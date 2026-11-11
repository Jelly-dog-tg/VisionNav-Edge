"""Train BC + PPO residual navigation V2.

BC receives the original 40-D observation inside the environment.
PPO receives the augmented 44-D observation.

Run from the project root:
    python -m visionnav.learning.train_bc_residual_ppo_v2
"""

import importlib.util
import json
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from visionnav.learning.bc_residual_navigation_env_v2 import (
    BCResidualNavigationEnvV2,
)


ROOT = Path(__file__).resolve().parents[2]

BC_CHECKPOINT = (
    ROOT / "models" / "behavior_cloning" / "bc_best.pth"
)

MODEL_DIR = ROOT / "models" / "bc_residual_ppo_v2"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"
LOG_DIR = ROOT / "logs" / "p6_training" / "bc_residual_ppo_v2"

SEED = 42
TOTAL_TIMESTEPS = 200_000
CHECKPOINT_FREQ = 20_000

# Keep the V1 PPO optimization settings as the starting baseline.
PPO_CONFIG = {
    "learning_rate": 1e-4,
    "n_steps": 2048,
    "batch_size": 128,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.005,
    "vf_coef": 0.5,
    "policy_kwargs": {
        "net_arch": {
            "pi": [128, 128],
            "vf": [128, 128],
        }
    },
}


class PeriodicSaveCallback(BaseCallback):
    """Save both PPO weights and matching VecNormalize statistics."""

    def __init__(self, save_freq, save_dir, verbose=1):
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.save_dir = Path(save_dir)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq != 0:
            return True

        step_dir = self.save_dir / f"step_{self.num_timesteps:07d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        model_path = step_dir / "ppo_v2"
        norm_path = step_dir / "vecnormalize.pkl"

        self.model.save(str(model_path))

        if isinstance(self.training_env, VecNormalize):
            self.training_env.save(str(norm_path))
        else:
            raise RuntimeError(
                "Training environment is not wrapped by VecNormalize."
            )

        if self.verbose:
            print(f"\nCheckpoint saved at {self.num_timesteps} timesteps:")
            print("Model:", model_path.with_suffix(".zip"))
            print("Normalization:", norm_path)

        return True


def make_env():
    """Create one monitored V2 environment."""
    env = BCResidualNavigationEnvV2(
        checkpoint_path=BC_CHECKPOINT,
        delta_v_max=0.05,
        delta_w_max=0.4,
        lookahead_distance_m=0.5,
    )

    # Monitor provides ep_rew_mean / ep_len_mean and episode success data.
    return Monitor(
        env,
        info_keywords=("success", "collision"),
    )


def main():
    if not BC_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"BC checkpoint not found: {BC_CHECKPOINT}"
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    set_random_seed(SEED)

    print("========== BC-PPO V2 TRAINING ==========")
    print("Project root:", ROOT)
    print("BC checkpoint:", BC_CHECKPOINT)
    print("PPO observation dimension: 44")
    print("BC observation dimension: 40")
    print("Action dimension: 2")
    print("Device: CPU")
    print("Total timesteps:", TOTAL_TIMESTEPS)
    print("Checkpoint frequency:", CHECKPOINT_FREQ)

    # One PyBullet environment, matching the V1 single-environment setup.
    base_vec_env = DummyVecEnv([make_env])
    base_vec_env.seed(SEED)

    # Normalize PPO observations, but keep reward scale unchanged.
    # BC receives its raw 40-D observation internally before this wrapper.
    env = VecNormalize(
        base_vec_env,
        training=True,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=PPO_CONFIG["gamma"],
    )

    callback = PeriodicSaveCallback(
        save_freq=CHECKPOINT_FREQ,
        save_dir=CHECKPOINT_DIR,
        verbose=1,
    )

    config = {
        "experiment": "BC-PPO V2",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": SEED,
        "total_timesteps": TOTAL_TIMESTEPS,
        "checkpoint_frequency": CHECKPOINT_FREQ,
        "device": "cpu",
        "environment": "BCResidualNavigationEnvV2",
        "bc_checkpoint": str(BC_CHECKPOINT),
        "bc_observation_dim": 40,
        "ppo_observation_dim": 44,
        "action_dim": 2,
        "delta_v_max": 0.05,
        "delta_w_max": 0.4,
        "lookahead_distance_m": 0.5,
        "vecnormalize": {
            "norm_obs": True,
            "norm_reward": False,
            "clip_obs": 10.0,
        },
        "ppo": {
            "learning_rate": PPO_CONFIG["learning_rate"],
            "n_steps": PPO_CONFIG["n_steps"],
            "batch_size": PPO_CONFIG["batch_size"],
            "n_epochs": PPO_CONFIG["n_epochs"],
            "gamma": PPO_CONFIG["gamma"],
            "gae_lambda": PPO_CONFIG["gae_lambda"],
            "clip_range": PPO_CONFIG["clip_range"],
            "ent_coef": PPO_CONFIG["ent_coef"],
            "vf_coef": PPO_CONFIG["vf_coef"],
            "net_arch": {
                "pi": [128, 128],
                "vf": [128, 128],
            },
        },
    }

    config_path = MODEL_DIR / "training_config.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    tensorboard_log = (
        str(LOG_DIR)
        if importlib.util.find_spec("tensorboard") is not None
        else None
    )
    if tensorboard_log is None:
        print("TensorBoard package not found; continuing with console logging.")

    model = None

    try:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=PPO_CONFIG["learning_rate"],
            n_steps=PPO_CONFIG["n_steps"],
            batch_size=PPO_CONFIG["batch_size"],
            n_epochs=PPO_CONFIG["n_epochs"],
            gamma=PPO_CONFIG["gamma"],
            gae_lambda=PPO_CONFIG["gae_lambda"],
            clip_range=PPO_CONFIG["clip_range"],
            ent_coef=PPO_CONFIG["ent_coef"],
            vf_coef=PPO_CONFIG["vf_coef"],
            policy_kwargs=PPO_CONFIG["policy_kwargs"],
            seed=SEED,
            device="cpu",
            verbose=1,
            tensorboard_log=tensorboard_log,
        )

        print("\nModel initialized.")
        print("Policy observation space:", model.observation_space)
        print("Policy action space:", model.action_space)

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback,
            tb_log_name="bc_residual_ppo_v2",
            progress_bar=False,
        )

        final_model_path = MODEL_DIR / "bc_residual_ppo_v2_final"
        final_norm_path = MODEL_DIR / "vecnormalize.pkl"

        model.save(str(final_model_path))
        env.save(str(final_norm_path))

        print("\n========== TRAINING COMPLETED ==========")
        print("Final model:", final_model_path.with_suffix(".zip"))
        print("VecNormalize:", final_norm_path)
        print("Training config:", config_path)
        print("TensorBoard logs:", LOG_DIR)

    finally:
        env.close()


if __name__ == "__main__":
    main()
