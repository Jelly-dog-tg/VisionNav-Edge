"""Train one independent BC-PPO V2 run with a selectable random seed.

Run from the project root, for example:
    python -m visionnav.learning.train_bc_residual_ppo_v2_seed --seed 123
    python -m visionnav.learning.train_bc_residual_ppo_v2_seed --seed 2026

Each seed is saved into its own model and log directories. Existing runs are
never overwritten.
"""

import argparse
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
DEFAULT_BC_CHECKPOINT = (
    ROOT / "models" / "behavior_cloning" / "bc_best.pth"
)

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an independent BC-PPO V2 run."
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--checkpoint-freq", type=int, default=20_000)
    parser.add_argument("--bc-checkpoint", type=Path, default=DEFAULT_BC_CHECKPOINT)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional unique run name. Default: seed_<seed>.",
    )
    return parser.parse_args()


class PeriodicSaveCallback(BaseCallback):
    """Save PPO weights and matching VecNormalize statistics together."""

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

        if not isinstance(self.training_env, VecNormalize):
            raise RuntimeError(
                "Training environment is not wrapped by VecNormalize."
            )

        self.training_env.save(str(norm_path))

        print(f"\nCheckpoint saved at {self.num_timesteps} timesteps:")
        print("Model:", model_path.with_suffix(".zip"))
        print("Normalization:", norm_path)
        return True


def make_env(bc_checkpoint):
    env = BCResidualNavigationEnvV2(
        checkpoint_path=bc_checkpoint,
        delta_v_max=0.05,
        delta_w_max=0.4,
        lookahead_distance_m=0.5,
    )
    # Monitor records ep_rew_mean and ep_len_mean without depending on
    # optional success/collision keys in terminal info.
    return Monitor(env)


def main():
    args = parse_args()

    if args.seed < 0:
        raise ValueError("--seed must be a non-negative integer.")
    if args.timesteps <= 0:
        raise ValueError("--timesteps must be positive.")
    if args.checkpoint_freq <= 0:
        raise ValueError("--checkpoint-freq must be positive.")
    if not args.bc_checkpoint.is_file():
        raise FileNotFoundError(
            f"BC checkpoint not found: {args.bc_checkpoint}"
        )

    run_name = args.run_name or f"seed_{args.seed}"
    if not run_name.replace("_", "").replace("-", "").isalnum():
        raise ValueError("--run-name may contain only letters, digits, _ and -.")

    model_dir = ROOT / "models" / f"bc_residual_ppo_v2_{run_name}"
    checkpoint_dir = model_dir / "checkpoints"
    log_dir = ROOT / "logs" / "p6_training" / f"bc_residual_ppo_v2_{run_name}"

    final_model_path = model_dir / "bc_residual_ppo_v2_final.zip"
    final_norm_path = model_dir / "vecnormalize.pkl"

    # Prevent accidentally overwriting a completed or partially completed run.
    if model_dir.exists() and any(model_dir.iterdir()):
        raise FileExistsError(
            f"Run directory already exists and is not empty:\n{model_dir}\n"
            "Choose a new --run-name or move the old directory first."
        )

    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Seed Python, NumPy, PyTorch and the SB3 environment/action generators.
    set_random_seed(args.seed)

    print("========== BC-PPO V2 INDEPENDENT TRAINING ==========")
    print("Seed:", args.seed)
    print("Run name:", run_name)
    print("BC checkpoint:", args.bc_checkpoint)
    print("PPO observation dimension: 44")
    print("BC observation dimension: 40")
    print("Action dimension: 2")
    print("Device: CPU")
    print("Total timesteps:", args.timesteps)
    print("Checkpoint frequency:", args.checkpoint_freq)
    print("Model directory:", model_dir)
    print("Log directory:", log_dir)

    base_vec_env = DummyVecEnv([
        lambda: make_env(args.bc_checkpoint)
    ])
    base_vec_env.seed(args.seed)

    env = VecNormalize(
        base_vec_env,
        training=True,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        gamma=PPO_CONFIG["gamma"],
    )

    callback = PeriodicSaveCallback(
        save_freq=args.checkpoint_freq,
        save_dir=checkpoint_dir,
        verbose=1,
    )

    config = {
        "experiment": "BC-PPO V2 independent seed run",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "run_name": run_name,
        "total_timesteps_requested": args.timesteps,
        "checkpoint_frequency": args.checkpoint_freq,
        "device": "cpu",
        "environment": "BCResidualNavigationEnvV2",
        "bc_checkpoint": str(args.bc_checkpoint),
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

    config_path = model_dir / "training_config.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    tensorboard_log = (
        str(log_dir)
        if importlib.util.find_spec("tensorboard") is not None
        else None
    )
    if tensorboard_log is None:
        print("TensorBoard not installed; console logging will be used.")

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
            seed=args.seed,
            device="cpu",
            verbose=1,
            tensorboard_log=tensorboard_log,
        )

        print("\nModel initialized.")
        print("Policy observation space:", model.observation_space)
        print("Policy action space:", model.action_space)

        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            tb_log_name=run_name,
            progress_bar=False,
        )

        model.save(str(model_dir / "bc_residual_ppo_v2_final"))
        env.save(str(final_norm_path))

        print("\n========== INDEPENDENT TRAINING COMPLETED ==========")
        print("Seed:", args.seed)
        print("Final model:", final_model_path)
        print("VecNormalize:", final_norm_path)
        print("Training config:", config_path)
        print("TensorBoard logs:", log_dir)

    finally:
        env.close()


if __name__ == "__main__":
    main()
