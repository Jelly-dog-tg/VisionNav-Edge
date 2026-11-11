"""Fine-tune BC + PPO Residual V2 into V3.

Run from the project root:
    python -m visionnav.learning.train_bc_residual_ppo_v3 \
        --seed 42 --timesteps 100000 --residual-scale 0.25

This script:
- Loads an existing V2 PPO checkpoint and its matching VecNormalize stats.
- Wraps the unchanged V2 environment with ResidualScaleWrapper.
- Continues training; it does not initialize a new Actor/Critic.
- Saves all V3 artifacts to a new seed-specific directory.
"""

import argparse
import importlib.util
import json
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import get_schedule_fn, set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from visionnav.learning.bc_residual_navigation_env_v2 import (
    BCResidualNavigationEnvV2,
)
from visionnav.learning.residual_scale_wrapper import ResidualScaleWrapper


ROOT = Path(__file__).resolve().parents[2]

BC_CHECKPOINT = ROOT / "models" / "behavior_cloning" / "bc_best.pth"

V2_MODEL_DIRS = {
    42: ROOT / "models" / "bc_residual_ppo_v2",
    123: ROOT / "models" / "bc_residual_ppo_v2_seed_123",
    2026: ROOT / "models" / "bc_residual_ppo_v2_seed_2026",
}

V3_MODEL_ROOT = ROOT / "models"
V3_LOG_ROOT = ROOT / "logs" / "p6_training"

DEFAULT_SEED = 42
DEFAULT_ADDITIONAL_TIMESTEPS = 100_000
DEFAULT_RESIDUAL_SCALE = 0.25
CHECKPOINT_FREQ = 20_000
LEARNING_RATE = 5e-5

# These match the V2 training configuration. Only the learning rate is changed.
PPO_CONFIG = {
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


class PeriodicV3SaveCallback(BaseCallback):
    """Save V3 weights and matching VecNormalize statistics periodically."""

    def __init__(self, save_freq: int, save_dir: Path, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.save_dir = Path(save_dir)

    def _on_step(self) -> bool:
        # n_calls counts newly collected environment steps (one env is used).
        if self.n_calls % self.save_freq != 0:
            return True

        step_dir = self.save_dir / f"step_{self.num_timesteps:07d}"
        step_dir.mkdir(parents=True, exist_ok=False)

        model_path = step_dir / "bc_residual_ppo_v3"
        vecnormalize_path = step_dir / "vecnormalize.pkl"

        self.model.save(str(model_path))

        training_env = self.training_env
        if not isinstance(training_env, VecNormalize):
            raise RuntimeError(
                "Expected VecNormalize as the outer training environment."
            )
        training_env.save(str(vecnormalize_path))

        if self.verbose:
            print(f"\n[V3 CHECKPOINT] Timesteps: {self.num_timesteps}")
            print("Model:", model_path.with_suffix(".zip"))
            print("VecNormalize:", vecnormalize_path)

        return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune an existing BC-PPO V2 checkpoint into V3."
    )
    parser.add_argument(
        "--seed",
        type=int,
        choices=(42, 123, 2026),
        default=DEFAULT_SEED,
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_ADDITIONAL_TIMESTEPS,
        help="Additional environment steps, not the cumulative total.",
    )
    parser.add_argument(
        "--residual-scale",
        type=float,
        default=DEFAULT_RESIDUAL_SCALE,
    )
    return parser.parse_args()


def make_monitored_env(seed: int, residual_scale: float):
    """Create the exact V3 action path around the unchanged V2 environment."""
    def _make():
        env = BCResidualNavigationEnvV2(
            checkpoint_path=BC_CHECKPOINT,
            delta_v_max=0.05,
            delta_w_max=0.4,
            lookahead_distance_m=0.5,
        )
        env = ResidualScaleWrapper(
            env,
            residual_scale=residual_scale,
        )
        env = Monitor(
            env,
            info_keywords=("success", "collision"),
        )
        return env

    vec_env = DummyVecEnv([_make])
    vec_env.seed(seed)
    return vec_env


def resolve_v2_files(seed: int):
    model_dir = V2_MODEL_DIRS[seed]
    model_path = model_dir / "bc_residual_ppo_v2_final.zip"
    vecnormalize_path = model_dir / "vecnormalize.pkl"
    return model_dir, model_path, vecnormalize_path


def ensure_output_is_safe(output_dir: Path):
    """Never overwrite an existing V3 run."""
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"V3 output directory already exists and is not empty:\n"
            f"{output_dir}\n"
            "Refusing to overwrite existing training results."
        )
    output_dir.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()

    if args.timesteps <= 0:
        raise ValueError("--timesteps must be a positive integer.")
    if not 0.0 <= args.residual_scale <= 1.0:
        raise ValueError("--residual-scale must be within [0, 1].")

    if not BC_CHECKPOINT.is_file():
        raise FileNotFoundError(f"BC checkpoint not found: {BC_CHECKPOINT}")

    v2_dir, v2_model_path, v2_vecnormalize_path = resolve_v2_files(args.seed)
    for path in (v2_model_path, v2_vecnormalize_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required V2 file not found: {path}")

    model_dir = V3_MODEL_ROOT / f"bc_residual_ppo_v3_seed_{args.seed}"
    checkpoint_dir = model_dir / "checkpoints"
    log_dir = V3_LOG_ROOT / f"bc_residual_ppo_v3_seed_{args.seed}"

    ensure_output_is_safe(model_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    set_random_seed(args.seed)

    print("=" * 64)
    print("BC-PPO V3 FINE-TUNING")
    print("=" * 64)
    print("Project root:", ROOT)
    print("Seed:", args.seed)
    print("Device: CPU")
    print("BC checkpoint:", BC_CHECKPOINT)
    print("V2 model:", v2_model_path)
    print("V2 VecNormalize:", v2_vecnormalize_path)
    print("Residual scale:", args.residual_scale)
    print("Additional timesteps:", args.timesteps)
    print("Learning rate:", LEARNING_RATE)
    print("Checkpoint frequency:", CHECKPOINT_FREQ)
    print("Output:", model_dir)

    base_vec_env = make_monitored_env(
        seed=args.seed,
        residual_scale=args.residual_scale,
    )

    # Load V2 observation statistics onto the same 44-D observation space.
    env = VecNormalize.load(
        str(v2_vecnormalize_path),
        base_vec_env,
    )
    env.training = True
    env.norm_reward = False

    tensorboard_log = (
        str(log_dir)
        if importlib.util.find_spec("tensorboard") is not None
        else None
    )

    model = None
    try:
        # Critical: PPO.load restores the trained Actor/Critic and optimizer state.
        model = PPO.load(
            str(v2_model_path),
            env=env,
            device="cpu",
            tensorboard_log=tensorboard_log,
        )

        previous_timesteps = int(model.num_timesteps)

        # Re-seed the continuation and replace the LR schedule with a constant
        # 5e-5 schedule. PPO's train() updates optimizer LR from lr_schedule.
        model.set_random_seed(args.seed)
        model.learning_rate = LEARNING_RATE
        model.lr_schedule = get_schedule_fn(LEARNING_RATE)
        for parameter_group in model.policy.optimizer.param_groups:
            parameter_group["lr"] = LEARNING_RATE

        # Keep the V2 optimization setup; only LR and residual scale change.
        loaded_config = {
            "n_steps": int(model.n_steps),
            "batch_size": int(model.batch_size),
            "n_epochs": int(model.n_epochs),
            "gamma": float(model.gamma),
            "gae_lambda": float(model.gae_lambda),
            "clip_range": float(PPO_CONFIG["clip_range"]),
            "ent_coef": float(model.ent_coef),
            "vf_coef": float(model.vf_coef),
            "net_arch": PPO_CONFIG["policy_kwargs"]["net_arch"],
        }

        print("\n========== MODEL LOADED ==========")
        print("Previous timesteps:", previous_timesteps)
        print("Policy observation space:", model.observation_space)
        print("Policy action space:", model.action_space)
        print("Loaded n_steps:", model.n_steps)
        print("Loaded batch_size:", model.batch_size)
        print("Loaded n_epochs:", model.n_epochs)
        print("Effective learning rate:", LEARNING_RATE)

        if model.observation_space.shape != (44,):
            raise RuntimeError(
                f"Expected V2 PPO observation shape (44,), got "
                f"{model.observation_space.shape}"
            )
        if model.action_space.shape != (2,):
            raise RuntimeError(
                f"Expected PPO residual action shape (2,), got "
                f"{model.action_space.shape}"
            )

        config = {
            "experiment": "BC-PPO V3 fine-tuning",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "seed": args.seed,
            "device": "cpu",
            "source_v2_model": str(v2_model_path),
            "source_v2_vecnormalize": str(v2_vecnormalize_path),
            "bc_checkpoint": str(BC_CHECKPOINT),
            "previous_timesteps": previous_timesteps,
            "additional_timesteps": int(args.timesteps),
            "expected_cumulative_timesteps": previous_timesteps + int(args.timesteps),
            "checkpoint_frequency": CHECKPOINT_FREQ,
            "residual_scale": float(args.residual_scale),
            "delta_v_max": 0.05,
            "delta_w_max": 0.4,
            "lookahead_distance_m": 0.5,
            "bc_observation_dim": 40,
            "ppo_observation_dim": 44,
            "action_dim": 2,
            "learning_rate": LEARNING_RATE,
            "ppo": loaded_config,
            "vecnormalize": {
                "loaded_from_v2": True,
                "training": True,
                "norm_obs": bool(env.norm_obs),
                "norm_reward": bool(env.norm_reward),
                "clip_obs": float(env.clip_obs),
                "gamma": float(env.gamma),
            },
            "task_sampling": (
                "Uses the existing V2 environment's online random task "
                "sampler on each reset; fixed benchmark task files are not "
                "passed to the training environment."
            ),
        }
        config_path = model_dir / "training_config.json"
        config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        callback = PeriodicV3SaveCallback(
            save_freq=CHECKPOINT_FREQ,
            save_dir=checkpoint_dir,
            verbose=1,
        )

        print("\n========== TRAINING START ==========")
        print("Target additional steps:", args.timesteps)
        print(
            "Expected final timestep:",
            previous_timesteps + args.timesteps,
        )
        print("Residual scaling is applied once by ResidualScaleWrapper.")
        print("No fixed benchmark tasks are passed into training.")

        model.learn(
            total_timesteps=args.timesteps,
            callback=callback,
            tb_log_name=f"bc_residual_ppo_v3_seed_{args.seed}",
            reset_num_timesteps=False,
            progress_bar=False,
        )

        final_model_path = model_dir / "bc_residual_ppo_v3_final"
        final_vecnormalize_path = model_dir / "vecnormalize.pkl"

        model.save(str(final_model_path))
        env.save(str(final_vecnormalize_path))

        final_config = dict(config)
        final_config["completed_at"] = datetime.now().isoformat(timespec="seconds")
        final_config["final_timesteps"] = int(model.num_timesteps)
        final_config["final_model"] = str(final_model_path.with_suffix(".zip"))
        final_config["final_vecnormalize"] = str(final_vecnormalize_path)
        config_path.write_text(
            json.dumps(final_config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("\n========== PPO V3 TRAINING COMPLETED ==========")
        print("Final timesteps:", model.num_timesteps)
        print("Final model:", final_model_path.with_suffix(".zip"))
        print("VecNormalize:", final_vecnormalize_path)
        print("Training config:", config_path)
        print("TensorBoard logs:", log_dir)

    finally:
        env.close()


if __name__ == "__main__":
    main()
