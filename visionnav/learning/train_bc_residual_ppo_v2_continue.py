"""Continue existing 200k BC-PPO V2 checkpoints for 200k more steps.

Run from project root:
    python -m visionnav.learning.train_bc_residual_ppo_v2_continue --seed 42
    python -m visionnav.learning.train_bc_residual_ppo_v2_continue --seed 123
    python -m visionnav.learning.train_bc_residual_ppo_v2_continue --seed 2026

The original 200k model and VecNormalize files are never overwritten.
A separate *_400k directory is created for each continued run.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from visionnav.learning.bc_residual_navigation_env_v2 import (
    BCResidualNavigationEnvV2,
)


ROOT = Path(__file__).resolve().parents[2]
BC_CHECKPOINT = ROOT / "models" / "behavior_cloning" / "bc_best.pth"

ADDITIONAL_TIMESTEPS_DEFAULT = 200_000
CHECKPOINT_FREQ_DEFAULT = 20_000


def parse_args():
    parser = argparse.ArgumentParser(
        description="Continue a trained BC-PPO V2 model without reinitializing it."
    )
    parser.add_argument("--seed", type=int, required=True, choices=[42, 123, 2026])
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=None,
        help="Optional source model directory. Defaults to the known 200k directory for this seed.",
    )
    parser.add_argument(
        "--additional-timesteps",
        type=int,
        default=ADDITIONAL_TIMESTEPS_DEFAULT,
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=CHECKPOINT_FREQ_DEFAULT,
    )
    parser.add_argument(
        "--continuation-seed",
        type=int,
        default=None,
        help="Seed for the newly created environment/RNG stream. Defaults to seed + 200000.",
    )
    return parser.parse_args()


def default_source_dir(seed):
    # Seed 42 was the original V2 run; Seeds 123 and 2026 used the
    # independent-seed training script.
    if seed == 42:
        return ROOT / "models" / "bc_residual_ppo_v2"
    return ROOT / "models" / f"bc_residual_ppo_v2_seed_{seed}"


def resolve_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def make_env():
    env = BCResidualNavigationEnvV2(
        checkpoint_path=BC_CHECKPOINT,
        delta_v_max=0.05,
        delta_w_max=0.4,
        lookahead_distance_m=0.5,
    )
    return env


class PeriodicSaveCallback(BaseCallback):
    """Save PPO and its matching VecNormalize state at the same timestep."""

    def __init__(self, save_freq, save_dir, verbose=1):
        super().__init__(verbose)
        self.save_freq = int(save_freq)
        self.save_dir = Path(save_dir)

    def _on_step(self):
        if self.n_calls % self.save_freq != 0:
            return True

        step_dir = self.save_dir / f"step_{self.num_timesteps:07d}"
        step_dir.mkdir(parents=True, exist_ok=True)

        model_path = step_dir / "ppo_v2"
        norm_path = step_dir / "vecnormalize.pkl"

        self.model.save(str(model_path))

        if not isinstance(self.training_env, VecNormalize):
            raise RuntimeError("Expected VecNormalize as the training environment.")

        self.training_env.save(str(norm_path))

        print(f"\nCheckpoint saved at total timestep {self.num_timesteps}:")
        print("Model:", model_path.with_suffix(".zip"))
        print("VecNormalize:", norm_path)
        return True


def main():
    args = parse_args()

    if args.additional_timesteps <= 0:
        raise ValueError("--additional-timesteps must be positive.")
    if args.checkpoint_freq <= 0:
        raise ValueError("--checkpoint-freq must be positive.")
    if not BC_CHECKPOINT.is_file():
        raise FileNotFoundError(f"BC checkpoint not found: {BC_CHECKPOINT}")

    source_dir = (
        resolve_path(args.source_dir)
        if args.source_dir is not None
        else default_source_dir(args.seed)
    )
    source_model = source_dir / "bc_residual_ppo_v2_final.zip"
    source_norm = source_dir / "vecnormalize.pkl"

    for required in (source_model, source_norm):
        if not required.is_file():
            raise FileNotFoundError(
                f"Required 200k source file not found: {required}"
            )

    output_dir = ROOT / "models" / f"bc_residual_ppo_v2_seed_{args.seed}_400k"
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory already exists and is not empty:\n{output_dir}\n"
            "The script will not overwrite a previous continuation run."
        )

    checkpoint_dir = output_dir / "checkpoints"
    log_dir = ROOT / "logs" / "p6_training" / f"bc_residual_ppo_v2_seed_{args.seed}_400k"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    continuation_seed = (
        args.continuation_seed
        if args.continuation_seed is not None
        else args.seed + 200_000
    )
    set_random_seed(continuation_seed)

    print("========== BC-PPO V2 CONTINUED TRAINING ==========")
    print("Original training seed:", args.seed)
    print("Continuation RNG seed:", continuation_seed)
    print("Source model:", source_model)
    print("Source VecNormalize:", source_norm)
    print("Additional timesteps:", args.additional_timesteps)
    print("Checkpoint frequency:", args.checkpoint_freq)
    print("Output directory:", output_dir)

    base_env = DummyVecEnv([make_env])
    base_env.seed(continuation_seed)

    # Restore the 200k observation-normalization statistics, then continue
    # updating them during the additional training.
    env = VecNormalize.load(str(source_norm), base_env)
    env.training = True
    env.norm_reward = False

    try:
        # Loading the SB3 checkpoint restores policy weights and, where
        # supported by the saved checkpoint, optimizer state.
        model = PPO.load(str(source_model), env=env, device="cpu")

        starting_timesteps = int(model.num_timesteps)
        if not 180_000 <= starting_timesteps <= 220_000:
            raise RuntimeError(
                f"Expected a checkpoint near 200k steps, but loaded "
                f"{starting_timesteps}. Refusing to label it as a 400k run."
            )

        optimizer_state_entries = None
        optimizer = getattr(model.policy, "optimizer", None)
        if optimizer is not None:
            optimizer_state_entries = len(optimizer.state)
            if optimizer_state_entries == 0:
                raise RuntimeError(
                    "The loaded PPO optimizer has no saved state. "
                    "To avoid silently restarting Adam during a continuation "
                    "experiment, training has been stopped. Check the source "
                    "checkpoint and Stable-Baselines3 version first."
                )

        # The RNG state of the original live process is not stored as a
        # complete simulator snapshot. Start a deterministic new RNG stream.
        model.set_random_seed(continuation_seed)
        model.tensorboard_log = str(log_dir)

        print("\nLoaded PPO model.")
        print("Starting timesteps:", starting_timesteps)
        print("Optimizer state entries:", optimizer_state_entries)
        print("Observation space:", model.observation_space)
        print("Action space:", model.action_space)
        print("Learning rate:", model.learning_rate)
        print("N steps:", model.n_steps)
        print("Batch size:", model.batch_size)
        print("Epochs:", model.n_epochs)

        callback = PeriodicSaveCallback(
            save_freq=args.checkpoint_freq,
            save_dir=checkpoint_dir,
            verbose=1,
        )

        model.learn(
            total_timesteps=args.additional_timesteps,
            callback=callback,
            tb_log_name=f"seed_{args.seed}_continued",
            reset_num_timesteps=False,
            progress_bar=False,
        )

        final_timesteps = int(model.num_timesteps)
        final_model_path = output_dir / "bc_residual_ppo_v2_final.zip"
        final_norm_path = output_dir / "vecnormalize.pkl"

        model.save(str(output_dir / "bc_residual_ppo_v2_final"))
        env.save(str(final_norm_path))

        config = {
            "experiment": "BC-PPO V2 continued training",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "original_training_seed": args.seed,
            "continuation_rng_seed": continuation_seed,
            "source_model": str(source_model),
            "source_vecnormalize": str(source_norm),
            "starting_timesteps": starting_timesteps,
            "additional_timesteps_requested": args.additional_timesteps,
            "final_timesteps_actual": final_timesteps,
            "optimizer_state_entries_after_load": optimizer_state_entries,
            "bc_checkpoint": str(BC_CHECKPOINT),
            "device": "cpu",
            "environment": "BCResidualNavigationEnvV2",
            "observation_dim_ppo": 44,
            "observation_dim_bc": 40,
            "action_dim": 2,
            "delta_v_max": 0.05,
            "delta_w_max": 0.4,
            "lookahead_distance_m": 0.5,
            "vecnormalize": {
                "norm_obs": env.norm_obs,
                "norm_reward": env.norm_reward,
                "clip_obs": env.clip_obs,
            },
            "ppo": {
                "learning_rate": float(model.learning_rate),
                "n_steps": model.n_steps,
                "batch_size": model.batch_size,
                "n_epochs": model.n_epochs,
                "gamma": model.gamma,
                "gae_lambda": model.gae_lambda,
                "clip_range": model.clip_range(1.0),
            },
            "note": (
                "Continued from the saved 200k policy checkpoint. "
                "The simulator/RNG live state was not restored; a new "
                "deterministic continuation RNG stream was used."
            ),
        }
        (output_dir / "continuation_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print("\n========== CONTINUED TRAINING COMPLETED ==========")
        print("Seed:", args.seed)
        print("Starting timesteps:", starting_timesteps)
        print("Final timesteps:", final_timesteps)
        print("Final model:", final_model_path)
        print("VecNormalize:", final_norm_path)
        print("Config:", output_dir / "continuation_config.json")
        print("Logs:", log_dir)

    finally:
        env.close()


if __name__ == "__main__":
    main()
