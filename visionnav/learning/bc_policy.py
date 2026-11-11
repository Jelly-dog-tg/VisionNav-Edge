
"""Inference interface for the trained Behavior Cloning policy."""

from pathlib import Path

import numpy as np
import torch

from visionnav.learning.bc_model import BehaviorCloningMLP


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CHECKPOINT = (
    ROOT / "models" / "behavior_cloning" / "bc_best.pth"
)


class BehaviorCloningPolicy:

    def __init__(self, checkpoint_path=DEFAULT_CHECKPOINT):

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        checkpoint_path = Path(checkpoint_path)

        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=True,
        )

        input_dim = checkpoint["input_dim"]
        output_dim = checkpoint["output_dim"]

        if input_dim != 40 or output_dim != 2:
            raise ValueError("Unexpected model dimensions")

        # Reconstruct the trained network.
        self.model = BehaviorCloningMLP(
            input_dim=input_dim,
            output_dim=output_dim,
        ).to(self.device)

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        # Restore the exact training normalization statistics.
        normalization = checkpoint["normalization"]

        self.obs_mean = normalization["obs_mean"].to(
            device=self.device,
            dtype=torch.float32,
        )

        self.obs_std = normalization["obs_std"].to(
            device=self.device,
            dtype=torch.float32,
        )

        self.action_mean = normalization["action_mean"].to(
            device=self.device,
            dtype=torch.float32,
        )

        self.action_std = normalization["action_std"].to(
            device=self.device,
            dtype=torch.float32,
        )

        self.best_epoch = checkpoint["epoch"]

        print("BC POLICY LOADED")
        print("Device:", self.device)
        print("Best epoch:", self.best_epoch)

    def predict(self, observation):
        """
        Convert one raw 40-dimensional observation
        into physical robot velocity commands.

        Returns:
            np.ndarray: [linear_velocity, angular_velocity]
        """

        observation = np.asarray(
            observation,
            dtype=np.float32,
        )

        if observation.shape != (40,):
            raise ValueError(
                f"Expected observation shape (40,), "
                f"got {observation.shape}"
            )

        if not np.isfinite(observation).all():
            raise ValueError(
                "Observation contains NaN or Inf"
            )

        x = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        )

        # Normalize using training statistics.
        x = (x - self.obs_mean) / self.obs_std

        with torch.inference_mode():

            # Predict normalized actions.
            normalized_action = self.model(x)

            # Restore physical velocity units.
            action = (
                normalized_action * self.action_std
                + self.action_mean
            )

        action = action.cpu().numpy().astype(np.float32)

        if action.shape != (2,):
            raise RuntimeError("Invalid predicted action shape")

        if not np.isfinite(action).all():
            raise RuntimeError("Model produced invalid actions")

        return action


def main():

    policy = BehaviorCloningPolicy()

    # Load a few held-out observations for an inference smoke test.
    data_dir = ROOT / "logs" / "p5_dataset"

    observations = np.load(
        data_dir / "observations.npy",
        allow_pickle=False,
    )

    episode_ids = np.load(
        data_dir / "episode_ids.npy",
        allow_pickle=False,
    )

    import json

    split_info = json.loads(
        (data_dir / "dataset_splits.json").read_text(
            encoding="utf-8"
        )
    )

    test_ids = np.asarray(
        split_info["test_episode_ids"],
        dtype=np.int64,
    )

    episode_ids = np.rint(episode_ids).astype(np.int64)

    test_indices = np.flatnonzero(
        np.isin(episode_ids, test_ids)
    )

    if len(test_indices) == 0:
        raise RuntimeError("No test observations found")

    positions = np.linspace(
        0,
        len(test_indices) - 1,
        num=min(5, len(test_indices)),
        dtype=int,
    )

    selected_indices = test_indices[positions]

    print("\nBC INFERENCE TEST")
    print("-----------------")

    for index in selected_indices:

        observation = observations[index]

        linear_v, angular_w = policy.predict(
            observation
        )

        print(
            f"Sample {index:5d} | "
            f"Episode {episode_ids[index]:2d} | "
            f"v={linear_v:+.4f} m/s | "
            f"w={angular_w:+.4f} rad/s"
        )

    print("\nBC INFERENCE PASSED")


if __name__ == "__main__":
    main()
