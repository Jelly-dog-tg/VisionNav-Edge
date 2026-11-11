
"""Dataset preparation for VisionNav-Edge Behavior Cloning."""

from pathlib import Path
import json

import numpy as np
import torch
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "logs" / "p5_dataset"

SPLIT_SEED = 2026
OBSERVATION_DIM = 40
ACTION_DIM = 2


class BehaviorCloningDataset(Dataset):
    """PyTorch dataset for expert observation-action pairs."""

    def __init__(
        self,
        observations,
        actions,
        obs_mean,
        obs_std,
        action_mean,
        action_std,
    ):
        # Normalize observations and expert actions.
        normalized_obs = (
            observations - obs_mean
        ) / obs_std

        normalized_actions = (
            actions - action_mean
        ) / action_std

        self.observations = torch.from_numpy(
            np.ascontiguousarray(
                normalized_obs, dtype=np.float32
            )
        )

        self.actions = torch.from_numpy(
            np.ascontiguousarray(
                normalized_actions, dtype=np.float32
            )
        )

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, index):
        return (
            self.observations[index],
            self.actions[index],
        )


def load_raw_data():
    """Load and validate the original expert dataset."""

    observations = np.load(
        DATA_DIR / "observations.npy",
        allow_pickle=False,
    )

    actions = np.load(
        DATA_DIR / "actions.npy",
        allow_pickle=False,
    )

    episode_ids = np.load(
        DATA_DIR / "episode_ids.npy",
        allow_pickle=False,
    )

    metadata = json.loads(
        (DATA_DIR / "metadata.json").read_text(
            encoding="utf-8"
        )
    )

    if observations.ndim != 2:
        raise ValueError("Invalid observation dimensions")

    if actions.ndim != 2:
        raise ValueError("Invalid action dimensions")

    if observations.shape[1] != OBSERVATION_DIM:
        raise ValueError("Expected 40 observation features")

    if actions.shape[1] != ACTION_DIM:
        raise ValueError("Expected 2 action features")

    if len(observations) != len(actions):
        raise ValueError("Observation/action count mismatch")

    if episode_ids.ndim != 1:
        raise ValueError("Invalid episode ID dimensions")

    if len(episode_ids) != len(observations):
        raise ValueError("Episode ID count mismatch")

    if metadata["sample_count"] != len(observations):
        raise ValueError("Metadata sample count mismatch")

    # Convert integer-valued float32 IDs to int64.
    if not np.all(np.isfinite(episode_ids)):
        raise ValueError("Invalid episode IDs")

    rounded_ids = np.rint(episode_ids)

    if not np.allclose(episode_ids, rounded_ids):
        raise ValueError("Episode IDs must be integers")

    episode_ids = rounded_ids.astype(np.int64)

    observations = observations.astype(
        np.float32, copy=False
    )

    actions = actions.astype(
        np.float32, copy=False
    )

    if not np.all(np.isfinite(observations)):
        raise ValueError("Observation contains NaN or Inf")

    if not np.all(np.isfinite(actions)):
        raise ValueError("Action contains NaN or Inf")

    # Only successful episodes are allowed in training data.
    successful_ids = {
        int(ep["episode_id"])
        for ep in metadata["episodes"]
        if ep["success"]
    }

    actual_ids = set(np.unique(episode_ids).tolist())

    if not actual_ids.issubset(successful_ids):
        raise ValueError(
            "Failed episode found in training data"
        )

    return observations, actions, episode_ids, metadata


def split_by_episode(episode_ids):
    """Split episodes without mixing frames from one episode."""

    unique_ids = np.unique(episode_ids)

    if len(unique_ids) < 3:
        raise ValueError("At least 3 episodes are required")

    rng = np.random.default_rng(SPLIT_SEED)

    shuffled_ids = rng.permutation(unique_ids)

    n = len(shuffled_ids)

    n_train = int(round(n * 0.8))
    n_val = int(round(n * 0.1))
    n_test = n - n_train - n_val

    if min(n_train, n_val, n_test) < 1:
        raise ValueError("Invalid episode split")

    train_ids = shuffled_ids[:n_train]
    val_ids = shuffled_ids[n_train:n_train + n_val]
    test_ids = shuffled_ids[n_train + n_val:]

    return train_ids, val_ids, test_ids


def calculate_normalization(observations, actions):
    """Calculate normalization statistics using training data only."""

    obs_mean = observations.mean(axis=0)
    obs_std = observations.std(axis=0)

    action_mean = actions.mean(axis=0)
    action_std = actions.std(axis=0)

    # Avoid division by zero for constant features.
    obs_std = np.where(obs_std < 1e-6, 1.0, obs_std)
    action_std = np.where(action_std < 1e-6, 1.0, action_std)

    return obs_mean, obs_std, action_mean, action_std


def prepare_datasets():
    """Prepare train, validation and test datasets."""

    observations, actions, episode_ids, metadata = (
        load_raw_data()
    )

    train_ids, val_ids, test_ids = split_by_episode(
        episode_ids
    )

    train_mask = np.isin(episode_ids, train_ids)
    val_mask = np.isin(episode_ids, val_ids)
    test_mask = np.isin(episode_ids, test_ids)

    x_train = observations[train_mask]
    y_train = actions[train_mask]

    x_val = observations[val_mask]
    y_val = actions[val_mask]

    x_test = observations[test_mask]
    y_test = actions[test_mask]

    # Statistics must be calculated from training data only.
    stats = calculate_normalization(x_train, y_train)

    obs_mean, obs_std, action_mean, action_std = stats

    train_dataset = BehaviorCloningDataset(
        x_train, y_train,
        obs_mean, obs_std,
        action_mean, action_std,
    )

    val_dataset = BehaviorCloningDataset(
        x_val, y_val,
        obs_mean, obs_std,
        action_mean, action_std,
    )

    test_dataset = BehaviorCloningDataset(
        x_test, y_test,
        obs_mean, obs_std,
        action_mean, action_std,
    )

    # Save split information and normalization parameters.
    split_info = {
        "seed": SPLIT_SEED,
        "method": "episode_level",
        "train_episode_ids": train_ids.tolist(),
        "val_episode_ids": val_ids.tolist(),
        "test_episode_ids": test_ids.tolist(),
        "train_samples": len(x_train),
        "val_samples": len(x_val),
        "test_samples": len(x_test),
    }

    (DATA_DIR / "dataset_splits.json").write_text(
        json.dumps(split_info, indent=2),
        encoding="utf-8",
    )

    np.savez(
        DATA_DIR / "normalization.npz",
        obs_mean=obs_mean,
        obs_std=obs_std,
        action_mean=action_mean,
        action_std=action_std,
    )

    print("\nBEHAVIOR CLONING DATASET READY")
    print("--------------------------------")
    print(f"Total samples: {len(observations)}")
    print(f"Train samples: {len(x_train)}")
    print(f"Val samples:   {len(x_val)}")
    print(f"Test samples:  {len(x_test)}")
    print()
    print(f"Train episodes: {len(train_ids)}")
    print(f"Val episodes:   {len(val_ids)}")
    print(f"Test episodes:  {len(test_ids)}")
    print()
    print("Observation shape:", train_dataset[0][0].shape)
    print("Action shape:", train_dataset[0][1].shape)

    print("\nAction normalization:")
    print("Mean:", action_mean)
    print("Std:", action_std)

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    prepare_datasets()
