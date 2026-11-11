
"""Train a Behavior Cloning MLP using expert demonstrations."""

import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from visionnav.learning.bc_model import BehaviorCloningMLP
from visionnav.learning.dataset import (
    prepare_datasets,
    DATA_DIR,
    OBSERVATION_DIM,
    ACTION_DIM,
)


ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = ROOT / "models" / "behavior_cloning"
LOG_DIR = ROOT / "logs" / "p5_training"

CHECKPOINT_PATH = MODEL_DIR / "bc_best.pth"
HISTORY_PATH = LOG_DIR / "training_history.json"

# Training configuration
SEED = 42
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 15


def set_seed(seed):
    """Set random seeds for reproducible training."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train the model for one epoch."""

    model.train()

    total_loss = 0.0
    total_samples = 0

    for observations, actions in loader:

        observations = observations.to(device)
        actions = actions.to(device)

        optimizer.zero_grad(set_to_none=True)

        predictions = model(observations)

        loss = criterion(predictions, actions)

        if not torch.isfinite(loss):
            raise RuntimeError("Training loss is NaN or Inf")

        loss.backward()
        optimizer.step()

        batch_size = observations.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


def evaluate_loss(model, loader, criterion, device):
    """Evaluate normalized action MSE."""

    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():

        for observations, actions in loader:

            observations = observations.to(device)
            actions = actions.to(device)

            predictions = model(observations)

            loss = criterion(predictions, actions)

            batch_size = observations.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return total_loss / total_samples


def evaluate_physical_rmse(
    model,
    loader,
    action_mean,
    action_std,
    device,
):
    """Evaluate action RMSE in physical units."""

    model.eval()

    squared_error = torch.zeros(
        ACTION_DIM, device=device
    )

    total_samples = 0

    action_mean = torch.as_tensor(
        action_mean, dtype=torch.float32, device=device
    )

    action_std = torch.as_tensor(
        action_std, dtype=torch.float32, device=device
    )

    with torch.inference_mode():

        for observations, normalized_actions in loader:

            observations = observations.to(device)
            normalized_actions = normalized_actions.to(device)

            predictions = model(observations)

            predicted_actions = (
                predictions * action_std + action_mean
            )

            actual_actions = (
                normalized_actions * action_std + action_mean
            )

            squared_error += (
                predicted_actions - actual_actions
            ).square().sum(dim=0)

            total_samples += observations.size(0)

    return torch.sqrt(
        squared_error / total_samples
    ).cpu().numpy()


def main():

    set_seed(SEED)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("DEVICE:", device)

    # Load prepared datasets.
    train_dataset, val_dataset, test_dataset = (
        prepare_datasets()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # Load normalization statistics calculated from training data.
    with np.load(
        DATA_DIR / "normalization.npz",
        allow_pickle=False,
    ) as stats:

        obs_mean = stats["obs_mean"].copy()
        obs_std = stats["obs_std"].copy()
        action_mean = stats["action_mean"].copy()
        action_std = stats["action_std"].copy()

    # Initialize model.
    model = BehaviorCloningMLP(
        input_dim=OBSERVATION_DIM,
        output_dim=ACTION_DIM,
    ).to(device)

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    print("\nBEHAVIOR CLONING TRAINING")
    print("-------------------------")
    print("Training samples:", len(train_dataset))
    print("Validation samples:", len(val_dataset))
    print("Test samples:", len(test_dataset))
    print("Parameters:", sum(
        p.numel() for p in model.parameters()
    ))

    print("\nStarting training...\n")

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    history = []

    for epoch in range(1, EPOCHS + 1):

        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
        )

        val_loss = evaluate_loss(
            model,
            val_loader,
            criterion,
            device,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        })

        improved = val_loss < best_val_loss

        if improved:

            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0

            # Save the best checkpoint.
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": OBSERVATION_DIM,
                    "output_dim": ACTION_DIM,
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "normalization": {
                        "obs_mean": torch.from_numpy(obs_mean),
                        "obs_std": torch.from_numpy(obs_std),
                        "action_mean": torch.from_numpy(action_mean),
                        "action_std": torch.from_numpy(action_std),
                    },
                    "config": {
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "seed": SEED,
                    },
                },
                CHECKPOINT_PATH,
            )

        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 5 == 0 or improved:
            print(
                f"Epoch [{epoch:03d}/{EPOCHS}] "
                f"Train Loss: {train_loss:.6f} "
                f"Val Loss: {val_loss:.6f}"
                + ("  * BEST" if improved else "")
            )

        if epochs_without_improvement >= PATIENCE:
            print("\nEarly stopping triggered.")
            break

    # Restore the best validation checkpoint.
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # Evaluate the test set only after model selection is complete.
    test_loss = evaluate_loss(
        model,
        test_loader,
        criterion,
        device,
    )

    physical_rmse = evaluate_physical_rmse(
        model,
        test_loader,
        action_mean,
        action_std,
        device,
    )

    history_data = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "test_linear_velocity_rmse_mps": float(
            physical_rmse[0]
        ),
        "test_angular_velocity_rmse_radps": float(
            physical_rmse[1]
        ),
        "history": history,
    }

    HISTORY_PATH.write_text(
        json.dumps(history_data, indent=2),
        encoding="utf-8",
    )

    print("\nTRAINING COMPLETED")
    print("-------------------")
    print("Best epoch:", best_epoch)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Test loss: {test_loss:.6f}")

    print(
        f"Linear velocity RMSE: {physical_rmse[0]:.5f} m/s"
    )

    print(
        f"Angular velocity RMSE: {physical_rmse[1]:.5f} rad/s"
    )

    print("Model saved:", CHECKPOINT_PATH)
    print("History saved:", HISTORY_PATH)


if __name__ == "__main__":
    main()
