"""P8.3: Train a compact state-based ACT-style policy from P8.1 expert windows.

Run from project root:
    python experiments/p8_train_act.py

Selection: deterministic z=0 validation L1, not posterior reconstruction L1.
The independent test split is evaluated exactly once after selecting the
best validation checkpoint. No PyBullet environment is required for training.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visionnav.learning.act_model import ACTConfig, ACTPolicy, act_loss

DATA_DIR = ROOT / "logs/p8_act_dataset"
MODEL_DIR = ROOT / "models/act"
LOG_DIR = ROOT / "logs/p8_act_training"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_loader(split: str, batch_size: int, device: torch.device, seed: int):
    x_path = DATA_DIR / f"{split}_observations.npy"
    y_path = DATA_DIR / f"{split}_action_chunks.npy"
    x = np.load(x_path, allow_pickle=False)
    y = np.load(y_path, allow_pickle=False)
    if x.ndim != 3 or x.shape[1:] != (8, 40):
        raise ValueError(f"Wrong {split} observation shape: {x.shape}")
    if y.shape != (len(x), 16, 2):
        raise ValueError(f"Wrong {split} action shape: {y.shape}")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError(f"Nonfinite values in {split}")
    ds = TensorDataset(
        torch.from_numpy(np.ascontiguousarray(x)),
        torch.from_numpy(np.ascontiguousarray(y)),
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=0,  # robust on Windows
        pin_memory=(device.type == "cuda"),
        generator=torch.Generator().manual_seed(seed),
    )


@torch.inference_mode()
def evaluate(model: ACTPolicy, loader, device, normalization) -> dict:
    """Evaluate the real deployment mode, using z=0 and NO future actions."""
    model.eval()
    total_abs_norm = 0.0
    total_values = 0
    sum_sq_physical = torch.zeros(2, dtype=torch.float64)
    total_actions = 0
    mean = torch.as_tensor(normalization["action_mean"], device=device)
    std = torch.as_tensor(normalization["action_std"], device=device)
    for obs, target in loader:
        obs = obs.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        prediction, _, _ = model(obs)
        err = prediction - target
        total_abs_norm += err.abs().sum().item()
        total_values += err.numel()
        phys_err = err * std.view(1, 1, 2)
        sum_sq_physical += phys_err.square().sum(dim=(0, 1)).cpu().to(torch.float64)
        total_actions += target.shape[0] * target.shape[1]
    if total_values == 0:
        raise RuntimeError("Empty evaluation split")
    rmse = torch.sqrt(sum_sq_physical / total_actions).tolist()
    return {
        "inference_l1_normalized": total_abs_norm / total_values,
        "linear_velocity_rmse_mps": float(rmse[0]),
        "angular_velocity_rmse_radps": float(rmse[1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--kl-weight", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch-size, and patience must be positive")
    if args.kl_weight < 0 or args.lr <= 0:
        raise ValueError("kl-weight must be >=0 and lr must be >0")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )
    best_path = MODEL_DIR / "act_best.pth"
    history_path = LOG_DIR / "training_history.csv"
    summary_path = LOG_DIR / "training_summary.json"
    existing = [p for p in [best_path, history_path, summary_path] if p.exists()]
    if existing:
        raise FileExistsError(
            f"Existing P8 training artifacts: {existing}. "
            "Keep the original results; move them to a backup directory before rerunning."
        )

    set_seed(args.seed)
    train_loader = get_loader("train", args.batch_size, device, args.seed)
    val_loader = get_loader("val", args.batch_size, device, args.seed)
    test_loader = get_loader("test", args.batch_size, device, args.seed)
    with np.load(DATA_DIR / "normalization.npz", allow_pickle=False) as f:
        normalization = {name: f[name].copy() for name in f.files}
    with (DATA_DIR / "act_dataset_metadata.json").open(encoding="utf-8") as f:
        dataset_meta = json.load(f)

    config = ACTConfig()
    model = ACTPolicy(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"P8 ACT training | device={device} | parameters={n_params:,}")
    print(f"Windows train={len(train_loader.dataset)}, val={len(val_loader.dataset)}, test={len(test_loader.dataset)}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    best_epoch = None
    bad_epochs = 0
    rows = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        accumulated = 0.0
        accumulated_recon = 0.0
        accumulated_kl = 0.0
        observed = 0
        for obs, target in train_loader:
            obs = obs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            prediction, mu, logvar = model(obs, target)
            loss, metrics = act_loss(
                prediction, target, mu, logvar, kl_weight=args.kl_weight
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            n = len(obs)
            accumulated += loss.item() * n
            accumulated_recon += metrics["reconstruction_l1"].item() * n
            accumulated_kl += metrics["kl"].item() * n
            observed += n

        val_metrics = evaluate(model, val_loader, device, normalization)
        val_l1 = val_metrics["inference_l1_normalized"]
        row = {
            "epoch": epoch,
            "train_loss": accumulated / observed,
            "train_reconstruction_l1": accumulated_recon / observed,
            "train_kl": accumulated_kl / observed,
            "val_inference_l1_normalized": val_l1,
            "val_linear_velocity_rmse_mps": val_metrics["linear_velocity_rmse_mps"],
            "val_angular_velocity_rmse_radps": val_metrics["angular_velocity_rmse_radps"],
        }
        rows.append(row)
        print(
            f"Epoch {epoch:03d} | train={row['train_loss']:.5f} | "
            f"val inference L1={val_l1:.5f} | KL={row['train_kl']:.5f}"
        )
        # Select checkpoint based on the same z=0 inference path used at deployment.
        if val_l1 < best_val - 1e-6:
            best_val = val_l1
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_config": asdict(config),
                    "normalization": {
                        k: torch.from_numpy(v.copy())
                        for k, v in normalization.items()
                    },
                    "seed": args.seed,
                    "epoch": epoch,
                    "best_val_inference_l1": best_val,
                    "control_dt_s": dataset_meta["control_dt_s"],
                    "history_len": config.history_len,
                    "action_chunk_size": config.chunk_size,
                    "execution_steps": 4,
                    "kl_weight": args.kl_weight,
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping: {bad_epochs} epochs without improvement")
                break

    # PyTorch 2.6+ defaults to weights_only=True; the checkpoint contains
    # ordinary tensors and primitive configuration values only.
    saved = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, normalization)
    with history_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "stage": "P8.3",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "train_windows": len(train_loader.dataset),
        "val_windows": len(val_loader.dataset),
        "test_windows": len(test_loader.dataset),
        "train_episodes": dataset_meta["splits"]["train"]["episode_count"],
        "val_episodes": dataset_meta["splits"]["val"]["episode_count"],
        "test_episodes": dataset_meta["splits"]["test"]["episode_count"],
        "model_parameters": n_params,
        "config": asdict(config),
        "training_options": vars(args),
        "epochs_completed": len(rows),
        "best_epoch": best_epoch,
        "best_val_inference_l1_normalized": best_val,
        "test_metrics": test_metrics,
        "note": "Test metrics are OFFLINE deterministic z=0 action prediction, not PyBullet navigation success.",
        "model_file": str(best_path.relative_to(ROOT)),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\n=== P8.3 TRAINING COMPLETE ===")
    print("Best epoch:", best_epoch)
    print("Best validation inference L1:", round(best_val, 6))
    print("Test velocity RMSE (m/s):", round(test_metrics["linear_velocity_rmse_mps"], 6))
    print("Test yaw-rate RMSE (rad/s):", round(test_metrics["angular_velocity_rmse_radps"], 6))
    print("Checkpoint:", best_path)
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
