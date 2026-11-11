"""P8.3-V2: deployment-prior-aligned ACT-style CVAE training.

Changes from P8.3:
- Backpropagate directly through the deterministic z=0 inference branch.
- Emphasize the first four actions that the ACT controller actually executes.
- Retain the posterior CVAE as a *small* auxiliary objective with stronger KL.
- Select checkpoint on the validation first-4-action L1 from the deployable branch.
- Freeze the same data split and BC normalization. Never overwrite original P8.3.

Run from the repository root:
    python experiments/p8_train_act_v2.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visionnav.learning.act_model import ACTConfig, ACTPolicy
from experiments.p8_train_act import DATA_DIR, evaluate, get_loader, set_seed

MODEL_DIR = ROOT / "models" / "act_v2"
LOG_DIR = ROOT / "logs" / "p8_act_training_v2"


def train_objective(
    model: ACTPolicy,
    obs: torch.Tensor,
    target: torch.Tensor,
    executed: int,
    early_weight: float,
    posterior_weight: float,
    kl_weight: float,
    temporal_weight: float,
):
    # This EXACT deterministic inference path is used online in ACTChunkController.
    prior, _, _ = model(obs)
    err = (prior - target).abs()
    steps = target.shape[1]
    weights = torch.ones(steps, device=target.device, dtype=target.dtype)
    weights[:executed] = early_weight
    prior_l1 = (err * weights.view(1, -1, 1)).mean() / weights.mean()

    # Preserve ACT's CVAE pathway as an auxiliary objective, not as the main task.
    posterior, mu, logvar = model(obs, target)
    posterior_l1 = nn.functional.l1_loss(posterior, target)
    kl_per_dim = -0.5 * (1 + logvar - mu.square() - logvar.exp())
    kl = kl_per_dim.mean()

    # Match the expert's *changes* in velocity, not just its absolute values.
    prior_delta = prior[:, 1:] - prior[:, :-1]
    expert_delta = target[:, 1:] - target[:, :-1]
    temporal_l1 = nn.functional.l1_loss(prior_delta, expert_delta)

    loss = (prior_l1 + posterior_weight * posterior_l1
            + kl_weight * kl + temporal_weight * temporal_l1)
    return loss, {
        "deploy_prior_l1": float(prior_l1.detach()),
        "posterior_l1": float(posterior_l1.detach()),
        "kl": float(kl.detach()),
        "temporal_l1": float(temporal_l1.detach()),
    }


@torch.inference_mode()
def val_metrics(model: ACTPolicy, loader, device: torch.device, norm, first_steps: int):
    model.eval()
    accum_first = 0.0
    accum_all = 0.0
    first_count = 0
    all_count = 0
    # Keep physical RMSE comparable with V1's offline report.
    rmse_sq_sum = torch.zeros(2, dtype=torch.float64)
    n_physical_actions = 0
    action_std = torch.as_tensor(norm["action_std"], device=device)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred, _, _ = model(x)
        err = pred - y
        accum_first += err[:, :first_steps].abs().sum().item()
        first_count += err[:, :first_steps].numel()
        accum_all += err.abs().sum().item()
        all_count += err.numel()
        physical_err = err * action_std.view(1, 1, -1)
        rmse_sq_sum += physical_err.square().sum(dim=(0, 1)).cpu().to(torch.float64)
        n_physical_actions += y.shape[0] * y.shape[1]

    if first_count == 0 or all_count == 0:
        raise ValueError("Empty validation split")
    rmse = torch.sqrt(rmse_sq_sum / n_physical_actions).tolist()
    return {
        "first4_l1_normalized": accum_first / first_count,
        "all16_l1_normalized": accum_all / all_count,
        "linear_velocity_rmse_mps": float(rmse[0]),
        "angular_velocity_rmse_radps": float(rmse[1]),
    }


def main():
    p = argparse.ArgumentParser(description="P8.3-V2 prior-aligned ACT training")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--first-steps", type=int, default=4)
    p.add_argument("--early-weight", type=float, default=2.0)
    p.add_argument("--posterior-weight", type=float, default=0.10)
    p.add_argument("--kl-weight", type=float, default=0.02)
    p.add_argument("--temporal-weight", type=float, default=0.03)
    args = p.parse_args()

    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        p.error("epochs, batch-size and patience must be positive")
    if not 1 <= args.first_steps <= 16 or args.early_weight < 1:
        p.error("first-steps must be in 1..16 and early-weight must be >=1")
    if args.lr <= 0 or min(args.posterior_weight, args.kl_weight,
                           args.temporal_weight) < 0:
        p.error("Learning rate must be positive and regularization weights nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested, but torch.cuda.is_available() is False")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    set_seed(args.seed)
    train_loader = get_loader("train", args.batch_size, device, args.seed)
    val_loader = get_loader("val", args.batch_size, device, args.seed)
    test_loader = get_loader("test", args.batch_size, device, args.seed)
    with np.load(DATA_DIR / "normalization.npz", allow_pickle=False) as f:
        normalization = {k: f[k].copy() for k in f.files}
    with (DATA_DIR / "act_dataset_metadata.json").open(encoding="utf-8") as f:
        dataset_meta = json.load(f)

    config = ACTConfig()
    model = ACTPolicy(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    checkpoint_path = MODEL_DIR / "act_best.pth"
    history_path = LOG_DIR / "training_history.csv"
    summary_path = LOG_DIR / "training_summary.json"
    outputs = [checkpoint_path, history_path, summary_path]
    if any(path.exists() for path in outputs):
        raise FileExistsError("V2 training output already exists; old V1 and V2 runs are protected. "
                              "Do not delete a completed run unintentionally.")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    best = float("inf")
    best_epoch = 0
    bad_epochs = 0
    history = []
    print(f"P8.3-V2 | device={device} | params={sum(p.numel() for p in model.parameters()):,}")
    print(f"Train={len(train_loader.dataset)} Val={len(val_loader.dataset)} Test={len(test_loader.dataset)}")
    print("Selection metric: deterministic z=0 first-4-action validation L1")

    for epoch in range(1, args.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "deploy_prior_l1": 0.0, "posterior_l1": 0.0,
                  "kl": 0.0, "temporal_l1": 0.0}
        observed = 0
        for obs, target in train_loader:
            obs = obs.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss, parts = train_objective(
                model, obs, target, args.first_steps, args.early_weight,
                args.posterior_weight, args.kl_weight, args.temporal_weight,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss in epoch {epoch}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            n = len(obs)
            totals["loss"] += float(loss.detach()) * n
            for key, val in parts.items():
                totals[key] += val * n
            observed += n

        validation = val_metrics(model, val_loader, device, normalization, args.first_steps)
        row = {"epoch": epoch}
        row.update({"train_" + k: v / observed for k, v in totals.items()})
        row.update({"val_" + k: v for k, v in validation.items()})
        history.append(row)
        score = validation["first4_l1_normalized"]
        print(f"epoch {epoch:03d} | train prior={row['train_deploy_prior_l1']:.4f} | "
              f"val first4={score:.4f} | val all16={validation['all16_l1_normalized']:.4f} | "
              f"KL={row['train_kl']:.4f}", flush=True)

        if score < best - 1e-6:
            best = score
            best_epoch = epoch
            bad_epochs = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": asdict(config),
                "normalization": {k: torch.from_numpy(v.copy()) for k, v in normalization.items()},
                "seed": args.seed,
                "epoch": epoch,
                "best_val_first4_inference_l1": best,
                "control_dt_s": dataset_meta["control_dt_s"],
                "history_len": config.history_len,
                "action_chunk_size": config.chunk_size,
                "execution_steps": args.first_steps,
                "training_variant": "deployment_prior_aligned_v2",
            }, checkpoint_path)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model_state_dict"])
    final_val = val_metrics(model, val_loader, device, normalization, args.first_steps)
    final_test = val_metrics(model, test_loader, device, normalization, args.first_steps)
    with history_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "stage": "P8.3-V2", "created_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device), "train_windows": len(train_loader.dataset),
        "val_windows": len(val_loader.dataset), "test_windows": len(test_loader.dataset),
        "best_epoch": best_epoch, "epochs_completed": len(history),
        "best_val_first4_l1_normalized": best,
        "best_validation_metrics": final_val, "test_metrics": final_test,
        "config": asdict(config), "training_options": vars(args),
        "model_file": str(checkpoint_path.relative_to(ROOT)),
        "baseline_v1_unchanged": True,
        "note": "Offline deterministic z=0 prediction; navigation requires separate PyBullet rollout.",
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== P8.3-V2 COMPLETE ===")
    print("Best epoch:", best_epoch)
    print("Val first4:", round(final_val["first4_l1_normalized"], 6))
    print("Val all16:", round(final_val["all16_l1_normalized"], 6))
    print("Test linear RMSE:", round(final_test["linear_velocity_rmse_mps"], 6))
    print("Test angular RMSE:", round(final_test["angular_velocity_rmse_radps"], 6))
    print("Saved model:", checkpoint_path)
    print("Saved summary:", summary_path)


if __name__ == "__main__":
    main()
