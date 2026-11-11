"""P8.6-B | Finetune ACT V2 on successful BC+PPO teacher rollouts.

- ACT V2 pretrained weights are the initialization (NOT retrained from scratch).
- Teacher trajectories carry the main supervision: 8x40 -> 16x2.
- A smaller P5 PID BC batch is used as retention regularization.
- Optimize the exact deterministic deployment branch (z=0), with emphasis
  on its first four executed actions.
- Episode-level train/val/test splits and original BC normalization reused.
- Original checkpoints and original test data are not changed.

From project root:
  python experiments/p8_train_act_distill.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visionnav.learning.act_model import ACTConfig, ACTPolicy
from visionnav.learning.act_dataset import build_split_windows
from experiments.p8_train_act_v2 import train_objective, val_metrics

TEACHER_DIR = ROOT / "logs" / "p8_ppo_teacher_distillation"
PID_ACT_DIR = ROOT / "logs" / "p8_act_dataset"
BASE_MODEL = ROOT / "models" / "act_v2" / "act_best.pth"
OUTPUT_MODEL = ROOT / "models" / "act_distilled" / "act_best.pth"
OUTPUT_LOG = ROOT / "logs" / "p8_act_distillation"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_normalization():
    with np.load(PID_ACT_DIR / "normalization.npz", allow_pickle=False) as f:
        return {k: f[k].astype(np.float32).copy() for k in f.files}


def load_teacher_split(split, norm, teacher_dir):
    split_file = json.loads((teacher_dir / "episode_splits.json").read_text(encoding="utf-8"))
    ids = split_file[f"{split}_episode_ids"]
    observations = np.load(teacher_dir / "teacher_observations.npy", allow_pickle=False)
    actions = np.load(teacher_dir / "teacher_actions.npy", allow_pickle=False)
    eids = np.load(teacher_dir / "teacher_episode_ids.npy", allow_pickle=False)
    x, y, window_eids, anchors, info = build_split_windows(
        observations, actions, eids, ids,
    )
    x = ((x - norm["obs_mean"][None, None, :]) /
         norm["obs_std"][None, None, :]).astype(np.float32)
    y = ((y - norm["action_mean"][None, None, :]) /
         norm["action_std"][None, None, :]).astype(np.float32)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError("Non-finite teacher windows after BC normalization")
    return TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), info


def load_pid_split(split):
    x = np.load(PID_ACT_DIR / f"{split}_observations.npy", allow_pickle=False)
    y = np.load(PID_ACT_DIR / f"{split}_action_chunks.npy", allow_pickle=False)
    return TensorDataset(torch.from_numpy(x.copy()), torch.from_numpy(y.copy()))


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    p = argparse.ArgumentParser(description="P8.6 ACT teacher-policy distillation")
    p.add_argument("--teacher-dir", type=Path, default=TEACHER_DIR)
    p.add_argument("--base-model", type=Path, default=BASE_MODEL)
    p.add_argument("--output-model", type=Path, default=OUTPUT_MODEL)
    p.add_argument("--output-log", type=Path, default=OUTPUT_LOG)
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--pid-retention-weight", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = p.parse_args()
    if args.epochs < 1 or args.batch_size < 2 or args.patience < 1 or args.lr <= 0:
        p.error("epochs, patience, lr positive, batch-size >= 2")
    if not (0 <= args.pid_retention_weight <= 1):
        p.error("pid-retention-weight must be in [0,1]")
    teacher_dir = args.teacher_dir if args.teacher_dir.is_absolute() else ROOT / args.teacher_dir
    base_model = args.base_model if args.base_model.is_absolute() else ROOT / args.base_model
    output_model = args.output_model if args.output_model.is_absolute() else ROOT / args.output_model
    output_log = args.output_log if args.output_log.is_absolute() else ROOT / args.output_log
    output_history = output_log / "training_history.csv"
    output_summary = output_log / "training_summary.json"
    for path in [output_model, output_history, output_summary]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing experiment: {path}")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but not available")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    set_seed(args.seed)
    norm = load_normalization()
    base = torch.load(base_model, map_location="cpu", weights_only=True)
    if base.get("training_variant") != "deployment_prior_aligned_v2":
        raise ValueError("Base model must be ACT V2, deployment-prior-aligned")
    if "normalization" in base:
        for k in ["obs_mean", "obs_std", "action_mean", "action_std"]:
            value = base["normalization"][k]
            value = value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
            if not np.allclose(value, norm[k], atol=1e-6, rtol=1e-6):
                raise ValueError(f"Base ACT V2 normalization mismatch: {k}")
    config = ACTConfig(**base["model_config"])
    if (config.history_len, config.obs_dim, config.chunk_size, config.action_dim) != (8, 40, 16, 2):
        raise ValueError("Only [8,40] -> [16,2] ACT supported")
    model = ACTPolicy(config).to(device)
    model.load_state_dict(base["model_state_dict"])
    norm_copy = {k: torch.as_tensor(v.copy()) for k, v in norm.items()}

    train_ds, train_meta = load_teacher_split("train", norm, teacher_dir)
    val_ds, val_meta = load_teacher_split("val", norm, teacher_dir)
    test_ds, test_meta = load_teacher_split("test", norm, teacher_dir)
    pid_train_ds = load_pid_split("train")
    pid_val_ds = load_pid_split("val")

    teacher_train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    teacher_val = DataLoader(val_ds, batch_size=args.batch_size * 2)
    teacher_test = DataLoader(test_ds, batch_size=args.batch_size * 2)
    pid_train = DataLoader(pid_train_ds, batch_size=max(16, args.batch_size // 2), shuffle=True)
    pid_val = DataLoader(pid_val_ds, batch_size=args.batch_size * 2)
    pid_iter = iter(pid_train)
    baseline_pid_val = val_metrics(model, pid_val, device, norm, 4)["first4_l1_normalized"]
    baseline_teacher_val = val_metrics(model, teacher_val, device, norm, 4)["first4_l1_normalized"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_log.mkdir(parents=True, exist_ok=True)
    best_score = float("inf")
    best_epoch = 0
    patience_count = 0
    history = []
    print(f"P8.6 ACT DISTILLATION | device={device} | teacher train={len(train_ds)} "
          f"val={len(val_ds)} test={len(test_ds)} | PID retention={len(pid_train_ds)}", flush=True)
    print(f"Initial teacher val first4={baseline_teacher_val:.4f}; "
          f"PID val first4={baseline_pid_val:.4f}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        weighted_loss = 0.0
        seen = 0
        for obs_teacher, act_teacher in teacher_train:
            obs_teacher, act_teacher = obs_teacher.to(device), act_teacher.to(device)
            try:
                obs_pid, act_pid = next(pid_iter)
            except StopIteration:
                pid_iter = iter(pid_train)
                obs_pid, act_pid = next(pid_iter)
            obs_pid, act_pid = obs_pid.to(device), act_pid.to(device)

            loss_teacher, metrics_teacher = train_objective(
                model, obs_teacher, act_teacher, executed=4, early_weight=2.0,
                posterior_weight=.10, kl_weight=.02, temporal_weight=.03,
            )
            if args.pid_retention_weight:
                loss_pid, _ = train_objective(
                    model, obs_pid, act_pid, executed=4, early_weight=2.0,
                    posterior_weight=.10, kl_weight=.02, temporal_weight=.03,
                )
                loss = loss_teacher + args.pid_retention_weight * loss_pid
            else:
                loss = loss_teacher
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite ACT distillation loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            weighted_loss += float(loss.item()) * len(obs_teacher)
            seen += len(obs_teacher)

        # Validation only: never select checkpoint on held-out teacher test.
        tv = val_metrics(model, teacher_val, device, norm, 4)
        pv = val_metrics(model, pid_val, device, norm, 4)
        # Mild retention penalty only when the PID validation error degrades.
        retention_penalty = .2 * max(0., pv["first4_l1_normalized"] - baseline_pid_val)
        selection_score = tv["first4_l1_normalized"] + retention_penalty
        row = {
            "epoch": epoch, "train_weighted_loss": weighted_loss / seen,
            "teacher_val_first4_l1": tv["first4_l1_normalized"],
            "teacher_val_all16_l1": tv["all16_l1_normalized"],
            "teacher_val_linear_rmse_mps": tv["linear_velocity_rmse_mps"],
            "teacher_val_angular_rmse_radps": tv["angular_velocity_rmse_radps"],
            "pid_val_first4_l1": pv["first4_l1_normalized"],
            "selection_score": selection_score,
        }
        history.append(row)
        print(f"epoch {epoch:03d}: teacher_val_first4={tv['first4_l1_normalized']:.4f} "
              f"pid_val_first4={pv['first4_l1_normalized']:.4f} "
              f"score={selection_score:.4f}", flush=True)
        if selection_score < best_score - 1e-5:
            best_score = selection_score
            best_epoch = epoch
            patience_count = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": asdict(config),
                "normalization": norm_copy,
                "epoch": epoch, "seed": args.seed,
                "best_val_first4_inference_l1": tv["first4_l1_normalized"],
                "history_len": 8, "action_chunk_size": 16,
                "execution_steps": 4, "control_dt_s": 10/240,
                "training_variant": "bc_ppo_teacher_distilled_v1",
                "base_model_sha256": sha256_file(base_model),
                "teacher_dataset_summary": str(teacher_dir / "collection_summary.json"),
            }, output_model)
        else:
            patience_count += 1
            if patience_count >= args.patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    best = torch.load(output_model, map_location="cpu", weights_only=True)
    model.load_state_dict(best["model_state_dict"])
    teacher_validation = val_metrics(model, teacher_val, device, norm, 4)
    teacher_test_result = val_metrics(model, teacher_test, device, norm, 4)
    pid_validation = val_metrics(model, pid_val, device, norm, 4)
    with output_history.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    summary = {
        "stage": "P8.6-ACT-teacher-distillation",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "teacher_type": "BC+PPO V3 plus explicit labeled terminal supervisor",
        "pretrained_model": str(base_model),
        "pretrained_model_sha256": sha256_file(base_model),
        "best_epoch": best_epoch, "epochs_completed": len(history),
        "best_model_path": str(output_model),
        "teacher_train_windows": len(train_ds),
        "teacher_val_windows": len(val_ds),
        "teacher_test_windows": len(test_ds),
        "original_pid_retention_train_windows": len(pid_train_ds),
        "baseline_teacher_val_first4_l1": baseline_teacher_val,
        "baseline_pid_val_first4_l1": baseline_pid_val,
        "distilled_teacher_val_metrics": teacher_validation,
        "distilled_teacher_test_metrics": teacher_test_result,
        "distilled_pid_val_metrics": pid_validation,
        "config": asdict(config),
        "options": {**vars(args), "teacher_dir": str(teacher_dir),
                    "base_model": str(base_model), "output_model": str(output_model),
                    "output_log": str(output_log)},
        "warning": "Offline imitation error is not navigation success; evaluate identical held-out tasks with identical terminal supervisor",
    }
    output_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("\n=== P8.6 DISTILLATION COMPLETE ===", flush=True)
    print("Best epoch:", best_epoch)
    print("Teacher val first4:", round(teacher_validation["first4_l1_normalized"], 5))
    print("Teacher test first4:", round(teacher_test_result["first4_l1_normalized"], 5))
    print("PID val first4:", round(pid_validation["first4_l1_normalized"], 5))
    print("Model:", output_model)
    print("Summary:", output_summary)


if __name__ == "__main__":
    main()
