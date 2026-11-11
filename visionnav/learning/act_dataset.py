
"""
P8.1 - ACT Sequential Dataset Builder

Input:
    8 consecutive observations, each with 40 dimensions.

Target:
    16 consecutive actions, each with 2 dimensions.

The original BC episode split and normalization statistics
are reused without modifying the source dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# ============================================================
# 1. Project configuration
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

SOURCE_DIR = ROOT / "logs" / "p5_dataset"
DEFAULT_OUTPUT_DIR = ROOT / "logs" / "p8_act_dataset"

OBS_DIM = 40
ACTION_DIM = 2

HISTORY_LEN = 8
ACTION_CHUNK_SIZE = 16
EXECUTION_STEPS = 4
REPLANNING_INTERVAL = 4

SPLIT_KEYS = {
    "train": "train_episode_ids",
    "val": "val_episode_ids",
    "test": "test_episode_ids",
}

SOURCE_FILES = [
    "observations.npy",
    "actions.npy",
    "episode_ids.npy",
    "metadata.json",
    "dataset_splits.json",
    "normalization.npz",
]

OUTPUT_FILES = [
    "train_observations.npy",
    "train_action_chunks.npy",
    "val_observations.npy",
    "val_action_chunks.npy",
    "test_observations.npy",
    "test_action_chunks.npy",
    "train_episode_ids.npy",
    "val_episode_ids.npy",
    "test_episode_ids.npy",
    "train_anchor_indices.npy",
    "val_anchor_indices.npy",
    "test_anchor_indices.npy",
    "normalization.npz",
    "source_dataset_splits.json",
    "act_dataset_metadata.json",
    "act_dataset_report.md",
]


# ============================================================
# 2. Basic validation
# ============================================================

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()

    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def validate_integer_ids(values, name: str):
    values = np.asarray(values)

    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")

    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{name} must be numeric")

    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or Inf")

    if not np.all(values == np.round(values)):
        raise ValueError(f"{name} contains non-integer IDs")

    return values.astype(np.int64)


def validate_episode_contiguity(episode_ids):
    """
    Each Episode must occupy one continuous segment
    in the original dataset.
    """

    if len(episode_ids) == 0:
        raise ValueError("Empty episode_ids")

    seen = set()
    previous = int(episode_ids[0])
    seen.add(previous)

    for value in episode_ids[1:]:
        current = int(value)

        if current != previous:
            if current in seen:
                raise ValueError(
                    f"Episode {current} appears in multiple segments"
                )

            seen.add(current)
            previous = current


def validate_splits(actual_ids, split_payload):
    actual_ids = set(map(int, actual_ids))

    splits = {}

    for split_name, key in SPLIT_KEYS.items():

        if key not in split_payload:
            raise KeyError(f"Missing split key: {key}")

        ids = validate_integer_ids(
            split_payload[key],
            key,
        )

        if len(ids) != len(set(ids.tolist())):
            raise ValueError(
                f"Duplicate Episode IDs in {split_name}"
            )

        splits[split_name] = ids.tolist()

    train = set(splits["train"])
    val = set(splits["val"])
    test = set(splits["test"])

    if train & val:
        raise ValueError("Train/Val Episode leakage")

    if train & test:
        raise ValueError("Train/Test Episode leakage")

    if val & test:
        raise ValueError("Val/Test Episode leakage")

    assigned = train | val | test

    if assigned != actual_ids:
        raise ValueError(
            "Episode split mismatch. "
            f"Unassigned={sorted(actual_ids - assigned)}, "
            f"Unknown={sorted(assigned - actual_ids)}"
        )

    return splits


# ============================================================
# 3. ACT sliding-window construction
# ============================================================

def build_split_windows(
    observations,
    actions,
    episode_ids,
    selected_episode_ids,
    history_len=HISTORY_LEN,
    action_chunk_size=ACTION_CHUNK_SIZE,
):
    """
    Build ACT sequences without crossing Episode boundaries.

    Input:
        observations[t-history_len+1 : t+1]

    Target:
        actions[t : t+action_chunk_size]

    Returns:
        observation_windows
        action_windows
        sample_episode_ids
        anchor_indices
        statistics
    """

    observations = np.asarray(
        observations,
        dtype=np.float32,
    )

    actions = np.asarray(
        actions,
        dtype=np.float32,
    )

    episode_ids = np.asarray(
        episode_ids,
        dtype=np.int64,
    )

    selected = set(map(int, selected_episode_ids))

    actual = set(np.unique(episode_ids).tolist())

    unknown = selected - actual

    if unknown:
        raise ValueError(
            f"Unknown selected Episodes: {sorted(unknown)}"
        )

    obs_windows = []
    action_windows = []
    sample_episode_ids = []
    anchor_indices = []

    episode_lengths = {}

    excluded_anchor_steps = 0
    short_episode_count = 0

    # Preserve original Episode order.
    episode_order = list(dict.fromkeys(episode_ids.tolist()))

    for eid in episode_order:

        if eid not in selected:
            continue

        positions = np.flatnonzero(episode_ids == eid)

        start = int(positions[0])
        length = len(positions)

        if positions[-1] - start + 1 != length:
            raise ValueError(
                f"Episode {eid} is not contiguous"
            )

        episode_lengths[str(eid)] = length

        valid_count = max(
            0,
            length - history_len - action_chunk_size + 2,
        )

        excluded_anchor_steps += length - valid_count

        if valid_count == 0:
            short_episode_count += 1
            continue

        # First valid anchor: history_len - 1
        # Last valid anchor: length - action_chunk_size
        for local_t in range(
            history_len - 1,
            length - action_chunk_size + 1,
        ):

            anchor = start + local_t

            obs_start = anchor - history_len + 1
            obs_end = anchor + 1

            action_start = anchor
            action_end = anchor + action_chunk_size

            obs_window = observations[
                obs_start:obs_end
            ]

            action_window = actions[
                action_start:action_end
            ]

            if obs_window.shape != (
                history_len,
                OBS_DIM,
            ):
                raise RuntimeError(
                    f"Invalid observation window: {obs_window.shape}"
                )

            if action_window.shape != (
                action_chunk_size,
                ACTION_DIM,
            ):
                raise RuntimeError(
                    f"Invalid action window: {action_window.shape}"
                )

            obs_windows.append(obs_window)
            action_windows.append(action_window)

            sample_episode_ids.append(eid)
            anchor_indices.append(anchor)

    if not obs_windows:
        raise ValueError(
            "No valid ACT windows generated"
        )

    obs_windows = np.stack(
        obs_windows
    ).astype(np.float32)

    action_windows = np.stack(
        action_windows
    ).astype(np.float32)

    sample_episode_ids = np.asarray(
        sample_episode_ids,
        dtype=np.int64,
    )

    anchor_indices = np.asarray(
        anchor_indices,
        dtype=np.int64,
    )

    statistics = {
        "episode_count": len(episode_lengths),
        "episode_lengths": episode_lengths,
        "window_count": len(obs_windows),
        "short_episode_count": short_episode_count,
        "excluded_anchor_steps": excluded_anchor_steps,
    }

    return (
        obs_windows,
        action_windows,
        sample_episode_ids,
        anchor_indices,
        statistics,
    )


# ============================================================
# 4. Normalization
# ============================================================

def load_normalization(path: Path):

    required = [
        "obs_mean",
        "obs_std",
        "action_mean",
        "action_std",
    ]

    with np.load(path, allow_pickle=False) as data:

        missing = [
            key for key in required
            if key not in data.files
        ]

        if missing:
            raise KeyError(
                f"Missing normalization parameters: {missing}"
            )

        stats = {
            key: np.asarray(data[key], dtype=np.float32).copy()
            for key in required
        }

    expected_shapes = {
        "obs_mean": (OBS_DIM,),
        "obs_std": (OBS_DIM,),
        "action_mean": (ACTION_DIM,),
        "action_std": (ACTION_DIM,),
    }

    for key, expected in expected_shapes.items():

        if stats[key].shape != expected:
            raise ValueError(
                f"{key}: expected {expected}, "
                f"got {stats[key].shape}"
            )

        if not np.isfinite(stats[key]).all():
            raise ValueError(
                f"{key} contains NaN or Inf"
            )

    if np.any(stats["obs_std"] <= 0):
        raise ValueError("Invalid observation std")

    if np.any(stats["action_std"] <= 0):
        raise ValueError("Invalid action std")

    return stats


def normalize_windows(obs, actions, stats):

    obs_norm = (
        obs - stats["obs_mean"][None, None, :]
    ) / stats["obs_std"][None, None, :]

    action_norm = (
        actions - stats["action_mean"][None, None, :]
    ) / stats["action_std"][None, None, :]

    obs_norm = obs_norm.astype(np.float32)
    action_norm = action_norm.astype(np.float32)

    if not np.isfinite(obs_norm).all():
        raise ValueError("Normalized observations invalid")

    if not np.isfinite(action_norm).all():
        raise ValueError("Normalized actions invalid")

    return obs_norm, action_norm


# ============================================================
# 5. Build and save ACT dataset
# ============================================================

def build_act_dataset(output_dir=None):

    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else DEFAULT_OUTPUT_DIR
    )

    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir

    print("\n========== P8.1 ACT DATASET ==========")

    # --------------------------------------------------------
    # Load source data
    # --------------------------------------------------------

    for name in SOURCE_FILES:

        path = SOURCE_DIR / name

        if not path.is_file():
            raise FileNotFoundError(path)

    observations = np.load(
        SOURCE_DIR / "observations.npy",
        allow_pickle=False,
    )

    actions = np.load(
        SOURCE_DIR / "actions.npy",
        allow_pickle=False,
    )

    raw_episode_ids = np.load(
        SOURCE_DIR / "episode_ids.npy",
        allow_pickle=False,
    )

    episode_ids = validate_integer_ids(
        raw_episode_ids,
        "episode_ids",
    )

    metadata = load_json(
        SOURCE_DIR / "metadata.json"
    )

    split_payload = load_json(
        SOURCE_DIR / "dataset_splits.json"
    )

    stats = load_normalization(
        SOURCE_DIR / "normalization.npz"
    )

    # --------------------------------------------------------
    # Validate source arrays
    # --------------------------------------------------------

    if observations.ndim != 2:
        raise ValueError("Observations must be 2D")

    if observations.shape[1] != OBS_DIM:
        raise ValueError(
            f"Expected {OBS_DIM} observation dimensions"
        )

    if actions.ndim != 2:
        raise ValueError("Actions must be 2D")

    if actions.shape[1] != ACTION_DIM:
        raise ValueError(
            f"Expected {ACTION_DIM} action dimensions"
        )

    if len(observations) != len(actions):
        raise ValueError("Observation/action length mismatch")

    if len(observations) != len(episode_ids):
        raise ValueError("Episode ID length mismatch")

    for name, array in [
        ("observations", observations),
        ("actions", actions),
    ]:
        if not np.isfinite(array).all():
            raise ValueError(
                f"{name} contains NaN or Inf"
            )

    validate_episode_contiguity(episode_ids)

    actual_episode_ids = np.unique(episode_ids)

    splits = validate_splits(
        actual_episode_ids,
        split_payload,
    )

    control_dt = float(metadata["control_dt_s"])

    if not np.isfinite(control_dt) or control_dt <= 0:
        raise ValueError("Invalid control_dt_s")

    metadata_episode_count = metadata.get(
        "episode_count"
    )

    print("Source samples:", len(observations))
    print("Actual Episodes:", len(actual_episode_ids))
    print("Metadata Episode count:", metadata_episode_count)
    print("Control dt:", control_dt)

    # --------------------------------------------------------
    # Build each split independently
    # --------------------------------------------------------

    results = {}

    for split_name, selected_ids in splits.items():

        print(f"\nBuilding {split_name}...")

        result = build_split_windows(
            observations,
            actions,
            episode_ids,
            selected_ids,
        )

        (
            obs_windows,
            action_windows,
            sample_ids,
            anchors,
            window_stats,
        ) = result

        # Reuse original BC training-only statistics.
        obs_norm, action_norm = normalize_windows(
            obs_windows,
            action_windows,
            stats,
        )

        results[split_name] = {
            "observations": obs_norm,
            "actions": action_norm,
            "episode_ids": sample_ids,
            "anchor_indices": anchors,
            "statistics": window_stats,
        }

        print(
            f"{split_name}: "
            f"{len(obs_norm)} windows"
        )

    # --------------------------------------------------------
    # Output directory and overwrite protection
    # --------------------------------------------------------

    existing = [
        name for name in OUTPUT_FILES
        if (output_dir / name).exists()
    ]

    if existing:
        raise FileExistsError(
            f"Output files already exist in {output_dir}: "
            f"{existing}. Choose another --output-dir "
            "to preserve previous results."
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Save arrays and metadata
    # --------------------------------------------------------

    for split_name, result in results.items():

        np.save(
            output_dir / f"{split_name}_observations.npy",
            result["observations"],
        )

        np.save(
            output_dir / f"{split_name}_action_chunks.npy",
            result["actions"],
        )

        np.save(
            output_dir / f"{split_name}_episode_ids.npy",
            result["episode_ids"],
        )

        np.save(
            output_dir / f"{split_name}_anchor_indices.npy",
            result["anchor_indices"],
        )

    np.savez(
        output_dir / "normalization.npz",
        **stats,
    )

    with open(
        output_dir / "source_dataset_splits.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            split_payload,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # Source file hashes
    # --------------------------------------------------------

    source_hashes = {}

    for name in SOURCE_FILES:

        path = SOURCE_DIR / name

        source_hashes[name] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    split_metadata = {}

    for split_name, result in results.items():

        split_metadata[split_name] = {
            "episode_ids": splits[split_name],
            "episode_count": result["statistics"]["episode_count"],
            "window_count": result["statistics"]["window_count"],
            "observation_shape": list(
                result["observations"].shape
            ),
            "action_shape": list(
                result["actions"].shape
            ),
            "short_episode_count": result["statistics"][
                "short_episode_count"
            ],
            "excluded_anchor_steps": result["statistics"][
                "excluded_anchor_steps"
            ],
            "episode_lengths": result["statistics"][
                "episode_lengths"
            ],
        }

    total_windows = sum(
        item["window_count"]
        for item in split_metadata.values()
    )

    absent_ids = sorted(
        set(range(
            int(actual_episode_ids.min()),
            int(actual_episode_ids.max()) + 1,
        )) - set(actual_episode_ids.tolist())
    )

    metadata_output = {
        "stage": "P8.1",
        "project": "VisionNav-Edge",
        "created_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "source_directory": str(SOURCE_DIR.relative_to(ROOT)),
        "output_directory": str(output_dir.relative_to(ROOT)),

        "source_samples": len(observations),
        "actual_episode_count": len(actual_episode_ids),
        "actual_episode_ids": actual_episode_ids.tolist(),

        "metadata_episode_count": metadata_episode_count,
        "numeric_episode_ids_absent_from_arrays": absent_ids,

        "observation_dim": OBS_DIM,
        "action_dim": ACTION_DIM,

        "history_len": HISTORY_LEN,
        "action_chunk_size": ACTION_CHUNK_SIZE,
        "execution_steps": EXECUTION_STEPS,
        "replanning_interval": REPLANNING_INTERVAL,

        "control_dt_s": control_dt,

        "history_offsets": list(
            range(-(HISTORY_LEN - 1), 1)
        ),

        "action_offsets": list(
            range(ACTION_CHUNK_SIZE)
        ),

        "history_span_between_first_last_s": (
            HISTORY_LEN - 1
        ) * control_dt,

        "action_chunk_span_between_first_last_s": (
            ACTION_CHUNK_SIZE - 1
        ) * control_dt,

        "action_chunk_horizon_s": (
            ACTION_CHUNK_SIZE
        ) * control_dt,

        "execution_duration_s": (
            EXECUTION_STEPS
        ) * control_dt,

        "window_formula": (
            "max(episode_length - history_len "
            "- action_chunk_size + 2, 0)"
        ),

        "normalization": {
            "applied": True,
            "method": "(x - mean) / std",
            "source": "logs/p5_dataset/normalization.npz",
            "calculated_using_training_data_only": True,
            "inference_requires_action_denormalization": True,
        },

        "splits": split_metadata,

        "total_windows": total_windows,

        "data_leakage_check": {
            "episode_level_split": True,
            "cross_episode_windows": False,
            "train_val_overlap": False,
            "train_test_overlap": False,
            "val_test_overlap": False,
            "normalization_reused": True,
        },

        "source_file_hashes": source_hashes,
    }

    with open(
        output_dir / "act_dataset_metadata.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata_output,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # Generate report
    # --------------------------------------------------------

    lines = [
        "# P8.1 ACT Dataset Report",
        "",
        f"- Generated: {metadata_output['created_utc']}",
        f"- Source samples: {len(observations)}",
        f"- Actual Episodes: {len(actual_episode_ids)}",
        f"- Metadata Episode count: {metadata_episode_count}",
        f"- Control frequency: {1.0 / control_dt:.2f} Hz",
        f"- Observation dimension: {OBS_DIM}",
        f"- Action dimension: {ACTION_DIM}",
        f"- History length: {HISTORY_LEN}",
        f"- Action chunk length: {ACTION_CHUNK_SIZE}",
        "",
        "## Dataset Statistics",
        "",
        "| Split | Episodes | Windows | Observation shape | Action shape |",
        "|---|---:|---:|---|---|",
    ]

    for name, item in split_metadata.items():

        lines.append(
            f"| {name} | {item['episode_count']} "
            f"| {item['window_count']} "
            f"| {item['observation_shape']} "
            f"| {item['action_shape']} |"
        )

    lines.extend([
        "",
        f"**Total windows: {total_windows}**",
        "",
        "## Normalization",
        "",
        "Reused the existing BC normalization.npz.",
        "Observation and action statistics are applied separately.",
        "No validation or test statistics are recalculated.",
        "",
        "## Data Leakage",
        "",
        "- Episode-level split reused.",
        "- No cross-Episode windows.",
        "- No split overlap.",
        "- Original source data remains unchanged.",
        "",
        "## Time Alignment",
        "",
        "Input: observations[t-7:t+1].",
        "Target: actions[t:t+16].",
        "Each input window ends at the first predicted action time.",
        "",
        "## Source Integrity",
        "",
        "SHA256 hashes are recorded in act_dataset_metadata.json.",
        "",
        "## Result",
        "",
        "ACT sequential dataset construction completed.",
    ])

    report = "\n".join(lines)

    with open(
        output_dir / "act_dataset_report.md",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(report)

    print("\n========== P8.1 COMPLETE ==========")

    for name, item in split_metadata.items():
        print(
            f"{name}: "
            f"{item['window_count']} windows"
        )

    print("Total windows:", total_windows)
    print("Output:", output_dir)
    print("====================================")

    return metadata_output


# ============================================================
# 6. Synthetic self-tests
# ============================================================

def run_self_tests():

    print("\n========== ACT DATASET SELF TESTS ==========")

    # Test 1: Basic alignment.
    obs = np.arange(
        30 * OBS_DIM,
        dtype=np.float32,
    ).reshape(30, OBS_DIM)

    actions = np.arange(
        30 * ACTION_DIM,
        dtype=np.float32,
    ).reshape(30, ACTION_DIM)

    ids = np.zeros(30, dtype=np.int64)

    x, y, eids, anchors, info = build_split_windows(
        obs,
        actions,
        ids,
        [0],
    )

    assert x.shape == (8, 8, 40)
    assert y.shape == (8, 16, 2)

    assert anchors[0] == 7
    assert anchors[-1] == 14

    np.testing.assert_array_equal(
        x[0],
        obs[0:8],
    )

    np.testing.assert_array_equal(
        y[0],
        actions[7:23],
    )

    print("[PASS] Window shape and temporal alignment")

    # Test 2: Episode boundaries.
    obs2 = np.zeros((50, 40), dtype=np.float32)
    obs2[:25] = 10
    obs2[25:] = 20

    act2 = np.zeros((50, 2), dtype=np.float32)
    act2[:25] = 1
    act2[25:] = 2

    ids2 = np.array(
        [1] * 25 + [2] * 25,
        dtype=np.int64,
    )

    x2, y2, e2, a2, info2 = build_split_windows(
        obs2,
        act2,
        ids2,
        [1, 2],
    )

    assert len(x2) == 6
    assert np.all(e2[:3] == 1)
    assert np.all(e2[3:] == 2)

    assert np.all(x2[:3] == 10)
    assert np.all(x2[3:] == 20)

    assert np.all(y2[:3] == 1)
    assert np.all(y2[3:] == 2)

    print("[PASS] No cross-Episode windows")

    # Test 3: Short Episodes.
    obs3 = np.zeros((45, 40), dtype=np.float32)
    act3 = np.zeros((45, 2), dtype=np.float32)

    ids3 = np.array(
        [1] * 22 + [2] * 23,
        dtype=np.int64,
    )

    x3, y3, e3, a3, info3 = build_split_windows(
        obs3,
        act3,
        ids3,
        [1, 2],
    )

    assert len(x3) == 1
    assert e3[0] == 2
    assert info3["short_episode_count"] == 1

    print("[PASS] Short Episode handling")

    # Test 4: Split isolation.
    fake_splits = {
        "train_episode_ids": [1, 2],
        "val_episode_ids": [3],
        "test_episode_ids": [4],
    }

    validated = validate_splits(
        [1, 2, 3, 4],
        fake_splits,
    )

    assert len(validated["train"]) == 2

    print("[PASS] Episode-level split validation")

    # Test 5: Normalization.
    fake_stats = {
        "obs_mean": np.zeros(40, dtype=np.float32),
        "obs_std": np.ones(40, dtype=np.float32),
        "action_mean": np.zeros(2, dtype=np.float32),
        "action_std": np.ones(2, dtype=np.float32),
    }

    nx, ny = normalize_windows(
        x[:1],
        y[:1],
        fake_stats,
    )

    np.testing.assert_allclose(nx, x[:1])
    np.testing.assert_allclose(ny, y[:1])

    print("[PASS] Normalization interface")

    print("\nAll 5 synthetic tests passed.")


# ============================================================
# 7. Entry point
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="P8.1 ACT Dataset Builder"
    )

    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run synthetic unit tests only",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory",
    )

    args = parser.parse_args()

    if args.self_test:
        run_self_tests()
    else:
        build_act_dataset(args.output_dir)
