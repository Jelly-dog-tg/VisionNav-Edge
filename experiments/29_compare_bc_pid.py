
"""P5-7.4: Compare BC and PID benchmark results."""

import csv
import json
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "logs" / "p5_evaluation"


def load_results(filename):
    path = RESULT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    results = {}

    for row in rows:
        episode_id = int(row["episode_id"])

        results[episode_id] = {
            "success": row["success"].lower() == "true",
            "collision": row["collision"].lower() == "true",
            "timeout": row["timeout"].lower() == "true",
            "steps": int(row["steps"]),
            "distance": float(row["actual_travel_distance_m"]),
            "error": float(row["final_goal_error_m"]),
            "time": float(row["elapsed_time_s"]),
        }

    return results


def calculate_metrics(results):
    rows = list(results.values())
    successful = [r for r in rows if r["success"]]

    def average(items, key):
        return mean([r[key] for r in items]) if items else None

    return {
        "total_tasks": len(rows),
        "success_count": len(successful),
        "success_rate": len(successful) / len(rows),
        "collision_count": sum(r["collision"] for r in rows),
        "collision_rate": sum(r["collision"] for r in rows) / len(rows),
        "timeout_count": sum(r["timeout"] for r in rows),

        "all_tasks": {
            "mean_steps": average(rows, "steps"),
            "mean_distance_m": average(rows, "distance"),
            "mean_error_m": average(rows, "error"),
            "mean_time_s": average(rows, "time"),
        },

        "successful_tasks_only": {
            "mean_steps": average(successful, "steps"),
            "mean_distance_m": average(successful, "distance"),
            "mean_error_m": average(successful, "error"),
            "mean_time_s": average(successful, "time"),
        },
    }


def main():

    bc = load_results("bc_results.csv")
    pid = load_results("pid_results.csv")

    if set(bc) != set(pid):
        raise ValueError(
            "BC and PID task IDs do not match."
        )

    task_ids = sorted(bc)

    # Compare task outcomes individually.
    both_success = []
    bc_only = []
    pid_only = []
    both_failed = []

    task_comparisons = []

    for task_id in task_ids:

        b = bc[task_id]
        p = pid[task_id]

        if b["success"] and p["success"]:
            both_success.append(task_id)

        elif b["success"]:
            bc_only.append(task_id)

        elif p["success"]:
            pid_only.append(task_id)

        else:
            both_failed.append(task_id)

        task_comparisons.append({
            "episode_id": task_id,
            "bc_success": b["success"],
            "pid_success": p["success"],
            "bc_collision": b["collision"],
            "pid_collision": p["collision"],
            "bc_steps": b["steps"],
            "pid_steps": p["steps"],
            "bc_distance_m": b["distance"],
            "pid_distance_m": p["distance"],
            "bc_error_m": b["error"],
            "pid_error_m": p["error"],
            "bc_time_s": b["time"],
            "pid_time_s": p["time"],
        })

    # Compare efficiency on identical successful tasks.
    paired = []

    for task_id in both_success:

        b = bc[task_id]
        p = pid[task_id]

        paired.append({
            "episode_id": task_id,
            "steps_delta_bc_minus_pid": b["steps"] - p["steps"],
            "distance_delta_bc_minus_pid": b["distance"] - p["distance"],
            "error_delta_bc_minus_pid": b["error"] - p["error"],
            "time_delta_bc_minus_pid": b["time"] - p["time"],
        })

    def paired_statistics(key):

        values = [r[key] for r in paired]

        if not values:
            return None

        return {
            "mean_bc_minus_pid": mean(values),
            "bc_better_count": sum(v < 0 for v in values),
            "pid_better_count": sum(v > 0 for v in values),
            "tie_count": sum(v == 0 for v in values),
        }

    summary = {
        "total_tasks": len(task_ids),

        "bc": calculate_metrics(bc),
        "pid": calculate_metrics(pid),

        "task_outcomes": {
            "both_success": len(both_success),
            "bc_only_success": len(bc_only),
            "pid_only_success": len(pid_only),
            "both_failed": len(both_failed),
            "bc_only_success_ids": bc_only,
            "pid_only_success_ids": pid_only,
            "both_failed_ids": both_failed,
        },

        "paired_successful_task_count": len(paired),

        "paired_efficiency_comparison": {
            "steps": paired_statistics(
                "steps_delta_bc_minus_pid"
            ),
            "distance": paired_statistics(
                "distance_delta_bc_minus_pid"
            ),
            "error": paired_statistics(
                "error_delta_bc_minus_pid"
            ),
            "time": paired_statistics(
                "time_delta_bc_minus_pid"
            ),
        },
    }

    # Save detailed per-task comparison.
    task_path = RESULT_DIR / "comparison_by_task.csv"

    with task_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(task_comparisons[0].keys()),
        )

        writer.writeheader()
        writer.writerows(task_comparisons)

    # Save paired successful-task comparison.
    paired_path = RESULT_DIR / "paired_success_comparison.csv"

    with paired_path.open(
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        fields = [
            "episode_id",
            "steps_delta_bc_minus_pid",
            "distance_delta_bc_minus_pid",
            "error_delta_bc_minus_pid",
            "time_delta_bc_minus_pid",
        ]

        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(paired)

    # Save complete summary.
    summary_path = RESULT_DIR / "comparison.json"

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n========== BC vs PID COMPARISON ==========")

    print("Total tasks:", len(task_ids))
    print("Both successful:", len(both_success))
    print("BC only successful:", len(bc_only))
    print("PID only successful:", len(pid_only))
    print("Both failed:", len(both_failed))

    print("\nSuccess rates:")
    print(f"BC:  {summary['bc']['success_rate']:.1%}")
    print(f"PID: {summary['pid']['success_rate']:.1%}")

    print("\nSuccessful tasks only:")

    for metric, values in summary[
        "paired_efficiency_comparison"
    ].items():

        if values is not None:
            print(
                f"{metric}: "
                f"BC better={values['bc_better_count']}, "
                f"PID better={values['pid_better_count']}, "
                f"Tie={values['tie_count']}, "
                f"Mean delta={values['mean_bc_minus_pid']:.4f}"
            )

    print("\nSaved:", task_path)
    print("Saved:", paired_path)
    print("Saved:", summary_path)

    print("\nP5-7.4 COMPARISON COMPLETED")


if __name__ == "__main__":
    main()
