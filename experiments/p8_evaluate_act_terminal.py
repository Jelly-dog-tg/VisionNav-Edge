"""P8.4 follow-up: ACT V2 with explicitly disclosed goal-proximity stopping.

Run from project root:
  python experiments/p8_evaluate_act_terminal.py \
    --model models/act_v2/act_best.pth \
    --benchmark logs/p5_evaluation_bc+pid/benchmark_tasks.json \
    --max-tasks 3 --start-index 0 --seed 42

This is ACT + a deterministic terminal stopping supervisor; it is NOT an
ACT-only evaluation. The supervisor uses the known geometric navigation goal,
not a hidden visual target position. Original environment, collision checking,
and success criteria remain unchanged. No BC/PPO/Pure Pursuit fallback.
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from p8_evaluate_act import load_tasks, resolve_benchmark  # noqa: E402
from visionnav.learning.act_controller import ACTChunkController  # noqa: E402
from visionnav.learning.rl_navigation_env import (  # noqa: E402
    MAX_LINEAR_V,
    MAX_ANGULAR_W,
    RobotNavigationEnv,
)

OUTPUT_ROOT = ROOT / 'logs' / 'p8_act_evaluation_terminal'


def run_one(env, controller, task, seed, max_steps, task_index,
            brake_distance, release_distance, slow_distance):
    controller.reset()
    observation, initial = env.reset(seed=seed, options={'task': task})
    goal = np.asarray(env.goal_xy, dtype=np.float64)
    if goal.shape != (2,) or not np.isfinite(goal).all():
        raise ValueError('Valid geometric navigation goal required')
    pose = np.asarray(env.robot.get_pose()[:2], dtype=np.float64)
    initial_distance = float(np.linalg.norm(goal - pose))
    min_distance = initial_distance
    distance_travelled = 0.0
    total_reward = 0.0
    brake_active = False
    brake_events = 0
    slow_steps = 0
    brake_steps = 0
    steps_in_goal_zone = 0
    control_trace = []
    info = {}
    terminated = truncated = False
    t0 = time.perf_counter()

    for step in range(max_steps):
        # ACT still observes and predicts every step, including the brake phase.
        # The supervisor changes only the execution command near the *known*
        # geometric goal, never the simulated state or success criterion.
        predicted_action, diagnostics = controller.next_action(observation)
        action = predicted_action.copy()
        current_position = np.asarray(env.robot.get_pose()[:2], dtype=np.float64)
        distance_before = float(np.linalg.norm(goal - current_position))

        if brake_active and distance_before >= release_distance:
            brake_active = False
        if not brake_active and distance_before <= brake_distance:
            brake_active = True
            brake_events += 1

        if brake_active and distance_before >= release_distance:
            brake_active = False

        if not brake_active and distance_before <= brake_distance:
            brake_active = True
            brake_events += 1


        
        mode = 'act'

        if brake_active:
            # Stop only after entering the smaller final region.
            action = np.array([-1.0, 0.0], dtype=np.float32)
            brake_steps += 1
            mode = 'terminal_brake'

        elif distance_before <= slow_distance:
            # Limit speed during terminal approach.
            max_v = min(
                MAX_LINEAR_V,
                0.04 + 0.30 * distance_before
            )

            # Ensure the robot continues approaching the goal.
            min_creep_v = 0.025

            requested_v = (
                (float(action[0]) + 1.0)
                * 0.5 * MAX_LINEAR_V
            )

            executed_v = float(np.clip(
                requested_v,
                min_creep_v,
                max_v
            ))

            action[0] = (
                2.0 * executed_v / MAX_LINEAR_V - 1.0
            )

            if distance_before <= 0.12:
                mode = 'terminal_creep'
            else:
                mode = 'terminal_slow'

            slow_steps += 1


        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        new_position = np.asarray(env.robot.get_pose()[:2], dtype=np.float64)
        distance_travelled += float(np.linalg.norm(new_position - pose))
        pose = new_position
        distance_after = float(info['goal_distance_m'])
        min_distance = min(min_distance, distance_after)
        if distance_after <= 0.12:
            steps_in_goal_zone += 1

        control_trace.append({
            'step': step + 1,
            'goal_distance_before_m': distance_before,
            'goal_distance_after_m': distance_after,
            'controller_mode': mode,
            'act_pred_v_mps': diagnostics['v_mps'],
            'act_pred_w_radps': diagnostics['w_radps'],
            'executed_v_mps': (float(action[0]) + 1.0) * 0.5 * MAX_LINEAR_V,
            'executed_w_radps': float(action[1]) * MAX_ANGULAR_W,
            'measured_v_mps': float(info['linear_speed_mps']),
            'measured_w_radps': float(info['angular_speed_radps']),
            'success': bool(info['success']),
            'collision': bool(info['collision']),
        })
        if terminated or truncated:
            break

    elapsed = time.perf_counter() - t0
    success = bool(info.get('success', False))
    collision = bool(info.get('collision', False))
    return {
        'task_index': int(task_index),
        'task_success': success,
        'collision': collision,
        'timeout': not success and not collision,
        'steps': int(step + 1),
        'initial_goal_distance_m': initial_distance,
        'minimum_goal_distance_m': float(min_distance),
        'goal_distance_m': float(info['goal_distance_m']),
        'final_linear_speed_mps': float(info['linear_speed_mps']),
        'final_angular_speed_radps': float(info['angular_speed_radps']),
        'steps_within_goal_tolerance': int(steps_in_goal_zone),
        'brake_entries': int(brake_events),
        'terminal_brake_steps': int(brake_steps),
        'terminal_slow_steps': int(slow_steps),
        'actual_path_length_m': float(distance_travelled),
        'total_reward': float(total_reward),
        'elapsed_wall_s': float(elapsed),
        'replans': int(controller.replans),
        'act_action_clip_count': int(controller.clipped_commands),
        'benchmark_goal_xy': initial['goal_xy'],
    }, control_trace


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=Path, default=None)
    parser.add_argument('--model', type=Path, default=ROOT/'models/act_v2/act_best.pth')
    parser.add_argument('--max-tasks', type=int, default=3, help='0 = all tasks')
    parser.add_argument('--start-index', type=int, default=0)
    parser.add_argument('--max-steps', type=int, default=2500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    parser.add_argument('--brake-distance', type=float, default=0.09)
    parser.add_argument('--release-distance', type=float, default=0.13)
    parser.add_argument('--slow-distance', type=float, default=0.30)
    args = parser.parse_args()
    if (args.max_tasks < 0 or args.start_index < 0 or args.max_steps < 1 or
        not (0 < args.brake_distance < args.release_distance < args.slow_distance)):
        parser.error('Invalid task count, limits, or terminal stopping distances')

    benchmark = resolve_benchmark(args.benchmark)
    all_tasks = load_tasks(benchmark)
    end = len(all_tasks) if args.max_tasks == 0 else min(
        len(all_tasks), args.start_index + args.max_tasks)
    selection = list(enumerate(all_tasks))[args.start_index:end]
    if not selection:
        parser.error('No tasks selected')

    model_path = args.model if args.model.is_absolute() else ROOT / args.model
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    controller = ACTChunkController(model_path, device=args.device)
    out_dir = OUTPUT_ROOT / datetime.now(timezone.utc).strftime('run_%Y%m%d_%H%M%S_%f')
    out_dir.mkdir(parents=True, exist_ok=False)
    env = RobotNavigationEnv()
    results = []
    print('P8.4 ACT + TERMINAL STOP |',len(selection), 'tasks')
    print('Model:', model_path)
    print('Benchmark:', benchmark)
    try:
        for index, task in selection:
            try:
                result, trace = run_one(
                    env, controller, task, args.seed, args.max_steps, index,
                    args.brake_distance, args.release_distance, args.slow_distance)
                with open(out_dir / f'task_{index:03d}_trace.csv', 'w',
                          encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(trace[0]))
                    writer.writeheader()
                    writer.writerows(trace)
            except Exception as exc:
                result = {'task_index': index, 'task_success': False,
                          'collision': None, 'runtime_error': repr(exc)}
            results.append(result)
            (out_dir / f'task_{index:03d}.json').write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
            print(f"task={index} success={result['task_success']} "
                  f"distance={result.get('goal_distance_m','ERR')} "
                  f"brake={result.get('brake_entries','ERR')} "
                  f"runtime_error={result.get('runtime_error','none')}")
    finally:
        env.close()

    completed = [item for item in results if 'runtime_error' not in item]
    summary = {
        'stage': 'P8.4-terminal',
        'controller': 'ACT V2 + explicit deterministic goal-proximity slow/brake',
        'important_disclosure': ('NOT ACT-alone: terminal slow/brake supervisor '
                                 'uses known geometric navigation goal. '
                                 'Original PyBullet success/collision checks unchanged.'),
        'model': str(model_path), 'benchmark': str(benchmark),
        'benchmark_role': ('development_validation' if 'validation' in benchmark.name.lower()
                           else 'existing_fixed_benchmark'),
        'seed': args.seed, 'max_steps': args.max_steps,
        'terminal_settings': {
            'brake_distance_m': args.brake_distance,
            'release_distance_m': args.release_distance,
            'slow_distance_m': args.slow_distance,
        },
        'task_count_requested': len(selection),
        'task_count_completed': len(completed),
        'runtime_error_count': len(results) - len(completed),
        'successful_tasks': sum(bool(x['task_success']) for x in completed),
        'collisions': sum(bool(x['collision']) for x in completed),
        'timeouts': sum(bool(x['timeout']) for x in completed),
        'success_rate': (sum(bool(x['task_success']) for x in completed) / len(completed)
                         if completed else None),
        'results': results,
    }
    report = out_dir / 'evaluation_summary.json'
    report.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n=== P8.4 ACT + TERMINAL STOP EVALUATION ===')
    print('Success:', summary['successful_tasks'], '/', len(completed))
    print('Collisions:', summary['collisions'])
    print('Runtime errors:', summary['runtime_error_count'])
    print('Report:', report)


if __name__ == '__main__':
    main()
