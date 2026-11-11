"""P1.4/P1.5: 540 read-only PPO-V2 inference episodes and V3 readiness report.

No training. Three isolated worker processes use the same evaluator, each with
one default DIRECT client. All models and VecNormalize statistics are read-only.
Run: python experiments/37_evaluate_ppo_v3_validation.py
Resume an interrupted, unchanged experiment with --resume (append only).
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from statistics import mean, stdev
import sys
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visionnav.learning.bc_policy import DEFAULT_CHECKPOINT  # noqa: E402
from visionnav.learning.bc_residual_navigation_env_v2 import BCResidualNavigationEnvV2  # noqa: E402
from visionnav.learning.residual_scale_wrapper import ResidualScaleWrapper  # noqa: E402

SEEDS = (42, 123, 2026)
SCALES = (0.0, 0.25, 0.50)
MAX_STEPS = 2500
ORIGINAL_TASKS = ROOT / "logs/p5_evaluation/benchmark_tasks.json"
VALIDATION_TASKS = ROOT / "logs/p6_evaluation/benchmark_tasks_v3_validation.json"
OUTPUT = ROOT / "logs/p6_evaluation/ppo_v3_validation"
HISTORICAL = ROOT / "logs/p6_evaluation/ppo_residual_scale_ablation/200k/all_seeds_episode_results.csv"

# Defined BEFORE generating/observing validation results; engineering gates,
# not a significance test or proof of performance after new V3 training.
READINESS_CRITERIA = {
    "validation_success_gain_min": .05,
    "seeds_not_below_bc_min": 2,
    "collision_rate_increase_max": 0.0,
    "mean_final_error_increase_max_m": .02,
    "cross_seed_sample_std_max_pp": 10.0,
    "original_benchmark_success_gain_min": 0.0,
    "prefer_half_only_if_std_advantage_pp_at_least": 2.5,
    "prefer_half_only_if_success_advantage_at_least": .025,
}


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "experiments" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ablation = load_script("34_evaluate_ppo_residual_scale_ablation")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, payload, exclusive=True):
    with Path(path).open("x" if exclusive else "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def write_csv(path, rows):
    if not rows:
        raise ValueError("Cannot write an empty result table")
    with Path(path).open("x", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key:json.dumps(value,sort_keys=True) if isinstance(value,(dict,list)) else value
                          for key,value in row.items()} for row in rows)


def read_rows(path):
    if not Path(path).exists():
        return []
    converted = []
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            result = {}
            for key,value in row.items():
                if value in ("True", "False"):
                    result[key] = value == "True"
                elif key in ("seed", "task_id", "steps", "clipped_action_count", "invalid_action_count"):
                    result[key] = int(value)
                else:
                    try:
                        result[key] = float(value)
                    except (ValueError, TypeError):
                        result[key] = value
            converted.append(result)
    return converted


def build_cases():
    cases = []
    for dataset,path,expected_count in (("regression", ORIGINAL_TASKS, 20),
                                         ("validation", VALIDATION_TASKS, 40)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        tasks = ablation.extract_tasks(payload)
        if len(tasks) != expected_count:
            raise ValueError(f"Expected {expected_count} {dataset} tasks")
        if dataset == "validation" and payload.get("generation", {}).get("excluded_unique_endpoints") != 80:
            raise ValueError("Validation task independence has not been certified")
        cases.extend({"dataset": dataset, "task": task} for task in tasks)
    if len({(c['dataset'], c['task']['task_id']) for c in cases}) != 60:
        raise ValueError("Duplicate benchmark task IDs within a dataset")
    return cases


def make_vector_env(seed, scale, norm_path):
    def factory():
        raw = BCResidualNavigationEnvV2(checkpoint_path=DEFAULT_CHECKPOINT,
            delta_v_max=.05, delta_w_max=.4, lookahead_distance_m=.5)
        return Monitor(ResidualScaleWrapper(raw, residual_scale=scale),
                       info_keywords=("success", "collision"))
    base = DummyVecEnv([factory])
    base.seed(seed)
    try:
        env = VecNormalize.load(str(norm_path), base)
        env.training = False
        env.norm_reward = False
        return env
    except Exception:
        base.close()
        raise


def evaluate_task(env, model, seed, scale, case, max_steps=MAX_STEPS):
    task = case["task"]
    env.venv.set_options({"task": task})
    observation = env.reset()
    previous = np.asarray(task["start_xy"], dtype=np.float64)
    distance, episode_return = 0., 0.
    raw_values, residuals, corrections, deviations, activations, clipped = [], [], [], [], [], []
    started = time.perf_counter()
    done, final_info = False, {}
    for step in range(1, max_steps + 1):
        if model is None:
            action = np.zeros((1, 2), dtype=np.float32)
        else:
            action, _ = model.predict(observation, deterministic=True)
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (1, 2) or not np.isfinite(action).all():
            raise ValueError("INVALID_POLICY_ACTION: refusing to fabricate a completed episode")
        # Scaling occurs ONLY in ResidualScaleWrapper, including BC-only scale 0.
        observation, rewards, dones, infos = env.step(action)
        info = final_info = infos[0]
        if observation.shape != (1, 44) or not np.isfinite(observation).all() or not np.isfinite(rewards).all():
            raise ValueError("Invalid observation/reward in validation rollout")
        diagnostic = info["v3_residual"]
        expected = (np.clip(action[0], -1., 1.) * scale).astype(np.float32)
        if not np.allclose(info["residual_action"], expected, atol=1e-7):
            raise RuntimeError("Action transformed more than once")
        correction = np.array([diagnostic["delta_v_mps"], diagnostic["delta_w_radps"]])
        if not np.allclose(correction, expected * [.05, .4], atol=1e-7):
            raise RuntimeError("Physical correction does not match V2's original limits")
        episode_return += float(rewards[0])
        position = np.asarray(info["position_xy"], dtype=np.float64)
        distance += float(np.linalg.norm(position - previous))
        previous = position
        raw_values.append(action[0].copy())
        residuals.append(np.asarray(diagnostic["scaled_action"]))
        corrections.append(correction)
        deviations.append(float(info["path_deviation_m"]))
        activations.append(bool(diagnostic["residual_active"]))
        clipped.append(bool(diagnostic["action_clipped"]))
        if bool(dones[0]):
            done = True
            break
    raw_values, residuals, corrections = map(np.asarray, (raw_values, residuals, corrections))
    success, collision = bool(final_info["success"]), bool(final_info["collision"])
    return {
        "seed": seed, "dataset": case["dataset"], "scale": scale,
        "controller": "BC only" if scale == 0 else f"BC + PPO V2 scale {scale:.2f}",
        "task_id": int(task["task_id"]), "success": success, "collision": collision,
        "timeout": not success and not collision, "steps": step,
        "path_length_m": ablation.route_length(task["waypoints_world"]),
        "actual_distance_m": distance, "final_goal_error_m": float(final_info["goal_distance_m"]),
        "final_speed_mps": float(final_info["linear_speed_mps"]),
        "final_angular_speed_radps": float(final_info["angular_speed_radps"]),
        "elapsed_time_s": time.perf_counter() - started, "episode_return": episode_return,
        "mean_path_deviation_m": float(np.mean(deviations)),
        "mean_abs_raw_policy_v": float(np.abs(raw_values[:,0]).mean()),
        "mean_abs_raw_policy_w": float(np.abs(raw_values[:,1]).mean()),
        "mean_abs_applied_residual_v": float(np.abs(residuals[:,0]).mean()),
        "mean_abs_applied_residual_w": float(np.abs(residuals[:,1]).mean()),
        "mean_delta_v_mps": float(corrections[:,0].mean()),
        "mean_delta_w_radps": float(corrections[:,1].mean()),
        "mean_abs_delta_v_mps": float(np.abs(corrections[:,0]).mean()),
        "mean_abs_delta_w_radps": float(np.abs(corrections[:,1]).mean()),
        "max_abs_delta_v_mps": float(np.abs(corrections[:,0]).max()),
        "max_abs_delta_w_radps": float(np.abs(corrections[:,1]).max()),
        "residual_active_fraction": float(np.mean(activations)),
        "clipped_action_count": sum(clipped), "action_clip_fraction": float(np.mean(clipped)),
        "invalid_action_count": 0,
        "failure_reason": "SUCCESS" if success else "COLLISION" if collision else "TIMEOUT",
        "terminated_or_truncated": done,
    }


def run_seed(seed, cases, output_dir, max_steps=MAX_STEPS, scales=SCALES):
    output_dir = Path(output_dir)
    progress_file = output_dir / f"v3_seed_{seed}_progress.csv"
    rows = read_rows(progress_file)
    completed = {(r['scale'], r['dataset'], r['task_id']) for r in rows}
    if len(completed) != len(rows):
        raise ValueError("Duplicate rows in seed progress file")
    _, model_path, norm_path = ablation.checkpoint_paths(seed, "200k")
    for scale in scales:
        pending = [case for case in cases if (scale, case['dataset'], int(case['task']['task_id'])) not in completed]
        if not pending:
            continue
        set_random_seed(seed)
        env = make_vector_env(seed, scale, norm_path)
        try:
            # BC-only never loads a PPO/TD3 or other auxiliary policy.
            model = None if scale == 0 else PPO.load(str(model_path), env=env, device="cpu")
            for case in pending:
                row = evaluate_task(env, model, seed, scale, case, max_steps)
                mode = "a" if progress_file.exists() else "x"
                with progress_file.open(mode, newline="", encoding="utf-8-sig") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    if mode == "x":
                        writer.writeheader()
                    writer.writerow(row)
                rows.append(row)
                completed.add((scale, case['dataset'], row['task_id']))
                if len(rows) % 5 == 0 or row['task_id'] == 0:
                    print(f"seed={seed} completed={len(rows)}/{len(cases)*len(scales)} "
                          f"scale={scale:.2f} {case['dataset']} task={row['task_id']} "
                          f"success={row['success']} steps={row['steps']}", flush=True)
        finally:
            env.close()
    return rows


def summarize(rows):
    result = ablation.summarize(rows)
    result["timeout_rate"] = result["timeouts"] / len(rows)
    result["mean_delta_v_mps"] = mean(r["mean_delta_v_mps"] for r in rows)
    result["mean_delta_w_radps"] = mean(r["mean_delta_w_radps"] for r in rows)
    result["action_clip_fraction"] = mean(r["action_clip_fraction"] for r in rows)
    result["invalid_action_count"] = sum(r["invalid_action_count"] for r in rows)
    return result


def aggregate_results(rows):
    per_seed, comparison = [], []
    for dataset in ("regression", "validation"):
        for scale in SCALES:
            group = []
            for seed in SEEDS:
                selected = [r for r in rows if (r['seed'],r['dataset'],r['scale']) == (seed,dataset,scale)]
                item = {"seed": seed, "dataset": dataset, "scale": scale, **summarize(selected)}
                per_seed.append(item)
                group.append(item)
            comparison.append({"dataset": dataset, "scale": scale,
                "total_episodes": sum(g["total_tasks"] for g in group),
                "mean_success_rate": mean(g["success_rate"] for g in group),
                "std_success_rate_pp": stdev(g["success_rate"] for g in group)*100,
                **{key: mean(g[metric] for g in group) for key,metric in (
                    ("mean_collision_rate","collision_rate"), ("mean_timeout_rate","timeout_rate"),
                    ("mean_steps","average_steps"), ("mean_actual_distance_m","average_actual_distance_m"),
                    ("mean_final_error_m","average_final_error_m"), ("mean_reward","average_reward"),
                    ("mean_path_deviation_m","average_path_deviation_m"),
                    ("mean_abs_delta_v_mps","average_mean_abs_delta_v_mps"),
                    ("mean_abs_delta_w_radps","average_mean_abs_delta_w_radps"),
                    ("mean_residual_active_fraction","average_residual_active_fraction"),
                    ("mean_delta_v_mps","mean_delta_v_mps"), ("mean_delta_w_radps","mean_delta_w_radps"),
                    ("action_clip_fraction","action_clip_fraction"))},
                "invalid_action_count": sum(g['invalid_action_count'] for g in group),
                "per_seed_success_rates": {str(g['seed']):g['success_rate'] for g in group}})
    return per_seed, comparison


def paired_comparisons(rows):
    lookup = {(r['seed'],r['dataset'],r['task_id'],r['scale']):r for r in rows}
    paired = []
    for seed in SEEDS:
        for dataset in ("regression", "validation"):
            task_ids = sorted({r['task_id'] for r in rows if r['seed']==seed and r['dataset']==dataset})
            for base_scale,candidate_scale in ((0.,.25),(0.,.5),(.25,.5)):
                for task_id in task_ids:
                    a = lookup[(seed,dataset,task_id,base_scale)]
                    b = lookup[(seed,dataset,task_id,candidate_scale)]
                    paired.append({"seed":seed,"dataset":dataset,"task_id":task_id,
                        "baseline_scale":base_scale,"candidate_scale":candidate_scale,
                        "baseline_success":a['success'],"candidate_success":b['success'],
                        "candidate_win":not a['success'] and b['success'],
                        "baseline_win":a['success'] and not b['success'],
                        "both_successful":a['success'] and b['success'],
                        "both_failed":not a['success'] and not b['success'],
                        "baseline_collision":a['collision'],"candidate_collision":b['collision'],
                        "baseline_final_error_m":a['final_goal_error_m'],
                        "candidate_final_error_m":b['final_goal_error_m'],
                        **{f"delta_{name}_candidate_minus_baseline":b[name]-a[name]
                           for name in ('steps','final_goal_error_m','actual_distance_m',
                                        'mean_path_deviation_m','episode_return')}})
    return paired


def historical_regression(rows):
    historical = read_rows(HISTORICAL)
    lookup = {(r['seed'],float(r['scale']),int(r['task_id'])):r for r in historical}
    mismatches, checked = [], 0
    for row in rows:
        if row['dataset'] != 'regression':
            continue
        reference = lookup[(row['seed'],row['scale'],row['task_id'])]
        checked += 1
        fields = [name for name in ('success','collision','timeout','steps') if row[name]!=reference[name]]
        fields += [name for name in ('final_goal_error_m','episode_return','actual_distance_m')
                   if not np.isclose(row[name],reference[name],rtol=0,atol=1e-6)]
        if fields:
            mismatches.append({"seed":row['seed'],"scale":row['scale'],"task_id":row['task_id'],"fields":fields})
    return {"checked_episodes":checked,"mismatches":mismatches,"passed":checked==180 and not mismatches,
            "source":str(HISTORICAL.relative_to(ROOT)),"sha256":sha256(HISTORICAL)}


def readiness(per_seed, comparisons, regression):
    indexed = {(r['dataset'],r['scale']):r for r in comparisons}
    decisions = []
    for scale in (.25,.5):
        current, base = indexed[('validation',scale)], indexed[('validation',0.)]
        original, original_base = indexed[('regression',scale)],indexed[('regression',0.)]
        noninferior = sum(next(r for r in per_seed if (r['seed'],r['dataset'],r['scale'])==(seed,'validation',scale))['success_rate']
                         >= next(r for r in per_seed if (r['seed'],r['dataset'],r['scale'])==(seed,'validation',0.))['success_rate']
                         for seed in SEEDS)
        gain = current['mean_success_rate']-base['mean_success_rate']
        gates = {
            "validation_gain_at_least_5pp":gain >= READINESS_CRITERIA['validation_success_gain_min']-1e-12,
            "at_least_two_seeds_not_below_bc":noninferior >= READINESS_CRITERIA['seeds_not_below_bc_min'],
            "collision_rate_not_increased":current['mean_collision_rate'] <= base['mean_collision_rate']+READINESS_CRITERIA['collision_rate_increase_max']+1e-12
                and original['mean_collision_rate'] <= original_base['mean_collision_rate']+READINESS_CRITERIA['collision_rate_increase_max']+1e-12,
            "final_error_not_worse_by_over_2cm":current['mean_final_error_m'] <= base['mean_final_error_m']+READINESS_CRITERIA['mean_final_error_increase_max_m'],
            "cross_seed_std_at_most_10pp":current['std_success_rate_pp'] <= READINESS_CRITERIA['cross_seed_sample_std_max_pp']+1e-12,
            "original_benchmark_trend_not_worse":original['mean_success_rate']-original_base['mean_success_rate'] >= READINESS_CRITERIA['original_benchmark_success_gain_min']-1e-12,
            "no_invalid_actions":current['invalid_action_count']==0,
            "v2_regression_reproduced":regression['passed'],
        }
        decisions.append({"scale":scale,"qualified":all(gates.values()),"gates":gates,
            "validation_success_gain_pp":gain*100,"noninferior_seed_count":noninferior})
    eligible = [r['scale'] for r in decisions if r['qualified']]
    recommended = eligible[0] if eligible else None
    if len(eligible)==2:
        quarter,half=indexed[('validation',.25)],indexed[('validation',.5)]
        if (quarter['std_success_rate_pp']-half['std_success_rate_pp'] >= READINESS_CRITERIA['prefer_half_only_if_std_advantage_pp_at_least']-1e-12
                and half['mean_success_rate']-quarter['mean_success_rate'] >= READINESS_CRITERIA['prefer_half_only_if_success_advantage_at_least']-1e-12):
            recommended=.5
    return {"recommend_training":recommended is not None,"recommended_residual_scale":recommended,
            "criteria":READINESS_CRITERIA,"decisions":decisions,
            "selection_rule":"Prefer 0.25 if both qualify; 0.50 must improve success by >=2.5pp AND reduce seed std by >=2.5pp",
            "scope":"Readiness to consider a new V3 training experiment, not evidence of a trained V3 policy"}


def write_report(output, summary, comparisons, per_seed, paired):
    decision = summary['readiness']
    lines = ['# PPO V3 训练准备度报告', '',
        f"建议开始 PPO V3 训练：**{'是' if decision['recommend_training'] else '否'}**。",
        f"推荐 residual_scale：**{decision['recommended_residual_scale']}**。", '',
        '本轮仅使用既有 200k PPO V2 模型做确定性推理，没有训练或更新任何模型及归一化统计。',
        '新 40 项是用于参数选择的独立验证集，不能称为最终无偏测试集；V3 训练后必须另设最终测试集。', '',
        '## 工程判据（运行前固定）', '',
        '- 验证成功率提高至少 5pp，至少 2/3 种子不低于 BC。',
        '- 原/新任务的碰撞率均不增加，验证集平均终点误差最多恶化 0.02m。',
        '- 验证集跨种子样本标准差不超过 10pp，旧基准平均成功率不得低于 BC。',
        '- 无非法动作，并完整复现旧基准的 180 项相关消融结果。',
        '- 两者都合格时默认选 0.25；0.50 须同时多提高至少 2.5pp 成功率、少至少 2.5pp 种子标准差才优先。', '',
        '## 总体结果', '',
        '| 数据集 | Scale | 平均成功率 | 种子标准差 | 碰撞率 | 超时率 | 平均终点误差(m) | 平均路径偏差(m) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in comparisons:
        lines.append(f"| {row['dataset']} | {row['scale']:.2f} | {row['mean_success_rate']:.2%} | "
            f"{row['std_success_rate_pp']:.2f}pp | {row['mean_collision_rate']:.2%} | {row['mean_timeout_rate']:.2%} | "
            f"{row['mean_final_error_m']:.4f} | {row['mean_path_deviation_m']:.4f} |")
    quarter = next(row for row in comparisons if row['dataset']=='validation' and row['scale']==.25)
    half = next(row for row in comparisons if row['dataset']=='validation' and row['scale']==.5)
    lines += ['', '方案权衡：', '',
        f"- 新验证集上 0.25/0.50 的成功率分别为 {quarter['mean_success_rate']:.2%}/{half['mean_success_rate']:.2%}，"
        f"跨种子标准差为 {quarter['std_success_rate_pp']:.2f}/{half['std_success_rate_pp']:.2f}pp。",
        f"- 0.25/0.50 的平均终点误差分别为 {quarter['mean_final_error_m']:.4f}/{half['mean_final_error_m']:.4f}m；"
        '成功率、干预幅度、稳定性与精度需要共同权衡，按上述预先规则选择，不能只看奖励。',
        '- 平均终点误差包含全部成功及失败任务，不是仅成功任务的误差。']
    lines += ['', '## 各种子结果', '', '| 数据集 | 种子 | Scale | 成功率 | 碰撞率 | 终点误差(m) |', '|---|---:|---:|---:|---:|---:|']
    for row in per_seed:
        lines.append(f"| {row['dataset']} | {row['seed']} | {row['scale']:.2f} | {row['success_rate']:.2%} | "
                     f"{row['collision_rate']:.2%} | {row['average_final_error_m']:.4f} |")
    lines += ['', '## 判定与逐任务配对', '']
    for item in decision['decisions']:
        failed=[name for name,passed in item['gates'].items() if not passed]
        lines.append(f"- Scale {item['scale']:.2f}：{'合格' if item['qualified'] else '不合格'}；"
                     f"验证提升 {item['validation_success_gain_pp']:.2f}pp；不低于 BC 的种子 {item['noninferior_seed_count']}/3；未满足项：{failed}。")
    for dataset in ('regression','validation'):
        for a,b in ((0.,.25),(0.,.5),(.25,.5)):
            selected=[r for r in paired if (r['dataset'],r['baseline_scale'],r['candidate_scale'])==(dataset,a,b)]
            lines.append(f"- {dataset}, {b:.2f} 对 {a:.2f}：候选独赢 {sum(r['candidate_win'] for r in selected)}，"
                f"基线独赢 {sum(r['baseline_win'] for r in selected)}，共同成功 {sum(r['both_successful'] for r in selected)}，"
                f"共同失败 {sum(r['both_failed'] for r in selected)}；详见 paired_comparison.csv。")
    lines += ['', '## 回归和风险', '',
        f"- 旧结果回归：{summary['v2_regression']['checked_episodes']} 项；不一致 {len(summary['v2_regression']['mismatches'])} 项。",
        '- 同一任务在三种子 BC-only 中重复，不应把重复基线当作三个独立任务样本；种子标准差只反映当前三个模型。',
        '- 40 项验证用于选参数，有选择偏差；固定仓库、小样本、三个种子均限制外推范围。',
        '- 独立端点检查范围是用户确认的数据集 40 端点 + 旧基准 40 端点；不等于地图、全轨迹或 PPO 在线训练状态完全未见。',
        '- 本轮评估的是 V2 策略缩放后的部署表现，不能保证重新训练的 V3 达到相同成功率。',
        '- 原 V2 的碰撞判定、奖励、终止条件、运动模型均保留；未新增安全控制或碰撞算法。',
        '- 诊断中的物理残差来自 V2 residual_correction，位于最终速度裁剪之前，并非从轮速反推的实际机器人速度差。',
        '- 原始动作指 model.predict 返回给环境的动作；SB3 自身可能已裁剪该动作，wrapper 的限幅率不代表网络潜在动作的限幅率。', '',
        '## 下一阶段（本次不实施）', '',
        '- 新增独立 train_bc_residual_ppo_v3.py，不修改 V2 训练文件；训练/评估共用 V2 → ResidualScaleWrapper → Monitor → DummyVecEnv → VecNormalize。',
        f"- {'采用推荐 scale '+str(decision['recommended_residual_scale']) if decision['recommend_training'] else '暂不启动训练，先处理上述未满足项及失败任务'}；保留 0.50/0.25 为对照，不启用动态门控或课程学习。",
        '- BC 权重、40/44 维观测、2 维动作和 V2 物理修正上限保持不变；调用方不得再次乘 scale。',
        '- 原 V2 PPO 超参数作为讨论起点，本次没有改动；训练步数、起始权重及其他训练参数由用户另行确定。',
        '- 使用独立 V3 模型、日志和 VecNormalize 路径；评估加载对应 V3 统计、training=False、norm_reward=False。',
        '- 新增与当前验证集分离的最终测试集；训练收益以成功率/碰撞/误差和配对任务评估判断，不能只看奖励。', '',
        '- 下一阶段 V3 训练采样还须排除这 40 项验证任务的 80 个端点，并保护未来最终测试集；在独立 V3 训练工厂中处理，不修改 V2 基线。', '',
        '## 可追溯文件', '',
        '- validation_manifest.json：数据/模型/统计/控制源码哈希及预先判据。',
        '- all_episode_results.csv：540 条原始结果；per_seed_summary.csv：18 组种子统计。',
        '- scale_comparison.csv：两套任务各三种控制器；paired_comparison.csv：540 条配对记录。',
        '- validation_summary.json：完整统计、判据逐项结果和正式建议。', '']
    (Path(output)/'PPO_V3_TRAINING_READINESS_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')


def manifest():
    paths=[DEFAULT_CHECKPOINT,ORIGINAL_TASKS,VALIDATION_TASKS,HISTORICAL,
           ROOT/'visionnav/learning/bc_residual_navigation_env_v2.py',ROOT/'visionnav/learning/rl_navigation_env.py',
           ROOT/'visionnav/learning/residual_scale_wrapper.py',ROOT/'scripts/diff_drive_robot.py',
           ROOT/'scripts/05_environment_builder.py',ROOT/'experiments/astar_planner.py',ROOT/'experiments/occupancy_grid.py']
    models={}
    for seed in SEEDS:
        _,model,norm=ablation.checkpoint_paths(seed,'200k');paths.extend((model,norm))
        models[str(seed)]={'model':str(model.relative_to(ROOT)),'vecnormalize':str(norm.relative_to(ROOT))}
    return {'experiment':'PPO V3 preparation validation; no training','seeds':list(SEEDS),'scales':list(SCALES),
        'tasks_per_seed_scale':60,'expected_episodes':540,'max_steps':MAX_STEPS,
        'model_set':'original 200k PPO V2','models':models,'criteria_before_evaluation':READINESS_CRITERIA,
        'source_sha256':{str(path.relative_to(ROOT)):sha256(path)for path in paths},
        'wrapper_order':['BCResidualNavigationEnvV2','ResidualScaleWrapper','Monitor','DummyVecEnv','VecNormalize'],
        'validation_role':'parameter-selection validation, not a final unbiased test set'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,choices=(1,2,3),default=3)
    parser.add_argument('--output-dir',type=Path,default=OUTPUT)
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();cases=build_cases();configuration=manifest();out=args.output_dir
    manifest_path=out/'validation_manifest.json'
    if args.resume:
        if json.loads(manifest_path.read_text(encoding='utf-8'))!=configuration:
            raise ValueError('Resume manifest mismatch; refusing to mix experiments')
        if (out/'validation_summary.json').exists():
            raise FileExistsError('Experiment already completed; refusing to overwrite its outputs')
    else:
        out.mkdir(parents=True,exist_ok=False)
        write_json(manifest_path,configuration)
    print('Starting 540 deterministic inference episodes; no training; workers=',args.workers,flush=True)
    all_rows,errors=[],[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs={pool.submit(run_seed,seed,cases,str(out)):seed for seed in SEEDS}
        for job in as_completed(jobs):
            seed=jobs[job]
            try:
                result=job.result();all_rows.extend(result)
                print(f'Completed seed {seed}: {len(result)} episodes',flush=True)
            except Exception as exc:
                errors.append({'seed':seed,'error':repr(exc)})
            write_json(out/'progress.json',{'completed_episodes':len(all_rows),'errors':errors},exclusive=False)
    if errors:
        raise RuntimeError(f'Incomplete experiment; retained append-only progress: {errors}')
    expected={(seed,scale,case['dataset'],int(case['task']['task_id']))for seed in SEEDS for scale in SCALES for case in cases}
    actual={(r['seed'],r['scale'],r['dataset'],r['task_id'])for r in all_rows}
    if len(all_rows)!=540 or actual!=expected:
        raise RuntimeError('Incomplete or duplicated episode matrix')
    all_rows.sort(key=lambda r:(r['seed'],r['scale'],r['dataset'],r['task_id']))
    per_seed,comparison=aggregate_results(all_rows);paired=paired_comparisons(all_rows)
    regression=historical_regression(all_rows);decision=readiness(per_seed,comparison,regression)
    summary={'completed':True,'total_episodes':len(all_rows),'per_seed_summary':per_seed,
        'scale_comparison':comparison,'v2_regression':regression,'readiness':decision,
        'paired_rows':len(paired),'manifest':configuration}
    write_csv(out/'all_episode_results.csv',all_rows)
    write_csv(out/'per_seed_summary.csv',per_seed)
    write_csv(out/'scale_comparison.csv',comparison)
    write_csv(out/'paired_comparison.csv',paired)
    write_json(out/'validation_summary.json',summary)
    write_report(out,summary,comparison,per_seed,paired)
    for row in comparison:
        print(row['dataset'],f"scale={row['scale']:.2f}",f"success={row['mean_success_rate']:.2%}",
              f"std={row['std_success_rate_pp']:.2f}pp",flush=True)
    print('All 540 evaluations completed; recommend training:',decision['recommend_training'],
          'recommended scale:',decision['recommended_residual_scale'],flush=True)


if __name__=='__main__':
    main()
