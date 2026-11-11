
"""P4-4: Language-driven multi-target visual navigation planning."""

import sys
import json
import base64
import time
import importlib.util
import urllib.request
from pathlib import Path
from dataclasses import asdict

import cv2
import numpy as np
import pybullet as p
import pybullet_data


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "experiments"))

import occupancy_grid as occ
import astar_planner as planner

from scripts.diff_drive_robot import DiffDriveRobot
from visionnav.simulation.semantic_scene import (
    add_semantic_targets,
    add_static_detour_obstacle,
)
from visionnav.simulation.observation_adapter import (
    PyBulletObservationAdapter,
)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


EnvironmentBuilder = load_module(
    "env_builder",
    ROOT / "scripts/05_environment_builder.py",
).EnvironmentBuilder

qwen_bbox = load_module(
    "qwen_bbox",
    ROOT / "experiments/23_qwen_target_bbox.py",
)
predict_bbox = qwen_bbox.predict_bbox

estimate_position = load_module(
    "rgbd_localization",
    ROOT / "experiments/24_localize_qwen_bbox.py",
).estimate_position


def understand_instruction(instruction, diagnostic_path=None):
    """Use Qwen to identify the requested target from Chinese text."""

    prompt = (
        "Extract the single physical object that the user "
        "wants the robot to approach. Translate its color "
        "and object type into an accurate English target description. "
        "Preserve every explicitly stated spatial relation, including "
        "floor versus shelf placement. Do not invent attributes or "
        "spatial relations. Examples: "
        "地上的红色箱子 -> red box on the floor; "
        "货架上的红色箱子、货架上方或货架顶部的红色箱子 -> red box on top of the shelf; "
        "蓝色圆柱体 -> blue cylinder; "
        "绿色球体 -> green sphere. "
        "Ignore navigation actions and obstacles to avoid. "
        'Return only JSON with one key "target". '
        "If no unambiguous target is specified, use null. "
        f"User instruction: {instruction}"
    )

    payload = {
        "model": "qwen3-vl:2b-instruct",
        "messages": [{
            "role": "user",
            "content": prompt,
        }],
        "format": "json",
        "stream": False,
        "options": {
            "temperature": 0,
            "num_predict": 96,
        },
    }

    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/chat",
        data=json.dumps(
            payload, ensure_ascii=False
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(
        request, timeout=600
    ) as response:
        result = json.load(response)

    if diagnostic_path is not None:
        save_json(diagnostic_path, {"instruction": instruction,
                                   "raw_response": result["message"]["content"]})
    answer = json.loads(result["message"]["content"])
    target = answer.get("target")

    if not isinstance(target, str):
        raise ValueError("Qwen did not identify a target")

    target = target.strip()

    if not target or len(target) > 240:
        raise ValueError("Invalid target description")

    # Spatial constraints come from the user, never from model-added placement.
    normalized_target = qwen_bbox.canonical_target_description(target, instruction)
    if diagnostic_path is not None:
        save_json(diagnostic_path, {"instruction": instruction,
            "raw_response": result["message"]["content"], "model_target": target,
            "normalized_target": normalized_target,
            "required_spatial_relation": qwen_bbox.spatial_requirement("", instruction)})
    return normalized_target



def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")


def checked_bbox(value):
    return qwen_bbox.validate_bbox(value)


def verify_target(image_b64, target, instruction, diagnostics=None):
    diagnostics = [] if diagnostics is None else diagnostics
    return qwen_bbox.visual_request(image_b64,
        qwen_bbox.visual_prompt(target, instruction, verifier=True), diagnostics)


def classify_detection(bbox, verification_raw, target=None, instruction="", diagnostics=None,
                       image_size=None):
    """Shared protocol parsing; preserve independent verifier and IoU>=0.5."""
    if target is None:
        raise ValueError("TARGET_PROTOCOL_INVALID: target is required")
    partial = {}
    try:
        answer = qwen_bbox.parse_visual_answer(verification_raw, target,
            qwen_bbox.spatial_requirement(target, instruction), diagnostics=partial)
    finally:
        if diagnostics is not None:
            diagnostics["verifier_protocol_details"] = partial
            diagnostics["verifier_bbox"] = partial.get("bbox_2d")
    if diagnostics is not None:
        diagnostics["verifier_normalized"] = answer
        diagnostics["verifier_bbox"] = answer["bbox_2d"]
        diagnostics["semantic_verification_status"] = answer["status"]
    bbox, _ = qwen_bbox.normalize_bbox(bbox, scale="0_1000")
    if answer["status"] == "TARGET_AMBIGUOUS":
        return "TARGET_AMBIGUOUS"
    verified = answer["bbox_2d"]
    if answer["status"] != "TARGET_FOUND":
        return "TARGET_NOT_VISIBLE" if bbox is None else "TARGET_RECOGNITION_MISMATCH"
    if image_size is not None:
        qwen_bbox.bbox_to_pixels(verified, *image_size)
    if bbox is None:
        return "TARGET_RECOGNITION_MISMATCH"
    iou = qwen_bbox.compute_iou(bbox, verified)
    if diagnostics is not None:
        diagnostics["bbox_iou"] = iou
        if iou < .5:
            diagnostics["rejection_detail"] = f"BBOX_IOU_BELOW_THRESHOLD: {iou:.6f} < 0.5"
    return "TARGET_FOUND" if iou >= .5 else "TARGET_RECOGNITION_MISMATCH"


def robot_geometry(robot):
    """Own collision footprint measured at the common initial heading, not target GT."""
    pose = robot.get_pose()
    if abs(pose[2]) > 0.01:
        raise ValueError("Measure footprint before active rotation")
    bounds = [p.getAABB(robot.robot_id, link)
              for link in range(-1, p.getNumJoints(robot.robot_id))
              if p.getCollisionShapeData(robot.robot_id, link)]
    lo = np.min([b[0] for b in bounds], axis=0)
    hi = np.max([b[1] for b in bounds], axis=0)
    front = float(hi[0] - pose[0])
    rear = float(pose[0] - lo[0])
    half_width = float(max(hi[1] - pose[1], pose[1] - lo[1]))
    return {"front_m": front, "rear_m": rear, "half_width_m": half_width,
            "length_m": front + rear, "width_m": 2 * half_width,
            "radius_m": float(np.hypot(max(front, rear), half_width)),
            "top_z_m": float(hi[2]),
            "center_z_offset_m": float((hi[2] + lo[2]) / 2
                                        - p.getBasePositionAndOrientation(robot.robot_id)[0][2])}


def search_action(robot, ground_id, angular_speed, physics_steps=10):
    """Physical wheel motion only; shared with execution for deterministic replay."""
    robot.set_velocity(0.0, float(angular_speed))
    for _ in range(physics_steps):
        p.stepSimulation()
        for contact in p.getContactPoints(bodyA=robot.robot_id):
            if contact[2] != ground_id and contact[9] > 0.1:
                robot.set_velocity(0.0, 0.0)
                raise RuntimeError("SEARCH_COLLISION")


def turn_to_heading(robot, ground_id, heading, commands):
    for _ in range(240):
        pose = robot.get_pose()
        error = float(np.arctan2(np.sin(heading - pose[2]), np.cos(heading - pose[2])))
        if abs(error) < 0.025:
            break
        speed = float(np.clip(2 * error, -0.8, 0.8))
        search_action(robot, ground_id, speed)
        commands.append([speed, 10])
    else:
        raise RuntimeError("SEARCH_ROTATION_TIMEOUT")
    for _ in range(12):
        search_action(robot, ground_id, 0.0)
        commands.append([0.0, 10])
    if np.linalg.norm(robot.get_pose()[:2]) > 0.10:
        raise RuntimeError("SEARCH_EXCESSIVE_TRANSLATION")


def validate_candidate_position(bbox, depth_path, obs, target, instruction, geometry, record):
    """Localize this frame's candidate, then check its requested spatial relation."""
    depth = np.load(depth_path, allow_pickle=False)
    if depth.shape != (obs.camera.height, obs.camera.width):
        raise ValueError("RGBD_LOCALIZATION_FAILED: depth/camera size mismatch")
    # Never reuse the calibration of an earlier observation direction.
    surface_xyz = np.asarray(
        estimate_position(bbox, depth, asdict(obs.camera)), dtype=float)
    if surface_xyz.shape != (3,) or not np.all(np.isfinite(surface_xyz)):
        raise ValueError("RGBD_LOCALIZATION_FAILED: invalid target XYZ")
    record["candidate_surface_xyz"] = surface_xyz.tolist()
    # Preserve the existing height threshold; a ground box cannot become a shelf box.
    elevated = bool(surface_xyz[2] > max(0.45, geometry["top_z_m"] + 0.15))
    requested_shelf = ("shelf" in target.lower()
                       or any(w in instruction for w in ("货架", "架顶", "架子上")))
    requested_floor = (any(w in target.lower() for w in ("floor", "ground"))
                       or any(w in instruction for w in ("地面", "地上")))
    if (requested_shelf and not elevated) or (requested_floor and elevated):
        raise ValueError(
            "TARGET_RECOGNITION_MISMATCH: RGB-D height contradicts spatial relation "
            f"(candidate z={surface_xyz[2]:.3f}m)")
    # Shelf-top support/color/category are checked by verify_target in this same
    # RGB frame; sufficient height alone does not establish the shelf relation.
    return surface_xyz, elevated


def match_candidate_bboxes(detector_raw, verifier_raw, target, instruction, image_size, record):
    """Invalid-detector recovery, not a lower IoU threshold for two valid boxes."""
    q=qwen_bbox;relation=q.spatial_requirement(target,instruction)
    raw=json.loads(detector_raw)
    detector=None;invalid=None
    try:
        detector=q.parse_visual_answer(detector_raw,target,relation)
        if detector["status"]!="TARGET_FOUND":
            raise ValueError("TARGET_RECOGNITION_MISMATCH: detector is not FOUND")
        q.check_bbox_reasonableness(detector["bbox_2d"],*image_size)
    except ValueError as exc:
        if not str(exc).startswith(("TARGET_BBOX_INVALID","TARGET_BBOX_TOO_SMALL","TARGET_BBOX_ABNORMAL")):
            raise
        invalid=str(exc)
        # A bbox failure is not permission to override semantic contradictions.
        color,category=q.target_identity(target)
        if (not isinstance(raw,dict) or raw.get("status")!="TARGET_FOUND"
            or not all(raw.get(k) is True for k in ("category_match","color_match","spatial_match"))
            or q.CATEGORIES.get(q.normalized_text(raw.get("observed_category")))!=category
            or q.COLORS.get(q.normalized_text(raw.get("observed_color")))!=color):
            raise ValueError("TARGET_RECOGNITION_MISMATCH: invalid detector semantics")
        observed=q.normalize_relation(raw.get("spatial_relation"))
        if relation!="unspecified" and observed!=relation:
            raise ValueError("TARGET_RECOGNITION_MISMATCH: detector spatial conflict")
    verifier=q.parse_visual_answer(verifier_raw,target,relation)
    record.update(verification_raw=verifier_raw,verifier_normalized=verifier,
                  verifier_bbox=verifier["bbox_2d"],semantic_verification_status=verifier["status"])
    if verifier["status"]!="TARGET_FOUND":
        raise ValueError("TARGET_RECOGNITION_MISMATCH: verifier did not confirm target")
    q.check_bbox_reasonableness(verifier["bbox_2d"],*image_size)
    original=detector["bbox_2d"] if detector is not None else None
    record["bbox_iou"]=q.compute_iou(original,verifier["bbox_2d"]) if original is not None else None
    record["detector_normalized"] = detector
    if invalid is not None:
        record.update(bbox_source="verifier_fallback",fallback_used=True,
            detector_bbox_rejection=invalid,iou_applicable=False,
            detector_coordinate_scale=detector.get("coordinate_scale") if detector else None,
            bbox_coordinate_scale=verifier["coordinate_scale"])
        return verifier["bbox_2d"]
    if record["bbox_iou"]<.5:
        raise ValueError(f"BBOX_IOU_BELOW_THRESHOLD: {record['bbox_iou']:.6f} < 0.5")
    record.update(bbox_source="detector",fallback_used=False,iou_applicable=True,
                  bbox_coordinate_scale=detector["coordinate_scale"])
    return original


def evaluate_observation(obs, rgb_path, depth_path, target, instruction, geometry, record,
                         allow_verifier_bbox_fallback=False):
    """Verify one synchronized robot-camera observation; no motion or GT lookup."""
    image_b64 = base64.b64encode(Path(rgb_path).read_bytes()).decode("ascii")
    record.update(validated=False, target=target, instruction=instruction,
        required_spatial_relation=qwen_bbox.spatial_requirement(target, instruction),
        detector_status="NOT_RUN", semantic_verification_status="NOT_RUN",
        rgbd_verification_status="NOT_RUN", hsv_verification_status="NOT_RUN",
        candidate_accepted=False, bbox_iou=None, verifier_bbox=None,
        bbox_source=None, fallback_used=False)
    record["visual_attempts"] = []
    for attempt_index in range(2):  # One total same-frame retry, never recapture/move.
        attempt = {"attempt": attempt_index, "detector": [], "verifier": []}
        record["visual_attempts"].append(attempt)
        record.pop("bbox_pixels", None)
        record.pop("rejection_detail", None)
        record.update(verifier_bbox=None, bbox_iou=None, semantic_verification_status="NOT_RUN")
        try:
            _, raw = predict_bbox(image_b64, target, diagnostics=attempt["detector"],
                max_retries=0, instruction=instruction)
            record["detector_raw"] = raw
            if allow_verifier_bbox_fallback and isinstance(json.loads(raw),dict) and json.loads(raw).get("status")=="TARGET_FOUND":
                record["detector_status"]="TARGET_FOUND"
                verify_raw=verify_target(image_b64,target,instruction,attempt["verifier"])
                record["verification_raw"]=verify_raw
                bbox_norm=match_candidate_bboxes(raw,verify_raw,target,instruction,
                    (obs.camera.width,obs.camera.height),record)
                record["bbox_normalized"]=bbox_norm
                record["bbox_pixels"]=qwen_bbox.bbox_to_pixels(bbox_norm,obs.camera.width,obs.camera.height)
                status="TARGET_FOUND"
            else:
                answer = qwen_bbox.parse_visual_answer(raw, target, record["required_spatial_relation"])
                record["detector_normalized"] = answer
                bbox_norm = answer["bbox_2d"]
                record["bbox_normalized"] = bbox_norm
                record["bbox_coordinate_scale"] = answer["coordinate_scale"]
                record["detector_status"] = answer["status"]
                if answer["status"] != "TARGET_FOUND":
                    status = answer["status"]
                else:
                    record["bbox_pixels"] = qwen_bbox.bbox_to_pixels(
                        bbox_norm, obs.camera.width, obs.camera.height)
                    verify_raw = verify_target(image_b64, target, instruction, attempt["verifier"])
                    record["verification_raw"] = verify_raw
                    status = classify_detection(bbox_norm, verify_raw, target, instruction,
                        diagnostics=record, image_size=(obs.camera.width, obs.camera.height))
            record["detection_status"] = status
            record["status"] = status
            if status != "TARGET_FOUND":
                record["rejection_reason"] = record.get("rejection_detail", status)
                return None
            break
        except json.JSONDecodeError as exc:
            attempt["error_type"] = "JSONDecodeError"
            attempt["error"] = str(exc)
        except (ValueError, TypeError, KeyError) as exc:
            attempt["error_type"] = "INVALID_VISUAL_RESPONSE"
            attempt["error"] = str(exc)
        if attempt["verifier"]:
            record["semantic_verification_status"] = "CANDIDATE_REJECTED"
        else:
            record["detector_status"] = "CANDIDATE_REJECTED"
        if attempt_index == 1:
            record["status"] = "CANDIDATE_REJECTED"
            record["rejection_reason"] = attempt["error_type"] + ": " + attempt["error"]
            return None
    try:
        surface_xyz, elevated = validate_candidate_position(
            record["bbox_pixels"], depth_path, obs, target, instruction, geometry, record)
    except (ValueError, TypeError, KeyError) as exc:
        record["rgbd_verification_status"] = "CANDIDATE_REJECTED"
        record["rgbd_rejection_reason"] = str(exc)
        record["rejection_reason"] = str(exc)
        record["status"] = "CANDIDATE_REJECTED"
        return None
    record["rgbd_verification_status"] = "TARGET_VALIDATED"
    record["target_surface_xyz"] = surface_xyz.tolist()
    record["rgbd_result_xyz"] = surface_xyz.tolist()
    record["elevated_target"] = elevated
    record["status"] = "TARGET_VALIDATED"
    record.pop("rejection_reason", None)
    record["validated"] = True
    return surface_xyz, elevated


def acquire_target(robot, adapter, ground_id, instruction, target, output_dir, run_id,
                   geometry):
    """Accept only a semantically AND geometrically validated candidate, within six views."""
    commands, records = [], []
    trace_path = output_dir / f"detection_{run_id}.json"
    for index, heading in enumerate(np.arange(6) * np.pi / 3):
        if index:
            turn_to_heading(robot, ground_id, float(heading), commands)
        obs = adapter.capture(timestamp_s=(20 + sum(c[1] for c in commands)) / 240,
            scene_id="warehouse_test_detour", task_id="semantic_navigation",
            frame_index=index)
        # Unique full-frame file: these exact bytes go to BOTH Qwen calls.
        rgb_path = output_dir / f"qwen_input_{run_id}_{index}.png"
        rgb_path.write_bytes(Path(obs.rgb_path).read_bytes())
        depth_path = output_dir / f"qwen_depth_{run_id}_{index}.npy"
        depth_path.write_bytes(Path(obs.depth_path).read_bytes())
        record = {"view": index, "instruction": instruction, "target": target,
            "rgb_path": str(rgb_path), "depth_path": str(depth_path),
            "camera": asdict(obs.camera), "robot_state": asdict(obs.state),
            "validated": False, "status": "OBSERVING"}
        records.append(record)
        save_json(trace_path, records)
        bbox = None
        try:
            result = evaluate_observation(
                obs, rgb_path, depth_path, target, instruction, geometry, record)
            bbox = record.get("bbox_pixels")
            if result is not None:
                surface_xyz, elevated = result
                print(f"View {index}: TARGET_VALIDATED; XYZ={surface_xyz}", flush=True)
                return obs, bbox, surface_xyz, elevated, commands, str(trace_path)
            print(f"View {index}: {record['status']}; continuing search", flush=True)
        except (ValueError, TypeError, KeyError) as exc:
            # Bad bbox/depth/semantics reject this view, not the entire search.
            record["status"] = "CANDIDATE_REJECTED"
            record["rejection_reason"] = str(exc)
            print(f"View {index}: candidate rejected: {exc}; continuing search", flush=True)
        except Exception as exc:
            record["status"] = "OBSERVATION_ERROR"
            record["error"] = str(exc)
            raise
        finally:
            # Persist a preview for every attempted view, including rejected
            # candidates and semantic/height failures, before continuing/returning.
            save_json(trace_path, records)
            preview = cv2.imread(str(rgb_path))
            if preview is None:
                raise RuntimeError("Cannot read captured RGB")
            bbox = record.get("bbox_pixels")
            if bbox is not None:
                cv2.rectangle(preview, tuple(map(int, bbox[:2])),
                              tuple(map(int, bbox[2:])), (0, 255, 255), 2)
            cv2.putText(preview, record["status"], (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1)
            preview_path = output_dir / f"detected_{run_id}_{index}.png"
            record["preview_path"] = str(preview_path)
            if not cv2.imwrite(str(preview_path), preview):
                raise RuntimeError("Could not save detection preview")
            (output_dir / "detected_target.png").write_bytes(preview_path.read_bytes())
            save_json(trace_path, records)
    reasons = [r.get("rejection_reason", r["status"]) for r in records]
    if any(r.get("detection_status") == "TARGET_AMBIGUOUS" for r in records):
        failure = "TARGET_AMBIGUOUS"
    elif all(r.get("detection_status") == "TARGET_NOT_VISIBLE" for r in records):
        failure = "TARGET_NOT_VISIBLE"
    else:
        failure = "TARGET_NOT_VERIFIED"
    raise RuntimeError(f"{failure}: bounded rotation search exhausted; "
        f"per-view failures: {reasons}. "
        "原地旋转后仍无验证通过的目标；目标可能被遮挡，需要移动到新的观察位置，未猜测位置。")


def build_visual_docking_context(obs, record, geometry, grid, arena_size, previous_context=None):
    """Local geometry ONLY from validated RGB/bbox/depth and this frame's camera.

    Target samples remain in collision geometry. Only their conservative ordinary
    inflation is replaced locally by swept footprint + observed free-space proof.
    No segmentation, body handles or semantic-scene metadata enter this context.
    """
    if not record.get("validated") or not record.get("candidate_accepted", False):
        raise ValueError("LOCAL_DOCKING_REQUIRES_VERIFIED_RGBD")
    rgb = cv2.imread(str(obs.rgb_path))
    depth = np.load(obs.depth_path, allow_pickle=False)
    c = obs.camera
    if rgb is None or depth.shape != (c.height,c.width):
        raise ValueError("LOCAL_DOCKING_FRAME_INVALID")
    T = np.asarray(c.T_world_camera,dtype=float).reshape(4,4)
    vv,uu = np.mgrid[0:c.height:3,0:c.width:3]
    z = depth[vv,uu]
    valid = np.isfinite(z) & (z > 0) & (z < 8.)
    camera_points=np.stack([(uu-c.cx)*z/c.fx,(vv-c.cy)*z/c.fy,z],axis=-1)
    world=camera_points@T[:3,:3].T+T[:3,3]
    high=valid & (world[:,:,2] > .035) & (world[:,:,2] < geometry["top_z_m"]+.15)
    box=record["bbox_pixels"]
    in_box=(uu>=box[0])&(uu<=box[2])&(vv>=box[1])&(vv<=box[3])
    # Conservative color-consistent visual target samples; background keeps inflation.
    hsv=cv2.cvtColor(rgb,cv2.COLOR_BGR2HSV)[vv,uu]
    color=record.get("color_check") or record["target"].split()[0]
    h,saturation,value=hsv[:,:,0],hsv[:,:,1],hsv[:,:,2]
    colored=(saturation>=70)&(value>=40)
    ranges={"red":((h<=10)|(h>=170)),"green":((h>=35)&(h<=85)),"blue":((h>=90)&(h<=135))}
    target=in_box & colored & ranges[color]
    surface=np.asarray(record["target_surface_xyz"],dtype=float)
    target &= np.linalg.norm(world-surface,axis=-1)<.5
    if np.count_nonzero(target & high)<8:
        raise ValueError("LOCAL_DOCKING_INSUFFICIENT_TARGET_DEPTH")
    context={"points":world[high],"ordinary_points":world[high & ~target],
        "depth":depth,"T":T,"camera":c,"grid":grid,"arena_size":arena_size,
        "resolution":occ.RESOLUTION,"geometry":geometry,
        "inflation_m":max(occ.SAFETY_MARGIN,geometry["radius_m"]+occ.RESOLUTION/np.sqrt(2)),
        "target_sample_count":int(np.count_nonzero(target & high))}
    if previous_context is not None:
        # Static scene: retain measured obstacles AND already certified free space
        # from this same run. A new view's blind area is not a new obstacle.
        context["previous_context"]=previous_context
        context["points"]=np.concatenate([previous_context["points"],context["points"]])
        context["ordinary_points"]=np.concatenate([previous_context["ordinary_points"],context["ordinary_points"]])
    return context


def visual_footprint_clear(pose, context):
    """Retain ordinary obstacle inflation and require visible free space at exceptions."""
    g=context["geometry"]
    center=np.asarray(pose[:2],dtype=float)
    heading=np.array([np.cos(pose[2]),np.sin(pose[2])])
    lateral=np.array([-heading[1],heading[0]])
    points=context["points"]
    delta=points[:,:2]-center
    forward=delta@heading; side=delta@lateral
    # Real target and all other measured surfaces remain physical collision barriers.
    if np.any((forward>=-g["rear_m"]-.015)&(forward<=g["front_m"]+.015)
              &(np.abs(side)<=g["half_width_m"]+.015)):
        context["last_rejection"]="MEASURED_SURFACE_INTERSECTS_FOOTPRINT"
        return False
    ordinary=context["ordinary_points"]
    if len(ordinary):
        distances=np.linalg.norm(ordinary[:,:2]-center,axis=1)
        if np.min(distances)<context.get("inflation_m",occ.SAFETY_MARGIN):
            context["last_rejection"]=f"ORDINARY_INFLATION: nearest={ordinary[np.argmin(distances)].tolist()}, distance={np.min(distances):.4f}"
            return False
    center_cell=planner.world_to_grid(center,context["resolution"],context["arena_size"],context["grid"])
    if center_cell is not None and context["grid"][center_cell]==0:
        return True
    xs=np.linspace(-g["rear_m"],g["front_m"],max(3,int(g["length_m"]/.035)+1))
    ys=np.linspace(-g["half_width_m"],g["half_width_m"],max(3,int(g["width_m"]/.035)+1))
    footprint=np.array([center+x*heading+y*lateral for x in xs for y in ys])
    needs=[]
    for xy in footprint:
        cell=planner.world_to_grid(xy,context["resolution"],context["arena_size"],context["grid"])
        if cell is None:return False
        if context["grid"][cell] != 0:
            # A free global cell certifies a robot-radius disk, regardless of yaw.
            # Reuse those geometric free-space certificates at body points that
            # are below/behind the current camera; never identify an obstacle.
            radius=context.get("inflation_m",occ.SAFETY_MARGIN)
            reach=int(np.ceil(radius/context["resolution"]))+1
            certified=False
            for row in range(max(0,cell[0]-reach),min(context["grid"].shape[0],cell[0]+reach+1)):
                for col in range(max(0,cell[1]-reach),min(context["grid"].shape[1],cell[1]+reach+1)):
                    if context["grid"][row,col]==0:
                        world=np.asarray(planner.grid_to_world((row,col),context["resolution"],context["arena_size"]))
                        if np.linalg.norm(world-xy)<=radius-.01:
                            certified=True;break
                if certified:break
            if not certified:needs.append(xy)
    if not needs:return True
    # Every footprint region inside a globally inflated cell must be certified
    # free at body heights by actual depth rays. Unknown/occluded regions fail closed.
    sample=np.array([[xy[0],xy[1],z] for xy in needs for z in (.04,g["top_z_m"]*.65,g["top_z_m"])])
    T=context["T"];camera=(sample-T[:3,3])@T[:3,:3]
    if np.any(camera[:,2]<=.02):
        context["last_rejection"]="UNOBSERVED_BODY_BEHIND_CAMERA"
        return visual_footprint_clear(pose,context["previous_context"]) if "previous_context" in context else False
    c=context["camera"]
    u=np.rint(c.fx*camera[:,0]/camera[:,2]+c.cx).astype(int)
    v=np.rint(c.fy*camera[:,1]/camera[:,2]+c.cy).astype(int)
    outside=(u<0)|(u>=c.width)|(v<0)|(v>=c.height)
    if np.any(outside):
        context["last_rejection"]=f"UNOBSERVED_BODY_OUTSIDE_FRAME: {sample[np.flatnonzero(outside)[0]].tolist()}"
        return visual_footprint_clear(pose,context["previous_context"]) if "previous_context" in context else False
    measured=context["depth"][v,u]
    free=bool(np.all(np.isfinite(measured)&(measured>camera[:,2]+.015)))
    if not free:
        context["last_rejection"]="DEPTH_DOES_NOT_CERTIFY_FREE_BODY_VOLUME"
        if "previous_context" in context:return visual_footprint_clear(pose,context["previous_context"])
    return free


def visual_corridor_clear(start, end, yaw, context):
    steps=max(2,int(np.linalg.norm(np.asarray(end)-np.asarray(start))/.01)+2)
    return all(visual_footprint_clear(np.array([*xy,yaw]),context)
               for xy in np.linspace(start,end,steps))


def select_local_docking(pre_xy, surface_xyz, context):
    """Continuous straight docking corridor on the currently observed face."""
    pre=np.asarray(pre_xy,dtype=float);surface=np.asarray(surface_xyz[:2],dtype=float)
    delta=surface-pre;distance=float(np.linalg.norm(delta))
    if distance<1e-6:raise RuntimeError("NO_SAFE_LOCAL_DOCKING_CORRIDOR")
    direction=delta/distance
    goal=surface-direction*(context["geometry"]["front_m"]+.12)
    yaw=float(np.arctan2(direction[1],direction[0]))
    if distance < context["geometry"]["front_m"]+.08:
        raise RuntimeError("LOCAL_DOCKING_ALREADY_TOO_CLOSE")
    if not visual_corridor_clear(pre,goal,yaw,context):
        raise RuntimeError("NO_SAFE_LOCAL_DOCKING_CORRIDOR: "+context.get("last_rejection","visual footprint/free-space check"))
    return goal,yaw


def select_parking(grid, start_cell, start, surface, view_xy, geometry, arena_size):
    """Select a GLOBAL inflated-map pre-stop; local docking enforces the final gap."""
    resolution = occ.RESOLUTION
    side = view_xy - surface
    if np.linalg.norm(side) < 1e-6:
        raise RuntimeError("Invalid target viewing direction")
    side /= np.linalg.norm(side)
    # Four-connected component labels are only a reachability prefilter.
    # With A* corner-cutting forbidden, diagonal moves cannot join these components.
    _, labels = cv2.connectedComponents((grid == 0).astype(np.uint8), connectivity=4)
    cells = np.argwhere((grid == 0) & (labels == labels[start_cell]))
    points = np.array([planner.grid_to_world(tuple(c), resolution, arena_size) for c in cells])
    offsets = points - surface
    distances = np.linalg.norm(offsets, axis=1)
    # Remain on the observed face side, including for elevated shelf targets.
    eligible = ((offsets @ side >= 0.5 * distances)
                & (distances >= geometry["front_m"] + 0.08))
    ids = np.flatnonzero(eligible)
    ids = sorted(ids, key=lambda i: (float(distances[i]), float(np.linalg.norm(points[i] - start))))
    expanded_total = 0
    for i in ids:
        path, expanded = planner.astar(grid, start_cell, tuple(cells[i]), resolution)
        expanded_total += expanded
        if path:
            goal = points[i]
            yaw = float(np.arctan2(*(surface - goal)[::-1]))
            gap = float(distances[i] - geometry["front_m"])
            return goal, path, expanded_total, float(distances[i]), yaw, gap
    raise RuntimeError("No reachable safe parking cell on the observed target side")


def main():
    instruction = input(
        "请输入导航指令："
    ).strip()

    if not instruction:
        raise ValueError("Navigation instruction is empty")

    output_dir = ROOT / "logs" / "p4_live"

    run_id = str(time.time_ns())
    output_path = ROOT / "logs/p4_visual_goal_plan.json"
    # Invalidate the previous plan before any perception/search can fail.
    save_json(output_path, {"status": "PLANNING", "instruction": instruction,
                            "run_id": run_id})
    client = p.connect(p.DIRECT)
    if client < 0:
        raise RuntimeError("PyBullet DIRECT connection failed")

    try:
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath()
        )
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1 / 240)

        plane_id = p.loadURDF("plane.urdf")

        builder = EnvironmentBuilder(client)
        builder.create_warehouse_environment(
            arena_size=10.0
        )

        add_semantic_targets(layout="test")
        add_static_detour_obstacle()

        robot = DiffDriveRobot(
            str(ROOT / "urdf/diff_drive_robot.urdf"),
            start_pos=[0, 0, 0.1],
            start_yaw=0.0,
        )

        # Let the robot settle before taking the image.
        for _ in range(20):
            p.stepSimulation()

        adapter = PyBulletObservationAdapter(
            robot=robot,
            output_dir=output_dir,
            width=640,
            height=480,
        )

        geometry = robot_geometry(robot)
        print("\n1. Understanding instruction...", flush=True)
        target_name = understand_instruction(
            instruction, output_dir / f"instruction_{run_id}.json")
        print(f"Recognized target: {target_name}", flush=True)
        print("\n2. Visual detection and bounded active search...", flush=True)
        (obs, bbox, surface_xyz, elevated,
         search_commands, detection_trace) = acquire_target(
            robot, adapter, plane_id, instruction, target_name, output_dir, run_id,
            geometry)
        # acquire_target returns only after this frame's RGB-D and spatial
        # validation pass. Keep the exact accepted estimate and observation pose.
        start_pose = robot.get_pose().copy()
        start = start_pose[:2].copy()
        surface = surface_xyz[:2].copy()
        print(f"Validated target XYZ: {surface_xyz}")

        # The planning scene is temporary. Remove the robot
        # so it is not treated as an obstacle in the grid.
        p.removeBody(robot.robot_id)

        obstacles = occ.collect_obstacle_aabbs(
            plane_id
        )

        arena_size = 10.0
        resolution = occ.RESOLUTION

        # Never reduce global inflation. Increase it if own footprint plus
        # cell discretization exceeds the existing safety margin.
        inflation = max(occ.SAFETY_MARGIN,
                        geometry["radius_m"] + resolution / np.sqrt(2))
        extra = inflation - occ.SAFETY_MARGIN
        grid = occ.create_grid([(x0-extra, y0-extra, x1+extra, y1+extra)
                                for x0, y0, x1, y1 in obstacles], arena_size)

        start_cell = planner.world_to_grid(
            start,
            resolution,
            arena_size,
            grid,
        )

        if (
            start_cell is None
            or grid[start_cell] != 0
        ):
            raise RuntimeError(
                "Start position is blocked"
            )

        print("\n4. A* planning...")

        (selected_goal, selected_path, expanded_nodes, selected_clearance,
         goal_yaw, planned_gap) = select_parking(
            grid, start_cell, start, surface,
            np.array([obs.state.x_m, obs.state.y_m]), geometry, arena_size)
        close_feasible = 0.08 <= planned_gap <= 0.15
        parking_reason = ("Preferred front gap is reachable" if close_feasible else
            "Nearest A*-reachable cell on the observed face side; preserved obstacle "
            "inflation, footprint and grid resolution prevent a 0.08-0.15m gap")
        if elevated:
            parking_reason += "; elevated target: ground observation stop in front of the visible shelf side"
        print(f"Parking: {parking_reason}; planned front gap={planned_gap:.3f}m")

        waypoints = [
            [
                float(v)
                for v in planner.grid_to_world(
                    node, resolution, arena_size
                )
            ]
            for node in selected_path
        ]

        # Diagnose the direct route in the same occupancy grid.
        # A clear direct route is valid for multi-target navigation.
        straight_blocked = False

        for fraction in np.linspace(0.05, 0.95, 100):
            point = (
                start * (1 - fraction)
                + selected_goal * fraction
            )

            cell = planner.world_to_grid(
                point,
                resolution,
                arena_size,
                grid,
            )

            if cell is None or grid[cell] != 0:
                straight_blocked = True
                break

        output = {
            "status": "READY",
            "plan_version": 2,
            "scene_id": "warehouse_test_detour",
            "layout": "test",
            "run_id": run_id,
            "detection_trace": detection_trace,
            "perception_status": "TARGET_FOUND",  # P26-compatible; validated below.
            "target_validated": True,
            "search_commands": search_commands,
            "start_pose": start_pose.tolist(),
            "goal_yaw_rad": goal_yaw,
            "robot_geometry": geometry,
            "map_inflation_m": inflation,
            "map_resolution_m": resolution,
            "elevated_target": elevated,
            "planned_front_gap_m": planned_gap,
            "close_approach_feasible": close_feasible,
            "parking_reason": parking_reason,
            "instruction": instruction,
            "target": target_name,
            "bbox_pixels": bbox,
            "target_surface_xyz": [
                float(v) for v in surface_xyz
            ],
            "target_surface_xy": surface.tolist(),
            "target_standoff_m": selected_clearance,
            "start_xy": start.tolist(),
            "stop_goal_xy": selected_goal.tolist(),
            "waypoints_world": waypoints,
            "expanded_nodes": int(expanded_nodes),
            "straight_path_blocked": straight_blocked,
        }

        output_path = (
            ROOT / "logs/p4_visual_goal_plan.json"
        )

        output_path.write_text(
            json.dumps(
                output,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )

        print("\nP4-4 SEMANTIC PLANNING PASSED")
        print(f"Instruction: {instruction}")
        print(f"Target: {target_name}")
        print(f"Target XY: {surface}")
        print(f"Safe stop: {selected_goal}")
        print(f"Path nodes: {len(waypoints)}")
        print(f"Expanded nodes: {expanded_nodes}")
        print(f"Straight path blocked: {straight_blocked}")
        print(f"Saved: {output_path}")

    except Exception as exc:
        save_json(output_path, {"status": "FAILED", "instruction": instruction,
            "run_id": run_id, "error": str(exc),
            "detection_trace": str(output_dir / f"detection_{run_id}.json")})
        print(f"Navigation planning failed: {exc}", flush=True)
        raise
    finally:
        if p.isConnected(client):
            p.disconnect(client)


if __name__ == "__main__":
    main()
