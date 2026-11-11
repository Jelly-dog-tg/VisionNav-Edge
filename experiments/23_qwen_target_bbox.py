
"""P3-4: Qwen3-VL bounding-box prediction and GT evaluation."""

import base64
import json
import math
import re
import urllib.request
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "logs" / "p2_dataset"
MODEL = "qwen3-vl:2b-instruct"
API = "http://127.0.0.1:11434/api/chat"

TARGETS = {
    "red": "red box",
    "blue": "blue cylinder",
    "green": "green sphere",
}


RELATIONS = {"floor": "floor", "on the floor": "floor", "on the ground": "floor",
    "shelf_top": "shelf_top", "on top of the shelf": "shelf_top", "atop the shelf": "shelf_top",
    "unspecified": "unspecified", "none": "unspecified"}
CATEGORIES = {"box": "box", "red box": "box", "cube": "box", "block": "box", "square": "box",
    "sphere": "sphere", "ball": "sphere", "cylinder": "cylinder"}
COLORS = {"red": "red", "green": "green", "blue": "blue"}


def normalized_text(value):
    return " ".join(value.strip().lower().split()) if isinstance(value, str) else ""


def normalize_relation(value):
    key = normalized_text(value)
    if key not in RELATIONS:
        raise ValueError("TARGET_SPATIAL_RELATION_UNKNOWN")
    return RELATIONS[key]


def target_identity(target):
    """Exact supported descriptions, never substring-based identity acceptance."""
    text = normalized_text(target)
    for phrase in ("on top of the shelf", "atop the shelf", "on the shelf", "on the floor", "on the ground"):
        text = text.replace(phrase, "").strip()
    match = re.fullmatch(r"(red|green|blue) (box|cube|block|square|sphere|ball|cylinder)", text)
    if not match:
        raise ValueError("TARGET_AMBIGUOUS_OR_UNSUPPORTED: invalid target identity")
    return match[1], CATEGORIES[match[2]]


def spatial_requirement(target, instruction=""):
    # A supplied user instruction is authoritative, even if it specifies no relation.
    text = (instruction if instruction else target).lower()
    shelf = any(w in text for w in ("shelf", "货架", "架顶", "架子上"))
    floor = any(w in text for w in ("floor", "ground", "地面", "地上"))
    if shelf and floor:
        raise ValueError("TARGET_SPATIAL_RELATION_CONFLICT")
    return "shelf_top" if shelf else "floor" if floor else "unspecified"


def canonical_target_description(target, instruction):
    color, category = target_identity(target)
    relation = spatial_requirement("", instruction)
    suffix = {"unspecified": "", "floor": " on the floor", "shelf_top": " on top of the shelf"}
    return f"{color} {category}" + suffix[relation]


def normalize_bbox(value, scale=None):
    if value is None:
        return None, None
    if (not isinstance(value, list) or len(value) != 4
            or any(type(v) not in (int, float) for v in value)):
        raise ValueError("TARGET_BBOX_INVALID: expected four finite JSON numbers")
    values = [float(v) for v in value]
    if not all(math.isfinite(v) and 0 <= v <= 1000 for v in values):
        raise ValueError("TARGET_BBOX_INVALID: nonfinite/out-of-range")
    used = scale or ("0_1" if all(v <= 1 for v in values) else "0_1000")
    if used == "0_1":
        values = [v*1000 for v in values]
    if (used not in ("0_1", "0_1000") or any(v > 1000 for v in values)
            or values[0] >= values[2] or values[1] >= values[3]):
        raise ValueError("TARGET_BBOX_INVALID: scale/order")
    return values, used


def check_bbox_reasonableness(box, width, height):
    pixels=bbox_to_pixels(box,width,height)
    area=(box[2]-box[0])*(box[3]-box[1])/1_000_000
    if area>=.90:
        raise ValueError("TARGET_BBOX_ABNORMAL_FULL_FRAME")
    return pixels


def validate_bbox(value):
    return normalize_bbox(value)[0]


def bbox_to_pixels(box, width, height):
    """Input is already normalized to 0..1000; do not infer scale twice."""
    box, _ = normalize_bbox(box, scale="0_1000")
    if box is None:
        raise ValueError("TARGET_BBOX_INVALID: no pixel box")
    pixels = [box[0]*width/1000, box[1]*height/1000,
              box[2]*width/1000, box[3]*height/1000]
    if pixels[2]-pixels[0] < 1 or pixels[3]-pixels[1] < 1:
        raise ValueError("TARGET_BBOX_TOO_SMALL: less than one pixel")
    return pixels


def visual_prompt(target, instruction="", verifier=False):
    relation = spatial_requirement(target, instruction)
    spatial = {"unspecified": "No spatial relation is requested. Do not invent one.",
        "floor": "Require visible evidence the object is on the floor.",
        "shelf_top": "Require visible support ON TOP of the shelf, not floor, inside, or merely high in the image."}[relation]
    role = "Independently verify" if verifier else "Locate"
    return (f"{role} the exact target: {target}. User instruction: {instruction}. "
        "Check color AND category in this image. " + spatial +
        " Do not guess hidden objects. Multiple indistinguishable matches: TARGET_AMBIGUOUS. "
        "Return JSON only: status (TARGET_FOUND/TARGET_NOT_VISIBLE/TARGET_AMBIGUOUS), "
        "category_match (boolean), color_match (boolean), spatial_match (boolean; true when unspecified), "
        "observed_category (box/sphere/cylinder or null; object category, not its 2D silhouette), "
        "observed_color (red/green/blue or null), "
        f"spatial_relation (must be {relation}), bbox_valid (boolean), "
        "bbox_2d ([xmin,ymin,xmax,ymax] normalized 0..1000 or null), "
        "matching_boxes (all matching boxes, [] if absent), failure_reason (short code). "
        "TARGET_FOUND requires exactly one valid box and all match flags true. "
        "bbox_2d:null means no detection; never use [null,null,null,null]. No long explanation.")


def normalize_visual_answer(answer, target, relation, diagnostics=None):
    """One strict protocol normalizer shared by detector, verifier and replay."""
    if not isinstance(answer, dict):
        raise ValueError("TARGET_PROTOCOL_INVALID: expected object")
    status = answer.get("status")
    if status not in ("TARGET_FOUND", "TARGET_NOT_VISIBLE", "TARGET_AMBIGUOUS"):
        raise ValueError("TARGET_PROTOCOL_INVALID: status")
    for key in ("category_match", "color_match", "spatial_match", "bbox_valid"):
        if type(answer.get(key)) is not bool:
            raise ValueError("TARGET_PROTOCOL_INVALID: missing boolean " + key)
    if not isinstance(answer.get("failure_reason"), str):
        raise ValueError("TARGET_PROTOCOL_INVALID: failure_reason")
    if "bbox_2d" not in answer and status == "TARGET_FOUND":
        raise ValueError("TARGET_BBOX_INVALID: FOUND missing bbox_2d")
    raw_box = answer.get("bbox_2d")
    # Only an internally consistent explicit non-detection may normalize four nulls.
    if (status == "TARGET_NOT_VISIBLE" and raw_box == [None]*4
            and answer["bbox_valid"] is False and answer.get("matching_boxes") == []
            and not answer["category_match"] and not answer["color_match"]
            and answer.get("observed_category") is None and answer.get("observed_color") is None):
        raw_box = None
    box, scale = normalize_bbox(raw_box)
    raw_boxes = answer.get("matching_boxes")
    if not isinstance(raw_boxes, list):
        raise ValueError("TARGET_PROTOCOL_INVALID: matching_boxes")
    boxes, scales = [], []
    for item in raw_boxes:
        if isinstance(item, dict):
            if "bbox_2d" not in item:
                raise ValueError("TARGET_BBOX_INVALID: matching object missing bbox_2d")
            # Optional labels are corroborating evidence, never a substitute for flags.
            if "label" in item:
                label_identity = target_identity(item["label"])
                if label_identity != target_identity(target):
                    raise ValueError("TARGET_RECOGNITION_MISMATCH: matching label identity")
            item = item["bbox_2d"]
        current, used = normalize_bbox(item)
        if current is None:
            raise ValueError("TARGET_BBOX_INVALID: null matching box")
        boxes.append(current); scales.append(used)
    observed_relation = normalize_relation(answer.get("spatial_relation"))
    required = normalize_relation(relation)
    if diagnostics is not None:
        diagnostics.update(bbox_2d=box, matching_boxes=boxes, coordinate_scale=scale,
                           matching_box_scales=scales, observed_spatial_relation=observed_relation)
    if answer["bbox_valid"] != (box is not None):
        raise ValueError("TARGET_RECOGNITION_MISMATCH: bbox validity conflict")
    if status == "TARGET_NOT_VISIBLE":
        if (box is not None or boxes or answer["category_match"] or answer["color_match"]
                or answer.get("observed_category") is not None or answer.get("observed_color") is not None):
            raise ValueError("TARGET_RECOGNITION_MISMATCH: absent with detection evidence")
    elif status == "TARGET_FOUND":
        if box is None or len(boxes) != 1 or boxes[0] != box:
            raise ValueError("TARGET_RECOGNITION_MISMATCH: FOUND box count/disagreement")
        if not all(answer[k] for k in ("category_match", "color_match", "spatial_match")):
            raise ValueError("TARGET_RECOGNITION_MISMATCH: status/match flags")
        expected_color, expected_category = target_identity(target)
        raw_category = normalized_text(answer.get("observed_category"))
        category = CATEGORIES.get(raw_category)
        color = COLORS.get(normalized_text(answer.get("observed_color")))
        if category != expected_category or color != expected_color:
            raise ValueError("TARGET_RECOGNITION_MISMATCH: observed category/color")
        if raw_category == "red box" and color != "red":
            raise ValueError("TARGET_RECOGNITION_MISMATCH: category contains conflicting color")
        # No user relation means no placement constraint; known observed placement is okay.
        if required != "unspecified" and observed_relation != required:
            raise ValueError("TARGET_RECOGNITION_MISMATCH: spatial conflict")
    normalized = dict(answer, bbox_2d=box, matching_boxes=boxes,
        observed_category=CATEGORIES.get(normalized_text(answer.get("observed_category"))),
        observed_color=COLORS.get(normalized_text(answer.get("observed_color"))),
        spatial_relation=observed_relation, coordinate_scale=scale,
        matching_box_scales=scales, required_spatial_relation=required)
    return normalized


def parse_visual_answer(raw, target, relation, diagnostics=None):
    return normalize_visual_answer(json.loads(raw), target, relation, diagnostics)


def validate_visual_answer(answer, target, relation):
    normalized = normalize_visual_answer(answer, target, relation)
    return normalized["bbox_2d"] if normalized["status"] == "TARGET_FOUND" else None


def visual_response_schema():
    """Ollama constrained JSON output; structure does not assert recognition."""
    box = {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1000},
           "minItems": 4, "maxItems": 4}
    properties = {
        "status": {"enum": ["TARGET_FOUND", "TARGET_NOT_VISIBLE", "TARGET_AMBIGUOUS"]},
        "category_match": {"type": "boolean"}, "color_match": {"type": "boolean"},
        "spatial_match": {"type": "boolean"}, "bbox_valid": {"type": "boolean"},
        "observed_category": {"enum": ["box", "sphere", "cylinder", None]},
        "observed_color": {"enum": ["red", "green", "blue", None]},
        "spatial_relation": {"enum": ["unspecified", "floor", "shelf_top"]},
        "bbox_2d": {"anyOf": [box, {"type": "null"}]},
        "matching_boxes": {"type": "array", "items": box},
        "failure_reason": {"type": "string"}}
    return {"type": "object", "properties": properties, "required": list(properties),
            "additionalProperties": False}


def visual_request(image_b64, prompt, diagnostics):
    """Save prompt/raw response/metadata even when response content is invalid JSON."""
    attempt = {"prompt": prompt, "raw": None, "response_metadata": None}
    diagnostics.append(attempt)
    payload = {"model": MODEL, "messages": [{"role": "user", "content": prompt,
        "images": [image_b64]}], "format": visual_response_schema(), "stream": False,
        "options": {"temperature": 0, "num_predict": 384}}
    attempt["response_schema"] = payload["format"]
    request = urllib.request.Request(API, data=json.dumps(payload).encode("utf8"),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=600) as response:
        envelope = response.read().decode("utf8")
    attempt["response_raw"] = envelope
    result = json.loads(envelope)
    attempt["response_metadata"] = {k: v for k, v in result.items() if k != "message"}
    raw = result["message"]["content"]
    attempt["raw"] = raw
    return raw


def predict_bbox(image_b64, target, *, diagnostics=None, max_retries=1, instruction=""):
    """Keep the existing (bbox, raw) interface; invalid output fails closed."""
    if max_retries not in (0, 1):
        raise ValueError("At most one retry is allowed")
    diagnostics = [] if diagnostics is None else diagnostics
    prompt = visual_prompt(target, instruction)
    raw = ""
    for attempt in range(max_retries + 1):
        try:
            raw = visual_request(image_b64, prompt, diagnostics)
            answer = parse_visual_answer(raw, target, spatial_requirement(target, instruction))
            diagnostics[-1]["normalized"] = answer
            box = answer["bbox_2d"] if answer["status"] == "TARGET_FOUND" else None
            return box, raw
        except json.JSONDecodeError as exc:
            diagnostics[-1]["error_type"] = "JSONDecodeError"
            diagnostics[-1]["error"] = str(exc)
        except (ValueError, TypeError, KeyError) as exc:
            diagnostics[-1]["error_type"] = "INVALID_VISUAL_RESPONSE"
            diagnostics[-1]["error"] = str(exc)
    return None, raw


def compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


def main():
    dataset = json.loads(
        (DATA_DIR / "dataset_index.json").read_text(
            encoding="utf-8"
        )
    )

    samples = [
        s for s in dataset["samples"]
        if s["split"] == "test"
    ]

    if len(samples) != 3:
        raise ValueError("Expected three test targets")

    image_path = (
        ROOT / samples[0]["model_input"]["rgb_path"]
    )

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(image_path)

    height, width = image.shape[:2]

    image_b64 = base64.b64encode(
        image_path.read_bytes()
    ).decode("ascii")

    results = []

    for sample in samples:
        gt = sample["ground_truth"]

        target = TARGETS[gt["color"]]

        bbox_norm, raw = predict_bbox(
            image_b64, target
        )

        result = {
            "target": target,
            "raw_answer": raw,
            "bbox_normalized": bbox_norm,
            "bbox_pixels": None,
            "iou": None,
        }

        if bbox_norm is None:
            print(f"{target}: NO VALID BBOX")
            results.append(result)
            continue

        try:
            bbox = bbox_to_pixels(bbox_norm, width, height)
        except ValueError as exc:
            result["rejection_reason"] = str(exc)
            results.append(result)
            continue

        # GT is used only for evaluation.
        mask = np.load(
            ROOT / gt["mask_path"],
            allow_pickle=False,
        )

        ys, xs = np.where(mask == gt["mask_id"])

        if len(xs) == 0:
            raise ValueError("GT target pixels missing")

        gt_bbox = [
            float(xs.min()),
            float(ys.min()),
            float(xs.max() + 1),
            float(ys.max() + 1),
        ]

        iou = compute_iou(bbox, gt_bbox)

        result["bbox_pixels"] = bbox
        result["iou"] = float(iou)

        results.append(result)

        cv2.rectangle(
            image,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            (0, 255, 255),
            2,
        )

        cv2.putText(
            image,
            target,
            (int(bbox[0]), max(20, int(bbox[1]) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        print(f"{target}: IoU={iou:.3f}")

    output_image = DATA_DIR / "p3_qwen_bboxes.png"
    output_json = DATA_DIR / "p3_qwen_bboxes.json"

    cv2.imwrite(str(output_image), image)

    output_json.write_text(
        json.dumps(
            results,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nP3-4 BBOX TEST FINISHED")
    print(f"Image: {output_image}")
    print(f"Results: {output_json}")


if __name__ == "__main__":
    main()
