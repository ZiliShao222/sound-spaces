#!/usr/bin/env python3

"""Build multi-goal trajectory dataset from valid_instances.json.

The input is the new-format file produced by export_valid_instances.py:
  output/<scene>/valid_instances.json

Main controls:
  - number of trajectories
  - min/max goals per trajectory
  - input valid_instances path
  - output trajectory dataset path
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import random
import re
import sys
from itertools import permutations
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from tqdm import tqdm
import numpy as np


def _infer_semantic_id(instance_key: str, raw_value: Any) -> Optional[int]:
    if isinstance(raw_value, int):
        return int(raw_value)
    if isinstance(raw_value, str) and raw_value.isdigit():
        return int(raw_value)
    match = re.search(r"_(\d+)$", instance_key)
    if match:
        return int(match.group(1))
    return None


def _is_vec3(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(v, (int, float)) for v in value)
    )


def _round_vec3(value: Optional[List[float]], digits: int = 3) -> Optional[List[float]]:
    if value is None or not _is_vec3(value):
        return None
    return [round(float(v), digits) for v in value]


def _load_multimodal_module() -> Any:
    module_path = Path(__file__).with_name("generate_multimodal_starts.py")
    spec = importlib.util.spec_from_file_location("generate_multimodal_starts", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load generate_multimodal_starts.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _resolve_scene_name(payload: Dict[str, Any], override: Optional[str]) -> str:
    if isinstance(override, str) and override.strip():
        return override.strip()

    scene_name = payload.get("scene_name")
    if isinstance(scene_name, str) and scene_name.strip():
        return scene_name.strip()

    scene_id = payload.get("scene_id")
    if isinstance(scene_id, str) and scene_id.strip():
        return Path(scene_id).stem

    raise RuntimeError("Unable to infer scene_name from input JSON.")


def _infer_scene_split(scene_id: Any) -> str:
    if not isinstance(scene_id, str) or not scene_id.strip():
        return "val"
    parts = [part.strip() for part in scene_id.split("/") if part.strip()]
    for split in ("train", "val", "test", "minival"):
        if split in parts:
            return split
    return "val"


def _normalize_episode_scene_id(scene_id: Any) -> str:
    if not isinstance(scene_id, str) or not scene_id.strip():
        return ""
    raw = scene_id.strip()
    if raw.startswith("mp3d/"):
        return raw[len("mp3d/") :]
    return raw


def _sound_id(scene_split: str, category: str) -> str:
    return f"{scene_split}/{category}.wav"


def _round_quat(values: List[float], digits: int = 6) -> List[float]:
    return [round(float(v), digits) for v in values]


def _serialize_floor_levels(floor_levels: List[Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for floor in floor_levels:
        rows.append(
            {
                "floor_index": int(getattr(floor, "index", -1)),
                "floor_y": round(float(getattr(floor, "y", 0.0)), 3),
                "min_y": round(float(getattr(floor, "min_y", 0.0)), 3),
                "max_y": round(float(getattr(floor, "max_y", 0.0)), 3),
                "triangle_count": int(getattr(floor, "count", 0)),
                "area_m2": round(float(getattr(floor, "area_m2", 0.0)), 3),
                "projected_area_m2": round(
                    float(getattr(floor, "projected_area_m2", 0.0)), 3
                ),
            }
        )
    return rows


def _nearest_floor(y_value: float, floor_levels: List[Any]) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    if not floor_levels:
        return None, None, None
    closest = min(
        floor_levels,
        key=lambda floor: abs(float(y_value) - float(getattr(floor, "y", y_value))),
    )
    floor_y = float(getattr(closest, "y", y_value))
    return int(getattr(closest, "index", -1)), floor_y, float(y_value) - floor_y


def _nearest_floor_y_value(y_value: float, floor_levels: List[Any]) -> float:
    if not floor_levels:
        return float(y_value)
    closest = min(
        floor_levels,
        key=lambda floor: abs(float(y_value) - float(getattr(floor, "y", y_value))),
    )
    return float(getattr(closest, "y", y_value))


def _snap_navmesh_point(
    pathfinder: Any,
    point: List[float],
    max_snap_distance: float,
) -> Optional[np.ndarray]:
    try:
        snapped = np.array(pathfinder.snap_point(np.array(point, dtype=np.float32)), dtype=np.float32)
    except Exception:
        return None
    if snapped.shape[0] != 3 or not np.all(np.isfinite(snapped)):
        return None
    if float(max_snap_distance) > 0.0:
        snap_xz_error = float(
            np.linalg.norm((snapped - np.array(point, dtype=np.float32))[[0, 2]])
        )
        if snap_xz_error > float(max_snap_distance):
            return None
    if hasattr(pathfinder, "is_navigable"):
        try:
            if not bool(pathfinder.is_navigable(snapped)):
                return None
        except Exception:
            return None
    return snapped


def _goal_nav_candidates(
    goal_record: Dict[str, Any],
    floor_levels: List[Any],
) -> List[List[float]]:
    center = goal_record.get("center")
    bbox_size = goal_record.get("bbox_size")
    nav_position = goal_record.get("nav_position")
    nav_floor_y = goal_record.get("nav_floor_y")

    candidates: List[List[float]] = []
    if _is_vec3(center):
        center_vec = [float(v) for v in center]
        base_y = float(center_vec[1])
        if _is_vec3(bbox_size):
            base_y = float(center_vec[1] - 0.5 * float(bbox_size[1]))

        if isinstance(nav_floor_y, (int, float)):
            floor_y = float(nav_floor_y)
        else:
            floor_y = _nearest_floor_y_value(base_y, floor_levels)

        candidates.append([float(center_vec[0]), floor_y, float(center_vec[2])])
        candidates.append([float(center_vec[0]), float(base_y), float(center_vec[2])])
        candidates.append(center_vec)

    if _is_vec3(nav_position):
        nav_vec = [float(v) for v in nav_position]
        if isinstance(nav_floor_y, (int, float)):
            candidates.append([float(nav_vec[0]), float(nav_floor_y), float(nav_vec[2])])
        else:
            candidates.append(
                [
                    float(nav_vec[0]),
                    _nearest_floor_y_value(float(nav_vec[1]), floor_levels),
                    float(nav_vec[2]),
                ]
            )
        candidates.append(nav_vec)

    deduped: List[List[float]] = []
    seen = set()
    for candidate in candidates:
        key = tuple(round(float(v), 3) for v in candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)
    return deduped


def _resolve_goal_navmesh_point(
    pathfinder: Any,
    goal_record: Dict[str, Any],
    floor_levels: List[Any],
    max_snap_distance: float,
) -> Optional[np.ndarray]:
    candidates = _goal_nav_candidates(goal_record, floor_levels)
    if not candidates:
        return None
    for candidate in candidates:
        snapped = _snap_navmesh_point(
            pathfinder,
            candidate,
            max_snap_distance=float(max_snap_distance),
        )
        if snapped is not None:
            return snapped
    return None


def _build_start_state(
    multimodal_module: Any,
    sim: Any,
    rng: random.Random,
    floor_levels: List[Any],
    min_clearance: float,
    max_attempts: int,
    floor_height_tolerance: float,
    width: int,
    height: int,
    hfov: float,
    sensor_height: float,
) -> Dict[str, Any]:
    start_position = multimodal_module._sample_start_state(
        pathfinder=sim.pathfinder,
        rng=rng,
        min_clearance=float(min_clearance),
        max_attempts=int(max_attempts),
        floor_levels=floor_levels,
        floor_height_tolerance=float(floor_height_tolerance),
    )
    yaw = rng.uniform(-math.pi, math.pi)
    rotation = multimodal_module._yaw_to_quaternion(yaw)
    return {
        "position": [round(float(v), 3) for v in start_position.tolist()],
        "rotation": _round_quat(rotation),
    }


def _build_pathfinder_simulator(
    scene_path: Path,
    scene_dataset_config: Optional[Path],
    exp_config: Path,
    width: int,
    height: int,
    hfov: float,
    sensor_height: float,
    gpu_device_id: int,
) -> Any:
    try:
        from ss_baselines.av_nav.config.default import get_task_config
        from habitat.sims import make_sim
    except Exception as exc:
        raise RuntimeError(
            "Failed to import Habitat/SoundSpaces dependencies needed for simulator start sampling."
        ) from exc

    cfg = get_task_config(config_paths=[str(exp_config)])
    cfg.defrost()
    cfg.SIMULATOR.TYPE = "ContinuousSoundSpacesSim"
    cfg.SIMULATOR.SCENE = str(scene_path)
    if scene_dataset_config is not None:
        cfg.SIMULATOR.SCENE_DATASET = str(scene_dataset_config)
    cfg.SIMULATOR.AUDIO.ENABLED = False
    cfg.SIMULATOR.CREATE_RENDERER = False
    if hasattr(cfg.SIMULATOR, "HABITAT_SIM_V0") and hasattr(
        cfg.SIMULATOR.HABITAT_SIM_V0, "GPU_DEVICE_ID"
    ):
        cfg.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = int(gpu_device_id)

    cfg.SIMULATOR.RGB_SENSOR.WIDTH = int(width)
    cfg.SIMULATOR.RGB_SENSOR.HEIGHT = int(height)
    cfg.SIMULATOR.SEMANTIC_SENSOR.WIDTH = int(width)
    cfg.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = int(height)
    if hasattr(cfg.SIMULATOR.RGB_SENSOR, "HFOV"):
        cfg.SIMULATOR.RGB_SENSOR.HFOV = float(hfov)
    if hasattr(cfg.SIMULATOR.SEMANTIC_SENSOR, "HFOV"):
        cfg.SIMULATOR.SEMANTIC_SENSOR.HFOV = float(hfov)
    if hasattr(cfg.SIMULATOR.RGB_SENSOR, "POSITION"):
        cfg.SIMULATOR.RGB_SENSOR.POSITION = [0.0, float(sensor_height), 0.0]
    if hasattr(cfg.SIMULATOR.SEMANTIC_SENSOR, "POSITION"):
        cfg.SIMULATOR.SEMANTIC_SENSOR.POSITION = [0.0, float(sensor_height), 0.0]

    # No rendering needed for start sampling; keep simulator in pathfinder-only mode.
    cfg.SIMULATOR.AGENT_0.SENSORS = []

    cfg.freeze()
    sim = make_sim(cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR)
    sim.reset()
    return sim


def _image_summary(image_payload: Dict[str, Any]) -> Dict[str, Any]:
    render_views = image_payload.get("render_views", [])
    num_views = int(image_payload.get("num_views", len(render_views)))
    summary: Dict[str, Any] = {
        "output_dir": image_payload.get("output_dir"),
        "views_json": image_payload.get("views_json"),
        "invalid_views_json": image_payload.get("invalid_views_json"),
        "num_views": num_views,
        "num_invalid_views": int(image_payload.get("num_invalid_views", 0)),
    }
    if isinstance(render_views, list) and render_views:
        summary["primary_view"] = render_views[0]
    return summary


def _safe_float(value: Any, default: float = -1.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return float(default)


def _render_views_from_instance(instance_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    image_payload = instance_record.get("image")
    if not isinstance(image_payload, dict):
        return []
    render_views = image_payload.get("render_views")
    if not isinstance(render_views, list):
        return []
    return [view for view in render_views if isinstance(view, dict)]


def _max_yolo_confidence(view: Dict[str, Any]) -> float:
    detections = view.get("yolo_matched_detections")
    if not isinstance(detections, list):
        return -1.0
    confidences = [
        _safe_float(det.get("confidence"), default=-1.0)
        for det in detections
        if isinstance(det, dict)
    ]
    if not confidences:
        return -1.0
    return float(max(confidences))


def _compact_render_view(view: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {}
    for key in (
        "resolution",
        "hfov",
        "position",
        "agent_base_position",
        "rotation",
        "radius",
        "angle_deg",
        "frame_cov",
        "iou",
        "yolo_num_detections",
        "yolo_matched_detections",
    ):
        if key in view:
            compact[key] = view[key]
    return compact


def _select_goal_render_view(
    render_views: List[Dict[str, Any]],
    rng: random.Random,
    strategy: str,
) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    if not render_views:
        return None, None

    if strategy == "random":
        index = int(rng.randrange(len(render_views)))
        return index, render_views[index]

    best_index = 0
    best_score = (
        _safe_float(render_views[0].get("iou"), default=-1.0),
        _safe_float(render_views[0].get("frame_cov"), default=-1.0),
        _max_yolo_confidence(render_views[0]),
    )
    for idx in range(1, len(render_views)):
        candidate = render_views[idx]
        score = (
            _safe_float(candidate.get("iou"), default=-1.0),
            _safe_float(candidate.get("frame_cov"), default=-1.0),
            _max_yolo_confidence(candidate),
        )
        if score > best_score:
            best_score = score
            best_index = idx
    return best_index, render_views[best_index]


def _build_goal_input_entry(
    record: Dict[str, Any],
    instance_record: Dict[str, Any],
    scene_split: str,
    rng: random.Random,
    view_strategy: str,
) -> Tuple[Dict[str, Any], str]:
    instance_key = str(record["instance_key"])
    category = str(record["category"])
    semantic_id = record.get("semantic_id")
    sound_id = _sound_id(scene_split, category)

    render_views = _render_views_from_instance(instance_record)
    selected_view_index, selected_view = _select_goal_render_view(
        render_views=render_views,
        rng=rng,
        strategy=str(view_strategy),
    )
    has_image = selected_view is not None and selected_view_index is not None

    available_modalities: List[str] = ["audio", "object"]
    if has_image:
        available_modalities.extend(["image", "text_description"])

    entry: Dict[str, Any] = {
        "instance_key": instance_key,
        "category": category,
        "semantic_id": semantic_id,
        "available_modalities": available_modalities,
        "primary_non_audio_modality": "image" if has_image else "object",
        "audio": {
            "sound_id": sound_id,
        },
        "object": {
            "instance_key": instance_key,
            "category": category,
            "semantic_id": semantic_id,
        },
        "image": None,
        "text_description": None,
    }
    if has_image:
        image_ref = {
            "view_id": f"{instance_key}#view_{int(selected_view_index)}",
            "selected_view_index": int(selected_view_index),
            "num_candidate_views": int(len(render_views)),
            "view_selection_strategy": str(view_strategy),
            "render_view": _compact_render_view(selected_view),
        }
        entry["image"] = image_ref
        entry["text_description"] = {
            "description_id": image_ref["view_id"],
            "source": "image_view",
            "text": None,
        }

    return entry, sound_id


def _goal_has_image(goal_input: Dict[str, Any]) -> bool:
    image_payload = goal_input.get("image")
    return isinstance(image_payload, dict)


def _assign_goal_modalities(
    goal_inputs: List[Dict[str, Any]],
    rng: random.Random,
    prefer_image_text_per_episode: bool,
    force_split_image_and_text: bool = False,
) -> Tuple[List[str], Dict[str, Any]]:
    image_candidates = [
        idx for idx, goal_input in enumerate(goal_inputs) if _goal_has_image(goal_input)
    ]
    goal_modalities: List[str] = [
        "image" if idx in image_candidates else "object"
        for idx in range(len(goal_inputs))
    ]

    if force_split_image_and_text and len(image_candidates) >= 2:
        goal_modalities[image_candidates[0]] = "image"
        goal_modalities[image_candidates[1]] = "text_description"
    elif prefer_image_text_per_episode and len(image_candidates) >= 2:
        text_idx = int(rng.choice(image_candidates))
        goal_modalities[text_idx] = "text_description"

    coverage = {
        "has_image_goal": bool("image" in goal_modalities),
        "has_text_description_goal": bool("text_description" in goal_modalities),
        "num_image_capable_goals": int(len(image_candidates)),
    }
    return goal_modalities, coverage


def _to_task_modality_token(goal_input: Dict[str, Any], modality: str) -> str:
    if modality == "text_description":
        return "description"
    if modality == "object":
        return "object"
    if modality == "image":
        image_payload = goal_input.get("image")
        if isinstance(image_payload, dict) and isinstance(
            image_payload.get("selected_view_index"), int
        ):
            return f"image_{int(image_payload['selected_view_index'])}"
        return "image_0"
    return str(modality)


def _record_has_image(record: Dict[str, Any]) -> bool:
    return bool(record.get("has_image"))


def _candidate_usage(
    record: Dict[str, Any],
    instance_usage_counter: Optional[Counter],
) -> int:
    if instance_usage_counter is None:
        return 0
    return int(instance_usage_counter.get(str(record.get("instance_key")), 0))


def _pick_low_usage_candidate(
    candidates: List[Dict[str, Any]],
    instance_usage_counter: Optional[Counter],
    rng: random.Random,
) -> Optional[Dict[str, Any]]:
    if not candidates:
        return None
    min_usage = min(_candidate_usage(candidate, instance_usage_counter) for candidate in candidates)
    best = [
        candidate
        for candidate in candidates
        if int(_candidate_usage(candidate, instance_usage_counter)) == int(min_usage)
    ]
    return rng.choice(best)


def _enforce_image_goal_quota(
    sampled_records: List[Dict[str, Any]],
    all_records: List[Dict[str, Any]],
    desired_image_goals: int,
    unique_categories: bool,
    instance_usage_counter: Optional[Counter],
    rng: random.Random,
) -> Optional[List[Dict[str, Any]]]:
    if desired_image_goals < 0:
        desired_image_goals = 0

    selected = list(sampled_records)
    selected_keys = {str(record["instance_key"]) for record in selected}
    selected_categories = [str(record["category"]) for record in selected]

    def _replace_once(require_image: bool) -> bool:
        if require_image:
            replace_indices = [
                idx for idx, record in enumerate(selected) if not _record_has_image(record)
            ]
        else:
            replace_indices = [
                idx for idx, record in enumerate(selected) if _record_has_image(record)
            ]
        if not replace_indices:
            return False

        rng.shuffle(replace_indices)
        replace_indices.sort(
            key=lambda idx: _candidate_usage(selected[idx], instance_usage_counter),
            reverse=True,
        )

        for replace_idx in replace_indices:
            old_record = selected[replace_idx]
            old_key = str(old_record["instance_key"])
            old_category = str(old_record["category"])

            candidates: List[Dict[str, Any]] = []
            for candidate in all_records:
                candidate_key = str(candidate["instance_key"])
                if candidate_key in selected_keys:
                    continue
                if bool(_record_has_image(candidate)) != bool(require_image):
                    continue

                candidate_category = str(candidate["category"])
                if unique_categories:
                    occupied = {
                        str(selected[idx]["category"])
                        for idx in range(len(selected))
                        if idx != int(replace_idx)
                    }
                    if candidate_category in occupied:
                        continue

                candidates.append(candidate)

            picked = _pick_low_usage_candidate(
                candidates,
                instance_usage_counter=instance_usage_counter,
                rng=rng,
            )
            if picked is None:
                continue

            selected[replace_idx] = picked
            selected_keys.remove(old_key)
            selected_keys.add(str(picked["instance_key"]))
            selected_categories[replace_idx] = str(picked["category"])
            return True
        return False

    while True:
        current_image_count = int(sum(_record_has_image(record) for record in selected))
        if current_image_count == int(desired_image_goals):
            break
        if current_image_count < int(desired_image_goals):
            if not _replace_once(require_image=True):
                return None
        else:
            if not _replace_once(require_image=False):
                return None

    return selected


def _flatten_instances(
    payload: Dict[str, Any],
    require_image: bool,
    include_image_summary: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    nested_instances = payload.get("instances")
    if not isinstance(nested_instances, dict):
        raise RuntimeError("Input valid_instances.json does not contain 'instances' dict.")

    flat_records: List[Dict[str, Any]] = []
    total_category_counts: Counter = Counter()
    eligible_category_counts: Counter = Counter()
    no_image_count = 0

    for category, per_category in nested_instances.items():
        if not isinstance(per_category, dict):
            continue
        category_name = str(category)
        for instance_key, record in per_category.items():
            if not isinstance(record, dict):
                continue

            key = str(instance_key)
            total_category_counts[category_name] += 1

            image_payload = record.get("image")
            has_image = (
                isinstance(image_payload, dict)
                and isinstance(image_payload.get("render_views"), list)
                and len(image_payload.get("render_views", [])) > 0
            )
            if require_image and not has_image:
                no_image_count += 1
                continue

            semantic_id = _infer_semantic_id(key, record.get("semantic_id"))
            item: Dict[str, Any] = {
                "instance_key": key,
                "category": category_name,
                "semantic_id": semantic_id,
                "has_image": bool(has_image),
                "center": _round_vec3(record.get("center")),
                "bbox_size": _round_vec3(record.get("bbox_size")),
                "nav_position": _round_vec3(record.get("nav_position")),
                "nav_floor_index": record.get("nav_floor_index"),
                "nav_floor_y": record.get("nav_floor_y"),
                "horizontal_radius": record.get("horizontal_radius"),
            }
            if include_image_summary and has_image:
                item["image"] = _image_summary(image_payload)

            flat_records.append(item)
            eligible_category_counts[category_name] += 1

    stats = {
        "num_instances_total": int(sum(total_category_counts.values())),
        "num_instances_eligible": int(len(flat_records)),
        "num_instances_excluded_no_image": int(no_image_count),
        "total_category_counts": dict(sorted(total_category_counts.items())),
        "eligible_category_counts": dict(sorted(eligible_category_counts.items())),
    }
    return flat_records, stats


def _build_instance_lookup(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Build direct instance_key -> full instance record mapping.

    The source is the full `instances` block from valid_instances.json (unfiltered),
    so downstream code can resolve `trajectory.goals[i]` directly.
    """
    nested_instances = payload.get("instances")
    if not isinstance(nested_instances, dict):
        return {}

    lookup: Dict[str, Dict[str, Any]] = {}
    for category, per_category in nested_instances.items():
        if not isinstance(per_category, dict):
            continue
        category_name = str(category)
        for instance_key, record in per_category.items():
            if not isinstance(record, dict):
                continue
            key = str(instance_key)
            full_record = copy.deepcopy(record)
            if "category" not in full_record:
                full_record["category"] = category_name
            if "semantic_id" not in full_record:
                full_record["semantic_id"] = _infer_semantic_id(
                    key, full_record.get("semantic_id")
                )

            full_record.pop("nav_floor_index", None)
            full_record.pop("nav_floor_y", None)
            full_record.pop("nav_floor_delta", None)

            image_payload = full_record.get("image")
            if isinstance(image_payload, dict):
                image_payload.pop("output_dir", None)
                image_payload.pop("views_json", None)
                image_payload.pop("invalid_views_json", None)
                image_payload.pop("num_invalid_views", None)

            render_views = _render_views_from_instance(full_record)
            has_image = len(render_views) > 0
            full_record["modalities"] = {
                "audio": True,
                "object": True,
                "image": bool(has_image),
                "text_description": bool(has_image),
            }
            full_record["preferred_non_audio_modality"] = (
                "image" if has_image else "object"
            )
            full_record["num_image_views"] = int(len(render_views))

            lookup[key] = full_record
    return lookup


def _sample_goals_for_trajectory(
    records: List[Dict[str, Any]],
    grouped_by_category: Dict[str, List[Dict[str, Any]]],
    num_goals: int,
    rng: random.Random,
    unique_categories: bool,
    instance_usage_counter: Optional[Counter] = None,
    coverage_explore_ratio: float = 0.15,
) -> List[Dict[str, Any]]:
    use_balanced_sampling = (
        instance_usage_counter is not None
        and float(coverage_explore_ratio) < 1.0
        and rng.random() >= float(coverage_explore_ratio)
    )

    def _usage_of(record: Dict[str, Any]) -> int:
        if instance_usage_counter is None:
            return 0
        return int(instance_usage_counter.get(str(record["instance_key"]), 0))

    if use_balanced_sampling and unique_categories:
        categories = [category for category, items in grouped_by_category.items() if items]
        if num_goals > len(categories):
            raise RuntimeError(
                f"Cannot sample {num_goals} unique categories from {len(categories)} available categories."
            )

        selected_categories: List[str] = []
        remaining_categories = list(categories)
        for _ in range(int(num_goals)):
            category_scores = {
                category: min(
                    _usage_of(record)
                    for record in grouped_by_category.get(category, [])
                )
                for category in remaining_categories
            }
            best_score = min(category_scores.values())
            best_categories = [
                category
                for category, score in category_scores.items()
                if int(score) == int(best_score)
            ]
            chosen_category = str(rng.choice(best_categories))
            selected_categories.append(chosen_category)
            remaining_categories.remove(chosen_category)

        goals: List[Dict[str, Any]] = []
        for category in selected_categories:
            candidates = grouped_by_category.get(category, [])
            if not candidates:
                continue
            best_usage = min(_usage_of(record) for record in candidates)
            best_candidates = [
                record
                for record in candidates
                if int(_usage_of(record)) == int(best_usage)
            ]
            goals.append(rng.choice(best_candidates))

        if len(goals) != int(num_goals):
            raise RuntimeError(
                "Failed to sample balanced goals under unique category constraints."
            )
        rng.shuffle(goals)
        return goals

    if use_balanced_sampling and not unique_categories:
        if num_goals > len(records):
            raise RuntimeError(
                f"Cannot sample {num_goals} unique instances from {len(records)} eligible instances."
            )
        remaining = list(records)
        goals: List[Dict[str, Any]] = []
        for _ in range(int(num_goals)):
            best_usage = min(_usage_of(record) for record in remaining)
            best_candidates = [
                record
                for record in remaining
                if int(_usage_of(record)) == int(best_usage)
            ]
            chosen = rng.choice(best_candidates)
            goals.append(chosen)
            remaining.remove(chosen)
        return goals

    if unique_categories:
        categories = [category for category, items in grouped_by_category.items() if items]
        if num_goals > len(categories):
            raise RuntimeError(
                f"Cannot sample {num_goals} unique categories from {len(categories)} available categories."
            )
        selected_categories = rng.sample(categories, num_goals)
        goals = [rng.choice(grouped_by_category[category]) for category in selected_categories]
        rng.shuffle(goals)
        return goals

    if num_goals > len(records):
        raise RuntimeError(
            f"Cannot sample {num_goals} unique instances from {len(records)} eligible instances."
        )
    return rng.sample(records, num_goals)


def _compute_total_geodesic_distances_from_matrix(
    distance_matrix: List[List[Optional[float]]],
) -> Tuple[Optional[float], Optional[float]]:
    """Compute ordered and unordered total geodesic distances from a distance matrix.

    Matrix layout:
      - index 0: start
      - index 1..n: goals in sampled order

    Returns:
      - ordered total: start->goal1->goal2->... (sampled order)
      - unordered total: minimum over all permutations of goals
    """

    if not distance_matrix:
        return None, None

    n = int(len(distance_matrix) - 1)
    if n <= 0:
        return 0.0, 0.0
    if n > 8:
        raise RuntimeError(
            "Brute-force unordered distance computation supports up to 8 goals. "
            f"Got n={n}."
        )

    def _path_length(order: Tuple[int, ...]) -> Optional[float]:
        first_leg = distance_matrix[0][order[0]]
        if first_leg is None:
            return None
        total = float(first_leg)
        for idx in range(len(order) - 1):
            leg = distance_matrix[order[idx]][order[idx + 1]]
            if leg is None:
                return None
            total += float(leg)
        return total

    ordered_order = tuple(range(1, n + 1))
    ordered_total = _path_length(ordered_order)

    best_unordered: Optional[float] = None
    for perm in permutations(ordered_order):
        candidate = _path_length(perm)
        if candidate is None:
            continue
        if best_unordered is None or float(candidate) < float(best_unordered):
            best_unordered = float(candidate)

    ordered_rounded = (
        round(float(ordered_total), 3) if ordered_total is not None else None
    )
    unordered_rounded = (
        round(float(best_unordered), 3) if best_unordered is not None else None
    )
    return ordered_rounded, unordered_rounded


def _build_trajectories(
    records: List[Dict[str, Any]],
    instance_lookup: Dict[str, Dict[str, Any]],
    num_trajectories: int,
    min_goals: int,
    max_goals: int,
    seed: int,
    unique_categories: bool,
    scene_name: str,
    start_sampler: Optional[Callable[[random.Random], Dict[str, Any]]] = None,
    start_goal_distance_fn: Optional[
        Callable[[List[float], Dict[str, Any]], Optional[float]]
    ] = None,
    goal_pair_distance_fn: Optional[
        Callable[[Dict[str, Any], Dict[str, Any]], Optional[float]]
    ] = None,
    distance_max_attempts: int = 100,
    min_goal_distance: float = 4.0,
    scene_split: str = "val",
    audio_duration: int = 25,
    audio_offset_min: int = 0,
    audio_offset_max: int = 50,
    audio_schedule: str = "round_robin",
    goal_image_view_strategy: str = "best_iou",
    balance_instance_coverage: bool = True,
    coverage_explore_ratio: float = 0.15,
    prefer_image_text_per_episode: bool = True,
    min_image_goals_per_episode: int = 1,
    max_image_goals_per_episode: int = 2,
    episode_scene_id: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rng = random.Random(int(seed))
    instance_usage_counter: Counter = Counter()

    total_image_records = int(sum(_record_has_image(record) for record in records))
    if total_image_records <= 0:
        min_image_goals_per_episode = 0
        max_image_goals_per_episode = 0

    grouped_by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped_by_category[record["category"]].append(record)

    max_possible = (
        len(grouped_by_category) if unique_categories else len(records)
    )
    effective_max_goals = min(int(max_goals), int(max_possible))
    if int(min_goals) > effective_max_goals:
        raise RuntimeError(
            "Sampling constraints are infeasible: "
            f"min_goals={int(min_goals)} but max_possible_goals={effective_max_goals}."
        )

    trajectories: List[Dict[str, Any]] = []
    goal_count_hist: Counter = Counter()

    for trajectory_index in tqdm(range(int(num_trajectories)), desc="generating trajectories"):
        goal_count = rng.randint(int(min_goals), int(effective_max_goals))
        while True:
            best_payload: Optional[Dict[str, Any]] = None
            best_null_count: Optional[int] = None

            desired_image_min = max(0, int(min_image_goals_per_episode))
            desired_image_max = max(0, int(max_image_goals_per_episode))
            if desired_image_min > desired_image_max:
                desired_image_min = desired_image_max

            required_min_image_goals = 0
            force_split_image_and_text = False
            if int(total_image_records) < 2:
                desired_image_min = 0
            else:
                # Rule by episode goal count:
                # - goals >= 4: must include both image and text_description modalities
                # - goals >= 3: must include at least one image/text_description modality
                # - others: fallback to CLI range config
                if int(goal_count) >= 4:
                    required_min_image_goals = 2
                    force_split_image_and_text = True
                elif int(goal_count) >= 3:
                    required_min_image_goals = 1

            desired_image_min = max(int(desired_image_min), int(required_min_image_goals))
            desired_image_max = max(int(desired_image_max), int(required_min_image_goals))

            desired_image_min = min(desired_image_min, int(goal_count), int(total_image_records))
            desired_image_max = min(desired_image_max, int(goal_count), int(total_image_records))
            if int(desired_image_min) < int(required_min_image_goals):
                if int(goal_count) > 2:
                    goal_count -= 1
                    continue
                raise RuntimeError(
                    "Insufficient image-capable goals for modality constraints: "
                    f"goal_count={int(goal_count)}, required_min_image_goals={int(required_min_image_goals)}, "
                    f"available_image_capable_instances={int(total_image_records)}."
                )
            if desired_image_max < desired_image_min:
                desired_image_max = desired_image_min

            if int(desired_image_max) >= int(desired_image_min):
                desired_image_goals = rng.randint(int(desired_image_min), int(desired_image_max))
            else:
                desired_image_goals = int(desired_image_min)

            for attempt_idx in range(max(1, int(distance_max_attempts))):
                active_usage_counter: Optional[Counter] = None
                active_explore_ratio = 1.0
                if bool(balance_instance_coverage):
                    active_usage_counter = instance_usage_counter
                    if int(distance_max_attempts) <= 1:
                        active_explore_ratio = float(coverage_explore_ratio)
                    else:
                        relax_threshold = int(max(1, round(0.7 * int(distance_max_attempts))))
                        if int(attempt_idx) < relax_threshold:
                            active_explore_ratio = float(coverage_explore_ratio)
                        else:
                            active_explore_ratio = 1.0

                sampled_records = _sample_goals_for_trajectory(
                    records=records,
                    grouped_by_category=grouped_by_category,
                    num_goals=goal_count,
                    rng=rng,
                    unique_categories=unique_categories,
                    instance_usage_counter=active_usage_counter,
                    coverage_explore_ratio=float(active_explore_ratio),
                )

                adjusted_records = _enforce_image_goal_quota(
                    sampled_records=sampled_records,
                    all_records=records,
                    desired_image_goals=int(desired_image_goals),
                    unique_categories=bool(unique_categories),
                    instance_usage_counter=active_usage_counter,
                    rng=rng,
                )
                if adjusted_records is None:
                    continue
                sampled_records = adjusted_records

                goal_instance_keys = [str(record["instance_key"]) for record in sampled_records]
                goal_categories = [str(record["category"]) for record in sampled_records]
                goal_inputs: List[Dict[str, Any]] = []
                goal_sound_ids: List[str] = []
                for record in sampled_records:
                    instance_key = str(record["instance_key"])
                    full_record = instance_lookup.get(instance_key, {})
                    goal_input, sound_id = _build_goal_input_entry(
                        record=record,
                        instance_record=full_record,
                        scene_split=scene_split,
                        rng=rng,
                        view_strategy=str(goal_image_view_strategy),
                    )
                    goal_inputs.append(goal_input)
                    goal_sound_ids.append(sound_id)

                goal_modalities, modality_coverage = _assign_goal_modalities(
                    goal_inputs=goal_inputs,
                    rng=rng,
                    prefer_image_text_per_episode=bool(prefer_image_text_per_episode),
                    force_split_image_and_text=bool(force_split_image_and_text),
                )

                if int(total_image_records) >= 2:
                    if int(goal_count) >= 4:
                        if not (
                            bool(modality_coverage.get("has_image_goal"))
                            and bool(modality_coverage.get("has_text_description_goal"))
                        ):
                            continue
                    elif int(goal_count) >= 3:
                        if not (
                            bool(modality_coverage.get("has_image_goal"))
                            or bool(modality_coverage.get("has_text_description_goal"))
                        ):
                            continue

                for index, modality in enumerate(goal_modalities):
                    goal_inputs[index]["selected_non_audio_modality"] = str(modality)

                goal_tasks: List[List[Any]] = []
                for record, goal_input, modality in zip(
                    sampled_records,
                    goal_inputs,
                    goal_modalities,
                ):
                    task_instance_key = str(record.get("instance_key"))
                    semantic_id = record.get("semantic_id")
                    if not isinstance(semantic_id, int):
                        inferred = _infer_semantic_id(
                            str(record.get("instance_key")),
                            semantic_id,
                        )
                        if inferred is None:
                            raise RuntimeError(
                                "Failed to infer semantic_id for goal record: "
                                f"{record.get('instance_key')}"
                            )
                        semantic_id = int(inferred)

                    goal_tasks.append(
                        [
                            task_instance_key,
                            _to_task_modality_token(goal_input, str(modality)),
                        ]
                    )

                start_state = start_sampler(rng) if start_sampler is not None else None

                distance_matrix: Optional[List[List[Optional[float]]]] = None
                ordered_total_geodesic_distance: Optional[float] = None
                unordered_total_geodesic_distance: Optional[float] = None
                if (
                    start_goal_distance_fn is not None
                    and isinstance(start_state, dict)
                    and _is_vec3(start_state.get("position"))
                ):
                    start_position = [float(v) for v in start_state["position"]]
                    n_goals = int(len(sampled_records))
                    distance_matrix = [
                        [None for _ in range(n_goals + 1)] for _ in range(n_goals + 1)
                    ]
                    for idx in range(n_goals + 1):
                        distance_matrix[idx][idx] = 0.0

                    for goal_idx, goal_record in enumerate(sampled_records):
                        distance = start_goal_distance_fn(start_position, goal_record)
                        distance_matrix[0][goal_idx + 1] = distance
                        distance_matrix[goal_idx + 1][0] = distance

                    if goal_pair_distance_fn is not None:
                        for idx in range(n_goals):
                            for jdx in range(idx + 1, n_goals):
                                pair_distance = goal_pair_distance_fn(
                                    sampled_records[idx],
                                    sampled_records[jdx],
                                )
                                distance_matrix[idx + 1][jdx + 1] = pair_distance
                                distance_matrix[jdx + 1][idx + 1] = pair_distance

                    (
                        ordered_total_geodesic_distance,
                        unordered_total_geodesic_distance,
                    ) = _compute_total_geodesic_distances_from_matrix(distance_matrix)

                start_position_payload = [0.0, 0.0, 0.0]
                start_rotation_payload = [0.0, 0.0, 0.0, 1.0]
                if isinstance(start_state, dict):
                    if _is_vec3(start_state.get("position")):
                        start_position_payload = [
                            float(v) for v in start_state.get("position", start_position_payload)
                        ]
                    if isinstance(start_state.get("rotation"), list) and len(start_state.get("rotation")) == 4:
                        start_rotation_payload = [
                            float(v) for v in start_state.get("rotation", start_rotation_payload)
                        ]

                trajectory_payload: Dict[str, Any] = {
                    "episode_id": str(trajectory_index),
                    "scene_id": str(episode_scene_id),
                    "start_position": start_position_payload,
                    "start_rotation": start_rotation_payload,
                    "num_goals": int(goal_count),
                    "goals": goal_tasks,
                    "ordered_total_geodesic_distance": ordered_total_geodesic_distance,
                    "unordered_total_geodesic_distance": unordered_total_geodesic_distance,
                    "object_category": goal_categories[0] if goal_categories else None,
                    "sound_id": goal_sound_ids[0] if goal_sound_ids else None,
                    "offset": str(
                        rng.randint(int(audio_offset_min), int(audio_offset_max))
                    ),
                    "duration": str(int(audio_duration)),
                    "sound_sources": [
                        {"sound_id": sound_id} for sound_id in goal_sound_ids
                    ],
                    "sound_source_schedule": [
                        str(audio_schedule),
                        int(audio_duration),
                    ],
                    "goal_instance_keys": goal_instance_keys,
                }
                distance_constraints_checked = bool(start_goal_distance_fn is not None)
                valid_distances = True
                if distance_constraints_checked:
                    if distance_matrix is None:
                        valid_distances = False
                    else:
                        n_goals = int(len(sampled_records))
                        for goal_idx in range(n_goals):
                            distance = distance_matrix[0][goal_idx + 1]
                            if distance is None or float(distance) < float(min_goal_distance):
                                valid_distances = False
                                break

                        if valid_distances and goal_pair_distance_fn is not None:
                            for idx in range(n_goals):
                                if not valid_distances:
                                    break
                                for jdx in range(idx + 1, n_goals):
                                    distance = distance_matrix[idx + 1][jdx + 1]
                                    if (
                                        distance is None
                                        or float(distance) < float(min_goal_distance)
                                    ):
                                        valid_distances = False
                                        break
                else:
                    valid_distances = False

                trajectory_payload["_distance_constraints_checked"] = bool(
                    distance_constraints_checked
                )
                trajectory_payload["_distance_constraints_satisfied"] = bool(
                    valid_distances
                )

                if valid_distances:
                    best_payload = trajectory_payload
                    break

                if distance_matrix is not None:
                    n_goals = int(len(sampled_records))
                    required_values: List[Optional[float]] = []
                    required_values.extend(
                        distance_matrix[0][goal_idx + 1]
                        for goal_idx in range(n_goals)
                    )
                    for idx in range(n_goals):
                        for jdx in range(idx + 1, n_goals):
                            required_values.append(distance_matrix[idx + 1][jdx + 1])
                    null_count = int(sum(value is None for value in required_values))
                else:
                    null_count = int(goal_count + (goal_count * (goal_count - 1)) // 2)
                if best_null_count is None or null_count < best_null_count:
                    best_payload = trajectory_payload
                    best_null_count = null_count

            if best_payload is None:
                if int(goal_count) > 2:
                    goal_count -= 1
                    continue
                raise RuntimeError("Failed to build trajectory payload")

            if bool(best_payload.get("_distance_constraints_checked")) and not bool(
                best_payload.get("_distance_constraints_satisfied")
            ):
                if int(goal_count) > 2:
                    goal_count -= 1
                    continue
                raise RuntimeError(
                    "Failed to satisfy min-distance constraints for episode "
                    f"{trajectory_index} after {int(distance_max_attempts)} attempts."
                )

            trajectories.append(best_payload)
            for instance_key in best_payload.get("goal_instance_keys", []):
                if isinstance(instance_key, str):
                    instance_usage_counter[instance_key] += 1
            if "goal_instance_keys" in best_payload:
                best_payload.pop("goal_instance_keys", None)
            if "_distance_constraints_checked" in best_payload:
                best_payload.pop("_distance_constraints_checked", None)
            if "_distance_constraints_satisfied" in best_payload:
                best_payload.pop("_distance_constraints_satisfied", None)
            goal_count_hist[int(goal_count)] += 1
            break

    distinct_goal_instances = int(len(instance_usage_counter))
    reused_instance_count = int(
        sum(int(count > 1) for count in instance_usage_counter.values())
    )
    max_instance_reuse = int(max(instance_usage_counter.values(), default=0))

    summary = {
        "num_trajectories": int(len(trajectories)),
        "goal_count_histogram": {
            str(k): int(v) for k, v in sorted(goal_count_hist.items())
        },
        "effective_max_goals": int(effective_max_goals),
        "num_distinct_goal_instances": distinct_goal_instances,
        "num_reused_goal_instances": reused_instance_count,
        "max_instance_reuse_count": max_instance_reuse,
    }
    return trajectories, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a trajectory dataset from valid_instances.json"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input valid_instances.json path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output trajectory dataset JSON (default: <input_dir>/trajectory_dataset.json)",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=20,
        help="Number of trajectories to generate",
    )
    parser.add_argument(
        "--min-goals",
        type=int,
        default=2,
        help="Minimum goals in one trajectory",
    )
    parser.add_argument(
        "--max-goals",
        type=int,
        default=4,
        help="Maximum goals in one trajectory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed",
    )
    parser.add_argument(
        "--require-image",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Only use instances with at least one saved render view (default: no)",
    )
    parser.add_argument(
        "--unique-categories",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Enforce unique goal categories within one trajectory (default: yes)",
    )
    parser.add_argument(
        "--include-image-summary",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Attach compact image info for each goal (default: yes)",
    )
    parser.add_argument(
        "--sim-start",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Sample each trajectory start from simulator navmesh (default: yes)",
    )
    parser.add_argument(
        "--scene-name",
        type=str,
        default=None,
        help="Optional scene name override when resolving GLB",
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=Path("data/scene_datasets/mp3d"),
        help="Root directory containing MP3D scenes",
    )
    parser.add_argument(
        "--scene-dataset-config",
        type=Path,
        default=Path("data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"),
        help="Path to scene_dataset_config JSON",
    )
    parser.add_argument(
        "--exp-config",
        type=Path,
        default=Path("configs/omni-long/mp3d/omni-long_semantic_audio.yaml"),
        help="Habitat task config used to instantiate simulator",
    )
    parser.add_argument(
        "--gpu-device-id",
        type=int,
        default=0,
        help="Habitat simulator GPU device id",
    )
    parser.add_argument("--width", type=int, default=512, help="Sensor width")
    parser.add_argument("--height", type=int, default=512, help="Sensor height")
    parser.add_argument("--hfov", type=float, default=90.0, help="Sensor HFOV")
    parser.add_argument(
        "--sensor-height",
        type=float,
        default=1.25,
        help="Sensor height above agent base",
    )
    parser.add_argument(
        "--start-min-clearance",
        type=float,
        default=0.5,
        help="Minimum obstacle clearance for sampled start positions",
    )
    parser.add_argument(
        "--start-max-attempts",
        type=int,
        default=1000,
        help="Maximum attempts when sampling one start position",
    )
    parser.add_argument(
        "--floor-level-tolerance",
        type=float,
        default=0.35,
        help="Height clustering tolerance when discovering floor levels",
    )
    parser.add_argument(
        "--floor-height-tolerance",
        type=float,
        default=0.25,
        help="Allowed y-diff between sampled start and dominant floors",
    )
    parser.add_argument(
        "--max-floor-levels",
        type=int,
        default=3,
        help="Number of dominant floors retained for start sampling",
    )
    parser.add_argument(
        "--max-snap-distance",
        type=float,
        default=2.0,
        help="Maximum x-z snap distance when projecting goals to navmesh",
    )
    parser.add_argument(
        "--distance-max-attempts",
        type=int,
        default=20000,
        help="Max re-sampling attempts per trajectory to reduce unreachable goal distances",
    )
    parser.add_argument(
        "--min-goal-distance",
        type=float,
        default=4.0,
        help="Minimum geodesic distance (meters) for start-goal and goal-goal pairs",
    )
    parser.add_argument(
        "--audio-duration",
        type=int,
        default=25,
        help="Audio playback duration for round-robin schedule",
    )
    parser.add_argument(
        "--audio-offset-min",
        type=int,
        default=0,
        help="Minimum random audio offset",
    )
    parser.add_argument(
        "--audio-offset-max",
        type=int,
        default=50,
        help="Maximum random audio offset",
    )
    parser.add_argument(
        "--audio-schedule",
        type=str,
        default="round_robin",
        help="Audio source schedule mode",
    )
    parser.add_argument(
        "--goal-image-view-strategy",
        type=str,
        choices=["best_iou", "random"],
        default="best_iou",
        help="How to pick one render view for each goal when multiple views exist",
    )
    parser.add_argument(
        "--balance-instance-coverage",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Prefer rarely-used instances so trajectories cover more unique instances",
    )
    parser.add_argument(
        "--coverage-explore-ratio",
        type=float,
        default=0.15,
        help="Random-explore ratio for coverage-aware sampling (0~1)",
    )
    parser.add_argument(
        "--prefer-image-text-per-episode",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Prefer at least one image-goal and one text-description-goal per episode",
    )
    parser.add_argument(
        "--min-image-goals-per-episode",
        type=int,
        default=1,
        help="Minimum number of image-capable goals per episode",
    )
    parser.add_argument(
        "--max-image-goals-per-episode",
        type=int,
        default=2,
        help="Maximum number of image-capable goals per episode",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if int(args.num_trajectories) <= 0:
        raise RuntimeError("--num-trajectories must be > 0")
    if int(args.min_goals) <= 0 or int(args.max_goals) <= 0:
        raise RuntimeError("--min-goals and --max-goals must be > 0")
    if int(args.min_goals) > int(args.max_goals):
        raise RuntimeError("--min-goals cannot be larger than --max-goals")
    if float(args.coverage_explore_ratio) < 0.0 or float(args.coverage_explore_ratio) > 1.0:
        raise RuntimeError("--coverage-explore-ratio must be within [0, 1]")
    if int(args.min_image_goals_per_episode) < 0 or int(args.max_image_goals_per_episode) < 0:
        raise RuntimeError("--min-image-goals-per-episode and --max-image-goals-per-episode must be >= 0")
    if int(args.min_image_goals_per_episode) > int(args.max_image_goals_per_episode):
        raise RuntimeError("--min-image-goals-per-episode cannot be larger than --max-image-goals-per-episode")

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise RuntimeError(f"Input file not found: {input_path}")

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else input_path.parent / "trajectory_dataset.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_path.read_text())
    scene_name = _resolve_scene_name(payload, args.scene_name)
    scene_id = payload.get("scene_id")
    scene_split = _infer_scene_split(scene_id)
    episode_scene_id = _normalize_episode_scene_id(scene_id)

    records, pool_stats = _flatten_instances(
        payload=payload,
        require_image=bool(args.require_image),
        include_image_summary=bool(args.include_image_summary),
    )
    instance_lookup = _build_instance_lookup(payload)
    if not records:
        raise RuntimeError("No eligible instances found after filtering.")

    start_sampler: Optional[Callable[[random.Random], Dict[str, Any]]] = None
    start_goal_distance_fn: Optional[
        Callable[[List[float], Dict[str, Any]], Optional[float]]
    ] = None
    goal_pair_distance_fn: Optional[
        Callable[[Dict[str, Any], Dict[str, Any]], Optional[float]]
    ] = None
    dominant_floors_payload = payload.get("dominant_floors", [])
    start_sampling_metadata: Dict[str, Any] = {
        "sim_start": bool(args.sim_start),
    }
    simulator = None
    try:
        if args.sim_start:
            multimodal_module = _load_multimodal_module()
            scene_dir = args.scene_dir.expanduser().resolve()
            scene_dataset_config = (
                args.scene_dataset_config.expanduser().resolve()
                if args.scene_dataset_config is not None
                else None
            )
            exp_config = args.exp_config.expanduser().resolve()
            scene_path = multimodal_module._resolve_scene_path(scene_dir, scene_name)
            scene_id = multimodal_module._scene_id_for_dataset(scene_path, scene_dir)
            episode_scene_id = _normalize_episode_scene_id(scene_id)

            simulator = _build_pathfinder_simulator(
                scene_path=scene_path,
                scene_dataset_config=scene_dataset_config,
                exp_config=exp_config,
                width=int(args.width),
                height=int(args.height),
                hfov=float(args.hfov),
                sensor_height=float(args.sensor_height),
                gpu_device_id=int(args.gpu_device_id),
            )

            navmesh_vertices = multimodal_module._collect_navmesh_triangle_vertices(
                simulator.pathfinder
            )
            all_floor_levels = multimodal_module._discover_floor_levels(
                vertices=navmesh_vertices,
                floor_level_tolerance=float(args.floor_level_tolerance),
            )
            dominant_floors = multimodal_module._select_dominant_floor_levels(
                all_floor_levels,
                max_floor_levels=int(args.max_floor_levels),
            )
            if not dominant_floors:
                raise RuntimeError(
                    "No dominant floor levels discovered from simulator navmesh."
                )

            dominant_floors_payload = _serialize_floor_levels(dominant_floors)
            start_sampling_metadata.update(
                {
                    "scene_path": str(scene_path),
                    "num_navmesh_height_bands": int(len(all_floor_levels)),
                    "num_dominant_floors": int(len(dominant_floors)),
                    "start_min_clearance": float(args.start_min_clearance),
                    "start_max_attempts": int(args.start_max_attempts),
                    "gpu_device_id": int(args.gpu_device_id),
                    "floor_level_tolerance": float(args.floor_level_tolerance),
                    "floor_height_tolerance": float(args.floor_height_tolerance),
                    "max_floor_levels": int(args.max_floor_levels),
                    "max_snap_distance": float(args.max_snap_distance),
                    "distance_max_attempts": int(args.distance_max_attempts),
                    "min_goal_distance": float(args.min_goal_distance),
                    "audio_duration": int(args.audio_duration),
                    "audio_offset_min": int(args.audio_offset_min),
                    "audio_offset_max": int(args.audio_offset_max),
                    "audio_schedule": str(args.audio_schedule),
                }
            )

            def _sample_start(rng: random.Random) -> Dict[str, Any]:
                return _build_start_state(
                    multimodal_module=multimodal_module,
                    sim=simulator,
                    rng=rng,
                    floor_levels=dominant_floors,
                    min_clearance=float(args.start_min_clearance),
                    max_attempts=int(args.start_max_attempts),
                    floor_height_tolerance=float(args.floor_height_tolerance),
                    width=int(args.width),
                    height=int(args.height),
                    hfov=float(args.hfov),
                    sensor_height=float(args.sensor_height),
                )

            start_sampler = _sample_start

            def _start_to_goal_ground_distance(
                start_position: List[float],
                goal_record: Dict[str, Any],
            ) -> Optional[float]:
                if not _is_vec3(start_position):
                    return None
                start_snapped = _snap_navmesh_point(
                    simulator.pathfinder,
                    [float(v) for v in start_position],
                    max_snap_distance=float(args.max_snap_distance),
                )
                if start_snapped is None:
                    return None

                snapped_goal = _resolve_goal_navmesh_point(
                    simulator.pathfinder,
                    goal_record,
                    dominant_floors,
                    max_snap_distance=float(args.max_snap_distance),
                )
                if snapped_goal is None:
                    return None
                distance = float(
                    multimodal_module._geodesic_distance(
                        simulator.pathfinder,
                        np.array(start_snapped, dtype=np.float32),
                        np.array(snapped_goal, dtype=np.float32),
                    )
                )
                if not math.isfinite(distance):
                    return None
                return round(distance, 3)

            start_goal_distance_fn = _start_to_goal_ground_distance

            def _goal_pair_ground_distance(
                goal_a: Dict[str, Any],
                goal_b: Dict[str, Any],
            ) -> Optional[float]:
                snapped_a = _resolve_goal_navmesh_point(
                    simulator.pathfinder,
                    goal_a,
                    dominant_floors,
                    max_snap_distance=float(args.max_snap_distance),
                )
                snapped_b = _resolve_goal_navmesh_point(
                    simulator.pathfinder,
                    goal_b,
                    dominant_floors,
                    max_snap_distance=float(args.max_snap_distance),
                )
                if snapped_a is None or snapped_b is None:
                    return None
                distance = float(
                    multimodal_module._geodesic_distance(
                        simulator.pathfinder,
                        np.array(snapped_a, dtype=np.float32),
                        np.array(snapped_b, dtype=np.float32),
                    )
                )
                if not math.isfinite(distance):
                    return None
                return round(distance, 3)

            goal_pair_distance_fn = _goal_pair_ground_distance

        episodes, sampling_summary = _build_trajectories(
            records=records,
            instance_lookup=instance_lookup,
            num_trajectories=int(args.num_trajectories),
            min_goals=int(args.min_goals),
            max_goals=int(args.max_goals),
            seed=int(args.seed),
            unique_categories=bool(args.unique_categories),
            scene_name=scene_name,
            start_sampler=start_sampler,
            start_goal_distance_fn=start_goal_distance_fn,
            goal_pair_distance_fn=goal_pair_distance_fn,
            distance_max_attempts=int(args.distance_max_attempts),
            min_goal_distance=float(args.min_goal_distance),
            scene_split=str(scene_split),
            audio_duration=int(args.audio_duration),
            audio_offset_min=int(args.audio_offset_min),
            audio_offset_max=int(args.audio_offset_max),
            audio_schedule=str(args.audio_schedule),
            goal_image_view_strategy=str(args.goal_image_view_strategy),
            balance_instance_coverage=bool(args.balance_instance_coverage),
            coverage_explore_ratio=float(args.coverage_explore_ratio),
            prefer_image_text_per_episode=bool(args.prefer_image_text_per_episode),
            min_image_goals_per_episode=int(args.min_image_goals_per_episode),
            max_image_goals_per_episode=int(args.max_image_goals_per_episode),
            episode_scene_id=str(episode_scene_id),
        )
    finally:
        if simulator is not None and hasattr(simulator, "close"):
            simulator.close()

    output_payload: Dict[str, Any] = {
        "dataset": "valid_instance_trajectory_dataset",
        "version": "0.1",
        "scene_name": scene_name,
        "scene_id": episode_scene_id,
        "source_valid_instances": str(input_path),
        "sampling": {
            "num_trajectories": int(args.num_trajectories),
            "min_goals": int(args.min_goals),
            "max_goals": int(args.max_goals),
            "seed": int(args.seed),
            "require_image": bool(args.require_image),
            "unique_categories": bool(args.unique_categories),
            "include_image_summary": bool(args.include_image_summary),
            "goal_image_view_strategy": str(args.goal_image_view_strategy),
            "balance_instance_coverage": bool(args.balance_instance_coverage),
            "coverage_explore_ratio": float(args.coverage_explore_ratio),
            "prefer_image_text_per_episode": bool(args.prefer_image_text_per_episode),
            "min_image_goals_per_episode": int(args.min_image_goals_per_episode),
            "max_image_goals_per_episode": int(args.max_image_goals_per_episode),
            "goal_modality_policy": {
                "audio": "always",
                "object": "always",
                "image": "use_when_render_views_exist",
                "text_description": "bind_to_selected_image_view_when_available",
            },
            "start_sampling": start_sampling_metadata,
        },
        "episodes": episodes,
        "instances": instance_lookup,
    }

    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, indent=2, ensure_ascii=False)

    print(f"Input valid instances: {input_path}")
    print(f"Eligible instances: {pool_stats['num_instances_eligible']}")
    print(f"Generated episodes: {len(episodes)}")
    print(f"Saved trajectory dataset to: {output_path}")


if __name__ == "__main__":
    main()
