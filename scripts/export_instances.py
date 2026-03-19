#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import quaternion
from habitat.utils.geometry_utils import quaternion_rotate_vector
from PIL import Image
from tqdm import tqdm


def _load_module(filename: str, module_name: str) -> Any:
    module_path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_multimodal_module() -> Any:
    return _load_module("generate_multimodal_starts.py", "generate_multimodal_starts")


def _load_imagenav_module() -> Any:
    return _load_module(
        "generate_imagenav_eval_dataset.py",
        "generate_imagenav_eval_dataset",
    )


def _load_description_module() -> Any:
    return _load_module(
        "generate_instance_descriptions_qwen.py",
        "generate_instance_descriptions_qwen",
    )


def _round_list(values: np.ndarray, digits: int = 3) -> List[float]:
    return [round(float(value), digits) for value in values.tolist()]


def _grid_values(start: float, end: float, spacing: float) -> np.ndarray:
    num_steps = max(1, int(math.ceil((float(end) - float(start)) / float(spacing))))
    return np.linspace(
        float(start),
        float(start) + float(num_steps) * float(spacing),
        num_steps + 1,
        dtype=np.float32,
    )


def _quat_to_list(quat: Any) -> List[float]:
    if all(hasattr(quat, attr) for attr in ("x", "y", "z", "w")):
        return [float(quat.x), float(quat.y), float(quat.z), float(quat.w)]
    if hasattr(quat, "vector") and hasattr(quat, "scalar"):
        vector = quat.vector
        return [
            float(vector[0]),
            float(vector[1]),
            float(vector[2]),
            float(quat.scalar),
        ]
    if isinstance(quat, (list, tuple, np.ndarray)) and len(quat) == 4:
        return [float(value) for value in quat]
    return [0.0, 0.0, 0.0, 1.0]


def _quat_from_list(values: Any) -> Any:
    if not isinstance(values, (list, tuple, np.ndarray)) or len(values) != 4:
        return quaternion.quaternion(1.0, 0.0, 0.0, 0.0)
    quat = quaternion.quaternion(
        float(values[3]),
        float(values[0]),
        float(values[1]),
        float(values[2]),
    )
    quat_norm = float(np.linalg.norm(quaternion.as_float_array(quat)))
    if quat_norm <= 1e-8:
        return quaternion.quaternion(1.0, 0.0, 0.0, 0.0)
    return quat / quat_norm


def _rotation_matrix_to_quaternion(rotation_matrix: np.ndarray) -> Any:
    matrix = np.array(rotation_matrix, dtype=np.float64)
    trace = float(np.trace(matrix))

    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * scale
        qx = (matrix[2, 1] - matrix[1, 2]) / scale
        qy = (matrix[0, 2] - matrix[2, 0]) / scale
        qz = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        qw = (matrix[2, 1] - matrix[1, 2]) / scale
        qx = 0.25 * scale
        qy = (matrix[0, 1] + matrix[1, 0]) / scale
        qz = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        qw = (matrix[0, 2] - matrix[2, 0]) / scale
        qx = (matrix[0, 1] + matrix[1, 0]) / scale
        qy = 0.25 * scale
        qz = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        qw = (matrix[1, 0] - matrix[0, 1]) / scale
        qx = (matrix[0, 2] + matrix[2, 0]) / scale
        qy = (matrix[1, 2] + matrix[2, 1]) / scale
        qz = 0.25 * scale

    quat = quaternion.quaternion(qw, qx, qy, qz)
    quat_norm = float(np.linalg.norm(quaternion.as_float_array(quat)))
    if quat_norm <= 1e-8:
        return quaternion.quaternion(1.0, 0.0, 0.0, 0.0)
    return quat / quat_norm


def _sensor_world_position(base_position: np.ndarray, sensor_height: float) -> np.ndarray:
    sensor_position = np.asarray(base_position, dtype=np.float32).copy()
    sensor_position[1] += float(sensor_height)
    return sensor_position


def _look_at_quaternion(
    base_position: np.ndarray,
    target: np.ndarray,
    sensor_height: float,
    horizontal_only: bool,
) -> Any:
    sensor_origin = _sensor_world_position(base_position, sensor_height)
    forward = np.asarray(target, dtype=np.float32) - sensor_origin
    if horizontal_only:
        forward[1] = 0.0
        if float(np.linalg.norm(forward)) <= 1e-6:
            forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= 1e-6:
        return quaternion.quaternion(1.0, 0.0, 0.0, 0.0)
    forward = forward / forward_norm

    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(forward, world_up))) >= 0.999:
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    right = np.cross(forward, world_up)
    right_norm = float(np.linalg.norm(right))
    if right_norm <= 1e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    else:
        right = right / right_norm

    up = np.cross(right, forward)
    up_norm = float(np.linalg.norm(up))
    if up_norm <= 1e-6:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    else:
        up = up / up_norm

    rotation_matrix = np.stack([right, up, -forward], axis=1).astype(np.float32)
    return _rotation_matrix_to_quaternion(rotation_matrix)


def _point_inside_aabb(
    point: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> bool:
    point_arr = np.asarray(point, dtype=np.float32)
    return bool(np.all(point_arr >= bbox_min) and np.all(point_arr <= bbox_max))


def _point_to_aabb_distance(
    point: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
) -> float:
    point_arr = np.asarray(point, dtype=np.float32)
    delta = np.maximum(np.maximum(bbox_min - point_arr, 0.0), point_arr - bbox_max)
    return float(np.linalg.norm(delta))


def _observe_semantic(
    sim: Any,
    inner_sim: Any,
    semantic_uuid: str,
    base_position: np.ndarray,
    rotation: Any,
) -> np.ndarray:
    sim.set_agent_state(base_position, rotation, reset_sensors=False)
    observations = inner_sim.get_sensor_observations()
    semantic = observations.get(semantic_uuid)
    if semantic is None:
        raise RuntimeError("Missing semantic observation.")
    semantic = np.asarray(semantic)
    if semantic.ndim == 3:
        semantic = semantic[:, :, 0]
    return semantic


def _semantic_instance_mask(semantic: np.ndarray, semantic_id: int) -> np.ndarray:
    return semantic.astype(np.int64, copy=False) == int(semantic_id)


def _view_point_visible(semantic: np.ndarray, semantic_id: int) -> bool:
    semantic_mask = _semantic_instance_mask(semantic, int(semantic_id))
    return bool(np.count_nonzero(semantic_mask) > 0)


def _supporting_floor(
    floors: List[Any],
    center_y: float,
) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    if not floors:
        return None, None, None
    ordered = sorted(floors, key=lambda level: float(level.y))
    selected = ordered[0]
    for level in ordered:
        if float(center_y) >= float(level.y) - 1e-6:
            selected = level
        else:
            break
    delta = float(center_y) - float(selected.y)
    return int(selected.index), float(selected.y), float(delta)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export scene instances, viewpoints, and YOLO-filtered render views."
    )
    parser.add_argument("scene_name", type=str, help="Scene id or .glb name")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (defaults to output_val/<scene>/instances.json)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output_val"),
        help="Root output directory for exported instances JSON",
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
        help="Path to MP3D scene_dataset_config JSON",
    )
    parser.add_argument(
        "--exp-config",
        type=Path,
        default=Path("configs/omni-long/mp3d/omni-long_semantic_audio.yaml"),
        help="Habitat task config used to instantiate simulator",
    )
    parser.add_argument("--width", type=int, default=512, help="Sensor width")
    parser.add_argument("--height", type=int, default=512, help="Sensor height")
    parser.add_argument("--hfov", type=float, default=90.0, help="Sensor HFOV")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--save-images",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Save raw RGB and YOLO-annotated render images (default: yes)",
    )
    parser.add_argument(
        "--yolo-model",
        type=Path,
        default=Path("models/yolo26x.pt"),
        help="YOLO weights for render-time filtering",
    )
    parser.add_argument(
        "--yolo-device",
        type=str,
        default=None,
        help="YOLO inference device",
    )
    parser.add_argument(
        "--yolo-conf-threshold",
        type=float,
        default=0.7,
        help="Minimum matched YOLO confidence for saved render views",
    )
    parser.add_argument(
        "--yolo-iou-threshold",
        type=float,
        default=0.45,
        help="YOLO NMS IoU threshold",
    )
    parser.add_argument(
        "--yolo-max-det",
        type=int,
        default=50,
        help="Maximum YOLO detections per rendered image",
    )
    parser.add_argument(
        "--yolo-aliases-json",
        type=Path,
        default=None,
        help="Optional JSON mapping for YOLO label aliases",
    )
    parser.add_argument(
        "--horizontal-only-rotation",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Constrain viewpoint rotation to yaw-only (default: yes)",
    )
    parser.add_argument(
        "--sensor-height",
        type=float,
        default=1.25,
        help="Sensor height above agent base",
    )
    parser.add_argument(
        "--floor-level-tolerance",
        type=float,
        default=0.35,
        help="Height clustering tolerance when identifying dominant floor levels",
    )
    parser.add_argument(
        "--floor-height-tolerance",
        type=float,
        default=1.0,
        help="Allowed height difference between projected nav points and dominant floors",
    )
    parser.add_argument(
        "--min-viewpoint-distance",
        type=float,
        default=0.1,
        help="Minimum sensor-to-instance surface distance for exported view_points",
    )
    parser.add_argument(
        "--max-viewpoint-distance",
        type=float,
        default=0.6,
        help="Maximum sensor-to-instance surface distance for exported view_points",
    )
    parser.add_argument(
        "--min-render-viewpoint-distance",
        type=float,
        default=0.5,
        help="Minimum sensor-to-instance surface distance for render_view_points",
    )
    parser.add_argument(
        "--max-render-viewpoint-distance",
        type=float,
        default=1.25,
        help="Maximum sensor-to-instance surface distance for render_view_points",
    )
    parser.add_argument(
        "--max-render-viewpoints",
        type=int,
        default=3,
        help="Maximum number of YOLO-passed render viewpoints saved per instance",
    )
    parser.add_argument(
        "--generate-descriptions",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Generate English descriptions for instances using rendered views",
    )
    parser.add_argument(
        "--description-api-key",
        type=str,
        default='sk-QgWdM03NkfNrFfMA576126F43fAa4b0eBb635d80C6D2Cc91',
        help="VLM API key. If omitted, uses DASHSCOPE_API_KEY/QWEN_API_KEY/OPENAI_API_KEY.",
    )
    parser.add_argument(
        "--description-api-base",
        type=str,
        default="https://api.vveai.com/v1",
        help="OpenAI-compatible API base URL for description model.",
    )
    parser.add_argument(
        "--description-model",
        type=str,
        default="qwen3-vl-plus",
        help="VLM model name for description generation.",
    )
    parser.add_argument(
        "--description-max-images",
        type=int,
        default=3,
        help="Maximum number of views per instance sent to VLM.",
    )
    parser.add_argument(
        "--description-retries",
        type=int,
        default=2,
        help="Retry count when VLM output fails constraints.",
    )
    parser.add_argument(
        "--description-timeout",
        type=int,
        default=120,
        help="HTTP timeout (seconds) for VLM requests.",
    )
    parser.add_argument(
        "--description-sleep",
        type=float,
        default=0.2,
        help="Sleep seconds between VLM calls.",
    )
    parser.add_argument(
        "--description-overwrite",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Overwrite existing description if present.",
    )
    parser.add_argument(
        "--description-max-instances",
        type=int,
        default=None,
        help="Optional cap for number of instances to generate descriptions for.",
    )
    parser.add_argument(
        "--max-floor-levels",
        type=int,
        default=3,
        help="Maximum number of dominant floor bands used for viewpoint search",
    )
    return parser.parse_args()


def _build_imagenav_args(args: argparse.Namespace, output_root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        scene_name=args.scene_name,
        target_category=None,
        yolo_model=args.yolo_model,
        yolo_device=args.yolo_device,
        yolo_conf_threshold=float(args.yolo_conf_threshold),
        yolo_iou_threshold=float(args.yolo_iou_threshold),
        yolo_max_det=int(args.yolo_max_det),
        yolo_aliases_json=args.yolo_aliases_json,
        scene_dir=args.scene_dir,
        scene_dataset_config=args.scene_dataset_config,
        exp_config=args.exp_config,
        output_root=output_root,
        output_json=None,
        width=args.width,
        height=args.height,
        hfov=args.hfov,
        seed=int(args.seed),
        viewpoint_sampling_mode="uniform_floor_grid",
        uniform_grid_step=0.2,
        horizontal_only_rotation=args.horizontal_only_rotation,
        sensor_height=args.sensor_height,
        search_radius=1.0,
        min_surface_offset=0.25,
        candidate_angle_step=30,
        radial_step=0.1,
        radial_step_jitter=0.05,
        angle_jitter=10.0,
        min_edge_clearance=0.05,
        min_iou=0.2,
        min_target_detection_coverage=0.1,
        max_floor_levels=args.max_floor_levels,
        floor_level_tolerance=args.floor_level_tolerance,
        nearby_object_radius=3.0,
        max_viewpoints=24,
        max_saved_views=3,
        min_final_view_angle_sep=35,
        min_sensor_offset=1.25,
        max_sensor_offset=1.5,
        sensor_offset_step=0.125,
        floor_height_tolerance=0.25,
        min_viewpoint_angle_sep=30,
        min_viewpoint_separation=0.45,
        max_snap_distance=0.6,
        object_clearance=0.1,
        position_adjust_max=0.25,
        position_adjust_step=0.05,
        render_validation_retry_radius=0.25,
        render_validation_retry_max_attempts=10,
        save_images=bool(args.save_images),
    )


def _view_point_angle_deg(center: np.ndarray, position: np.ndarray) -> float:
    delta = np.asarray(position, dtype=np.float32) - np.asarray(center, dtype=np.float32)
    return float((math.degrees(math.atan2(float(delta[2]), float(delta[0]))) + 360.0) % 360.0)


def _view_point_sort_key(view_point: dict) -> tuple:
    position = view_point["agent_state"]["position"]
    return (
        -int(view_point.get("semantic_pixels", 0)),
        float(view_point.get("sensor_surface_distance", 1e9)),
        float(view_point.get("_angle_deg", 0.0)),
        float(position[0]),
        float(position[2]),
    )


def _select_uniform_view_points(
    raw_view_points: List[dict],
    min_count: int = 4,
    max_count: int = 12,
) -> List[dict]:
    if not raw_view_points:
        return []

    target_count = min(int(max_count), len(raw_view_points))
    if target_count <= 0:
        return []

    bins = [[] for _ in range(target_count)]
    for view_point in raw_view_points:
        angle_deg = float(view_point.get("_angle_deg", 0.0)) % 360.0
        bin_index = min(target_count - 1, int((angle_deg / 360.0) * target_count))
        bins[bin_index].append(view_point)

    for bucket in bins:
        bucket.sort(key=_view_point_sort_key)

    selected: List[dict] = []
    for bucket in bins:
        if bucket and len(selected) < target_count:
            selected.append(bucket.pop(0))

    while len(selected) < target_count:
        progressed = False
        for bucket in bins:
            if bucket and len(selected) < target_count:
                selected.append(bucket.pop(0))
                progressed = True
        if not progressed:
            break

    selected.sort(
        key=lambda item: (
            float(item.get("_angle_deg", 0.0)),
            _view_point_sort_key(item),
        )
    )
    trimmed = []
    for view_point in selected:
        output_view = dict(view_point)
        output_view.pop("_angle_deg", None)
        trimmed.append(output_view)

    if len(trimmed) >= int(min_count):
        return trimmed
    return trimmed


def _collect_candidate_view_points(
    scanner: Any,
    obj: Any,
    floor_y: Optional[float],
    args: argparse.Namespace,
    *,
    min_distance: Optional[float] = None,
    max_distance: Optional[float] = None,
) -> List[dict]:
    pathfinder = scanner.pathfinder
    sim = scanner.sim
    inner_sim = getattr(scanner, "inner_sim", getattr(sim, "_sim", sim))
    semantic_uuid = getattr(scanner, "semantic_uuid", None) or "semantic"

    sensor_height = float(args.sensor_height)
    sample_y = float(floor_y if floor_y is not None else obj.aabb_min[1])
    min_distance = float(
        args.min_viewpoint_distance if min_distance is None else min_distance
    )
    max_distance = float(
        args.max_viewpoint_distance if max_distance is None else max_distance
    )

    raw_view_points: List[dict] = []
    seen = set()
    for x_value in _grid_values(float(obj.aabb_min[0]) - 1.0, float(obj.aabb_max[0]) + 1.0, 0.1):
        for z_value in _grid_values(float(obj.aabb_min[2]) - 1.0, float(obj.aabb_max[2]) + 1.0, 0.1):
            point = np.array([float(x_value), sample_y, float(z_value)], dtype=np.float32)
            if not pathfinder.is_navigable(point):
                snapped = np.asarray(pathfinder.snap_point(point), dtype=np.float32)
                if snapped.shape != (3,) or not np.all(np.isfinite(snapped)):
                    continue
                if not pathfinder.is_navigable(snapped):
                    continue
                if float(np.linalg.norm((snapped - point)[[0, 2]])) > 0.1:
                    continue
                if abs(float(snapped[1]) - sample_y) > min(float(args.floor_height_tolerance), 0.25):
                    continue
                point = snapped

            dedup_key = tuple(int(round(float(value) / 0.05)) for value in point.tolist())
            if dedup_key in seen:
                continue
            seen.add(dedup_key)

            sensor_position = _sensor_world_position(point, sensor_height)
            if _point_inside_aabb(sensor_position, obj.aabb_min, obj.aabb_max):
                continue

            sensor_surface_distance = _point_to_aabb_distance(
                sensor_position,
                obj.aabb_min,
                obj.aabb_max,
            )
            if (
                sensor_surface_distance < min_distance
                or sensor_surface_distance > max_distance
            ):
                continue

            rotation = _look_at_quaternion(
                point,
                obj.center,
                sensor_height,
                bool(getattr(args, "horizontal_only_rotation", True)),
            )

            rounded_position = [round(float(value), 3) for value in point.tolist()]
            rounded_rotation = [
                round(float(value), 6) for value in _quat_to_list(rotation)
            ]
            stored_point = np.array(rounded_position, dtype=np.float32)
            stored_rotation = _quat_from_list(rounded_rotation)
            try:
                semantic = _observe_semantic(
                    sim,
                    inner_sim,
                    semantic_uuid,
                    stored_point,
                    stored_rotation,
                )
            except Exception:
                continue

            if not _view_point_visible(semantic, int(obj.semantic_id)):
                continue
            semantic_pixels = int(
                np.count_nonzero(_semantic_instance_mask(semantic, int(obj.semantic_id)))
            )

            raw_view_points.append(
                {
                    "agent_state": {
                        "position": rounded_position,
                        "rotation": rounded_rotation,
                    },
                    "semantic_pixels": semantic_pixels,
                    "sensor_surface_distance": round(float(sensor_surface_distance), 3),
                    "_angle_deg": round(_view_point_angle_deg(obj.center, stored_point), 6),
                }
            )

    return raw_view_points


def _collect_view_points(
    scanner: Any,
    obj: Any,
    floor_y: Optional[float],
    args: argparse.Namespace,
) -> List[dict]:
    raw_view_points = _collect_candidate_view_points(scanner, obj, floor_y, args)

    return _select_uniform_view_points(raw_view_points, min_count=4, max_count=12)


def _collect_render_view_candidates(
    scanner: Any,
    obj: Any,
    floor_y: Optional[float],
    args: argparse.Namespace,
) -> List[dict]:
    raw_view_points = _collect_candidate_view_points(
        scanner,
        obj,
        floor_y,
        args,
        min_distance=float(args.min_render_viewpoint_distance),
        max_distance=float(args.max_render_viewpoint_distance),
    )
    return _select_uniform_view_points(raw_view_points, min_count=0, max_count=12)


def _bbox_overlaps_middle_half(
    bbox: Optional[Tuple[int, int, int, int]],
    image_width: int,
) -> bool:
    if bbox is None:
        return False
    center_min_x = 0.25 * float(image_width)
    center_max_x = 0.75 * float(image_width)
    bbox_min_x, _, bbox_max_x, _ = bbox
    return float(bbox_max_x) >= center_min_x and float(bbox_min_x) <= center_max_x


def _render_views_from_candidates(
    scanner: Any,
    obj: Any,
    candidates: List[dict],
    output_root: Path,
    scene_name: str,
    max_saved_views: int,
) -> Tuple[List[Dict[str, Any]], Optional[str], Optional[str]]:
    render_views: List[Dict[str, Any]] = []
    object_output_dir = (
        output_root
        / scene_name
        / scanner._category_dirname(obj.category)
        / str(int(obj.semantic_id))
    )

    for view_index, candidate in enumerate(candidates):
        agent_state = candidate.get("agent_state", {})
        position_values = agent_state.get("position")
        rotation_values = agent_state.get("rotation")
        if not isinstance(position_values, list) or len(position_values) != 3:
            continue
        if not isinstance(rotation_values, list) or len(rotation_values) != 4:
            continue

        base_position = np.array(position_values, dtype=np.float32)
        rotation = _quat_from_list(rotation_values)
        sensor_position = _sensor_world_position(
            base_position,
            float(scanner.args.sensor_height),
        )

        try:
            rgb, semantic = scanner._observe(
                base_position,
                rotation,
                target_id=int(obj.semantic_id),
            )
        except Exception:
            continue

        yolo_valid, yolo_matches, yolo_num_detections, yolo_annotated = (
            scanner._validate_with_yolo(rgb, obj.category)
        )
        if not yolo_valid:
            continue

        entity_mask = scanner._entity_mask_from_aabb(obj, sensor_position, rotation)
        semantic_mask = scanner._semantic_instance_mask(semantic, int(obj.semantic_id))
        bbox = scanner._mask_bbox(semantic_mask)
        if not _bbox_overlaps_middle_half(bbox, int(scanner.args.width)):
            continue
        visible_ratio = scanner._mask_visible_ratio(entity_mask)
        mask_iou = scanner._mask_iou(entity_mask, semantic_mask)
        floor_level = scanner._nearest_floor_level(float(base_position[1]))

        goal_name = scanner._goal_filename(
            semantic_id=int(obj.semantic_id),
            sensor_position=sensor_position,
            rotation=rotation,
            object_center=obj.center,
            surface_distance=float(candidate.get("sensor_surface_distance", 0.0)),
            frame_cov=visible_ratio,
            iou=mask_iou,
        )
        yolo_name = f"yolo_{goal_name}"

        if bool(getattr(scanner.args, "save_images", True)):
            object_output_dir.mkdir(parents=True, exist_ok=True)
            goal_path = object_output_dir / goal_name
            yolo_path = object_output_dir / yolo_name
            Image.fromarray(rgb).save(goal_path)
            if yolo_annotated is not None:
                yolo_annotated.save(yolo_path)

        render_views.append(
            {
                "agent_state": {
                    "position": [round(float(value), 3) for value in base_position.tolist()],
                    "rotation": [round(float(value), 6) for value in _quat_to_list(rotation)],
                },
                "position": [round(float(value), 3) for value in sensor_position.tolist()],
                "floor_index": (
                    int(floor_level.index) if floor_level is not None else None
                ),
                "floor_y": (
                    round(float(floor_level.y), 3) if floor_level is not None else None
                ),
                "semantic_pixels": int(candidate.get("semantic_pixels", 0)),
                "sensor_surface_distance": round(
                    float(candidate.get("sensor_surface_distance", 0.0)),
                    3,
                ),
                "angle_deg": round(
                    float(
                        candidate.get(
                            "_angle_deg",
                            _view_point_angle_deg(obj.center, base_position),
                        )
                    ),
                    3,
                ),
                "bbox": list(bbox) if bbox is not None else None,
                "frame_cov": round(100.0 * float(visible_ratio), 1),
                "iou": round(100.0 * float(mask_iou), 1),
                "yolo_num_detections": int(yolo_num_detections),
                "yolo_matched_detections": yolo_matches,
            }
        )
        if len(render_views) >= int(max_saved_views):
            break

    if not render_views:
        return render_views, None, None

    object_output_dir.mkdir(parents=True, exist_ok=True)
    render_views_json = object_output_dir / "render_views.json"
    render_views_json.write_text(
        json.dumps(
            {
                "object_id": int(obj.semantic_id),
                "object_category": obj.category,
                "object_center": _round_list(obj.center),
                "num_render_views": len(render_views),
                "render_views": render_views,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return (
        render_views,
        object_output_dir.relative_to(output_root).as_posix(),
        render_views_json.relative_to(output_root).as_posix(),
    )


def _resolve_render_image_candidates(
    instance_payload: Dict[str, Any],
    output_root: Path,
    max_images: int,
    description_module: Any,
) -> List[Tuple[Path, str, float]]:
    render_output_dir = instance_payload.get("render_output_dir")
    if not isinstance(render_output_dir, str) or not render_output_dir.strip():
        return []

    render_dir = (output_root / render_output_dir).resolve()
    if not render_dir.is_dir():
        return []

    candidates: List[Tuple[Path, str, float]] = []
    for image_path in sorted(render_dir.glob("*.png")):
        if image_path.name.startswith("yolo_"):
            continue
        if not image_path.is_file():
            continue
        rel = image_path.relative_to(output_root).as_posix()
        candidates.append((image_path, rel, 0.0))

    if max_images > 0:
        candidates = candidates[:max_images]
    return candidates


def _generate_descriptions_for_instances(
    instances: Dict[str, Dict[str, Dict[str, Any]]],
    *,
    description_module: Any,
    output_root: Path,
    api_base: str,
    api_key: str,
    model: str,
    max_images: int,
    retries: int,
    timeout: int,
    sleep_seconds: float,
    overwrite: bool,
    max_instances: Optional[int],
) -> Dict[str, Any]:
    flat_instances: List[Tuple[str, str, Dict[str, Any]]] = []
    for category, entries in instances.items():
        if not isinstance(entries, dict):
            continue
        for instance_key, instance_payload in entries.items():
            if isinstance(instance_payload, dict):
                flat_instances.append((category, str(instance_key), instance_payload))

    flat_instances.sort(key=lambda item: (item[0], item[1]))
    if isinstance(max_instances, int) and max_instances > 0:
        flat_instances = flat_instances[:max_instances]

    total = len(flat_instances)
    stats: Dict[str, Any] = {
        "enabled": True,
        "model": model,
        "api_base": api_base,
        "processed": total,
        "updated": 0,
        "skipped_no_image": 0,
        "skipped_existing": 0,
        "failed": 0,
    }
    if total == 0:
        return stats

    print(f"Generating descriptions with {model} for {total} instances...")
    progress_bar = tqdm(
        flat_instances,
        total=total,
        desc="Descriptions",
        unit="instance",
        dynamic_ncols=True,
    )
    for idx, (_, instance_key, instance_payload) in enumerate(progress_bar, start=1):
        instance_payload.pop("description_meta", None)

        render_views = instance_payload.get("render_view_points")
        if not isinstance(render_views, list) or len(render_views) == 0:
            stats["skipped_no_image"] += 1
            continue

        existing_description = instance_payload.get("description")
        if (
            not overwrite
            and isinstance(existing_description, str)
            and existing_description.strip()
        ):
            stats["skipped_existing"] += 1
            continue

        candidates = _resolve_render_image_candidates(
            instance_payload,
            output_root,
            max(1, int(max_images)),
            description_module,
        )
        if len(candidates) == 0:
            stats["failed"] += 1
            tqdm.write(f"[{idx}/{total}] {instance_key}: missing image files")
            continue

        category_name = str(instance_payload.get("category", "object")).strip() or "object"
        image_paths = [item[0] for item in candidates]
        rel_paths = [item[1] for item in candidates]
        try:
            description, status = description_module._generate_description(
                api_base=api_base,
                api_key=api_key,
                model=model,
                category=category_name,
                image_paths=image_paths,
                timeout=max(1, int(timeout)),
                retries=max(1, int(retries)),
            )
        except Exception as exc:
            stats["failed"] += 1
            tqdm.write(f"[{idx}/{total}] {instance_key}: api_error {exc}")
            if sleep_seconds > 0:
                time.sleep(float(sleep_seconds))
            continue

        if description is None:
            stats["failed"] += 1
            tqdm.write(f"[{idx}/{total}] {instance_key}: invalid ({status})")
            if sleep_seconds > 0:
                time.sleep(float(sleep_seconds))
            continue

        instance_payload["description"] = description
        modalities = instance_payload.get("modalities")
        if isinstance(modalities, list):
            if "description" not in modalities:
                modalities.append("description")
        else:
            instance_payload["modalities"] = ["description"]
        stats["updated"] += 1
        tqdm.write(f"[{idx}/{total}] {instance_key}: {description}")
        if sleep_seconds > 0:
            time.sleep(float(sleep_seconds))

    progress_bar.close()
    return stats


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    multimodal = _load_multimodal_module()
    imagenav = _load_imagenav_module()

    if args.generate_descriptions and not args.save_images:
        print("Description generation enabled; forcing --save-images.")
        args.save_images = True

    scene_dir = args.scene_dir.expanduser().resolve()
    scene_path = multimodal._resolve_scene_path(scene_dir, args.scene_name)
    scene_id = multimodal._scene_id_for_dataset(scene_path, scene_dir)

    output_root = args.output_root.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else output_root / args.scene_name / "instances.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scanner_args = _build_imagenav_args(args, output_root)

    scanner = None
    scene_objects = []
    dominant_floors = []
    instances: Dict[str, Dict[str, Dict[str, Any]]] = {}
    description_summary: Optional[Dict[str, Any]] = None
    try:
        scanner = imagenav.ImageNavDeterministicScanner(scanner_args)

        vertices = multimodal._collect_navmesh_triangle_vertices(scanner.pathfinder)
        floor_levels = multimodal._discover_floor_levels(
            vertices,
            floor_level_tolerance=float(args.floor_level_tolerance),
        )
        dominant_floors = [
            level
            for level in multimodal._select_dominant_floor_levels(
                floor_levels,
                max_floor_levels=int(args.max_floor_levels),
            )
            if float(level.projected_area_m2) >= 25.0
        ]
        scene_objects = multimodal._collect_scene_objects(scanner.sim)
        for obj in tqdm(
            scene_objects,
            desc=f"Exporting {args.scene_name}",
            unit="instance",
            dynamic_ncols=True,
        ):
            goal_probe = np.array(
                [float(obj.center[0]), float(obj.aabb_min[1]), float(obj.center[2])],
                dtype=np.float32,
            )
            nav_pos, nav_reason = multimodal._project_goal_to_navmesh(
                goal_probe,
                scanner.pathfinder,
                dominant_floors,
                float(args.floor_height_tolerance),
            )
            _, floor_y, _ = _supporting_floor(dominant_floors, float(obj.center[1]))
            view_points = _collect_view_points(scanner, obj, floor_y, args)
            render_view_candidates = _collect_render_view_candidates(
                scanner,
                obj,
                floor_y,
                args,
            )
            render_views, render_output_dir, render_views_json = _render_views_from_candidates(
                scanner,
                obj,
                render_view_candidates,
                output_root,
                args.scene_name,
                max_saved_views=int(args.max_render_viewpoints),
            )
            payload = {
                "semantic_id": int(obj.semantic_id),
                "category": obj.category,
                "center": _round_list(obj.center),
                "bbox_size": _round_list(obj.sizes),
                "bbox_min": _round_list(obj.aabb_min),
                "bbox_max": _round_list(obj.aabb_max),
                "horizontal_radius": round(float(obj.horizontal_radius), 3),
                "nav_position": _round_list(nav_pos) if nav_pos is not None else None,
                "view_points": view_points,
                "num_view_points": int(len(view_points)),
                "render_view_points": render_views,
                "num_render_view_points": int(len(render_views)),
            }
            if render_output_dir is not None:
                payload["render_output_dir"] = render_output_dir
            if render_views_json is not None:
                payload["render_views_json"] = render_views_json
            if nav_reason != "ok":
                payload["nav_projection_reason"] = nav_reason
            instances.setdefault(obj.category, {})[
                f"{obj.category}_{int(obj.semantic_id)}"
            ] = payload
    finally:
        if scanner is not None:
            scanner.close()

    if args.generate_descriptions:
        description_module = _load_description_module()
        description_api_key = (
            args.description_api_key
            or os.environ.get("DASHSCOPE_API_KEY")
            or os.environ.get("QWEN_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not description_api_key:
            raise RuntimeError(
                "Description generation enabled but API key is missing. "
                "Provide --description-api-key or set DASHSCOPE_API_KEY/QWEN_API_KEY/OPENAI_API_KEY."
            )
        description_summary = _generate_descriptions_for_instances(
            instances,
            description_module=description_module,
            output_root=output_root,
            api_base=str(args.description_api_base),
            api_key=str(description_api_key),
            model=str(args.description_model),
            max_images=int(args.description_max_images),
            retries=int(args.description_retries),
            timeout=int(args.description_timeout),
            sleep_seconds=float(args.description_sleep),
            overwrite=bool(args.description_overwrite),
            max_instances=args.description_max_instances,
        )

    category_counts = {category: len(entries) for category, entries in instances.items()}
    output = {
        "scene_name": args.scene_name,
        "scene_id": scene_id,
        "num_scene_objects": len(scene_objects),
        "category_counts": category_counts,
        "dominant_floors": [
            {
                "floor_index": int(level.index),
                "floor_y": round(float(level.y), 3),
                "min_y": round(float(level.min_y), 3),
                "max_y": round(float(level.max_y), 3),
                "projected_area_m2": round(float(level.projected_area_m2), 3),
            }
            for level in dominant_floors
        ],
        "instances": instances,
    }
    if description_summary is not None:
        output["description_generation"] = description_summary
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"Saved {sum(category_counts.values())} instances to: {output_path}")


if __name__ == "__main__":
    main()
