#!/usr/bin/env python3

"""Export all valid object instances for a scene (no start/goal distance checks)."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _load_multimodal_module() -> Any:
    module_path = Path(__file__).with_name("generate_multimodal_starts.py")
    spec = importlib.util.spec_from_file_location(
        "generate_multimodal_starts", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load generate_multimodal_starts.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_imagenav_module() -> Any:
    module_path = Path(__file__).with_name("generate_imagenav_eval_dataset.py")
    spec = importlib.util.spec_from_file_location(
        "generate_imagenav_eval_dataset", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load generate_imagenav_eval_dataset.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_description_module() -> Any:
    module_path = Path(__file__).with_name("generate_instance_descriptions_qwen.py")
    spec = importlib.util.spec_from_file_location(
        "generate_instance_descriptions_qwen", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load generate_instance_descriptions_qwen.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _round_list(values: np.ndarray, digits: int = 3) -> List[float]:
    return [round(float(v), digits) for v in values.tolist()]


def _normalize_category_key(category: str) -> str:
    return category.strip().lower().replace(" ", "_")


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _nearest_floor(
    floors: List[Any],
    y_value: float,
) -> Tuple[Optional[int], Optional[float], Optional[float]]:
    if not floors:
        return None, None, None
    closest = min(floors, key=lambda level: abs(float(level.y) - float(y_value)))
    delta = float(y_value) - float(closest.y)
    return int(closest.index), float(closest.y), float(delta)


def _summarize_invalid_views(
    invalid_views: List[Dict[str, Any]],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    reason_counts: Dict[str, int] = {}
    stage_counts: Dict[str, int] = {}
    for view in invalid_views:
        reason = str(view.get("reason", "unknown"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        stage = str(view.get("stage", "unknown"))
        stage_counts[stage] = stage_counts.get(stage, 0) + 1
    return reason_counts, stage_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export all valid instances for a scene based on navmesh rules."
    )
    parser.add_argument("scene_name", type=str, help="Scene id or .glb name")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (defaults to output/<scene>/valid_instances.json)",
    )
    parser.add_argument(
        "--image-report",
        type=Path,
        default=None,
        help="Output JSON path for instances without saved images",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Root output directory for rendered images",
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
        default=Path("configs/semantic_audionav/av_nav/mp3d/semantic_audiogoal.yaml"),
        help="Habitat task config used to instantiate simulator",
    )
    parser.add_argument("--width", type=int, default=512, help="Sensor width")
    parser.add_argument("--height", type=int, default=512, help="Sensor height")
    parser.add_argument("--hfov", type=float, default=90.0, help="Sensor HFOV")
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed for reproducible rendering candidates",
    )
    parser.add_argument(
        "--horizontal-only-rotation",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Constrain render rotation to yaw-only (horizontal view) (default: yes)",
    )
    parser.add_argument(
        "--save-images",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Save rendered RGB/YOLO images (default: yes)",
    )
    parser.add_argument(
        "--yolo-model",
        type=Path,
        default=Path("models/yolo26x.pt"),
        help="YOLO weights for render-time validation (default: models/yolo26x.pt)",
    )
    parser.add_argument(
        "--yolo-device",
        type=str,
        default=None,
        help="YOLO inference device (e.g., cpu, 0, cuda:0)",
    )
    parser.add_argument(
        "--yolo-conf-threshold",
        type=float,
        default=0.7,
        help="YOLO confidence threshold for render-time validation",
    )
    parser.add_argument(
        "--yolo-iou-threshold",
        type=float,
        default=0.45,
        help="YOLO NMS IoU threshold for render-time validation",
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
        help="Allowed height difference between nav points and dominant floors",
    )
    parser.add_argument(
        "--max-floor-levels",
        type=int,
        default=3,
        help="Maximum number of dominant floor bands used for sampling",
    )
    parser.add_argument(
        "--footprint-snap-distance",
        type=float,
        default=0.25,
        help="Max xz snap distance when validating object footprint navigability",
    )
    parser.add_argument(
        "--include-rejections",
        action="store_true",
        help="Include rejected instances with reasons",
    )
    parser.add_argument(
        "--generate-descriptions",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Generate English descriptions for instances with rendered images using VLM (default: yes).",
    )
    parser.add_argument(
        "--description-api-key",
        type=str,
        default="sk-QgWdM03NkfNrFfMA576126F43fAa4b0eBb635d80C6D2Cc91",
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
    return parser.parse_args()


def _build_imagenav_args(
    args: argparse.Namespace,
    output_root: Path,
) -> argparse.Namespace:
    return argparse.Namespace(
        scene_name=args.scene_name,
        target_category=None,
        yolo_model=args.yolo_model,
        yolo_device=args.yolo_device,
        yolo_conf_threshold=args.yolo_conf_threshold,
        yolo_iou_threshold=args.yolo_iou_threshold,
        yolo_max_det=args.yolo_max_det,
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
        horizontal_only_rotation=args.horizontal_only_rotation,
        sensor_height=args.sensor_height,
        search_radius=1.0,
        min_surface_offset=0.5,
        candidate_angle_step=30,
        radial_step=0.15,
        radial_step_jitter=0.05,
        angle_jitter=10.0,
        min_edge_clearance=0.1,
        min_iou=0.2,
        max_floor_levels=args.max_floor_levels,
        floor_level_tolerance=args.floor_level_tolerance,
        nearby_object_radius=3.0,
        max_viewpoints=3,
        min_sensor_offset=1.25,
        max_sensor_offset=1.5,
        sensor_offset_step=0.125,
        floor_height_tolerance=0.25,
        min_viewpoint_angle_sep=45,
        min_viewpoint_separation=0.75,
        max_snap_distance=0.5,
        object_clearance=0.1,
        position_adjust_max=0.25,
        position_adjust_step=0.05,
        render_validation_retry_radius=0.25,
        render_validation_retry_max_attempts=10,
        save_images=bool(args.save_images),
    )


def _write_render_metadata(
    scanner: Any,
    target: Any,
    target_views: List[Dict[str, Any]],
    invalid_views: List[Dict[str, Any]],
    object_output_dir: Path,
) -> Tuple[Path, Path]:
    object_output_dir.mkdir(parents=True, exist_ok=True)
    object_views_json = object_output_dir / "views.json"
    invalid_views_json = object_output_dir / "invalid_views.json"
    target_floor_level = scanner._nearest_floor_level(
        float(target.center[1] - 0.5 * target.sizes[1])
    )
    object_payload = {
        "object_id": int(target.semantic_id),
        "object_category": target.category_name,
        "object_center": _round_list(target.center),
        "object_sizes": _round_list(target.sizes),
        "horizontal_radius": round(float(target.horizontal_radius), 3),
        "object_base_y": round(float(target.center[1] - 0.5 * target.sizes[1]), 3),
        "floor_index": (
            int(target_floor_level.index) if target_floor_level is not None else None
        ),
        "floor_y": (
            round(float(target_floor_level.y), 3)
            if target_floor_level is not None
            else None
        ),
        "num_views": len(target_views),
        "views": target_views,
    }
    object_views_json.write_text(json.dumps(object_payload, indent=2, ensure_ascii=False))

    invalid_payload = {
        "object_id": int(target.semantic_id),
        "object_category": target.category_name,
        "object_center": _round_list(target.center),
        "object_sizes": _round_list(target.sizes),
        "horizontal_radius": round(float(target.horizontal_radius), 3),
        "num_invalid_views": len(invalid_views),
        "invalid_views": invalid_views,
    }
    invalid_views_json.write_text(
        json.dumps(invalid_payload, indent=2, ensure_ascii=False)
    )

    return object_views_json, invalid_views_json


def _generate_descriptions_for_valid_instances(
    valid_instances: Dict[str, Dict[str, Dict[str, Any]]],
    *,
    description_module: Any,
    output_json_path: Path,
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
    for category, entries in valid_instances.items():
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

    print(
        "Generating descriptions with {} for {} instances...".format(model, total)
    )

    for idx, (_, instance_key, instance_payload) in enumerate(flat_instances, start=1):
        image_payload = instance_payload.get("image")
        render_views = (
            image_payload.get("render_views")
            if isinstance(image_payload, dict)
            else None
        )
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

        candidates = description_module._resolve_instance_image_candidates(
            input_path=output_json_path,
            output_root=output_root,
            instance_key=instance_key,
            max_images=max(1, int(max_images)),
        )

        if len(candidates) == 0:
            instance_payload["description_meta"] = {
                "status": "missing_images",
                "model": model,
            }
            stats["failed"] += 1
            print("[{}/{}] {}: missing image files".format(idx, total, instance_key))
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
            instance_payload["description_meta"] = {
                "status": "api_error",
                "model": model,
                "error": str(exc),
                "source_images": rel_paths,
            }
            stats["failed"] += 1
            print("[{}/{}] {}: api_error {}".format(idx, total, instance_key, exc))
            if sleep_seconds > 0:
                time.sleep(float(sleep_seconds))
            continue

        if description is None:
            instance_payload["description_meta"] = {
                "status": status,
                "model": model,
                "source_images": rel_paths,
            }
            stats["failed"] += 1
            print("[{}/{}] {}: invalid ({})".format(idx, total, instance_key, status))
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

        instance_payload["description_meta"] = {
            "status": "ok",
            "model": model,
            "word_count": description_module._word_count(description),
            "source_images": rel_paths,
            "num_source_views": len(rel_paths),
        }
        stats["updated"] += 1
        print("[{}/{}] {}: {}".format(idx, total, instance_key, description))

        if sleep_seconds > 0:
            time.sleep(float(sleep_seconds))

    return stats


def main() -> None:
    args = parse_args()
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    module = _load_multimodal_module()
    imagenav_module = _load_imagenav_module()

    if args.generate_descriptions and not args.save_images:
        print("Description generation enabled; forcing --save-images.")
        args.save_images = True

    scene_dir = args.scene_dir.expanduser().resolve()
    scene_path = module._resolve_scene_path(scene_dir, args.scene_name)
    scene_id = module._scene_id_for_dataset(scene_path, scene_dir)
    output_root = args.output_root.expanduser().resolve()
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else output_root / args.scene_name / "valid_instances.json"
    )
    image_report_path = (
        args.image_report.expanduser().resolve()
        if args.image_report is not None
        else output_root / args.scene_name / "invalid_image_instances.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_report_path.parent.mkdir(parents=True, exist_ok=True)

    image_scanner = None
    try:
        imagenav_args = _build_imagenav_args(args, output_root)
        image_scanner = imagenav_module.ImageNavDeterministicScanner(imagenav_args)
        target_lookup: Dict[Tuple[str, int], Any] = {
            (_normalize_category_key(target.category_name), int(target.semantic_id)): target
            for target in image_scanner.scene_objects
        }

        vertices = module._collect_navmesh_triangle_vertices(image_scanner.pathfinder)
        floor_levels = module._discover_floor_levels(
            vertices, floor_level_tolerance=float(args.floor_level_tolerance)
        )
        dominant_floors = [
            level
            for level in module._select_dominant_floor_levels(
                floor_levels, max_floor_levels=int(args.max_floor_levels)
            )
            if float(level.projected_area_m2) >= 25.0
        ]

        scene_objects = module._collect_scene_objects(image_scanner.sim)
        accessible_objects, footprint_rejections = module._filter_accessible_objects(
            scene_objects,
            pathfinder=image_scanner.pathfinder,
            snap_distance=float(args.footprint_snap_distance),
            floor_levels=dominant_floors,
            floor_height_tolerance=float(args.floor_height_tolerance),
        )

        valid_instances: Dict[str, Dict[str, Dict[str, Any]]] = {}
        no_image_instances: List[Dict[str, Any]] = []
        projection_rejections: List[Dict[str, Any]] = []
        for obj in accessible_objects:
            nav_pos, nav_reason = module._project_goal_to_navmesh(
                obj.center,
                image_scanner.pathfinder,
                dominant_floors,
                float(args.floor_height_tolerance),
            )
            if nav_pos is None:
                projection_rejections.append(
                    {
                        "semantic_id": int(obj.semantic_id),
                        "category": obj.category,
                        "reason": nav_reason,
                    }
                )
                continue
            floor_index, floor_y, floor_delta = _nearest_floor(dominant_floors, nav_pos[1])
            category_bucket = valid_instances.setdefault(obj.category, {})
            key = f"{obj.category}_{int(obj.semantic_id)}"
            instance_payload = {
                "semantic_id": int(obj.semantic_id),
                "category": obj.category,
                "center": _round_list(obj.center),
                "bbox_size": _round_list(obj.sizes),
                "bbox_min": _round_list(obj.aabb_min),
                "bbox_max": _round_list(obj.aabb_max),
                "horizontal_radius": round(float(obj.horizontal_radius), 3),
                "nav_position": _round_list(nav_pos),
                "nav_floor_index": floor_index,
                "nav_floor_y": round(float(floor_y), 3) if floor_y is not None else None,
                "nav_floor_delta": round(float(floor_delta), 3)
                if floor_delta is not None
                else None,
            }

            target_key = (_normalize_category_key(obj.category), int(obj.semantic_id))
            render_target = target_lookup.get(target_key)
            if render_target is None:
                sizes = np.array(obj.sizes, dtype=np.float32)
                center = np.array(obj.center, dtype=np.float32)
                horizontal_radius = 0.5 * float(max(sizes[0], sizes[2]))
                render_target = imagenav_module.SemanticTarget(
                    semantic_id=int(obj.semantic_id),
                    category_name=obj.category,
                    center=center,
                    sizes=sizes,
                    horizontal_radius=horizontal_radius,
                )

            category_dir = image_scanner._category_dirname(render_target.category_name)
            object_output_dir = (
                output_root / args.scene_name / category_dir / str(render_target.semantic_id)
            )
            object_output_dir.mkdir(parents=True, exist_ok=True)
            target_views, invalid_views = image_scanner._scan_target(
                render_target, object_output_dir
            )
            if len(target_views) > 3:
                target_views = target_views[:3]
            object_views_json, invalid_views_json = _write_render_metadata(
                image_scanner,
                render_target,
                target_views,
                invalid_views,
                object_output_dir,
            )
            pose_only_views = [
                image_scanner._pose_only_view(view) for view in target_views
            ]
            if pose_only_views:
                instance_payload["image"] = {
                    "output_dir": object_output_dir.relative_to(output_root).as_posix(),
                    "views_json": object_views_json.relative_to(output_root).as_posix(),
                    "invalid_views_json": invalid_views_json.relative_to(
                        output_root
                    ).as_posix(),
                    "render_views": pose_only_views,
                    "num_views": len(target_views),
                    "num_invalid_views": len(invalid_views),
                }
            else:
                reason_counts, stage_counts = _summarize_invalid_views(invalid_views)
                no_image_instances.append(
                    {
                        "semantic_id": int(obj.semantic_id),
                        "category": obj.category,
                        "key": key,
                        "output_dir": object_output_dir.relative_to(output_root).as_posix(),
                        "views_json": object_views_json.relative_to(output_root).as_posix(),
                        "invalid_views_json": invalid_views_json.relative_to(
                            output_root
                        ).as_posix(),
                        "num_invalid_views": len(invalid_views),
                        "invalid_views": invalid_views,
                        "reason_counts": reason_counts,
                        "stage_counts": stage_counts,
                    }
                )

            category_bucket[key] = instance_payload
    finally:
        if image_scanner is not None:
            image_scanner.close()

    description_summary: Optional[Dict[str, Any]] = None
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

        description_summary = _generate_descriptions_for_valid_instances(
            valid_instances,
            description_module=description_module,
            output_json_path=output_path,
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

    category_counts: Dict[str, int] = {
        category: len(entries) for category, entries in valid_instances.items()
    }

    payload: Dict[str, Any] = {
        "scene_name": args.scene_name,
        "scene_id": scene_id,
        "num_scene_objects": len(scene_objects),
        "num_accessible_objects": len(accessible_objects),
        "num_valid_instances": sum(category_counts.values()),
        "category_counts": category_counts,
        "image_report_json": _relative_to_root(image_report_path, output_root),
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
        "instances": valid_instances,
    }

    if description_summary is not None:
        payload["description_generation"] = description_summary

    if args.include_rejections:
        payload["footprint_rejections"] = [
            {
                "semantic_id": semantic_id,
                "category": category,
                "reason": reason,
            }
            for (semantic_id, category), reason in footprint_rejections.items()
        ]
        payload["projection_rejections"] = projection_rejections

    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    image_report_payload = {
        "scene_name": args.scene_name,
        "scene_id": scene_id,
        "num_instances": sum(category_counts.values()),
        "num_instances_without_images": len(no_image_instances),
        "instances_without_images": no_image_instances,
    }
    image_report_path.write_text(
        json.dumps(image_report_payload, indent=2, ensure_ascii=False)
    )
    total_instances = sum(category_counts.values())
    print(
        f"Saved {total_instances} valid instances across {len(valid_instances)} categories to: {output_path}"
    )


if __name__ == "__main__":
    main()
