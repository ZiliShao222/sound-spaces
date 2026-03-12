#!/usr/bin/env python3

"""Deterministically scan MP3D semantic targets and export ImageNav goal images."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import logging
import math
import os
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


def _bootstrap_native_dependencies() -> None:
    """Load known native deps before importing habitat-sim.

    Set ``SOUNDSPACES_FORCE_SYSTEM_LIBZ=1`` to preload system zlib in
    environments where RLRAudioPropagation links against an incompatible libz.
    """
    try:
        import quaternion  # noqa: F401
    except Exception:
        pass

    if os.environ.get("SOUNDSPACES_FORCE_SYSTEM_LIBZ", "0") != "1":
        return

    candidates = (
        "/lib/x86_64-linux-gnu/libz.so.1",
        "/usr/lib/x86_64-linux-gnu/libz.so.1",
    )
    for candidate in candidates:
        if not os.path.isfile(candidate):
            continue
        try:
            ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue

    libz_path = ctypes.util.find_library("z")
    if libz_path is None:
        return
    try:
        ctypes.CDLL(libz_path, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass


_bootstrap_native_dependencies()

import habitat_sim
import numpy as np
import quaternion
import soundspaces  # noqa: F401
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat.sims import make_sim
from habitat_sim.utils.common import quat_from_angle_axis
from PIL import Image
from ss_baselines.av_nav.config.default import get_task_config
from soundspaces.continuous_simulator import _quat_to_list
from tqdm import tqdm
from yolo26_demo import (
    annotate_image as annotate_yolo_image,
    deduplicate as yolo_deduplicate,
    load_aliases as load_yolo_aliases,
    load_yolo,
    matches_target as yolo_matches_target,
    normalize_label as yolo_normalize_label,
    run_inference as run_yolo_inference,
)


MP3D_TARGET_CATEGORIES: Tuple[str, ...] = (
    "chair",
    "table",
    "picture",
    "cabinet",
    "cushion",
    "sofa",
    "bed",
    "chest_of_drawers",
    "plant",
    "sink",
    "toilet",
    "stool",
    "towel",
    "tv_monitor",
    "shower",
    "bathtub",
    "counter",
    "fireplace",
    "gym_equipment",
    "seating",
    "clothes",
)

YOLO_TARGET_ALIASES: Dict[str, Tuple[str, ...]] = {
    "chair": ("chair",),
    "table": ("table", "dining table", "desk", "coffee table", "end table"),
    "picture": ("picture", "painting", "poster", "wall art", "photo"),
    "cabinet": ("cabinet", "cupboard"),
    "cushion": ("cushion", "pillow"),
    "sofa": ("sofa", "couch"),
    "bed": ("bed",),
    "chest_of_drawers": (
        "chest_of_drawers",
        "chest of drawers",
        "dresser",
        "drawer cabinet",
    ),
    "plant": ("plant", "potted plant"),
    "sink": ("sink", "basin"),
    "toilet": ("toilet",),
    "stool": ("stool",),
    "towel": ("towel",),
    "tv_monitor": ("tv_monitor", "tv", "television", "monitor", "screen"),
    "shower": ("shower", "shower stall"),
    "bathtub": ("bathtub", "bath tub", "tub"),
    "counter": ("counter", "countertop"),
    "fireplace": ("fireplace",),
    "gym_equipment": (
        "gym_equipment",
        "gym equipment",
        "exercise equipment",
        "treadmill",
        "exercise bike",
        "elliptical",
    ),
    "seating": ("seating", "seat", "chair", "sofa", "couch", "bench", "stool"),
    "clothes": ("clothes", "clothing", "shirt", "pants", "jacket", "coat", "dress"),
}


@dataclass
class SemanticTarget:
    """Semantic target with geometry used for scanning."""

    semantic_id: int
    category_name: str
    center: np.ndarray
    sizes: np.ndarray
    horizontal_radius: float
    object_id_raw: Optional[str] = None
    object_name: Optional[str] = None
    room_id: Optional[str] = None
    room_name: Optional[str] = None
    level_id: Optional[int] = None


@dataclass
class ViewpointCandidate:
    """Renderable viewpoint candidate selected for one target object."""

    floor_position: np.ndarray
    sensor_position: np.ndarray
    base_position: np.ndarray
    rotation: Any
    surface_distance: float
    angle_deg: int
    edge_clearance: float
    nearby_count: int
    nearby_penalty: float
    sensor_clearance: float
    sensor_offset: float
    floor_level_index: Optional[int]
    floor_level_y: float


@dataclass
class FloorLevel:
    """Dominant navigable floor band inferred from navmesh vertex heights."""

    y: float
    min_y: float
    max_y: float
    count: int
    area_m2: float = 0.0
    projected_area_m2: float = 0.0
    index: int = -1


class ImageNavDeterministicScanner:
    """Minimal deterministic scanner for semantic-object ImageNav goals."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize scanner state and simulator resources."""
        self.args = args
        self.logger = logging.getLogger(self.__class__.__name__)

        self.scene_dir = args.scene_dir.expanduser().resolve()
        self.output_root = args.output_root.expanduser().resolve()

        self.scene_path = self._resolve_scene_path(args.scene_name)
        self.scene_name = args.scene_name
        self.scene_id = self._scene_id_for_dataset(self.scene_path)
        self.target_category = (
            args.target_category.strip().lower()
            if args.target_category is not None and args.target_category.strip()
            else None
        )
        self.target_categories = set(MP3D_TARGET_CATEGORIES)
        self.target_label = (
            args.target_category.strip()
            if self.target_category is not None
            else "all_mp3d_target_categories"
        )
        self.yolo_model = None
        self.yolo_aliases = self._build_yolo_aliases(args.yolo_aliases_json)
        self.clearance_ignore_categories = {"floor", "ceiling"}
        self.nearby_ignore_categories = {
            "floor",
            "ceiling",
            "wall",
            "door",
            "window",
            "stairs",
            "stair",
            "railing",
            "beam",
            "column",
        }

        self.sim = self._build_simulator(self.scene_path)
        self.inner_sim = getattr(self.sim, "_sim", self.sim)
        self.pathfinder = self.sim.pathfinder
        self.rgb_uuid = self._get_sensor_uuid("rgb")
        self.semantic_uuid = self._get_sensor_uuid("semantic")
        self.sensor_height = float(self.args.sensor_height)
        if (
            abs(float(self.args.min_sensor_offset) - self.sensor_height) > 1e-6
            or abs(float(self.args.max_sensor_offset) - self.sensor_height) > 1e-6
        ):
            self.logger.warning(
                "Ignoring --min-sensor-offset/--max-sensor-offset for rendering "
                "because Habitat's sensor is fixed at --sensor-height above the agent base."
            )
        self.scene_objects = self._collect_scene_objects()
        self.navmesh_triangle_vertices = self._collect_navmesh_triangle_vertices()
        self.navmesh_vertices = self._collect_navmesh_vertices(
            self.navmesh_triangle_vertices
        )
        self.all_floor_levels = self._discover_floor_levels()
        self.floor_levels = self._select_dominant_floor_levels(self.all_floor_levels)
        self.largest_navmesh_island_radius = (
            self._compute_largest_navmesh_island_radius()
        )
        self._initialize_yolo_validator()

        if self.all_floor_levels:
            all_floor_summary = [
                {
                    "band_index": int(level.index),
                    "y": round(level.y, 3),
                    "min_y": round(level.min_y, 3),
                    "max_y": round(level.max_y, 3),
                    "triangles": int(level.count),
                    "area_m2": round(level.area_m2, 3),
                    "projected_area_m2": round(level.projected_area_m2, 3),
                }
                for level in self.all_floor_levels
            ]
            self.logger.info("All navmesh height bands: %s", all_floor_summary)

        if self.floor_levels:
            dominant_summary = [
                {
                    "floor_index": int(level.index),
                    "y": round(level.y, 3),
                    "min_y": round(level.min_y, 3),
                    "max_y": round(level.max_y, 3),
                    "triangles": int(level.count),
                    "area_m2": round(level.area_m2, 3),
                    "projected_area_m2": round(level.projected_area_m2, 3),
                }
                for level in self.floor_levels
            ]
            self.logger.info("Dominant floor levels: %s", dominant_summary)

    def _serialize_floor_level(self, floor_level: FloorLevel) -> Dict[str, Any]:
        """Convert one floor level into JSON-friendly scene metadata."""
        total_triangles = max(1, int(sum(level.count for level in self.all_floor_levels)))
        total_projected_area = max(
            1e-8,
            float(sum(level.projected_area_m2 for level in self.all_floor_levels)),
        )
        return {
            "floor_index": int(floor_level.index),
            "floor_y": round(float(floor_level.y), 3),
            "min_y": round(float(floor_level.min_y), 3),
            "max_y": round(float(floor_level.max_y), 3),
            "height_span": round(float(floor_level.max_y - floor_level.min_y), 3),
            "triangle_count": int(floor_level.count),
            "triangle_ratio": round(float(floor_level.count) / float(total_triangles), 6),
            "area_m2": round(float(floor_level.area_m2), 6),
            "projected_area_m2": round(float(floor_level.projected_area_m2), 6),
            "projected_area_ratio": round(
                float(floor_level.projected_area_m2) / float(total_projected_area),
                6,
            ),
            "selected_as_dominant": any(
                int(level.index) == int(floor_level.index) for level in self.floor_levels
            ),
        }

    def _serialize_scene_object(self, scene_object: SemanticTarget) -> Dict[str, Any]:
        """Convert one semantic object into JSON-friendly scene metadata."""
        base_y = float(scene_object.center[1] - 0.5 * scene_object.sizes[1])
        top_y = float(scene_object.center[1] + 0.5 * scene_object.sizes[1])
        aabb_min = np.array(scene_object.center, dtype=np.float32) - 0.5 * np.array(
            scene_object.sizes,
            dtype=np.float32,
        )
        aabb_max = np.array(scene_object.center, dtype=np.float32) + 0.5 * np.array(
            scene_object.sizes,
            dtype=np.float32,
        )
        floor_level = self._nearest_floor_level(base_y)
        return {
            "semantic_id": int(scene_object.semantic_id),
            "object_id_raw": scene_object.object_id_raw,
            "object_name": scene_object.object_name,
            "category": scene_object.category_name,
            "center": [round(float(v), 3) for v in scene_object.center],
            "sizes": [round(float(v), 3) for v in scene_object.sizes],
            "aabb_min": [round(float(v), 3) for v in aabb_min],
            "aabb_max": [round(float(v), 3) for v in aabb_max],
            "horizontal_radius": round(float(scene_object.horizontal_radius), 3),
            "base_y": round(base_y, 3),
            "top_y": round(top_y, 3),
            "floor_index": self._floor_level_index(floor_level),
            "floor_y": (
                round(float(floor_level.y), 3) if floor_level is not None else None
            ),
            "base_to_floor_delta": (
                round(base_y - float(floor_level.y), 3)
                if floor_level is not None
                else None
            ),
            "room_id": scene_object.room_id,
            "room_name": scene_object.room_name,
            "level_id": scene_object.level_id,
            "matches_target_category": self._matches_requested_target(
                scene_object.category_name
            ),
        }

    def _build_objects_payload(self) -> Dict[str, Any]:
        """Build scene-level object metadata grouped by semantic category."""
        categories: Dict[str, List[Dict[str, Any]]] = {}
        for scene_object in self.scene_objects:
            category_name = scene_object.category_name or "<unknown>"
            categories.setdefault(category_name, []).append(
                self._serialize_scene_object(scene_object)
            )

        sorted_categories = sorted(
            categories.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )
        category_counts = {
            category_name: len(objects)
            for category_name, objects in sorted_categories
        }
        categories_payload = {
            category_name: {
                "count": len(objects),
                "objects": sorted(objects, key=lambda item: item["semantic_id"]),
            }
            for category_name, objects in sorted_categories
        }

        return {
            "scene_name": self.scene_name,
            "scene_id": self.scene_id,
            "target_category": self.target_label,
            "target_categories": self._active_target_categories(),
            "total_objects": len(self.scene_objects),
            "num_categories": len(sorted_categories),
            "category_counts": category_counts,
            "categories": categories_payload,
        }

    def _json_value(self, value: Any, float_decimals: int = 3) -> Any:
        """Convert numpy-heavy values into JSON-friendly Python primitives."""
        if value is None or isinstance(value, (str, bool)):
            return value
        if isinstance(value, np.ndarray):
            return [self._json_value(item, float_decimals=float_decimals) for item in value.tolist()]
        if isinstance(value, (list, tuple)):
            return [self._json_value(item, float_decimals=float_decimals) for item in value]
        if isinstance(value, dict):
            return {
                str(key): self._json_value(item, float_decimals=float_decimals)
                for key, item in value.items()
            }
        if isinstance(value, (np.integer, int)):
            return int(value)
        if isinstance(value, (np.floating, float)):
            return round(float(value), float_decimals)
        return value

    def _invalid_view_record(
        self,
        *,
        target: SemanticTarget,
        stage: str,
        reason: str,
        floor_position: Optional[np.ndarray] = None,
        base_position: Optional[np.ndarray] = None,
        sensor_position: Optional[np.ndarray] = None,
        rotation: Optional[Any] = None,
        angle_deg: Optional[int] = None,
        surface_distance: Optional[float] = None,
        candidate_source: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build one structured invalid-view record for later debugging."""
        reason_detail = self._format_invalid_reason(reason, details)
        payload: Dict[str, Any] = {
            "semantic_id": int(target.semantic_id),
            "category": target.category_name,
            "stage": stage,
            "reason": reason,
            "reason_detail": reason_detail,
            "object_center": self._json_value(target.center),
            "floor_position": self._json_value(floor_position),
            "agent_base_position": self._json_value(base_position),
            "position": self._json_value(sensor_position),
            "angle_deg": int(angle_deg) if angle_deg is not None else None,
            "surface_distance": (
                round(float(surface_distance), 3)
                if surface_distance is not None
                else None
            ),
            "candidate_source": candidate_source,
        }
        if rotation is not None:
            payload["rotation"] = [round(float(v), 6) for v in _quat_to_list(rotation)]
        if details:
            payload["details"] = self._json_value(details)
        return payload

    def _format_invalid_reason(
        self,
        reason: str,
        details: Optional[Dict[str, Any]],
    ) -> Optional[str]:
        """Create a human-readable explanation for an invalid-view reason."""
        details = details or {}

        def _fmt(value: Optional[float]) -> Optional[str]:
            if value is None:
                return None
            try:
                return f"{float(value):.3f}"
            except Exception:
                return None

        if reason == "y_mismatch":
            point_y = _fmt(details.get("point_y"))
            target_floor_y = _fmt(details.get("target_floor_y"))
            y_diff = _fmt(details.get("y_diff"))
            tol = f"{float(self.args.floor_height_tolerance):.3f}"
            if point_y and target_floor_y and y_diff:
                return (
                    f"abs(point_y-target_floor_y)={y_diff} > floor_height_tolerance={tol} "
                    f"(point_y={point_y}, target_floor_y={target_floor_y})"
                )
        if reason == "too_far":
            center_distance = _fmt(details.get("center_distance"))
            max_distance = _fmt(details.get("max_center_distance"))
            if center_distance and max_distance:
                return f"center_distance={center_distance} > max_center_distance={max_distance}"
        if reason == "too_near":
            center_distance = _fmt(details.get("center_distance"))
            min_distance = _fmt(details.get("min_center_distance"))
            if center_distance and min_distance:
                return f"center_distance={center_distance} < min_center_distance={min_distance}"
        if reason == "edge_reject":
            edge_clearance = _fmt(details.get("edge_clearance"))
            min_edge = _fmt(details.get("min_edge_clearance"))
            if edge_clearance and min_edge:
                return f"edge_clearance={edge_clearance} < min_edge_clearance={min_edge}"
        if reason == "invalid_floor_navmesh":
            navigable = details.get("is_navigable")
            floor_match = details.get("matches_floor_level")
            parts = []
            if navigable is not None:
                parts.append(f"is_navigable={bool(navigable)}")
            if floor_match is not None:
                parts.append(f"matches_floor_level={bool(floor_match)}")
            if parts:
                return "invalid floor navmesh point (" + ", ".join(parts) + ")"
            return "invalid floor navmesh point"
        if reason == "sensor_pose_none":
            return "no valid sensor pose found for this floor candidate"
        if reason == "diversity_reject":
            min_angle_diff = _fmt(details.get("min_angle_diff"))
            min_angle_threshold = _fmt(details.get("min_angle_threshold"))
            min_separation = _fmt(details.get("min_separation"))
            min_separation_threshold = _fmt(details.get("min_separation_threshold"))
            parts = []
            if min_angle_diff and min_angle_threshold:
                parts.append(f"min_angle_diff={min_angle_diff} < {min_angle_threshold}")
            if min_separation and min_separation_threshold:
                parts.append(
                    f"min_separation={min_separation} < {min_separation_threshold}"
                )
            if parts:
                return "diversity constraints failed (" + ", ".join(parts) + ")"
            return "diversity constraints failed"
        if reason == "observe_failure":
            return "failed to render sensor observations"
        if reason == "iou_reject":
            iou = _fmt(details.get("iou"))
            min_iou = _fmt(details.get("min_iou"))
            if iou and min_iou:
                return f"iou={iou} < min_iou={min_iou}"
        if reason == "yolo_reject":
            yolo_num = details.get("yolo_num_detections")
            yolo_matches = details.get("yolo_matched_detections")
            threshold = details.get("yolo_conf_threshold")
            if threshold is not None:
                return (
                    f"yolo matched detections below threshold {float(threshold):.2f} "
                    f"(num_detections={yolo_num}, matches={yolo_matches})"
                )
            return "yolo validation failed"
        return None

    def _build_yolo_aliases(self, alias_path: Optional[Path]) -> Dict[str, List[str]]:
        """Merge built-in MP3D-to-YOLO aliases with optional user overrides."""
        aliases: Dict[str, List[str]] = {
            yolo_normalize_label(key): yolo_deduplicate([key, *values])
            for key, values in YOLO_TARGET_ALIASES.items()
        }
        user_aliases = load_yolo_aliases(alias_path) if alias_path is not None else {}
        for key, values in user_aliases.items():
            normalized = yolo_normalize_label(key)
            aliases[normalized] = yolo_deduplicate([normalized, *aliases.get(normalized, []), *values])
        return aliases

    def _initialize_yolo_validator(self) -> None:
        """Load YOLO26 validation model once if configured."""
        if self.args.yolo_model is None:
            self.logger.info("YOLO26 validation disabled; saving views without detector check")
            return
        model_path = self.args.yolo_model.expanduser().resolve()
        self.yolo_model = load_yolo(model_path)
        self.logger.info("Loaded YOLO26 validation model: %s", model_path)

    def _yolo_target_labels(self, category_name: str) -> List[str]:
        """Return the target label list used to validate one rendered object."""
        return yolo_deduplicate([category_name])

    def _validate_with_yolo(
        self,
        rgb: np.ndarray,
        category_name: str,
    ) -> Tuple[bool, List[Dict[str, Any]], int, Optional[Image.Image]]:
        """Return whether YOLO26 recognizes the rendered object category."""
        if self.yolo_model is None:
            return True, [], 0, None

        detections = run_yolo_inference(
            self.yolo_model,
            rgb,
            device=self.args.yolo_device,
            conf_threshold=float(self.args.yolo_conf_threshold),
            iou_threshold=float(self.args.yolo_iou_threshold),
            max_det=int(self.args.yolo_max_det),
        )
        target_labels = self._yolo_target_labels(category_name)
        matched = [
            detection.to_dict()
            for detection in detections
            if (
                float(detection.confidence) >= float(self.args.yolo_conf_threshold)
                and yolo_matches_target(
                    detection.class_name,
                    target_labels,
                    self.yolo_aliases,
                )
            )
        ]
        annotated = annotate_yolo_image(
            rgb,
            detections,
            target_labels=target_labels,
            aliases=self.yolo_aliases,
        )
        return len(matched) > 0, matched, len(detections), annotated

    def _normalize_category_name(self, category_name: str) -> str:
        """Return canonical category token used for matching."""
        return category_name.strip().lower()

    def _active_target_categories(self) -> List[str]:
        """Return the active target categories for this run."""
        if self.target_category is not None:
            return [self.target_category]
        return list(MP3D_TARGET_CATEGORIES)

    def _matches_requested_target(self, category_name: str) -> bool:
        """Return whether an object category should be scanned in this run."""
        normalized = self._normalize_category_name(category_name)
        if self.target_category is not None:
            return self.target_category in normalized
        return normalized in self.target_categories

    def _pose_only_view(self, view: Dict[str, Any]) -> Dict[str, Any]:
        """Return a compact render-parameter view payload without image paths."""
        return {
            "resolution": {
                "width": int(self.args.width),
                "height": int(self.args.height),
            },
            "hfov": float(self.args.hfov),
            "position": list(view["position"]),
            "agent_base_position": list(view["agent_base_position"]),
            "rotation": list(view["rotation"]),
            "radius": view["radius"],
            "angle_deg": view["angle_deg"],
            "frame_cov": view["frame_cov"],
            "iou": view["iou"],
            "yolo_num_detections": int(view.get("yolo_num_detections", 0)),
            "yolo_matched_detections": list(view.get("yolo_matched_detections", [])),
        }

    def _category_dirname(self, category_name: str) -> str:
        """Return a filesystem-friendly directory name for one category."""
        normalized = category_name.strip().lower()
        normalized = normalized.replace("/", "_")
        normalized = normalized.replace("\\", "_")
        normalized = "_".join(normalized.split())
        return normalized or "unknown"

    def close(self) -> None:
        """Close simulator resources."""
        if self.sim is not None:
            self.sim.close()

    def run(self) -> Path:
        """Scan target objects, save valid views, and write a compact JSON manifest."""
        scene_output_dir = self.output_root / self.scene_name
        scene_output_dir.mkdir(parents=True, exist_ok=True)

        objects_metadata_path = scene_output_dir / "objects.json"
        objects_payload = self._build_objects_payload()
        with objects_metadata_path.open("w", encoding="utf-8") as file_obj:
            json.dump(objects_payload, file_obj, indent=2, ensure_ascii=False)

        floor_metadata_path = scene_output_dir / "floor_metadata.json"
        floor_metadata_payload = {
            "scene_name": self.scene_name,
            "scene_id": self.scene_id,
            "num_navmesh_height_bands": len(self.all_floor_levels),
            "num_dominant_floor_levels": len(self.floor_levels),
            "all_navmesh_height_bands": [
                self._serialize_floor_level(level) for level in self.all_floor_levels
            ],
            "dominant_floor_levels": [
                self._serialize_floor_level(level) for level in self.floor_levels
            ],
        }
        with floor_metadata_path.open("w", encoding="utf-8") as file_obj:
            json.dump(floor_metadata_payload, file_obj, indent=2, ensure_ascii=False)

        self.logger.info(
            "Exported %d objects across %d categories: %s",
            int(objects_payload["total_objects"]),
            int(objects_payload["num_categories"]),
            objects_metadata_path,
        )

        targets = self._discover_targets()
        if self.target_category is None:
            self.logger.info(
                "Discovered %d MP3D target objects across %d target categories",
                len(targets),
                len(
                    {
                        self._normalize_category_name(target.category_name)
                        for target in targets
                    }
                ),
            )
        else:
            self.logger.info("Discovered %d '%s' objects", len(targets), self.target_label)

        goals: Dict[str, Dict[str, Dict[str, Any]]] = {}
        all_views: List[Dict[str, Any]] = []

        progress_desc = (
            f"Scanning {self.target_label.title()}s"
            if self.target_category is not None
            else "Scanning MP3D Targets"
        )
        progress_unit = self.target_category or "target"

        for target in tqdm(targets, desc=progress_desc, unit=progress_unit):
            category_output_dir = scene_output_dir / self._category_dirname(
                target.category_name
            )
            category_output_dir.mkdir(parents=True, exist_ok=True)

            object_output_dir = category_output_dir / str(target.semantic_id)
            object_output_dir.mkdir(parents=True, exist_ok=True)

            target_views, invalid_views = self._scan_target(target, object_output_dir)
            if len(target_views) > 3:
                target_views = target_views[:3]
            pose_only_views = [self._pose_only_view(view) for view in target_views]
            all_views.extend(pose_only_views)
            object_views_json = object_output_dir / "views.json"
            invalid_views_json = object_output_dir / "invalid_views.json"
            target_floor_level = self._nearest_floor_level(
                float(target.center[1] - 0.5 * target.sizes[1])
            )
            object_payload = {
                "object_id": int(target.semantic_id),
                "object_category": target.category_name,
                "object_center": [round(float(v), 3) for v in target.center],
                "object_sizes": [round(float(v), 3) for v in target.sizes],
                "horizontal_radius": round(float(target.horizontal_radius), 3),
                "object_base_y": round(
                    float(target.center[1] - 0.5 * target.sizes[1]),
                    3,
                ),
                "floor_index": (
                    int(target_floor_level.index)
                    if target_floor_level is not None
                    else None
                ),
                "floor_y": (
                    round(float(target_floor_level.y), 3)
                    if target_floor_level is not None
                    else None
                ),
                "num_views": len(target_views),
                "views": target_views,
            }
            with object_views_json.open("w", encoding="utf-8") as file_obj:
                json.dump(object_payload, file_obj, indent=2, ensure_ascii=False)

            invalid_payload = {
                "object_id": int(target.semantic_id),
                "object_category": target.category_name,
                "object_center": [round(float(v), 3) for v in target.center],
                "object_sizes": [round(float(v), 3) for v in target.sizes],
                "horizontal_radius": round(float(target.horizontal_radius), 3),
                "num_invalid_views": len(invalid_views),
                "invalid_views": invalid_views,
            }
            with invalid_views_json.open("w", encoding="utf-8") as file_obj:
                json.dump(invalid_payload, file_obj, indent=2, ensure_ascii=False)

            if pose_only_views:
                category_key = self._category_dirname(target.category_name)
                category_goals = goals.setdefault(category_key, {})
                category_goals[str(target.semantic_id)] = {
                    "object_id": int(target.semantic_id),
                    "object_category": target.category_name,
                    "output_dir": object_output_dir.relative_to(self.output_root).as_posix(),
                    "views_json": object_views_json.relative_to(self.output_root).as_posix(),
                    "invalid_views_json": invalid_views_json.relative_to(self.output_root).as_posix(),
                    "render_views": pose_only_views,
                    "num_views": len(target_views),
                    "num_invalid_views": len(invalid_views),
                }

        output_json = (
            self.args.output_json.expanduser().resolve()
            if self.args.output_json is not None
            else scene_output_dir / "imagenav_eval_episodes.json"
        )
        output_json.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "episodes": [],
            "goals": goals,
            "views": all_views,
            "metadata": {
                "scene_name": self.scene_name,
                "scene_id": self.scene_id,
                "target_category": self.target_label,
                "target_categories": self._active_target_categories(),
                "manifest_format": "render_poses_only",
                "render_sensor": {
                    "width": int(self.args.width),
                    "height": int(self.args.height),
                    "hfov": float(self.args.hfov),
                    "sensor_height": float(self.args.sensor_height),
                },
                "scanner": "deterministic_open_floor_search",
                "search_radius": float(self.args.search_radius),
                "min_surface_offset": float(self.args.min_surface_offset),
                "min_iou": float(self.args.min_iou),
                "yolo_validation_enabled": bool(self.yolo_model is not None),
                "yolo_model": (
                    str(self.args.yolo_model.expanduser().resolve())
                    if self.args.yolo_model is not None
                    else None
                ),
                "yolo_conf_threshold": float(self.args.yolo_conf_threshold),
                "yolo_iou_threshold": float(self.args.yolo_iou_threshold),
                "yolo_max_det": int(self.args.yolo_max_det),
                "candidate_angle_step_deg": int(self.args.candidate_angle_step),
                "radial_step": float(self.args.radial_step),
                "radial_step_jitter": float(self.args.radial_step_jitter),
                "angle_jitter_deg": float(self.args.angle_jitter),
                "min_edge_clearance": float(self.args.min_edge_clearance),
                "nearby_object_radius": float(self.args.nearby_object_radius),
                "max_viewpoints": int(self.args.max_viewpoints),
                "min_sensor_offset": float(self.args.min_sensor_offset),
                "max_sensor_offset": float(self.args.max_sensor_offset),
                "sensor_offset_step": float(self.args.sensor_offset_step),
                "max_floor_levels": int(self.args.max_floor_levels),
                "floor_level_tolerance": float(self.args.floor_level_tolerance),
                "num_navmesh_height_bands": len(self.all_floor_levels),
                "num_floor_levels": len(self.floor_levels),
                "all_navmesh_height_bands": [
                    self._serialize_floor_level(level) for level in self.all_floor_levels
                ],
                "dominant_floor_levels": [
                    self._serialize_floor_level(level) for level in self.floor_levels
                ],
                "objects_json": objects_metadata_path.relative_to(
                    self.output_root
                ).as_posix(),
                "floor_metadata_json": floor_metadata_path.relative_to(
                    self.output_root
                ).as_posix(),
                "num_targets": len(targets),
                "num_saved_views": len(all_views),
            },
        }

        with output_json.open("w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, indent=2, ensure_ascii=False)

        self.logger.info("Saved %d valid goal views", len(all_views))
        self.logger.info("Wrote manifest JSON: %s", output_json)
        return output_json

    def _resolve_scene_path(self, scene_name: str) -> Path:
        """Resolve a scene name to an existing MP3D `.glb` path."""
        candidate = Path(scene_name)
        if candidate.suffix == ".glb":
            if candidate.is_absolute() and candidate.is_file():
                return candidate
            local_candidate = (self.scene_dir / candidate).resolve()
            if local_candidate.is_file():
                return local_candidate

        direct = (self.scene_dir / scene_name / f"{scene_name}.glb").resolve()
        if direct.is_file():
            return direct

        recursive = sorted(self.scene_dir.glob(f"**/{scene_name}.glb"))
        if recursive:
            return recursive[0].resolve()

        raise RuntimeError(
            f"Failed to resolve scene '{scene_name}' under {self.scene_dir}"
        )

    def _scene_id_for_dataset(self, scene_path: Path) -> str:
        """Build MP3D-style scene id for metadata compatibility."""
        try:
            relative = scene_path.relative_to(self.scene_dir).as_posix()
        except Exception:
            relative = scene_path.name

        if relative.startswith("mp3d/"):
            return relative
        return f"mp3d/{relative}"

    def _build_simulator(self, scene_path: Path):
        """Create `ContinuousSoundSpacesSim` with RGB and semantic sensors."""
        cfg = get_task_config(config_paths=[str(self.args.exp_config)])
        cfg.defrost()
        cfg.SIMULATOR.TYPE = "ContinuousSoundSpacesSim"
        cfg.SIMULATOR.SCENE = str(scene_path)
        cfg.SIMULATOR.SCENE_DATASET = str(self.args.scene_dataset_config)
        cfg.SIMULATOR.AUDIO.ENABLED = False

        cfg.SIMULATOR.RGB_SENSOR.WIDTH = int(self.args.width)
        cfg.SIMULATOR.RGB_SENSOR.HEIGHT = int(self.args.height)
        cfg.SIMULATOR.SEMANTIC_SENSOR.WIDTH = int(self.args.width)
        cfg.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = int(self.args.height)

        if hasattr(cfg.SIMULATOR.RGB_SENSOR, "HFOV"):
            cfg.SIMULATOR.RGB_SENSOR.HFOV = float(self.args.hfov)
        if hasattr(cfg.SIMULATOR.SEMANTIC_SENSOR, "HFOV"):
            cfg.SIMULATOR.SEMANTIC_SENSOR.HFOV = float(self.args.hfov)

        if hasattr(cfg.SIMULATOR.RGB_SENSOR, "POSITION"):
            cfg.SIMULATOR.RGB_SENSOR.POSITION = [0.0, self.args.sensor_height, 0.0]
        if hasattr(cfg.SIMULATOR.SEMANTIC_SENSOR, "POSITION"):
            cfg.SIMULATOR.SEMANTIC_SENSOR.POSITION = [0.0, self.args.sensor_height, 0.0]

        sensors = list(getattr(cfg.SIMULATOR.AGENT_0, "SENSORS", []))
        for sensor_name in ("RGB_SENSOR", "SEMANTIC_SENSOR"):
            if sensor_name not in sensors:
                sensors.append(sensor_name)
        cfg.SIMULATOR.AGENT_0.SENSORS = sensors

        cfg.freeze()
        sim = make_sim(cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR)
        sim.reset()
        return sim

    def _get_sensor_uuid(self, sensor_type: str) -> str:
        """Return configured sensor UUID for RGB or semantic streams."""
        if sensor_type == "rgb" and hasattr(self.sim, "_get_rgb_uuid"):
            rgb_uuid = self.sim._get_rgb_uuid()
            if rgb_uuid is not None:
                return rgb_uuid
        if sensor_type == "semantic" and hasattr(self.sim, "_get_semantic_uuid"):
            semantic_uuid = self.sim._get_semantic_uuid()
            if semantic_uuid is not None:
                return semantic_uuid
        return "rgb" if sensor_type == "rgb" else "semantic"

    def _semantic_objects(self) -> List[Any]:
        """Get semantic objects from simulator scene graph."""
        if hasattr(self.sim, "_semantic_objects"):
            try:
                return list(self.sim._semantic_objects())
            except Exception:
                pass

        semantic_scene = getattr(self.inner_sim, "semantic_scene", None)
        if semantic_scene is None:
            return []
        try:
            return list(semantic_scene.objects)
        except Exception:
            return []

    def _safe_semantic_id(self, obj: Any) -> Optional[int]:
        """Extract semantic id from semantic object metadata."""
        for attr_name in ("semantic_id", "semanticID"):
            value = getattr(obj, attr_name, None)
            if value is None:
                continue
            try:
                return int(value)
            except Exception:
                continue
        return None

    def _safe_category_name(self, obj: Any) -> str:
        """Extract category name from semantic object metadata."""
        category = getattr(obj, "category", None)
        if category is None:
            return ""
        try:
            return str(category.name())
        except Exception:
            return ""

    def _safe_object_id_raw(self, obj: Any) -> Optional[str]:
        """Extract raw scene object id as a stable string when available."""
        for attr_name in ("id", "object_id", "objectID"):
            value = getattr(obj, attr_name, None)
            if value is None:
                continue
            try:
                return str(int(value))
            except Exception:
                try:
                    return str(value)
                except Exception:
                    continue
        return None

    def _safe_object_name(self, obj: Any) -> Optional[str]:
        """Extract human-readable object name when available."""
        for attr_name in ("name", "label"):
            value = getattr(obj, attr_name, None)
            if value is None:
                continue
            try:
                text = str(value)
            except Exception:
                continue
            if text:
                return text
        return None

    def _safe_int(self, value: Any) -> Optional[int]:
        """Best-effort integer conversion helper."""
        if value is None:
            return None
        try:
            return int(value)
        except Exception:
            return None

    def _safe_str(self, value: Any) -> Optional[str]:
        """Best-effort string conversion helper."""
        if value is None:
            return None
        try:
            text = str(value)
        except Exception:
            return None
        return text if text else None

    def _safe_room_info(self, obj: Any) -> Tuple[Optional[str], Optional[str], Optional[int]]:
        """Extract room/region/level metadata from a semantic object when available."""
        room_id = None
        room_name = None
        level_id = None

        for attr_name in ("room_id", "region_id"):
            room_id = self._safe_str(getattr(obj, attr_name, None))
            if room_id is not None:
                break

        room_name = self._safe_str(getattr(obj, "room_name", None))
        level_id = self._safe_int(getattr(obj, "level_id", None))

        region = getattr(obj, "region", None)
        if region is not None:
            if room_id is None:
                for attr_name in ("id", "region_id", "regionID"):
                    room_id = self._safe_str(getattr(region, attr_name, None))
                    if room_id is not None:
                        break

            if room_name is None:
                category = getattr(region, "category", None)
                if category is not None:
                    try:
                        room_name = str(category.name())
                    except Exception:
                        room_name = None
                if room_name is None:
                    for attr_name in ("name", "label"):
                        room_name = self._safe_str(getattr(region, attr_name, None))
                        if room_name is not None:
                            break

            if level_id is None:
                level = getattr(region, "level", None)
                if level is not None:
                    for attr_name in ("id", "level_id", "levelID"):
                        level_id = self._safe_int(getattr(level, attr_name, None))
                        if level_id is not None:
                            break

        return room_id, room_name, level_id

    def _collect_scene_objects(self) -> List[SemanticTarget]:
        """Collect all semantic objects with valid AABBs for collision filtering."""
        scene_objects: List[SemanticTarget] = []
        seen_ids: set[int] = set()

        for obj in self._semantic_objects():
            semantic_id = self._safe_semantic_id(obj)
            if semantic_id is None or semantic_id in seen_ids:
                continue

            aabb = getattr(obj, "aabb", None)
            if aabb is None:
                continue

            center = np.array(aabb.center, dtype=np.float32)
            sizes = np.array(aabb.sizes, dtype=np.float32)
            if sizes.size < 3 or not np.all(np.isfinite(sizes)):
                continue

            horizontal_radius = 0.5 * float(max(sizes[0], sizes[2]))
            room_id, room_name, level_id = self._safe_room_info(obj)
            scene_objects.append(
                SemanticTarget(
                    semantic_id=int(semantic_id),
                    category_name=self._safe_category_name(obj),
                    center=center,
                    sizes=sizes,
                    horizontal_radius=horizontal_radius,
                    object_id_raw=self._safe_object_id_raw(obj),
                    object_name=self._safe_object_name(obj),
                    room_id=room_id,
                    room_name=room_name,
                    level_id=level_id,
                )
            )
            seen_ids.add(int(semantic_id))

        scene_objects.sort(key=lambda obj: obj.semantic_id)
        return scene_objects

    def _discover_targets(self) -> List[SemanticTarget]:
        """Collect all semantic objects matching the current target selection."""
        return [
            obj for obj in self.scene_objects if self._matches_requested_target(obj.category_name)
        ]

    def _collect_navmesh_triangle_vertices(self) -> np.ndarray:
        """Collect raw navmesh triangle vertices from the pathfinder."""
        if self.pathfinder is None or not hasattr(self.pathfinder, "build_navmesh_vertices"):
            return np.zeros((0, 3), dtype=np.float32)

        try:
            vertices = np.array(list(self.pathfinder.build_navmesh_vertices()), dtype=np.float32)
        except Exception:
            return np.zeros((0, 3), dtype=np.float32)

        if vertices.ndim != 2 or vertices.shape[1] != 3:
            return np.zeros((0, 3), dtype=np.float32)
        vertices = vertices[np.all(np.isfinite(vertices), axis=1)]
        return vertices

    def _collect_navmesh_vertices(self, raw_vertices: np.ndarray) -> np.ndarray:
        """Collect unique navmesh vertices for deterministic floor candidate search."""
        vertices = np.array(raw_vertices, dtype=np.float32)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
            return np.zeros((0, 3), dtype=np.float32)

        rounded = np.round(vertices, 3)
        _, unique_indices = np.unique(rounded, axis=0, return_index=True)
        return vertices[np.sort(unique_indices)]

    def _triangle_area_3d(self, triangle: np.ndarray) -> float:
        """Compute 3D area of a navmesh triangle."""
        edge_a = np.array(triangle[1], dtype=np.float32) - np.array(triangle[0], dtype=np.float32)
        edge_b = np.array(triangle[2], dtype=np.float32) - np.array(triangle[0], dtype=np.float32)
        return 0.5 * float(np.linalg.norm(np.cross(edge_a, edge_b)))

    def _triangle_area_xz(self, triangle: np.ndarray) -> float:
        """Compute horizontal projected area of a triangle on the X-Z plane."""
        p0 = np.array(triangle[0], dtype=np.float32)[[0, 2]]
        p1 = np.array(triangle[1], dtype=np.float32)[[0, 2]]
        p2 = np.array(triangle[2], dtype=np.float32)[[0, 2]]
        return 0.5 * abs(
            float((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]))
        )

    def _discover_floor_levels(self) -> List[FloorLevel]:
        """Cluster all navmesh triangles by height and accumulate their areas."""
        raw_vertices = np.array(self.navmesh_triangle_vertices, dtype=np.float32)
        if raw_vertices.ndim != 2 or raw_vertices.shape[1] != 3 or len(raw_vertices) < 3:
            return []
        triangle_count = len(raw_vertices) // 3
        if triangle_count <= 0:
            return []

        triangles = raw_vertices[: triangle_count * 3].reshape(triangle_count, 3, 3)
        triangle_centers_y = np.mean(triangles[:, :, 1], axis=1)
        sort_indices = np.argsort(triangle_centers_y)
        sorted_triangles = triangles[sort_indices]
        sorted_y = triangle_centers_y[sort_indices]

        tolerance = max(1e-3, float(self.args.floor_level_tolerance))
        clusters: List[List[Tuple[float, np.ndarray]]] = []
        current_cluster: List[Tuple[float, np.ndarray]] = [
            (float(sorted_y[0]), np.array(sorted_triangles[0], dtype=np.float32))
        ]

        for value, triangle in zip(sorted_y[1:], sorted_triangles[1:]):
            y_value = float(value)
            current_min = min(item[0] for item in current_cluster)
            current_max = max(item[0] for item in current_cluster)
            next_min = min(current_min, y_value)
            next_max = max(current_max, y_value)
            if (next_max - next_min) <= tolerance:
                current_cluster.append((y_value, np.array(triangle, dtype=np.float32)))
            else:
                clusters.append(current_cluster)
                current_cluster = [(y_value, np.array(triangle, dtype=np.float32))]
        clusters.append(current_cluster)

        floor_levels = [
            FloorLevel(
                y=float(np.mean([item[0] for item in cluster], dtype=np.float32)),
                min_y=float(min(item[0] for item in cluster)),
                max_y=float(max(item[0] for item in cluster)),
                count=len(cluster),
                area_m2=float(
                    sum(self._triangle_area_3d(item[1]) for item in cluster)
                ),
                projected_area_m2=float(
                    sum(self._triangle_area_xz(item[1]) for item in cluster)
                ),
            )
            for cluster in clusters
            if cluster
        ]
        floor_levels.sort(key=lambda level: level.y)
        for idx, floor_level in enumerate(floor_levels):
            floor_level.index = idx
        return floor_levels

    def _select_dominant_floor_levels(
        self,
        floor_levels: List[FloorLevel],
    ) -> List[FloorLevel]:
        """Keep only the largest floor-like height bands for viewpoint sampling."""
        if not floor_levels:
            return []
        dominant = sorted(
            floor_levels,
            key=lambda level: (-level.projected_area_m2, -level.area_m2, level.y),
        )[: max(1, int(self.args.max_floor_levels))]
        dominant.sort(key=lambda level: level.y)
        return dominant

    def _nearest_floor_level(self, y_value: float) -> Optional[FloorLevel]:
        """Return the dominant floor level closest to a y-coordinate."""
        if not self.floor_levels:
            return None
        return min(self.floor_levels, key=lambda level: abs(level.y - float(y_value)))

    def _floor_level_index(self, floor_level: Optional[FloorLevel]) -> Optional[int]:
        """Return stable floor index for a floor level."""
        if floor_level is None:
            return None
        return int(floor_level.index)

    def _matches_floor_level(self, y_value: float, floor_level: Optional[FloorLevel]) -> bool:
        """Check whether a point height belongs to a chosen dominant floor level."""
        if floor_level is None:
            return True
        return abs(float(y_value) - float(floor_level.y)) <= float(
            self.args.floor_height_tolerance
        )

    def _snap_to_navmesh(self, point: np.ndarray) -> np.ndarray:
        """Project point to nearest navigable point."""
        snapped = self.pathfinder.snap_point(point)
        return np.array(snapped, dtype=np.float32)

    def _sensor_world_position(self, base_position: np.ndarray) -> np.ndarray:
        """Return sensor world position from agent base pose."""
        sensor_position = np.array(base_position, dtype=np.float32)
        sensor_position[1] += self.sensor_height
        return sensor_position

    def _look_at_quaternion(
        self,
        base_position: np.ndarray,
        target: np.ndarray,
    ) -> Any:
        """Compute a roll-free look-at quaternion from sensor origin to target center."""
        sensor_origin = self._sensor_world_position(base_position)
        forward = np.array(target, dtype=np.float32) - sensor_origin
        horizontal_only = bool(getattr(self.args, "horizontal_only_rotation", True))
        if horizontal_only:
            forward[1] = 0.0
            horizontal_norm = float(np.linalg.norm(forward))
            if horizontal_norm <= 1e-6:
                forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        forward_norm = float(np.linalg.norm(forward))
        if forward_norm <= 1e-6:
            return quat_from_angle_axis(
                0.0,
                np.array([0.0, 1.0, 0.0], dtype=np.float32),
            )
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

        rotation_matrix = np.stack(
            [right, up, -forward],
            axis=1,
        ).astype(np.float32)
        return self._rotation_matrix_to_quaternion(rotation_matrix)

    def _rotation_matrix_to_quaternion(self, rotation_matrix: np.ndarray) -> Any:
        """Convert a 3x3 rotation matrix into a quaternion understood by Habitat."""
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

    def _compute_largest_navmesh_island_radius(self) -> Optional[float]:
        """Return the largest reachable navmesh island radius in the scene."""
        if self.pathfinder is None:
            return None
        if not hasattr(self.pathfinder, "build_navmesh_vertices"):
            return None
        if not hasattr(self.pathfinder, "island_radius"):
            return None

        try:
            navmesh_vertices = list(self.pathfinder.build_navmesh_vertices())
        except Exception:
            return None
        if not navmesh_vertices:
            return None

        island_sizes: List[float] = []
        for vertex in navmesh_vertices:
            try:
                island_radius = float(self.pathfinder.island_radius(vertex))
            except Exception:
                continue
            if math.isfinite(island_radius):
                island_sizes.append(island_radius)
        if not island_sizes:
            return None
        return max(island_sizes)

    def _distance_to_navmesh_edge(self, point: np.ndarray) -> float:
        """Estimate clearance from a floor point to the nearest navmesh boundary."""
        if self.pathfinder is None:
            return 0.0
        if not hasattr(self.pathfinder, "distance_to_closest_obstacle"):
            return 0.0
        try:
            clearance = float(self.pathfinder.distance_to_closest_obstacle(point))
        except Exception:
            return 0.0
        if not math.isfinite(clearance):
            return 0.0
        return max(0.0, clearance)

    def _is_valid_floor_navmesh_point(
        self,
        point: np.ndarray,
        floor_level: Optional[FloorLevel] = None,
    ) -> bool:
        """Check that a point is navigable and belongs to a dominant floor level."""
        if self.pathfinder is None:
            return False
        if hasattr(self.pathfinder, "is_navigable"):
            try:
                if not bool(self.pathfinder.is_navigable(point)):
                    return False
            except Exception:
                return False
        return self._matches_floor_level(float(point[1]), floor_level)

    def _snap_to_floor_navmesh(
        self,
        point: np.ndarray,
        floor_level: Optional[FloorLevel] = None,
    ) -> Optional[np.ndarray]:
        """Snap a point to navigable space on a dominant floor level."""
        if self.pathfinder is None or not hasattr(self.pathfinder, "snap_point"):
            return None
        try:
            snapped = np.array(self.pathfinder.snap_point(point), dtype=np.float32)
        except Exception:
            return None
        if snapped.shape[0] != 3 or not np.all(np.isfinite(snapped)):
            return None
        snap_xz_error = float(np.linalg.norm((snapped - point)[[0, 2]]))
        if snap_xz_error > float(self.args.max_snap_distance):
            return None
        if not self._is_valid_floor_navmesh_point(snapped, floor_level=floor_level):
            return None
        return snapped

    def _target_floor_y(self, target: SemanticTarget) -> float:
        """Approximate the target's supporting dominant floor height."""
        object_base_y = float(target.center[1] - 0.5 * target.sizes[1])
        floor_level = self._nearest_floor_level(object_base_y)
        if floor_level is None:
            return object_base_y
        return float(floor_level.y)

    def _point_to_aabb_distance_2d(
        self,
        point: np.ndarray,
        obj: SemanticTarget,
    ) -> float:
        """Compute shortest x-z distance from a point to an object's AABB footprint."""
        dx = abs(float(point[0]) - float(obj.center[0])) - 0.5 * float(obj.sizes[0])
        dz = abs(float(point[2]) - float(obj.center[2])) - 0.5 * float(obj.sizes[2])
        outside = np.maximum(np.array([dx, dz], dtype=np.float32), 0.0)
        return float(np.linalg.norm(outside))

    def _point_to_aabb_distance(
        self,
        point: np.ndarray,
        obj: SemanticTarget,
    ) -> float:
        """Compute shortest 3D distance from a point to an object's AABB."""
        half_sizes = 0.5 * np.array(obj.sizes, dtype=np.float32)
        delta = np.abs(np.array(point, dtype=np.float32) - np.array(obj.center, dtype=np.float32)) - half_sizes
        outside = np.maximum(delta, 0.0)
        return float(np.linalg.norm(outside))

    def _is_clear_of_scene_objects(
        self,
        point: np.ndarray,
        target_id: Optional[int] = None,
    ) -> Tuple[bool, Optional[SemanticTarget], Optional[float]]:
        """Check that a point is not inside or too close to any semantic object."""
        clearance = float(self.args.object_clearance)
        for obj in self.scene_objects:
            category_name = obj.category_name.strip().lower()
            if category_name in self.clearance_ignore_categories:
                continue
            if target_id is not None and int(obj.semantic_id) == int(target_id):
                continue
            dist = self._point_to_aabb_distance(point, obj)
            if dist < clearance:
                return False, obj, dist
        return True, None, None

    def _nearby_object_stats_2d(
        self,
        floor_point: np.ndarray,
        target_id: Optional[int] = None,
    ) -> Tuple[int, float]:
        """Count and penalize nearby non-floor objects around a floor viewpoint."""
        nearby_radius = float(self.args.nearby_object_radius)
        nearby_count = 0
        nearby_penalty = 0.0

        for obj in self.scene_objects:
            category_name = obj.category_name.strip().lower()
            if category_name in self.nearby_ignore_categories:
                continue
            if target_id is not None and int(obj.semantic_id) == int(target_id):
                continue

            dist_2d = self._point_to_aabb_distance_2d(floor_point, obj)
            if dist_2d > nearby_radius:
                continue
            nearby_count += 1
            nearby_penalty += 1.0 / max(dist_2d, 0.1)

        return nearby_count, nearby_penalty

    def _min_sensor_clearance(
        self,
        sensor_position: np.ndarray,
        target_id: Optional[int] = None,
    ) -> float:
        """Compute minimum 3D clearance from a sensor point to nearby semantic objects."""
        min_clearance = float("inf")
        for obj in self.scene_objects:
            category_name = obj.category_name.strip().lower()
            if category_name in self.clearance_ignore_categories:
                continue
            if target_id is not None and int(obj.semantic_id) == int(target_id):
                continue
            min_clearance = min(
                min_clearance,
                self._point_to_aabb_distance(sensor_position, obj),
            )
        if not math.isfinite(min_clearance):
            return 999.0
        return float(min_clearance)

    def _sensor_offset_candidates(self) -> List[float]:
        """Return the physically valid sensor offset above the agent base.

        In Habitat-Sim, the sensor pose is configured relative to the agent body.
        To preserve the default agent viewpoint, the body must stay on the floor
        and the world sensor height is always `base_y + sensor_height`.
        """
        return [float(self.sensor_height)]

    def _candidate_rng(self, target: SemanticTarget) -> np.random.RandomState:
        """Return a deterministic RNG for candidate jitter per target."""
        seed_payload = f"{self.scene_name}:{int(target.semantic_id)}".encode("utf-8")
        seed = zlib.crc32(seed_payload) & 0xFFFFFFFF
        return np.random.RandomState(seed)

    def _estimate_angle_deg(
        self,
        floor_position: np.ndarray,
        target: SemanticTarget,
    ) -> int:
        """Estimate x-z viewing angle around the target center."""
        delta = np.array(floor_position, dtype=np.float32) - np.array(target.center, dtype=np.float32)
        angle_deg = math.degrees(math.atan2(float(delta[2]), float(delta[0]))) % 360.0
        return int(round(angle_deg)) % 360

    def _build_navmesh_ring_candidates(self, target: SemanticTarget) -> np.ndarray:
        """Build ring-based floor candidates with radial and angular jitter."""
        floor_y = self._target_floor_y(target)
        floor_level = self._nearest_floor_level(floor_y)
        candidates: List[np.ndarray] = []
        min_radius = float(self.args.min_surface_offset) + float(target.horizontal_radius)
        max_radius = float(self.args.search_radius) + float(target.horizontal_radius)
        radial_step = max(1e-6, float(self.args.radial_step))
        radial_jitter = max(0.0, float(self.args.radial_step_jitter))
        angle_jitter = max(0.0, float(self.args.angle_jitter))
        angles_deg = list(range(0, 360, int(self.args.candidate_angle_step)))
        rng = self._candidate_rng(target)
        for base_angle in angles_deg:
            radius = min_radius
            while radius <= max_radius + 1e-6:
                jittered_angle = float(base_angle) + rng.uniform(-angle_jitter, angle_jitter)
                angle_rad = math.radians(jittered_angle)
                probe = np.array(
                    [
                        target.center[0] + float(radius) * math.cos(angle_rad),
                        floor_y,
                        target.center[2] + float(radius) * math.sin(angle_rad),
                    ],
                    dtype=np.float32,
                )
                snapped = self._snap_to_floor_navmesh(probe, floor_level=floor_level)
                if snapped is not None:
                    candidates.append(snapped)
                step = radial_step
                if radial_jitter > 0.0:
                    step += rng.uniform(-radial_jitter, radial_jitter)
                    step = max(step, 1e-6)
                radius += step
        if not candidates:
            return np.zeros((0, 3), dtype=np.float32)
        rounded = np.round(np.array(candidates, dtype=np.float32), 3)
        _, unique_indices = np.unique(rounded, axis=0, return_index=True)
        return np.array(candidates, dtype=np.float32)[np.sort(unique_indices)]

    def _rank_floor_candidates(
        self,
        target: SemanticTarget,
        min_edge_clearance: float,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
        """Rank candidate floor viewpoints around a target by openness."""
        floor_y = self._target_floor_y(target)
        floor_level = self._nearest_floor_level(floor_y)
        diagnostics: Dict[str, Any] = {
            "target_floor_y": round(float(floor_y), 3),
            "target_floor_index": self._floor_level_index(floor_level),
            "raw_candidates": 0,
            "y_mismatch": 0,
            "too_far": 0,
            "too_near": 0,
            "valid_mask_pass": 0,
            "invalid_floor_navmesh": 0,
            "edge_reject": 0,
            "ranked_floor": 0,
            "min_center_distance": 0.0,
            "max_center_distance": 0.0,
            "min_edge_clearance": round(float(min_edge_clearance), 3),
        }
        invalid_views: List[Dict[str, Any]] = []
        candidate_arrays: List[np.ndarray] = []
        candidate_sources: List[str] = []

        ring_candidates = self._build_navmesh_ring_candidates(target)
        if len(ring_candidates) > 0:
            candidate_arrays.append(ring_candidates)
            candidate_sources.extend(["ring_candidate"] * len(ring_candidates))

        if candidate_arrays:
            raw_candidates = np.concatenate(candidate_arrays, axis=0)
            rounded = np.round(raw_candidates, 3)
            _, unique_indices = np.unique(rounded, axis=0, return_index=True)
            unique_indices = np.sort(unique_indices)
            raw_candidates = raw_candidates[unique_indices]
            raw_sources = [candidate_sources[index] for index in unique_indices]
        else:
            raw_candidates = np.zeros((0, 3), dtype=np.float32)
            raw_sources = []

        diagnostics["raw_candidates"] = int(len(raw_candidates))

        if len(raw_candidates) == 0:
            return [], diagnostics, invalid_views

        floor_tolerance = float(self.args.floor_height_tolerance)
        delta_xz = raw_candidates[:, [0, 2]] - np.array(target.center[[0, 2]], dtype=np.float32)
        dist_2d = np.linalg.norm(delta_xz, axis=1)
        min_distance = float(target.horizontal_radius) + float(self.args.min_surface_offset)
        max_distance = float(target.horizontal_radius) + float(self.args.search_radius)
        diagnostics["min_center_distance"] = round(min_distance, 3)
        diagnostics["max_center_distance"] = round(max_distance, 3)

        y_diff = np.abs(raw_candidates[:, 1] - floor_y)
        diagnostics["y_mismatch"] = int(np.sum(y_diff > floor_tolerance))
        diagnostics["too_far"] = int(np.sum(dist_2d > max_distance))
        diagnostics["too_near"] = int(np.sum(dist_2d < min_distance))
        valid_mask = (
            y_diff <= floor_tolerance
        ) & (
            dist_2d <= max_distance
        ) & (
            dist_2d >= min_distance
        )
        diagnostics["valid_mask_pass"] = int(np.sum(valid_mask))

        ranked: List[Dict[str, Any]] = []

        for point, point_dist, point_y_diff, point_source in zip(
            raw_candidates,
            dist_2d,
            y_diff,
            raw_sources,
        ):
            common_details = {
                "center_distance": float(point_dist),
                "min_center_distance": float(min_distance),
                "max_center_distance": float(max_distance),
                "point_y": float(point[1]),
                "target_floor_y": float(floor_y),
                "y_diff": float(point_y_diff),
                "floor_height_tolerance": float(self.args.floor_height_tolerance),
            }
            if point_y_diff > floor_tolerance:
                invalid_views.append(
                    self._invalid_view_record(
                        target=target,
                        stage="floor_candidate",
                        reason="y_mismatch",
                        floor_position=point,
                        angle_deg=self._estimate_angle_deg(point, target),
                        surface_distance=max(0.0, float(point_dist) - float(target.horizontal_radius)),
                        candidate_source=point_source,
                        details=common_details,
                    )
                )
                continue
            if point_dist > max_distance:
                invalid_views.append(
                    self._invalid_view_record(
                        target=target,
                        stage="floor_candidate",
                        reason="too_far",
                        floor_position=point,
                        angle_deg=self._estimate_angle_deg(point, target),
                        surface_distance=max(0.0, float(point_dist) - float(target.horizontal_radius)),
                        candidate_source=point_source,
                        details=common_details,
                    )
                )
                continue
            if point_dist < min_distance:
                invalid_views.append(
                    self._invalid_view_record(
                        target=target,
                        stage="floor_candidate",
                        reason="too_near",
                        floor_position=point,
                        angle_deg=self._estimate_angle_deg(point, target),
                        surface_distance=max(0.0, float(point_dist) - float(target.horizontal_radius)),
                        candidate_source=point_source,
                        details=common_details,
                    )
                )
                continue
            if not self._is_valid_floor_navmesh_point(point, floor_level=floor_level):
                diagnostics["invalid_floor_navmesh"] += 1
                is_navigable = None
                if self.pathfinder is not None and hasattr(self.pathfinder, "is_navigable"):
                    try:
                        is_navigable = bool(self.pathfinder.is_navigable(point))
                    except Exception:
                        is_navigable = False
                matches_floor_level = self._matches_floor_level(
                    float(point[1]),
                    floor_level,
                )
                invalid_views.append(
                    self._invalid_view_record(
                        target=target,
                        stage="floor_candidate",
                        reason="invalid_floor_navmesh",
                        floor_position=point,
                        angle_deg=self._estimate_angle_deg(point, target),
                        surface_distance=max(0.0, float(point_dist) - float(target.horizontal_radius)),
                        candidate_source=point_source,
                        details={
                            **common_details,
                            "is_navigable": is_navigable,
                            "matches_floor_level": matches_floor_level,
                        },
                    )
                )
                continue
            edge_clearance = self._distance_to_navmesh_edge(point)
            if edge_clearance < min_edge_clearance:
                diagnostics["edge_reject"] += 1
                invalid_views.append(
                    self._invalid_view_record(
                        target=target,
                        stage="floor_candidate",
                        reason="edge_reject",
                        floor_position=point,
                        angle_deg=self._estimate_angle_deg(point, target),
                        surface_distance=max(0.0, float(point_dist) - float(target.horizontal_radius)),
                        candidate_source=point_source,
                        details={
                            **common_details,
                            "edge_clearance": float(edge_clearance),
                            "min_edge_clearance": float(min_edge_clearance),
                        },
                    )
                )
                continue
            ranked.append(
                {
                    "floor_position": np.array(point, dtype=np.float32),
                    "edge_clearance": float(edge_clearance),
                    "nearby_count": 0,
                    "nearby_penalty": 0.0,
                    "surface_distance": max(0.0, float(point_dist) - float(target.horizontal_radius)),
                    "angle_deg": self._estimate_angle_deg(point, target),
                    "candidate_source": point_source,
                }
            )

        ranked.sort(
            key=lambda item: (
                -item["edge_clearance"],
                -item["surface_distance"],
                item["angle_deg"],
            )
        )
        diagnostics["ranked_floor"] = int(len(ranked))
        return ranked, diagnostics, invalid_views

    def _select_sensor_pose(
        self,
        floor_position: np.ndarray,
        target: SemanticTarget,
        floor_stats: Dict[str, Any],
    ) -> Optional[ViewpointCandidate]:
        """Choose the most open sensor height above a valid floor viewpoint."""
        best_candidate: Optional[ViewpointCandidate] = None
        floor_level = self._nearest_floor_level(float(floor_position[1]))
        floor_level_y = (
            float(floor_level.y)
            if floor_level is not None
            else float(floor_position[1])
        )

        for sensor_offset in self._sensor_offset_candidates():
            base_position = np.array(floor_position, dtype=np.float32)
            sensor_position = self._sensor_world_position(base_position)
            sensor_clearance = self._min_sensor_clearance(
                sensor_position,
                target_id=target.semantic_id,
            )

            rotation = self._look_at_quaternion(base_position, target.center)
            candidate = ViewpointCandidate(
                floor_position=np.array(floor_position, dtype=np.float32),
                sensor_position=sensor_position,
                base_position=base_position,
                rotation=rotation,
                surface_distance=float(floor_stats["surface_distance"]),
                angle_deg=int(floor_stats["angle_deg"]),
                edge_clearance=float(floor_stats["edge_clearance"]),
                nearby_count=int(floor_stats["nearby_count"]),
                nearby_penalty=float(floor_stats["nearby_penalty"]),
                sensor_clearance=float(sensor_clearance),
                sensor_offset=float(sensor_offset),
                floor_level_index=self._floor_level_index(floor_level),
                floor_level_y=float(floor_level_y),
            )
            if best_candidate is None:
                best_candidate = candidate
                continue

            if candidate.sensor_clearance > best_candidate.sensor_clearance + 1e-6:
                best_candidate = candidate
            elif abs(candidate.sensor_clearance - best_candidate.sensor_clearance) <= 1e-6:
                if candidate.sensor_offset > best_candidate.sensor_offset:
                    best_candidate = candidate

        return best_candidate

    def _is_diverse_viewpoint(
        self,
        candidate: ViewpointCandidate,
        selected: List[ViewpointCandidate],
    ) -> bool:
        """Enforce angular and spatial diversity across selected viewpoints."""
        for chosen in selected:
            angle_diff = abs(candidate.angle_deg - chosen.angle_deg) % 360
            angle_diff = min(angle_diff, 360 - angle_diff)
            if angle_diff < int(self.args.min_viewpoint_angle_sep):
                return False
            dist_2d = float(
                np.linalg.norm(
                    (candidate.floor_position - chosen.floor_position)[[0, 2]]
                )
            )
            if dist_2d < float(self.args.min_viewpoint_separation):
                return False
        return True

    def _direction_adjustments(self) -> List[float]:
        """Return radial forward/backward adjustments along one sampling direction."""
        max_adjust = float(self.args.position_adjust_max)
        step = float(self.args.position_adjust_step)
        adjustments = [0.0]
        if step <= 0.0 or max_adjust <= 0.0:
            return adjustments

        steps = int(math.floor(max_adjust / step + 1e-6))
        for idx in range(1, steps + 1):
            delta = round(idx * step, 6)
            adjustments.append(delta)
            adjustments.append(-delta)
        if adjustments[-2] != max_adjust:
            adjustments.append(max_adjust)
            adjustments.append(-max_adjust)
        return adjustments

    def _has_line_of_sight(
        self,
        sensor_position: np.ndarray,
        target: SemanticTarget,
    ) -> bool:
        """Check whether a ray from sensor to target center reaches object surface."""
        if not hasattr(self.inner_sim, "cast_ray"):
            return True

        direction = np.array(target.center, dtype=np.float32) - np.array(
            sensor_position,
            dtype=np.float32,
        )
        ray_distance = float(np.linalg.norm(direction))
        if ray_distance <= 1e-6:
            return True

        surface_distance = max(0.0, ray_distance - float(target.horizontal_radius))
        if surface_distance <= 1e-6:
            return True

        ray = habitat_sim.geo.Ray(sensor_position, direction / ray_distance)
        try:
            hits = self.inner_sim.cast_ray(ray).hits
        except Exception:
            return False

        if len(hits) == 0:
            return False

        first_hit_distance = float(getattr(hits[0], "ray_distance", float("inf")))
        return first_hit_distance >= (surface_distance - 0.05)

    def _observe(
        self,
        base_position: np.ndarray,
        rotation: Any,
        target_id: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Render RGB and semantic observations at the given agent state."""
        self.sim.set_agent_state(base_position, rotation, reset_sensors=False)
        obs = self.inner_sim.get_sensor_observations()

        rgb = obs.get(self.rgb_uuid)
        semantic = obs.get(self.semantic_uuid)
        if rgb is None or semantic is None:
            raise RuntimeError("Missing RGB or semantic observations.")

        rgb = np.asarray(rgb)
        if rgb.ndim != 3:
            raise RuntimeError(f"Unexpected RGB observation shape: {rgb.shape}")
        if rgb.shape[2] == 4:
            rgb = rgb[:, :, :3]
        elif rgb.shape[2] == 1:
            rgb = np.repeat(rgb, 3, axis=2)
        elif rgb.shape[2] != 3:
            raise RuntimeError(f"Unsupported RGB channel count: {rgb.shape[2]}")
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        rgb = np.ascontiguousarray(rgb)

        if semantic.ndim == 3:
            semantic = semantic[:, :, 0]
        if target_id is not None:
            unique_ids = np.unique(semantic.astype(np.int64, copy=False))
            unique_ids_list = [int(value) for value in unique_ids.tolist()]
            self.logger.debug(
                "[ID Debug] Target: %s | Unique IDs in frame: %s.",
                str(int(target_id)),
                unique_ids_list,
            )
        return rgb, semantic

    def _image_center_pixel(self) -> Tuple[float, float]:
        """Return the image center pixel in `(x, y)` order."""
        return (
            0.5 * (float(self.args.width) - 1.0),
            0.5 * (float(self.args.height) - 1.0),
        )

    def _camera_intrinsics(self) -> Tuple[float, float, float, float]:
        """Return pinhole intrinsics `(fx, fy, cx, cy)` from configured FoV."""
        width = float(self.args.width)
        height = float(self.args.height)
        hfov_rad = math.radians(float(self.args.hfov))
        fx = 0.5 * width / math.tan(0.5 * hfov_rad)
        vfov_rad = 2.0 * math.atan(math.tan(0.5 * hfov_rad) * (height / width))
        fy = 0.5 * height / math.tan(0.5 * vfov_rad)
        cx, cy = self._image_center_pixel()
        return fx, fy, cx, cy

    def _camera_axes(
        self,
        rotation: Any,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build camera right/up/forward axes from the exact render quaternion."""
        right = quaternion_rotate_vector(
            rotation,
            np.array([1.0, 0.0, 0.0], dtype=np.float32),
        )
        up = quaternion_rotate_vector(
            rotation,
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        forward = quaternion_rotate_vector(
            rotation,
            np.array([0.0, 0.0, -1.0], dtype=np.float32),
        )

        right = np.array(right, dtype=np.float32)
        up = np.array(up, dtype=np.float32)
        forward = np.array(forward, dtype=np.float32)

        right /= max(float(np.linalg.norm(right)), 1e-8)
        up /= max(float(np.linalg.norm(up)), 1e-8)
        forward /= max(float(np.linalg.norm(forward)), 1e-8)
        return right, up, forward

    def _forward_alignment_deg(
        self,
        sensor_position: np.ndarray,
        rotation: Any,
        target: np.ndarray,
    ) -> float:
        """Return the angle between render forward and target direction in degrees."""
        _, _, forward = self._camera_axes(rotation)
        target_direction = np.array(target, dtype=np.float32) - np.array(
            sensor_position,
            dtype=np.float32,
        )
        target_norm = float(np.linalg.norm(target_direction))
        if target_norm <= 1e-6:
            return 0.0
        target_direction /= target_norm
        cosine = float(np.clip(np.dot(forward, target_direction), -1.0, 1.0))
        return math.degrees(math.acos(cosine))

    def _project_world_point(
        self,
        point: np.ndarray,
        sensor_position: np.ndarray,
        rotation: Any,
    ) -> Optional[Tuple[float, float, float]]:
        """Project a world-space point into image pixel coordinates."""
        right, up, forward = self._camera_axes(rotation)
        rel = np.array(point, dtype=np.float32) - np.array(sensor_position, dtype=np.float32)

        cam_x = float(np.dot(rel, right))
        cam_y = float(np.dot(rel, up))
        cam_z = float(np.dot(rel, forward))
        if cam_z <= 1e-4:
            return None

        fx, fy, cx, cy = self._camera_intrinsics()
        pixel_x = cx + fx * (cam_x / cam_z)
        pixel_y = cy - fy * (cam_y / cam_z)
        return pixel_x, pixel_y, cam_z

    def _entity_mask_from_aabb(
        self,
        target: SemanticTarget,
        sensor_position: np.ndarray,
        rotation: Any,
    ) -> np.ndarray:
        """Build an entity mask by intersecting camera rays with the object's 3D AABB.

        This avoids relying on semantic ids. The result is an instance-specific
        geometry mask, not a rectangular bbox fill.
        """
        width = int(self.args.width)
        height = int(self.args.height)
        fx, fy, cx, cy = self._camera_intrinsics()
        right, up, forward = self._camera_axes(rotation)

        pixel_x = np.arange(width, dtype=np.float32)
        pixel_y = np.arange(height, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(pixel_x, pixel_y)

        cam_x = (grid_x - cx) / fx
        cam_y = (cy - grid_y) / fy

        ray_dirs = (
            forward.reshape(1, 1, 3)
            + cam_x[..., None] * right.reshape(1, 1, 3)
            + cam_y[..., None] * up.reshape(1, 1, 3)
        )
        ray_norms = np.linalg.norm(ray_dirs, axis=2, keepdims=True)
        ray_dirs = ray_dirs / np.maximum(ray_norms, 1e-8)

        box_min = np.array(target.center, dtype=np.float32) - 0.5 * np.array(
            target.sizes,
            dtype=np.float32,
        )
        box_max = np.array(target.center, dtype=np.float32) + 0.5 * np.array(
            target.sizes,
            dtype=np.float32,
        )
        origin = np.array(sensor_position, dtype=np.float32)

        with np.errstate(divide="ignore", invalid="ignore"):
            inv_dirs = 1.0 / ray_dirs
            t0 = (box_min.reshape(1, 1, 3) - origin.reshape(1, 1, 3)) * inv_dirs
            t1 = (box_max.reshape(1, 1, 3) - origin.reshape(1, 1, 3)) * inv_dirs

        t_near = np.maximum.reduce(np.minimum(t0, t1), axis=2)
        t_far = np.minimum.reduce(np.maximum(t0, t1), axis=2)
        return (t_far >= np.maximum(t_near, 0.0))

    def _mask_bbox(self, mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Compute image-space bbox from a boolean entity mask."""
        ys, xs = np.where(mask)
        if ys.size == 0:
            return None
        return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

    def _mask_visible_ratio(self, mask: np.ndarray) -> float:
        """Estimate object image coverage from the entity mask."""
        if mask.size == 0:
            return 0.0
        return float(np.mean(mask.astype(np.float32, copy=False)))

    def _semantic_instance_mask(
        self,
        semantic: np.ndarray,
        semantic_id: int,
    ) -> np.ndarray:
        """Return the per-pixel semantic mask for the requested semantic id."""
        semantic_ids = semantic.astype(np.int64, copy=False)
        return semantic_ids == int(semantic_id)

    def _mask_iou(self, mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        """Compute IoU between two boolean masks."""
        mask_a_bool = mask_a.astype(bool, copy=False)
        mask_b_bool = mask_b.astype(bool, copy=False)
        union = np.logical_or(mask_a_bool, mask_b_bool)
        union_sum = int(np.count_nonzero(union))
        if union_sum <= 0:
            return 0.0
        intersection = np.logical_and(mask_a_bool, mask_b_bool)
        intersection_sum = int(np.count_nonzero(intersection))
        return float(intersection_sum) / float(union_sum)

    def _mask_to_image(self, mask: np.ndarray) -> Image.Image:
        """Convert a boolean entity mask into a plain binary image."""
        return Image.fromarray(mask.astype(np.uint8) * 255)

    def _semantic_color(self, semantic_id: int) -> Tuple[int, int, int]:
        """Return a deterministic RGB color for a semantic id."""
        sid = int(semantic_id)
        if sid == 0:
            return (0, 0, 0)
        return (
            (sid * 37) % 256,
            (sid * 67) % 256,
            (sid * 97) % 256,
        )

    def _semantic_to_image(self, semantic: np.ndarray) -> Image.Image:
        """Convert raw semantic ids into a colored semantic visualization."""
        semantic_ids = semantic.astype(np.int64, copy=False)
        semantic_rgb = np.zeros((*semantic_ids.shape, 3), dtype=np.uint8)
        for semantic_id in np.unique(semantic_ids):
            semantic_rgb[semantic_ids == semantic_id] = self._semantic_color(
                int(semantic_id)
            )
        return Image.fromarray(semantic_rgb)

    def _format_float(self, value: float) -> str:
        """Format scalar used in deterministic output filenames."""
        return f"{float(value):.1f}"

    def _format_percent(self, value: float) -> str:
        """Format a ratio as percentage for deterministic output filenames."""
        return self._format_float(100.0 * float(value))

    def _goal_filename(
        self,
        semantic_id: int,
        sensor_position: np.ndarray,
        rotation: Any,
        object_center: np.ndarray,
        surface_distance: float,
        frame_cov: float,
        iou: float,
    ) -> str:
        """Build filename with semantic id, object center, sensor pose, and distance."""
        pos = np.array(sensor_position, dtype=np.float32)
        center = np.array(object_center, dtype=np.float32)
        quat_xyzw = _quat_to_list(rotation)
        ix, iy, iz, w = [float(value) for value in quat_xyzw]

        return (
            f"{int(semantic_id)}"
            f"_p_{self._format_float(pos[0])}"
            f"_{self._format_float(pos[1])}"
            f"_{self._format_float(pos[2])}"
            f"_r_{self._format_float(w)}"
            f"_{self._format_float(ix)}"
            f"_{self._format_float(iy)}"
            f"_{self._format_float(iz)}"
            f"_c_{self._format_float(center[0])}"
            f"_{self._format_float(center[1])}"
            f"_{self._format_float(center[2])}"
            f"_d_{self._format_float(surface_distance)}"
            f"_fc_{self._format_percent(frame_cov)}"
            f"_iou_{self._format_percent(iou)}.png"
        )

    def _surface_distance_from_floor_point(
        self,
        floor_position: np.ndarray,
        target: SemanticTarget,
    ) -> float:
        """Compute floor-viewpoint distance to target surface in the x-z plane."""
        delta_xz = (
            np.array(floor_position[[0, 2]], dtype=np.float32)
            - np.array(target.center[[0, 2]], dtype=np.float32)
        )
        center_dist = float(np.linalg.norm(delta_xz))
        return max(0.0, center_dist - float(target.horizontal_radius))

    def _resample_candidates_around_viewpoint(
        self,
        target: SemanticTarget,
        seed_candidate: ViewpointCandidate,
        retry_radius: float,
        max_retries: int,
    ) -> List[ViewpointCandidate]:
        """Sample nearby fallback viewpoints around one failed render candidate."""
        retries: List[ViewpointCandidate] = []
        if max_retries <= 0 or retry_radius <= 1e-6:
            return retries

        seed_floor = np.array(seed_candidate.floor_position, dtype=np.float32)
        floor_level = self._nearest_floor_level(float(seed_floor[1]))
        seen = {
            (
                round(float(seed_floor[0]), 3),
                round(float(seed_floor[1]), 3),
                round(float(seed_floor[2]), 3),
            )
        }
        golden_angle_rad = math.radians(137.50776405003785)
        base_angle_rad = math.radians(float(seed_candidate.angle_deg % 360))

        for attempt_idx in range(max_retries):
            radius = float(retry_radius) * math.sqrt(
                float(attempt_idx + 1) / float(max_retries)
            )
            theta = base_angle_rad + float(attempt_idx + 1) * golden_angle_rad
            probe = np.array(
                [
                    float(seed_floor[0]) + radius * math.cos(theta),
                    float(seed_floor[1]),
                    float(seed_floor[2]) + radius * math.sin(theta),
                ],
                dtype=np.float32,
            )
            snapped = self._snap_to_floor_navmesh(probe, floor_level=floor_level)
            if snapped is None:
                continue

            key = (
                round(float(snapped[0]), 3),
                round(float(snapped[1]), 3),
                round(float(snapped[2]), 3),
            )
            if key in seen:
                continue
            seen.add(key)

            base_position = np.array(snapped, dtype=np.float32)
            sensor_position = self._sensor_world_position(base_position)
            rotation = self._look_at_quaternion(base_position, target.center)
            edge_clearance = self._distance_to_navmesh_edge(base_position)
            nearby_count, nearby_penalty = self._nearby_object_stats_2d(
                base_position,
                target_id=target.semantic_id,
            )
            sensor_clearance = self._min_sensor_clearance(
                sensor_position,
                target_id=target.semantic_id,
            )
            retry_floor = self._nearest_floor_level(float(base_position[1]))
            retry_floor_y = (
                float(retry_floor.y)
                if retry_floor is not None
                else float(base_position[1])
            )

            retries.append(
                ViewpointCandidate(
                    floor_position=base_position,
                    sensor_position=sensor_position,
                    base_position=base_position,
                    rotation=rotation,
                    surface_distance=self._surface_distance_from_floor_point(
                        base_position,
                        target,
                    ),
                    angle_deg=self._estimate_angle_deg(base_position, target),
                    edge_clearance=float(edge_clearance),
                    nearby_count=int(nearby_count),
                    nearby_penalty=float(nearby_penalty),
                    sensor_clearance=float(sensor_clearance),
                    sensor_offset=float(self.sensor_height),
                    floor_level_index=self._floor_level_index(retry_floor),
                    floor_level_y=float(retry_floor_y),
                )
            )
            if len(retries) >= int(max_retries):
                break

        return retries

    def _scan_target(
        self,
        target: SemanticTarget,
        object_output_dir: Path,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Select up to three dominant-floor viewpoints around a target and render them."""
        views: List[Dict[str, Any]] = []
        observe_failures = 0
        selected_candidates: List[ViewpointCandidate] = []
        ranked_floor_candidates, rank_diagnostics, invalid_views = self._rank_floor_candidates(
            target,
            min_edge_clearance=float(self.args.min_edge_clearance),
        )
        scan_diagnostics: Dict[str, Any] = {
            **rank_diagnostics,
            "sensor_pose_none": 0,
            "diversity_reject": 0,
            "selected": 0,
            "observe_failures": 0,
            "iou_reject": 0,
            "yolo_reject": 0,
            "rendered_views": 0,
        }
        max_viewpoints = min(int(self.args.max_viewpoints), 3)
        for floor_stats in ranked_floor_candidates:
            if len(selected_candidates) >= max_viewpoints:
                break
            candidate = self._select_sensor_pose(
                floor_position=floor_stats["floor_position"],
                target=target,
                floor_stats=floor_stats,
            )
            if candidate is None:
                scan_diagnostics["sensor_pose_none"] += 1
                invalid_views.append(
                    self._invalid_view_record(
                        target=target,
                        stage="sensor_pose",
                        reason="sensor_pose_none",
                        floor_position=floor_stats["floor_position"],
                        angle_deg=int(floor_stats["angle_deg"]),
                        surface_distance=float(floor_stats["surface_distance"]),
                        candidate_source=floor_stats.get("candidate_source"),
                        details={
                            "edge_clearance": float(floor_stats["edge_clearance"]),
                        },
                    )
                )
                continue
            if not self._is_diverse_viewpoint(candidate, selected_candidates):
                scan_diagnostics["diversity_reject"] += 1
                min_angle_diff = None
                min_separation = None
                for chosen in selected_candidates:
                    angle_diff = abs(candidate.angle_deg - chosen.angle_deg) % 360
                    angle_diff = min(angle_diff, 360 - angle_diff)
                    dist_2d = float(
                        np.linalg.norm(
                            (candidate.floor_position - chosen.floor_position)[[0, 2]]
                        )
                    )
                    min_angle_diff = (
                        angle_diff
                        if min_angle_diff is None
                        else min(min_angle_diff, angle_diff)
                    )
                    min_separation = (
                        dist_2d
                        if min_separation is None
                        else min(min_separation, dist_2d)
                    )
                invalid_views.append(
                    self._invalid_view_record(
                        target=target,
                        stage="selection",
                        reason="diversity_reject",
                        floor_position=candidate.floor_position,
                        base_position=candidate.base_position,
                        sensor_position=candidate.sensor_position,
                        rotation=candidate.rotation,
                        angle_deg=int(candidate.angle_deg),
                        surface_distance=float(candidate.surface_distance),
                        candidate_source=floor_stats.get("candidate_source"),
                        details={
                            "edge_clearance": float(candidate.edge_clearance),
                            "sensor_clearance": float(candidate.sensor_clearance),
                            "min_angle_diff": min_angle_diff,
                            "min_angle_threshold": float(self.args.min_viewpoint_angle_sep),
                            "min_separation": min_separation,
                            "min_separation_threshold": float(
                                self.args.min_viewpoint_separation
                            ),
                        },
                    )
                )
                continue
            selected_candidates.append(candidate)
            scan_diagnostics["selected"] = int(len(selected_candidates))

        retry_radius = max(
            0.0,
            float(getattr(self.args, "render_validation_retry_radius", 0.25)),
        )
        retry_max_attempts = max(
            0,
            int(getattr(self.args, "render_validation_retry_max_attempts", 10)),
        )

        for candidate in selected_candidates:
            attempt_candidates: List[ViewpointCandidate] = [candidate]
            attempt_candidates.extend(
                self._resample_candidates_around_viewpoint(
                    target=target,
                    seed_candidate=candidate,
                    retry_radius=retry_radius,
                    max_retries=retry_max_attempts,
                )
            )

            for retry_index, active_candidate in enumerate(attempt_candidates):
                rotation = active_candidate.rotation
                quat_xyzw = _quat_to_list(rotation)
                alignment_deg = self._forward_alignment_deg(
                    active_candidate.sensor_position,
                    rotation,
                    target.center,
                )
                self.logger.debug(
                    "[Align] %s %d | forward-target angle = %.3f deg | retry=%d",
                    self.target_label,
                    target.semantic_id,
                    alignment_deg,
                    int(retry_index),
                )

                try:
                    rgb, semantic = self._observe(
                        active_candidate.base_position,
                        rotation,
                        target_id=target.semantic_id,
                    )
                except Exception:
                    observe_failures += 1
                    scan_diagnostics["observe_failures"] = int(observe_failures)
                    invalid_views.append(
                        self._invalid_view_record(
                            target=target,
                            stage="render",
                            reason="observe_failure",
                            floor_position=active_candidate.floor_position,
                            base_position=active_candidate.base_position,
                            sensor_position=active_candidate.sensor_position,
                            rotation=active_candidate.rotation,
                            angle_deg=int(active_candidate.angle_deg),
                            surface_distance=float(active_candidate.surface_distance),
                            details={
                                "edge_clearance": float(active_candidate.edge_clearance),
                                "sensor_clearance": float(active_candidate.sensor_clearance),
                                "retry_index": int(retry_index),
                                "retry_radius": float(retry_radius),
                                "retry_max_attempts": int(retry_max_attempts),
                            },
                        )
                    )
                    continue

                center_projection = self._project_world_point(
                    target.center,
                    sensor_position=active_candidate.sensor_position,
                    rotation=rotation,
                )
                if center_projection is None:
                    center_pixel = self._image_center_pixel()
                else:
                    center_pixel = (center_projection[0], center_projection[1])

                entity_mask = self._entity_mask_from_aabb(
                    target,
                    active_candidate.sensor_position,
                    rotation,
                )
                semantic_mask = self._semantic_instance_mask(semantic, target.semantic_id)
                bbox = self._mask_bbox(entity_mask)
                visible_ratio = self._mask_visible_ratio(entity_mask)
                mask_iou = self._mask_iou(entity_mask, semantic_mask)
                if mask_iou < float(self.args.min_iou):
                    scan_diagnostics["iou_reject"] += 1
                    invalid_views.append(
                        self._invalid_view_record(
                            target=target,
                            stage="render_validation",
                            reason="iou_reject",
                            floor_position=active_candidate.floor_position,
                            base_position=active_candidate.base_position,
                            sensor_position=active_candidate.sensor_position,
                            rotation=active_candidate.rotation,
                            angle_deg=int(active_candidate.angle_deg),
                            surface_distance=float(active_candidate.surface_distance),
                            details={
                                "iou": float(mask_iou),
                                "min_iou": float(self.args.min_iou),
                                "frame_cov": float(visible_ratio),
                                "bbox": list(bbox) if bbox is not None else None,
                                "retry_index": int(retry_index),
                                "retry_radius": float(retry_radius),
                                "retry_max_attempts": int(retry_max_attempts),
                            },
                        )
                    )
                    continue
                yolo_valid, yolo_matches, yolo_num_detections, yolo_annotated = self._validate_with_yolo(
                    rgb,
                    target.category_name,
                )
                if not yolo_valid:
                    scan_diagnostics["yolo_reject"] += 1
                    invalid_views.append(
                        self._invalid_view_record(
                            target=target,
                            stage="render_validation",
                            reason="yolo_reject",
                            floor_position=active_candidate.floor_position,
                            base_position=active_candidate.base_position,
                            sensor_position=active_candidate.sensor_position,
                            rotation=active_candidate.rotation,
                            angle_deg=int(active_candidate.angle_deg),
                            surface_distance=float(active_candidate.surface_distance),
                            details={
                                "iou": float(mask_iou),
                                "frame_cov": float(visible_ratio),
                                "yolo_num_detections": int(yolo_num_detections),
                                "yolo_matched_detections": yolo_matches,
                                "yolo_conf_threshold": float(self.args.yolo_conf_threshold),
                                "retry_index": int(retry_index),
                                "retry_radius": float(retry_radius),
                                "retry_max_attempts": int(retry_max_attempts),
                            },
                        )
                    )
                    continue

                goal_name = self._goal_filename(
                    semantic_id=target.semantic_id,
                    sensor_position=active_candidate.sensor_position,
                    rotation=rotation,
                    object_center=target.center,
                    surface_distance=active_candidate.surface_distance,
                    frame_cov=visible_ratio,
                    iou=mask_iou,
                )
                yolo_name = f"yolo_{goal_name}"

                goal_path = object_output_dir / goal_name
                yolo_path = object_output_dir / yolo_name
                save_images = bool(getattr(self.args, "save_images", True))
                goal_rel = None
                yolo_rel = None
                if save_images:
                    Image.fromarray(rgb).save(goal_path)
                    if yolo_annotated is not None:
                        yolo_annotated.save(yolo_path)
                    goal_rel = goal_path.relative_to(self.output_root).as_posix()
                    yolo_rel = (
                        yolo_path.relative_to(self.output_root).as_posix()
                        if yolo_annotated is not None
                        else None
                    )
                view_payload = {
                    "semantic_id": int(target.semantic_id),
                    "category": target.category_name,
                    "position": [round(float(v), 3) for v in active_candidate.sensor_position],
                    "agent_base_position": [round(float(v), 3) for v in active_candidate.base_position],
                    "floor_position": [round(float(v), 3) for v in active_candidate.floor_position],
                    "floor_index": active_candidate.floor_level_index,
                    "floor_y": round(float(active_candidate.floor_level_y), 3),
                    "object_center": [round(float(v), 3) for v in target.center],
                    "rotation": [round(float(v), 6) for v in quat_xyzw],
                    "object_center_pixel": [round(float(v), 3) for v in center_pixel],
                    "bbox": list(bbox) if bbox is not None else None,
                    "radius": round(active_candidate.surface_distance, 3),
                    "angle_deg": int(active_candidate.angle_deg),
                    "visible_ratio": round(visible_ratio, 3),
                    "frame_cov": round(100.0 * visible_ratio, 1),
                    "iou": round(100.0 * mask_iou, 1),
                    "edge_clearance": round(active_candidate.edge_clearance, 3),
                    "nearby_count": int(active_candidate.nearby_count),
                    "nearby_penalty": round(active_candidate.nearby_penalty, 3),
                    "sensor_clearance": round(active_candidate.sensor_clearance, 3),
                    "sensor_offset": round(active_candidate.sensor_offset, 3),
                    "retry_index": int(retry_index),
                    "yolo_num_detections": int(yolo_num_detections),
                    "yolo_matched_detections": yolo_matches,
                    "goal_image": goal_rel,
                    "yolo_image": yolo_rel,
                }
                views.append(view_payload)
                scan_diagnostics["rendered_views"] = int(len(views))
                break

        self.logger.info(
            "%s %d (%s): %d valid views",
            self.target_label,
            target.semantic_id,
            target.category_name,
            len(views),
        )
        if not views:
            self.logger.info(
                (
                    "%s %d rejects | raw=%d y_mismatch=%d too_near=%d too_far=%d "
                    "valid_mask=%d invalid_floor=%d edge=%d ranked_floor=%d "
                    "sensor_pose_none=%d diversity=%d selected=%d observe=%d iou=%d yolo=%d "
                    "floor_index=%s floor_y=%.3f center_dist=[%.3f, %.3f]"
                ),
                self.target_label,
                target.semantic_id,
                int(scan_diagnostics["raw_candidates"]),
                int(scan_diagnostics["y_mismatch"]),
                int(scan_diagnostics["too_near"]),
                int(scan_diagnostics["too_far"]),
                int(scan_diagnostics["valid_mask_pass"]),
                int(scan_diagnostics["invalid_floor_navmesh"]),
                int(scan_diagnostics["edge_reject"]),
                int(scan_diagnostics["ranked_floor"]),
                int(scan_diagnostics["sensor_pose_none"]),
                int(scan_diagnostics["diversity_reject"]),
                int(scan_diagnostics["selected"]),
                int(scan_diagnostics["observe_failures"]),
                int(scan_diagnostics["iou_reject"]),
                int(scan_diagnostics["yolo_reject"]),
                str(scan_diagnostics["target_floor_index"]),
                float(scan_diagnostics["target_floor_y"]),
                float(scan_diagnostics["min_center_distance"]),
                float(scan_diagnostics["max_center_distance"]),
            )
        return views, invalid_views


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for deterministic semantic-target scanning."""
    parser = argparse.ArgumentParser(
        description=(
            "Select up to three open floor viewpoints near each semantic target, "
            "lift the sensor by 1.25-1.5m, and export RGB/annotated/semantic views."
        )
    )
    parser.add_argument(
        "scene_name",
        nargs="?",
        default=None,
        type=str,
        help=(
            "Optional MP3D scene id, e.g. QUCTc6BB5sX. If omitted, scans all scenes "
            "under the MP3D val split."
        ),
    )
    parser.add_argument(
        "--target-category",
        type=str,
        default=None,
        help=(
            "Optional semantic category substring to match; if omitted, scans all "
            "21 MP3D target categories (chair, table, picture, cabinet, cushion, "
            "sofa, bed, chest_of_drawers, plant, sink, toilet, stool, towel, "
            "tv_monitor, shower, bathtub, counter, fireplace, gym_equipment, "
            "seating, clothes)"
        ),
    )
    parser.add_argument(
        "--yolo-model",
        type=Path,
        default=Path("models/yolo26x.pt"),
        help="YOLO weights used to validate rendered views before saving",
    )
    parser.add_argument(
        "--yolo-device",
        type=str,
        default=None,
        help="YOLO inference device, e.g. cpu, 0, cuda:0",
    )
    parser.add_argument(
        "--yolo-conf-threshold",
        type=float,
        default=0.59,
        help="Minimum YOLO detection confidence for save-time validation",
    )
    parser.add_argument(
        "--yolo-iou-threshold",
        type=float,
        default=0.45,
        help="YOLO NMS IoU threshold for save-time validation",
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
        help="Optional JSON mapping MP3D target labels to YOLO class-name aliases",
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
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Root output directory",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional explicit output manifest path",
    )
    parser.add_argument(
        "--save-images",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Save rendered RGB/YOLO images to disk (default: save)",
    )
    parser.add_argument("--width", type=int, default=512, help="RGB/semantic width")
    parser.add_argument("--height", type=int, default=512, help="RGB/semantic height")
    parser.add_argument("--hfov", type=float, default=90.0, help="Sensor horizontal FoV")
    parser.add_argument(
        "--horizontal-only-rotation",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Constrain render rotation to yaw-only (horizontal view) (default: yes)",
    )
    parser.add_argument(
        "--sensor-height",
        type=float,
        default=1.25,
        help="Configured Habitat sensor height above the agent base",
    )
    parser.add_argument(
        "--search-radius",
        type=float,
        default=1.0,
        help="Maximum 2D search radius from the target surface to candidate floor points",
    )
    parser.add_argument(
        "--min-surface-offset",
        type=float,
        default=0.5,
        help="Minimum 2D offset from the target surface to a floor viewpoint",
    )
    parser.add_argument(
        "--candidate-angle-step",
        type=int,
        default=30,
        help="Angular step for deterministic fallback floor probes",
    )
    parser.add_argument(
        "--radial-step",
        type=float,
        default=0.15,
        help="Radial step (meters) between ring samples along one direction",
    )
    parser.add_argument(
        "--radial-step-jitter",
        type=float,
        default=0.05,
        help="Random jitter (meters) applied to each radial step",
    )
    parser.add_argument(
        "--angle-jitter",
        type=float,
        default=10.0,
        help="Angular jitter (degrees) applied to each candidate direction",
    )
    parser.add_argument(
        "--min-edge-clearance",
        type=float,
        default=0.2,
        help="Required clearance from the viewpoint to navmesh boundaries",
    )
    parser.add_argument(
        "--min-iou",
        type=float,
        default=0.2,
        help="Reject rendered views whose geometry-vs-semantic IoU is below this threshold",
    )
    parser.add_argument(
        "--max-floor-levels",
        type=int,
        default=3,
        help="Maximum number of dominant navigable floor levels kept in one scene",
    )
    parser.add_argument(
        "--floor-level-tolerance",
        type=float,
        default=0.35,
        help="Height clustering tolerance when identifying dominant floor levels",
    )
    parser.add_argument(
        "--nearby-object-radius",
        type=float,
        default=3.0,
        help="Legacy openness parameter kept for compatibility",
    )
    parser.add_argument(
        "--max-viewpoints",
        type=int,
        default=3,
        help="Maximum number of rendered viewpoints per target",
    )
    parser.add_argument(
        "--min-sensor-offset",
        type=float,
        default=1.25,
        help="Legacy parameter kept for compatibility; rendering uses --sensor-height",
    )
    parser.add_argument(
        "--max-sensor-offset",
        type=float,
        default=1.5,
        help="Legacy parameter kept for compatibility; rendering uses --sensor-height",
    )
    parser.add_argument(
        "--sensor-offset-step",
        type=float,
        default=0.125,
        help="Legacy parameter kept for compatibility",
    )
    parser.add_argument(
        "--min-sensor-clearance",
        type=float,
        default=0.1,
        help="Legacy sensor-clearance parameter kept for compatibility",
    )
    parser.add_argument(
        "--floor-height-tolerance",
        type=float,
        default=2.75,
        help="Allowed Y difference between candidate floor points and the target support floor",
    )
    parser.add_argument(
        "--min-viewpoint-angle-sep",
        type=int,
        default=45,
        help="Minimum angular separation between selected viewpoints around one target",
    )
    parser.add_argument(
        "--min-viewpoint-separation",
        type=float,
        default=0.75,
        help="Minimum 2D separation between selected floor viewpoints",
    )
    parser.add_argument(
        "--radius-min",
        type=float,
        default=0.5,
        help="Legacy fallback radius parameter kept for compatibility",
    )
    parser.add_argument(
        "--radius-max",
        type=float,
        default=2.0,
        help="Legacy fallback radius parameter kept for compatibility",
    )
    parser.add_argument(
        "--radius-step",
        type=float,
        default=0.5,
        help="Legacy fallback radius parameter kept for compatibility",
    )
    parser.add_argument(
        "--angle-step",
        type=int,
        default=10,
        help="Legacy fallback angle parameter kept for compatibility",
    )
    parser.add_argument(
        "--max-snap-distance",
        type=float,
        default=0.5,
        help="Max allowed x-z distance between intended and snapped points",
    )
    parser.add_argument(
        "--object-clearance",
        type=float,
        default=0.1,
        help="Minimum 3D clearance to any semantic object AABB",
    )
    parser.add_argument(
        "--position-adjust-max",
        type=float,
        default=0.25,
        help="Maximum forward/backward radial adjustment for one direction",
    )
    parser.add_argument(
        "--position-adjust-step",
        type=float,
        default=0.05,
        help="Step size for radial adjustment search",
    )
    parser.add_argument(
        "--min-floor-island-radius",
        type=float,
        default=1.5,
        help="Legacy navmesh-island parameter kept for compatibility",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Python logging level",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    """Configure script logging format and level."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def discover_val_scene_names(scene_dir: Path) -> Tuple[Path, List[str]]:
    """Discover available MP3D val-scene ids from the local scene directory."""
    resolved_scene_dir = scene_dir.expanduser().resolve()
    val_root = (
        (resolved_scene_dir / "val").resolve()
        if (resolved_scene_dir / "val").is_dir()
        else resolved_scene_dir
    )
    if not val_root.is_dir():
        raise RuntimeError(f"MP3D val scene directory does not exist: {val_root}")

    scene_names: List[str] = []
    for candidate_dir in sorted(path for path in val_root.iterdir() if path.is_dir()):
        scene_name = candidate_dir.name
        if (candidate_dir / f"{scene_name}.glb").is_file():
            scene_names.append(scene_name)

    if not scene_names:
        raise RuntimeError(f"No MP3D val scenes found under {val_root}")
    return val_root, scene_names


def main() -> None:
    """Entrypoint for deterministic semantic-target scanner CLI."""
    args = parse_args()
    configure_logging(args.log_level)
    logger = logging.getLogger("generate_imagenav_eval_dataset")

    if args.scene_name is not None:
        scanner = ImageNavDeterministicScanner(args)
        try:
            output_path = scanner.run()
            print(f"Deterministic scan manifest saved at: {output_path}")
        finally:
            scanner.close()
        return

    if args.output_json is not None:
        raise ValueError(
            "--output-json requires an explicit scene_name; omit it only for per-scene default manifests"
        )

    val_root, scene_names = discover_val_scene_names(args.scene_dir)
    logger.info(
        "No scene_name provided; scanning %d MP3D val scenes from %s",
        len(scene_names),
        val_root,
    )

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    batch_summary: Dict[str, Any] = {
        "split": "val",
        "scene_root": str(val_root),
        "num_scenes": len(scene_names),
        "succeeded": [],
        "failed": [],
    }

    for scene_name in scene_names:
        logger.info("Starting val scene: %s", scene_name)
        scene_args = argparse.Namespace(**vars(args))
        scene_args.scene_name = scene_name

        scanner: Optional[ImageNavDeterministicScanner] = None
        try:
            scanner = ImageNavDeterministicScanner(scene_args)
            output_path = scanner.run()
            batch_summary["succeeded"].append(
                {
                    "scene_name": scene_name,
                    "manifest_json": str(output_path),
                }
            )
            print(f"Deterministic scan manifest saved at: {output_path}")
        except Exception as exc:
            logger.exception("Failed to process val scene %s", scene_name)
            batch_summary["failed"].append(
                {
                    "scene_name": scene_name,
                    "error": str(exc),
                }
            )
        finally:
            if scanner is not None:
                scanner.close()

    summary_path = output_root / "mp3d_val_scan_summary.json"
    with summary_path.open("w", encoding="utf-8") as file_obj:
        json.dump(batch_summary, file_obj, indent=2, ensure_ascii=False)

    logger.info(
        "Finished MP3D val scan: %d succeeded, %d failed",
        len(batch_summary["succeeded"]),
        len(batch_summary["failed"]),
    )
    print(f"Val split summary saved at: {summary_path}")


if __name__ == "__main__":
    main()
