#!/usr/bin/env python3

"""Generate multimodal episode start states with navmesh clearance checks."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import quaternion  # noqa: F401
except Exception:
    pass

import numpy as np
import habitat_sim
from ss_baselines.av_nav.config.default import get_task_config
from habitat.sims import make_sim


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


@dataclass
class SceneObject:
    """Semantic object info used for goal sampling."""

    semantic_id: int
    category: str
    center: np.ndarray
    sizes: np.ndarray
    horizontal_radius: float
    aabb_min: np.ndarray
    aabb_max: np.ndarray


@dataclass(frozen=True)
class RejectionNote:
    semantic_id: int
    category: str
    reason: str


@dataclass(frozen=True)
class GoalCandidate:
    obj: SceneObject
    nav_position: np.ndarray
    distance_to_start: float


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


def _resolve_scene_path(scene_dir: Path, scene_name: str) -> Path:
    candidate = Path(scene_name)
    if candidate.suffix == ".glb":
        if candidate.is_absolute() and candidate.is_file():
            return candidate
        local_candidate = (scene_dir / candidate).resolve()
        if local_candidate.is_file():
            return local_candidate

    direct = (scene_dir / scene_name / f"{scene_name}.glb").resolve()
    if direct.is_file():
        return direct

    recursive = sorted(scene_dir.glob(f"**/{scene_name}.glb"))
    if recursive:
        return recursive[0].resolve()

    raise RuntimeError(f"Failed to resolve scene '{scene_name}' under {scene_dir}")


def _scene_id_for_dataset(scene_path: Path, scene_dir: Path) -> str:
    try:
        relative = scene_path.relative_to(scene_dir).as_posix()
    except Exception:
        relative = scene_path.name
    if relative.startswith("mp3d/"):
        return relative
    return f"mp3d/{relative}"


def _build_simulator(
    scene_path: Path,
    scene_dataset_config: Optional[Path],
    exp_config: Path,
    width: int,
    height: int,
    hfov: float,
    sensor_height: float,
):
    cfg = get_task_config(config_paths=[str(exp_config)])
    cfg.defrost()
    cfg.SIMULATOR.TYPE = "ContinuousSoundSpacesSim"
    cfg.SIMULATOR.SCENE = str(scene_path)
    if scene_dataset_config is not None:
        cfg.SIMULATOR.SCENE_DATASET = str(scene_dataset_config)
    cfg.SIMULATOR.AUDIO.ENABLED = False
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

    sensors = list(getattr(cfg.SIMULATOR.AGENT_0, "SENSORS", []))
    for sensor_name in ("RGB_SENSOR", "SEMANTIC_SENSOR"):
        if sensor_name not in sensors:
            sensors.append(sensor_name)
    cfg.SIMULATOR.AGENT_0.SENSORS = sensors

    cfg.freeze()
    sim = make_sim(cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR)
    sim.reset()
    return sim


def _yaw_to_quaternion(yaw_rad: float) -> List[float]:
    half = 0.5 * float(yaw_rad)
    return [0.0, math.sin(half), 0.0, math.cos(half)]


def _normalize_category(category: str) -> str:
    return category.strip().lower().replace(" ", "_")


def _semantic_objects(sim: Any) -> List[Any]:
    if hasattr(sim, "_semantic_objects"):
        try:
            return list(sim._semantic_objects())
        except Exception:
            pass
    inner_sim = getattr(sim, "_sim", sim)
    semantic_scene = getattr(inner_sim, "semantic_scene", None)
    if semantic_scene is None:
        return []
    try:
        return list(semantic_scene.objects)
    except Exception:
        return []


def _safe_semantic_id(obj: Any) -> Optional[int]:
    for attr_name in ("semantic_id", "semanticID"):
        value = getattr(obj, attr_name, None)
        if value is None:
            continue
        try:
            return int(value)
        except Exception:
            continue
    return None


def _collect_scene_objects(sim: Any) -> List[SceneObject]:
    objects: List[SceneObject] = []
    for obj in _semantic_objects(sim):
        category = getattr(obj, "category", None)
        if category is None:
            continue
        try:
            category_name = str(category.name())
        except Exception:
            continue
        normalized = _normalize_category(category_name)
        if normalized not in MP3D_TARGET_CATEGORIES:
            continue
        aabb = getattr(obj, "aabb", None)
        if aabb is None:
            continue
        semantic_id = _safe_semantic_id(obj)
        if semantic_id is None:
            continue
        center = np.array(aabb.center, dtype=np.float32)
        sizes = np.array(aabb.sizes, dtype=np.float32)
        aabb_min = center - 0.5 * sizes
        aabb_max = center + 0.5 * sizes
        horizontal_radius = 0.5 * math.sqrt(float(sizes[0]) ** 2 + float(sizes[2]) ** 2)
        objects.append(
            SceneObject(
                semantic_id=int(semantic_id),
                category=normalized,
                center=center,
                sizes=sizes,
                horizontal_radius=horizontal_radius,
                aabb_min=aabb_min,
                aabb_max=aabb_max,
            )
        )
    return objects


def _group_objects_by_category(objects: Sequence[SceneObject]) -> Dict[str, List[SceneObject]]:
    grouped: Dict[str, List[SceneObject]] = {}
    for obj in objects:
        grouped.setdefault(obj.category, []).append(obj)
    for category, items in grouped.items():
        items.sort(key=lambda item: item.semantic_id)
    return grouped


def _load_image_goal_map(path: Path) -> Tuple[Dict[tuple[str, int], Dict[str, Any]], Dict[str, Any]]:
    if not path.exists():
        return {}, {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}, {}
    goals = data.get("goals", {})
    if not isinstance(goals, dict):
        return {}, {}
    mapping: Dict[tuple[str, int], Dict[str, Any]] = {}
    for category, per_id in goals.items():
        if not isinstance(per_id, dict):
            continue
        for semantic_id, goal in per_id.items():
            try:
                semantic_int = int(semantic_id)
            except (TypeError, ValueError):
                continue
            if isinstance(goal, dict):
                mapping[(str(category), semantic_int)] = goal
    return mapping, goals


def _is_valid_image_goal(goal: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(goal, dict):
        return False
    render_views = goal.get("render_views")
    if isinstance(render_views, list) and len(render_views) > 0:
        return True
    num_views = goal.get("num_views")
    if isinstance(num_views, (int, float)) and num_views > 0:
        return True
    return False


def _build_goal_candidates_for_start(
    objects: Sequence[SceneObject],
    pathfinder: Any,
    floor_levels: Sequence[FloorLevel],
    floor_height_tolerance: float,
    min_goal_distance: float,
    start_position: np.ndarray,
    rejection_log: Optional[Dict[Tuple[int, str], List[str]]] = None,
) -> List[GoalCandidate]:
    candidates: List[GoalCandidate] = []
    for obj in objects:
        goal_nav, nav_reason = _project_goal_to_navmesh(
            obj.center,
            pathfinder,
            floor_levels,
            floor_height_tolerance,
        )
        if goal_nav is None:
            if rejection_log is not None:
                rejection_log.setdefault((int(obj.semantic_id), obj.category), []).append(
                    f"project_fail:{nav_reason}"
                )
            continue
        distance_to_start = _geodesic_distance(pathfinder, start_position, goal_nav)
        if not math.isfinite(distance_to_start):
            if rejection_log is not None:
                rejection_log.setdefault((int(obj.semantic_id), obj.category), []).append(
                    "no_geodesic_path"
                )
            continue
        if distance_to_start < float(min_goal_distance):
            if rejection_log is not None:
                rejection_log.setdefault((int(obj.semantic_id), obj.category), []).append(
                    f"too_close_to_start dist={distance_to_start:.3f} "
                    f"min={float(min_goal_distance):.3f}"
                )
            continue
        candidates.append(
            GoalCandidate(
                obj=obj,
                nav_position=goal_nav,
                distance_to_start=distance_to_start,
            )
        )
    return candidates


def _sample_goal_set(
    candidates: Sequence[GoalCandidate],
    num_goals: int,
    min_goal_distance: float,
    pathfinder: Any,
    rng: random.Random,
    max_attempts: int,
    category_priority: Sequence[str],
) -> Optional[List[GoalCandidate]]:
    if len(candidates) < num_goals:
        return None
    candidates_by_category: Dict[str, List[GoalCandidate]] = {}
    for candidate in candidates:
        candidates_by_category.setdefault(candidate.obj.category, []).append(candidate)
    categories = [c for c in category_priority if c in candidates_by_category]
    if not categories:
        return None
    attempts = max(1, int(max_attempts))
    for _ in range(attempts):
        if len(categories) >= num_goals:
            target_categories = rng.sample(categories, num_goals)
        else:
            target_categories = list(categories)
            while len(target_categories) < num_goals:
                target_categories.append(rng.choice(categories))
        rng.shuffle(target_categories)
        chosen: List[GoalCandidate] = []
        chosen_navs: List[np.ndarray] = []
        used_keys: set[tuple[str, int]] = set()
        success = True
        for category in target_categories:
            options = list(candidates_by_category.get(category, []))
            rng.shuffle(options)
            picked = False
            for candidate in options:
                key = (candidate.obj.category, int(candidate.obj.semantic_id))
                if key in used_keys:
                    continue
                valid = True
                for other_nav in chosen_navs:
                    dist = _geodesic_distance(
                        pathfinder, candidate.nav_position, other_nav
                    )
                    if not math.isfinite(dist) or dist < float(min_goal_distance):
                        valid = False
                        break
                if not valid:
                    continue
                chosen.append(candidate)
                chosen_navs.append(candidate.nav_position)
                used_keys.add(key)
                picked = True
                break
            if not picked:
                success = False
                break
        if success and len(chosen) == num_goals:
            return chosen
    return None


def _print_rejection_report(
    title: str,
    rejections: Dict[Tuple[int, str], List[str]] | Dict[Tuple[int, str], str],
) -> None:
    if not rejections:
        return
    print(title)
    if isinstance(next(iter(rejections.values())), list):
        sorted_items = sorted(
            rejections.items(), key=lambda item: (item[0][1], item[0][0])
        )
        for (semantic_id, category), reasons in sorted_items:
            unique_reasons = sorted(set(reasons))
            print(f"  [{category}] id={semantic_id}: " + "; ".join(unique_reasons))
    else:
        sorted_items = sorted(
            rejections.items(), key=lambda item: (item[0][1], item[0][0])
        )
        for (semantic_id, category), reason in sorted_items:
            print(f"  [{category}] id={semantic_id}: {reason}")


def _point_in_aabb(point: np.ndarray, obj: SceneObject) -> bool:
    return bool(
        (point[0] >= obj.aabb_min[0])
        and (point[0] <= obj.aabb_max[0])
        and (point[1] >= obj.aabb_min[1])
        and (point[1] <= obj.aabb_max[1])
        and (point[2] >= obj.aabb_min[2])
        and (point[2] <= obj.aabb_max[2])
    )


def _object_has_navigable_footprint(
    obj: SceneObject,
    objects: Sequence[SceneObject],
    pathfinder: Any,
    snap_distance: float,
    floor_levels: Sequence[FloorLevel],
    floor_height_tolerance: float,
    grid_steps: int = 3,
) -> Tuple[bool, str]:
    if pathfinder is None or not hasattr(pathfinder, "snap_point"):
        return True, "pathfinder_unavailable"
    if grid_steps < 1:
        grid_steps = 1
    xs = np.linspace(float(obj.aabb_min[0]), float(obj.aabb_max[0]), grid_steps)
    zs = np.linspace(float(obj.aabb_min[2]), float(obj.aabb_max[2]), grid_steps)
    base_y = float(obj.aabb_min[1])
    reasons: Counter[str] = Counter()
    total_probes = 0
    for x in xs:
        for z in zs:
            total_probes += 1
            probe = np.array([x, base_y, z], dtype=np.float32)
            try:
                snapped = np.array(pathfinder.snap_point(probe), dtype=np.float32)
            except Exception:
                reasons["snap_exception"] += 1
                continue
            if snapped.shape[0] != 3 or not np.all(np.isfinite(snapped)):
                reasons["snap_invalid"] += 1
                continue
            xz_error = float(np.linalg.norm((snapped - probe)[[0, 2]]))
            if xz_error > float(snap_distance):
                reasons["snap_too_far"] += 1
                continue
            if hasattr(pathfinder, "is_navigable"):
                try:
                    if not bool(pathfinder.is_navigable(snapped)):
                        reasons["not_navigable"] += 1
                        continue
                except Exception:
                    reasons["nav_check_error"] += 1
                    continue
            if not _matches_floor_level(
                float(snapped[1]),
                floor_levels,
                floor_height_tolerance,
            ):
                reasons["not_dominant_floor"] += 1
                continue
            occupied = False
            for other in objects:
                if other.semantic_id == obj.semantic_id:
                    continue
                if _point_in_aabb(snapped, other):
                    occupied = True
                    break
            if occupied:
                reasons["occupied_by_other"] += 1
                continue
            return True, "ok"
    if not reasons:
        return False, "no_valid_probe"
    reason_counts = ", ".join(
        f"{key}={count}" for key, count in reasons.most_common()
    )
    detail = (
        f"footprint_failed probes={total_probes} "
        f"snap_distance={float(snap_distance):.3f} ({reason_counts})"
    )
    return False, detail


def _filter_accessible_objects(
    objects: Sequence[SceneObject],
    pathfinder: Any,
    snap_distance: float,
    floor_levels: Sequence[FloorLevel],
    floor_height_tolerance: float,
) -> Tuple[List[SceneObject], Dict[Tuple[int, str], str]]:
    filtered: List[SceneObject] = []
    rejected: Dict[Tuple[int, str], str] = {}
    for obj in objects:
        ok, reason = _object_has_navigable_footprint(
            obj,
            objects=objects,
            pathfinder=pathfinder,
            snap_distance=snap_distance,
            floor_levels=floor_levels,
            floor_height_tolerance=floor_height_tolerance,
        )
        if ok:
            filtered.append(obj)
        else:
            rejected[(int(obj.semantic_id), obj.category)] = reason
    return filtered, rejected


def _project_goal_to_navmesh(
    point: np.ndarray,
    pathfinder: Any,
    floor_levels: Sequence[FloorLevel],
    floor_height_tolerance: float,
) -> Tuple[Optional[np.ndarray], str]:
    """Project a goal position vertically onto the navmesh (keep xz, adjust y)."""
    if pathfinder is None or not hasattr(pathfinder, "snap_point"):
        return None, "pathfinder_unavailable"
    try:
        snapped = np.array(pathfinder.snap_point(point), dtype=np.float32)
    except Exception:
        return None, "snap_exception"
    if snapped.shape[0] != 3 or not np.all(np.isfinite(snapped)):
        return None, "snap_invalid"
    if not _matches_floor_level(
        float(snapped[1]),
        floor_levels,
        floor_height_tolerance,
    ):
        return None, (
            f"off_dominant_floor snapped_y={float(snapped[1]):.3f} "
            f"tolerance={float(floor_height_tolerance):.3f}"
        )
    if hasattr(pathfinder, "is_navigable"):
        try:
            if not bool(pathfinder.is_navigable(snapped)):
                return None, "projected_not_navigable"
        except Exception:
            return None, "nav_check_error"
    return snapped, "ok"


def _geodesic_distance(pathfinder: Any, start: np.ndarray, end: np.ndarray) -> float:
    if pathfinder is None or not hasattr(pathfinder, "find_path"):
        return float("inf")
    try:
        start_snap = np.array(pathfinder.snap_point(start), dtype=np.float32)
        end_snap = np.array(pathfinder.snap_point(end), dtype=np.float32)
    except Exception:
        return float("inf")
    path = habitat_sim.ShortestPath()
    path.requested_start = start_snap
    path.requested_end = end_snap
    found = pathfinder.find_path(path)
    if (not found) or not np.isfinite(path.geodesic_distance):
        return float("inf")
    return float(path.geodesic_distance)


def _init_instance_pools(
    grouped: Dict[str, List[SceneObject]],
    rng: random.Random,
) -> Dict[str, Dict[str, Any]]:
    pools: Dict[str, Dict[str, Any]] = {}
    for category, items in grouped.items():
        items_copy = list(items)
        rng.shuffle(items_copy)
        pools[category] = {"items": items_copy, "index": 0}
    return pools


def _pick_instance(
    pools: Dict[str, Dict[str, Any]],
    category: str,
    rng: random.Random,
) -> SceneObject:
    pool = pools[category]
    items: List[SceneObject] = pool["items"]
    index = int(pool["index"])
    if index >= len(items):
        rng.shuffle(items)
        index = 0
    obj = items[index]
    pool["index"] = index + 1
    return obj


def _choose_goal_categories(
    rng: random.Random,
    available_categories: Sequence[str],
    num_goals: int,
) -> List[str]:
    available_list = list(available_categories)
    if not available_list:
        return []
    if len(available_list) >= num_goals:
        return rng.sample(available_list, num_goals)
    chosen = list(available_list)
    while len(chosen) < num_goals:
        chosen.append(rng.choice(available_list))
    return chosen

def _collect_navmesh_triangle_vertices(pathfinder: Any) -> np.ndarray:
    """Collect raw navmesh triangle vertices from the pathfinder."""
    if pathfinder is None or not hasattr(pathfinder, "build_navmesh_vertices"):
        return np.zeros((0, 3), dtype=np.float32)
    try:
        vertices = np.array(list(pathfinder.build_navmesh_vertices()), dtype=np.float32)
    except Exception:
        return np.zeros((0, 3), dtype=np.float32)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        return np.zeros((0, 3), dtype=np.float32)
    return vertices[np.all(np.isfinite(vertices), axis=1)]


def _triangle_area_3d(triangle: np.ndarray) -> float:
    edge_a = np.array(triangle[1], dtype=np.float32) - np.array(triangle[0], dtype=np.float32)
    edge_b = np.array(triangle[2], dtype=np.float32) - np.array(triangle[0], dtype=np.float32)
    return 0.5 * float(np.linalg.norm(np.cross(edge_a, edge_b)))


def _triangle_area_xz(triangle: np.ndarray) -> float:
    p0 = np.array(triangle[0], dtype=np.float32)[[0, 2]]
    p1 = np.array(triangle[1], dtype=np.float32)[[0, 2]]
    p2 = np.array(triangle[2], dtype=np.float32)[[0, 2]]
    return 0.5 * abs(
        float((p1[0] - p0[0]) * (p2[1] - p0[1]) - (p1[1] - p0[1]) * (p2[0] - p0[0]))
    )


def _discover_floor_levels(
    vertices: np.ndarray,
    floor_level_tolerance: float,
) -> List[FloorLevel]:
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3:
        return []
    triangle_count = len(vertices) // 3
    if triangle_count <= 0:
        return []
    triangles = vertices[: triangle_count * 3].reshape(triangle_count, 3, 3)
    triangle_centers_y = np.mean(triangles[:, :, 1], axis=1)
    sort_indices = np.argsort(triangle_centers_y)
    sorted_triangles = triangles[sort_indices]
    sorted_y = triangle_centers_y[sort_indices]
    tolerance = max(1e-3, float(floor_level_tolerance))

    clusters: List[List[tuple[float, np.ndarray]]] = []
    current_cluster: List[tuple[float, np.ndarray]] = [
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
            area_m2=float(sum(_triangle_area_3d(item[1]) for item in cluster)),
            projected_area_m2=float(sum(_triangle_area_xz(item[1]) for item in cluster)),
        )
        for cluster in clusters
        if cluster
    ]
    floor_levels.sort(key=lambda level: level.y)
    for idx, floor_level in enumerate(floor_levels):
        floor_level.index = idx
    return floor_levels


def _select_dominant_floor_levels(
    floor_levels: Sequence[FloorLevel],
    max_floor_levels: int,
) -> List[FloorLevel]:
    if not floor_levels:
        return []
    dominant = sorted(
        floor_levels,
        key=lambda level: (-level.projected_area_m2, -level.area_m2, level.y),
    )[: max(1, int(max_floor_levels))]
    dominant.sort(key=lambda level: level.y)
    return dominant


def _matches_floor_level(
    y_value: float,
    floor_levels: Sequence[FloorLevel],
    floor_height_tolerance: float,
) -> bool:
    if not floor_levels:
        return True
    tolerance = float(floor_height_tolerance)
    return any(abs(float(y_value) - float(level.y)) <= tolerance for level in floor_levels)


def _sample_start_state(
    pathfinder: Any,
    rng: random.Random,
    min_clearance: float,
    max_attempts: int,
    floor_levels: Sequence[FloorLevel],
    floor_height_tolerance: float,
) -> np.ndarray:
    if pathfinder is None or not hasattr(pathfinder, "get_random_navigable_point"):
        raise RuntimeError("Simulator pathfinder is unavailable for sampling.")

    for _ in range(max_attempts):
        point = np.array(pathfinder.get_random_navigable_point(), dtype=np.float32)
        if hasattr(pathfinder, "is_navigable"):
            try:
                if not bool(pathfinder.is_navigable(point)):
                    continue
            except Exception:
                continue
        clearance = 0.0
        if hasattr(pathfinder, "distance_to_closest_obstacle"):
            try:
                clearance = float(pathfinder.distance_to_closest_obstacle(point))
            except Exception:
                clearance = 0.0
        if clearance < float(min_clearance):
            continue
        if not _matches_floor_level(point[1], floor_levels, floor_height_tolerance):
            continue
        return point
    raise RuntimeError("Failed to sample a valid start position within max attempts.")


def _build_goal_payload(
    goal_id: int,
    obj: SceneObject,
    geodesic_distance_to_start: float,
    nav_position: np.ndarray,
) -> Dict[str, Any]:
    return {
        "goal_id": int(goal_id),
        "semantic_id": int(obj.semantic_id),
        "category": obj.category,
        "geodesic_distance_to_start": round(float(geodesic_distance_to_start), 3),
        "nav_position": [round(float(v), 3) for v in nav_position.tolist()],
        "goal_state": {
            "object_center": [round(float(v), 3) for v in obj.center.tolist()],
            "bbox_size": [round(float(v), 3) for v in obj.sizes.tolist()],
            "radius": round(float(obj.horizontal_radius), 3),
        },
        "modalities": {
            "text": {"label": obj.category},
            "image": {"render_views": []},
            "audio": {"source_position": [round(float(v), 3) for v in obj.center.tolist()]},
        },
    }


def build_episode_payload(
    scene_name: str,
    scene_id: str,
    episode_index: int,
    position: np.ndarray,
    yaw_rad: float,
    width: int,
    height: int,
    hfov: float,
    sensor_height: float,
    order_type: str,
    goals: List[Dict[str, Any]],
    goal_pair_distances: List[Dict[str, Any]],
    sampling_warning: Optional[str] = None,
) -> Dict[str, Any]:
    payload = {
        "episode_id": f"{scene_name}_{episode_index:06d}",
        "scene_name": scene_name,
        "scene_id": scene_id,
        "order_type": order_type,
        "num_goals": int(len(goals)),
        "start_state": {
            "position": [round(float(v), 3) for v in position.tolist()],
            "rotation": [round(float(v), 6) for v in _yaw_to_quaternion(yaw_rad)],
            "sensor": {
                "width": int(width),
                "height": int(height),
                "hfov": float(hfov),
                "sensor_height": float(sensor_height),
            },
        },
        "goals": goals,
        "goal_pair_distances": goal_pair_distances,
    }
    if sampling_warning:
        payload["goal_sampling_warning"] = sampling_warning
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample multimodal episode start states with navmesh clearance."
    )
    parser.add_argument("scene_name", type=str, help="MP3D scene id, e.g. QUCTc6BB5sX")
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=2,
        help="Number of episode starts to sample",
    )
    parser.add_argument(
        "--min-goals",
        type=int,
        default=2,
        help="Minimum number of goals per episode",
    )
    parser.add_argument(
        "--max-goals",
        type=int,
        default=4,
        help="Maximum number of goals per episode",
    )
    parser.add_argument(
        "--min-goal-distance",
        type=float,
        default=4.0,
        help="Minimum geodesic distance between goals and start (meters)",
    )
    parser.add_argument(
        "--goal-max-attempts",
        type=int,
        default=200,
        help="Max attempts to sample goals that satisfy distance constraints",
    )
    parser.add_argument(
        "--footprint-snap-distance",
        type=float,
        default=0.25,
        help="Max xz snap distance when validating object footprint navigability",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (defaults to output/<scene>/multimodal_episodes.json)",
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
        "--sensor-height",
        type=float,
        default=1.25,
        help="Sensor height above agent base",
    )
    parser.add_argument(
        "--min-clearance",
        type=float,
        default=0.5,
        help="Minimum clearance to navmesh obstacles",
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
        default=0.02,
        help="Allowed height difference between start position and dominant floors",
    )
    parser.add_argument(
        "--max-floor-levels",
        type=int,
        default=3,
        help="Maximum number of dominant floor bands used for sampling",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=500,
        help="Max sampling attempts per episode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--order-type",
        type=str,
        default="unordered",
        choices=("ordered", "unordered"),
        help="Goal ordering type",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_dir = args.scene_dir.expanduser().resolve()
    scene_path = _resolve_scene_path(scene_dir, args.scene_name)
    scene_id = _scene_id_for_dataset(scene_path, scene_dir)
    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else Path("output") / args.scene_name / "multimodal_episodes.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image_goal_path = output_path.parent / "imagenav_eval_episodes.json"
    _image_goal_map, image_goal_payload = _load_image_goal_map(image_goal_path)
    if not image_goal_payload:
        raise RuntimeError(
            "No valid image goals found. Expected imagenav_eval_episodes.json at "
            f"{image_goal_path} with non-empty goals."
        )

    rng = random.Random(int(args.seed))
    sim = _build_simulator(
        scene_path=scene_path,
        scene_dataset_config=args.scene_dataset_config,
        exp_config=args.exp_config,
        width=args.width,
        height=args.height,
        hfov=args.hfov,
        sensor_height=args.sensor_height,
    )
    try:
        vertices = _collect_navmesh_triangle_vertices(sim.pathfinder)
        floor_levels = _discover_floor_levels(
            vertices,
            floor_level_tolerance=float(args.floor_level_tolerance),
        )
        dominant_floors = _select_dominant_floor_levels(
            floor_levels,
            max_floor_levels=int(args.max_floor_levels),
        )

        scene_objects = _collect_scene_objects(sim)
        accessible_objects, footprint_rejections = _filter_accessible_objects(
            scene_objects,
            pathfinder=sim.pathfinder,
            snap_distance=float(args.footprint_snap_distance),
            floor_levels=dominant_floors,
            floor_height_tolerance=float(args.floor_height_tolerance),
        )
        goal_objects = list(accessible_objects)
        if not goal_objects:
            raise RuntimeError("No accessible objects with valid image goals found.")
        grouped_objects = _group_objects_by_category(goal_objects)
        available_categories = [
            category
            for category in MP3D_TARGET_CATEGORIES
            if category in grouped_objects
        ]
        if not available_categories:
            raise RuntimeError("No target categories found in the scene.")
        episodes: List[Dict[str, Any]] = []
        for idx in range(int(args.num_episodes)):
            num_goals = rng.randint(int(args.min_goals), int(args.max_goals))
            episode_attempts = 0
            last_candidate_rejections: Dict[Tuple[int, str], List[str]] = {}
            while True:
                if episode_attempts >= int(args.max_attempts):
                    _print_rejection_report(
                        "Footprint rejections (instance not navigable):",
                        footprint_rejections,
                    )
                    _print_rejection_report(
                        "Candidate rejections (per start sample):",
                        last_candidate_rejections,
                    )
                    raise RuntimeError(
                        f"Unable to sample {num_goals} goals after {int(args.max_attempts)} "
                        "episode attempts; try lowering --min-goal-distance or increasing "
                        "--goal-max-attempts."
                    )
                episode_attempts += 1
                try:
                    position = _sample_start_state(
                        sim.pathfinder,
                        rng=rng,
                        min_clearance=float(args.min_clearance),
                        max_attempts=int(args.max_attempts),
                        floor_levels=dominant_floors,
                        floor_height_tolerance=float(args.floor_height_tolerance),
                    )
                except RuntimeError:
                    continue
                yaw = rng.uniform(0.0, 2.0 * math.pi)
                candidate_rejections: Dict[Tuple[int, str], List[str]] = {}
                candidates = _build_goal_candidates_for_start(
                    objects=goal_objects,
                    pathfinder=sim.pathfinder,
                    floor_levels=dominant_floors,
                    floor_height_tolerance=float(args.floor_height_tolerance),
                    min_goal_distance=float(args.min_goal_distance),
                    start_position=position,
                    rejection_log=candidate_rejections,
                )
                last_candidate_rejections = candidate_rejections
                if len(candidates) < num_goals:
                    continue
                selected = _sample_goal_set(
                    candidates=candidates,
                    num_goals=num_goals,
                    min_goal_distance=float(args.min_goal_distance),
                    pathfinder=sim.pathfinder,
                    rng=rng,
                    max_attempts=int(args.goal_max_attempts),
                    category_priority=available_categories,
                )
                if not selected:
                    continue
                goals: List[Dict[str, Any]] = [
                    _build_goal_payload(
                        goal_id=index,
                        obj=candidate.obj,
                        geodesic_distance_to_start=candidate.distance_to_start,
                        nav_position=candidate.nav_position,
                    )
                    for index, candidate in enumerate(selected)
                ]
                break
            episodes.append(
                build_episode_payload(
                    scene_name=args.scene_name,
                    scene_id=scene_id,
                    episode_index=idx,
                    position=position,
                    yaw_rad=yaw,
                    width=args.width,
                    height=args.height,
                    hfov=args.hfov,
                    sensor_height=args.sensor_height,
                    order_type=args.order_type,
                    goals=goals,
                    goal_pair_distances=[
                        {
                            "goal_id_a": goals[i]["goal_id"],
                            "goal_id_b": goals[j]["goal_id"],
                            "geodesic_distance": round(
                                float(
                                    _geodesic_distance(
                                        sim.pathfinder,
                                        np.array(goals[i]["nav_position"], dtype=np.float32),
                                        np.array(goals[j]["nav_position"], dtype=np.float32),
                                    )
                                ),
                                3,
                            ),
                        }
                        for i in range(len(goals))
                        for j in range(i + 1, len(goals))
                    ],
                )
            )
    finally:
        sim.close()

    payload = {
        "dataset": "multimodal_lifelong_nav_preview",
        "version": "0.1",
        "episodes": episodes,
        "image_goal": image_goal_payload,
        "metadata": {
            "scene_name": args.scene_name,
            "scene_id": scene_id,
            "modalities": ["text", "image", "audio"],
            "sampling": {
                "min_clearance": float(args.min_clearance),
                "max_attempts": int(args.max_attempts),
                "seed": int(args.seed),
                "floor_level_tolerance": float(args.floor_level_tolerance),
                "floor_height_tolerance": float(args.floor_height_tolerance),
                "max_floor_levels": int(args.max_floor_levels),
                "min_goals": int(args.min_goals),
                "max_goals": int(args.max_goals),
                "min_goal_distance": float(args.min_goal_distance),
                "goal_max_attempts": int(args.goal_max_attempts),
                "footprint_snap_distance": float(args.footprint_snap_distance),
            },
            "goal_categories": available_categories,
            "goal_category_counts": {
                category: len(grouped_objects[category]) for category in available_categories
            },
            "num_scene_objects": len(scene_objects),
            "num_accessible_objects": len(accessible_objects),
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
        },
    }
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(episodes)} episode starts to: {output_path}")


if __name__ == "__main__":
    main()
