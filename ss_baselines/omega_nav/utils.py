from __future__ import annotations

import copy
import hashlib
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import yaml


DEFAULT_OMEGA_CONFIG: Dict[str, Any] = {
    "goal_encoder": {
        "embedding_dim": 32,
        "image_histogram_bins": 8,
    },
    "visual": {
        "describe_every": 5,
        "top_m": 3,
        "min_visible_pixels": 24,
        "max_summary_objects": 4,
        "visible_similarity_threshold": 0.5,
    },
    "audio": {
        "aggregation_window": 2,
        "detection_threshold": 0.60,
        "ear_distance_m": 0.18,
        "speed_of_sound_mps": 343.0,
        "reference_rms": 0.05,
        "sampling_rate_hz": 16000,
        "step_time_s": 0.25,
        "protocol_cycle_steps": 25,
        "protocol_transient_steps": 3,
        "trace_history_size": 32,
        "gcc_interp": 8,
        "bearing_peak_exclusion_bins": 8,
        "bearing_min_peak_ratio": 1.05,
        "bearing_min_confidence": 0.15,
        "bearing_subframe_samples": 1024,
        "bearing_subframe_hop_samples": 512,
        "bearing_min_valid_subframes": 2,
        "bearing_max_spread_deg": 12.0,
        "world_tracker_angle_bin_deg": 2.0,
        "world_tracker_min_evidence_frames": 3,
        "world_tracker_min_heading_span_deg": 45.0,
        "world_tracker_min_confidence": 0.3,
        "world_tracker_max_posterior_entropy": 0.98,
        "world_tracker_background_history_size": 8,
        "world_tracker_background_min_history": 2,
        "world_tracker_background_strength": 0.75,
        "background_similarity": 0.05,
        "distance_scale_m": 1.0,
    },
    "depth": {
        "voxel_size_m": 0.10,
        "map_resolution_m": 0.04,
        "map_size_cells": 800,
        "max_depth_m": 6.0,
        "min_depth_m": 0.1,
        "sample_stride": 16,
        "hfov_deg": 90.0,
        "frontier_max_points": 24,
        "assume_normalized_depth": True,
    },
    "memory": {
        "working_window": 8,
        "episodic_decay": 0.2,
        "episodic_prune_threshold": 0.05,
        "revisit_radius_m": 1.5,
    },
    "planner": {
        "submit_distance_m": 1.0,
        "distance_penalty": 0.05,
        "visible_priority_bonus": 2.0,
        "audio_priority_bonus": 1.2,
        "hint_priority_bonus": 0.8,
        "audio_submit_similarity": 0.92,
        "forward_submit_angle_deg": 20.0,
        "audio_submit_distance_cap": 1.25,
    },
    "navigator": {
        "follower_goal_radius": 1e-3,
    },
    "policy": {
        "submit_action_name": "LIFELONG_SUBMIT",
        "stop_action_name": "STOP",
    },
}


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_omega_config(
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    config = copy.deepcopy(DEFAULT_OMEGA_CONFIG)
    if config_path:
        payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            config = deep_update(config, payload)
    if isinstance(overrides, dict) and overrides:
        config = deep_update(config, overrides)
    return config


def extract_rgb(observations: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    if not isinstance(observations, dict):
        return None
    for key in ("rgb", "RGB_SENSOR", "rgb_sensor"):
        value = observations.get(key)
        if value is not None:
            return np.asarray(value, dtype=np.uint8)
    return None


def extract_depth(
    observations: Optional[Dict[str, Any]],
    *,
    max_depth_m: float,
    assume_normalized: bool,
) -> Optional[np.ndarray]:
    if not isinstance(observations, dict):
        return None
    depth = None
    for key in ("depth", "DEPTH_SENSOR", "depth_sensor"):
        value = observations.get(key)
        if value is not None:
            depth = np.asarray(value, dtype=np.float32)
            break
    if depth is None:
        return None
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        return None
    if assume_normalized and float(np.nanmax(depth)) <= 1.0 + 1e-6:
        depth = depth * float(max_depth_m)
    return np.clip(depth, 0.0, float(max_depth_m))


def extract_audio(observations: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    if not isinstance(observations, dict):
        return None
    for key in ("audiogoal", "AUDIOGOAL_SENSOR", "audio", "audio_sensor"):
        value = observations.get(key)
        if value is not None:
            return np.asarray(value, dtype=np.float32)
    return None


def extract_semantic(observations: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    if not isinstance(observations, dict):
        return None
    for key in ("semantic", "SEMANTIC_SENSOR", "semantic_sensor"):
        value = observations.get(key)
        if value is None:
            continue
        semantic = np.asarray(value)
        if semantic.ndim == 3:
            semantic = semantic[:, :, 0]
        if semantic.ndim == 2:
            return semantic.astype(np.int64, copy=False)
    return None


def ensure_vec3(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        return None
    return array


def rotate_vector_by_quaternion(quaternion_xyzw: Any, vector: Any) -> np.ndarray:
    quat = np.asarray(quaternion_xyzw, dtype=np.float32).reshape(4)
    vec = np.asarray(vector, dtype=np.float32).reshape(3)
    quat_xyz = quat[:3]
    quat_w = float(quat[3])
    uv = np.cross(quat_xyz, vec)
    uuv = np.cross(quat_xyz, uv)
    return vec + 2.0 * (quat_w * uv + uuv)


def heading_from_quaternion(quaternion_xyzw: Any) -> float:
    heading_vector = rotate_vector_by_quaternion(quaternion_xyzw, np.asarray([0.0, 0.0, -1.0], dtype=np.float32))
    return float(np.arctan2(float(heading_vector[0]), float(-heading_vector[2])))


def extract_pose(observations: Optional[Dict[str, Any]]) -> Optional[np.ndarray]:
    if not isinstance(observations, dict):
        return None
    for key in ("full_pose_sensor", "FULL_POSE_SENSOR"):
        value = observations.get(key)
        if value is None:
            continue
        pose = np.asarray(value, dtype=np.float32).reshape(-1)
        if pose.size >= 7 and np.all(np.isfinite(pose[:7])):
            return pose[:7]
    return None


def pose_to_position(pose: Any) -> Optional[np.ndarray]:
    if pose is None:
        return None
    array = np.asarray(pose, dtype=np.float32).reshape(-1)
    if array.size < 3 or not np.all(np.isfinite(array[:3])):
        return None
    return np.asarray(array[:3], dtype=np.float32)


def pose_to_heading(pose: Any) -> Optional[float]:
    if pose is None:
        return None
    array = np.asarray(pose, dtype=np.float32).reshape(-1)
    if array.size < 7 or not np.all(np.isfinite(array[3:7])):
        return None
    return heading_from_quaternion(array[3:7])


def rotate_local_offset(heading_rad: float, vector: Sequence[float]) -> np.ndarray:
    offset = np.asarray(vector, dtype=np.float32).reshape(3)
    angle = float(heading_rad)
    cos_angle = float(math.cos(angle))
    sin_angle = float(math.sin(angle))
    x_coord = float(offset[0]) * cos_angle + float(offset[2]) * sin_angle
    z_coord = -float(offset[0]) * sin_angle + float(offset[2]) * cos_angle
    return np.asarray([x_coord, float(offset[1]), z_coord], dtype=np.float32)


def relative_bearing_from_pose_deg(pose: Any, target_position: Any) -> float:
    agent_position = pose_to_position(pose)
    heading_rad = pose_to_heading(pose)
    target = ensure_vec3(target_position)
    if agent_position is None or heading_rad is None or target is None:
        return 0.0
    delta = np.asarray(target, dtype=np.float32) - np.asarray(agent_position, dtype=np.float32)
    delta[1] = 0.0
    norm = float(np.linalg.norm(delta[[0, 2]]))
    if norm <= 1e-6:
        return 0.0
    target_heading = float(math.atan2(float(delta[0]), float(-delta[2])))
    angle = float(np.degrees(target_heading - float(heading_rad)))
    while angle <= -180.0:
        angle += 360.0
    while angle > 180.0:
        angle -= 360.0
    return angle


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def hash_seed(value: str) -> int:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def hash_embedding(value: str, dim: int) -> np.ndarray:
    rng = np.random.default_rng(hash_seed(value))
    vec = rng.normal(size=(int(dim),)).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    if norm <= 1e-6:
        return np.zeros((int(dim),), dtype=np.float32)
    return vec / norm


def image_histogram_embedding(image: np.ndarray, bins: int) -> np.ndarray:
    if image is None:
        return np.zeros((int(bins) * 3,), dtype=np.float32)
    array = np.asarray(image, dtype=np.uint8)
    if array.ndim != 3 or array.shape[-1] != 3:
        return np.zeros((int(bins) * 3,), dtype=np.float32)
    histograms = []
    for channel in range(3):
        hist, _ = np.histogram(array[:, :, channel], bins=int(bins), range=(0, 256), density=True)
        histograms.append(hist.astype(np.float32))
    embedding = np.concatenate(histograms, axis=0)
    norm = float(np.linalg.norm(embedding))
    if norm <= 1e-6:
        return embedding
    return embedding / norm


def cosine_similarity(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> float:
    if a is None or b is None:
        return 0.0
    aa = np.asarray(a, dtype=np.float32).reshape(-1)
    bb = np.asarray(b, dtype=np.float32).reshape(-1)
    if aa.size == 0 or bb.size == 0 or aa.shape != bb.shape:
        return 0.0
    denom = float(np.linalg.norm(aa) * np.linalg.norm(bb))
    if denom <= 1e-6:
        return 0.0
    return float(np.clip(np.dot(aa, bb) / denom, -1.0, 1.0))


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0 or xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _quat_xyzw(rotation: Any) -> Tuple[float, float, float, float]:
    if rotation is None:
        return 0.0, 0.0, 0.0, 1.0
    if isinstance(rotation, np.ndarray):
        array = np.asarray(rotation, dtype=np.float32).reshape(-1)
        if array.size >= 4:
            return float(array[0]), float(array[1]), float(array[2]), float(array[3])
        return 0.0, 0.0, 0.0, 1.0
    if hasattr(rotation, "imag") and hasattr(rotation, "real"):
        imag = np.asarray(rotation.imag, dtype=np.float32).reshape(-1)
        if imag.size >= 3:
            return float(imag[0]), float(imag[1]), float(imag[2]), float(rotation.real)
    if all(hasattr(rotation, key) for key in ("x", "y", "z", "w")):
        return float(rotation.x), float(rotation.y), float(rotation.z), float(rotation.w)
    array = np.asarray(rotation, dtype=np.float32).reshape(-1)
    if array.size >= 4:
        return float(array[0]), float(array[1]), float(array[2]), float(array[3])
    return 0.0, 0.0, 0.0, 1.0


def rotate_vector(rotation: Any, vector: Sequence[float]) -> np.ndarray:
    x, y, z, w = _quat_xyzw(rotation)
    vx, vy, vz = [float(v) for v in vector]
    q_vec = np.asarray([x, y, z], dtype=np.float32)
    uv = np.cross(q_vec, np.asarray([vx, vy, vz], dtype=np.float32))
    uuv = np.cross(q_vec, uv)
    return np.asarray([vx, vy, vz], dtype=np.float32) + 2.0 * (w * uv + uuv)


def forward_vector(rotation: Any) -> np.ndarray:
    forward = rotate_vector(rotation, (0.0, 0.0, -1.0))
    forward[1] = 0.0
    norm = float(np.linalg.norm(forward[[0, 2]]))
    if norm <= 1e-6:
        return np.asarray([0.0, 0.0, -1.0], dtype=np.float32)
    return forward / norm


def yaw_from_rotation(rotation: Any) -> float:
    forward = forward_vector(rotation)
    return float(math.atan2(float(forward[0]), float(-forward[2])))


def relative_bearing_deg(agent_position: np.ndarray, rotation: Any, target_position: np.ndarray) -> float:
    delta = np.asarray(target_position, dtype=np.float32) - np.asarray(agent_position, dtype=np.float32)
    delta[1] = 0.0
    norm = float(np.linalg.norm(delta[[0, 2]]))
    if norm <= 1e-6:
        return 0.0
    delta = delta / norm
    forward = forward_vector(rotation)
    cross = float(forward[0] * delta[2] - forward[2] * delta[0])
    dot = float(forward[0] * delta[0] + forward[2] * delta[2])
    return float(np.degrees(math.atan2(cross, dot)))


def coarse_direction_from_angle(angle_deg: float) -> str:
    if angle_deg <= -15.0:
        return "left"
    if angle_deg >= 15.0:
        return "right"
    return "forward"


def chinese_direction_from_angle(angle_deg: float) -> str:
    rounded = int(round(abs(float(angle_deg)) / 5.0) * 5)
    if rounded < 5:
        return "前方"
    side = "左" if float(angle_deg) < 0.0 else "右"
    if rounded < 60:
        return f"{side}前方 {rounded}°"
    if rounded <= 120:
        return f"{side}侧 {rounded}°"
    return f"{side}后方 {rounded}°"


def sector_free_space(depth: Optional[np.ndarray]) -> Dict[str, float]:
    if depth is None or depth.ndim != 2 or depth.size == 0:
        return {"left": 0.0, "forward": 0.0, "right": 0.0}
    width = depth.shape[1]
    thirds = [depth[:, : width // 3], depth[:, width // 3 : 2 * width // 3], depth[:, 2 * width // 3 :]]
    labels = ("left", "forward", "right")
    result: Dict[str, float] = {}
    for label, sector in zip(labels, thirds):
        finite = sector[np.isfinite(sector)]
        result[label] = float(np.median(finite)) if finite.size > 0 else 0.0
    return result


def world_to_grid(position: np.ndarray, origin: np.ndarray, resolution_m: float, grid_size: int) -> Optional[Tuple[int, int]]:
    pos = ensure_vec3(position)
    org = ensure_vec3(origin)
    if pos is None or org is None:
        return None
    half = int(grid_size) // 2
    gx = int(round((float(pos[0]) - float(org[0])) / float(resolution_m))) + half
    gz = int(round((float(pos[2]) - float(org[2])) / float(resolution_m))) + half
    if 0 <= gx < int(grid_size) and 0 <= gz < int(grid_size):
        return gx, gz
    return None


def grid_to_world(cell: Tuple[int, int], origin: np.ndarray, resolution_m: float, grid_size: int) -> np.ndarray:
    half = int(grid_size) // 2
    gx, gz = int(cell[0]), int(cell[1])
    return np.asarray(
        [
            float(origin[0]) + float(gx - half) * float(resolution_m),
            float(origin[1]),
            float(origin[2]) + float(gz - half) * float(resolution_m),
        ],
        dtype=np.float32,
    )


def interpolate_grid_line(start: Tuple[int, int], end: Tuple[int, int]) -> Iterable[Tuple[int, int]]:
    x0, y0 = int(start[0]), int(start[1])
    x1, y1 = int(end[0]), int(end[1])
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        yield x0, y0
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def as_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): as_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_serializable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value
