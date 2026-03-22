from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ss_baselines.omega_nav.utils import as_serializable


@dataclass
class MapObservation:
    rgb: np.ndarray
    depth: np.ndarray
    position: np.ndarray
    heading_rad: float
    step_index: int


@dataclass
class SemanticVoxelMapState:
    occupancy: np.ndarray
    explored: np.ndarray
    frontier: np.ndarray
    goal_maps: Dict[str, np.ndarray] = field(default_factory=dict)
    goal_peaks: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    voxel_count: int = 0
    occupied_voxel_count: int = 0
    semantic_voxel_count: int = 0

    def summary(self) -> Dict[str, Any]:
        return {
            "occupancy_shape": list(self.occupancy.shape),
            "voxel_count": int(self.voxel_count),
            "occupied_voxel_count": int(self.occupied_voxel_count),
            "semantic_voxel_count": int(self.semantic_voxel_count),
            "explored_cells": int(self.explored.sum()),
            "frontier_cells": int(self.frontier.sum()),
            "goal_map_keys": list(self.goal_maps.keys()),
            "goal_peak_keys": list(self.goal_peaks.keys()),
        }

    def to_dict(self) -> Dict[str, Any]:
        return as_serializable(
            {
                "occupancy": self.occupancy,
                "explored": self.explored,
                "frontier": self.frontier,
                "goal_maps": self.goal_maps,
                "goal_peaks": self.goal_peaks,
                "voxel_count": self.voxel_count,
                "occupied_voxel_count": self.occupied_voxel_count,
                "semantic_voxel_count": self.semantic_voxel_count,
            }
        )


@dataclass
class AudioObservationPacket:
    step_index: int = -1
    sample_rate_hz: int = 0
    frame_samples: int = 0
    valid_samples: int = 0
    channel_count: int = 0
    nonzero_samples: int = 0
    stereo_audio: Optional[np.ndarray] = None
    mono_audio: Optional[np.ndarray] = None
    position: Optional[np.ndarray] = None
    heading_rad: float = 0.0

    def summary(self) -> Dict[str, Any]:
        stereo_shape = []
        mono_samples = 0
        if isinstance(self.stereo_audio, np.ndarray):
            stereo_shape = list(self.stereo_audio.shape)
        if isinstance(self.mono_audio, np.ndarray):
            mono_samples = int(self.mono_audio.size)
        return {
            "step_index": int(self.step_index),
            "sample_rate_hz": int(self.sample_rate_hz),
            "frame_samples": int(self.frame_samples),
            "valid_samples": int(self.valid_samples),
            "channel_count": int(self.channel_count),
            "nonzero_samples": int(self.nonzero_samples),
            "stereo_shape": stereo_shape,
            "mono_samples": int(mono_samples),
            "frame_duration_sec": round(float(self.frame_samples / max(int(self.sample_rate_hz), 1)), 2),
            "valid_duration_sec": round(float(self.valid_samples / max(int(self.sample_rate_hz), 1)), 2),
            "position": [round(float(value), 2) for value in np.asarray(self.position, dtype=np.float32)]
            if self.position is not None
            else [],
            "heading_deg": round(float(np.degrees(float(self.heading_rad))), 2),
        }

    def to_dict(self) -> Dict[str, Any]:
        return as_serializable(self.summary())


@dataclass
class AudioProtocolState:
    goal_id: str = ""
    goal_index: int = -1
    goal_count: int = 0
    cycle_steps: int = 0
    transient_steps: int = 0
    step_time_s: float = 0.0
    window_index: int = -1
    slot_step: int = -1
    stable_start_step: int = 0
    stable_end_step: int = -1
    is_transient: bool = False
    is_stable: bool = False
    queue_depth: int = 0
    expected_labels: Tuple[str, ...] = ()

    def summary(self) -> Dict[str, Any]:
        return {
            "goal_id": str(self.goal_id),
            "goal_index": int(self.goal_index),
            "goal_count": int(self.goal_count),
            "cycle_steps": int(self.cycle_steps),
            "transient_steps": int(self.transient_steps),
            "step_time_s": round(float(self.step_time_s), 2),
            "window_index": int(self.window_index),
            "slot_step": int(self.slot_step),
            "stable_start_step": int(self.stable_start_step),
            "stable_end_step": int(self.stable_end_step),
            "is_transient": bool(self.is_transient),
            "is_stable": bool(self.is_stable),
            "queue_depth": int(self.queue_depth),
            "expected_labels": [str(label) for label in self.expected_labels],
        }

    def to_dict(self) -> Dict[str, Any]:
        return as_serializable(self.summary())


@dataclass
class AudioBearingEstimate:
    is_valid: bool = False
    reason: str = ""
    relative_bearing_rad: float = 0.0
    relative_bearing_deg: float = 0.0
    tau_s: float = 0.0
    tau_samples: float = 0.0
    max_tau_s: float = 0.0
    peak: float = 0.0
    peak_ratio: float = 0.0
    confidence: float = 0.0
    direction: str = ""

    def summary(self) -> Dict[str, Any]:
        return {
            "is_valid": bool(self.is_valid),
            "reason": str(self.reason),
            "relative_bearing_rad": round(float(self.relative_bearing_rad), 4),
            "relative_bearing_deg": round(float(self.relative_bearing_deg), 2),
            "tau_s": float(self.tau_s),
            "tau_samples": round(float(self.tau_samples), 2),
            "max_tau_s": float(self.max_tau_s),
            "peak": round(float(self.peak), 3),
            "peak_ratio": round(float(self.peak_ratio), 3),
            "confidence": round(float(self.confidence), 3),
            "direction": str(self.direction),
        }

    def to_dict(self) -> Dict[str, Any]:
        return as_serializable(self.summary())


@dataclass
class AudioRay2D:
    is_valid: bool = False
    reason: str = ""
    goal_id: str = ""
    step_index: int = -1
    origin_xz: Optional[np.ndarray] = None
    direction_xz: Optional[np.ndarray] = None
    world_bearing_rad: float = 0.0
    world_bearing_deg: float = 0.0
    confidence: float = 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "is_valid": bool(self.is_valid),
            "reason": str(self.reason),
            "goal_id": str(self.goal_id),
            "step_index": int(self.step_index),
            "origin_xz": np.round(np.asarray(self.origin_xz, dtype=np.float32), 3).tolist()
            if self.origin_xz is not None
            else [],
            "direction_xz": np.round(np.asarray(self.direction_xz, dtype=np.float32), 4).tolist()
            if self.direction_xz is not None
            else [],
            "world_bearing_rad": round(float(self.world_bearing_rad), 4),
            "world_bearing_deg": round(float(self.world_bearing_deg), 2),
            "confidence": round(float(self.confidence), 3),
        }

    def to_dict(self) -> Dict[str, Any]:
        return as_serializable(self.summary())


@dataclass
class AudioBelief2DState:
    goal_id: str = ""
    ray_count: int = 0
    total_mass: float = 0.0
    peak_value: float = 0.0
    entropy: float = 0.0
    peak_cell: Tuple[int, int] = (-1, -1)
    peak_world: Optional[np.ndarray] = None

    def summary(self) -> Dict[str, Any]:
        return {
            "goal_id": str(self.goal_id),
            "ray_count": int(self.ray_count),
            "total_mass": round(float(self.total_mass), 3),
            "peak_value": round(float(self.peak_value), 3),
            "entropy": round(float(self.entropy), 3),
            "peak_cell": [int(self.peak_cell[0]), int(self.peak_cell[1])],
            "peak_world": np.round(np.asarray(self.peak_world, dtype=np.float32), 3).tolist()
            if self.peak_world is not None
            else [],
        }

    def to_dict(self) -> Dict[str, Any]:
        return as_serializable(self.summary())


@dataclass
class AudioTriangulation2D:
    is_valid: bool = False
    reason: str = ""
    goal_id: str = ""
    ray_count: int = 0
    point_xz: Optional[np.ndarray] = None
    condition_number: float = 0.0
    mean_residual_m: float = 0.0
    min_pair_angle_deg: float = 0.0
    confidence: float = 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "is_valid": bool(self.is_valid),
            "reason": str(self.reason),
            "goal_id": str(self.goal_id),
            "ray_count": int(self.ray_count),
            "point_xz": np.round(np.asarray(self.point_xz, dtype=np.float32), 3).tolist()
            if self.point_xz is not None
            else [],
            "condition_number": round(float(self.condition_number), 3),
            "mean_residual_m": round(float(self.mean_residual_m), 3),
            "min_pair_angle_deg": round(float(self.min_pair_angle_deg), 2),
            "confidence": round(float(self.confidence), 3),
        }

    def to_dict(self) -> Dict[str, Any]:
        return as_serializable(self.summary())


@dataclass
class AudioTarget2p5D:
    goal_id: str = ""
    is_valid: bool = False
    reason: str = ""
    xz_source: str = ""
    height_source: str = "unknown"
    target_xz: Optional[np.ndarray] = None
    target_xyz: Optional[np.ndarray] = None
    confidence: float = 0.0
    visual_score: float = 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "goal_id": str(self.goal_id),
            "is_valid": bool(self.is_valid),
            "reason": str(self.reason),
            "xz_source": str(self.xz_source),
            "height_source": str(self.height_source),
            "target_xz": np.round(np.asarray(self.target_xz, dtype=np.float32), 3).tolist()
            if self.target_xz is not None
            else [],
            "target_xyz": np.round(np.asarray(self.target_xyz, dtype=np.float32), 3).tolist()
            if self.target_xyz is not None
            else [],
            "confidence": round(float(self.confidence), 3),
            "visual_score": round(float(self.visual_score), 3),
        }

    def to_dict(self) -> Dict[str, Any]:
        return as_serializable(self.summary())


@dataclass
class AudioPerceptionState:
    is_active: bool = False
    rms: float = 0.0
    peak: float = 0.0
    label: str = ""
    score: float = 0.0
    score_threshold: float = 0.0
    scores: Dict[str, float] = field(default_factory=dict)
    scope: str = ""
    packet: Optional[AudioObservationPacket] = None
    protocol: AudioProtocolState = field(default_factory=AudioProtocolState)
    bearing: AudioBearingEstimate = field(default_factory=AudioBearingEstimate)
    ray: AudioRay2D = field(default_factory=AudioRay2D)
    belief: AudioBelief2DState = field(default_factory=AudioBelief2DState)
    triangulation: AudioTriangulation2D = field(default_factory=AudioTriangulation2D)
    target_2p5d: AudioTarget2p5D = field(default_factory=AudioTarget2p5D)
    queue_depths: Dict[str, int] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        expected_labels = tuple(self.protocol.expected_labels)
        label_matches_expected = bool(self.label) and bool(expected_labels) and self.label in expected_labels
        return {
            "is_active": bool(self.is_active),
            "rms": round(float(self.rms), 2),
            "peak": round(float(self.peak), 2),
            "label": str(self.label),
            "score": round(float(self.score), 2),
            "score_threshold": round(float(self.score_threshold), 2),
            "scope": str(self.scope),
            "scores": {str(key): round(float(value), 2) for key, value in self.scores.items()},
            "packet": self.packet.summary() if self.packet is not None else {},
            "protocol": self.protocol.summary(),
            "bearing": self.bearing.summary(),
            "ray": self.ray.summary(),
            "belief": self.belief.summary(),
            "triangulation": self.triangulation.summary(),
            "target_2p5d": self.target_2p5d.summary(),
            "queue_depths": {str(key): int(value) for key, value in self.queue_depths.items()},
            "label_matches_expected": bool(label_matches_expected),
        }

    def to_dict(self) -> Dict[str, Any]:
        return as_serializable(self.summary())


@dataclass
class PerceptionOutput:
    step_index: int
    map_state: SemanticVoxelMapState
    audio_state: AudioPerceptionState = field(default_factory=AudioPerceptionState)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": int(self.step_index),
            "map_state": self.map_state.to_dict(),
            "audio_state": self.audio_state.to_dict(),
        }
