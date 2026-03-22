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
    audio_peak: Dict[str, Any] = field(default_factory=dict)
    voxel_count: int = 0
    occupied_voxel_count: int = 0
    semantic_voxel_count: int = 0
    audio_voxel_count: int = 0

    def summary(self) -> Dict[str, Any]:
        return {
            "occupancy_shape": list(self.occupancy.shape),
            "voxel_count": int(self.voxel_count),
            "occupied_voxel_count": int(self.occupied_voxel_count),
            "semantic_voxel_count": int(self.semantic_voxel_count),
            "audio_voxel_count": int(self.audio_voxel_count),
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
                "audio_peak": self.audio_peak,
                "voxel_count": self.voxel_count,
                "occupied_voxel_count": self.occupied_voxel_count,
                "semantic_voxel_count": self.semantic_voxel_count,
                "audio_voxel_count": self.audio_voxel_count,
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
            "expected_labels": [str(label) for label in self.expected_labels],
        }

    def to_dict(self) -> Dict[str, Any]:
        return as_serializable(self.summary())


@dataclass
class AudioBearingEstimate:
    is_valid: bool = False
    reason: str = ""
    frame_reason: str = ""
    relative_bearing_rad: float = 0.0
    relative_bearing_deg: float = 0.0
    world_bearing_rad: float = 0.0
    world_bearing_deg: float = 0.0
    tau_s: float = 0.0
    tau_samples: float = 0.0
    predicted_tau_s: float = 0.0
    predicted_tau_samples: float = 0.0
    max_tau_s: float = 0.0
    peak: float = 0.0
    peak_ratio: float = 0.0
    confidence: float = 0.0
    posterior_peak: float = 0.0
    posterior_entropy: float = 1.0
    posterior_margin: float = 0.0
    heading_span_deg: float = 0.0
    evidence_frame_count: int = 0
    window_frame_count: int = 0
    direction: str = ""
    valid_subframe_count: int = 0
    total_subframe_count: int = 0
    spread_deg: float = 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "is_valid": bool(self.is_valid),
            "reason": str(self.reason),
            "frame_reason": str(self.frame_reason),
            "relative_bearing_deg": round(float(self.relative_bearing_deg), 2),
            "world_bearing_deg": round(float(self.world_bearing_deg), 2),
            "tau_samples": round(float(self.tau_samples), 2),
            "predicted_tau_samples": round(float(self.predicted_tau_samples), 2),
            "peak": round(float(self.peak), 3),
            "peak_ratio": round(float(self.peak_ratio), 3),
            "confidence": round(float(self.confidence), 3),
            "posterior_peak": round(float(self.posterior_peak), 4),
            "posterior_entropy": round(float(self.posterior_entropy), 4),
            "posterior_margin": round(float(self.posterior_margin), 4),
            "heading_span_deg": round(float(self.heading_span_deg), 2),
            "evidence_frame_count": int(self.evidence_frame_count),
            "window_frame_count": int(self.window_frame_count),
            "direction": str(self.direction),
            "valid_subframe_count": int(self.valid_subframe_count),
            "total_subframe_count": int(self.total_subframe_count),
            "spread_deg": round(float(self.spread_deg), 2),
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
            "bearing": self.bearing.summary(),
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
