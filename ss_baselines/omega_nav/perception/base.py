from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

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
        }

    def to_dict(self) -> Dict[str, Any]:
        return as_serializable(
            {
                "occupancy": self.occupancy,
                "explored": self.explored,
                "frontier": self.frontier,
                "goal_maps": self.goal_maps,
                "voxel_count": self.voxel_count,
                "occupied_voxel_count": self.occupied_voxel_count,
                "semantic_voxel_count": self.semantic_voxel_count,
            }
        )


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

    def summary(self) -> Dict[str, Any]:
        return {
            "is_active": bool(self.is_active),
            "rms": float(self.rms),
            "peak": float(self.peak),
            "label": str(self.label),
            "score": float(self.score),
            "score_threshold": float(self.score_threshold),
            "scope": str(self.scope),
            "scores": {str(key): float(value) for key, value in self.scores.items()},
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
