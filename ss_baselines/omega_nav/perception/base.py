from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ss_baselines.omega_nav.utils import as_serializable


@dataclass
class GoalSpec:
    goal_id: str
    goal_index: int
    modality: str
    category: str
    text_query: str
    image_description: str = ""
    image_embedding: Optional[np.ndarray] = None
    reference_image: Optional[np.ndarray] = None
    semantic_id: Optional[int] = None
    room_name: str = ""
    sound_id: str = ""
    object_position: Optional[np.ndarray] = None
    sound_position: Optional[np.ndarray] = None
    view_positions: Tuple[np.ndarray, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)

    def prompt_text(self) -> str:
        parts = [self.text_query.strip()]
        if self.image_description.strip() and self.image_description.strip() != self.text_query.strip():
            parts.append(self.image_description.strip())
        return " | ".join(part for part in parts if part)

    def to_dict(self) -> Dict[str, Any]:
        return as_serializable(asdict(self))


@dataclass
class ObjectRegionMatch:
    goal_id: str
    goal_index: int
    category: str
    similarity: float
    visible: bool
    pixel_count: int = 0
    visible_ratio: float = 0.0
    bbox: Optional[Tuple[int, int, int, int]] = None
    estimated_distance_m: Optional[float] = None
    relative_angle_deg: Optional[float] = None
    relative_direction: str = "forward"
    target_position: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return as_serializable(asdict(self))


@dataclass
class AudioMatch:
    goal_id: str
    goal_index: int
    category: str
    similarity: float
    aggregated_similarity: float
    detected: bool
    direction_text: str = "未知"
    relative_angle_deg: Optional[float] = None
    itd_seconds: Optional[float] = None
    distance_m: Optional[float] = None
    sound_position: Optional[np.ndarray] = None

    def to_dict(self) -> Dict[str, Any]:
        return as_serializable(asdict(self))


@dataclass
class SemanticMapState:
    occupancy: np.ndarray
    visited: np.ndarray
    frontier: np.ndarray
    agent_cell: Tuple[int, int]
    origin_world: np.ndarray
    resolution_m: float
    frontier_world_positions: Tuple[np.ndarray, ...] = ()
    explored_ratio: float = 0.0
    free_space_by_direction: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return as_serializable(asdict(self))


@dataclass
class PerceptionOutput:
    step_index: int
    scene_description: str
    visual_matches: Dict[str, ObjectRegionMatch]
    audio_matches: Dict[str, AudioMatch]
    top_clip_matches: Tuple[ObjectRegionMatch, ...]
    semantic_map: SemanticMapState
    observation_summary: str

    def visual_match(self, goal_id: str) -> Optional[ObjectRegionMatch]:
        return self.visual_matches.get(goal_id)

    def audio_match(self, goal_id: str) -> Optional[AudioMatch]:
        return self.audio_matches.get(goal_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_index": int(self.step_index),
            "scene_description": self.scene_description,
            "visual_matches": {key: value.to_dict() for key, value in self.visual_matches.items()},
            "audio_matches": {key: value.to_dict() for key, value in self.audio_matches.items()},
            "top_clip_matches": [match.to_dict() for match in self.top_clip_matches],
            "semantic_map": self.semantic_map.to_dict(),
            "observation_summary": self.observation_summary,
        }
