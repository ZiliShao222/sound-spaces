from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from ss_baselines.omega_nav.perception.base import MapObservation, PerceptionOutput
from ss_baselines.omega_nav.perception.siglip_encoder import SigLIPEncoder
from ss_baselines.omega_nav.perception.voxel_map import SemanticVoxelMap
from ss_baselines.omega_nav.utils import extract_depth, extract_pose, extract_rgb, pose_to_heading, pose_to_position


class PerceptionEncoder:
    def __init__(self, config: Dict[str, Any], device: torch.device, goal_encoder_model_name: str) -> None:
        self._depth_cfg = dict(config.get("depth", {}))
        self._max_depth_m = float(self._depth_cfg.get("max_depth_m", 6.0))
        self._assume_normalized_depth = bool(self._depth_cfg.get("assume_normalized_depth", True))
        self._siglip = SigLIPEncoder(goal_encoder_model_name, device)
        self._map = SemanticVoxelMap(self._depth_cfg, self._siglip)
        self._goal_embeddings: Dict[str, np.ndarray] = {}
        self._last_output: Optional[PerceptionOutput] = None

    def reset(self, observations: Optional[Dict[str, Any]] = None) -> None:
        position = np.zeros(3, dtype=np.float32)
        if observations is not None:
            pose = extract_pose(observations)
            position = np.asarray(pose_to_position(pose), dtype=np.float32)
        self._map.reset(position)
        self._goal_embeddings = {}
        self._last_output = None

    def encode_goals(self, goal_ids: Sequence[str], goal_payloads: Sequence[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        self._goal_embeddings = {}
        for goal_id, payload in zip(goal_ids, goal_payloads):
            embedding = self._encode_goal_payload(payload)
            if embedding is not None:
                self._goal_embeddings[str(goal_id)] = embedding
        return dict(self._goal_embeddings)

    def percept(self, observations: Dict[str, Any], step_index: int) -> PerceptionOutput:
        observation = self._build_observation(observations, step_index)
        self._map.update(observation)
        output = PerceptionOutput(int(step_index), self._map.build_state(self._goal_embeddings))
        self._last_output = output
        return output

    def goal_embedding_dims(self) -> Dict[str, int]:
        return {goal_id: int(embedding.shape[0]) for goal_id, embedding in self._goal_embeddings.items()}

    def _encode_goal_payload(self, payload: Dict[str, Any]) -> Optional[np.ndarray]:
        modality = str(payload.get("modality", "")).lower()
        if modality == "image" and isinstance(payload.get("image"), np.ndarray):
            return self._siglip.encode_image(np.asarray(payload["image"], dtype=np.uint8))
        if modality == "object":
            text = str(payload.get("category", ""))
        else:
            text = str(payload.get("text") or payload.get("category") or "")
        text = text.strip()
        if not text:
            return None
        return self._siglip.encode_text(text)

    def _build_observation(self, observations: Dict[str, Any], step_index: int) -> MapObservation:
        rgb = extract_rgb(observations)
        depth = extract_depth(
            observations,
            max_depth_m=self._max_depth_m,
            assume_normalized=self._assume_normalized_depth,
        )
        pose = extract_pose(observations)
        position = pose_to_position(pose)
        heading = pose_to_heading(pose)
        assert rgb is not None
        assert depth is not None
        assert position is not None
        assert heading is not None
        return MapObservation(
            rgb=np.asarray(rgb, dtype=np.uint8),
            depth=np.asarray(depth, dtype=np.float32),
            position=np.asarray(position, dtype=np.float32),
            heading_rad=float(heading),
            step_index=int(step_index),
        )
