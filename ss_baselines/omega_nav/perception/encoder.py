from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ss_baselines.omega_nav.perception.base import AudioPerceptionState, MapObservation, PerceptionOutput
from ss_baselines.omega_nav.perception.siglip_encoder import SigLIPEncoder
from ss_baselines.omega_nav.perception.voxel_map import SemanticVoxelMap
from ss_baselines.omega_nav.utils import cosine_similarity, extract_audio, extract_depth, extract_pose, extract_rgb, pose_to_heading, pose_to_position


class PerceptionEncoder:
    def __init__(self, config: Dict[str, Any], device: torch.device, goal_encoder_model_name: str) -> None:
        self._audio_cfg = dict(config.get("audio", {}))
        self._depth_cfg = dict(config.get("depth", {}))
        self._audio_aggregation_window = max(int(self._audio_cfg.get("aggregation_window", 10)), 1)
        self._audio_match_threshold = float(self._audio_cfg.get("detection_threshold", 0.65))
        self._audio_reference_rms = float(self._audio_cfg.get("reference_rms", 0.05))
        self._audio_top_k = max(int(self._audio_cfg.get("anchor_count", 5)), 1)
        self._max_depth_m = float(self._depth_cfg.get("max_depth_m", 6.0))
        self._assume_normalized_depth = bool(self._depth_cfg.get("assume_normalized_depth", True))
        self._siglip = SigLIPEncoder(goal_encoder_model_name, device)
        self._map = SemanticVoxelMap(self._depth_cfg, self._siglip)
        self._audio_prototype_library: Optional[Any] = None
        self._audio_candidates: Dict[str, np.ndarray] = {}
        self._audio_history: List[np.ndarray] = []
        self._goal_embeddings: Dict[str, np.ndarray] = {}
        self._last_output: Optional[PerceptionOutput] = None

    def reset(self, observations: Optional[Dict[str, Any]] = None) -> None:
        position = np.zeros(3, dtype=np.float32)
        if observations is not None:
            pose = extract_pose(observations)
            position = np.asarray(pose_to_position(pose), dtype=np.float32)
        self._map.reset(position)
        self._audio_candidates = {}
        self._audio_history = []
        self._goal_embeddings = {}
        self._last_output = None

    def set_audio_runtime(self, prototype_library: Any) -> None:
        self._audio_prototype_library = prototype_library

    def set_audio_candidates(self, candidates: Dict[str, np.ndarray]) -> None:
        self._audio_candidates = {
            str(label): np.asarray(embedding, dtype=np.float32)
            for label, embedding in candidates.items()
        }

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
        output = PerceptionOutput(
            int(step_index),
            self._map.build_state(self._goal_embeddings),
            self._perceive_audio(observations),
        )
        print(
            "[omega_nav][percept] "
            + json.dumps(
                {
                    "step_index": int(step_index),
                    "visual": {
                        "rgb_shape": list(observation.rgb.shape),
                        "depth_shape": list(observation.depth.shape),
                        "depth_min_m": float(np.min(observation.depth)),
                        "depth_max_m": float(np.max(observation.depth)),
                    },
                    "pose": {
                        "position": np.round(observation.position, 3).tolist(),
                        "heading_deg": round(float(np.degrees(observation.heading_rad)), 2),
                    },
                    "audio": output.audio_state.summary(),
                    "map": output.map_state.summary(),
                },
                ensure_ascii=False,
            )
        )
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

    def _perceive_audio(self, observations: Dict[str, Any]) -> AudioPerceptionState:
        if self._audio_prototype_library is None:
            return AudioPerceptionState(score_threshold=float(self._audio_match_threshold))
        audio = extract_audio(observations)
        if audio is None:
            return AudioPerceptionState(score_threshold=float(self._audio_match_threshold))
        mono = self._to_mono(audio)
        rms = float(np.sqrt(np.mean(np.square(mono)))) if mono.size > 0 else 0.0
        peak = float(np.max(np.abs(mono - float(np.mean(mono))))) if mono.size > 0 else 0.0
        if mono.size == 0 or peak <= max(self._audio_reference_rms * 1e-4, 1e-8):
            self._audio_history = []
            return AudioPerceptionState(
                is_active=False,
                rms=rms,
                peak=peak,
                score_threshold=float(self._audio_match_threshold),
            )
        embedding = np.asarray(self._audio_prototype_library.encode_audio_observation(audio), dtype=np.float32)
        self._audio_history.append(embedding)
        if len(self._audio_history) > self._audio_aggregation_window:
            self._audio_history = self._audio_history[-self._audio_aggregation_window :]
        smoothed = np.mean(np.stack(self._audio_history, axis=0), axis=0).astype(np.float32)
        norm = float(np.linalg.norm(smoothed))
        if norm > 1e-6:
            smoothed = smoothed / norm
        scores, scope = self._score_audio(smoothed)
        if not scores:
            return AudioPerceptionState(
                is_active=True,
                rms=rms,
                peak=peak,
                score_threshold=float(self._audio_match_threshold),
            )
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_scores = {str(label): float(score) for label, score in ranked[: self._audio_top_k]}
        top_label, top_score = ranked[0]
        label = str(top_label) if float(top_score) >= self._audio_match_threshold else ""
        return AudioPerceptionState(
            is_active=True,
            rms=rms,
            peak=peak,
            label=label,
            score=float(top_score),
            score_threshold=float(self._audio_match_threshold),
            scores=top_scores,
            scope=str(scope),
        )

    def _score_audio(self, embedding: np.ndarray) -> Tuple[Dict[str, float], str]:
        if len(self._audio_candidates) > 0:
            return {
                str(label): float(cosine_similarity(embedding, candidate))
                for label, candidate in self._audio_candidates.items()
            }, "episode"
        return self._audio_prototype_library.score_embedding(embedding), "global"

    def _to_mono(self, audio: np.ndarray) -> np.ndarray:
        array = np.asarray(audio, dtype=np.float32)
        if array.ndim == 1:
            return array.reshape(-1)
        if array.ndim == 2 and array.shape[0] <= 4:
            return np.mean(array, axis=0, dtype=np.float32)
        if array.ndim == 2 and array.shape[1] <= 4:
            return np.mean(array, axis=1, dtype=np.float32)
        return array.reshape(-1)
