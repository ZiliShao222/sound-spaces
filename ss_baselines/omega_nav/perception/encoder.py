from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ss_baselines.omega_nav.audio_localization import (
    estimate_horizontal_bearing,
    project_bearing_to_world_ray,
    summarize_belief_map,
    triangulate_rays_2d,
    update_belief_map_from_ray,
)
from ss_baselines.omega_nav.perception.base import (
    AudioObservationPacket,
    AudioPerceptionState,
    AudioProtocolState,
    AudioRay2D,
    AudioTarget2p5D,
    MapObservation,
    PerceptionOutput,
    SemanticVoxelMapState,
)
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
        self._audio_ear_distance_m = float(self._audio_cfg.get("ear_distance_m", 0.18))
        self._audio_speed_of_sound_mps = float(self._audio_cfg.get("speed_of_sound_mps", 343.0))
        self._audio_sampling_rate_hz = max(int(self._audio_cfg.get("sampling_rate_hz", 16000)), 1)
        self._audio_step_time_s = max(float(self._audio_cfg.get("step_time_s", 0.25)), 1e-3)
        self._audio_waveform_window_sec = max(float(self._audio_cfg.get("waveform_window_sec", 1.0)), self._audio_step_time_s)
        self._audio_protocol_cycle_steps = max(int(self._audio_cfg.get("protocol_cycle_steps", 25)), 1)
        self._audio_protocol_transient_steps = max(int(self._audio_cfg.get("protocol_transient_steps", 3)), 0)
        self._audio_trace_history_size = max(int(self._audio_cfg.get("trace_history_size", 32)), 1)
        self._audio_goal_queue_size = max(int(self._audio_cfg.get("goal_queue_size", 24)), 1)
        self._audio_gcc_interp = max(int(self._audio_cfg.get("gcc_interp", 8)), 1)
        self._audio_bearing_peak_exclusion_bins = max(int(self._audio_cfg.get("bearing_peak_exclusion_bins", 8)), 1)
        self._audio_bearing_min_peak_ratio = float(self._audio_cfg.get("bearing_min_peak_ratio", 1.05))
        self._audio_bearing_min_confidence = float(self._audio_cfg.get("bearing_min_confidence", 0.15))
        self._audio_belief_ray_sigma_m = float(self._audio_cfg.get("belief_ray_sigma_m", 0.75))
        self._audio_belief_max_range_m = float(self._audio_cfg.get("belief_max_range_m", 0.0))
        self._audio_triangulation_min_rays = max(int(self._audio_cfg.get("triangulation_min_rays", 3)), 2)
        self._audio_triangulation_history_size = max(int(self._audio_cfg.get("triangulation_history_size", 24)), 1)
        self._audio_triangulation_max_condition_number = float(self._audio_cfg.get("triangulation_max_condition_number", 80.0))
        self._audio_triangulation_min_pair_angle_deg = float(self._audio_cfg.get("triangulation_min_pair_angle_deg", 5.0))
        self._audio_target_visual_score_threshold = float(self._audio_cfg.get("target_visual_score_threshold", 0.5))
        self._max_depth_m = float(self._depth_cfg.get("max_depth_m", 6.0))
        self._assume_normalized_depth = bool(self._depth_cfg.get("assume_normalized_depth", True))
        self._audio_grid_resolution_m = float(self._depth_cfg.get("map_resolution_m", 0.25))
        self._audio_grid_size = int(self._depth_cfg.get("map_size_cells", 256))
        self._siglip = SigLIPEncoder(goal_encoder_model_name, device)
        self._map = SemanticVoxelMap(self._depth_cfg, self._siglip)
        self._audio_prototype_library: Optional[Any] = None
        self._audio_candidates: Dict[str, np.ndarray] = {}
        self._audio_history: List[np.ndarray] = []
        self._audio_goal_order: Tuple[str, ...] = ()
        self._audio_goal_expected_labels: Dict[str, Tuple[str, ...]] = {}
        self._audio_goal_packet_queues: Dict[str, List[AudioObservationPacket]] = {}
        self._audio_goal_ray_queues: Dict[str, List[AudioRay2D]] = {}
        self._audio_goal_belief_maps: Dict[str, np.ndarray] = {}
        self._audio_trace: List[Dict[str, Any]] = []
        self._audio_origin_world = np.zeros(3, dtype=np.float32)
        self._audio_grid_x_world = np.zeros((self._audio_grid_size, self._audio_grid_size), dtype=np.float32)
        self._audio_grid_z_world = np.zeros((self._audio_grid_size, self._audio_grid_size), dtype=np.float32)
        self._goal_embeddings: Dict[str, np.ndarray] = {}
        self._last_output: Optional[PerceptionOutput] = None

    def reset(self, observations: Optional[Dict[str, Any]] = None) -> None:
        position = np.zeros(3, dtype=np.float32)
        if observations is not None:
            pose = extract_pose(observations)
            position = np.asarray(pose_to_position(pose), dtype=np.float32)
        self._map.reset(position)
        self._audio_origin_world = np.asarray(position, dtype=np.float32).copy()
        self._init_audio_grid_world()
        self._audio_candidates = {}
        self._audio_history = []
        self._audio_goal_order = ()
        self._audio_goal_expected_labels = {}
        self._audio_goal_packet_queues = {}
        self._audio_goal_ray_queues = {}
        self._audio_goal_belief_maps = {}
        self._audio_trace = []
        self._goal_embeddings = {}
        self._last_output = None

    def set_audio_runtime(self, prototype_library: Any) -> None:
        self._audio_prototype_library = prototype_library

    def set_audio_candidates(self, candidates: Dict[str, np.ndarray]) -> None:
        self._audio_candidates = {
            str(label): np.asarray(embedding, dtype=np.float32)
            for label, embedding in candidates.items()
        }

    def set_audio_goal_cycle(
        self,
        goal_ids: Sequence[str],
        goal_labels: Optional[Dict[str, Sequence[str]]] = None,
    ) -> None:
        self._audio_goal_order = tuple(str(goal_id) for goal_id in goal_ids)
        self._audio_goal_expected_labels = {}
        labels = goal_labels or {}
        for goal_id in self._audio_goal_order:
            values = labels.get(goal_id, ())
            if isinstance(values, str):
                values = (values,)
            self._audio_goal_expected_labels[goal_id] = tuple(
                str(value) for value in values if str(value).strip()
            )
        self._audio_goal_packet_queues = {goal_id: [] for goal_id in self._audio_goal_order}
        self._audio_goal_ray_queues = {goal_id: [] for goal_id in self._audio_goal_order}
        self._audio_goal_belief_maps = {
            goal_id: np.zeros((self._audio_grid_size, self._audio_grid_size), dtype=np.float32)
            for goal_id in self._audio_goal_order
        }
        self._audio_trace = []

    def audio_debug_state(self) -> Dict[str, Any]:
        return {
            "goal_order": list(self._audio_goal_order),
            "goal_expected_labels": {
                str(goal_id): list(labels)
                for goal_id, labels in self._audio_goal_expected_labels.items()
            },
            "queue_depths": self._current_queue_depths(),
            "spatial_summaries": self._current_spatial_summaries(),
            "recent_trace": list(self._audio_trace),
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
        map_state = self._map.build_state(self._goal_embeddings)
        audio_state = self._perceive_audio(observations, observation)
        audio_state.target_2p5d = self._build_audio_target_2p5d(map_state, audio_state)
        output = PerceptionOutput(
            int(step_index),
            map_state,
            audio_state,
        )
        map_summary = output.map_state.summary()
        map_summary.pop("goal_map_keys", None)
        map_summary.pop("goal_peak_keys", None)
        print(
            "[omega_nav][percept] "
            + json.dumps(
                {
                    "step_index": int(step_index),
                    "audio": output.audio_state.summary(),
                    "map": map_summary,
                },
                ensure_ascii=False,
            )
        )
        print('\n')
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

    def _perceive_audio(self, observations: Dict[str, Any], observation: MapObservation) -> AudioPerceptionState:
        protocol = self._build_audio_protocol_state(observation.step_index)
        audio = extract_audio(observations)
        if audio is None:
            state = AudioPerceptionState(
                score_threshold=float(self._audio_match_threshold),
                protocol=protocol,
                queue_depths=self._current_queue_depths(),
            )
            self._append_audio_trace(state)
            return state
        stereo = self._to_channel_first_audio(audio)
        mono = self._to_mono(stereo)
        packet = self._build_audio_packet(stereo, mono, observation)
        protocol.queue_depth = self._append_audio_packet(protocol.goal_id, packet)
        mono_signal = packet.mono_audio if isinstance(packet.mono_audio, np.ndarray) else mono
        rms = float(np.sqrt(np.mean(np.square(mono_signal)))) if mono_signal.size > 0 else 0.0
        peak = float(np.max(np.abs(mono_signal - float(np.mean(mono_signal))))) if mono_signal.size > 0 else 0.0
        bearing = estimate_horizontal_bearing(
            packet,
            protocol,
            ear_distance_m=float(self._audio_ear_distance_m),
            speed_of_sound_mps=float(self._audio_speed_of_sound_mps),
            reference_rms=float(self._audio_reference_rms),
            gcc_interp=int(self._audio_gcc_interp),
            peak_exclusion_bins=int(self._audio_bearing_peak_exclusion_bins),
            min_peak_ratio=float(self._audio_bearing_min_peak_ratio),
            min_confidence=float(self._audio_bearing_min_confidence),
        )
        ray = project_bearing_to_world_ray(packet, protocol, bearing)
        belief_state, triangulation = self._update_audio_spatial_state(protocol.goal_id, ray)
        if mono_signal.size == 0 or peak <= max(self._audio_reference_rms * 1e-4, 1e-8):
            self._audio_history = []
            state = AudioPerceptionState(
                is_active=True,
                rms=rms,
                peak=peak,
                score_threshold=float(self._audio_match_threshold),
                packet=packet,
                protocol=protocol,
                bearing=bearing,
                ray=ray,
                belief=belief_state,
                triangulation=triangulation,
                queue_depths=self._current_queue_depths(),
            )
            self._append_audio_trace(state)
            return state
        if self._audio_prototype_library is None:
            state = AudioPerceptionState(
                is_active=True,
                rms=rms,
                peak=peak,
                score_threshold=float(self._audio_match_threshold),
                packet=packet,
                protocol=protocol,
                bearing=bearing,
                ray=ray,
                belief=belief_state,
                triangulation=triangulation,
                queue_depths=self._current_queue_depths(),
            )
            self._append_audio_trace(state)
            return state
        self._audio_history.append(np.asarray(mono_signal, dtype=np.float32).reshape(-1))
        history_steps = max(int(round(self._audio_waveform_window_sec / self._audio_step_time_s)), 1)
        if len(self._audio_history) > history_steps:
            self._audio_history = self._audio_history[-history_steps:]
        waveform = np.concatenate(self._audio_history, axis=0).astype(np.float32)
        embedding = np.asarray(self._audio_prototype_library.encode_audio_observation(waveform), dtype=np.float32)
        scores, scope = self._score_audio(embedding)
        if not scores:
            state = AudioPerceptionState(
                is_active=True,
                rms=rms,
                peak=peak,
                score_threshold=float(self._audio_match_threshold),
                packet=packet,
                protocol=protocol,
                bearing=bearing,
                ray=ray,
                belief=belief_state,
                triangulation=triangulation,
                queue_depths=self._current_queue_depths(),
            )
            self._append_audio_trace(state)
            return state
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        top_scores = {str(label): float(score) for label, score in ranked[: self._audio_top_k]}
        top_label, top_score = ranked[0]
        label = str(top_label) if float(top_score) >= self._audio_match_threshold else ""
        state = AudioPerceptionState(
            is_active=True,
            rms=rms,
            peak=peak,
            label=label,
            score=float(top_score),
            score_threshold=float(self._audio_match_threshold),
            scores=top_scores,
            scope=str(scope),
            packet=packet,
            protocol=protocol,
            bearing=bearing,
            ray=ray,
            belief=belief_state,
            triangulation=triangulation,
            queue_depths=self._current_queue_depths(),
        )
        self._append_audio_trace(state)
        return state

    def _score_audio(self, embedding: np.ndarray) -> Tuple[Dict[str, float], str]:
        if len(self._audio_candidates) > 0:
            return {
                str(label): float(cosine_similarity(embedding, candidate))
                for label, candidate in self._audio_candidates.items()
            }, "episode"
        return self._audio_prototype_library.score_embedding(embedding), "global"

    def _build_audio_packet(
        self,
        stereo: np.ndarray,
        mono: np.ndarray,
        observation: MapObservation,
    ) -> AudioObservationPacket:
        audio = np.asarray(stereo, dtype=np.float32)
        channel_count = int(audio.shape[0]) if audio.ndim == 2 else 1
        frame_samples = int(audio.shape[1]) if audio.ndim == 2 else int(audio.size)
        configured_valid_samples = max(int(round(self._audio_sampling_rate_hz * self._audio_step_time_s)), 1)
        valid_samples = min(frame_samples, configured_valid_samples)
        trimmed_audio = np.asarray(audio[:, :valid_samples], dtype=np.float32) if audio.ndim == 2 else audio.reshape(1, -1)[:, :valid_samples]
        trimmed_mono = np.asarray(mono[:valid_samples], dtype=np.float32).reshape(-1)
        nonzero_samples = int(np.count_nonzero(np.any(np.abs(audio) > 1e-8, axis=0))) if audio.ndim == 2 else int(np.count_nonzero(np.abs(audio) > 1e-8))
        return AudioObservationPacket(
            step_index=int(observation.step_index),
            sample_rate_hz=int(self._audio_sampling_rate_hz),
            frame_samples=int(frame_samples),
            valid_samples=int(valid_samples),
            channel_count=int(channel_count),
            nonzero_samples=int(nonzero_samples),
            stereo_audio=trimmed_audio,
            mono_audio=trimmed_mono,
            position=np.asarray(observation.position, dtype=np.float32),
            heading_rad=float(observation.heading_rad),
        )

    def _build_audio_protocol_state(self, step_index: int) -> AudioProtocolState:
        cycle_steps = max(int(self._audio_protocol_cycle_steps), 1)
        transient_steps = max(int(self._audio_protocol_transient_steps), 0)
        slot_step = int(step_index % cycle_steps)
        window_index = int(step_index // cycle_steps)
        stable_start_step = int(min(transient_steps, cycle_steps))
        stable_stop = int(max(cycle_steps - transient_steps, stable_start_step))
        stable_end_step = int(stable_stop - 1) if stable_stop > stable_start_step else int(stable_start_step - 1)
        goal_count = len(self._audio_goal_order)
        is_stable = bool(goal_count > 0 and stable_start_step <= slot_step < stable_stop)
        is_transient = bool(goal_count > 0 and not is_stable)
        goal_index = -1
        goal_id = ""
        expected_labels: Tuple[str, ...] = ()
        if goal_count > 0:
            goal_index = int(window_index % goal_count)
            goal_id = str(self._audio_goal_order[goal_index])
            expected_labels = tuple(self._audio_goal_expected_labels.get(goal_id, ()))
        return AudioProtocolState(
            goal_id=goal_id,
            goal_index=int(goal_index),
            goal_count=int(goal_count),
            cycle_steps=int(cycle_steps),
            transient_steps=int(transient_steps),
            step_time_s=float(self._audio_step_time_s),
            window_index=int(window_index),
            slot_step=int(slot_step),
            stable_start_step=int(stable_start_step),
            stable_end_step=int(stable_end_step),
            is_transient=bool(is_transient),
            is_stable=bool(is_stable),
            expected_labels=expected_labels,
        )

    def _append_audio_packet(self, goal_id: str, packet: AudioObservationPacket) -> int:
        if not goal_id:
            return 0
        queue = self._audio_goal_packet_queues.setdefault(str(goal_id), [])
        queue.append(packet)
        if len(queue) > self._audio_goal_queue_size:
            del queue[:-self._audio_goal_queue_size]
        return int(len(queue))

    def _append_audio_trace(self, state: AudioPerceptionState) -> None:
        packet_summary = state.packet.summary() if state.packet is not None else {}
        protocol_summary = state.protocol.summary()
        heading_deg = packet_summary.get("heading_deg", 0.0)
        expected_labels = list(protocol_summary.get("expected_labels", []))
        label_matches_expected = None
        if expected_labels:
            label_matches_expected = bool(state.label) and str(state.label) in expected_labels
        entry = {
            "step_index": int(packet_summary.get("step_index", -1)),
            "goal_id": str(protocol_summary.get("goal_id", "")),
            "goal_index": int(protocol_summary.get("goal_index", -1)),
            "slot_step": int(protocol_summary.get("slot_step", -1)),
            "window_index": int(protocol_summary.get("window_index", -1)),
            "is_transient": bool(protocol_summary.get("is_transient", False)),
            "is_stable": bool(protocol_summary.get("is_stable", False)),
            "queue_depth": int(protocol_summary.get("queue_depth", 0)),
            "expected_labels": expected_labels,
            "audio_shape": list(packet_summary.get("stereo_shape", [])),
            "frame_samples": int(packet_summary.get("frame_samples", 0)),
            "valid_samples": int(packet_summary.get("valid_samples", 0)),
            "position": list(packet_summary.get("position", [])),
            "heading_deg": round(float(heading_deg), 2),
            "is_active": bool(state.is_active),
            "rms": float(state.rms),
            "peak": float(state.peak),
            "bearing_valid": bool(state.bearing.is_valid),
            "bearing_deg": float(state.bearing.relative_bearing_deg),
            "bearing_confidence": float(state.bearing.confidence),
            "bearing_reason": str(state.bearing.reason),
            "ray_valid": bool(state.ray.is_valid),
            "world_bearing_deg": float(state.ray.world_bearing_deg),
            "belief_peak_value": float(state.belief.peak_value),
            "belief_peak_world": list(state.belief.peak_world) if state.belief.peak_world is not None else [],
            "triangulation_valid": bool(state.triangulation.is_valid),
            "triangulation_point_xz": list(state.triangulation.point_xz) if state.triangulation.point_xz is not None else [],
            "triangulation_reason": str(state.triangulation.reason),
            "target_valid": bool(state.target_2p5d.is_valid),
            "target_xz_source": str(state.target_2p5d.xz_source),
            "target_height_source": str(state.target_2p5d.height_source),
            "target_xz": list(state.target_2p5d.target_xz) if state.target_2p5d.target_xz is not None else [],
            "target_xyz": list(state.target_2p5d.target_xyz) if state.target_2p5d.target_xyz is not None else [],
            "label": str(state.label),
            "score": float(state.score),
            "label_matches_expected": label_matches_expected,
        }
        self._audio_trace.append(entry)
        if len(self._audio_trace) > self._audio_trace_history_size:
            self._audio_trace = self._audio_trace[-self._audio_trace_history_size :]

    def _current_queue_depths(self) -> Dict[str, int]:
        return {
            str(goal_id): int(len(self._audio_goal_packet_queues.get(goal_id, [])))
            for goal_id in self._audio_goal_order
        }

    def _current_spatial_summaries(self) -> Dict[str, Dict[str, Any]]:
        summaries: Dict[str, Dict[str, Any]] = {}
        for goal_id in self._audio_goal_order:
            belief_map = self._audio_goal_belief_maps.get(goal_id)
            belief_state = summarize_belief_map(
                belief_map if belief_map is not None else np.zeros((self._audio_grid_size, self._audio_grid_size), dtype=np.float32),
                goal_id=str(goal_id),
                ray_count=len(self._audio_goal_ray_queues.get(goal_id, [])),
                origin_world=self._audio_origin_world,
                resolution_m=self._audio_grid_resolution_m,
                grid_size=self._audio_grid_size,
            )
            triangulation = triangulate_rays_2d(
                tuple(self._audio_goal_ray_queues.get(goal_id, [])),
                goal_id=str(goal_id),
                min_rays=int(self._audio_triangulation_min_rays),
                max_condition_number=float(self._audio_triangulation_max_condition_number),
                min_pair_angle_deg=float(self._audio_triangulation_min_pair_angle_deg),
            )
            summaries[str(goal_id)] = {
                "belief": belief_state.summary(),
                "triangulation": triangulation.summary(),
            }
        return summaries

    def _build_audio_target_2p5d(
        self,
        map_state: SemanticVoxelMapState,
        audio_state: AudioPerceptionState,
    ) -> AudioTarget2p5D:
        goal_id = str(audio_state.protocol.goal_id or "")
        if not goal_id:
            return AudioTarget2p5D(reason="missing_goal_id")
        target_xz = None
        xz_source = ""
        confidence = 0.0
        reason = "no_horizontal_estimate"
        if audio_state.triangulation.is_valid and audio_state.triangulation.point_xz is not None:
            target_xz = np.asarray(audio_state.triangulation.point_xz, dtype=np.float32).reshape(2)
            xz_source = "triangulation"
            confidence = float(audio_state.triangulation.confidence)
            reason = "ok"
        elif audio_state.belief.ray_count > 0 and audio_state.belief.peak_world is not None:
            target_xz = np.asarray(audio_state.belief.peak_world, dtype=np.float32).reshape(2)
            xz_source = "belief_peak"
            mass = max(float(audio_state.belief.total_mass), 1e-6)
            confidence = float(np.clip(float(audio_state.belief.peak_value) / mass, 0.0, 1.0))
            reason = "belief_only"
        goal_peak = map_state.goal_peaks.get(goal_id, {}) if isinstance(map_state.goal_peaks, dict) else {}
        visual_score = float(goal_peak.get("score", 0.0)) if isinstance(goal_peak, dict) else 0.0
        visual_position = np.asarray(goal_peak.get("position", []), dtype=np.float32).reshape(-1) if isinstance(goal_peak, dict) else np.zeros((0,), dtype=np.float32)
        visual_available = bool(visual_position.size >= 3 and visual_score >= float(self._audio_target_visual_score_threshold))
        if target_xz is None and visual_available:
            target_xz = np.asarray([float(visual_position[0]), float(visual_position[2])], dtype=np.float32)
            xz_source = "visual_fallback"
            confidence = float(np.clip(visual_score, 0.0, 1.0))
            reason = "visual_fallback"
        if target_xz is None:
            return AudioTarget2p5D(goal_id=goal_id, reason=reason, visual_score=visual_score)
        target_xyz = None
        height_source = "unknown"
        if visual_available:
            target_xyz = np.asarray([float(target_xz[0]), float(visual_position[1]), float(target_xz[1])], dtype=np.float32)
            height_source = "visual"
            confidence = max(float(confidence), float(np.clip(visual_score, 0.0, 1.0)))
        return AudioTarget2p5D(
            goal_id=goal_id,
            is_valid=True,
            reason=reason,
            xz_source=xz_source,
            height_source=height_source,
            target_xz=np.asarray(target_xz, dtype=np.float32),
            target_xyz=np.asarray(target_xyz, dtype=np.float32) if target_xyz is not None else None,
            confidence=float(np.clip(confidence, 0.0, 1.0)),
            visual_score=float(visual_score),
        )

    def _update_audio_spatial_state(self, goal_id: str, ray: AudioRay2D):
        goal_key = str(goal_id or "")
        if not goal_key:
            empty_belief = summarize_belief_map(
                np.zeros((self._audio_grid_size, self._audio_grid_size), dtype=np.float32),
                goal_id="",
                ray_count=0,
                origin_world=self._audio_origin_world,
                resolution_m=self._audio_grid_resolution_m,
                grid_size=self._audio_grid_size,
            )
            return empty_belief, triangulate_rays_2d(
                tuple(),
                goal_id="",
                min_rays=int(self._audio_triangulation_min_rays),
                max_condition_number=float(self._audio_triangulation_max_condition_number),
                min_pair_angle_deg=float(self._audio_triangulation_min_pair_angle_deg),
            )
        belief_map = self._audio_goal_belief_maps.setdefault(
            goal_key,
            np.zeros((self._audio_grid_size, self._audio_grid_size), dtype=np.float32),
        )
        ray_queue = self._audio_goal_ray_queues.setdefault(goal_key, [])
        if ray.is_valid:
            update_belief_map_from_ray(
                belief_map,
                self._audio_grid_x_world,
                self._audio_grid_z_world,
                ray,
                sigma_m=float(self._audio_belief_ray_sigma_m),
                max_range_m=float(self._resolved_belief_max_range_m()),
            )
            ray_queue.append(ray)
            if len(ray_queue) > self._audio_triangulation_history_size:
                del ray_queue[:-self._audio_triangulation_history_size]
        belief_state = summarize_belief_map(
            belief_map,
            goal_id=goal_key,
            ray_count=len(ray_queue),
            origin_world=self._audio_origin_world,
            resolution_m=self._audio_grid_resolution_m,
            grid_size=self._audio_grid_size,
        )
        triangulation = triangulate_rays_2d(
            tuple(ray_queue),
            goal_id=goal_key,
            min_rays=int(self._audio_triangulation_min_rays),
            max_condition_number=float(self._audio_triangulation_max_condition_number),
            min_pair_angle_deg=float(self._audio_triangulation_min_pair_angle_deg),
        )
        return belief_state, triangulation

    def _resolved_belief_max_range_m(self) -> float:
        if float(self._audio_belief_max_range_m) > 1e-6:
            return float(self._audio_belief_max_range_m)
        return float(self._audio_grid_resolution_m) * float(self._audio_grid_size) * 0.5

    def _init_audio_grid_world(self) -> None:
        half = int(self._audio_grid_size) // 2
        coords = (np.arange(self._audio_grid_size, dtype=np.float32) - float(half)) * float(self._audio_grid_resolution_m)
        x_world = float(self._audio_origin_world[0]) + coords
        z_world = float(self._audio_origin_world[2]) + coords
        self._audio_grid_x_world, self._audio_grid_z_world = np.meshgrid(x_world, z_world, indexing="ij")

    def _to_channel_first_audio(self, audio: np.ndarray) -> np.ndarray:
        array = np.asarray(audio, dtype=np.float32)
        if array.ndim == 1:
            return array.reshape(1, -1)
        if array.ndim == 2 and array.shape[0] <= 4:
            return array
        if array.ndim == 2 and array.shape[1] <= 4:
            return np.transpose(array, (1, 0))
        return array.reshape(1, -1)

    def _to_mono(self, audio: np.ndarray) -> np.ndarray:
        array = np.asarray(audio, dtype=np.float32)
        if array.ndim == 1:
            return array.reshape(-1)
        if array.ndim == 2 and array.shape[0] <= 4:
            return np.mean(array, axis=0, dtype=np.float32)
        if array.ndim == 2 and array.shape[1] <= 4:
            return np.mean(array, axis=1, dtype=np.float32)
        return array.reshape(-1)
