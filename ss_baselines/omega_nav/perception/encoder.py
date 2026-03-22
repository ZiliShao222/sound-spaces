from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ss_baselines.omega_nav.audio_bearing import WindowWorldAzimuthTracker, estimate_world_bearing
from ss_baselines.omega_nav.perception.base import (
    AudioObservationPacket,
    AudioPerceptionState,
    AudioProtocolState,
    MapObservation,
    PerceptionOutput,
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
        self._audio_gcc_interp = max(int(self._audio_cfg.get("gcc_interp", 8)), 1)
        self._audio_bearing_peak_exclusion_bins = max(int(self._audio_cfg.get("bearing_peak_exclusion_bins", 8)), 1)
        self._audio_bearing_min_peak_ratio = float(self._audio_cfg.get("bearing_min_peak_ratio", 1.05))
        self._audio_bearing_min_confidence = float(self._audio_cfg.get("bearing_min_confidence", 0.15))
        self._audio_bearing_subframe_samples = max(int(self._audio_cfg.get("bearing_subframe_samples", 1024)), 32)
        self._audio_bearing_subframe_hop_samples = max(int(self._audio_cfg.get("bearing_subframe_hop_samples", 512)), 1)
        self._audio_bearing_min_valid_subframes = max(int(self._audio_cfg.get("bearing_min_valid_subframes", 2)), 1)
        self._audio_bearing_max_spread_deg = max(float(self._audio_cfg.get("bearing_max_spread_deg", 12.0)), 0.0)
        self._audio_world_angle_bin_deg = max(float(self._audio_cfg.get("world_tracker_angle_bin_deg", 2.0)), 0.5)
        self._audio_world_min_evidence_frames = max(int(self._audio_cfg.get("world_tracker_min_evidence_frames", 3)), 1)
        self._audio_world_min_heading_span_deg = max(
            float(self._audio_cfg.get("world_tracker_min_heading_span_deg", 45.0)),
            0.0,
        )
        self._audio_world_min_confidence = max(float(self._audio_cfg.get("world_tracker_min_confidence", 0.3)), 0.0)
        self._audio_world_max_posterior_entropy = float(
            np.clip(self._audio_cfg.get("world_tracker_max_posterior_entropy", 0.98), 0.0, 1.0)
        )
        self._audio_world_background_history_size = max(
            int(self._audio_cfg.get("world_tracker_background_history_size", 8)),
            1,
        )
        self._audio_world_background_min_history = max(
            int(self._audio_cfg.get("world_tracker_background_min_history", 2)),
            1,
        )
        self._audio_world_background_strength = float(
            np.clip(self._audio_cfg.get("world_tracker_background_strength", 0.75), 0.0, 1.0)
        )
        self._max_depth_m = float(self._depth_cfg.get("max_depth_m", 6.0))
        self._assume_normalized_depth = bool(self._depth_cfg.get("assume_normalized_depth", True))
        self._siglip = SigLIPEncoder(goal_encoder_model_name, device)
        self._map = SemanticVoxelMap(self._depth_cfg, self._siglip)
        self._audio_world_tracker = WindowWorldAzimuthTracker(
            angle_bin_deg=float(self._audio_world_angle_bin_deg),
            min_evidence_frames=int(self._audio_world_min_evidence_frames),
            min_heading_span_deg=float(self._audio_world_min_heading_span_deg),
            min_confidence=float(self._audio_world_min_confidence),
            max_posterior_entropy=float(self._audio_world_max_posterior_entropy),
            background_history_size=int(self._audio_world_background_history_size),
            background_min_history=int(self._audio_world_background_min_history),
            background_strength=float(self._audio_world_background_strength),
        )
        self._audio_prototype_library: Optional[Any] = None
        self._audio_candidates: Dict[str, np.ndarray] = {}
        self._audio_history: List[np.ndarray] = []
        self._audio_goal_order: Tuple[str, ...] = ()
        self._audio_goal_expected_labels: Dict[str, Tuple[str, ...]] = {}
        self._audio_trace: List[Dict[str, Any]] = []
        self._goal_embeddings: Dict[str, np.ndarray] = {}
        self._last_output: Optional[PerceptionOutput] = None
        self._episode_index = -1

    def reset(self, observations: Optional[Dict[str, Any]] = None) -> None:
        position = np.zeros(3, dtype=np.float32)
        if observations is not None:
            pose = extract_pose(observations)
            position = np.asarray(pose_to_position(pose), dtype=np.float32)
        self._map.reset(position)
        self._episode_index += 1
        self._audio_candidates = {}
        self._audio_history = []
        self._audio_goal_order = ()
        self._audio_goal_expected_labels = {}
        self._audio_trace = []
        self._audio_world_tracker.reset()
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
        self._audio_trace = []
        self._audio_world_tracker.reset()

    def audio_debug_state(self) -> Dict[str, Any]:
        return {
            "goal_order": list(self._audio_goal_order),
            "goal_expected_labels": {
                str(goal_id): list(labels)
                for goal_id, labels in self._audio_goal_expected_labels.items()
            },
            "audio_map": self._map.audio_summary(),
            "world_bearing_tracker": self._audio_world_tracker.debug_state(),
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
        audio_state = self._perceive_audio(observations, observation)
        map_state = self._map.build_state(self._goal_embeddings)
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
            )
            self._append_audio_trace(state)
            return state
        stereo = self._to_channel_first_audio(audio)
        mono = self._to_mono(stereo)
        packet = self._build_audio_packet(stereo, mono, observation)
        mono_signal = packet.mono_audio if isinstance(packet.mono_audio, np.ndarray) else mono
        rms = float(np.sqrt(np.mean(np.square(mono_signal)))) if mono_signal.size > 0 else 0.0
        peak = float(np.max(np.abs(mono_signal - float(np.mean(mono_signal))))) if mono_signal.size > 0 else 0.0
        bearing = estimate_world_bearing(
            packet,
            protocol,
            tracker=self._audio_world_tracker,
            ear_distance_m=float(self._audio_ear_distance_m),
            speed_of_sound_mps=float(self._audio_speed_of_sound_mps),
            reference_rms=float(self._audio_reference_rms),
            gcc_interp=int(self._audio_gcc_interp),
            peak_exclusion_bins=int(self._audio_bearing_peak_exclusion_bins),
            min_peak_ratio=float(self._audio_bearing_min_peak_ratio),
            min_confidence=float(self._audio_bearing_min_confidence),
            subframe_samples=int(self._audio_bearing_subframe_samples),
            subframe_hop_samples=int(self._audio_bearing_subframe_hop_samples),
            min_valid_subframes=int(self._audio_bearing_min_valid_subframes),
            max_spread_deg=float(self._audio_bearing_max_spread_deg),
        )
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
            )
            return self._finalize_audio_state(state)
        if self._audio_prototype_library is None:
            state = AudioPerceptionState(
                is_active=True,
                rms=rms,
                peak=peak,
                score_threshold=float(self._audio_match_threshold),
                packet=packet,
                protocol=protocol,
                bearing=bearing,
            )
            return self._finalize_audio_state(state)
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
            )
            return self._finalize_audio_state(state)
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
        )
        return self._finalize_audio_state(state)

    def _finalize_audio_state(self, state: AudioPerceptionState) -> AudioPerceptionState:
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
        stable_stop = int(cycle_steps)
        stable_end_step = int(cycle_steps - 1) if stable_stop > stable_start_step else int(stable_start_step - 1)
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
            "slot_step": int(protocol_summary.get("slot_step", -1)),
            "window_index": int(protocol_summary.get("window_index", -1)),
            "is_transient": bool(protocol_summary.get("is_transient", False)),
            "is_stable": bool(protocol_summary.get("is_stable", False)),
            "expected_labels": expected_labels,
            "heading_deg": round(float(heading_deg), 2),
            "is_active": bool(state.is_active),
            "rms": float(state.rms),
            "peak": float(state.peak),
            "bearing_valid": bool(state.bearing.is_valid),
            "world_bearing_deg": float(state.bearing.world_bearing_deg),
            "relative_bearing_deg": float(state.bearing.relative_bearing_deg),
            "bearing_confidence": float(state.bearing.confidence),
            "bearing_reason": str(state.bearing.reason),
            "bearing_frame_reason": str(state.bearing.frame_reason),
            "bearing_heading_span_deg": float(state.bearing.heading_span_deg),
            "bearing_posterior_peak": float(state.bearing.posterior_peak),
            "bearing_posterior_entropy": float(state.bearing.posterior_entropy),
            "bearing_posterior_margin": float(state.bearing.posterior_margin),
            "bearing_evidence_frames": int(state.bearing.evidence_frame_count),
            "bearing_valid_subframes": int(state.bearing.valid_subframe_count),
            "bearing_total_subframes": int(state.bearing.total_subframe_count),
            "bearing_spread_deg": float(state.bearing.spread_deg),
            "label": str(state.label),
            "score": float(state.score),
            "label_matches_expected": label_matches_expected,
        }
        self._audio_trace.append(entry)
        if len(self._audio_trace) > self._audio_trace_history_size:
            self._audio_trace = self._audio_trace[-self._audio_trace_history_size :]

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
