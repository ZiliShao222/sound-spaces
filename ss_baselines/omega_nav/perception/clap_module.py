from __future__ import annotations

from collections import deque
from typing import Any, Deque, Dict, Optional, Sequence, Set, Tuple

import numpy as np

from ss_baselines.omega_nav.perception.base import AudioMatch, GoalSpec
from ss_baselines.omega_nav.utils import chinese_direction_from_angle


class OracleCLAPMatcher:
    def __init__(self, config: Dict[str, Any]):
        self._aggregation_window = max(int(config.get("aggregation_window", 10)), 1)
        self._threshold = float(config.get("detection_threshold", 0.65))
        self._ear_distance_m = float(config.get("ear_distance_m", 0.18))
        self._speed_of_sound_mps = float(config.get("speed_of_sound_mps", 343.0))
        self._reference_rms = max(float(config.get("reference_rms", 0.05)), 1e-4)
        self._sampling_rate_hz = max(int(config.get("sampling_rate_hz", 16000)), 1)
        self._background_similarity = float(config.get("background_similarity", 0.05))
        self._distance_scale_m = max(float(config.get("distance_scale_m", 1.0)), 1e-3)
        self._similarity_history: Dict[str, Deque[float]] = {}

    def reset(self, goal_specs: Sequence[GoalSpec]) -> None:
        self._similarity_history = {
            goal.goal_id: deque(maxlen=self._aggregation_window)
            for goal in goal_specs
        }

    def _audio_observation(self, observations: Dict[str, Any]) -> Optional[np.ndarray]:
        audio = observations.get("audiogoal")
        if audio is None:
            return None
        array = np.asarray(audio, dtype=np.float32)
        if array.ndim == 1:
            array = np.stack([array, array], axis=0)
        elif array.ndim == 2 and array.shape[0] != 2 and array.shape[1] == 2:
            array = array.T
        if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] <= 0:
            return None
        return array[:2]

    def _audio_energy_factor(self, audio: Optional[np.ndarray]) -> float:
        if audio is None:
            return 0.0
        rms = float(np.sqrt(np.mean(np.square(audio))))
        return float(np.clip(rms / self._reference_rms, 0.0, 1.0))

    def _relative_angle(self, audio: Optional[np.ndarray]) -> Tuple[Optional[float], Optional[float]]:
        if audio is None:
            return None, None
        left = np.asarray(audio[0], dtype=np.float32)
        right = np.asarray(audio[1], dtype=np.float32)
        if left.size == 0 or right.size == 0:
            return None, None

        left_zm = left - float(np.mean(left))
        right_zm = right - float(np.mean(right))
        max_lag = max(int(np.ceil((self._ear_distance_m / self._speed_of_sound_mps) * self._sampling_rate_hz * 1.5)), 1)
        corr = np.correlate(left_zm, right_zm, mode="full")
        center = right_zm.size - 1
        lo = max(0, center - max_lag)
        hi = min(corr.size, center + max_lag + 1)
        window = corr[lo:hi]
        lag = 0
        if window.size > 0:
            lag = int(np.argmax(window)) + lo - center
        itd_seconds = float(lag) / float(self._sampling_rate_hz)
        itd_ratio = np.clip(
            -itd_seconds * self._speed_of_sound_mps / max(self._ear_distance_m, 1e-6),
            -1.0,
            1.0,
        )
        itd_angle = float(np.degrees(np.arcsin(itd_ratio)))

        left_rms = float(np.sqrt(np.mean(np.square(left)))) + 1e-8
        right_rms = float(np.sqrt(np.mean(np.square(right)))) + 1e-8
        ild_db = 20.0 * float(np.log10(left_rms / right_rms))
        ild_angle = float(np.clip(-12.0 * ild_db, -90.0, 90.0))

        angle_deg = itd_angle if abs(itd_angle) >= 5.0 else ild_angle
        if abs(angle_deg) < 5.0 and abs(ild_angle) >= 5.0:
            angle_deg = ild_angle
        return float(angle_deg), float(itd_seconds)

    def _tracked_goal_state(
        self,
        goal_specs: Sequence[GoalSpec],
        pending_goal_ids: Optional[Sequence[str]],
        order_mode: str,
    ) -> Tuple[Set[str], Optional[str]]:
        tracked_goal_ids = set(pending_goal_ids or [goal.goal_id for goal in goal_specs])
        if str(order_mode).strip().lower() == "ordered" and pending_goal_ids:
            return tracked_goal_ids, str(pending_goal_ids[0])
        return tracked_goal_ids, None

    def match(
        self,
        env: Any,
        episode: Any,
        observations: Dict[str, Any],
        goal_specs: Sequence[GoalSpec],
        *,
        pending_goal_ids: Optional[Sequence[str]] = None,
        order_mode: str = "ordered",
    ) -> Dict[str, AudioMatch]:
        del env, episode
        if set(self._similarity_history.keys()) != {goal.goal_id for goal in goal_specs}:
            self.reset(goal_specs)

        audio = self._audio_observation(observations)
        base_similarity = self._audio_energy_factor(audio)
        angle_deg, itd_seconds = self._relative_angle(audio)
        direction_text = chinese_direction_from_angle(angle_deg) if angle_deg is not None else "未知"
        distance_m = None
        if base_similarity > 0.0:
            distance_m = float(np.clip(self._distance_scale_m / max(base_similarity, 1e-3), 0.25, 10.0))

        tracked_goal_ids, focus_goal_id = self._tracked_goal_state(goal_specs, pending_goal_ids, order_mode)
        ordered = str(order_mode).strip().lower() == "ordered"
        results: Dict[str, AudioMatch] = {}
        for goal in goal_specs:
            history = self._similarity_history.setdefault(goal.goal_id, deque(maxlen=self._aggregation_window))
            similarity = 0.0
            if goal.goal_id in tracked_goal_ids and ordered and focus_goal_id is not None:
                similarity = base_similarity if goal.goal_id == focus_goal_id else self._background_similarity * base_similarity
            history.append(float(similarity))
            aggregated = float(np.median(np.asarray(list(history), dtype=np.float32))) if history else float(similarity)
            detected = bool(goal.goal_id == focus_goal_id and aggregated >= self._threshold)
            results[goal.goal_id] = AudioMatch(
                goal_id=goal.goal_id,
                goal_index=int(goal.goal_index),
                category=str(goal.category),
                similarity=float(similarity),
                aggregated_similarity=float(aggregated),
                detected=detected,
                direction_text=str(direction_text),
                relative_angle_deg=float(angle_deg) if angle_deg is not None else None,
                itd_seconds=itd_seconds,
                distance_m=distance_m,
                sound_position=None,
            )
        return results
