from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ss_baselines.omega_nav.perception.base import AudioBearingEstimate, AudioObservationPacket, AudioProtocolState
from ss_baselines.omega_nav.utils import coarse_direction_from_angle


_EPS = 1e-6


def _wrap_angle_deg(angle_deg: float) -> float:
    wrapped = (float(angle_deg) + 180.0) % 360.0 - 180.0
    return float(wrapped)


def _normalize_curve(values: np.ndarray) -> np.ndarray:
    curve = np.asarray(values, dtype=np.float32).reshape(-1)
    if curve.size == 0:
        return curve
    curve = np.maximum(curve, 0.0)
    max_value = float(np.max(curve))
    if max_value <= _EPS:
        return np.full(curve.shape, 1.0 / float(curve.size), dtype=np.float32)
    curve = curve / max_value
    curve = np.maximum(curve, _EPS)
    curve /= float(np.sum(curve))
    return curve.astype(np.float32)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    weight_array = np.asarray(weights, dtype=np.float32).reshape(-1)
    if array.size == 0:
        return 0.0
    weight_sum = float(np.sum(weight_array))
    if weight_sum <= _EPS:
        return float(np.mean(array))
    return float(np.sum(array * weight_array) / weight_sum)


def _gcc_phat_curve(
    left: np.ndarray,
    right: np.ndarray,
    *,
    sample_rate_hz: int,
    max_tau_s: float,
    interp: int,
    exclusion_bins: int,
) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    left_signal = np.asarray(left, dtype=np.float32).reshape(-1)
    right_signal = np.asarray(right, dtype=np.float32).reshape(-1)
    if left_signal.size == 0 or right_signal.size == 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32), 0.0, 0.0, 0.0
    length = int(left_signal.size + right_signal.size)
    spectrum_left = np.fft.rfft(left_signal, n=length)
    spectrum_right = np.fft.rfft(right_signal, n=length)
    cross_power = spectrum_left * np.conj(spectrum_right)
    magnitude = np.maximum(np.abs(cross_power), 1e-12)
    cross_correlation = np.fft.irfft(cross_power / magnitude, n=int(max(int(interp), 1) * length))
    max_shift = min(int(max(int(interp), 1) * sample_rate_hz * max_tau_s), int(max(int(interp), 1) * length // 2))
    if max_shift <= 0:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32), 0.0, 0.0, 0.0
    window = np.concatenate((cross_correlation[-max_shift:], cross_correlation[: max_shift + 1]))
    abs_window = np.abs(window).astype(np.float32)
    peak_index = int(np.argmax(abs_window))
    peak = float(abs_window[peak_index])
    lower = max(0, peak_index - int(exclusion_bins))
    upper = min(abs_window.size, peak_index + int(exclusion_bins) + 1)
    masked = abs_window.copy()
    masked[lower:upper] = 0.0
    second_peak = float(np.max(masked)) if masked.size > 0 else 0.0
    peak_ratio = float(peak / max(second_peak, _EPS))
    shift = int(peak_index - max_shift)
    tau_s = float(shift) / float(max(int(interp), 1) * float(sample_rate_hz))
    tau_axis_s = np.arange(-max_shift, max_shift + 1, dtype=np.float32) / float(max(int(interp), 1) * float(sample_rate_hz))
    return tau_axis_s, _normalize_curve(abs_window), tau_s, peak, peak_ratio


def _subframe_starts(total_samples: int, window_samples: int, hop_samples: int) -> Tuple[int, ...]:
    total = max(int(total_samples), 0)
    window = max(min(int(window_samples), total), 1)
    hop = max(int(hop_samples), 1)
    if total <= window:
        return (0,)
    starts = list(range(0, total - window + 1, hop))
    tail_start = int(total - window)
    if len(starts) == 0 or starts[-1] != tail_start:
        starts.append(tail_start)
    return tuple(int(start) for start in starts)


@dataclass
class FrameDelayEvidence:
    is_valid: bool = False
    reason: str = ""
    tau_axis_s: Optional[np.ndarray] = None
    curve: Optional[np.ndarray] = None
    peak_tau_s: float = 0.0
    peak: float = 0.0
    peak_ratio: float = 0.0
    confidence: float = 0.0
    valid_subframe_count: int = 0
    total_subframe_count: int = 0
    spread_deg: float = 0.0
    weight: float = 0.0


class WindowWorldAzimuthTracker:
    def __init__(
        self,
        *,
        angle_bin_deg: float,
        min_evidence_frames: int,
        min_heading_span_deg: float,
        min_confidence: float,
        max_posterior_entropy: float,
        background_history_size: int,
        background_min_history: int,
        background_strength: float,
    ) -> None:
        bin_size = max(float(angle_bin_deg), 0.5)
        self._angle_grid_deg = np.arange(-180.0, 180.0, bin_size, dtype=np.float32)
        if self._angle_grid_deg.size == 0:
            self._angle_grid_deg = np.asarray([0.0], dtype=np.float32)
        self._min_evidence_frames = max(int(min_evidence_frames), 1)
        self._min_heading_span_deg = max(float(min_heading_span_deg), 0.0)
        self._min_confidence = max(float(min_confidence), 0.0)
        self._max_posterior_entropy = float(np.clip(max_posterior_entropy, 0.0, 1.0))
        self._background_history_size = max(int(background_history_size), 1)
        self._background_min_history = max(int(background_min_history), 1)
        self._background_strength = float(np.clip(background_strength, 0.0, 1.0))
        self.reset()

    def reset(self) -> None:
        self.window_index = -1
        self._log_scores = np.zeros(self._angle_grid_deg.shape[0], dtype=np.float32)
        self._frame_count = 0
        self._evidence_count = 0
        self._heading_unwrapped_deg: list[float] = []
        self._last_heading_deg: Optional[float] = None
        self._curve_history: list[np.ndarray] = []
        self._last_tau_axis_s: Optional[np.ndarray] = None
        self._last_curve: Optional[np.ndarray] = None

    def ensure_window(self, window_index: int) -> None:
        index = int(window_index)
        if self.window_index == index:
            return
        self.reset()
        self.window_index = index

    def observe(self, *, heading_rad: float, evidence: FrameDelayEvidence, max_tau_s: float) -> None:
        heading_deg = float(np.degrees(float(heading_rad)))
        self._frame_count += 1
        if self._last_heading_deg is None:
            self._heading_unwrapped_deg.append(heading_deg)
            self._last_heading_deg = heading_deg
        else:
            delta_deg = _wrap_angle_deg(heading_deg - float(self._last_heading_deg))
            self._heading_unwrapped_deg.append(float(self._heading_unwrapped_deg[-1]) + float(delta_deg))
            self._last_heading_deg = heading_deg
        if not evidence.is_valid or evidence.curve is None or evidence.tau_axis_s is None or evidence.curve.size == 0:
            return
        tau_axis_s = np.asarray(evidence.tau_axis_s, dtype=np.float32)
        raw_curve = _normalize_curve(np.asarray(evidence.curve, dtype=np.float32))
        curve = self._apply_background_suppression(raw_curve)
        self._last_tau_axis_s = tau_axis_s.copy()
        self._last_curve = curve.copy()
        predicted_tau_s = float(max_tau_s) * np.sin(np.deg2rad(self._angle_grid_deg - heading_deg))
        likelihood = np.interp(
            predicted_tau_s,
            tau_axis_s,
            curve,
            left=_EPS,
            right=_EPS,
        )
        likelihood = np.clip(likelihood, _EPS, 1.0)
        self._log_scores += float(max(evidence.weight, 1e-3)) * np.log(likelihood)
        self._evidence_count += 1
        self._curve_history.append(raw_curve.copy())
        if len(self._curve_history) > self._background_history_size:
            self._curve_history = self._curve_history[-self._background_history_size :]

    def build_estimate(
        self,
        *,
        heading_rad: float,
        frame_evidence: FrameDelayEvidence,
        max_tau_s: float,
    ) -> AudioBearingEstimate:
        heading_deg = float(np.degrees(float(heading_rad)))
        posterior = self._posterior()
        best_index = int(np.argmax(posterior)) if posterior.size > 0 else 0
        if self._evidence_count > 0 and posterior.size > 0:
            world_bearing_deg = float(self._angle_grid_deg[best_index])
            posterior_peak = float(posterior[best_index])
            if posterior.size > 1:
                second_peak = float(np.partition(posterior, -2)[-2])
            else:
                second_peak = 0.0
            posterior_margin = float(max(posterior_peak - second_peak, 0.0))
            posterior_entropy = self._posterior_entropy(posterior)
        else:
            world_bearing_deg = 0.0
            posterior_peak = 0.0
            posterior_margin = 0.0
            posterior_entropy = 1.0
        world_bearing_rad = float(np.deg2rad(world_bearing_deg))
        relative_bearing_deg = _wrap_angle_deg(world_bearing_deg - heading_deg)
        relative_bearing_rad = float(np.deg2rad(relative_bearing_deg))
        predicted_tau_s = float(max_tau_s) * float(np.sin(relative_bearing_rad))
        observed_tau_s = float(frame_evidence.peak_tau_s)
        if self._last_tau_axis_s is not None and self._last_curve is not None and self._last_curve.size > 0:
            observed_index = int(np.argmax(self._last_curve))
            observed_tau_s = float(self._last_tau_axis_s[observed_index])
        heading_span_deg = self.heading_span_deg
        confidence = self._posterior_confidence(posterior_peak, posterior_margin, posterior_entropy, heading_span_deg)
        if self._evidence_count < self._min_evidence_frames:
            reason = "insufficient_evidence_frames"
            is_valid = False
        elif heading_span_deg < self._min_heading_span_deg:
            reason = "insufficient_heading_span"
            is_valid = False
        elif posterior_entropy > self._max_posterior_entropy or confidence < self._min_confidence:
            reason = "ambiguous_world_posterior"
            is_valid = False
        else:
            reason = "ok"
            is_valid = True
        return AudioBearingEstimate(
            is_valid=bool(is_valid),
            reason=str(reason),
            frame_reason=str(frame_evidence.reason),
            relative_bearing_rad=float(relative_bearing_rad),
            relative_bearing_deg=float(relative_bearing_deg),
            world_bearing_rad=float(world_bearing_rad),
            world_bearing_deg=float(world_bearing_deg),
            tau_s=float(observed_tau_s),
            tau_samples=float(frame_evidence.peak_tau_s * float(max(int(16000), 1))),
            predicted_tau_s=float(predicted_tau_s),
            predicted_tau_samples=float(predicted_tau_s * float(max(int(16000), 1))),
            max_tau_s=float(max_tau_s),
            peak=float(frame_evidence.peak),
            peak_ratio=float(frame_evidence.peak_ratio),
            confidence=float(confidence),
            posterior_peak=float(posterior_peak),
            posterior_entropy=float(posterior_entropy),
            posterior_margin=float(posterior_margin),
            heading_span_deg=float(heading_span_deg),
            evidence_frame_count=int(self._evidence_count),
            window_frame_count=int(self._frame_count),
            direction=str(coarse_direction_from_angle(relative_bearing_deg)),
            valid_subframe_count=int(frame_evidence.valid_subframe_count),
            total_subframe_count=int(frame_evidence.total_subframe_count),
            spread_deg=float(frame_evidence.spread_deg),
        )

    @property
    def heading_span_deg(self) -> float:
        if len(self._heading_unwrapped_deg) < 2:
            return 0.0
        return float(max(self._heading_unwrapped_deg) - min(self._heading_unwrapped_deg))

    def debug_state(self) -> Dict[str, float]:
        posterior = self._posterior()
        best_index = int(np.argmax(posterior)) if posterior.size > 0 else 0
        return {
            "window_index": int(self.window_index),
            "frame_count": int(self._frame_count),
            "evidence_count": int(self._evidence_count),
            "background_history_count": int(len(self._curve_history)),
            "heading_span_deg": round(float(self.heading_span_deg), 2),
            "world_bearing_deg": round(float(self._angle_grid_deg[best_index]), 2) if posterior.size > 0 else 0.0,
            "posterior_peak": round(float(posterior[best_index]), 4) if posterior.size > 0 else 0.0,
            "posterior_entropy": round(float(self._posterior_entropy(posterior)), 4),
        }

    def _apply_background_suppression(self, curve: np.ndarray) -> np.ndarray:
        raw_curve = _normalize_curve(np.asarray(curve, dtype=np.float32))
        if len(self._curve_history) < self._background_min_history:
            return raw_curve
        history = np.stack(self._curve_history, axis=0).astype(np.float32)
        background_curve = np.median(history, axis=0).astype(np.float32)
        residual_curve = raw_curve - float(self._background_strength) * background_curve
        residual_curve = np.maximum(residual_curve, 0.0)
        if float(np.sum(residual_curve)) <= _EPS:
            return raw_curve
        return _normalize_curve(residual_curve)

    def _posterior(self) -> np.ndarray:
        if self._log_scores.size == 0:
            return np.zeros(0, dtype=np.float32)
        normalized = self._log_scores - float(np.max(self._log_scores))
        posterior = np.exp(normalized).astype(np.float32)
        total = float(np.sum(posterior))
        if total <= _EPS:
            return np.full(self._log_scores.shape, 1.0 / float(self._log_scores.size), dtype=np.float32)
        return posterior / total

    def _posterior_entropy(self, posterior: np.ndarray) -> float:
        if posterior.size <= 1:
            return 0.0
        array = np.clip(np.asarray(posterior, dtype=np.float32), _EPS, 1.0)
        entropy = float(-np.sum(array * np.log(array)))
        normalizer = float(np.log(float(array.size)))
        if normalizer <= _EPS:
            return 0.0
        return float(np.clip(entropy / normalizer, 0.0, 1.0))

    def _posterior_confidence(
        self,
        posterior_peak: float,
        posterior_margin: float,
        posterior_entropy: float,
        heading_span_deg: float,
    ) -> float:
        uniform_peak = 1.0 / float(max(self._angle_grid_deg.size, 1))
        peak_term = float(np.clip((float(posterior_peak) - uniform_peak) / max(1.0 - uniform_peak, _EPS), 0.0, 1.0))
        margin_term = float(np.clip(float(posterior_margin) / max(float(posterior_peak), _EPS), 0.0, 1.0))
        entropy_term = float(np.clip(1.0 - float(posterior_entropy), 0.0, 1.0))
        heading_term = float(
            np.clip(float(heading_span_deg) / max(float(self._min_heading_span_deg), 1.0), 0.0, 1.0)
        )
        evidence_term = float(
            np.clip(float(self._evidence_count) / max(float(self._min_evidence_frames), 1.0), 0.0, 1.0)
        )
        return float(
            np.clip(
                0.35 * peak_term + 0.25 * margin_term + 0.2 * entropy_term + 0.1 * heading_term + 0.1 * evidence_term,
                0.0,
                1.0,
            )
        )


def _build_frame_delay_evidence(
    packet: AudioObservationPacket,
    protocol: AudioProtocolState,
    *,
    ear_distance_m: float,
    speed_of_sound_mps: float,
    reference_rms: float,
    gcc_interp: int,
    peak_exclusion_bins: int,
    min_peak_ratio: float,
    min_confidence: float,
    subframe_samples: int,
    subframe_hop_samples: int,
    min_valid_subframes: int,
    max_spread_deg: float,
) -> FrameDelayEvidence:
    if not protocol.is_stable:
        return FrameDelayEvidence(reason="transient_window")
    if packet is None or not isinstance(packet.stereo_audio, np.ndarray):
        return FrameDelayEvidence(reason="missing_audio")
    stereo = np.asarray(packet.stereo_audio, dtype=np.float32)
    if stereo.ndim != 2 or stereo.shape[0] < 2:
        return FrameDelayEvidence(reason="insufficient_channels")
    if stereo.shape[1] < 32:
        return FrameDelayEvidence(reason="too_few_samples")
    if float(ear_distance_m) <= _EPS or float(speed_of_sound_mps) <= _EPS:
        return FrameDelayEvidence(reason="invalid_geometry")

    mono = (
        np.asarray(packet.mono_audio, dtype=np.float32).reshape(-1)
        if isinstance(packet.mono_audio, np.ndarray)
        else np.mean(stereo, axis=0, dtype=np.float32)
    )
    mono_peak = float(np.max(np.abs(mono - float(np.mean(mono))))) if mono.size > 0 else 0.0
    if mono.size == 0 or mono_peak <= max(float(reference_rms) * 1e-4, 1e-8):
        return FrameDelayEvidence(reason="low_energy")

    max_tau_s = float(ear_distance_m) / float(speed_of_sound_mps)
    total_samples = int(stereo.shape[1])
    window_samples = max(min(int(subframe_samples), total_samples), 32)
    hop_samples = max(int(subframe_hop_samples), 1)
    starts = _subframe_starts(total_samples, window_samples, hop_samples)
    sample_rate_hz = max(int(packet.sample_rate_hz), 1)
    energy_floor = max(float(reference_rms) * 1e-4, 1e-8)
    candidates = []
    valid_candidates = []

    for start in starts:
        end = int(start + window_samples)
        left = np.asarray(stereo[0, start:end], dtype=np.float32)
        right = np.asarray(stereo[1, start:end], dtype=np.float32)
        sub_mono = 0.5 * (left + right)
        sub_peak = float(np.max(np.abs(sub_mono - float(np.mean(sub_mono))))) if sub_mono.size > 0 else 0.0
        if sub_mono.size < 32 or sub_peak <= energy_floor:
            continue
        tau_axis_s, curve, peak_tau_s, peak, peak_ratio = _gcc_phat_curve(
            left,
            right,
            sample_rate_hz=sample_rate_hz,
            max_tau_s=max_tau_s,
            interp=max(int(gcc_interp), 1),
            exclusion_bins=max(int(peak_exclusion_bins), 1),
        )
        if curve.size == 0:
            continue
        peak_tau_s = float(np.clip(peak_tau_s, -max_tau_s, max_tau_s))
        sine_value = float(np.clip(peak_tau_s * float(speed_of_sound_mps) / float(ear_distance_m), -1.0, 1.0))
        bearing_deg = float(np.degrees(float(math.asin(sine_value))))
        peak_term = float(np.clip(peak / 0.2, 0.0, 1.0))
        peak_ratio_term = float(np.clip((peak_ratio - 1.0) / 2.0, 0.0, 1.0))
        confidence = float(np.clip(0.45 * peak_term + 0.55 * peak_ratio_term, 0.0, 1.0))
        candidate = {
            "tau_axis_s": tau_axis_s,
            "curve": curve,
            "peak_tau_s": float(peak_tau_s),
            "bearing_deg": float(bearing_deg),
            "peak": float(peak),
            "peak_ratio": float(peak_ratio),
            "confidence": float(confidence),
            "weight": float(max(confidence, 1e-3)),
        }
        candidates.append(candidate)
        if peak_ratio >= float(min_peak_ratio) and confidence >= float(min_confidence):
            valid_candidates.append(candidate)

    total_subframe_count = int(len(candidates))
    if total_subframe_count == 0:
        return FrameDelayEvidence(reason="low_energy")
    aggregate_set = valid_candidates if len(valid_candidates) > 0 else candidates
    weights = np.asarray([float(item["weight"]) for item in aggregate_set], dtype=np.float32)
    tau_axis_s = np.asarray(aggregate_set[0]["tau_axis_s"], dtype=np.float32)
    curve_stack = np.stack([np.asarray(item["curve"], dtype=np.float32) for item in aggregate_set], axis=0)
    if float(np.sum(weights)) <= _EPS:
        aggregate_curve = np.mean(curve_stack, axis=0)
    else:
        aggregate_curve = np.average(curve_stack, axis=0, weights=weights)
    aggregate_curve = _normalize_curve(aggregate_curve)
    peak_index = int(np.argmax(aggregate_curve))
    peak_tau_s = float(tau_axis_s[peak_index]) if tau_axis_s.size > 0 else 0.0
    peak_values = np.asarray([float(item["peak"]) for item in aggregate_set], dtype=np.float32)
    peak_ratio_values = np.asarray([float(item["peak_ratio"]) for item in aggregate_set], dtype=np.float32)
    confidence_values = np.asarray([float(item["confidence"]) for item in aggregate_set], dtype=np.float32)
    bearing_values_deg = np.asarray([float(item["bearing_deg"]) for item in aggregate_set], dtype=np.float32)
    relative_bearing_deg = float(np.degrees(np.arcsin(np.clip(peak_tau_s * float(speed_of_sound_mps) / float(ear_distance_m), -1.0, 1.0))))
    spread_deg = (
        float(np.sqrt(np.average(np.square(bearing_values_deg - relative_bearing_deg), weights=weights)))
        if bearing_values_deg.size > 0
        else 0.0
    )
    peak = float(_weighted_mean(peak_values, weights))
    peak_ratio = float(_weighted_mean(peak_ratio_values, weights))
    confidence = float(_weighted_mean(confidence_values, weights))
    valid_subframe_count = int(len(valid_candidates))
    required_valid_subframes = min(max(int(min_valid_subframes), 1), max(total_subframe_count, 1))
    if valid_subframe_count < required_valid_subframes:
        reason = "insufficient_valid_subframes"
        is_valid = False
    elif spread_deg > float(max_spread_deg):
        reason = "inconsistent_subframes"
        is_valid = False
    else:
        reason = "ok"
        is_valid = True
    valid_fraction = float(valid_subframe_count) / float(max(total_subframe_count, 1))
    spread_term = 1.0
    if float(max_spread_deg) > 0.0:
        spread_term = float(np.clip(1.0 - spread_deg / float(max_spread_deg), 0.0, 1.0))
    weight = float(np.clip(confidence * (0.5 + 0.5 * valid_fraction) * (0.5 + 0.5 * spread_term), 0.0, 1.0))
    return FrameDelayEvidence(
        is_valid=bool(is_valid),
        reason=str(reason),
        tau_axis_s=tau_axis_s,
        curve=aggregate_curve,
        peak_tau_s=float(peak_tau_s),
        peak=float(peak),
        peak_ratio=float(peak_ratio),
        confidence=float(confidence),
        valid_subframe_count=int(valid_subframe_count),
        total_subframe_count=int(total_subframe_count),
        spread_deg=float(spread_deg),
        weight=float(weight),
    )


def estimate_world_bearing(
    packet: AudioObservationPacket,
    protocol: AudioProtocolState,
    *,
    tracker: WindowWorldAzimuthTracker,
    ear_distance_m: float,
    speed_of_sound_mps: float,
    reference_rms: float,
    gcc_interp: int,
    peak_exclusion_bins: int,
    min_peak_ratio: float,
    min_confidence: float,
    subframe_samples: int,
    subframe_hop_samples: int,
    min_valid_subframes: int,
    max_spread_deg: float,
) -> AudioBearingEstimate:
    if packet is None:
        return AudioBearingEstimate(reason="missing_audio", frame_reason="missing_audio")
    if not protocol.is_stable:
        return AudioBearingEstimate(reason="transient_window", frame_reason="transient_window")
    tracker.ensure_window(protocol.window_index)
    frame_evidence = _build_frame_delay_evidence(
        packet,
        protocol,
        ear_distance_m=ear_distance_m,
        speed_of_sound_mps=speed_of_sound_mps,
        reference_rms=reference_rms,
        gcc_interp=gcc_interp,
        peak_exclusion_bins=peak_exclusion_bins,
        min_peak_ratio=min_peak_ratio,
        min_confidence=min_confidence,
        subframe_samples=subframe_samples,
        subframe_hop_samples=subframe_hop_samples,
        min_valid_subframes=min_valid_subframes,
        max_spread_deg=max_spread_deg,
    )
    max_tau_s = float(ear_distance_m) / float(speed_of_sound_mps)
    tracker.observe(heading_rad=float(packet.heading_rad), evidence=frame_evidence, max_tau_s=max_tau_s)
    estimate = tracker.build_estimate(
        heading_rad=float(packet.heading_rad),
        frame_evidence=frame_evidence,
        max_tau_s=max_tau_s,
    )
    estimate.tau_samples = float(estimate.tau_s * float(max(int(packet.sample_rate_hz), 1)))
    estimate.predicted_tau_samples = float(estimate.predicted_tau_s * float(max(int(packet.sample_rate_hz), 1)))
    return estimate


__all__ = ["WindowWorldAzimuthTracker", "estimate_world_bearing"]
