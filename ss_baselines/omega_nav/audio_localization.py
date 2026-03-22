from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from ss_baselines.omega_nav.perception.base import (
    AudioBearingEstimate,
    AudioBelief2DState,
    AudioObservationPacket,
    AudioProtocolState,
    AudioRay2D,
    AudioTriangulation2D,
)
from ss_baselines.omega_nav.utils import coarse_direction_from_angle


def wrap_angle_rad(angle_rad: float) -> float:
    angle = float(angle_rad)
    while angle <= -math.pi:
        angle += 2.0 * math.pi
    while angle > math.pi:
        angle -= 2.0 * math.pi
    return angle


def _gcc_phat_delay(
    left: np.ndarray,
    right: np.ndarray,
    *,
    sample_rate_hz: int,
    max_tau_s: float,
    interp: int,
    exclusion_bins: int,
) -> Tuple[float, float, float]:
    left_signal = np.asarray(left, dtype=np.float32).reshape(-1)
    right_signal = np.asarray(right, dtype=np.float32).reshape(-1)
    if left_signal.size == 0 or right_signal.size == 0:
        return 0.0, 0.0, 0.0
    length = int(left_signal.size + right_signal.size)
    spectrum_left = np.fft.rfft(left_signal, n=length)
    spectrum_right = np.fft.rfft(right_signal, n=length)
    cross_power = spectrum_left * np.conj(spectrum_right)
    magnitude = np.abs(cross_power)
    magnitude = np.maximum(magnitude, 1e-12)
    cross_correlation = np.fft.irfft(cross_power / magnitude, n=int(interp * length))
    max_shift = min(int(interp * sample_rate_hz * max_tau_s), int(interp * length // 2))
    if max_shift <= 0:
        return 0.0, 0.0, 0.0
    window = np.concatenate((cross_correlation[-max_shift:], cross_correlation[: max_shift + 1]))
    abs_window = np.abs(window)
    peak_index = int(np.argmax(abs_window))
    peak = float(abs_window[peak_index])
    lower = max(0, peak_index - int(exclusion_bins))
    upper = min(abs_window.size, peak_index + int(exclusion_bins) + 1)
    masked = abs_window.copy()
    masked[lower:upper] = 0.0
    second_peak = float(np.max(masked)) if masked.size > 0 else 0.0
    peak_ratio = float(peak / max(second_peak, 1e-6))
    shift = int(peak_index - max_shift)
    tau_s = float(shift) / float(int(interp) * float(sample_rate_hz))
    return tau_s, peak, peak_ratio


def estimate_horizontal_bearing(
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
) -> AudioBearingEstimate:
    if not protocol.is_stable:
        return AudioBearingEstimate(reason="transient_window")
    if packet is None or not isinstance(packet.stereo_audio, np.ndarray):
        return AudioBearingEstimate(reason="missing_audio")
    stereo = np.asarray(packet.stereo_audio, dtype=np.float32)
    if stereo.ndim != 2 or stereo.shape[0] < 2:
        return AudioBearingEstimate(reason="insufficient_channels")
    if stereo.shape[1] < 32:
        return AudioBearingEstimate(reason="too_few_samples")
    if float(ear_distance_m) <= 1e-6 or float(speed_of_sound_mps) <= 1e-6:
        return AudioBearingEstimate(reason="invalid_geometry")
    mono = np.asarray(packet.mono_audio, dtype=np.float32).reshape(-1) if isinstance(packet.mono_audio, np.ndarray) else np.mean(stereo, axis=0)
    mono_peak = float(np.max(np.abs(mono - float(np.mean(mono))))) if mono.size > 0 else 0.0
    if mono.size == 0 or mono_peak <= max(float(reference_rms) * 1e-4, 1e-8):
        return AudioBearingEstimate(reason="low_energy")
    max_tau_s = float(ear_distance_m) / float(speed_of_sound_mps)
    tau_s, peak, peak_ratio = _gcc_phat_delay(
        stereo[0],
        stereo[1],
        sample_rate_hz=max(int(packet.sample_rate_hz), 1),
        max_tau_s=max_tau_s,
        interp=max(int(gcc_interp), 1),
        exclusion_bins=max(int(peak_exclusion_bins), 1),
    )
    tau_s = float(np.clip(tau_s, -max_tau_s, max_tau_s))
    sine_value = float(np.clip(tau_s * float(speed_of_sound_mps) / float(ear_distance_m), -1.0, 1.0))
    relative_bearing_rad = float(math.asin(sine_value))
    relative_bearing_deg = float(np.degrees(relative_bearing_rad))
    edge_margin = 1.0 - min(abs(tau_s) / max(max_tau_s, 1e-8), 1.0)
    peak_term = float(np.clip(peak / 0.2, 0.0, 1.0))
    peak_ratio_term = float(np.clip((peak_ratio - 1.0) / 2.0, 0.0, 1.0))
    confidence = float(np.clip(0.35 * peak_term + 0.45 * peak_ratio_term + 0.20 * edge_margin, 0.0, 1.0))
    is_valid = bool(peak_ratio >= float(min_peak_ratio) and confidence >= float(min_confidence))
    reason = "ok" if is_valid else "low_confidence"
    return AudioBearingEstimate(
        is_valid=is_valid,
        reason=reason,
        relative_bearing_rad=float(relative_bearing_rad),
        relative_bearing_deg=float(relative_bearing_deg),
        tau_s=float(tau_s),
        tau_samples=float(tau_s * float(max(int(packet.sample_rate_hz), 1))),
        max_tau_s=float(max_tau_s),
        peak=float(peak),
        peak_ratio=float(peak_ratio),
        confidence=float(confidence),
        direction=str(coarse_direction_from_angle(relative_bearing_deg)),
    )


def project_bearing_to_world_ray(
    packet: AudioObservationPacket,
    protocol: AudioProtocolState,
    bearing: AudioBearingEstimate,
) -> AudioRay2D:
    if not bearing.is_valid:
        return AudioRay2D(reason=str(bearing.reason or "invalid_bearing"), goal_id=str(protocol.goal_id), step_index=int(packet.step_index))
    if packet is None or packet.position is None:
        return AudioRay2D(reason="missing_pose", goal_id=str(protocol.goal_id), step_index=int(packet.step_index))
    position = np.asarray(packet.position, dtype=np.float32).reshape(-1)
    if position.size < 3 or not np.all(np.isfinite(position[:3])):
        return AudioRay2D(reason="invalid_pose", goal_id=str(protocol.goal_id), step_index=int(packet.step_index))
    world_bearing_rad = wrap_angle_rad(float(packet.heading_rad) + float(bearing.relative_bearing_rad))
    direction_xz = np.asarray([
        math.sin(world_bearing_rad),
        -math.cos(world_bearing_rad),
    ], dtype=np.float32)
    norm = float(np.linalg.norm(direction_xz))
    if norm <= 1e-6:
        return AudioRay2D(reason="degenerate_direction", goal_id=str(protocol.goal_id), step_index=int(packet.step_index))
    direction_xz = direction_xz / norm
    return AudioRay2D(
        is_valid=True,
        reason="ok",
        goal_id=str(protocol.goal_id),
        step_index=int(packet.step_index),
        origin_xz=np.asarray([float(position[0]), float(position[2])], dtype=np.float32),
        direction_xz=np.asarray(direction_xz, dtype=np.float32),
        world_bearing_rad=float(world_bearing_rad),
        world_bearing_deg=float(np.degrees(world_bearing_rad)),
        confidence=float(bearing.confidence),
    )


def update_belief_map_from_ray(
    belief_map: np.ndarray,
    grid_x_world: np.ndarray,
    grid_z_world: np.ndarray,
    ray: AudioRay2D,
    *,
    sigma_m: float,
    max_range_m: float,
) -> np.ndarray:
    belief = np.asarray(belief_map, dtype=np.float32)
    if not ray.is_valid or ray.origin_xz is None or ray.direction_xz is None:
        return belief
    sigma = max(float(sigma_m), 1e-3)
    max_range = max(float(max_range_m), 1e-3)
    origin = np.asarray(ray.origin_xz, dtype=np.float32).reshape(2)
    direction = np.asarray(ray.direction_xz, dtype=np.float32).reshape(2)
    delta_x = np.asarray(grid_x_world, dtype=np.float32) - float(origin[0])
    delta_z = np.asarray(grid_z_world, dtype=np.float32) - float(origin[1])
    along = delta_x * float(direction[0]) + delta_z * float(direction[1])
    perp = np.abs(delta_x * float(direction[1]) - delta_z * float(direction[0]))
    forward_mask = (along >= 0.0) & (along <= max_range)
    update = np.exp(-0.5 * np.square(perp / sigma)).astype(np.float32)
    update *= forward_mask.astype(np.float32)
    belief += update * float(max(ray.confidence, 1e-3))
    return belief


def summarize_belief_map(
    belief_map: np.ndarray,
    *,
    goal_id: str,
    ray_count: int,
    origin_world: np.ndarray,
    resolution_m: float,
    grid_size: int,
) -> AudioBelief2DState:
    belief = np.asarray(belief_map, dtype=np.float32)
    if belief.size == 0:
        return AudioBelief2DState(goal_id=str(goal_id), ray_count=int(ray_count))
    peak_index = int(np.argmax(belief))
    peak_cell = np.unravel_index(peak_index, belief.shape)
    peak_value = float(belief[peak_cell])
    total_mass = float(np.sum(belief))
    entropy = 0.0
    if total_mass > 1e-8:
        probs = belief / total_mass
        valid = probs[probs > 1e-12]
        if valid.size > 0:
            entropy = float(-np.sum(valid * np.log(valid)))
    half = int(grid_size) // 2
    peak_world = np.asarray(
        [
            float(origin_world[0]) + float(int(peak_cell[0]) - half) * float(resolution_m),
            float(origin_world[2]) + float(int(peak_cell[1]) - half) * float(resolution_m),
        ],
        dtype=np.float32,
    )
    return AudioBelief2DState(
        goal_id=str(goal_id),
        ray_count=int(ray_count),
        total_mass=float(total_mass),
        peak_value=float(peak_value),
        entropy=float(entropy),
        peak_cell=(int(peak_cell[0]), int(peak_cell[1])),
        peak_world=peak_world,
    )


def triangulate_rays_2d(
    rays: Tuple[AudioRay2D, ...],
    *,
    goal_id: str,
    min_rays: int,
    max_condition_number: float,
    min_pair_angle_deg: float,
) -> AudioTriangulation2D:
    valid_rays = [ray for ray in rays if ray.is_valid and ray.origin_xz is not None and ray.direction_xz is not None]
    if len(valid_rays) < int(min_rays):
        return AudioTriangulation2D(reason="insufficient_rays", goal_id=str(goal_id), ray_count=len(valid_rays))
    identity = np.eye(2, dtype=np.float32)
    matrix_a = np.zeros((2, 2), dtype=np.float64)
    vector_b = np.zeros((2,), dtype=np.float64)
    pair_angles = []
    for ray in valid_rays:
        direction = np.asarray(ray.direction_xz, dtype=np.float64).reshape(2)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-8:
            continue
        direction = direction / direction_norm
        origin = np.asarray(ray.origin_xz, dtype=np.float64).reshape(2)
        projector = identity - np.outer(direction, direction)
        matrix_a += projector
        vector_b += projector @ origin
    for index, first in enumerate(valid_rays):
        d0 = np.asarray(first.direction_xz, dtype=np.float64).reshape(2)
        n0 = float(np.linalg.norm(d0))
        if n0 <= 1e-8:
            continue
        d0 = d0 / n0
        for second in valid_rays[index + 1 :]:
            d1 = np.asarray(second.direction_xz, dtype=np.float64).reshape(2)
            n1 = float(np.linalg.norm(d1))
            if n1 <= 1e-8:
                continue
            d1 = d1 / n1
            dot = float(np.clip(np.dot(d0, d1), -1.0, 1.0))
            angle = float(np.degrees(np.arccos(dot)))
            angle = min(angle, 180.0 - angle)
            pair_angles.append(angle)
    if not np.all(np.isfinite(matrix_a)) or not np.all(np.isfinite(vector_b)):
        return AudioTriangulation2D(reason="invalid_system", goal_id=str(goal_id), ray_count=len(valid_rays))
    eigenvalues = np.linalg.eigvalsh(matrix_a)
    min_eig = float(np.min(eigenvalues)) if eigenvalues.size > 0 else 0.0
    max_eig = float(np.max(eigenvalues)) if eigenvalues.size > 0 else 0.0
    if min_eig <= 1e-8:
        condition_number = float("inf")
    else:
        condition_number = float(max_eig / min_eig)
    min_angle = float(min(pair_angles)) if pair_angles else 0.0
    if not np.isfinite(condition_number) or condition_number > float(max_condition_number):
        return AudioTriangulation2D(
            reason="ill_conditioned",
            goal_id=str(goal_id),
            ray_count=len(valid_rays),
            condition_number=condition_number,
            min_pair_angle_deg=min_angle,
        )
    if min_angle < float(min_pair_angle_deg):
        return AudioTriangulation2D(
            reason="low_angle_diversity",
            goal_id=str(goal_id),
            ray_count=len(valid_rays),
            condition_number=condition_number,
            min_pair_angle_deg=min_angle,
        )
    point = np.linalg.pinv(matrix_a) @ vector_b
    point = np.asarray(point, dtype=np.float32).reshape(2)
    residuals = []
    for ray in valid_rays:
        origin = np.asarray(ray.origin_xz, dtype=np.float32).reshape(2)
        direction = np.asarray(ray.direction_xz, dtype=np.float32).reshape(2)
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            continue
        direction = direction / norm
        delta = point - origin
        residual = abs(float(delta[0] * direction[1] - delta[1] * direction[0]))
        residuals.append(residual)
    mean_residual = float(np.mean(residuals)) if residuals else 0.0
    condition_term = float(np.clip(1.0 - condition_number / max(float(max_condition_number), 1e-6), 0.0, 1.0))
    angle_term = float(np.clip(min_angle / max(float(min_pair_angle_deg) * 4.0, 1e-6), 0.0, 1.0))
    residual_term = float(np.clip(1.0 - mean_residual / 2.0, 0.0, 1.0))
    confidence = float(np.clip(0.4 * condition_term + 0.3 * angle_term + 0.3 * residual_term, 0.0, 1.0))
    return AudioTriangulation2D(
        is_valid=True,
        reason="ok",
        goal_id=str(goal_id),
        ray_count=len(valid_rays),
        point_xz=np.asarray(point, dtype=np.float32),
        condition_number=float(condition_number),
        mean_residual_m=float(mean_residual),
        min_pair_angle_deg=float(min_angle),
        confidence=float(confidence),
    )


__all__ = [
    "estimate_horizontal_bearing",
    "summarize_belief_map",
    "triangulate_rays_2d",
    "project_bearing_to_world_ray",
    "update_belief_map_from_ray",
    "wrap_angle_rad",
]
