from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ss_baselines.omega_nav.perception.base import MapObservation, SemanticVoxelMapState
from ss_baselines.omega_nav.perception.siglip_encoder import SigLIPEncoder
from ss_baselines.omega_nav.utils import rotate_local_offset, world_to_grid


@dataclass
class VoxelCell:
    log_odds: float = 0.0
    obs_count: int = 0
    last_seen_step: int = 0
    feature: Optional[np.ndarray] = None
    feature_weight: float = 0.0


class SemanticVoxelMap:
    def __init__(self, config: Dict[str, float], encoder: SigLIPEncoder) -> None:
        self._encoder = encoder
        self._voxel_size = float(config.get("voxel_size_m", config.get("map_resolution_m", 0.25)))
        self._grid_resolution_m = float(config.get("map_resolution_m", self._voxel_size))
        self._grid_size = int(config.get("map_size_cells", 256))
        self._max_depth_m = float(config.get("max_depth_m", 6.0))
        self._min_depth_m = float(config.get("min_depth_m", 0.1))
        self._hfov_deg = float(config.get("hfov_deg", 90.0))
        self._camera_height_m = float(config.get("camera_height_m", 0.88))
        self._free_delta = float(config.get("free_log_odds_delta", -0.4))
        self._occupied_delta = float(config.get("occupied_log_odds_delta", 0.85))
        self._min_log_odds = float(config.get("min_log_odds", -2.0))
        self._max_log_odds = float(config.get("max_log_odds", 3.5))
        self._occupied_threshold = float(config.get("occupied_threshold", 0.6))
        self._free_threshold = float(config.get("free_threshold", -0.3))
        self._occupancy_min_height_m = float(config.get("occupancy_min_height_m", 0.1))
        self._occupancy_max_height_m = float(config.get("occupancy_max_height_m", 1.5))
        self._ray_stride = max(int(config.get("sample_stride", 16)), 1)
        self._semantic_patch_size = max(int(config.get("semantic_patch_size", 64)), 16)
        self._semantic_patch_stride = max(int(config.get("semantic_patch_stride", 64)), 16)
        self._semantic_update_weight = float(config.get("semantic_update_weight", 1.0))
        self._origin_world = np.zeros(3, dtype=np.float32)
        self._voxels: Dict[Tuple[int, int, int], VoxelCell] = {}

    def reset(self, origin_world: np.ndarray) -> None:
        self._origin_world = np.asarray(origin_world, dtype=np.float32).copy()
        self._voxels.clear()

    def update(self, observation: MapObservation) -> None:
        self._integrate_depth(observation)
        self._integrate_semantics(observation)

    def build_state(self, goal_embeddings: Dict[str, np.ndarray]) -> SemanticVoxelMapState:
        occupancy = -np.ones((self._grid_size, self._grid_size), dtype=np.int8)
        explored = np.zeros((self._grid_size, self._grid_size), dtype=np.uint8)
        frontier = np.zeros((self._grid_size, self._grid_size), dtype=np.uint8)
        goal_maps = {goal_id: np.zeros((self._grid_size, self._grid_size), dtype=np.float32) for goal_id in goal_embeddings}
        goal_peaks = {
            goal_id: {"score": 0.0, "position": [], "cell": []}
            for goal_id in goal_embeddings
        }
        occupied_voxel_count = 0
        semantic_voxel_count = 0

        for index, cell in self._voxels.items():
            center = self._voxel_center(index)
            grid_cell = world_to_grid(center, self._origin_world, self._grid_resolution_m, self._grid_size)
            if grid_cell is None:
                continue
            gx, gz = grid_cell
            if cell.obs_count > 0:
                explored[gx, gz] = 1
            if self._occupancy_min_height_m <= center[1] <= self._occupancy_max_height_m:
                if cell.log_odds >= self._occupied_threshold:
                    occupancy[gx, gz] = 1
                elif occupancy[gx, gz] != 1 and cell.log_odds <= self._free_threshold:
                    occupancy[gx, gz] = 0
            if cell.log_odds >= self._occupied_threshold:
                occupied_voxel_count += 1
            if cell.feature is None or cell.log_odds < self._occupied_threshold:
                continue
            semantic_voxel_count += 1
            for goal_id, goal_embedding in goal_embeddings.items():
                score = float(np.dot(cell.feature, goal_embedding))
                if score > float(goal_maps[goal_id][gx, gz]):
                    goal_maps[goal_id][gx, gz] = score
                if score > float(goal_peaks[goal_id].get("score", 0.0)):
                    goal_peaks[goal_id] = {
                        "score": float(score),
                        "position": np.asarray(center, dtype=np.float32).copy(),
                        "cell": [int(index[0]), int(index[1]), int(index[2])],
                    }

        for gx in range(1, self._grid_size - 1):
            for gz in range(1, self._grid_size - 1):
                if occupancy[gx, gz] != 0:
                    continue
                if np.any(occupancy[gx - 1 : gx + 2, gz - 1 : gz + 2] < 0):
                    frontier[gx, gz] = 1

        return SemanticVoxelMapState(
            occupancy=occupancy,
            explored=explored,
            frontier=frontier,
            goal_maps=goal_maps,
            goal_peaks=goal_peaks,
            voxel_count=len(self._voxels),
            occupied_voxel_count=occupied_voxel_count,
            semantic_voxel_count=semantic_voxel_count,
        )

    def _integrate_depth(self, observation: MapObservation) -> None:
        height, width = observation.depth.shape
        camera_position = np.asarray(
            [observation.position[0], observation.position[1] + self._camera_height_m, observation.position[2]],
            dtype=np.float32,
        )
        for row in range(0, height, self._ray_stride):
            for col in range(0, width, self._ray_stride):
                ray_depth = float(observation.depth[row, col])
                if not np.isfinite(ray_depth) or ray_depth < self._min_depth_m:
                    continue
                ray_depth = min(ray_depth, self._max_depth_m)
                endpoint = self._pixel_to_world(observation, row, col, ray_depth)
                path = self._voxel_path(camera_position, endpoint)
                for index in path[:-1]:
                    self._update_log_odds(index, self._free_delta, observation.step_index)
                if ray_depth < self._max_depth_m * 0.98:
                    self._update_log_odds(path[-1], self._occupied_delta, observation.step_index)
                else:
                    self._update_log_odds(path[-1], self._free_delta, observation.step_index)

    def _integrate_semantics(self, observation: MapObservation) -> None:
        height, width = observation.depth.shape
        patch = self._semantic_patch_size
        stride = self._semantic_patch_stride
        patches: List[np.ndarray] = []
        indices: List[Tuple[int, int, int]] = []
        for top in range(0, max(height - patch + 1, 1), stride):
            for left in range(0, max(width - patch + 1, 1), stride):
                depth_patch = observation.depth[top : top + patch, left : left + patch]
                finite = depth_patch[np.isfinite(depth_patch)]
                if finite.size == 0:
                    continue
                patch_depth = float(np.median(finite))
                if patch_depth < self._min_depth_m or patch_depth >= self._max_depth_m * 0.98:
                    continue
                center_row = min(top + patch // 2, height - 1)
                center_col = min(left + patch // 2, width - 1)
                point = self._pixel_to_world(observation, center_row, center_col, patch_depth)
                index = self._voxel_index(point)
                cell = self._voxels.get(index)
                if cell is None or cell.log_odds < self._occupied_threshold:
                    continue
                patches.append(observation.rgb[top : top + patch, left : left + patch])
                indices.append(index)
        if len(patches) == 0:
            return
        embeddings = self._encoder.encode_images(patches)
        for index, feature in zip(indices, embeddings):
            self._update_feature(index, feature, observation.step_index)

    def _pixel_to_world(self, observation: MapObservation, row: int, col: int, depth: float) -> np.ndarray:
        height, width = observation.depth.shape
        hfov = float(np.deg2rad(self._hfov_deg))
        vfov = float(2.0 * np.arctan(np.tan(hfov / 2.0) * float(height) / float(width)))
        x_angle = ((float(col) + 0.5) / float(width) - 0.5) * hfov
        y_angle = ((float(row) + 0.5) / float(height) - 0.5) * vfov
        direction = np.asarray([np.tan(x_angle), -np.tan(y_angle), -1.0], dtype=np.float32)
        direction /= float(np.linalg.norm(direction))
        local = direction * float(depth)
        xz_offset = rotate_local_offset(observation.heading_rad, (local[0], 0.0, local[2]))
        return np.asarray(
            [
                observation.position[0] + xz_offset[0],
                observation.position[1] + self._camera_height_m + local[1],
                observation.position[2] + xz_offset[2],
            ],
            dtype=np.float32,
        )

    def _voxel_path(self, start: np.ndarray, end: np.ndarray) -> List[Tuple[int, int, int]]:
        distance = float(np.linalg.norm(end - start))
        steps = max(int(np.ceil(distance / max(self._voxel_size * 0.5, 1e-6))), 1)
        points = start[None, :] + (end - start)[None, :] * np.linspace(0.0, 1.0, steps + 1, dtype=np.float32)[:, None]
        path: List[Tuple[int, int, int]] = []
        last: Optional[Tuple[int, int, int]] = None
        for point in points:
            index = self._voxel_index(point)
            if index != last:
                path.append(index)
                last = index
        return path

    def _voxel_index(self, point: np.ndarray) -> Tuple[int, int, int]:
        coords = np.floor(np.asarray(point, dtype=np.float32) / self._voxel_size).astype(np.int32)
        return int(coords[0]), int(coords[1]), int(coords[2])

    def _voxel_center(self, index: Tuple[int, int, int]) -> np.ndarray:
        return (np.asarray(index, dtype=np.float32) + 0.5) * self._voxel_size

    def _update_log_odds(self, index: Tuple[int, int, int], delta: float, step_index: int) -> None:
        cell = self._voxels.get(index)
        if cell is None:
            cell = VoxelCell()
            self._voxels[index] = cell
        cell.log_odds = float(np.clip(cell.log_odds + delta, self._min_log_odds, self._max_log_odds))
        cell.obs_count += 1
        cell.last_seen_step = int(step_index)

    def _update_feature(self, index: Tuple[int, int, int], feature: np.ndarray, step_index: int) -> None:
        cell = self._voxels[index]
        if cell.feature is None:
            cell.feature = np.asarray(feature, dtype=np.float32)
            cell.feature_weight = float(self._semantic_update_weight)
            cell.last_seen_step = int(step_index)
            return
        weight = cell.feature_weight + float(self._semantic_update_weight)
        merged = (cell.feature * cell.feature_weight + feature * float(self._semantic_update_weight)) / weight
        merged /= max(float(np.linalg.norm(merged)), 1e-6)
        cell.feature = merged.astype(np.float32)
        cell.feature_weight = weight
        cell.last_seen_step = int(step_index)
