from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ss_baselines.omega_nav.perception.base import SemanticMapState
from ss_baselines.omega_nav.utils import (
    extract_depth,
    grid_to_world,
    interpolate_grid_line,
    rotate_vector,
    sector_free_space,
    world_to_grid,
)


class OracleDepthProcessor:
    def __init__(self, config: Dict[str, Any]):
        self._resolution_m = float(config.get("map_resolution_m", 0.25))
        self._grid_size = max(int(config.get("map_size_cells", 256)), 32)
        self._max_depth_m = float(config.get("max_depth_m", 6.0))
        self._min_depth_m = float(config.get("min_depth_m", 0.1))
        self._sample_stride = max(int(config.get("sample_stride", 16)), 1)
        self._hfov_deg = float(config.get("hfov_deg", 90.0))
        self._frontier_max_points = max(int(config.get("frontier_max_points", 24)), 1)
        self._assume_normalized_depth = bool(config.get("assume_normalized_depth", True))
        self._origin_world: Optional[np.ndarray] = None
        self._occupancy: Optional[np.ndarray] = None
        self._visited: Optional[np.ndarray] = None

    def reset(self, agent_position: Optional[np.ndarray] = None) -> None:
        self._occupancy = -np.ones((self._grid_size, self._grid_size), dtype=np.int8)
        self._visited = np.zeros((self._grid_size, self._grid_size), dtype=np.uint8)
        self._origin_world = np.asarray(agent_position if agent_position is not None else np.zeros(3), dtype=np.float32)

    def _ensure_state(self, agent_position: np.ndarray) -> None:
        if self._occupancy is None or self._visited is None or self._origin_world is None:
            self.reset(agent_position)

    def update(self, env: Any, observations: Dict[str, Any]) -> SemanticMapState:
        agent_state = env.sim.get_agent_state()
        agent_position = np.asarray(agent_state.position, dtype=np.float32)
        self._ensure_state(agent_position)
        assert self._occupancy is not None
        assert self._visited is not None
        assert self._origin_world is not None

        agent_cell = world_to_grid(agent_position, self._origin_world, self._resolution_m, self._grid_size)
        if agent_cell is None:
            self.reset(agent_position)
            agent_cell = world_to_grid(agent_position, self._origin_world, self._resolution_m, self._grid_size)
        assert agent_cell is not None

        self._visited[agent_cell[0], agent_cell[1]] = 1
        self._occupancy[agent_cell[0], agent_cell[1]] = 0

        depth = extract_depth(
            observations,
            max_depth_m=self._max_depth_m,
            assume_normalized=self._assume_normalized_depth,
        )
        free_space = sector_free_space(depth)

        if depth is not None:
            height, width = depth.shape
            cols = range(0, width, self._sample_stride)
            rows = [max(0, min(height - 1, int(height * ratio))) for ratio in (0.4, 0.5, 0.6)]
            hfov_rad = np.deg2rad(self._hfov_deg)

            for row in rows:
                for col in cols:
                    ray_depth = float(depth[row, col])
                    if not np.isfinite(ray_depth) or ray_depth < self._min_depth_m:
                        continue
                    ray_depth = min(ray_depth, self._max_depth_m)
                    x_angle = ((float(col) + 0.5) / float(max(width, 1)) - 0.5) * hfov_rad
                    local_endpoint = np.asarray(
                        [
                            np.sin(x_angle) * ray_depth,
                            0.0,
                            -np.cos(x_angle) * ray_depth,
                        ],
                        dtype=np.float32,
                    )
                    world_endpoint = agent_position + rotate_vector(agent_state.rotation, local_endpoint)
                    endpoint_cell = world_to_grid(world_endpoint, self._origin_world, self._resolution_m, self._grid_size)
                    if endpoint_cell is None:
                        continue
                    line = list(interpolate_grid_line(agent_cell, endpoint_cell))
                    if not line:
                        continue
                    for free_cell in line[:-1]:
                        self._occupancy[free_cell[0], free_cell[1]] = 0
                    self._occupancy[line[-1][0], line[-1][1]] = 1 if ray_depth < self._max_depth_m * 0.98 else 0

        frontier = np.zeros_like(self._occupancy, dtype=np.uint8)
        offsets = ((1, 0), (-1, 0), (0, 1), (0, -1))
        for gx in range(1, self._grid_size - 1):
            for gz in range(1, self._grid_size - 1):
                if self._occupancy[gx, gz] != 0:
                    continue
                for dx, dz in offsets:
                    if self._occupancy[gx + dx, gz + dz] < 0:
                        frontier[gx, gz] = 1
                        break

        frontier_cells = np.argwhere(frontier > 0)
        if frontier_cells.size > 0:
            ordering = np.argsort(np.linalg.norm(frontier_cells - np.asarray(agent_cell)[None, :], axis=1))
            frontier_cells = frontier_cells[ordering[: self._frontier_max_points]]
        frontier_world_positions: List[np.ndarray] = [
            grid_to_world((int(cell[0]), int(cell[1])), self._origin_world, self._resolution_m, self._grid_size)
            for cell in frontier_cells
        ]

        explored_ratio = float(np.count_nonzero(self._occupancy >= 0)) / float(self._occupancy.size)
        return SemanticMapState(
            occupancy=self._occupancy.copy(),
            visited=self._visited.copy(),
            frontier=frontier,
            agent_cell=(int(agent_cell[0]), int(agent_cell[1])),
            origin_world=self._origin_world.copy(),
            resolution_m=float(self._resolution_m),
            frontier_world_positions=tuple(frontier_world_positions),
            explored_ratio=float(explored_ratio),
            free_space_by_direction={key: float(value) for key, value in free_space.items()},
        )
