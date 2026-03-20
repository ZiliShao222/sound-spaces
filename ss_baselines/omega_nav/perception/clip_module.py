from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from ss_baselines.omega_nav.perception.base import GoalSpec, ObjectRegionMatch
from ss_baselines.omega_nav.utils import cosine_similarity, extract_depth, extract_rgb, image_histogram_embedding


class OracleCLIPDetector:
    def __init__(self, config: Dict[str, float]):
        self._top_m = max(int(config.get("top_m", 3)), 1)
        self._max_depth_m = float(config.get("max_depth_m", 6.0))
        self._assume_normalized_depth = bool(config.get("assume_normalized_depth", True))
        self._histogram_bins = max(int(config.get("image_histogram_bins", config.get("histogram_bins", 8))), 2)
        self._visible_similarity_threshold = float(config.get("visible_similarity_threshold", 0.5))

    def _regions(self, rgb: np.ndarray) -> List[Tuple[str, Tuple[int, int, int, int], np.ndarray]]:
        height, width = rgb.shape[:2]
        third = max(width // 3, 1)
        center_left = width // 4
        center_right = max(center_left + 1, (3 * width) // 4)
        center_top = height // 4
        center_bottom = max(center_top + 1, (3 * height) // 4)
        regions: List[Tuple[str, Tuple[int, int, int, int], np.ndarray]] = [
            ("left", (0, 0, third, height), rgb[:, :third]),
            ("forward", (third, 0, min(2 * third, width), height), rgb[:, third:min(2 * third, width)]),
            ("right", (min(2 * third, width - 1), 0, width, height), rgb[:, min(2 * third, width - 1):]),
            ("forward", (center_left, center_top, center_right, center_bottom), rgb[center_top:center_bottom, center_left:center_right]),
        ]
        return [(direction, bbox, crop) for direction, bbox, crop in regions if crop.size > 0]

    def detect(
        self,
        observations: Dict[str, np.ndarray],
        goal_specs: Sequence[GoalSpec],
    ) -> Tuple[Dict[str, ObjectRegionMatch], Tuple[ObjectRegionMatch, ...]]:
        rgb = extract_rgb(observations)
        depth = extract_depth(
            observations,
            max_depth_m=self._max_depth_m,
            assume_normalized=self._assume_normalized_depth,
        )
        matches: Dict[str, ObjectRegionMatch] = {}
        regions = self._regions(rgb) if rgb is not None and rgb.ndim == 3 else []

        for goal in goal_specs:
            similarity = 0.0
            best_bbox = None
            best_direction = "forward"
            estimated_distance = None
            region_embedding = None
            if goal.image_embedding is not None and regions:
                for direction, bbox, crop in regions:
                    if crop.ndim != 3 or crop.shape[0] <= 1 or crop.shape[1] <= 1:
                        continue
                    candidate_embedding = image_histogram_embedding(crop, self._histogram_bins)
                    candidate_similarity = max(cosine_similarity(candidate_embedding, goal.image_embedding), 0.0)
                    if candidate_similarity > similarity:
                        similarity = float(candidate_similarity)
                        best_bbox = bbox
                        best_direction = direction
                        region_embedding = crop

                if best_bbox is not None and depth is not None:
                    x1, y1, x2, y2 = best_bbox
                    depth_crop = depth[y1:y2, x1:x2]
                    finite = depth_crop[np.isfinite(depth_crop)]
                    if finite.size > 0:
                        estimated_distance = float(np.median(finite))

            visible = bool(similarity >= self._visible_similarity_threshold)
            pixel_count = 0
            visible_ratio = 0.0
            angle_deg = None
            if best_bbox is not None and rgb is not None:
                x1, y1, x2, y2 = best_bbox
                pixel_count = int(max(x2 - x1, 0) * max(y2 - y1, 0))
                visible_ratio = float(pixel_count) / float(max(rgb.shape[0] * rgb.shape[1], 1))
                center_x = 0.5 * float(x1 + x2)
                angle_deg = ((center_x / max(rgb.shape[1] - 1, 1)) - 0.5) * 90.0

            matches[goal.goal_id] = ObjectRegionMatch(
                goal_id=goal.goal_id,
                goal_index=int(goal.goal_index),
                category=str(goal.category),
                similarity=float(similarity),
                visible=visible,
                pixel_count=int(pixel_count),
                visible_ratio=float(visible_ratio),
                bbox=best_bbox,
                estimated_distance_m=estimated_distance,
                relative_angle_deg=float(angle_deg) if angle_deg is not None else None,
                relative_direction=str(best_direction),
                target_position=None,
            )

        ranked = tuple(sorted(matches.values(), key=lambda item: float(item.similarity), reverse=True)[: self._top_m])
        return matches, ranked
