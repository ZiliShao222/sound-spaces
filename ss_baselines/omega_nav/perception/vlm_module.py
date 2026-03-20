from __future__ import annotations

from typing import Dict, Optional, Sequence

from ss_baselines.omega_nav.perception.base import GoalSpec, ObjectRegionMatch, SemanticMapState


class OracleVLMDescriptor:
    def __init__(self, config: Dict[str, float]):
        self._describe_every = max(int(config.get("describe_every", 5)), 1)
        self._max_summary_objects = max(int(config.get("max_summary_objects", 4)), 1)
        self._last_description = ""
        self._last_step = -1

    def reset(self) -> None:
        self._last_description = ""
        self._last_step = -1

    def describe(
        self,
        *,
        step_index: int,
        goal_specs: Sequence[GoalSpec],
        visual_matches: Dict[str, ObjectRegionMatch],
        semantic_map: SemanticMapState,
    ) -> str:
        if self._last_description and int(step_index) % self._describe_every != 0:
            return self._last_description

        visible_matches = [match for match in visual_matches.values() if match.visible]
        visible_matches.sort(key=lambda item: float(item.similarity), reverse=True)
        if visible_matches:
            object_phrases = []
            for match in visible_matches[: self._max_summary_objects]:
                distance_text = (
                    f"，约 {float(match.estimated_distance_m):.1f}m"
                    if match.estimated_distance_m is not None
                    else ""
                )
                object_phrases.append(f"{match.category} 在{match.relative_direction}侧{distance_text}")
            object_sentence = "视野中可见 " + "；".join(object_phrases)
        else:
            object_sentence = "当前视野中没有可确认的目标物体"

        free_space = semantic_map.free_space_by_direction or {}
        if free_space:
            best_direction = max(free_space, key=free_space.get)
            map_sentence = (
                f"探索地图已覆盖 {semantic_map.explored_ratio * 100.0:.1f}% 区域，"
                f"{best_direction} 方向更开阔"
            )
        else:
            map_sentence = f"探索地图已覆盖 {semantic_map.explored_ratio * 100.0:.1f}% 区域"

        pending_names = [goal.category for goal in goal_specs]
        pending_sentence = f"任务目标包括：{', '.join(pending_names[:4])}" if pending_names else "当前没有待搜索目标"
        self._last_description = f"{pending_sentence}。{object_sentence}。{map_sentence}。"
        self._last_step = int(step_index)
        return self._last_description
