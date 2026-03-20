from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ss_baselines.omega_nav.memory import EpisodicMemoryEntry, HierarchicalMemory
from ss_baselines.omega_nav.perception.base import GoalSpec, PerceptionOutput
from ss_baselines.omega_nav.utils import (
    as_serializable,
    coarse_direction_from_angle,
    extract_pose,
    pose_to_position,
    relative_bearing_deg,
    relative_bearing_from_pose_deg,
)


@dataclass
class PlannerDecision:
    mark_complete: List[str]
    next_goal: Optional[str]
    action: str
    direction: str
    reasoning: str
    target_positions: Tuple[np.ndarray, ...] = ()
    goal_scores: Dict[str, float] = field(default_factory=dict)
    prompt_snapshot: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return as_serializable(
            {
                "mark_complete": self.mark_complete,
                "next_goal": self.next_goal,
                "action": self.action,
                "direction": self.direction,
                "reasoning": self.reasoning,
                "target_positions": self.target_positions,
                "goal_scores": self.goal_scores,
                "prompt_snapshot": self.prompt_snapshot,
            }
        )


class OmegaLLMPlanner:
    def __init__(self, config: Dict[str, Any]):
        self._submit_distance_m = float(config.get("submit_distance_m", 1.0))
        self._distance_penalty = float(config.get("distance_penalty", 0.05))
        self._visible_bonus = float(config.get("visible_priority_bonus", 2.0))
        self._audio_bonus = float(config.get("audio_priority_bonus", 1.2))
        self._hint_bonus = float(config.get("hint_priority_bonus", 0.8))
        self._audio_submit_similarity = float(config.get("audio_submit_similarity", 0.92))
        self._forward_submit_angle_deg = float(config.get("forward_submit_angle_deg", 20.0))
        self._audio_submit_distance_cap = float(config.get("audio_submit_distance_cap", 1.25))

    def _hint_distance(
        self,
        env: Any,
        episode: Any,
        hint: Optional[EpisodicMemoryEntry],
        observations: Optional[Dict[str, Any]] = None,
    ) -> Optional[float]:
        if hint is None or hint.position is None:
            return None
        target = np.asarray(hint.position, dtype=np.float32)
        if env is not None and hasattr(env, "sim"):
            current = np.asarray(env.sim.get_agent_state().position, dtype=np.float32)
            distance = env.sim.geodesic_distance(current, [target], episode)
            if distance is None or not np.isfinite(distance):
                return None
            return float(distance)
        current = pose_to_position(extract_pose(observations))
        if current is None:
            return None
        return float(np.linalg.norm((target - current)[[0, 2]]))

    def _observed_distance(
        self,
        env: Any,
        episode: Any,
        goal: GoalSpec,
        perception: PerceptionOutput,
        memory: HierarchicalMemory,
        observations: Optional[Dict[str, Any]] = None,
    ) -> Optional[float]:
        visual = perception.visual_match(goal.goal_id)
        if visual is not None and visual.estimated_distance_m is not None:
            return float(visual.estimated_distance_m)

        audio = perception.audio_match(goal.goal_id)
        if audio is not None and audio.distance_m is not None:
            return float(audio.distance_m)

        return self._hint_distance(env, episode, memory.best_hint(goal.goal_id), observations=observations)

    def _submit_candidate(
        self,
        env: Any,
        episode: Any,
        goal_specs_by_id: Dict[str, GoalSpec],
        perception: PerceptionOutput,
        memory: HierarchicalMemory,
        candidate_goal_ids: Sequence[str],
        submit_distance_m: float,
        observations: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        winner = None
        winner_distance = None
        for goal_id in candidate_goal_ids:
            goal = goal_specs_by_id.get(goal_id)
            if goal is None:
                continue
            visual = perception.visual_match(goal_id)
            if visual is not None and visual.visible:
                visual_distance = visual.estimated_distance_m
                if visual_distance is not None and float(visual_distance) <= float(submit_distance_m):
                    if winner_distance is None or float(visual_distance) < float(winner_distance):
                        winner = goal_id
                        winner_distance = float(visual_distance)
                    continue

            audio = perception.audio_match(goal_id)
            if audio is None or not audio.detected:
                continue
            if float(audio.aggregated_similarity) < float(self._audio_submit_similarity):
                continue
            if audio.relative_angle_deg is not None and abs(float(audio.relative_angle_deg)) > float(self._forward_submit_angle_deg):
                continue
            audio_distance = audio.distance_m
            if audio_distance is not None and float(audio_distance) > float(max(submit_distance_m, self._audio_submit_distance_cap)):
                continue
            hint_distance = self._hint_distance(env, episode, memory.best_hint(goal_id), observations=observations)
            effective_distance = float(audio_distance) if audio_distance is not None else hint_distance
            if winner_distance is None or effective_distance is None or float(effective_distance) < float(winner_distance):
                winner = goal_id
                winner_distance = effective_distance
        return winner

    def _goal_score(
        self,
        env: Any,
        episode: Any,
        goal: GoalSpec,
        perception: PerceptionOutput,
        memory: HierarchicalMemory,
        observations: Optional[Dict[str, Any]] = None,
    ) -> float:
        score = 0.0
        visual = perception.visual_match(goal.goal_id)
        audio = perception.audio_match(goal.goal_id)
        hint = memory.best_hint(goal.goal_id)
        if visual is not None:
            if visual.visible:
                score += self._visible_bonus + float(visual.similarity)
            else:
                score += 0.2 * float(visual.similarity)
        if audio is not None:
            if audio.detected:
                score += self._audio_bonus + float(audio.aggregated_similarity)
            else:
                score += 0.15 * float(audio.aggregated_similarity)
        if hint is not None:
            score += self._hint_bonus * float(hint.confidence)

        distance = self._observed_distance(env, episode, goal, perception, memory, observations=observations)
        if distance is not None:
            score -= self._distance_penalty * float(distance)
        return float(score)

    def _hint_target(
        self,
        hint: Optional[EpisodicMemoryEntry],
    ) -> Tuple[np.ndarray, ...]:
        if hint is not None and hint.position is not None:
            return (np.asarray(hint.position, dtype=np.float32),)
        return ()

    def decide(
        self,
        *,
        env: Any,
        episode: Any,
        observations: Optional[Dict[str, Any]] = None,
        goal_specs: Sequence[GoalSpec],
        perception: PerceptionOutput,
        memory: HierarchicalMemory,
        pending_goal_ids: Sequence[str],
        order_mode: str,
        step_index: int,
        submit_distance_m: Optional[float] = None,
    ) -> PlannerDecision:
        goal_specs_by_id = {goal.goal_id: goal for goal in goal_specs}
        submit_distance = float(submit_distance_m or self._submit_distance_m)
        normalized_mode = str(order_mode).strip().lower()
        ordered = normalized_mode == "ordered"

        candidate_submit_ids = list(pending_goal_ids[:1]) if ordered else list(pending_goal_ids)
        submit_goal_id = self._submit_candidate(
            env,
            episode,
            goal_specs_by_id,
            perception,
            memory,
            candidate_submit_ids,
            submit_distance,
            observations=observations,
        )
        if submit_goal_id is not None:
            reasoning = f"{submit_goal_id} 已进入可提交范围，优先执行提交。"
            prompt_snapshot = {
                "mode": normalized_mode,
                "pending_goals": list(pending_goal_ids),
                "visual": perception.scene_description,
                "working_memory": memory.working_memory(),
            }
            return PlannerDecision(
                mark_complete=[submit_goal_id],
                next_goal=submit_goal_id,
                action="approach_object",
                direction="forward",
                reasoning=reasoning,
                target_positions=(),
                prompt_snapshot=prompt_snapshot,
            )

        goal_scores: Dict[str, float] = {}
        if ordered:
            next_goal_id = pending_goal_ids[0] if pending_goal_ids else None
            if next_goal_id is not None:
                goal_scores[next_goal_id] = self._goal_score(
                    env,
                    episode,
                    goal_specs_by_id[next_goal_id],
                    perception,
                    memory,
                    observations=observations,
                )
        else:
            for goal_id in pending_goal_ids:
                goal = goal_specs_by_id.get(goal_id)
                if goal is None:
                    continue
                goal_scores[goal_id] = self._goal_score(
                    env,
                    episode,
                    goal,
                    perception,
                    memory,
                    observations=observations,
                )
            next_goal_id = max(goal_scores, key=goal_scores.get) if goal_scores else None

        if next_goal_id is None:
            return PlannerDecision(
                mark_complete=[],
                next_goal=None,
                action="explore_frontier",
                direction="forward",
                reasoning="没有剩余目标，准备停止。",
                target_positions=(),
                goal_scores=goal_scores,
                prompt_snapshot={"mode": normalized_mode, "pending_goals": list(pending_goal_ids)},
            )

        goal = goal_specs_by_id[next_goal_id]
        visual = perception.visual_match(next_goal_id)
        audio = perception.audio_match(next_goal_id)
        hint = memory.best_hint(next_goal_id)
        if visual is not None and visual.visible:
            action = "approach_object"
            target_positions = self._hint_target(hint)
            direction = visual.relative_direction
            reasoning = f"{goal.category} 在当前视野中出现了参考外观匹配，继续朝该方向靠近。"
        elif audio is not None and audio.detected:
            action = "navigate_to_sound"
            target_positions = self._hint_target(hint)
            if audio.relative_angle_deg is not None:
                direction = coarse_direction_from_angle(audio.relative_angle_deg)
            else:
                direction = "forward"
            reasoning = f"{goal.category} 的环境音仍可听见，沿 {audio.direction_text} 继续搜索。"
        elif hint is not None:
            action = "navigate_to_hint"
            target_positions = self._hint_target(hint)
            if hint.position is not None:
                if env is not None and hasattr(env, "sim"):
                    angle_deg = relative_bearing_deg(
                        np.asarray(env.sim.get_agent_state().position, dtype=np.float32),
                        env.sim.get_agent_state().rotation,
                        np.asarray(hint.position, dtype=np.float32),
                    )
                else:
                    angle_deg = relative_bearing_from_pose_deg(extract_pose(observations), np.asarray(hint.position, dtype=np.float32))
                direction = coarse_direction_from_angle(angle_deg)
            else:
                direction = "forward"
            reasoning = f"{goal.category} 存在历史线索，优先回访可疑区域。"
        else:
            action = "explore_frontier"
            target_positions = tuple(perception.semantic_map.frontier_world_positions[:3])
            free_space = perception.semantic_map.free_space_by_direction or {"forward": 0.0}
            direction = max(free_space, key=free_space.get)
            reasoning = f"{goal.category} 暂无线索，转向 frontier 探索。"

        prompt_snapshot = {
            "mode": normalized_mode,
            "pending_goals": list(pending_goal_ids),
            "memory": memory.to_prompt_summary(),
            "visual": perception.scene_description,
            "audio": {
                goal_id: match.to_dict()
                for goal_id, match in perception.audio_matches.items()
                if match.detected
            },
            "clip_top3": [match.to_dict() for match in perception.top_clip_matches],
            "working_memory": memory.working_memory(),
            "step_index": int(step_index),
        }
        return PlannerDecision(
            mark_complete=[],
            next_goal=next_goal_id,
            action=action,
            direction=str(direction),
            reasoning=reasoning,
            target_positions=target_positions,
            goal_scores=goal_scores,
            prompt_snapshot=prompt_snapshot,
        )
