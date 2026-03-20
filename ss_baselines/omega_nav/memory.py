from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Sequence

import numpy as np

from ss_baselines.omega_nav.perception.base import GoalSpec, PerceptionOutput, SemanticMapState
from ss_baselines.omega_nav.utils import as_serializable, ensure_vec3


@dataclass
class EpisodicMemoryEntry:
    goal_id: str
    source: str
    confidence: float
    description: str
    position: Optional[np.ndarray]
    step_index: int

    def to_dict(self) -> Dict[str, Any]:
        return as_serializable(
            {
                "goal_id": self.goal_id,
                "source": self.source,
                "confidence": float(self.confidence),
                "description": self.description,
                "position": self.position,
                "step_index": int(self.step_index),
            }
        )


class HierarchicalMemory:
    def __init__(self, config: Dict[str, Any]):
        self._working_window = max(int(config.get("working_window", 8)), 1)
        self._episodic_decay = float(config.get("episodic_decay", 0.2))
        self._episodic_prune_threshold = float(config.get("episodic_prune_threshold", 0.05))
        self._revisit_radius_m = float(config.get("revisit_radius_m", 1.5))
        self._goal_specs: Dict[str, GoalSpec] = {}
        self._working_memory: Deque[str] = deque(maxlen=self._working_window)
        self._episodic_memory: Dict[str, List[EpisodicMemoryEntry]] = {}
        self._semantic_map: Optional[SemanticMapState] = None

    def reset(self, goal_specs: Sequence[GoalSpec]) -> None:
        self._goal_specs = {goal.goal_id: goal for goal in goal_specs}
        self._working_memory = deque(maxlen=self._working_window)
        self._episodic_memory = {goal.goal_id: [] for goal in goal_specs}
        self._semantic_map = None

    def _remember(
        self,
        *,
        goal_id: str,
        source: str,
        confidence: float,
        description: str,
        position: Optional[np.ndarray],
        step_index: int,
    ) -> None:
        bucket = self._episodic_memory.setdefault(goal_id, [])
        pos = ensure_vec3(position)
        for entry in bucket:
            if entry.source != source:
                continue
            if pos is None or entry.position is None:
                continue
            if float(np.linalg.norm(pos - entry.position)) <= 0.75:
                entry.confidence = max(float(entry.confidence), float(confidence))
                entry.description = str(description)
                entry.position = pos
                entry.step_index = int(step_index)
                return
        bucket.append(
            EpisodicMemoryEntry(
                goal_id=str(goal_id),
                source=str(source),
                confidence=float(np.clip(confidence, 0.0, 1.0)),
                description=str(description),
                position=pos,
                step_index=int(step_index),
            )
        )

    def update(
        self,
        *,
        step_index: int,
        env: Any,
        perception: PerceptionOutput,
        pending_goal_ids: Sequence[str],
        order_mode: str,
    ) -> None:
        self._working_memory.append(str(perception.observation_summary))
        self._semantic_map = perception.semantic_map

        if str(order_mode).strip().lower() == "ordered":
            tracked_goal_ids = set(pending_goal_ids[:1])
        else:
            tracked_goal_ids = set(pending_goal_ids)

        for goal_id in tracked_goal_ids:
            visual = perception.visual_match(goal_id)
            if visual is not None and visual.visible:
                self._remember(
                    goal_id=goal_id,
                    source="vision",
                    confidence=max(0.75, float(visual.similarity)),
                    description=f"视觉确认 {visual.category} 在 {visual.relative_direction} 侧",
                    position=visual.target_position,
                    step_index=step_index,
                )

            audio = perception.audio_match(goal_id)
            if audio is not None and audio.detected:
                self._remember(
                    goal_id=goal_id,
                    source="audio",
                    confidence=max(0.4, float(audio.aggregated_similarity)),
                    description=f"音频线索：{audio.direction_text}",
                    position=audio.sound_position,
                    step_index=step_index,
                )

        self._decay_nearby_entries(env, step_index)

    def _decay_nearby_entries(self, env: Any, step_index: int) -> None:
        agent_position = np.asarray(env.sim.get_agent_state().position, dtype=np.float32)
        for goal_id, entries in list(self._episodic_memory.items()):
            updated: List[EpisodicMemoryEntry] = []
            for entry in entries:
                if int(entry.step_index) == int(step_index):
                    updated.append(entry)
                    continue
                if entry.position is not None:
                    distance = float(np.linalg.norm((entry.position - agent_position)[[0, 2]]))
                    if distance <= self._revisit_radius_m:
                        entry.confidence -= self._episodic_decay
                if entry.confidence > self._episodic_prune_threshold:
                    updated.append(entry)
            self._episodic_memory[goal_id] = updated

    def confirm_goal(self, goal_id: str) -> None:
        self._episodic_memory.pop(str(goal_id), None)

    def penalize_goal(self, goal_id: str, penalty: float = 0.25) -> None:
        entries = self._episodic_memory.get(str(goal_id), [])
        updated = []
        for entry in entries:
            entry.confidence -= float(penalty)
            if entry.confidence > self._episodic_prune_threshold:
                updated.append(entry)
        self._episodic_memory[str(goal_id)] = updated

    def best_hint(self, goal_id: str) -> Optional[EpisodicMemoryEntry]:
        entries = self._episodic_memory.get(str(goal_id), [])
        if not entries:
            return None
        return max(entries, key=lambda entry: float(entry.confidence))

    def clues_above(self, threshold: float = 0.5) -> List[Dict[str, Any]]:
        clues = []
        for entries in self._episodic_memory.values():
            for entry in entries:
                if float(entry.confidence) >= float(threshold):
                    clues.append(entry.to_dict())
        clues.sort(key=lambda item: float(item.get("confidence", 0.0)), reverse=True)
        return clues

    def exploration_progress(self) -> float:
        if self._semantic_map is None:
            return 0.0
        return float(self._semantic_map.explored_ratio)

    def semantic_map(self) -> Optional[SemanticMapState]:
        return self._semantic_map

    def working_memory(self) -> List[str]:
        return list(self._working_memory)

    def to_prompt_summary(self, threshold: float = 0.5) -> Dict[str, Any]:
        return {
            "clues": self.clues_above(threshold=threshold),
            "exploration_progress": float(self.exploration_progress()),
            "working_memory": self.working_memory(),
        }
