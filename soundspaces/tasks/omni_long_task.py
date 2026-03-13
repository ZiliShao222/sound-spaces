#!/usr/bin/env python3

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Set, Type

import attr

from habitat.config import Config
from habitat.core.dataset import Episode
from habitat.core.embodied_task import SimulatorTaskAction
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import EmbodiedTask, Measure, NavigationEpisode, NavigationTask

from soundspaces.tasks.semantic_audionav_task import SemanticAudioGoal


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def _normalize_order_mode(value: Any) -> str:
    mode = str(value or "ordered").strip().lower()
    if mode not in {"ordered", "unordered"}:
        return "ordered"
    return mode


def _normalize_task_specs(raw_tasks: Any, goal_count: int) -> List[List[str]]:
    specs: List[List[str]] = []
    if isinstance(raw_tasks, list):
        for idx, item in enumerate(raw_tasks):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                specs.append([str(item[0]), str(item[1])])
                continue
            if isinstance(item, dict):
                instance_key = item.get("instance_key", item.get("goal", idx))
                modality = item.get("modality", "object")
                specs.append([str(instance_key), str(modality)])
                continue
            specs.append([str(idx), "object"])

    if goal_count <= 0:
        return specs

    while len(specs) < goal_count:
        specs.append([str(len(specs)), "object"])
    return specs[:goal_count]


def _goal_success_distance(task_config: Config) -> float:
    success_cfg = getattr(task_config, "SUCCESS", None)
    value = getattr(success_cfg, "SUCCESS_DISTANCE", 1.0)
    return float(value)


@attr.s(auto_attribs=True, kw_only=True)
class OmniLongEpisode(NavigationEpisode):
    object_category: str
    sound_id: str
    sound_sources: Optional[List[dict]] = None
    sound_source_schedule: Optional[list] = None
    distractor_sound_id: Optional[str] = None
    distractor_position_index: Optional[int] = attr.ib(
        default=None,
        converter=lambda x: int(x) if x is not None else None,
    )
    offset: int = attr.ib(default=0, converter=int)
    duration: int = attr.ib(default=25, converter=int)
    num_goals: Optional[int] = attr.ib(
        default=None,
        converter=lambda x: int(x) if x is not None else None,
    )
    tasks: Optional[List[List[str]]] = None
    goal_order_mode: str = attr.ib(default="ordered", converter=_normalize_order_mode)
    ordered_total_geodesic_distance: Optional[float] = attr.ib(
        default=None,
        converter=_optional_float,
    )
    unordered_total_geodesic_distance: Optional[float] = attr.ib(
        default=None,
        converter=_optional_float,
    )

    @property
    def goals_key(self) -> str:
        return f"{os.path.basename(self.scene_id)}_{self.object_category}"


def merge_omni_long_sim_episode_config(
    sim_config: Config,
    episode: Type[Episode],
) -> Any:
    sim_config.defrost()
    sim_config.SCENE = episode.scene_id
    sim_config.freeze()

    if episode.start_position is None or episode.start_rotation is None:
        return sim_config

    agent_name = sim_config.AGENTS[sim_config.DEFAULT_AGENT_ID]
    agent_cfg = getattr(sim_config, agent_name)
    agent_cfg.defrost()
    agent_cfg.START_POSITION = episode.start_position
    agent_cfg.START_ROTATION = episode.start_rotation

    episode_goals = list(getattr(episode, "goals", []) or [])
    if episode_goals:
        goal_positions = [goal.position for goal in episode_goals]
        agent_cfg.GOAL_POSITION = goal_positions[0]
        agent_cfg.GOAL_POSITIONS = goal_positions

    if hasattr(episode, "sound_sources") and episode.sound_sources:
        sources: List[dict] = []
        for idx, source in enumerate(episode.sound_sources):
            source_dict = dict(source)
            if "position" not in source_dict and episode_goals:
                goal_pos = goal_positions[idx] if idx < len(goal_positions) else goal_positions[-1]
                source_dict["position"] = goal_pos
            sources.append(source_dict)
        if sources:
            agent_cfg.SOUND_SOURCES = sources
            first_sound = sources[0].get("sound_id")
            if first_sound is not None:
                agent_cfg.SOUND_ID = first_sound
    elif hasattr(episode, "sound_id"):
        agent_cfg.SOUND_ID = episode.sound_id

    if hasattr(episode, "sound_source_schedule") and episode.sound_source_schedule:
        agent_cfg.SOUND_SOURCE_SCHEDULE = episode.sound_source_schedule
    if hasattr(episode, "distractor_sound_id"):
        agent_cfg.DISTRACTOR_SOUND_ID = episode.distractor_sound_id
    if hasattr(episode, "distractor_position_index"):
        agent_cfg.DISTRACTOR_POSITION_INDEX = episode.distractor_position_index
    if hasattr(episode, "offset"):
        agent_cfg.OFFSET = episode.offset
    if hasattr(episode, "duration"):
        agent_cfg.DURATION = episode.duration

    agent_cfg.IS_SET_START_STATE = True
    agent_cfg.freeze()
    return sim_config


@registry.register_task(name="OmniLongSemanticAudioNav")
class OmniLongNavigationTask(NavigationTask):
    def overwrite_sim_config(self, sim_config: Any, episode: Type[Episode]) -> Any:
        return merge_omni_long_sim_episode_config(sim_config, episode)

    def reset(self, episode: Episode):
        self._all_goals: List[SemanticAudioGoal] = list(getattr(episode, "goals", []) or [])
        self._task_specs: List[List[str]] = _normalize_task_specs(
            getattr(episode, "tasks", None),
            len(self._all_goals),
        )
        self._mode = _normalize_order_mode(
            getattr(episode, "goal_order_mode", getattr(self._config, "GOAL_ORDER_MODE", "ordered"))
        )
        self._remaining_goal_indices: Set[int] = set(range(len(self._all_goals)))
        self._completed_goal_indices: List[int] = []
        self._current_goal_index: Optional[int] = None
        self._current_task_token: Optional[List[str]] = None
        self._submit_count: int = 0
        self._submit_limit: int = max(0, int(len(self._all_goals)) - 1)
        self._submit_rejected_count: int = 0

        if self._all_goals:
            if self._mode == "ordered":
                self._set_active_goal_index(episode, 0, apply_to_sim=False)
            else:
                start_idx = self._select_unordered_active_index(episode)
                self._set_active_goal_index(episode, start_idx, apply_to_sim=False)

        return super().reset(episode)

    @property
    def completed_goal_indices(self) -> Sequence[int]:
        return tuple(self._completed_goal_indices)

    @property
    def remaining_goal_indices(self) -> Sequence[int]:
        return tuple(sorted(self._remaining_goal_indices))

    @property
    def current_task_token(self) -> Optional[List[str]]:
        return self._current_task_token

    @property
    def submit_count(self) -> int:
        return int(self._submit_count)

    @property
    def submit_limit(self) -> int:
        return int(self._submit_limit)

    @property
    def submit_rejected_count(self) -> int:
        return int(self._submit_rejected_count)

    @property
    def remaining_submit_quota(self) -> int:
        return int(max(0, self._submit_limit - self._submit_count))

    def _check_episode_is_active(
        self,
        *args: Any,
        observations=None,
        action=None,
        episode: Episode,
        **kwargs: Any,
    ) -> bool:
        if getattr(self, "is_stop_called", False):
            return False
        if getattr(self, "is_submit_called", False):
            self.is_submit_called = False  # type: ignore
            if self._submit_count >= self._submit_limit:
                self._submit_rejected_count += 1
                return True
            self._submit_count += 1
            return self._handle_submit(episode)
        return True

    def _handle_submit(self, episode: Episode) -> bool:
        if not self._all_goals:
            return True
        if not self._remaining_goal_indices:
            return False

        success_distance = _goal_success_distance(self._config)

        if self._mode == "ordered":
            goal_index = self._current_goal_index
            if goal_index is None or goal_index not in self._remaining_goal_indices:
                return bool(self._remaining_goal_indices)
            current_goal = self._all_goals[goal_index]
            distance = self._distance_to_goal(current_goal, episode)
            if distance is None or float(distance) >= success_distance:
                return True
            self._mark_goal_completed(goal_index)
            if not self._remaining_goal_indices:
                return False
            next_index = min(self._remaining_goal_indices)
            self._set_active_goal_index(episode, next_index, apply_to_sim=True)
            return True

        goal_index = self._match_submitted_unordered_goal(episode, success_distance)
        if goal_index is None:
            return True
        self._mark_goal_completed(goal_index)
        if not self._remaining_goal_indices:
            return False
        next_index = self._select_unordered_active_index(episode)
        self._set_active_goal_index(episode, next_index, apply_to_sim=True)
        return True

    def _distance_to_goal(self, goal: SemanticAudioGoal, episode: Episode) -> Optional[float]:
        current_position = self._sim.get_agent_state().position
        distance_to_cfg = getattr(self._config, "DISTANCE_TO_GOAL", None)
        distance_to = getattr(distance_to_cfg, "DISTANCE_TO", "POINT")
        if distance_to == "VIEW_POINTS" and goal.view_points:
            targets = [view.agent_state.position for view in goal.view_points]
        else:
            targets = [goal.position]
        return self._sim.geodesic_distance(current_position, targets, episode)

    def _distance_to_goal_by_mode(self, episode: Episode) -> Optional[float]:
        if not self._all_goals:
            return 0.0

        if self._mode == "ordered":
            submit_count = int(getattr(self, "submit_count", 0))
            goal_index = min(max(0, submit_count), len(self._all_goals) - 1)
            goal = self._all_goals[goal_index]
            return self._distance_to_goal(goal, episode)

        if not self._remaining_goal_indices:
            return 0.0

        best_distance: Optional[float] = None
        for goal_index in sorted(self._remaining_goal_indices):
            goal = self._all_goals[goal_index]
            distance = self._distance_to_goal(goal, episode)
            if distance is None:
                continue
            if best_distance is None or float(distance) < float(best_distance):
                best_distance = float(distance)
        return best_distance

    def _match_submitted_unordered_goal(
        self,
        episode: Episode,
        success_distance: float,
    ) -> Optional[int]:
        winner_index: Optional[int] = None
        winner_distance: Optional[float] = None
        for goal_index in sorted(self._remaining_goal_indices):
            goal = self._all_goals[goal_index]
            distance = self._distance_to_goal(goal, episode)
            if distance is None:
                continue
            if float(distance) >= success_distance:
                continue
            if winner_distance is None or float(distance) < winner_distance:
                winner_distance = float(distance)
                winner_index = goal_index
        return winner_index

    def _select_unordered_active_index(self, episode: Episode) -> Optional[int]:
        if not self._remaining_goal_indices:
            return None
        current_position = self._sim.get_agent_state().position
        best_index: Optional[int] = None
        best_distance: Optional[float] = None
        for goal_index in sorted(self._remaining_goal_indices):
            goal = self._all_goals[goal_index]
            distance = self._sim.geodesic_distance(current_position, [goal.position], episode)
            if best_distance is None or float(distance) < best_distance:
                best_distance = float(distance)
                best_index = goal_index
        return best_index

    def _mark_goal_completed(self, goal_index: int) -> None:
        if goal_index in self._remaining_goal_indices:
            self._remaining_goal_indices.remove(goal_index)
        if goal_index not in self._completed_goal_indices:
            self._completed_goal_indices.append(goal_index)

    def _set_active_goal_index(
        self,
        episode: Episode,
        goal_index: Optional[int],
        apply_to_sim: bool,
    ) -> None:
        self._current_goal_index = goal_index
        if goal_index is None:
            self._current_task_token = None
            return

        episode.goals = [self._all_goals[goal_index]]
        if goal_index < len(self._task_specs):
            self._current_task_token = list(self._task_specs[goal_index])
        else:
            self._current_task_token = [str(goal_index), "object"]

        if apply_to_sim:
            self._apply_goal_to_sim(episode, goal_index)

    def _apply_goal_to_sim(self, episode: Episode, goal_index: int) -> None:
        if hasattr(self._sim, "set_active_sound_index"):
            try:
                self._sim.set_active_sound_index(goal_index)
                return
            except Exception:
                pass

        goal = self._all_goals[goal_index]
        sound_id = None
        sound_sources = getattr(episode, "sound_sources", None)
        if isinstance(sound_sources, list) and goal_index < len(sound_sources):
            candidate = sound_sources[goal_index]
            if isinstance(candidate, dict):
                sound_id = candidate.get("sound_id")
        if hasattr(self._sim, "set_active_goal"):
            try:
                self._sim.set_active_goal(goal.position, sound_id=sound_id)
            except Exception:
                pass


@registry.register_measure
class OmniLongDistanceToGoal(Measure):
    cls_uuid: str = "distance_to_goal"

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        self._config = config
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any):
        self.update_metric(episode=episode, task=task)

    def update_metric(self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any):
        if hasattr(task, "_distance_to_goal_by_mode"):
            distance = task._distance_to_goal_by_mode(episode)
            self._metric = float(distance) if distance is not None else float("inf")
            return

        goals = getattr(episode, "goals", ())
        if not goals:
            self._metric = 0.0
            return

        current_position = self._sim.get_agent_state().position
        self._metric = self._sim.geodesic_distance(
            current_position,
            [goals[0].position],
            episode,
        )


@registry.register_measure
class LifelongGoalsFound(Measure):
    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        self._config = config
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "lifelong_goals_found"

    def reset_metric(self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any):
        self.update_metric(episode=episode, task=task)

    def update_metric(self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any):
        done = getattr(task, "completed_goal_indices", ())
        self._metric = int(len(done))


@registry.register_measure
class LifelongGoalCompletion(Measure):
    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        self._config = config
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "lifelong_goal_completion"

    def reset_metric(self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any):
        self.update_metric(episode=episode, task=task)

    def update_metric(self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any):
        done = getattr(task, "completed_goal_indices", ())
        all_goals = getattr(task, "_all_goals", ())
        total = int(len(all_goals))
        if total <= 0:
            self._metric = 0.0
            return
        self._metric = float(len(done)) / float(total)


@registry.register_measure
class LifelongTaskSuccess(Measure):
    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        self._config = config
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "lifelong_task_success"

    def reset_metric(self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any):
        self.update_metric(episode=episode, task=task)

    def update_metric(self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any):
        remaining = getattr(task, "remaining_goal_indices", ())
        self._metric = 1.0 if len(remaining) == 0 else 0.0


@registry.register_task_action
class LifelongSubmitAction(SimulatorTaskAction):
    name: str = "LIFELONG_SUBMIT"

    def reset(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        task.is_submit_called = False  # type: ignore

    def step(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        task.is_submit_called = True  # type: ignore
        return self._sim.get_observations_at()  # type: ignore
