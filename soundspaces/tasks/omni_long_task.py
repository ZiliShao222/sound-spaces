#!/usr/bin/env python3

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Type

import attr
import numpy as np

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
    if isinstance(value, bool):
        return "ordered" if value else "unordered"

    if value is None:
        return "ordered"

    mode = str(value).strip().lower()
    if mode in {"ordered", "order", "true", "1", "yes", "y"}:
        return "ordered"
    if mode in {"unordered", "unorder", "false", "0", "no", "n"}:
        return "unordered"
    return "ordered"


def _optional_order_mode(value: Any) -> Optional[str]:
    if value is None:
        return None
    return _normalize_order_mode(value)


def _resolve_goal_order_mode(
    episode: Episode,
    task: Optional[EmbodiedTask] = None,
    default: Optional[str] = None,
) -> str:
    episode_mode = getattr(episode, "goal_order_mode", None)
    if episode_mode is not None:
        return _normalize_order_mode(episode_mode)

    if task is not None:
        task_config = getattr(task, "_config", None)
        config_mode = getattr(task_config, "GOAL_ORDER_MODE", None)
        if config_mode is not None:
            return _normalize_order_mode(config_mode)

        task_mode = getattr(task, "_mode", None)
        if task_mode is not None:
            return _normalize_order_mode(task_mode)

    if default is not None:
        return _normalize_order_mode(default)
    return "ordered"


def _goal_success_distance(task_config: Config) -> float:
    success_cfg = getattr(task_config, "SUCCESS", None)
    value = getattr(success_cfg, "SUCCESS_DISTANCE", 1.0)
    return float(value)


def _episode_reference_distance(episode: Episode, task: EmbodiedTask) -> float:
    mode = _resolve_goal_order_mode(episode, task=task)
    if mode == "unordered":
        distance = getattr(episode, "unordered_total_geodesic_distance", None)
    else:
        distance = getattr(episode, "ordered_total_geodesic_distance", None)

    try:
        distance_value = float(distance)
    except Exception:
        distance_value = 0.0

    if distance_value > 1e-6:
        return distance_value

    info = getattr(episode, "info", None)
    if isinstance(info, dict):
        try:
            fallback = float(info.get("geodesic_distance", 0.0))
        except Exception:
            fallback = 0.0
        if fallback > 1e-6:
            return fallback

    return 1e-6


def _task_goal_completion(task: EmbodiedTask) -> float:
    goal_states = tuple(getattr(task, "_goals_map", {}).values())
    total = int(len(goal_states))
    if total <= 0:
        return 0.0
    found = sum(1 for goal_state in goal_states if bool(goal_state.get("found", False)))
    return float(found) / float(total)


def _task_strict_success(task: EmbodiedTask) -> float:
    goal_states = tuple(getattr(task, "_goals_map", {}).values())
    if not goal_states:
        return 1.0
    return 1.0 if all(bool(goal_state.get("found", False)) for goal_state in goal_states) else 0.0


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
    goal_order_mode: Optional[str] = attr.ib(default=None, converter=_optional_order_mode)
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
        self._mode = _resolve_goal_order_mode(episode, task=self)
        self._submit_count: int = 0
        self._submit_limit: int = max(0, int(len(self._all_goals)) - 1)
        self._goals_map = self._build_goals_map(getattr(episode, "tasks", None))
        self._refresh_episode_goals(episode, apply_to_sim=False)

        return super().reset(episode)

    @property
    def all_goals(self) -> Sequence[SemanticAudioGoal]:
        return tuple(self._all_goals)

    @property
    def goal_state_map(self) -> Dict[str, Dict[str, Any]]:
        return {
            goal_key: {
                "goal_index": int(goal_state["goal_index"]),
                "descriptor": goal_state["descriptor"],
                "found": bool(goal_state["found"]),
            }
            for goal_key, goal_state in self._goals_map.items()
        }

    @property
    def order_enforced(self) -> bool:
        return bool(self._mode == "ordered")

    @property
    def submit_count(self) -> int:
        return int(self._submit_count)

    @property
    def submit_limit(self) -> int:
        return int(self._submit_limit)

    def _check_episode_is_active(
        self,
        *args: Any,
        observations=None,
        action=None,
        episode: Episode,
        **kwargs: Any,
    ) -> bool:
        if getattr(self, "is_stop_called", False):
            self.is_stop_called = False  # type: ignore
            self._handle_stop(episode)
            return False
        if getattr(self, "is_submit_called", False):
            self.is_submit_called = False  # type: ignore
            if self._submit_count >= self._submit_limit:
                self._handle_stop(episode)
                return False
            self._submit_count += 1
            return self._handle_submit(episode)
        return True

    def _handle_submit(self, episode: Episode) -> bool:
        goal_state = self._match_goal_state(episode)
        if goal_state is None:
            return True
        self._mark_goal_found(goal_state)
        if all(bool(state["found"]) for state in self._goals_map.values()):
            return False
        self._refresh_episode_goals(episode, apply_to_sim=True)
        return True

    def _handle_stop(self, episode: Episode) -> None:
        goal_state = self._match_goal_state(episode)
        if goal_state is not None:
            self._mark_goal_found(goal_state)

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
            active_goal_state = self._ordered_active_goal_state()
            if active_goal_state is None:
                return 0.0
            return self._distance_to_goal(active_goal_state["goal"], episode)

        if all(bool(goal_state["found"]) for goal_state in self._goals_map.values()):
            return 0.0

        best_distance: Optional[float] = None
        for goal_state in self._iter_unfound_goal_states():
            distance = self._distance_to_goal(goal_state["goal"], episode)
            if distance is None:
                continue
            if best_distance is None or float(distance) < float(best_distance):
                best_distance = float(distance)
        return best_distance

    def _build_goals_map(self, raw_tasks: Any) -> Dict[str, Dict[str, Any]]:
        goals_map: Dict[str, Dict[str, Any]] = {}
        for goal_index, goal in enumerate(self._all_goals):
            descriptor: Any = int(goal_index)
            if isinstance(raw_tasks, list) and goal_index < len(raw_tasks):
                raw_goal = raw_tasks[goal_index]
                if isinstance(raw_goal, (list, tuple)) and len(raw_goal) >= 2:
                    descriptor = (str(raw_goal[0]), str(raw_goal[1]))
                elif isinstance(raw_goal, dict):
                    descriptor = (
                        str(raw_goal.get("instance_key", raw_goal.get("goal", goal_index))),
                        str(raw_goal.get("modality", "object")),
                    )

            goal_key = "goal_{:03d}".format(int(goal_index))
            goals_map[goal_key] = {
                "goal_index": int(goal_index),
                "goal": goal,
                "descriptor": descriptor,
                "found": False,
            }
        return goals_map

    def _iter_unfound_goal_states(self):
        for goal_state in self._goals_map.values():
            if not bool(goal_state["found"]):
                yield goal_state

    def _ordered_active_goal_state(self):
        for goal_state in self._iter_unfound_goal_states():
            return goal_state
        return None

    def _match_goal_state(self, episode: Episode):
        if not self._goals_map:
            return None

        success_distance = _goal_success_distance(self._config)
        if self._mode == "ordered":
            active_goal_state = self._ordered_active_goal_state()
            if active_goal_state is None:
                return None
            distance = self._distance_to_goal(active_goal_state["goal"], episode)
            if distance is None or float(distance) >= success_distance:
                return None
            return active_goal_state

        winner_state = None
        winner_distance: Optional[float] = None
        for goal_state in self._iter_unfound_goal_states():
            distance = self._distance_to_goal(goal_state["goal"], episode)
            if distance is None:
                continue
            if float(distance) >= success_distance:
                continue
            if winner_distance is None or float(distance) < winner_distance:
                winner_distance = float(distance)
                winner_state = goal_state
        return winner_state

    def _mark_goal_found(self, goal_state: Dict[str, Any]) -> None:
        goal_state["found"] = True

    def _refresh_episode_goals(self, episode: Episode, apply_to_sim: bool) -> None:
        if self._mode != "ordered":
            episode.goals = list(self._all_goals)
            return

        active_goal_state = self._ordered_active_goal_state()
        if active_goal_state is None:
            episode.goals = list(self._all_goals)
            return

        goal_index = int(active_goal_state["goal_index"])
        episode.goals = [active_goal_state["goal"]]
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
        goal_states = tuple(getattr(task, "_goals_map", {}).values())
        self._metric = int(
            sum(1 for goal_state in goal_states if bool(goal_state.get("found", False)))
        )


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
        self._metric = _task_goal_completion(task)


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
        self._metric = _task_strict_success(task)


@registry.register_measure
class OmniLongSuccess(Measure):
    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        self._config = config
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "success"

    def reset_metric(self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any):
        self.update_metric(episode=episode, task=task)

    def update_metric(self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any):
        self._metric = _task_goal_completion(task)


@registry.register_measure
class OmniLongSPL(Measure):
    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._previous_position: Optional[np.ndarray] = None
        self._agent_episode_distance: float = 0.0
        self._reference_distance: float = 1e-6
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "spl"

    def reset_metric(self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any):
        self._previous_position = np.array(self._sim.get_agent_state().position, dtype=np.float32)
        self._agent_episode_distance = 0.0
        self._reference_distance = _episode_reference_distance(episode, task)
        self.update_metric(episode=episode, task=task)

    def _update_path_length(self) -> None:
        current_position = np.array(self._sim.get_agent_state().position, dtype=np.float32)
        if self._previous_position is not None:
            self._agent_episode_distance += float(
                np.linalg.norm(current_position - self._previous_position, ord=2)
            )
        self._previous_position = current_position

    def _success_value(self, task: EmbodiedTask) -> float:
        return _task_strict_success(task)

    def update_metric(self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any):
        self._update_path_length()
        success_value = self._success_value(task)
        self._metric = float(success_value) * (
            self._reference_distance
            / max(self._reference_distance, float(self._agent_episode_distance))
        )


@registry.register_measure
class OmniLongSoftSPL(OmniLongSPL):
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "softspl"

    def _success_value(self, task: EmbodiedTask) -> float:
        return _task_goal_completion(task)


@registry.register_task_action
class LifelongSubmitAction(SimulatorTaskAction):
    name: str = "LIFELONG_SUBMIT"

    def reset(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        task.is_submit_called = False  # type: ignore

    def step(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        task.is_submit_called = True  # type: ignore
        return self._sim.get_observations_at()  # type: ignore
