#!/usr/bin/env python3

from __future__ import annotations

import os
from itertools import permutations
from typing import Any, Dict, List, Optional, Sequence, Type

import attr
import clip
import numpy as np
import torch
from gym import spaces
from PIL import Image

from habitat.config import Config
from habitat.core.dataset import Episode
from habitat.core.embodied_task import SimulatorTaskAction
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes, Simulator
from habitat.tasks.nav.nav import DistanceToGoal, EmbodiedTask, Measure, NavigationEpisode, NavigationTask

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


def _mode_config_float(
    config: Any,
    mode: str,
    ordered_key: str,
    unordered_key: str,
    fallback_key: str,
    default: float,
) -> float:
    mode_key = ordered_key if _normalize_order_mode(mode) == "ordered" else unordered_key
    value = getattr(config, mode_key, None)
    if value is None:
        value = getattr(config, fallback_key, default)
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


def _task_goals_found(task: EmbodiedTask) -> int:
    goal_states = tuple(getattr(task, "_goals_map", {}).values())
    return int(sum(1 for goal_state in goal_states if bool(goal_state.get("found", False))))


def _task_active_goal_index(task: EmbodiedTask) -> Optional[int]:
    try:
        if hasattr(task, "_navigation_goal_state"):
            goal_state = task._navigation_goal_state()
        else:
            goal_state = task._ordered_active_goal_state()
    except Exception:
        goal_state = None
    if not isinstance(goal_state, dict):
        return None
    goal_index = goal_state.get("goal_index")
    if goal_index is None:
        return None
    return int(goal_index)


def _finite_distance(value: Any) -> Optional[float]:
    try:
        distance = float(value)
    except Exception:
        return None
    if not np.isfinite(distance):
        return None
    return distance


def _is_vec3(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(isinstance(v, (int, float)) for v in value)
    )


def _is_quat4(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(isinstance(v, (int, float)) for v in value)
    )


def _extract_rgb_from_observations(observations: Dict[str, Any]) -> Optional[np.ndarray]:
    for key in ("rgb", "RGB_SENSOR", "rgb_sensor"):
        if key in observations:
            candidate = observations[key]
            arr = np.asarray(candidate)
            if arr.ndim == 3:
                if arr.shape[2] == 4:
                    arr = arr[:, :, :3]
                elif arr.shape[2] == 1:
                    arr = np.repeat(arr, 3, axis=2)
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                return arr
    for value in observations.values():
        arr = np.asarray(value)
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return arr
    return None


def _goal_target_positions(goal: SemanticAudioGoal, task_config: Config) -> List[Any]:
    distance_to_cfg = getattr(task_config, "DISTANCE_TO_GOAL", None)
    distance_to = getattr(distance_to_cfg, "DISTANCE_TO", "POINT")
    if distance_to == "VIEW_POINTS" and getattr(goal, "view_points", None):
        positions: List[Any] = []
        for view in goal.view_points:
            agent_state = getattr(view, "agent_state", None)
            position = getattr(agent_state, "position", None)
            if position is not None:
                positions.append(position)
        if positions:
            return positions
    return [goal.position]


def _distance_from_position_to_goal(
    sim: Simulator,
    task_config: Config,
    position: Any,
    goal: SemanticAudioGoal,
    episode: Episode,
) -> Optional[float]:
    distance = sim.geodesic_distance(
        position,
        _goal_target_positions(goal, task_config),
        episode,
    )
    return _finite_distance(distance)


def _distance_between_goals(
    sim: Simulator,
    task_config: Config,
    source_goal: SemanticAudioGoal,
    target_goal: SemanticAudioGoal,
    episode: Episode,
) -> Optional[float]:
    best_distance: Optional[float] = None
    for source_position in _goal_target_positions(source_goal, task_config):
        pair_distance = _distance_from_position_to_goal(
            sim,
            task_config,
            source_position,
            target_goal,
            episode,
        )
        if pair_distance is None:
            continue
        if best_distance is None or float(pair_distance) < float(best_distance):
            best_distance = float(pair_distance)
    return best_distance


def _shortest_remaining_path_length(task: EmbodiedTask, episode: Episode) -> Optional[float]:
    goal_states = [
        goal_state
        for goal_state in getattr(task, "_goals_map", {}).values()
        if not bool(goal_state.get("found", False))
    ]
    if not goal_states:
        return 0.0

    current_position = task._sim.get_agent_state().position
    start_distances: Dict[int, float] = {}
    pair_distances: Dict[tuple, float] = {}

    for goal_state in goal_states:
        goal_index = int(goal_state["goal_index"])
        distance = _distance_from_position_to_goal(
            task._sim,
            task._config,
            current_position,
            goal_state["goal"],
            episode,
        )
        if distance is None:
            return None
        start_distances[goal_index] = float(distance)

    for source_state in goal_states:
        source_goal_index = int(source_state["goal_index"])
        for target_state in goal_states:
            target_goal_index = int(target_state["goal_index"])
            if source_goal_index == target_goal_index:
                continue
            pair_distance = _distance_between_goals(
                task._sim,
                task._config,
                source_state["goal"],
                target_state["goal"],
                episode,
            )
            if pair_distance is None:
                return None
            pair_distances[(source_goal_index, target_goal_index)] = float(pair_distance)

    best_total: Optional[float] = None
    goal_indices = [int(goal_state["goal_index"]) for goal_state in goal_states]
    for ordering in permutations(goal_indices):
        total = start_distances[ordering[0]]
        for source_goal_index, target_goal_index in zip(ordering[:-1], ordering[1:]):
            total += pair_distances[(source_goal_index, target_goal_index)]
        if best_total is None or float(total) < float(best_total):
            best_total = float(total)

    return best_total


def _ordered_inactive_goal_hits(task: EmbodiedTask, episode: Episode) -> int:
    if str(getattr(task, "goal_order_mode", "ordered")) != "ordered":
        return 0

    success_distance = _goal_success_distance(task._config)
    active_goal_state = None
    if hasattr(task, "_navigation_goal_state"):
        active_goal_state = task._navigation_goal_state(episode)

    hit_count = 0
    for goal_state in getattr(task, "_goals_map", {}).values():
        if bool(goal_state.get("found", False)):
            continue
        if active_goal_state is not None and int(goal_state["goal_index"]) == int(active_goal_state["goal_index"]):
            continue
        distance = task._distance_to_goal(goal_state["goal"], episode)
        if distance is not None and float(distance) < float(success_distance):
            hit_count += 1
    return int(hit_count)


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
        self._current_episode = episode
        self._mode = _resolve_goal_order_mode(episode, task=self)
        self._submit_count: int = 0
        self._submit_limit: int = max(0, int(len(self._all_goals)) - 1)
        self._goals_map = self._build_goals_map(getattr(episode, "tasks", None))
        self._unordered_goal_lock_index: Optional[int] = None
        self._unordered_goal_lock_best_distance: Optional[float] = None
        self._unordered_lock_candidate_index: Optional[int] = None
        self._unordered_lock_candidate_anchor_distance: Optional[float] = None
        self._refresh_episode_goals(episode, apply_to_sim=False)

        return super().reset(episode)

    @property
    def all_goals(self) -> Sequence[SemanticAudioGoal]:
        return tuple(self._all_goals)

    @property
    def order_enforced(self) -> bool:
        return bool(self._mode == "ordered")

    @property
    def goal_order_mode(self) -> str:
        return str(self._mode)

    @property
    def active_subtask_idx(self) -> int:
        active_goal_state = self._navigation_goal_state()
        if active_goal_state is None:
            return int(len(self._all_goals))
        return int(active_goal_state.get("goal_index", len(self._all_goals)))

    def _goal_state_by_index(self, goal_index: int) -> Optional[Dict[str, Any]]:
        goal_key = "goal_{:03d}".format(int(goal_index))
        goal_state = self._goals_map.get(goal_key)
        if not isinstance(goal_state, dict):
            return None
        return goal_state

    def _goal_state_distance(
        self,
        goal_state: Optional[Dict[str, Any]],
        episode: Optional[Episode] = None,
    ) -> Optional[float]:
        if not isinstance(goal_state, dict):
            return None
        reference_episode = episode if episode is not None else self._current_episode
        if reference_episode is None:
            return None
        return self._distance_to_goal(goal_state["goal"], reference_episode)

    def _clear_unordered_goal_lock(self) -> None:
        self._unordered_goal_lock_index = None
        self._unordered_goal_lock_best_distance = None
        self._unordered_lock_candidate_index = None
        self._unordered_lock_candidate_anchor_distance = None

    def _current_unordered_locked_goal_state(self) -> Optional[Dict[str, Any]]:
        if self._unordered_goal_lock_index is None:
            return None
        goal_state = self._goal_state_by_index(self._unordered_goal_lock_index)
        if goal_state is None or bool(goal_state.get("found", False)):
            return None
        return goal_state

    def _select_nearest_unfound_goal_state(
        self,
        episode: Optional[Episode] = None,
    ) -> Optional[Dict[str, Any]]:
        reference_episode = episode if episode is not None else self._current_episode
        winner_state = None
        winner_distance: Optional[float] = None
        for goal_state in self._iter_unfound_goal_states():
            if reference_episode is None:
                return goal_state
            distance = self._goal_state_distance(goal_state, reference_episode)
            if distance is None:
                continue
            if winner_distance is None or float(distance) < float(winner_distance):
                winner_distance = float(distance)
                winner_state = goal_state
        return winner_state

    def _update_unordered_goal_lock(self, episode: Optional[Episode] = None) -> None:
        if self._mode != "unordered":
            self._clear_unordered_goal_lock()
            return

        reference_episode = episode if episode is not None else self._current_episode
        if reference_episode is None:
            return

        if not bool(getattr(self._config, "UNORDERED_TARGET_LOCK_ENABLED", True)):
            self._clear_unordered_goal_lock()
            return

        locked_goal_state = self._current_unordered_locked_goal_state()
        if locked_goal_state is not None:
            current_distance = self._goal_state_distance(locked_goal_state, reference_episode)
            if current_distance is not None:
                if (
                    self._unordered_goal_lock_best_distance is None
                    or float(current_distance) < float(self._unordered_goal_lock_best_distance)
                ):
                    self._unordered_goal_lock_best_distance = float(current_distance)

                if bool(getattr(self._config, "UNORDERED_TARGET_LOCK_ALLOW_RELEASE", True)):
                    release_delta = float(
                        getattr(
                            self._config,
                            "UNORDERED_TARGET_LOCK_RELEASE_DISTANCE_DELTA",
                            2.0,
                        )
                    )
                    if (
                        self._unordered_goal_lock_best_distance is not None
                        and float(current_distance) - float(self._unordered_goal_lock_best_distance)
                        >= float(release_delta)
                    ):
                        self._clear_unordered_goal_lock()

        if self._current_unordered_locked_goal_state() is not None:
            return

        nearest_goal_state = self._select_nearest_unfound_goal_state(reference_episode)
        if nearest_goal_state is None:
            self._clear_unordered_goal_lock()
            return

        nearest_goal_index = int(nearest_goal_state["goal_index"])
        nearest_distance = self._goal_state_distance(nearest_goal_state, reference_episode)
        if nearest_distance is None:
            return

        acquire_progress = float(
            getattr(self._config, "UNORDERED_TARGET_LOCK_ACQUIRE_PROGRESS", 1.0)
        )

        if (
            self._unordered_lock_candidate_index != nearest_goal_index
            or self._unordered_lock_candidate_anchor_distance is None
        ):
            self._unordered_lock_candidate_index = nearest_goal_index
            self._unordered_lock_candidate_anchor_distance = float(nearest_distance)
            if acquire_progress <= 1e-6:
                self._unordered_goal_lock_index = nearest_goal_index
                self._unordered_goal_lock_best_distance = float(nearest_distance)
            return

        if (
            acquire_progress <= 1e-6
            or float(self._unordered_lock_candidate_anchor_distance) - float(nearest_distance)
            >= float(acquire_progress)
        ):
            self._unordered_goal_lock_index = nearest_goal_index
            self._unordered_goal_lock_best_distance = float(nearest_distance)


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

        active_goal_state = self._navigation_goal_state(episode)
        if active_goal_state is None:
            return 0.0
        return self._goal_state_distance(active_goal_state, episode)

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

    def _unordered_active_goal_state(self, episode: Optional[Episode] = None):
        reference_episode = episode if episode is not None else self._current_episode
        self._update_unordered_goal_lock(reference_episode)

        locked_goal_state = self._current_unordered_locked_goal_state()
        if locked_goal_state is not None:
            return locked_goal_state
        return self._select_nearest_unfound_goal_state(reference_episode)

    def _navigation_goal_state(self, episode: Optional[Episode] = None):
        if self._mode == "ordered":
            return self._ordered_active_goal_state()
        return self._unordered_active_goal_state(episode)

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
        goal_index = int(goal_state.get("goal_index", -1))
        if self._unordered_goal_lock_index is not None and int(self._unordered_goal_lock_index) == goal_index:
            self._clear_unordered_goal_lock()
        if self._unordered_lock_candidate_index is not None and int(self._unordered_lock_candidate_index) == goal_index:
            self._unordered_lock_candidate_index = None
            self._unordered_lock_candidate_anchor_distance = None

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
class OmniLongDistanceToGoalReward(Measure):
    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._previous_distance: Optional[float] = None
        self._previous_remaining_path_length: Optional[float] = None
        self._previous_active_goal_index: Optional[int] = None
        self._previous_goals_found: int = 0
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "omni_long_distance_to_goal_reward"

    def _unordered_dense_reward_mode(self) -> str:
        return str(
            getattr(self._config, "UNORDERED_DENSE_REWARD_MODE", "global_path_reduction")
        ).strip().lower()

    def _milestone_reward(self, mode: str, previous_goals_found: int, goals_found_delta: int) -> float:
        base_reward = _mode_config_float(
            self._config,
            mode,
            ordered_key="ORDERED_SUBMIT_SUCCESS_REWARD",
            unordered_key="UNORDERED_SUBMIT_SUCCESS_REWARD",
            fallback_key="SUBMIT_SUCCESS_REWARD",
            default=10.0,
        )
        reward_increment = _mode_config_float(
            self._config,
            mode,
            ordered_key="ORDERED_SUBMIT_SUCCESS_REWARD_INCREMENT",
            unordered_key="UNORDERED_SUBMIT_SUCCESS_REWARD_INCREMENT",
            fallback_key="SUBMIT_SUCCESS_REWARD_INCREMENT",
            default=0.0,
        )

        total_reward = 0.0
        for found_offset in range(int(goals_found_delta)):
            total_reward += float(base_reward) + float(reward_increment) * float(
                int(previous_goals_found) + int(found_offset)
            )
        return float(total_reward)

    def reset_metric(self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [
                DistanceToGoal.cls_uuid,
                "lifelong_goals_found",
            ],
        )
        if hasattr(task, "_update_unordered_goal_lock"):
            task._update_unordered_goal_lock(episode)
        self._previous_distance = _finite_distance(
            task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        )
        self._previous_remaining_path_length = _shortest_remaining_path_length(task, episode)
        self._previous_active_goal_index = _task_active_goal_index(task)
        self._previous_goals_found = _task_goals_found(task)
        self._metric = 0.0

    def update_metric(self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any):
        mode = getattr(task, "goal_order_mode", _resolve_goal_order_mode(episode, task=task))
        if hasattr(task, "_update_unordered_goal_lock"):
            task._update_unordered_goal_lock(episode)

        current_distance = _finite_distance(
            task.measurements.measures[DistanceToGoal.cls_uuid].get_metric()
        )
        current_remaining_path_length = _shortest_remaining_path_length(task, episode)
        current_goals_found = _task_goals_found(task)
        current_active_goal_index = _task_active_goal_index(task)

        goals_found_delta = max(0, int(current_goals_found) - int(self._previous_goals_found))
        goal_switched = goals_found_delta > 0 or (
            self._previous_active_goal_index is not None
            and current_active_goal_index is not None
            and int(current_active_goal_index) != int(self._previous_active_goal_index)
        )

        reward = 0.0
        distance_scale = _mode_config_float(
            self._config,
            mode,
            ordered_key="ORDERED_DISTANCE_REWARD_SCALE",
            unordered_key="UNORDERED_DISTANCE_REWARD_SCALE",
            fallback_key="DISTANCE_REWARD_SCALE",
            default=1.0,
        )

        if str(mode) == "ordered":
            if (
                not goal_switched
                and self._previous_distance is not None
                and current_distance is not None
            ):
                reward += float(self._previous_distance - current_distance) * float(distance_scale)

            inactive_goal_penalty = float(
                getattr(self._config, "ORDERED_INACTIVE_GOAL_PENALTY", 0.0)
            )
            if inactive_goal_penalty > 0.0:
                reward -= inactive_goal_penalty * float(_ordered_inactive_goal_hits(task, episode))
        else:
            unordered_dense_reward_mode = self._unordered_dense_reward_mode()
            if unordered_dense_reward_mode in {"global_path_reduction", "global_tsp_reduction", "tsp"}:
                if (
                    self._previous_remaining_path_length is not None
                    and current_remaining_path_length is not None
                ):
                    reward += (
                        float(self._previous_remaining_path_length)
                        - float(current_remaining_path_length)
                    ) * float(distance_scale)
            else:
                if (
                    not goal_switched
                    and self._previous_distance is not None
                    and current_distance is not None
                ):
                    reward += float(self._previous_distance - current_distance) * float(distance_scale)

        if goals_found_delta > 0:
            reward += self._milestone_reward(
                mode,
                previous_goals_found=self._previous_goals_found,
                goals_found_delta=goals_found_delta,
            )

        self._metric = float(reward)
        self._previous_distance = current_distance
        self._previous_remaining_path_length = current_remaining_path_length
        self._previous_active_goal_index = current_active_goal_index
        self._previous_goals_found = int(current_goals_found)


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


_OMNI_LONG_CLIP_CACHE: Dict[str, Any] = {}


def _load_omni_long_clip(model_name: str):
    cache_key = str(model_name)
    if cache_key not in _OMNI_LONG_CLIP_CACHE:
        model, preprocess = clip.load(model_name, device="cpu", jit=False)
        model.eval()
        _OMNI_LONG_CLIP_CACHE[cache_key] = (model, preprocess)
    return _OMNI_LONG_CLIP_CACHE[cache_key]


def _goal_modality_image_index(modality: str) -> int:
    token = str(modality).strip().lower()
    if not token.startswith("image"):
        return 0
    if "_" not in token:
        return 0
    suffix = token.split("_")[-1]
    if suffix.isdigit():
        return int(suffix)
    return 0


@registry.register_sensor
class OmniLongGoalSensor(Sensor):
    cls_uuid: str = "omni_long_goal"

    def __init__(self, *args: Any, sim: Simulator, config: Config, dataset=None, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._dataset = dataset
        self._clip_model_name = str(getattr(config, "CLIP_MODEL", "RN50"))
        self._max_goals = max(1, int(getattr(config, "MAX_GOALS", 5)))
        self._model, self._preprocess = _load_omni_long_clip(self._clip_model_name)
        self._embedding_dim = int(getattr(self._model.visual, "output_dim", 1024))
        self._text_cache: Dict[str, np.ndarray] = {}
        self._rendered_image_cache: Dict[str, np.ndarray] = {}
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._max_goals, self._embedding_dim),
            dtype=np.float32,
        )

    def _encode_text(self, text: str) -> np.ndarray:
        key = str(text).strip()
        if not key:
            return np.zeros((self._embedding_dim,), dtype=np.float32)
        if key not in self._text_cache:
            with torch.no_grad():
                tokens = clip.tokenize([key], truncate=True)
                embedding = self._model.encode_text(tokens).float()
                embedding = embedding / embedding.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            self._text_cache[key] = embedding[0].cpu().numpy().astype(np.float32)
        return self._text_cache[key]

    def _encode_rendered_rgb(self, rgb: np.ndarray) -> np.ndarray:
        image = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
        with torch.no_grad():
            image_tensor = self._preprocess(image).unsqueeze(0)
            embedding = self._model.encode_image(image_tensor).float()
            embedding = embedding / embedding.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return embedding[0].cpu().numpy().astype(np.float32)

    def _render_reference_rgb(
        self,
        instance_record: Dict[str, Any],
        image_index: int,
    ) -> Optional[np.ndarray]:
        image_payload = instance_record.get("image")
        if not isinstance(image_payload, dict):
            return None
        render_views = image_payload.get("render_views")
        if not isinstance(render_views, list) or len(render_views) == 0:
            return None

        if image_index < 0 or image_index >= len(render_views):
            image_index = 0

        render_view = render_views[image_index]
        if not isinstance(render_view, dict):
            return None

        position = render_view.get("agent_base_position")
        if not _is_vec3(position):
            position = render_view.get("position")
        rotation = render_view.get("rotation")
        if not _is_vec3(position) or not _is_quat4(rotation):
            return None

        try:
            observations = self._sim.get_observations_at(
                position=[float(v) for v in position],
                rotation=[float(v) for v in rotation],
                keep_agent_at_new_pose=False,
            )
        except Exception:
            return None

        if observations is None:
            return None
        return _extract_rgb_from_observations(observations)

    def _render_image_embedding(
        self,
        episode: OmniLongEpisode,
        instance_key: str,
        instance_record: Dict[str, Any],
        image_index: int,
    ) -> Optional[np.ndarray]:
        cache_key = "{scene}|{instance}|{index}".format(
            scene=str(getattr(episode, "scene_id", "")),
            instance=str(instance_key),
            index=int(max(0, image_index)),
        )
        if cache_key not in self._rendered_image_cache:
            rgb = self._render_reference_rgb(instance_record, image_index)
            if rgb is None:
                return None
            self._rendered_image_cache[cache_key] = self._encode_rendered_rgb(rgb)
        return self._rendered_image_cache[cache_key]

    def _lookup_goal_record(
        self,
        episode: OmniLongEpisode,
        instance_key: str,
    ) -> Optional[Dict[str, Any]]:
        scene_instances = getattr(self._dataset, "instances_by_scene", None)
        if not isinstance(scene_instances, dict):
            return None
        records = scene_instances.get(str(getattr(episode, "scene_id", "")))
        if not isinstance(records, dict):
            return None
        record = records.get(str(instance_key))
        if not isinstance(record, dict):
            return None
        return record

    def _goal_embedding_from_spec(
        self,
        episode: OmniLongEpisode,
        instance_key: str,
        modality: str,
    ) -> np.ndarray:
        record = self._lookup_goal_record(episode, instance_key)
        if not isinstance(record, dict):
            return np.zeros((self._embedding_dim,), dtype=np.float32)

        category = str(record.get("category", "object"))
        description = str(record.get("description", "")).strip()

        if modality.startswith("image"):
            image_idx = _goal_modality_image_index(modality)
            image_embedding = self._render_image_embedding(
                episode,
                instance_key,
                record,
                image_idx,
            )
            if image_embedding is not None:
                return image_embedding
            return np.zeros((self._embedding_dim,), dtype=np.float32)

        if modality == "description" and description:
            return self._encode_text(description)

        if description and modality.startswith("text"):
            return self._encode_text(description)

        return self._encode_text(category)

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: OmniLongEpisode,
        task: EmbodiedTask,
        **kwargs: Any,
    ) -> np.ndarray:
        task_specs = list(getattr(episode, "tasks", []) or [])
        if len(task_specs) > self._max_goals:
            raise RuntimeError(
                "OmniLongGoalSensor received more goals than configured capacity: "
                f"num_goals={len(task_specs)}, max_goals={self._max_goals}."
            )

        goal_embeddings = np.zeros(
            (self._max_goals, self._embedding_dim),
            dtype=np.float32,
        )
        for goal_index, task_spec in enumerate(task_specs):
            if not isinstance(task_spec, (list, tuple)) or len(task_spec) < 2:
                continue
            instance_key = str(task_spec[0])
            modality = str(task_spec[1])
            goal_embeddings[goal_index] = self._goal_embedding_from_spec(
                episode,
                instance_key,
                modality,
            )
        return goal_embeddings
