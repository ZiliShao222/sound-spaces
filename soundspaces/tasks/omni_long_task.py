#!/usr/bin/env python3

from __future__ import annotations

import os
from itertools import permutations
from typing import Any, Dict, List, Optional, Sequence, Type

import attr
import numpy as np
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
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return None


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

    distance_value = _optional_float(distance)
    if distance_value is None:
        distance_value = 0.0

    if distance_value > 1e-6:
        return distance_value

    info = getattr(episode, "info", None)
    if isinstance(info, dict):
        fallback = _optional_float(info.get("geodesic_distance", 0.0))
        if fallback is not None and fallback > 1e-6:
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
    if hasattr(task, "_navigation_goal_state"):
        goal_state = task._navigation_goal_state()
    else:
        goal_state = task._ordered_active_goal_state()
    if not isinstance(goal_state, dict):
        return None
    goal_index = goal_state.get("goal_index")
    if goal_index is None:
        return None
    return int(goal_index)


def _finite_distance(value: Any) -> Optional[float]:
    distance = _optional_float(value)
    if distance is None:
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
        self._last_action_feedback: Dict[str, Any] = {}
        self._refresh_episode_goals(episode, apply_to_sim=False)
        self._set_last_action_feedback(action_name=None, matched_goal_state=None)

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

    @property
    def remaining_submit_count(self) -> int:
        return max(0, int(self._submit_limit) - int(self._submit_count))

    def get_last_action_feedback(self) -> Dict[str, Any]:
        return dict(getattr(self, "_last_action_feedback", {}))

    def _goal_descriptor_payload(self, descriptor: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if isinstance(descriptor, (list, tuple)) and len(descriptor) >= 2:
            payload["instance_key"] = str(descriptor[0])
            payload["modality"] = str(descriptor[1])
        elif descriptor is not None:
            payload["descriptor"] = str(descriptor)
        return payload

    def _goal_state_payload(self, goal_state: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(goal_state, dict):
            return None

        goal = goal_state.get("goal")
        payload: Dict[str, Any] = {
            "goal_index": int(goal_state.get("goal_index", -1)),
            "goal_key": "goal_{:03d}".format(int(goal_state.get("goal_index", -1))),
        }
        payload.update(self._goal_descriptor_payload(goal_state.get("descriptor")))

        if goal is not None:
            object_category = getattr(goal, "object_category", None)
            object_name = getattr(goal, "object_name", None)
            object_id = getattr(goal, "object_id", None)
            position = getattr(goal, "position", None)
            room_name = getattr(goal, "room_name", None)
            if object_category is not None:
                payload["object_category"] = str(object_category)
            if object_name is not None:
                payload["object_name"] = str(object_name)
            if object_id is not None:
                payload["object_id"] = str(object_id)
            if room_name is not None:
                payload["room_name"] = str(room_name)
            if _is_vec3(position):
                payload["position"] = [float(v) for v in position]

        return payload

    def _set_last_action_feedback(
        self,
        action_name: Optional[str],
        matched_goal_state: Optional[Dict[str, Any]],
    ) -> None:
        goal_payload = self._goal_state_payload(matched_goal_state)
        self._last_action_feedback = {
            "action_name": action_name,
            "found_goal_this_step": bool(goal_payload is not None),
            "found_goal_index": int(goal_payload.get("goal_index", -1)) if goal_payload else -1,
            "remaining_submit_count": int(self.remaining_submit_count),
            "submit_count": int(self._submit_count),
            "submit_limit": int(self._submit_limit),
            "goal": goal_payload,
        }

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
        self._set_last_action_feedback(action_name=None, matched_goal_state=None)
        if getattr(self, "is_stop_called", False):
            self.is_stop_called = False  # type: ignore
            matched_goal_state = self._handle_stop(episode)
            self._set_last_action_feedback(action_name="STOP", matched_goal_state=matched_goal_state)
            return False
        if getattr(self, "is_submit_called", False):
            self.is_submit_called = False  # type: ignore
            if self._submit_count >= self._submit_limit:
                matched_goal_state = self._handle_stop(episode)
                self._set_last_action_feedback(
                    action_name="LIFELONG_SUBMIT",
                    matched_goal_state=matched_goal_state,
                )
                return False
            self._submit_count += 1
            is_active, matched_goal_state = self._handle_submit(episode)
            self._set_last_action_feedback(
                action_name="LIFELONG_SUBMIT",
                matched_goal_state=matched_goal_state,
            )
            return is_active
        return True

    def _handle_submit(self, episode: Episode) -> Any:
        goal_state = self._match_goal_state(episode)
        if goal_state is None:
            return True, None
        self._mark_goal_found(goal_state)
        if all(bool(state["found"]) for state in self._goals_map.values()):
            return False, goal_state
        self._refresh_episode_goals(episode, apply_to_sim=True)
        return True, goal_state

    def _handle_stop(self, episode: Episode):
        goal_state = self._match_goal_state(episode)
        if goal_state is not None:
            self._mark_goal_found(goal_state)
        return goal_state

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
            self._sim.set_active_sound_index(goal_index)
            return

        goal = self._all_goals[goal_index]
        sound_id = None
        sound_sources = getattr(episode, "sound_sources", None)
        if isinstance(sound_sources, list) and goal_index < len(sound_sources):
            candidate = sound_sources[goal_index]
            if isinstance(candidate, dict):
                sound_id = candidate.get("sound_id")
        if hasattr(self._sim, "set_active_goal"):
            self._sim.set_active_goal(goal.position, sound_id=sound_id)


@registry.register_measure
class OmniLongDistanceToGoal(Measure):
    cls_uuid: str = DistanceToGoal.cls_uuid

    def __init__(self, *args: Any, sim: Simulator, config: Config, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._previous_position: Optional[Sequence[float]] = None
        self._previous_state_signature: Optional[tuple] = None
        self._metric: float = 0.0
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _state_signature(self, task: EmbodiedTask, episode: Episode) -> tuple:
        mode = getattr(task, "goal_order_mode", _resolve_goal_order_mode(episode, task=task))
        if str(mode).strip().lower() == "ordered":
            return (str(mode), _task_active_goal_index(task), _task_goals_found(task))

        remaining_goal_indices = tuple(
            sorted(
                int(goal_state.get("goal_index", -1))
                for goal_state in getattr(task, "_goals_map", {}).values()
                if not bool(goal_state.get("found", False))
            )
        )
        return (str(mode), remaining_goal_indices)

    def _compute_distance(self, task: EmbodiedTask, episode: Episode) -> float:
        mode = getattr(task, "goal_order_mode", _resolve_goal_order_mode(episode, task=task))
        if str(mode).strip().lower() == "unordered":
            distance = _shortest_remaining_path_length(task, episode)
        else:
            distance = task._distance_to_goal_by_mode(episode)

        distance_value = _finite_distance(distance)
        if distance_value is not None:
            return float(max(0.0, distance_value))

        previous_value = _finite_distance(self._metric)
        if previous_value is not None:
            return float(max(0.0, previous_value))
        return 0.0

    def reset_metric(self, *args: Any, episode: Episode, task: EmbodiedTask, **kwargs: Any):
        self._previous_position = None
        self._previous_state_signature = None
        self._metric = 0.0
        self.update_metric(episode=episode, task=task)

    def update_metric(self, *args: Any, episode: Episode, task: EmbodiedTask, **kwargs: Any):
        current_position = np.array(self._sim.get_agent_state().position, dtype=np.float32)
        state_signature = self._state_signature(task, episode)

        if (
            self._previous_position is None
            or not np.allclose(self._previous_position, current_position, atol=1e-4)
            or self._previous_state_signature != state_signature
        ):
            self._metric = self._compute_distance(task, episode)
            self._previous_position = (
                float(current_position[0]),
                float(current_position[1]),
                float(current_position[2]),
            )
            self._previous_state_signature = state_signature


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
        self._previous_active_goal_index: Optional[int] = None
        self._previous_goals_found: int = 0
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "omni_long_distance_to_goal_reward"

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

        if (
            (str(mode) != "ordered" or not goal_switched)
            and self._previous_distance is not None
            and current_distance is not None
        ):
            reward += float(self._previous_distance - current_distance) * float(distance_scale)

        self._metric = float(reward)
        self._previous_distance = current_distance
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


def _goal_text_from_record(
    record: Dict[str, Any],
    modality: str,
    category_prompt_template: str,
) -> str:
    category = str(record.get("category", "object")).strip() or "object"
    description = str(record.get("description", "")).strip()
    modality_token = str(modality).strip().lower()

    if modality_token == "description" and description:
        return description
    if description and modality_token.startswith("text"):
        return description
    return str(category_prompt_template).format(category=category)


def _goal_modality_mask(modality: str) -> np.ndarray:
    modality_token = str(modality).strip().lower()
    if modality_token.startswith("image"):
        return np.asarray([0.0, 1.0], dtype=np.float32)
    return np.asarray([1.0, 0.0], dtype=np.float32)


def _resize_goal_rgb(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    resized = Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode="RGB")
    resized = resized.resize((int(width), int(height)), Image.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


class _OmniLongRawGoalSensorBase(Sensor):
    def __init__(self, *args: Any, sim: Simulator, config: Config, dataset=None, **kwargs: Any):
        self._sim = sim
        self._config = config
        self._dataset = dataset
        self._max_goals = max(1, int(getattr(config, "MAX_GOALS", 5)))
        self._category_prompt_template = str(
            getattr(config, "CATEGORY_PROMPT_TEMPLATE", "Find the {category}.")
        )
        self._image_width = max(1, int(getattr(config, "WIDTH", 224)))
        self._image_height = max(1, int(getattr(config, "HEIGHT", 224)))
        self._max_text_bytes = max(
            1,
            int(getattr(config, "MAX_TEXT_BYTES", getattr(config, "CONTEXT_LENGTH", 77))),
        )
        self._rendered_rgb_cache: Dict[str, np.ndarray] = {}
        self._text_bytes_cache: Dict[str, np.ndarray] = {}
        super().__init__(config=config)

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

        observations = self._sim.get_observations_at(
            position=[float(v) for v in position],
            rotation=[float(v) for v in rotation],
            keep_agent_at_new_pose=False,
        )

        if observations is None:
            return None
        rgb = _extract_rgb_from_observations(observations)
        if rgb is None:
            return None
        return _resize_goal_rgb(rgb, self._image_width, self._image_height)

    def _render_goal_rgb(
        self,
        episode: OmniLongEpisode,
        instance_key: str,
        instance_record: Dict[str, Any],
        image_index: int,
    ) -> Optional[np.ndarray]:
        cache_key = "{scene}|{instance}|{index}|{width}x{height}".format(
            scene=str(getattr(episode, "scene_id", "")),
            instance=str(instance_key),
            index=int(max(0, image_index)),
            width=int(self._image_width),
            height=int(self._image_height),
        )
        if cache_key not in self._rendered_rgb_cache:
            rgb = self._render_reference_rgb(instance_record, image_index)
            if rgb is None:
                return None
            self._rendered_rgb_cache[cache_key] = rgb
        return self._rendered_rgb_cache[cache_key]

    def _encode_goal_text_bytes(self, text: str) -> np.ndarray:
        key = str(text).strip()
        if not key:
            return np.zeros((self._max_text_bytes,), dtype=np.uint8)
        if key not in self._text_bytes_cache:
            encoded = key.encode("utf-8", errors="ignore")[: self._max_text_bytes]
            buffer = np.zeros((self._max_text_bytes,), dtype=np.uint8)
            if len(encoded) > 0:
                buffer[: len(encoded)] = np.frombuffer(encoded, dtype=np.uint8)
            self._text_bytes_cache[key] = buffer
        return self._text_bytes_cache[key]


@registry.register_sensor
class OmniLongGoalImageSensor(_OmniLongRawGoalSensorBase):
    cls_uuid: str = "omni_long_goal_image"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.COLOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self._max_goals, self._image_height, self._image_width, 3),
            dtype=np.uint8,
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: OmniLongEpisode,
        **kwargs: Any,
    ) -> np.ndarray:
        goal_images = np.zeros(
            (self._max_goals, self._image_height, self._image_width, 3),
            dtype=np.uint8,
        )
        for goal_index, task_spec in enumerate(list(getattr(episode, "tasks", []) or [])[: self._max_goals]):
            if not isinstance(task_spec, (list, tuple)) or len(task_spec) < 2:
                continue
            instance_key = str(task_spec[0])
            modality = str(task_spec[1])
            if not modality.strip().lower().startswith("image"):
                continue
            record = self._lookup_goal_record(episode, instance_key)
            if not isinstance(record, dict):
                continue
            rgb = self._render_goal_rgb(
                episode,
                instance_key,
                record,
                _goal_modality_image_index(modality),
            )
            if rgb is not None:
                goal_images[goal_index] = rgb
        return goal_images


@registry.register_sensor
class OmniLongGoalTextSensor(_OmniLongRawGoalSensorBase):
    cls_uuid: str = "omni_long_goal_text"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TEXT

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=255,
            shape=(self._max_goals, self._max_text_bytes),
            dtype=np.uint8,
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: OmniLongEpisode,
        **kwargs: Any,
    ) -> np.ndarray:
        text_bytes = np.zeros((self._max_goals, self._max_text_bytes), dtype=np.uint8)
        for goal_index, task_spec in enumerate(list(getattr(episode, "tasks", []) or [])[: self._max_goals]):
            if not isinstance(task_spec, (list, tuple)) or len(task_spec) < 2:
                continue
            instance_key = str(task_spec[0])
            modality = str(task_spec[1])
            record = self._lookup_goal_record(episode, instance_key)
            if not isinstance(record, dict):
                continue
            text = _goal_text_from_record(record, modality, self._category_prompt_template)
            text_bytes[goal_index] = self._encode_goal_text_bytes(text)
        return text_bytes


@registry.register_sensor
class OmniLongGoalModalitySensor(_OmniLongRawGoalSensorBase):
    cls_uuid: str = "omni_long_goal_modality"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._max_goals, 2),
            dtype=np.float32,
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: OmniLongEpisode,
        **kwargs: Any,
    ) -> np.ndarray:
        modality_mask = np.zeros((self._max_goals, 2), dtype=np.float32)
        for goal_index, task_spec in enumerate(list(getattr(episode, "tasks", []) or [])[: self._max_goals]):
            if not isinstance(task_spec, (list, tuple)) or len(task_spec) < 2:
                continue
            modality_mask[goal_index] = _goal_modality_mask(str(task_spec[1]))
        return modality_mask


@registry.register_sensor
class OmniLongGoalMaskSensor(_OmniLongRawGoalSensorBase):
    cls_uuid: str = "goal_mask"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0.0, high=1.0, shape=(self._max_goals,), dtype=np.float32)

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: OmniLongEpisode,
        task: Optional[EmbodiedTask] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        goal_mask = np.zeros((self._max_goals,), dtype=np.float32)
        goal_states = list(getattr(task, "_goals_map", {}).values()) if task is not None else []
        if goal_states:
            mode = str(getattr(task, "goal_order_mode", _resolve_goal_order_mode(episode, task=task)))
            if mode == "ordered":
                active_goal_state = None
                if hasattr(task, "_ordered_active_goal_state"):
                    active_goal_state = task._ordered_active_goal_state()
                if isinstance(active_goal_state, dict):
                    goal_index = int(active_goal_state.get("goal_index", -1))
                    if 0 <= goal_index < self._max_goals:
                        goal_mask[goal_index] = 1.0
                return goal_mask

            for goal_state in goal_states:
                goal_index = int(goal_state.get("goal_index", -1))
                if 0 <= goal_index < self._max_goals and not bool(goal_state.get("found", False)):
                    goal_mask[goal_index] = 1.0
            return goal_mask

        for goal_index, _ in enumerate(list(getattr(episode, "tasks", []) or [])[: self._max_goals]):
            goal_mask[goal_index] = 1.0
        return goal_mask


@registry.register_sensor
class OmniLongTaskModeSensor(_OmniLongRawGoalSensorBase):
    cls_uuid: str = "task_mode_flag"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TENSOR

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: OmniLongEpisode,
        task: Optional[EmbodiedTask] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        mode = str(getattr(task, "goal_order_mode", _resolve_goal_order_mode(episode, task=task)))
        return np.asarray([1.0 if mode == "unordered" else 0.0], dtype=np.float32)
