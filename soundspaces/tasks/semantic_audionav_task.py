#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, List, Optional, Type

import attr
import numpy as np
from gym import spaces

from habitat.config import Config
from habitat.core.dataset import Dataset, Episode
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, Sensor, SensorTypes, Simulator
from habitat.core.embodied_task import SimulatorTaskAction
from habitat.tasks.nav.nav import Measure, EmbodiedTask, Success
from habitat.core.utils import not_none_validator
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    NavigationTask,
)


@attr.s(auto_attribs=True, kw_only=True)
class SemanticAudioGoalNavEpisode(NavigationEpisode):
    r"""ObjectGoal Navigation Episode

    :param object_category: Category of the object
    """
    object_category: str
    sound_id: str
    sound_sources: Optional[List[dict]] = None
    sound_source_schedule: Optional[list] = None
    distractor_sound_id: str = None
    distractor_position_index: attr.ib(converter=int) = None
    offset: attr.ib(converter=int)
    duration: attr.ib(converter=int)
    num_goals: Optional[int] = attr.ib(
        default=None,
        converter=lambda x: int(x) if x is not None else None,
    )

    @property
    def goals_key(self) -> str:
        r"""The key to retrieve the goals
        """
        return f"{os.path.basename(self.scene_id)}_{self.object_category}"


@attr.s(auto_attribs=True)
class ObjectViewLocation:
    r"""ObjectViewLocation provides information about a position around an object goal
    usually that is navigable and the object is visible with specific agent
    configuration that episode's dataset was created.
     that is target for
    navigation. That can be specify object_id, position and object
    category. An important part for metrics calculation are view points that
     describe success area for the navigation.

    Args:
        agent_state: navigable AgentState with a position and a rotation where
        the object is visible.
        iou: an intersection of a union of the object and a rectangle in the
        center of view. This metric is used to evaluate how good is the object
        view form current position. Higher iou means better view, iou equals
        1.0 if whole object is inside of the rectangle and no pixel inside
        the rectangle belongs to anything except the object.
    """
    agent_state: AgentState
    iou: Optional[float]


@attr.s(auto_attribs=True, kw_only=True)
class SemanticAudioGoal(NavigationGoal):
    r"""Object goal provides information about an object that is target for
    navigation. That can be specify object_id, position and object
    category. An important part for metrics calculation are view points that
     describe success area for the navigation.

    Args:
        object_id: id that can be used to retrieve object from the semantic
        scene annotation
        object_id_raw: original object id from semantic scene (optional)
        object_name: name of the object
        object_category: object category name usually similar to scene semantic
        categories
        room_id: id of a room where object is located, can be used to retrieve
        room from the semantic scene annotation
        room_name: name of the room, where object is located
        view_points: navigable positions around the object with specified
        proximity of the object surface used for navigation metrics calculation.
        The object is visible from these positions.
    """

    object_id: str = attr.ib(default=None, validator=not_none_validator)
    object_id_raw: Optional[str] = None
    object_name: Optional[str] = None
    object_category: Optional[str] = None
    room_id: Optional[str] = None
    room_name: Optional[str] = None
    view_points: Optional[List[ObjectViewLocation]] = None


@registry.register_sensor
class SemanticAudioGoalSensor(Sensor):
    r"""A sensor for Object Goal specification as observations which is used in
    ObjectGoal Navigation. The goal is expected to be specified by object_id or
    semantic category id.
    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalSensor sensor. Can contain field
            GOAL_SPEC that specifies which id use for goal specification,
            GOAL_SPEC_MAX_VAL the maximum object_id possible used for
            observation space definition.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """
    cls_uuid: str = "objectgoal"

    def __init__(
        self, sim, config: Config, dataset: Dataset, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._dataset = dataset
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (1,)
        max_value = (self.config.GOAL_SPEC_MAX_VAL - 1,)
        if self.config.GOAL_SPEC == "TASK_CATEGORY_ID":
            max_value = max(
                self._dataset.category_to_task_category_id.values()
            )

        return spaces.Box(
            low=0, high=max_value, shape=sensor_shape, dtype=np.int64
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: SemanticAudioGoalNavEpisode,
        **kwargs: Any,
    ) -> Optional[int]:
        if self.config.GOAL_SPEC == "TASK_CATEGORY_ID":
            if len(episode.goals) == 0:
                logger.error(
                    f"No goal specified for episode {episode.episode_id}."
                )
                return None
            if not isinstance(episode.goals[0], SemanticAudioGoal):
                logger.error(
                    f"First goal should be ObjectGoal, episode {episode.episode_id}."
                )
                return None
            category_name = episode.object_category
            return np.array(
                [self._dataset.category_to_task_category_id[category_name]],
                dtype=np.int64,
            )
        elif self.config.GOAL_SPEC == "OBJECT_ID":
            return np.array([episode.goals[0].object_name_id], dtype=np.int64)
        else:
            raise RuntimeError(
                "Wrong GOAL_SPEC specified for ObjectGoalSensor."
            )


@registry.register_task(name="SemanticAudioNav")
class SemanticAudioNavigationTask(NavigationTask):
    r"""An Object Navigation Task class for a task specific methods.
        Used to explicitly state a type of the task in config.
    """

    def overwrite_sim_config(
            self, sim_config: Any, episode: Type[Episode]
    ) -> Any:
        return merge_sim_episode_config(sim_config, episode)

    def reset(self, episode: Episode):
        self._all_goals = list(episode.goals) if episode.goals else []
        self._goal_index = 0
        self._found_goal_indices = set()
        if self._all_goals:
            episode.goals = [self._all_goals[0]]
        return super().reset(episode)

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
            return self._handle_submit(episode)
        return True

    def _handle_submit(self, episode: Episode) -> bool:
        if not self._all_goals:
            return True
        goal = self._all_goals[self._goal_index]
        success_distance = getattr(self._config.SUCCESS, "SUCCESS_DISTANCE", 1.0)
        distance = self._distance_to_goal(goal, episode)
        if distance is None or distance >= success_distance:
            return True
        self._found_goal_indices.add(self._goal_index)
        if self._goal_index + 1 >= len(self._all_goals):
            return False
        self._goal_index += 1
        episode.goals = [self._all_goals[self._goal_index]]
        self._apply_goal_to_sim(episode, self._goal_index)
        return True

    def _distance_to_goal(self, goal: SemanticAudioGoal, episode: Episode) -> Optional[float]:
        current_position = self._sim.get_agent_state().position
        distance_to_cfg = getattr(self._config, "DISTANCE_TO_GOAL", None)
        distance_to = getattr(distance_to_cfg, "DISTANCE_TO", "POINT")
        if distance_to == "VIEW_POINTS" and goal.view_points:
            targets = [vp.agent_state.position for vp in goal.view_points]
        else:
            targets = [goal.position]
        return self._sim.geodesic_distance(current_position, targets, episode)

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
        if sound_sources and goal_index < len(sound_sources):
            sound_id = sound_sources[goal_index].get("sound_id")
        if hasattr(self._sim, "set_active_goal"):
            try:
                self._sim.set_active_goal(goal.position, sound_id=sound_id)
            except Exception:
                pass


def merge_sim_episode_config(
        sim_config: Config, episode: Type[Episode]
) -> Any:
    sim_config.defrost()
    # here's where the scene update happens, extract the scene name out of the path
    sim_config.SCENE = episode.scene_id
    sim_config.freeze()
    if (
            episode.start_position is not None
            and episode.start_rotation is not None
    ):
        agent_name = sim_config.AGENTS[sim_config.DEFAULT_AGENT_ID]
        agent_cfg = getattr(sim_config, agent_name)
        agent_cfg.defrost()
        agent_cfg.START_POSITION = episode.start_position
        agent_cfg.START_ROTATION = episode.start_rotation
        goal_positions = [g.position for g in episode.goals]
        agent_cfg.GOAL_POSITION = goal_positions[0]
        agent_cfg.GOAL_POSITIONS = goal_positions
        if hasattr(episode, "sound_sources") and episode.sound_sources:
            sound_sources = []
            for idx, src in enumerate(episode.sound_sources):
                src_dict = dict(src)
                if "position" not in src_dict:
                    pos = (
                        goal_positions[idx]
                        if idx < len(goal_positions)
                        else goal_positions[-1]
                    )
                    src_dict["position"] = pos
                sound_sources.append(src_dict)
            agent_cfg.SOUND_SOURCES = sound_sources
            agent_cfg.SOUND_ID = sound_sources[0]["sound_id"]
        else:
            agent_cfg.SOUND_ID = episode.sound_id
        if hasattr(episode, "sound_source_schedule") and episode.sound_source_schedule:
            agent_cfg.SOUND_SOURCE_SCHEDULE = episode.sound_source_schedule
        agent_cfg.DISTRACTOR_SOUND_ID = episode.distractor_sound_id
        agent_cfg.DISTRACTOR_POSITION_INDEX = episode.distractor_position_index
        agent_cfg.OFFSET = episode.offset
        agent_cfg.DURATION = episode.duration
        agent_cfg.IS_SET_START_STATE = True
        agent_cfg.freeze()
    return sim_config


@registry.register_measure
class SWS(Measure):
    r"""Success when silent
    """
    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "sws"

    def reset_metric(self, *args: Any, episode, **kwargs: Any):
        self._metric = None

    def update_metric(
        self, *args: Any, episode, action, task: EmbodiedTask, **kwargs: Any
    ):
        ep_success = task.measurements.measures[Success.cls_uuid].get_metric()
        self._metric = ep_success * self._sim.is_silent


@registry.register_task_action
class SubmitAction(SimulatorTaskAction):
    name: str = "SUBMIT"

    def reset(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        task.is_submit_called = False  # type: ignore

    def step(self, task: EmbodiedTask, *args: Any, **kwargs: Any):
        task.is_submit_called = True  # type: ignore
        return self._sim.get_observations_at()  # type: ignore
