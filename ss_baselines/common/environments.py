#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@baseline_registry.register_env(name="myEnv")` for reusability
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Type
import logging
import math

import habitat
import numpy as np
from habitat import Config, Dataset
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.geometry_utils import quaternion_rotate_vector
from ss_baselines.common.baseline_registry import baseline_registry


def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return baseline_registry.get_env(env_name)


@baseline_registry.register_env(name="AudioNavRLEnv")
class AudioNavRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._continuous = config.CONTINUOUS

        self._previous_target_distance = None
        self._previous_action = None
        self._episode_distance_covered = None
        self._success_distance = self._core_env_config.TASK.SUCCESS.SUCCESS_DISTANCE
        super().__init__(self._core_env_config, dataset)

    def reset(self):
        self._previous_action = None

        observations = super().reset()
        logging.debug(super().current_episode)

        if self._continuous:
            self._previous_target_distance = self._distance_target()
        else:
            self._previous_target_distance = self.habitat_env.current_episode.info[
                "geodesic_distance"
            ]
        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations):
        reward = 0

        if self._rl_config.WITH_TIME_PENALTY:
            reward += self._rl_config.SLACK_REWARD

        if self._rl_config.WITH_DISTANCE_REWARD:
            current_target_distance = self._distance_target()
            reward += (self._previous_target_distance - current_target_distance) * self._rl_config.DISTANCE_REWARD_SCALE
            self._previous_target_distance = current_target_distance

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD
            logging.debug('Reaching goal!')

        assert not math.isnan(reward)

        return reward

    def _distance_target(self):
        return self._env.get_metrics()['distance_to_goal']

    def _episode_success(self):
        if self._env.task.is_stop_called and \
                ((self._continuous and self._distance_target() < self._success_distance) or
                 (not self._continuous and self._env.sim.reaching_goal)):
            return True
        return False

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        info = dict(self.habitat_env.get_metrics())
        task = getattr(self._env, "task", None)
        if task is None:
            return info

        remaining_submit_count = getattr(task, "remaining_submit_count", None)
        if remaining_submit_count is not None:
            info["remaining_submit_count"] = int(remaining_submit_count)

        if hasattr(task, "get_last_action_feedback"):
            feedback = task.get_last_action_feedback()
            if isinstance(feedback, dict):
                info["last_action_feedback"] = feedback
                info["found_goal_this_step"] = int(bool(feedback.get("found_goal_this_step", False)))
                info["found_goal_index_this_step"] = int(feedback.get("found_goal_index", -1))
                if "remaining_submit_count" in feedback:
                    info["remaining_submit_count"] = int(feedback["remaining_submit_count"])
                if "submit_count" in feedback:
                    info["submit_count"] = int(feedback["submit_count"])
                if "submit_limit" in feedback:
                    info["submit_limit"] = int(feedback["submit_limit"])

        return info

    # for data collection
    def get_current_episode_id(self):
        return self.habitat_env.current_episode.episode_id


@baseline_registry.register_env(name="OmniLongRLEnv")
class OmniLongRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._reward_measure_name = getattr(
            self._rl_config, "REWARD_MEASURE", "omni_long_distance_to_goal_reward"
        )
        self._success_measure_name = getattr(
            self._rl_config, "SUCCESS_MEASURE", "lifelong_task_success"
        )
        self._end_on_success = bool(getattr(self._rl_config, "END_ON_SUCCESS", True))
        self._teacher_rng = np.random.default_rng(int(getattr(config, "SEED", 0)))
        self._teacher_goal_signature_state: Optional[Tuple[int, Any]] = None
        self._teacher_target_index: Optional[int] = None
        self._teacher_scan_target_index: Optional[int] = None
        self._teacher_scan_turns_done: int = 0
        self._teacher_blocked_target_indices: Set[int] = set()
        super().__init__(self._core_env_config, dataset)

    def _clear_teacher_goal_state(self) -> None:
        self._teacher_target_index = None
        self._teacher_scan_target_index = None
        self._teacher_scan_turns_done = 0
        self._teacher_blocked_target_indices = set()

    def _teacher_goal_signature(self, goal_state) -> Optional[Tuple[int, Any]]:
        if not isinstance(goal_state, dict):
            return None
        return (
            int(goal_state.get("goal_index", -1)),
            getattr(goal_state.get("goal"), "object_id", None),
        )

    def _teacher_success_distance(self) -> float:
        bc_cfg = getattr(self._rl_config, "BC", None)
        if bc_cfg is not None and getattr(bc_cfg, "expert_success_distance", None) is not None:
            return float(bc_cfg.expert_success_distance)
        success_cfg = getattr(self._core_env_config.TASK, "SUCCESS", None)
        if success_cfg is not None and getattr(success_cfg, "SUCCESS_DISTANCE", None) is not None:
            return float(success_cfg.SUCCESS_DISTANCE)
        return 1.0

    def _task_action_index(self, action_name: str) -> int:
        task = self.habitat_env.task
        for index, candidate_name in enumerate(task.actions.keys()):
            if str(candidate_name).upper() == str(action_name).upper():
                return int(index)
        raise KeyError(f"Unknown task action name: {action_name}")

    @staticmethod
    def _sim_action_name(action: int) -> str:
        if int(action) == int(HabitatSimActions.STOP):
            return "STOP"
        if int(action) == int(HabitatSimActions.MOVE_FORWARD):
            return "MOVE_FORWARD"
        if int(action) == int(HabitatSimActions.TURN_LEFT):
            return "TURN_LEFT"
        if int(action) == int(HabitatSimActions.TURN_RIGHT):
            return "TURN_RIGHT"
        raise ValueError(f"Unsupported simulator action: {action}")

    def _teacher_active_goal_state(self):
        task = self.habitat_env.task
        if not hasattr(task, "_navigation_goal_state"):
            return None
        return task._navigation_goal_state(self.habitat_env.current_episode)

    def _teacher_goal_distance(self, goal_state) -> Optional[float]:
        if not isinstance(goal_state, dict):
            return None
        task = self.habitat_env.task
        goal = goal_state.get("goal")
        if goal is None or not hasattr(task, "_distance_to_goal"):
            return None
        return task._distance_to_goal(goal, self.habitat_env.current_episode)

    def _teacher_goal_visible(self, goal_state) -> bool:
        if not isinstance(goal_state, dict):
            return False
        task = self.habitat_env.task
        if hasattr(task, "_goal_state_visible"):
            try:
                return bool(task._goal_state_visible(goal_state, self.habitat_env.current_episode))
            except Exception:
                return False
        return False

    def _teacher_target_positions(self, goal_state) -> List[np.ndarray]:
        if not isinstance(goal_state, dict):
            return []
        goal = goal_state.get("goal")
        if goal is None:
            return []

        distance_to_cfg = getattr(self._core_env_config.TASK, "DISTANCE_TO_GOAL", None)
        distance_to = str(getattr(distance_to_cfg, "DISTANCE_TO", "POINT"))
        current_position = self.habitat_env.sim.get_agent_state().position

        target_positions = []
        if distance_to == "VIEW_POINTS" and getattr(goal, "view_points", None):
            target_positions = [view.agent_state.position for view in goal.view_points]
        elif getattr(goal, "position", None) is not None:
            target_positions = [goal.position]

        if len(target_positions) == 0:
            return []

        prepared_targets: List[Tuple[float, np.ndarray]] = []
        for target_position in target_positions:
            candidate_target = np.asarray(target_position, dtype=np.float32)
            if hasattr(self.habitat_env.sim, "_snap_to_navmesh"):
                candidate_target = np.asarray(
                    self.habitat_env.sim._snap_to_navmesh(candidate_target),
                    dtype=np.float32,
                )
            elif hasattr(self.habitat_env.sim, "pathfinder") and hasattr(
                self.habitat_env.sim.pathfinder, "snap_point"
            ):
                candidate_target = np.asarray(
                    self.habitat_env.sim.pathfinder.snap_point(candidate_target),
                    dtype=np.float32,
                )
            distance = self.habitat_env.sim.geodesic_distance(
                current_position,
                [candidate_target],
                self.habitat_env.current_episode,
            )
            if distance is None or not np.isfinite(distance):
                distance = float(np.linalg.norm((candidate_target - np.asarray(current_position, dtype=np.float32))[[0, 2]]))
            prepared_targets.append((float(distance), candidate_target))

        if not prepared_targets:
            return []
        prepared_targets.sort(key=lambda item: float(item[0]))
        return [target for _, target in prepared_targets]

    def _teacher_target_distance(self, target_position: np.ndarray) -> Optional[float]:
        current_position = np.asarray(self.habitat_env.sim.get_agent_state().position, dtype=np.float32)
        distance = self.habitat_env.sim.geodesic_distance(
            current_position,
            [np.asarray(target_position, dtype=np.float32)],
            self.habitat_env.current_episode,
        )
        if distance is not None and np.isfinite(distance):
            return float(distance)
        planar = np.asarray(target_position, dtype=np.float32) - current_position
        planar[1] = 0.0
        return float(np.linalg.norm(planar[[0, 2]]))

    def _teacher_at_target_position(self, target_position: np.ndarray) -> bool:
        target_distance = self._teacher_target_distance(target_position)
        if target_distance is None:
            return False
        tolerance = max(
            0.1,
            float(getattr(self._core_env_config.SIMULATOR, "FORWARD_STEP_SIZE", 0.25)),
        )
        return float(target_distance) <= float(tolerance)

    def _teacher_select_target_index(self, target_positions: List[np.ndarray]) -> Optional[int]:
        if len(target_positions) == 0:
            return None
        if (
            self._teacher_target_index is not None
            and self._teacher_target_index not in self._teacher_blocked_target_indices
            and 0 <= int(self._teacher_target_index) < len(target_positions)
        ):
            return int(self._teacher_target_index)

        candidates = [
            index for index in range(len(target_positions)) if index not in self._teacher_blocked_target_indices
        ]
        if not candidates:
            return None
        selected_index = int(self._teacher_rng.choice(candidates))
        self._teacher_target_index = selected_index
        return selected_index

    def _teacher_start_scan(self, target_index: int) -> int:
        self._teacher_scan_target_index = int(target_index)
        self._teacher_scan_turns_done = 1
        self._teacher_target_index = int(target_index)
        return self._task_action_index("TURN_LEFT")

    def _teacher_continue_scan(self) -> Optional[int]:
        if self._teacher_scan_target_index is None:
            return None

        max_turns = max(
            1,
            int(getattr(getattr(self._rl_config, "BC", None), "teacher_visibility_scan_turns", 12)),
        )
        if int(self._teacher_scan_turns_done) >= max_turns:
            self._teacher_blocked_target_indices.add(int(self._teacher_scan_target_index))
            if self._teacher_target_index == int(self._teacher_scan_target_index):
                self._teacher_target_index = None
            self._teacher_scan_target_index = None
            self._teacher_scan_turns_done = 0
            return None

        self._teacher_scan_turns_done += 1
        return self._task_action_index("TURN_LEFT")

    def _teacher_next_waypoint(self, target_position: np.ndarray) -> np.ndarray:
        current_position = np.asarray(
            self.habitat_env.sim.get_agent_state().position,
            dtype=np.float32,
        )
        if not hasattr(self.habitat_env.sim, "get_straight_shortest_path_points"):
            return np.asarray(target_position, dtype=np.float32)

        path_points = self.habitat_env.sim.get_straight_shortest_path_points(
            current_position,
            target_position,
        )
        if len(path_points) == 0:
            return np.asarray(target_position, dtype=np.float32)

        forward_step_size = float(
            getattr(self._core_env_config.SIMULATOR, "FORWARD_STEP_SIZE", 0.25)
        )
        lookahead_distance = max(2.0 * forward_step_size, 0.5)
        previous_point = current_position
        traversed_distance = 0.0

        for path_point in path_points:
            candidate = np.asarray(path_point, dtype=np.float32)
            segment = candidate - previous_point
            segment[1] = 0.0
            segment_length = float(np.linalg.norm(segment[[0, 2]]))
            if segment_length <= 1e-3:
                previous_point = candidate
                continue

            traversed_distance += segment_length
            previous_point = candidate
            if traversed_distance >= lookahead_distance:
                return candidate

        return np.asarray(path_points[-1], dtype=np.float32)

    def _teacher_heading_sim_action(self, target_position: np.ndarray) -> int:
        state = self.habitat_env.sim.get_agent_state()
        current_position = np.asarray(state.position, dtype=np.float32)
        target_vector = np.asarray(target_position, dtype=np.float32) - current_position
        target_vector[1] = 0.0
        target_norm = float(np.linalg.norm(target_vector[[0, 2]]))
        if target_norm <= 1e-6:
            return int(HabitatSimActions.MOVE_FORWARD)

        target_vector = target_vector / max(target_norm, 1e-6)
        forward_vector = np.asarray(
            quaternion_rotate_vector(
                state.rotation,
                np.array([0.0, 0.0, -1.0], dtype=np.float32),
            ),
            dtype=np.float32,
        )
        forward_vector[1] = 0.0
        forward_norm = float(np.linalg.norm(forward_vector[[0, 2]]))
        if forward_norm <= 1e-6:
            return int(HabitatSimActions.MOVE_FORWARD)

        forward_vector = forward_vector / max(forward_norm, 1e-6)
        cross = float(forward_vector[0] * target_vector[2] - forward_vector[2] * target_vector[0])
        dot = float(forward_vector[0] * target_vector[0] + forward_vector[2] * target_vector[2])
        angle = float(np.arctan2(cross, dot))
        forward_step_size = float(
            getattr(self._core_env_config.SIMULATOR, "FORWARD_STEP_SIZE", 0.25)
        )
        turn_angle = float(getattr(self._core_env_config.SIMULATOR, "TURN_ANGLE", 30.0))
        lateral_offset = abs(cross) * target_norm
        move_angle_threshold = np.deg2rad(max(10.0, min(45.0, turn_angle)))
        lateral_threshold = max(0.5 * forward_step_size, 0.15)

        if dot > 0.0 and (
            abs(angle) <= float(move_angle_threshold)
            or lateral_offset <= float(lateral_threshold)
        ):
            return int(HabitatSimActions.MOVE_FORWARD)

        if angle < 0.0:
            return int(HabitatSimActions.TURN_LEFT)
        if angle > 0.0:
            return int(HabitatSimActions.TURN_RIGHT)
        return int(HabitatSimActions.MOVE_FORWARD)

    def get_teacher_action(self) -> int:
        goal_state = self._teacher_active_goal_state()
        if goal_state is None:
            self._teacher_goal_signature_state = None
            self._clear_teacher_goal_state()
            return self._task_action_index("STOP")

        goal_signature = self._teacher_goal_signature(goal_state)
        if goal_signature != self._teacher_goal_signature_state:
            self._teacher_goal_signature_state = goal_signature
            self._clear_teacher_goal_state()

        distance_to_goal = self._teacher_goal_distance(goal_state)
        success_distance = self._teacher_success_distance()
        goal_visible = self._teacher_goal_visible(goal_state)
        if (
            distance_to_goal is not None
            and float(distance_to_goal) < float(success_distance)
            and goal_visible
        ):
            self._clear_teacher_goal_state()
            task = self.habitat_env.task
            if int(getattr(task, "remaining_submit_count", 0)) > 0:
                return self._task_action_index("LIFELONG_SUBMIT")
            return self._task_action_index("STOP")

        target_positions = self._teacher_target_positions(goal_state)
        if len(target_positions) == 0:
            return self._task_action_index("STOP")

        if self._teacher_scan_target_index is not None:
            if (
                0 <= int(self._teacher_scan_target_index) < len(target_positions)
                and distance_to_goal is not None
                and float(distance_to_goal) < float(success_distance)
                and goal_visible
            ):
                self._clear_teacher_goal_state()
                task = self.habitat_env.task
                if int(getattr(task, "remaining_submit_count", 0)) > 0:
                    return self._task_action_index("LIFELONG_SUBMIT")
                return self._task_action_index("STOP")

            scan_action = self._teacher_continue_scan()
            if scan_action is not None:
                return int(scan_action)

        target_index = self._teacher_select_target_index(target_positions)
        if target_index is None:
            self._teacher_blocked_target_indices = set()
            target_index = self._teacher_select_target_index(target_positions)
            if target_index is None:
                return self._task_action_index("STOP")

        target_position = target_positions[int(target_index)]
        if self._teacher_at_target_position(target_position):
            if goal_visible and distance_to_goal is not None and float(distance_to_goal) < float(success_distance):
                self._clear_teacher_goal_state()
                task = self.habitat_env.task
                if int(getattr(task, "remaining_submit_count", 0)) > 0:
                    return self._task_action_index("LIFELONG_SUBMIT")
                return self._task_action_index("STOP")
            return self._teacher_start_scan(int(target_index))

        waypoint_position = self._teacher_next_waypoint(target_position)
        sim_action = self._teacher_heading_sim_action(waypoint_position)
        return self._task_action_index(self._sim_action_name(int(sim_action)))

    def _goal_order_mode(self) -> str:
        task = getattr(self._env, "task", None)
        mode = getattr(task, "goal_order_mode", None)
        if isinstance(mode, str) and mode.strip():
            return mode.strip().lower()
        fallback = getattr(self._core_env_config.TASK, "GOAL_ORDER_MODE", "ordered")
        return str(fallback).strip().lower()

    def _mode_reward_value(
        self,
        ordered_key: str,
        unordered_key: str,
        fallback_key: str,
        default: float,
    ) -> float:
        mode_key = ordered_key if self._goal_order_mode() == "ordered" else unordered_key
        value = getattr(self._rl_config, mode_key, None)
        if value is None:
            value = getattr(self._rl_config, fallback_key, default)
        return float(value)

    def _goal_completion_value(self) -> float:
        metrics = self._env.get_metrics()
        completion_value = metrics.get("lifelong_goal_completion", 0.0)
        return float(completion_value)

    def _time_penalty_scale(self) -> float:
        if not bool(getattr(self._rl_config, "DYNAMIC_TIME_PENALTY", True)):
            return 1.0

        decay = self._mode_reward_value(
            ordered_key="ORDERED_TIME_PENALTY_DECAY",
            unordered_key="UNORDERED_TIME_PENALTY_DECAY",
            fallback_key="TIME_PENALTY_DECAY",
            default=0.75,
        )
        min_scale = self._mode_reward_value(
            ordered_key="ORDERED_TIME_PENALTY_MIN_SCALE",
            unordered_key="UNORDERED_TIME_PENALTY_MIN_SCALE",
            fallback_key="TIME_PENALTY_MIN_SCALE",
            default=0.25,
        )

        completion = min(max(self._goal_completion_value(), 0.0), 1.0)
        scale = 1.0 - float(decay) * float(completion)
        return float(max(float(min_scale), min(1.0, float(scale))))

    def reset(self):
        self._teacher_goal_signature_state = None
        self._clear_teacher_goal_state()
        return super().reset()

    def step(self, *args, **kwargs):
        return super().step(*args, **kwargs)

    def _last_action_feedback(self) -> Dict[str, Any]:
        task = getattr(self._env, "task", None)
        if task is None or not hasattr(task, "get_last_action_feedback"):
            return {}

        feedback = task.get_last_action_feedback()
        if isinstance(feedback, dict):
            return feedback
        return {}

    def _action_feedback_reward(self) -> float:
        feedback = self._last_action_feedback()
        action_name = str(feedback.get("action_name") or "").strip().upper()
        found_goal = bool(feedback.get("found_goal_this_step", False))

        if not action_name:
            return 0.0

        if action_name == "STOP":
            return self._mode_reward_value(
                ordered_key="ORDERED_STOP_PENALTY",
                unordered_key="UNORDERED_STOP_PENALTY",
                fallback_key="STOP_PENALTY",
                default=-5.0,
            )

        if action_name == "LIFELONG_SUBMIT":
            reward = self._mode_reward_value(
                ordered_key="ORDERED_SUBMIT_PENALTY",
                unordered_key="UNORDERED_SUBMIT_PENALTY",
                fallback_key="SUBMIT_PENALTY",
                default=-2.0,
            )
            if found_goal:
                reward += self._mode_reward_value(
                    ordered_key="ORDERED_FOUND_GOAL_REWARD",
                    unordered_key="UNORDERED_FOUND_GOAL_REWARD",
                    fallback_key="FOUND_GOAL_REWARD",
                    default=10.0,
                )
            return float(reward)

        return 0.0

    def get_reward_range(self):
        slack_reward = self._mode_reward_value(
            ordered_key="ORDERED_SLACK_REWARD",
            unordered_key="UNORDERED_SLACK_REWARD",
            fallback_key="SLACK_REWARD",
            default=-0.01,
        )
        success_reward = self._mode_reward_value(
            ordered_key="ORDERED_SUCCESS_REWARD",
            unordered_key="UNORDERED_SUCCESS_REWARD",
            fallback_key="SUCCESS_REWARD",
            default=20.0,
        )

        stop_floor = self._mode_reward_value(
            ordered_key="ORDERED_STOP_PENALTY",
            unordered_key="UNORDERED_STOP_PENALTY",
            fallback_key="STOP_PENALTY",
            default=-5.0,
        )
        submit_floor = self._mode_reward_value(
            ordered_key="ORDERED_SUBMIT_PENALTY",
            unordered_key="UNORDERED_SUBMIT_PENALTY",
            fallback_key="SUBMIT_PENALTY",
            default=-2.0,
        )
        submit_success = submit_floor + self._mode_reward_value(
            ordered_key="ORDERED_FOUND_GOAL_REWARD",
            unordered_key="UNORDERED_FOUND_GOAL_REWARD",
            fallback_key="FOUND_GOAL_REWARD",
            default=10.0,
        )
        return (
            float(slack_reward) + min(0.0, float(stop_floor), float(submit_floor)) - 10.0,
            float(slack_reward) + max(float(submit_success), float(stop_floor) + float(success_reward)) + 10.0,
        )

    def _success_value(self) -> float:
        success_metric = self._env.get_metrics().get(self._success_measure_name, 0.0)
        return float(success_metric)

    def get_reward(self, observations):
        reward = 0.0
        if bool(getattr(self._rl_config, "WITH_TIME_PENALTY", True)):
            reward += self._mode_reward_value(
                ordered_key="ORDERED_SLACK_REWARD",
                unordered_key="UNORDERED_SLACK_REWARD",
                fallback_key="SLACK_REWARD",
                default=-0.01,
            )

        reward_metric = self._env.get_metrics().get(self._reward_measure_name, 0.0)
        reward += float(reward_metric)
        reward += self._action_feedback_reward()

        feedback = self._last_action_feedback()
        action_name = str(feedback.get("action_name") or "").strip().upper()
        if action_name == "STOP" and self._success_value() >= 1.0:
            reward += self._mode_reward_value(
                ordered_key="ORDERED_SUCCESS_REWARD",
                unordered_key="UNORDERED_SUCCESS_REWARD",
                fallback_key="SUCCESS_REWARD",
                default=20.0,
            )

        assert not math.isnan(reward)
        return reward

    def get_done(self, observations):
        if self._env.episode_over:
            return True
        if self._end_on_success and self._success_value() >= 1.0:
            return True
        return False

    def get_info(self, observations):
        info = dict(self.habitat_env.get_metrics())
        task = getattr(self._env, "task", None)
        if task is None:
            return info

        remaining_submit_count = getattr(task, "remaining_submit_count", None)
        if remaining_submit_count is not None:
            info["remaining_submit_count"] = int(remaining_submit_count)

        if hasattr(task, "get_last_action_feedback"):
            feedback = task.get_last_action_feedback()
            if isinstance(feedback, dict):
                info["last_action_feedback"] = feedback
                info["found_goal_this_step"] = int(
                    bool(feedback.get("found_goal_this_step", False))
                )
                info["found_goal_index_this_step"] = int(
                    feedback.get("found_goal_index", -1)
                )
                if "remaining_submit_count" in feedback:
                    info["remaining_submit_count"] = int(
                        feedback["remaining_submit_count"]
                    )
                if "submit_count" in feedback:
                    info["submit_count"] = int(feedback["submit_count"])
                if "submit_limit" in feedback:
                    info["submit_limit"] = int(feedback["submit_limit"])

        return info
