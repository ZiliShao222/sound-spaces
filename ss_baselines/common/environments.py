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

from typing import Optional, Type
import logging
import math

import habitat
from habitat import Config, Dataset
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
        return self.habitat_env.get_metrics()

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
        super().__init__(self._core_env_config, dataset)

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
        try:
            return float(completion_value)
        except Exception:
            return 0.0

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
        return super().reset()

    def step(self, *args, **kwargs):
        return super().step(*args, **kwargs)

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
            default=10.0,
        )
        return (
            float(slack_reward) - 10.0,
            float(success_reward) + 10.0,
        )

    def _success_value(self) -> float:
        success_metric = self._env.get_metrics().get(self._success_measure_name, 0.0)
        try:
            return float(success_metric)
        except Exception:
            return 0.0

    def get_reward(self, observations):
        reward = 0.0
        if bool(getattr(self._rl_config, "WITH_TIME_PENALTY", True)):
            reward += self._mode_reward_value(
                ordered_key="ORDERED_SLACK_REWARD",
                unordered_key="UNORDERED_SLACK_REWARD",
                fallback_key="SLACK_REWARD",
                default=-0.01,
            ) * self._time_penalty_scale()

        reward_metric = self._env.get_metrics().get(self._reward_measure_name, 0.0)
        try:
            reward += float(reward_metric)
        except Exception:
            pass

        if self._success_value() >= 1.0:
            reward += self._mode_reward_value(
                ordered_key="ORDERED_SUCCESS_REWARD",
                unordered_key="UNORDERED_SUCCESS_REWARD",
                fallback_key="SUCCESS_REWARD",
                default=10.0,
            )

        return reward

    def get_done(self, observations):
        if self._env.episode_over:
            return True
        if self._end_on_success and self._success_value() >= 1.0:
            return True
        return False

    def get_info(self, observations):
        return self.habitat_env.get_metrics()
