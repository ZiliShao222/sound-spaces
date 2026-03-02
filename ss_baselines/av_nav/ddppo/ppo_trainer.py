#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
import logging
from collections import deque, defaultdict
from typing import Dict, List, Any
import json
import random
import glob

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from numpy.linalg import norm
from gym import spaces

from habitat import Config, logger
from ss_baselines.common.utils import observations_to_image
from ss_baselines.common.base_trainer import BaseRLTrainer
from ss_baselines.common.baseline_registry import baseline_registry
from ss_baselines.common.env_utils import construct_envs
from ss_baselines.common.environments import get_env_class
from ss_baselines.common.rollout_storage import RolloutStorage
from ss_baselines.common.tensorboard_utils import TensorboardWriter
from ss_baselines.common.utils import (
    batch_obs,
    generate_video,
    linear_decay,
    plot_top_down_map,
    resize_observation,
    NpEncoder
)
from ss_baselines.av_nav.ppo.policy import AudioNavBaselinePolicy
from ss_baselines.av_nav.ppo.ppo import PPO
from ss_baselines.savi.ppo.slurm_utils import (
    EXIT,
    REQUEUE,
    load_interrupted_state,
    requeue_job,
    save_interrupted_state,
)
from soundspaces.tasks.shortest_path_follower import ShortestPathFollower


class DataParallelPassthrough(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


@baseline_registry.register_trainer(name="av_nav_ppo")
class PPOTrainer(BaseRLTrainer):
    r"""Trainer class for PPO algorithm

    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None

        self._static_smt_encoder = False
        self._encoder = None

    def _setup_actor_critic_agent(self, ppo_cfg: Config, observation_space=None) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        if observation_space is None:
            observation_space = self.envs.observation_spaces[0]

        self.actor_critic = AudioNavBaselinePolicy(
            observation_space=observation_space,
            action_space=self.envs.action_spaces[0],
            hidden_size=ppo_cfg.hidden_size,
            goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
            extra_rgb=self.config.EXTRA_RGB
        )

        self.agent = PPO(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
        )

        if self.config.RESUME:
            ckpt_dict = self.load_checkpoint('data/models/smt_with_pose/ckpt.400.pth', map_location="cpu")
            self.agent.actor_critic.net.visual_encoder.load_state_dict(self.search_dict(ckpt_dict, 'visual_encoder'))
            self.agent.actor_critic.net.goal_encoder.load_state_dict(self.search_dict(ckpt_dict, 'goal_encoder'))
            self.agent.actor_critic.net.action_encoder.load_state_dict(self.search_dict(ckpt_dict, 'action_encoder'))

        self.actor_critic.to(self.device)

    @staticmethod
    def search_dict(ckpt_dict, encoder_name):
        encoder_dict = {}
        for key, value in ckpt_dict['state_dict'].items():
            if encoder_name in key:
                encoder_dict['.'.join(key.split('.')[3:])] = value

        return encoder_dict

    def save_checkpoint(
        self, file_name: str, extra_state=None
    ) -> None:
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    def try_to_resume_checkpoint(self):
        checkpoints = glob.glob(f"{self.config.CHECKPOINT_FOLDER}/*.pth")
        if len(checkpoints) == 0:
            count_steps = 0
            count_checkpoints = 0
            start_update = 0
        else:
            last_ckpt = sorted(checkpoints, key=lambda x: int(x.split(".")[1]))[-1]
            checkpoint_path = last_ckpt
            # Restore checkpoints to models
            ckpt_dict = self.load_checkpoint(checkpoint_path)
            self.agent.load_state_dict(ckpt_dict["state_dict"])
            ckpt_id = int(last_ckpt.split("/")[-1].split(".")[1])
            count_steps = ckpt_dict["extra_state"]["step"]
            count_checkpoints = ckpt_id + 1
            start_update = ckpt_dict["config"].CHECKPOINT_INTERVAL * ckpt_id + 1
            print(f"Resuming checkpoint {last_ckpt} at {count_steps} frames")

        return count_steps, count_checkpoints, start_update

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    def _collect_rollout_step(
        self, rollouts, current_episode_reward, running_episode_stats
    ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()
        # sample actions
        with torch.no_grad():
            step_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }

            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states
            ) = self.actor_critic.act(
                step_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            )

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()

        outputs = self.envs.step([a[0].item() for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
        logging.debug('Reward: {}'.format(rewards[0]))

        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(observations, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float, device=current_episode_reward.device)
        rewards = rewards.unsqueeze(1)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones], dtype=torch.float, device=current_episode_reward.device
        )

        current_episode_reward += rewards
        running_episode_stats["reward"] += (1 - masks) * current_episode_reward
        running_episode_stats["count"] += 1 - masks
        for k, v in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v, dtype=torch.float, device=current_episode_reward.device
            ).unsqueeze(1)
            if k not in running_episode_stats:
                running_episode_stats[k] = torch.zeros_like(
                    running_episode_stats["count"]
                )

            running_episode_stats[k] += (1 - masks) * v

        current_episode_reward *= masks

        rollouts.insert(
            batch,
            recurrent_hidden_states,
            actions,
            actions_log_probs,
            values,
            rewards.to(device=self.device),
            masks.to(device=self.device),
        )

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs

    def _update_agent(self, ppo_cfg, rollouts):
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {
                k: v[-1] for k, v in rollouts.observations.items()
            }

            next_value = self.actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step]
            ).detach()

        rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        value_loss, action_loss, dist_entropy = self.agent.update(rollouts)

        rollouts.after_update()

        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
        )

    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """
        logger.info(f"config: {self.config}")
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)

        # add_signal_handlers()

        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME), workers_ignore_signals=True
        )

        ppo_cfg = self.config.RL.PPO
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)
        self._setup_actor_critic_agent(ppo_cfg)
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size
        )
        rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(observations)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        t_start = time.time()
        env_time = 0
        pth_time = 0
        count_steps = 0
        count_checkpoints = 0
        start_update = 0
        prev_time = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )

        interrupted_state = load_interrupted_state(model_dir=self.config.MODEL_DIR)
        if interrupted_state is not None:
            self.agent.load_state_dict(interrupted_state["state_dict"])
            self.agent.optimizer.load_state_dict(
                interrupted_state["optimizer_state"]
            )
            lr_scheduler.load_state_dict(interrupted_state["lr_scheduler_state"])

            requeue_stats = interrupted_state["requeue_stats"]
            env_time = requeue_stats["env_time"]
            pth_time = requeue_stats["pth_time"]
            count_steps = requeue_stats["count_steps"]
            count_checkpoints = requeue_stats["count_checkpoints"]
            start_update = requeue_stats["start_update"]
            prev_time = requeue_stats["prev_time"]

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            for update in range(start_update, self.config.NUM_UPDATES):
                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )

                if EXIT.is_set():
                    self.envs.close()

                    if REQUEUE.is_set():
                        requeue_stats = dict(
                            env_time=env_time,
                            pth_time=pth_time,
                            count_steps=count_steps,
                            count_checkpoints=count_checkpoints,
                            start_update=update,
                            prev_time=(time.time() - t_start) + prev_time,
                        )
                        save_interrupted_state(
                            dict(
                                state_dict=self.agent.state_dict(),
                                optimizer_state=self.agent.optimizer.state_dict(),
                                lr_scheduler_state=lr_scheduler.state_dict(),
                                config=self.config,
                                requeue_stats=requeue_stats,
                            ),
                            model_dir=self.config.MODEL_DIR
                        )
                        requeue_job()
                    return

                for step in range(ppo_cfg.num_steps):
                    delta_pth_time, delta_env_time, delta_steps = self._collect_rollout_step(
                        rollouts,
                        current_episode_reward,
                        running_episode_stats
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps += delta_steps

                delta_pth_time, value_loss, action_loss, dist_entropy = self._update_agent(
                    ppo_cfg, rollouts
                )
                pth_time += delta_pth_time

                deltas = {
                    k: (
                        (v[-1] - v[0]).sum().item()
                        if len(v) > 1
                        else v[0].sum().item()
                    )
                    for k, v in window_episode_stats.items()
                }
                deltas["count"] = max(deltas["count"], 1.0)

                writer.add_scalar(
                    "Metrics/reward", deltas["reward"] / deltas["count"], count_steps
                )

                # Check to see if there are any metrics
                # that haven't been logged yet
                metrics = {
                    k: v / deltas["count"]
                    for k, v in deltas.items()
                    if k not in {"reward", "count"}
                }
                if len(metrics) > 0:
                    # writer.add_scalars("metrics", metrics, count_steps)
                    for metric, value in metrics.items():
                        writer.add_scalar(f"Metrics/{metric}", value, count_steps)

                writer.add_scalar("Policy/value_loss", value_loss, count_steps)
                writer.add_scalar("Policy/policy_loss", action_loss, count_steps)
                writer.add_scalar("Policy/entropy_loss", dist_entropy, count_steps)
                writer.add_scalar('Policy/learning_rate', lr_scheduler.get_lr()[0], count_steps)

                # log stats
                if update > 0 and update % self.config.LOG_INTERVAL == 0:
                    logger.info(
                        "update: {}\tfps: {:.3f}\t".format(
                            update, count_steps / (time.time() - t_start)
                        )
                    )

                    logger.info(
                        "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                        "frames: {}".format(
                            update, env_time, pth_time, count_steps
                        )
                    )

                    logger.info(
                        "Average window size: {}  {}".format(
                            len(window_episode_stats["count"]),
                            "  ".join(
                                "{}: {:.3f}".format(k, v / deltas["count"])
                                for k, v in deltas.items()
                                if k != "count"
                            ),
                        )
                    )

                # checkpoint model
                if update % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(f"ckpt.{count_checkpoints}.pth")
                    count_checkpoints += 1

            self.envs.close()

    def _sample_random_actions(self, action_space, num_envs: int):
        action_ids = torch.randint(
                1,
                4,
                (num_envs, 1),
                device=self.device,
                dtype=torch.long,
            )
        return action_ids, [int(a.item()) for a in action_ids]
        # if isinstance(action_space, spaces.Discrete):
        #     if action_space.n <= 1:
        #         logger.warning(
        #             "Action space has <=1 action; falling back to action=0."
        #         )
        #         action_ids = torch.zeros(
        #             (num_envs, 1), device=self.device, dtype=torch.long
        #         )
        #         return action_ids, [0 for _ in range(num_envs)]
        #     action_ids = torch.randint(
        #         1,
        #         action_space.n,
        #         (num_envs, 1),
        #         device=self.device,
        #         dtype=torch.long,
        #     )
        #     return action_ids, [int(a[0].item()) for a in action_ids]

        # samples = [action_space.sample() for _ in range(num_envs)]
        # actions = torch.as_tensor(samples, device=self.device)
        # if actions.ndim == 1:
        #     actions = actions.unsqueeze(1)
        # return actions, samples

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> Dict:
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)

        random_policy = False
        ckpt_dict = None
        if checkpoint_path and os.path.isfile(checkpoint_path):
            try:
                ckpt_dict = self.load_checkpoint(
                    checkpoint_path, map_location="cpu"
                )
            except Exception as exc:
                logger.warning(
                    f"Failed to load ckpt {checkpoint_path}; "
                    f"falling back to random policy: {exc}"
                )
                random_policy = True
        else:
            logger.warning(
                f"Checkpoint not found at {checkpoint_path}; "
                "falling back to random policy."
            )
            random_policy = True

        if (
            not random_policy
            and ckpt_dict is not None
            and self.config.EVAL.USE_CKPT_CONFIG
        ):
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            if self.config.EVAL.USE_CKPT_CONFIG and random_policy:
                logger.warning(
                    "EVAL.USE_CKPT_CONFIG is True but no ckpt loaded; "
                    "using eval config only."
                )
            config = self.config.clone()

        ppo_cfg = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        if (
            self.config.DISPLAY_RESOLUTION
            != config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH
        ):
            model_resolution = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH
            config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH = (
                config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT
            ) = (
                config.TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH
            ) = (
                config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT
            ) = self.config.DISPLAY_RESOLUTION
        else:
            model_resolution = self.config.DISPLAY_RESOLUTION
        config.freeze()

        if random_policy:
            # Keep v0 action space so numeric action ids map to STOP/MOVE_FORWARD/TURN_LEFT/TURN_RIGHT.
            # Using "move-all" removes TURN_LEFT/RIGHT and breaks integer action ids.
            config.defrost()
            config.TASK_CONFIG.SIMULATOR.ACTION_SPACE_CONFIG = "v0"
            config.freeze()

        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()
        elif "top_down_map" in self.config.VISUALIZATION_OPTION:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.freeze()

        logger.info(f"env config: {config}")
        self.envs = construct_envs(
            config, get_env_class(config.ENV_NAME)
        )
        if self.config.DISPLAY_RESOLUTION != model_resolution:
            observation_space = self.envs.observation_spaces[0]
            observation_space.spaces["depth"] = spaces.Box(
                low=0,
                high=1,
                shape=(model_resolution, model_resolution, 1),
                dtype=np.uint8,
            )
            observation_space.spaces["rgb"] = spaces.Box(
                low=0,
                high=1,
                shape=(model_resolution, model_resolution, 3),
                dtype=np.uint8,
            )
        else:
            observation_space = self.envs.observation_spaces[0]
        self._setup_actor_critic_agent(ppo_cfg, observation_space)

        if config.FOLLOW_SHORTEST_PATH:
            follower = ShortestPathFollower(
                self.envs.workers[0]._env.habitat_env.sim, 0.5, False
            )

        if not random_policy and ckpt_dict is not None:
            self.agent.load_state_dict(ckpt_dict["state_dict"])
            self.actor_critic = self.agent.actor_critic

        self.metric_uuids = []
        for metric_name in self.config.TASK_CONFIG.TASK.MEASUREMENTS:
            metric_cfg = getattr(self.config.TASK_CONFIG.TASK, metric_name)
            measure_type = baseline_registry.get_measure(metric_cfg.TYPE)
            assert measure_type is not None, "invalid measurement type {}".format(
                metric_cfg.TYPE
            )
            self.metric_uuids.append(
                measure_type(sim=None, task=None, config=None)._get_uuid()
            )

        observations = self.envs.reset()
        if self.config.DISPLAY_RESOLUTION != model_resolution:
            resize_observation(observations, model_resolution)
        batch = batch_obs(observations, self.device)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.config.NUM_PROCESSES,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.config.NUM_PROCESSES, 1, device=self.device
        )
        stats_episodes = dict()

        rgb_frames = [[] for _ in range(self.config.NUM_PROCESSES)]
        audios = [[] for _ in range(self.config.NUM_PROCESSES)]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        t = tqdm(total=self.config.TEST_EPISODE_COUNT)
        while (
            len(stats_episodes) < self.config.TEST_EPISODE_COUNT
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            env_actions = None
            if random_policy and not config.FOLLOW_SHORTEST_PATH: 
                
                actions, env_actions = self._sample_random_actions(
                    self.envs.action_spaces[0], self.envs.num_envs
                )
                # env_actions = actions
                prev_actions.copy_(actions)
                print(f'env_actions: {env_actions}')
            else:
                with torch.no_grad():
                    _, actions, _, test_recurrent_hidden_states = (
                        self.actor_critic.act(
                            batch,
                            test_recurrent_hidden_states,
                            prev_actions,
                            not_done_masks,
                            deterministic=False,
                        )
                    )
                    prev_actions.copy_(actions)

            if config.FOLLOW_SHORTEST_PATH:
                actions = [
                    follower.get_next_action(
                        self.envs.workers[0]
                        ._env.habitat_env.current_episode.goals[0]
                        .view_points[0]
                        .agent_state.position
                    )
                ]
                outputs = self.envs.step(actions)
            elif env_actions is not None:
                outputs = self.envs.step(env_actions)
            else:
                outputs = self.envs.step([a[0].item() for a in actions])

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            for i in range(self.envs.num_envs):
                if len(self.config.VIDEO_OPTION) > 0:
                    if (
                        config.TASK_CONFIG.SIMULATOR.CONTINUOUS_VIEW_CHANGE
                        and "intermediate" in observations[i]
                    ):
                        for observation in observations[i]["intermediate"]:
                            frame = observations_to_image(
                                observation, infos[i]
                            )
                            rgb_frames[i].append(frame)
                        del observations[i]["intermediate"]

                    if "rgb" not in observations[i]:
                        observations[i]["rgb"] = np.zeros(
                            (
                                self.config.DISPLAY_RESOLUTION,
                                self.config.DISPLAY_RESOLUTION,
                                3,
                            )
                        )
                    frame = observations_to_image(observations[i], infos[i])
                    rgb_frames[i].append(frame)
                    audios[i].append(observations[i]["audiogoal"])

            if self.config.DISPLAY_RESOLUTION != model_resolution:
                resize_observation(observations, model_resolution)
            batch = batch_obs(observations, self.device)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            rewards = torch.tensor(
                rewards, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            for i in range(self.envs.num_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                if not_done_masks[i].item() == 0:
                    episode_stats = dict()
                    for metric_uuid in self.metric_uuids:
                        episode_stats[metric_uuid] = infos[i][metric_uuid]
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats["geodesic_distance"] = (
                        current_episodes[i].info["geodesic_distance"]
                    )
                    episode_stats["euclidean_distance"] = norm(
                        np.array(current_episodes[i].goals[0].position)
                        - np.array(current_episodes[i].start_position)
                    )
                    logger.info(
                        f"[Episode done] scene={current_episodes[i].scene_id} "
                        f"id={current_episodes[i].episode_id} stats={episode_stats}"
                    )
                    current_episode_reward[i] = 0
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats
                    t.update()

                    if len(self.config.VIDEO_OPTION) > 0:
                        fps = int(
                            1 / self.config.TASK_CONFIG.SIMULATOR.STEP_TIME
                        )
                        if "sound" in current_episodes[i].info:
                            sound = current_episodes[i].info["sound"]
                        else:
                            sound = (
                                current_episodes[i]
                                .sound_id.split("/")[1][:-4]
                            )
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i][:-1],
                            scene_name=current_episodes[i].scene_id.split(
                                "/"
                            )[3],
                            sound=sound,
                            sr=self.config.TASK_CONFIG.SIMULATOR.AUDIO.RIR_SAMPLING_RATE,
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metric_name="spl",
                            metric_value=infos[i]["spl"],
                            tb_writer=writer,
                            audios=audios[i][:-1],
                            fps=fps,
                        )

                        rgb_frames[i] = []
                        audios[i] = []

                    if "top_down_map" in self.config.VISUALIZATION_OPTION:
                        top_down_map = plot_top_down_map(
                            infos[i],
                            dataset=self.config.TASK_CONFIG.SIMULATOR.SCENE_DATASET,
                        )
                        scene = current_episodes[i].scene_id.split("/")[3]
                        writer.add_image(
                            "{}_{}_{}/{}".format(
                                config.EVAL.SPLIT,
                                scene,
                                current_episodes[i].episode_id,
                                config.BASE_TASK_CONFIG_PATH.split("/")[
                                    -1
                                ][:-5],
                            ),
                            top_down_map,
                            dataformats="WHC",
                        )

            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = sum(
                [v[stat_key] for v in stats_episodes.values()]
            )
        num_episodes = len(stats_episodes)

        stats_file = os.path.join(
            config.TENSORBOARD_DIR,
            "{}_stats_{}.json".format(config.EVAL.SPLIT, config.SEED),
        )
        new_stats_episodes = {
            ",".join(key): value for key, value in stats_episodes.items()
        }
        with open(stats_file, "w") as fo:
            json.dump(new_stats_episodes, fo, cls=NpEncoder)

        episode_reward_mean = aggregated_stats["reward"] / num_episodes
        episode_metrics_mean = {}
        for metric_uuid in self.metric_uuids:
            episode_metrics_mean[metric_uuid] = (
                aggregated_stats[metric_uuid] / num_episodes
            )

        logger.info(f"Average episode reward: {episode_reward_mean:.6f}")
        for metric_uuid in self.metric_uuids:
            logger.info(
                f"Average episode {metric_uuid}: "
                f"{episode_metrics_mean[metric_uuid]:.6f}"
            )

        if not config.EVAL.SPLIT.startswith("test"):
            writer.add_scalar(
                "{}/reward".format(config.EVAL.SPLIT),
                episode_reward_mean,
                checkpoint_index,
            )
            for metric_uuid in self.metric_uuids:
                writer.add_scalar(
                    f"{config.EVAL.SPLIT}/{metric_uuid}",
                    episode_metrics_mean[metric_uuid],
                    checkpoint_index,
                )

        self.envs.close()

        result = {"episode_reward_mean": episode_reward_mean}
        for metric_uuid in self.metric_uuids:
            result[
                "episode_{}_mean".format(metric_uuid)
            ] = episode_metrics_mean[metric_uuid]

        return result
