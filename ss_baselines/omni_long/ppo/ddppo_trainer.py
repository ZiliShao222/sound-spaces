#!/usr/bin/env python3

from __future__ import annotations

import os
import random
import time
from collections import defaultdict, deque
from typing import DefaultDict, Dict, List

import numpy as np
import torch
import torch.distributed as distrib
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, logger

from ss_baselines.common.baseline_registry import baseline_registry
from ss_baselines.common.env_utils import construct_envs
from ss_baselines.common.environments import get_env_class
from ss_baselines.common.tensorboard_utils import TensorboardWriter
from ss_baselines.common.utils import batch_obs, linear_decay
from ss_baselines.omni_long.ppo.ddppo import DDPPO
from ss_baselines.omni_long.ppo.policy import OmniLongBaselinePolicy
from ss_baselines.omni_long.ppo.ppo_trainer import OmniLongPPOTrainer, OmniLongRolloutStorage
from ss_baselines.savi.ddppo.algo.ddp_utils import init_distrib_slurm


@baseline_registry.register_trainer(name="OmniLongDDPPOTrainer")
class OmniLongDDPPOTrainer(OmniLongPPOTrainer):
    def __init__(self, config: Config):
        super().__init__(config)
        self.local_rank = 0
        self.world_rank = 0
        self.world_size = 1

    def _setup_actor_critic_agent(self, ppo_cfg: Config, observation_space=None) -> None:
        if self.world_rank == 0:
            logger.add_filehandler(self.config.LOG_FILE)
        if observation_space is None:
            observation_space = self.envs.observation_spaces[0]
        action_space = self.envs.action_spaces[0]

        self.actor_critic = OmniLongBaselinePolicy.from_config(
            config=self.config,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.actor_critic.to(self.device)

        self.agent = DDPPO(
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

    def save_checkpoint(self, file_name: str, extra_state=None) -> None:
        if self.world_rank != 0:
            return
        super().save_checkpoint(file_name, extra_state=extra_state)

    def _distributed_scalar_mean(self, value: float) -> float:
        tensor = torch.tensor([float(value)], dtype=torch.float32, device=self.device)
        distrib.all_reduce(tensor)
        tensor /= float(self.world_size)
        return float(tensor.item())

    def train(self) -> None:
        if os.environ.get("LOCAL_RANK") is not None:
            self.local_rank = int(os.environ["LOCAL_RANK"])
            self.world_rank = int(os.environ["RANK"])
            self.world_size = int(os.environ["WORLD_SIZE"])
            distrib.init_process_group(
                backend=self.config.RL.DDPPO.distrib_backend,
                init_method="env://",
            )
        else:
            self.local_rank, _ = init_distrib_slurm(self.config.RL.DDPPO.distrib_backend)
            self.world_rank = distrib.get_rank()
            self.world_size = distrib.get_world_size()

        self.config.defrost()
        self.config.TORCH_GPU_ID = self.local_rank
        self.config.SIMULATOR_GPU_ID = self.local_rank
        self.config.SEED += self.world_rank * self.config.NUM_PROCESSES
        self.config.TASK_CONFIG.SEED += self.world_rank * self.config.NUM_PROCESSES
        if self.world_rank != 0:
            self.config.LOG_FILE = ""
            self.config.TENSORBOARD_DIR = ""
            self.config.WANDB.ENABLED = False
        self.config.freeze()

        logger.info(
            "DDPPO init rank=%s world_size=%s local_rank=%s",
            self.world_rank,
            self.world_size,
            self.local_rank,
        )

        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)

        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if self.world_rank == 0:
            os.makedirs(self.config.MODEL_DIR, exist_ok=True)
            os.makedirs(self.config.CHECKPOINT_FOLDER, exist_ok=True)
            os.makedirs(self.config.TENSORBOARD_DIR, exist_ok=True)
        distrib.barrier()

        self.envs = construct_envs(self.config, get_env_class(self.config.ENV_NAME))
        env_observation_space = self.envs.observation_spaces[0]
        observation_space = OmniLongBaselinePolicy.build_policy_observation_space(
            self.config,
            env_observation_space,
        )
        action_space = self.envs.action_spaces[0]
        ppo_cfg = self.config.RL.PPO
        self._setup_actor_critic_agent(ppo_cfg, observation_space=observation_space)
        self.agent.init_distributed(find_unused_params=True)

        if self.world_rank == 0:
            logger.info(
                "agent number of trainable parameters: %s",
                sum(param.numel() for param in self.agent.parameters() if param.requires_grad),
            )

        rollouts = OmniLongRolloutStorage(
            num_steps=ppo_cfg.num_steps,
            num_envs=self.envs.num_envs,
            observation_space=observation_space,
            action_space=action_space,
            memory_size=ppo_cfg.transformer_memory_size,
            memory_dim=ppo_cfg.hidden_size,
        )
        rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        batch = self.actor_critic.prepare_observations(batch, masks=None)
        for sensor in batch:
            rollouts.observations[sensor][0].copy_(batch[sensor])
        initial_memory_tokens, initial_memory_mask = self.actor_critic.init_memory(
            batch_size=self.envs.num_envs,
            device=self.device,
        )
        rollouts.memory_tokens[0].copy_(initial_memory_tokens)
        rollouts.memory_mask[0].copy_(initial_memory_mask)

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device=self.device)
        running_episode_stats: Dict[str, torch.Tensor] = {
            "count": torch.zeros(self.envs.num_envs, 1, device=self.device),
            "reward": torch.zeros(self.envs.num_envs, 1, device=self.device),
        }
        window_episode_stats: DefaultDict[str, deque] = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda epoch: linear_decay(epoch, self.config.NUM_UPDATES),
        )

        t_start = time.time()
        env_time = 0.0
        pth_time = 0.0
        count_steps = 0
        count_checkpoints = 0
        interval_stage_times: DefaultDict[str, float] = defaultdict(float)

        writer_dir = self.config.TENSORBOARD_DIR if self.world_rank == 0 else None
        with TensorboardWriter(writer_dir, flush_secs=self.flush_secs) as writer:
            for update in range(self.config.NUM_UPDATES):
                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()
                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(update, self.config.NUM_UPDATES)

                for _ in range(ppo_cfg.num_steps):
                    delta_pth_time, delta_env_time, delta_steps, collect_stage_times = self._collect_rollout_step(
                        rollouts,
                        current_episode_reward,
                        running_episode_stats,
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps += delta_steps
                    for key, value in collect_stage_times.items():
                        interval_stage_times[key] += value

                delta_pth_time, value_loss, action_loss, dist_entropy, update_stage_times = self._update_agent(
                    ppo_cfg,
                    rollouts,
                )
                pth_time += delta_pth_time
                for key, value in update_stage_times.items():
                    interval_stage_times[key] += value

                if self.world_rank == 0:
                    for key, value in running_episode_stats.items():
                        reduced_value = value.clone()
                        distrib.reduce(reduced_value, dst=0)
                        window_episode_stats[key].append(reduced_value.cpu())
                else:
                    for value in running_episode_stats.values():
                        distrib.reduce(value.clone(), dst=0)

                global_count_steps = count_steps * self.world_size
                value_loss = self._distributed_scalar_mean(value_loss)
                action_loss = self._distributed_scalar_mean(action_loss)
                dist_entropy = self._distributed_scalar_mean(dist_entropy)

                if self.world_rank == 0:
                    deltas = {
                        key: ((values[-1] - values[0]).sum().item() if len(values) > 1 else values[0].sum().item())
                        for key, values in window_episode_stats.items()
                    }
                    deltas["count"] = max(1.0, deltas.get("count", 1.0))

                    writer.add_scalar("Metrics/reward", deltas.get("reward", 0.0) / deltas["count"], global_count_steps)
                    for metric, value in deltas.items():
                        if metric in {"reward", "count"}:
                            continue
                        writer.add_scalar(f"Metrics/{metric}", value / deltas["count"], global_count_steps)

                    writer.add_scalar("Policy/value_loss", value_loss, global_count_steps)
                    writer.add_scalar("Policy/policy_loss", action_loss, global_count_steps)
                    writer.add_scalar("Policy/entropy", dist_entropy, global_count_steps)
                    writer.add_scalar("Policy/lr", self.agent.optimizer.param_groups[0]["lr"], global_count_steps)
                    for stage_name, stage_value in interval_stage_times.items():
                        writer.add_scalar(f"Timing/{stage_name}", stage_value, global_count_steps)

                    if update > 0 and update % self.config.LOG_INTERVAL == 0:
                        logger.info(
                            "update: {}\tfps: {:.3f}".format(
                                update,
                                global_count_steps / max(1e-6, time.time() - t_start),
                            )
                        )
                        logger.info(
                            "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\tframes: {}".format(
                                update,
                                env_time,
                                pth_time,
                                global_count_steps,
                            )
                        )
                        logger.info(
                            "Average window size: {}  {}".format(
                                len(window_episode_stats.get("count", [])),
                                "  ".join(
                                    f"{key}: {value / deltas['count']:.3f}"
                                    for key, value in deltas.items()
                                    if key != "count"
                                ),
                            )
                        )
                        logger.info(
                            "Timing breakdown (interval): {}".format(
                                self._format_stage_timing_breakdown(dict(interval_stage_times))
                            )
                        )
                        logger.info(
                            "Timing ratio (interval): {}".format(
                                self._format_stage_timing_ratio(dict(interval_stage_times))
                            )
                        )
                        interval_stage_times = defaultdict(float)

                    if update % self.config.CHECKPOINT_INTERVAL == 0:
                        self.save_checkpoint(
                            f"ckpt.{count_checkpoints}.pth",
                            extra_state={
                                "update": update,
                                "count_steps": global_count_steps,
                            },
                        )
                        count_checkpoints += 1
                else:
                    if update % self.config.LOG_INTERVAL == 0:
                        interval_stage_times = defaultdict(float)

        self.envs.close()
        distrib.barrier()
