#!/usr/bin/env python3

from __future__ import annotations

import os
import random
import time
from collections import defaultdict, deque
from typing import DefaultDict, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, logger

from ss_baselines.common.baseline_registry import baseline_registry
from ss_baselines.common.env_utils import construct_envs
from ss_baselines.common.environments import get_env_class
from ss_baselines.common.tensorboard_utils import TensorboardWriter
from ss_baselines.common.utils import batch_obs, linear_decay
from ss_baselines.common.wandb_utils import WandbRun
from ss_baselines.omni_long.bc.bc import BehaviorCloning
from ss_baselines.omni_long.ppo.policy import OmniLongBaselinePolicy
from ss_baselines.omni_long.ppo.ppo_trainer import OmniLongPPOTrainer


class OmniLongBCRolloutStorage:
    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        observation_space,
        memory_size: int,
        memory_dim: int,
    ) -> None:
        self.num_steps = int(num_steps)
        self.num_envs = int(num_envs)
        self.step = 0
        self.observations: Dict[str, torch.Tensor] = {}
        for sensor_name, space in observation_space.spaces.items():
            self.observations[sensor_name] = torch.zeros(
                self.num_steps,
                self.num_envs,
                *space.shape,
            )
        self.prev_actions = torch.zeros(self.num_steps, self.num_envs, 1, dtype=torch.long)
        self.masks = torch.ones(self.num_steps, self.num_envs, 1)
        self.expert_actions = torch.zeros(self.num_steps, self.num_envs, 1, dtype=torch.long)
        self.memory_tokens = torch.zeros(
            self.num_steps,
            self.num_envs,
            int(memory_size),
            int(memory_dim),
        )
        self.memory_mask = torch.zeros(
            self.num_steps,
            self.num_envs,
            int(memory_size),
            dtype=torch.bool,
        )

    def to(self, device: torch.device) -> None:
        for sensor_name in self.observations:
            self.observations[sensor_name] = self.observations[sensor_name].to(device)
        self.prev_actions = self.prev_actions.to(device)
        self.masks = self.masks.to(device)
        self.expert_actions = self.expert_actions.to(device)
        self.memory_tokens = self.memory_tokens.to(device)
        self.memory_mask = self.memory_mask.to(device)

    def insert(
        self,
        observations: Dict[str, torch.Tensor],
        prev_actions: torch.Tensor,
        masks: torch.Tensor,
        expert_actions: torch.Tensor,
        memory_tokens: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> None:
        if self.step >= self.num_steps:
            raise RuntimeError(
                f"BC rollout storage overflow: step={self.step}, num_steps={self.num_steps}"
            )
        for sensor_name in observations:
            self.observations[sensor_name][self.step].copy_(observations[sensor_name])
        self.prev_actions[self.step].copy_(prev_actions)
        self.masks[self.step].copy_(masks)
        self.expert_actions[self.step].copy_(expert_actions)
        self.memory_tokens[self.step].copy_(memory_tokens)
        self.memory_mask[self.step].copy_(memory_mask)
        self.step += 1

    def after_update(self) -> None:
        self.step = 0

    @staticmethod
    def _flatten_helper(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(tensor.size(0) * tensor.size(1), *tensor.size()[2:])

    def feed_forward_generator(self, num_mini_batch: int):
        if self.step <= 0:
            return

        total_batch_size = self.step * self.num_envs
        num_mini_batch = max(1, min(int(num_mini_batch), int(total_batch_size)))
        permutation = torch.randperm(total_batch_size, device=self.prev_actions.device)
        mini_batches = torch.chunk(permutation, num_mini_batch)

        flat_observations = {
            sensor_name: self._flatten_helper(sensor_tensor[: self.step])
            for sensor_name, sensor_tensor in self.observations.items()
        }
        flat_prev_actions = self._flatten_helper(self.prev_actions[: self.step])
        flat_masks = self._flatten_helper(self.masks[: self.step])
        flat_expert_actions = self._flatten_helper(self.expert_actions[: self.step])
        flat_memory_tokens = self._flatten_helper(self.memory_tokens[: self.step])
        flat_memory_mask = self._flatten_helper(self.memory_mask[: self.step])

        for indices in mini_batches:
            if indices.numel() == 0:
                continue
            yield (
                {sensor_name: sensor_tensor[indices] for sensor_name, sensor_tensor in flat_observations.items()},
                flat_prev_actions[indices],
                flat_masks[indices],
                flat_expert_actions[indices],
                flat_memory_tokens[indices],
                flat_memory_mask[indices],
            )


@baseline_registry.register_trainer(name="OmniLongBCTrainer")
class OmniLongBCTrainer(OmniLongPPOTrainer):
    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        self.agent: Optional[BehaviorCloning] = None

    def _setup_actor_critic_agent(self, _algo_cfg: Config, observation_space=None) -> None:
        logger.add_filehandler(self.config.LOG_FILE)
        if observation_space is None:
            observation_space = self.envs.observation_spaces[0]
        action_space = self.envs.action_spaces[0]
        bc_cfg = self.config.RL.BC

        self.actor_critic = OmniLongBaselinePolicy.from_config(
            config=self.config,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.actor_critic.to(self.device)

        self.agent = BehaviorCloning(
            actor_critic=self.actor_critic,
            lr=bc_cfg.lr,
            eps=bc_cfg.eps,
            max_grad_norm=bc_cfg.max_grad_norm,
            bc_epoch=bc_cfg.bc_epoch,
            num_mini_batch=bc_cfg.num_mini_batch,
        )

    def _query_teacher_actions(self) -> torch.Tensor:
        action_names = ["get_teacher_action"] * self.envs.num_envs
        teacher_actions = self.envs.call(action_names)
        return torch.tensor(teacher_actions, dtype=torch.long, device=self.device).unsqueeze(-1)

    def _collect_bc_rollout_step(
        self,
        rollouts: OmniLongBCRolloutStorage,
        current_batch: Dict[str, torch.Tensor],
        prev_actions: torch.Tensor,
        masks: torch.Tensor,
        memory_tokens: torch.Tensor,
        memory_mask: torch.Tensor,
        current_episode_reward: torch.Tensor,
        running_episode_stats: Dict[str, torch.Tensor],
    ) -> Tuple[
        float,
        float,
        int,
        Dict[str, float],
        Dict[str, torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        pth_time = 0.0
        env_time = 0.0
        stage_times: Dict[str, float] = {}

        t_teacher = time.time()
        teacher_actions = self._query_teacher_actions()
        stage_times["collect.teacher_query"] = time.time() - t_teacher
        pth_time += stage_times["collect.teacher_query"]

        t_policy = time.time()
        with torch.no_grad():
            output = self.actor_critic(
                observations=current_batch,
                prev_actions=prev_actions,
                masks=masks,
                memory_tokens=memory_tokens,
                memory_mask=memory_mask,
            )
        next_memory_tokens = output.updated_memory_tokens
        next_memory_mask = output.updated_memory_mask
        stage_times["collect.policy_forward"] = time.time() - t_policy
        pth_time += stage_times["collect.policy_forward"]

        t_store = time.time()
        rollouts.insert(
            observations=current_batch,
            prev_actions=prev_actions,
            masks=masks,
            expert_actions=teacher_actions,
            memory_tokens=memory_tokens,
            memory_mask=memory_mask,
        )
        stage_times["collect.rollout_insert"] = time.time() - t_store
        pth_time += stage_times["collect.rollout_insert"]

        t_step = time.time()
        outputs = self.envs.step([int(action.item()) for action in teacher_actions])
        observations, rewards, dones, infos = [list(items) for items in zip(*outputs)]
        stage_times["collect.env_step"] = time.time() - t_step
        env_time += stage_times["collect.env_step"]

        t_batch = time.time()
        batch = batch_obs(observations, device=self.device)
        stage_times["collect.batch_obs"] = time.time() - t_batch

        t_tensor = time.time()
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float32,
            device=self.device,
        )
        batch = self.actor_critic.prepare_observations(batch, masks=next_masks)
        stage_times["collect.tensorize"] = time.time() - t_tensor

        t_episode_stats = time.time()
        current_episode_reward += rewards_tensor
        running_episode_stats["reward"] += (1.0 - next_masks) * current_episode_reward
        running_episode_stats["count"] += 1.0 - next_masks
        current_episode_reward *= next_masks
        stage_times["collect.episode_stats"] = time.time() - t_episode_stats

        t_metric_extract = time.time()
        extracted_metrics = self._extract_scalars_from_infos(infos)
        for key, values in extracted_metrics.items():
            metric_tensor = torch.tensor(values, dtype=torch.float32, device=self.device).unsqueeze(1)
            if key not in running_episode_stats:
                running_episode_stats[key] = torch.zeros_like(running_episode_stats["count"])
            running_episode_stats[key] += (1.0 - next_masks) * metric_tensor
        stage_times["collect.metric_extract"] = time.time() - t_metric_extract

        for stage_name, stage_value in stage_times.items():
            if stage_name != "collect.env_step":
                pth_time += stage_value

        return (
            pth_time,
            env_time,
            self.envs.num_envs,
            stage_times,
            batch,
            teacher_actions,
            next_masks,
            next_memory_tokens,
            next_memory_mask,
        )

    def _update_agent(
        self,
        bc_cfg: Config,
        rollouts: OmniLongBCRolloutStorage,
    ) -> Tuple[float, float, float, float, Dict[str, float]]:
        stage_times: Dict[str, float] = {}
        t_update = time.time()
        loss, ce_loss, entropy = self.agent.update(rollouts)
        stage_times["update.bc_update"] = time.time() - t_update

        t_after = time.time()
        rollouts.after_update()
        stage_times["update.after_update"] = time.time() - t_after

        pth_time = sum(stage_times.values())
        return pth_time, loss, ce_loss, entropy, stage_times

    def train(self) -> None:
        logger.info(f"config: {self.config}")
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)

        os.makedirs(self.config.MODEL_DIR, exist_ok=True)
        os.makedirs(self.config.CHECKPOINT_FOLDER, exist_ok=True)
        os.makedirs(self.config.TENSORBOARD_DIR, exist_ok=True)

        self.envs = construct_envs(self.config, get_env_class(self.config.ENV_NAME))
        env_observation_space = self.envs.observation_spaces[0]
        observation_space = OmniLongBaselinePolicy.build_policy_observation_space(
            self.config,
            env_observation_space,
        )
        bc_cfg = self.config.RL.BC
        self._setup_actor_critic_agent(bc_cfg, observation_space=observation_space)

        rollouts = OmniLongBCRolloutStorage(
            num_steps=bc_cfg.num_steps,
            num_envs=self.envs.num_envs,
            observation_space=observation_space,
            memory_size=self.config.RL.PPO.transformer_memory_size,
            memory_dim=self.config.RL.PPO.hidden_size,
        )
        rollouts.to(self.device)

        observations = self.envs.reset()
        current_batch = batch_obs(observations, device=self.device)
        current_batch = self.actor_critic.prepare_observations(current_batch, masks=None)
        prev_actions = torch.zeros(self.envs.num_envs, 1, device=self.device, dtype=torch.long)
        masks = torch.zeros(self.envs.num_envs, 1, device=self.device, dtype=torch.float32)
        memory_tokens, memory_mask = self.actor_critic.init_memory(
            batch_size=self.envs.num_envs,
            device=self.device,
        )

        current_episode_reward = torch.zeros(self.envs.num_envs, 1, device=self.device)
        running_episode_stats: Dict[str, torch.Tensor] = {
            "count": torch.zeros(self.envs.num_envs, 1, device=self.device),
            "reward": torch.zeros(self.envs.num_envs, 1, device=self.device),
        }
        window_episode_stats: DefaultDict[str, deque] = defaultdict(
            lambda: deque(maxlen=bc_cfg.reward_window_size)
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
        wandb_run = WandbRun(self.config, run_type="bc")

        with TensorboardWriter(self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs) as writer:
            for update in range(self.config.NUM_UPDATES):
                if bc_cfg.use_linear_lr_decay:
                    lr_scheduler.step()

                interval_stage_times: DefaultDict[str, float] = defaultdict(float)
                for _ in range(bc_cfg.num_steps):
                    (
                        delta_pth_time,
                        delta_env_time,
                        delta_steps,
                        collect_stage_times,
                        current_batch,
                        prev_actions,
                        masks,
                        memory_tokens,
                        memory_mask,
                    ) = self._collect_bc_rollout_step(
                        rollouts=rollouts,
                        current_batch=current_batch,
                        prev_actions=prev_actions,
                        masks=masks,
                        memory_tokens=memory_tokens,
                        memory_mask=memory_mask,
                        current_episode_reward=current_episode_reward,
                        running_episode_stats=running_episode_stats,
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps += delta_steps
                    for key, value in collect_stage_times.items():
                        interval_stage_times[key] += value

                delta_pth_time, loss, ce_loss, entropy, update_stage_times = self._update_agent(
                    bc_cfg,
                    rollouts,
                )
                pth_time += delta_pth_time
                for key, value in update_stage_times.items():
                    interval_stage_times[key] += value

                for key, value in running_episode_stats.items():
                    window_episode_stats[key].append(value.clone())

                deltas = {
                    key: ((values[-1] - values[0]).sum().item() if len(values) > 1 else values[0].sum().item())
                    for key, values in window_episode_stats.items()
                }
                completed_episodes = float(deltas.get("count", 0.0))
                deltas["count"] = max(1.0, completed_episodes)

                writer.add_scalar("Metrics/reward", deltas.get("reward", 0.0) / deltas["count"], count_steps)
                writer.add_scalar("Metrics/completed_episodes", completed_episodes, count_steps)
                for metric_name, metric_value in deltas.items():
                    if metric_name in {"reward", "count"}:
                        continue
                    writer.add_scalar(f"Metrics/{metric_name}", metric_value / deltas["count"], count_steps)

                writer.add_scalar("BehaviorCloning/loss", loss, count_steps)
                writer.add_scalar("BehaviorCloning/ce_loss", ce_loss, count_steps)
                writer.add_scalar("BehaviorCloning/entropy", entropy, count_steps)
                writer.add_scalar("BehaviorCloning/lr", self.agent.optimizer.param_groups[0]["lr"], count_steps)

                wandb_payload: Dict[str, float] = {
                    "Metrics/reward": float(deltas.get("reward", 0.0) / deltas["count"]),
                    "Metrics/completed_episodes": float(completed_episodes),
                    "BehaviorCloning/loss": float(loss),
                    "BehaviorCloning/ce_loss": float(ce_loss),
                    "BehaviorCloning/entropy": float(entropy),
                    "BehaviorCloning/lr": float(self.agent.optimizer.param_groups[0]["lr"]),
                    "System/fps": float(count_steps / max(1e-6, time.time() - t_start)),
                    "System/env_time": float(env_time),
                    "System/pth_time": float(pth_time),
                    "System/frames": float(count_steps),
                    "System/update": float(update),
                }
                for metric_name, metric_value in deltas.items():
                    if metric_name in {"reward", "count"}:
                        continue
                    wandb_payload[f"Metrics/{metric_name}"] = float(metric_value / deltas["count"])
                wandb_run.log(wandb_payload, step=count_steps)

                if update > 0 and update % self.config.LOG_INTERVAL == 0:
                    logger.info(
                        "update: {}\tfps: {:.3f}".format(
                            update,
                            count_steps / max(1e-6, time.time() - t_start),
                        )
                    )
                    logger.info(
                        "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\tframes: {}".format(
                            update,
                            env_time,
                            pth_time,
                            count_steps,
                        )
                    )
                    logger.info(
                        "Average window size: {}  completed_episodes: {:.0f}  {}".format(
                            len(window_episode_stats.get("count", [])),
                            completed_episodes,
                            "  ".join(
                                f"{key}: {value / deltas['count']:.3f}"
                                for key, value in deltas.items()
                                if key != "count"
                            ),
                        )
                    )
                    logger.info(
                        "BC stats: loss={:.4f} ce_loss={:.4f} entropy={:.4f}".format(
                            loss,
                            ce_loss,
                            entropy,
                        )
                    )

                if update % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        extra_state={
                            "step": count_steps,
                            "update": update,
                            "trainer": "bc",
                        },
                    )
                    count_checkpoints += 1

            self.envs.close()
        wandb_run.finish()


__all__ = ["OmniLongBCTrainer"]
