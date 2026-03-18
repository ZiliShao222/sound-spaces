#!/usr/bin/env python3

from __future__ import annotations

import os
import time
import random
from collections import defaultdict, deque
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, logger

from ss_baselines.common.base_trainer import BaseRLTrainer
from ss_baselines.common.baseline_registry import baseline_registry
from ss_baselines.common.env_utils import construct_envs
from ss_baselines.common.environments import get_env_class
from ss_baselines.common.tensorboard_utils import TensorboardWriter
from ss_baselines.common.utils import batch_obs, linear_decay
from ss_baselines.omni_long.ppo.policy import OmniLongBaselinePolicy
from ss_baselines.omni_long.ppo.ppo import PPO


class OmniLongRolloutStorage:
    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        observation_space,
        action_space,
        memory_size: int,
        memory_dim: int,
    ) -> None:
        self.num_steps = int(num_steps)
        self.num_envs = int(num_envs)
        self.step = 0
        self.observations: Dict[str, torch.Tensor] = {}
        for sensor_name, space in observation_space.spaces.items():
            self.observations[sensor_name] = torch.zeros(
                self.num_steps + 1,
                self.num_envs,
                *space.shape,
            )

        self.memory_tokens = torch.zeros(
            self.num_steps + 1,
            self.num_envs,
            int(memory_size),
            int(memory_dim),
        )
        self.memory_mask = torch.zeros(
            self.num_steps + 1,
            self.num_envs,
            int(memory_size),
            dtype=torch.bool,
        )
        self.rewards = torch.zeros(self.num_steps, self.num_envs, 1)
        self.value_preds = torch.zeros(self.num_steps + 1, self.num_envs, 1)
        self.returns = torch.zeros(self.num_steps + 1, self.num_envs, 1)
        self.action_log_probs = torch.zeros(self.num_steps, self.num_envs, 1)
        self.actions = torch.zeros(self.num_steps, self.num_envs, 1, dtype=torch.long)
        self.prev_actions = torch.zeros(self.num_steps + 1, self.num_envs, 1, dtype=torch.long)
        self.masks = torch.ones(self.num_steps + 1, self.num_envs, 1)
        self._action_space = action_space

    def to(self, device: torch.device) -> None:
        for sensor in self.observations:
            self.observations[sensor] = self.observations[sensor].to(device)
        self.memory_tokens = self.memory_tokens.to(device)
        self.memory_mask = self.memory_mask.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.prev_actions = self.prev_actions.to(device)
        self.masks = self.masks.to(device)

    def insert(
        self,
        observations: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        action_log_probs: torch.Tensor,
        value_preds: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
        memory_tokens: torch.Tensor,
        memory_mask: torch.Tensor,
    ) -> None:
        if self.step >= self.num_steps:
            raise RuntimeError(
                f"RolloutStorage overflow: step={self.step}, num_steps={self.num_steps}"
            )
        next_index = self.step + 1
        for sensor in observations:
            self.observations[sensor][next_index].copy_(observations[sensor])
        self.actions[self.step].copy_(actions)
        self.prev_actions[next_index].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[next_index].copy_(masks)
        self.memory_tokens[next_index].copy_(memory_tokens)
        self.memory_mask[next_index].copy_(memory_mask)
        self.step = next_index

    def after_update(self) -> None:
        for sensor in self.observations:
            self.observations[sensor][0].copy_(self.observations[sensor][self.step])
        self.prev_actions[0].copy_(self.prev_actions[self.step])
        self.masks[0].copy_(self.masks[self.step])
        self.memory_tokens[0].copy_(self.memory_tokens[self.step])
        self.memory_mask[0].copy_(self.memory_mask[self.step])
        self.step = 0

    def compute_returns(
        self,
        next_value: torch.Tensor,
        use_gae: bool,
        gamma: float,
        tau: float,
    ) -> None:
        self.value_preds[self.step].copy_(next_value)
        if use_gae:
            gae = 0.0
            for step in reversed(range(self.step)):
                delta = (
                    self.rewards[step]
                    + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                    - self.value_preds[step]
                )
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[self.step].copy_(next_value)
            for step in reversed(range(self.step)):
                self.returns[step] = (
                    self.returns[step + 1] * gamma * self.masks[step + 1]
                    + self.rewards[step]
                )

    @staticmethod
    def _flatten_helper(t: int, n: int, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(t * n, *tensor.size()[2:])

    def recurrent_generator(self, advantages: torch.Tensor, num_mini_batch: int):
        num_processes = self.num_envs
        assert num_processes >= num_mini_batch, (
            "Trainer requires the number of processes ({}) to be >= num_mini_batch ({})".format(
                num_processes,
                num_mini_batch,
            )
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes, device=advantages.device)

        for start_ind in range(0, num_processes, num_envs_per_batch):
            observations_batch: DefaultDict[str, List[torch.Tensor]] = defaultdict(list)
            actions_batch: List[torch.Tensor] = []
            prev_actions_batch: List[torch.Tensor] = []
            value_preds_batch: List[torch.Tensor] = []
            return_batch: List[torch.Tensor] = []
            masks_batch: List[torch.Tensor] = []
            old_action_log_probs_batch: List[torch.Tensor] = []
            adv_targ: List[torch.Tensor] = []
            memory_tokens_batch: List[torch.Tensor] = []
            memory_mask_batch: List[torch.Tensor] = []

            for offset in range(num_envs_per_batch):
                ind = int(perm[start_ind + offset].item())
                for sensor in self.observations:
                    observations_batch[sensor].append(self.observations[sensor][: self.step, ind])
                actions_batch.append(self.actions[: self.step, ind])
                prev_actions_batch.append(self.prev_actions[: self.step, ind])
                value_preds_batch.append(self.value_preds[: self.step, ind])
                return_batch.append(self.returns[: self.step, ind])
                masks_batch.append(self.masks[: self.step, ind])
                old_action_log_probs_batch.append(self.action_log_probs[: self.step, ind])
                adv_targ.append(advantages[: self.step, ind])
                memory_tokens_batch.append(self.memory_tokens[: self.step, ind])
                memory_mask_batch.append(self.memory_mask[: self.step, ind])

            T, N = self.step, num_envs_per_batch
            for sensor in observations_batch:
                observations_batch[sensor] = torch.stack(observations_batch[sensor], dim=1)
                observations_batch[sensor] = self._flatten_helper(T, N, observations_batch[sensor])

            actions_batch = self._flatten_helper(T, N, torch.stack(actions_batch, dim=1))
            prev_actions_batch = self._flatten_helper(T, N, torch.stack(prev_actions_batch, dim=1))
            value_preds_batch = self._flatten_helper(T, N, torch.stack(value_preds_batch, dim=1))
            return_batch = self._flatten_helper(T, N, torch.stack(return_batch, dim=1))
            masks_batch = self._flatten_helper(T, N, torch.stack(masks_batch, dim=1))
            old_action_log_probs_batch = self._flatten_helper(
                T,
                N,
                torch.stack(old_action_log_probs_batch, dim=1),
            )
            adv_targ = self._flatten_helper(T, N, torch.stack(adv_targ, dim=1))
            memory_tokens_batch = self._flatten_helper(T, N, torch.stack(memory_tokens_batch, dim=1))
            memory_mask_batch = self._flatten_helper(T, N, torch.stack(memory_mask_batch, dim=1))

            yield (
                observations_batch,
                actions_batch,
                prev_actions_batch,
                value_preds_batch,
                return_batch,
                masks_batch,
                old_action_log_probs_batch,
                adv_targ,
                memory_tokens_batch,
                memory_mask_batch,
            )


@baseline_registry.register_trainer(name="OmniLongPPOTrainer")
class OmniLongPPOTrainer(BaseRLTrainer):
    supported_tasks = ["OmniLongSemanticAudioNav"]
    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision"}

    def __init__(self, config: Optional[Config] = None):
        super().__init__(config)
        self.actor_critic: Optional[OmniLongBaselinePolicy] = None
        self.agent: Optional[PPO] = None
        self.envs = None
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    @classmethod
    def _extract_scalars_from_info(cls, info: Dict[str, Any]) -> Dict[str, float]:
        result: Dict[str, float] = {}
        for key, value in info.items():
            if key in cls.METRICS_BLACKLIST:
                continue
            if value is None:
                continue
            if isinstance(value, dict):
                nested = cls._extract_scalars_from_info(value)
                for nested_key, nested_value in nested.items():
                    full_key = f"{key}.{nested_key}"
                    if full_key not in cls.METRICS_BLACKLIST:
                        result[full_key] = nested_value
            elif np.size(value) == 1 and not isinstance(value, str):
                scalar = np.asarray(value)
                if scalar.size != 1:
                    continue
                if not (
                    np.issubdtype(scalar.dtype, np.number)
                    or np.issubdtype(scalar.dtype, np.bool_)
                ):
                    continue
                result[key] = float(scalar.reshape(()).item())
        return result

    @classmethod
    def _extract_scalars_from_infos(cls, infos: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        results: DefaultDict[str, List[float]] = defaultdict(list)
        for info in infos:
            for key, value in cls._extract_scalars_from_info(info).items():
                results[key].append(float(value))
        return results

    def _setup_actor_critic_agent(self, ppo_cfg: Config, observation_space=None) -> None:
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

    def save_checkpoint(self, file_name: str, extra_state=None) -> None:
        os.makedirs(self.config.CHECKPOINT_FOLDER, exist_ok=True)
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state
        torch.save(checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name))

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        return torch.load(checkpoint_path, *args, **kwargs)

    def _collect_rollout_step(
        self,
        rollouts: OmniLongRolloutStorage,
        current_episode_reward: torch.Tensor,
        running_episode_stats: Dict[str, torch.Tensor],
    ) -> Tuple[float, float, int, Dict[str, float]]:
        pth_time = 0.0
        env_time = 0.0
        stage_times: Dict[str, float] = {}

        t_sample_action = time.time()
        with torch.no_grad():
            step_observation = {k: v[rollouts.step] for k, v in rollouts.observations.items()}
            (
                predicted_values,
                actions,
                action_log_probs,
                next_memory_tokens,
                next_memory_mask,
                _,
            ) = self.actor_critic.act(
                observations=step_observation,
                prev_actions=rollouts.prev_actions[rollouts.step],
                masks=rollouts.masks[rollouts.step],
                memory_tokens=rollouts.memory_tokens[rollouts.step],
                memory_mask=rollouts.memory_mask[rollouts.step],
                deterministic=False,
            )
        stage_times["collect.policy_act"] = time.time() - t_sample_action
        pth_time += stage_times["collect.policy_act"]

        t_step_env = time.time()
        outputs = self.envs.step([int(a.item()) for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]
        stage_times["collect.env_step"] = time.time() - t_step_env
        env_time += stage_times["collect.env_step"]

        t_batch_obs = time.time()
        batch = batch_obs(observations, device=self.device)
        stage_times["collect.batch_obs"] = time.time() - t_batch_obs

        t_tensorize = time.time()
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float32,
            device=self.device,
        )
        batch = self.actor_critic.prepare_observations(batch, masks=masks)
        stage_times["collect.tensorize"] = time.time() - t_tensorize

        t_episode_stats = time.time()
        current_episode_reward += rewards_tensor
        running_episode_stats["reward"] += (1.0 - masks) * current_episode_reward
        running_episode_stats["count"] += 1.0 - masks
        current_episode_reward *= masks
        stage_times["collect.episode_stats"] = time.time() - t_episode_stats

        t_metric_extract = time.time()
        extracted_metrics = self._extract_scalars_from_infos(infos)
        for key, metric_values in extracted_metrics.items():
            metric_tensor = torch.tensor(metric_values, dtype=torch.float32, device=self.device).unsqueeze(1)
            if key not in running_episode_stats:
                running_episode_stats[key] = torch.zeros_like(running_episode_stats["count"])
            running_episode_stats[key] += (1.0 - masks) * metric_tensor
        stage_times["collect.metric_extract"] = time.time() - t_metric_extract

        t_rollout_insert = time.time()
        rollouts.insert(
            observations=batch,
            actions=actions,
            action_log_probs=action_log_probs,
            value_preds=predicted_values,
            rewards=rewards_tensor,
            masks=masks,
            memory_tokens=next_memory_tokens,
            memory_mask=next_memory_mask,
        )
        stage_times["collect.rollout_insert"] = time.time() - t_rollout_insert

        for stage_name, stage_value in stage_times.items():
            if stage_name != "collect.env_step":
                pth_time += stage_value
        return pth_time, env_time, self.envs.num_envs, stage_times

    def _update_agent(self, ppo_cfg: Config, rollouts: OmniLongRolloutStorage):
        stage_times: Dict[str, float] = {}
        t_update_model = time.time()

        t_bootstrap = time.time()
        with torch.no_grad():
            last_observation = {k: v[rollouts.step] for k, v in rollouts.observations.items()}
            next_value = self.actor_critic.get_value(
                observations=last_observation,
                prev_actions=rollouts.prev_actions[rollouts.step],
                masks=rollouts.masks[rollouts.step],
                memory_tokens=rollouts.memory_tokens[rollouts.step],
                memory_mask=rollouts.memory_mask[rollouts.step],
            ).detach()
        stage_times["update.bootstrap_value"] = time.time() - t_bootstrap

        t_returns = time.time()
        rollouts.compute_returns(next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau)
        stage_times["update.compute_returns"] = time.time() - t_returns

        t_ppo_update = time.time()
        value_loss, action_loss, dist_entropy = self.agent.update(rollouts)
        stage_times["update.ppo_update"] = time.time() - t_ppo_update

        t_after_update = time.time()
        rollouts.after_update()
        stage_times["update.after_update"] = time.time() - t_after_update

        total_update_time = time.time() - t_update_model
        accounted_time = sum(stage_times.values())
        if total_update_time > accounted_time:
            stage_times["update.overhead"] = total_update_time - accounted_time

        return total_update_time, value_loss, action_loss, dist_entropy, stage_times

    @staticmethod
    def _format_stage_timing_breakdown(stage_times: Dict[str, float]) -> str:
        ordered_keys = [
            "collect.policy_act",
            "collect.env_step",
            "collect.batch_obs",
            "collect.tensorize",
            "collect.episode_stats",
            "collect.metric_extract",
            "collect.rollout_insert",
            "update.bootstrap_value",
            "update.compute_returns",
            "update.ppo_update",
            "update.after_update",
            "update.overhead",
        ]
        parts: List[str] = []
        for key in ordered_keys:
            if key in stage_times:
                short_key = key.replace("collect.", "").replace("update.", "")
                parts.append(f"{short_key}: {stage_times[key]:.3f}s")
        for key in sorted(stage_times.keys()):
            if key not in ordered_keys:
                parts.append(f"{key}: {stage_times[key]:.3f}s")
        return "  ".join(parts)

    @staticmethod
    def _format_stage_timing_ratio(stage_times: Dict[str, float]) -> str:
        total_time = sum(stage_times.values())
        if total_time <= 0.0:
            return "n/a"

        ordered_keys = [
            "collect.policy_act",
            "collect.env_step",
            "collect.batch_obs",
            "collect.tensorize",
            "collect.episode_stats",
            "collect.metric_extract",
            "collect.rollout_insert",
            "update.bootstrap_value",
            "update.compute_returns",
            "update.ppo_update",
            "update.after_update",
            "update.overhead",
        ]
        parts: List[str] = []
        for key in ordered_keys:
            if key in stage_times:
                short_key = key.replace("collect.", "").replace("update.", "")
                parts.append(f"{short_key}: {100.0 * stage_times[key] / total_time:.1f}%")
        for key in sorted(stage_times.keys()):
            if key not in ordered_keys:
                parts.append(f"{key}: {100.0 * stage_times[key] / total_time:.1f}%")
        return "  ".join(parts)

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
        action_space = self.envs.action_spaces[0]
        ppo_cfg = self.config.RL.PPO
        self._setup_actor_critic_agent(ppo_cfg, observation_space=observation_space)

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

        with TensorboardWriter(self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs) as writer:
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

                for key, value in running_episode_stats.items():
                    window_episode_stats[key].append(value.clone())

                deltas = {
                    key: ((values[-1] - values[0]).sum().item() if len(values) > 1 else values[0].sum().item())
                    for key, values in window_episode_stats.items()
                }
                deltas["count"] = max(1.0, deltas.get("count", 1.0))

                writer.add_scalar("Metrics/reward", deltas.get("reward", 0.0) / deltas["count"], count_steps)
                for metric, value in deltas.items():
                    if metric in {"reward", "count"}:
                        continue
                    writer.add_scalar(f"Metrics/{metric}", value / deltas["count"], count_steps)

                writer.add_scalar("Policy/value_loss", value_loss, count_steps)
                writer.add_scalar("Policy/policy_loss", action_loss, count_steps)
                writer.add_scalar("Policy/entropy", dist_entropy, count_steps)
                writer.add_scalar("Policy/lr", self.agent.optimizer.param_groups[0]["lr"], count_steps)
                for stage_name, stage_value in interval_stage_times.items():
                    writer.add_scalar(f"Timing/{stage_name}", stage_value, count_steps)

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
                            "step": count_steps,
                            "update": update,
                        },
                    )
                    count_checkpoints += 1

            self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> Dict[str, float]:
        random.seed(self.config.SEED)
        np.random.seed(self.config.SEED)
        torch.manual_seed(self.config.SEED)

        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")
        if self.config.EVAL.USE_CKPT_CONFIG and ckpt_dict is not None:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        env_observation_space = self.envs.observation_spaces[0]
        observation_space = OmniLongBaselinePolicy.build_policy_observation_space(config, env_observation_space)
        self._setup_actor_critic_agent(config.RL.PPO, observation_space=observation_space)
        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic
        self.actor_critic.eval()

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)
        batch = self.actor_critic.prepare_observations(batch, masks=None)
        memory_tokens, memory_mask = self.actor_critic.init_memory(
            batch_size=self.envs.num_envs,
            device=self.device,
        )
        prev_actions = torch.zeros(self.envs.num_envs, 1, device=self.device, dtype=torch.long)
        masks = torch.ones(self.envs.num_envs, 1, device=self.device)

        stats_episodes: List[Dict[str, float]] = []
        while len(stats_episodes) < int(self.config.TEST_EPISODE_COUNT):
            with torch.no_grad():
                (
                    _,
                    actions,
                    _,
                    memory_tokens,
                    memory_mask,
                    _,
                ) = self.actor_critic.act(
                    observations=batch,
                    prev_actions=prev_actions,
                    masks=masks,
                    memory_tokens=memory_tokens,
                    memory_mask=memory_mask,
                    deterministic=True,
                )

            outputs = self.envs.step([int(a.item()) for a in actions])
            observations, _, dones, infos = [list(x) for x in zip(*outputs)]
            batch = batch_obs(observations, device=self.device)
            prev_actions = actions
            masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float32,
                device=self.device,
            )
            batch = self.actor_critic.prepare_observations(batch, masks=masks)

            for done, info in zip(dones, infos):
                if done:
                    stats_episodes.append(self._extract_scalars_from_info(info))
                    if len(stats_episodes) >= int(self.config.TEST_EPISODE_COUNT):
                        break

        aggregated: DefaultDict[str, List[float]] = defaultdict(list)
        for episode_stats in stats_episodes:
            for key, value in episode_stats.items():
                aggregated[key].append(float(value))
        means = {key: float(np.mean(values)) for key, values in aggregated.items() if len(values) > 0}

        if "reward" in means:
            writer.add_scalar(f"{config.EVAL.SPLIT}/reward", means["reward"], checkpoint_index)
        for metric, value in means.items():
            writer.add_scalar(f"{config.EVAL.SPLIT}/{metric}", value, checkpoint_index)

        logger.info(
            "Evaluation on split={} checkpoint={} episodes={} metrics={}".format(
                config.EVAL.SPLIT,
                checkpoint_path,
                len(stats_episodes),
                means,
            )
        )
        self.envs.close()
        return means


__all__ = ["OmniLongPPOTrainer"]
