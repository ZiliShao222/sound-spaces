#!/usr/bin/env python3

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim


EPS_PPO = 1e-5


class PPO(nn.Module):
    def __init__(
        self,
        actor_critic: nn.Module,
        clip_param: float,
        ppo_epoch: int,
        num_mini_batch: int,
        value_loss_coef: float,
        entropy_coef: float,
        lr: float | None = None,
        eps: float | None = None,
        max_grad_norm: float | None = None,
        use_clipped_value_loss: bool = True,
        use_normalized_advantage: bool = True,
    ) -> None:
        super().__init__()
        self.actor_critic = actor_critic
        self.clip_param = float(clip_param)
        self.ppo_epoch = int(ppo_epoch)
        self.num_mini_batch = int(num_mini_batch)
        self.value_loss_coef = float(value_loss_coef)
        self.entropy_coef = float(entropy_coef)
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = bool(use_clipped_value_loss)
        self.use_normalized_advantage = bool(use_normalized_advantage)
        self.device = next(actor_critic.parameters()).device
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr, eps=eps)

    def forward(self, *x, **kwargs):
        raise NotImplementedError

    def get_advantages(self, rollouts) -> torch.Tensor:
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        if not self.use_normalized_advantage:
            return advantages
        return (advantages - advantages.mean()) / (advantages.std() + EPS_PPO)

    def _compute_losses(
        self,
        values: torch.Tensor,
        action_log_probs: torch.Tensor,
        dist_entropy: torch.Tensor,
        old_action_log_probs_batch: torch.Tensor,
        adv_targ: torch.Tensor,
        value_preds_batch: torch.Tensor,
        return_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ratio = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = ratio * adv_targ
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
        action_loss = -torch.min(surr1, surr2).mean()

        if self.use_clipped_value_loss:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
                -self.clip_param,
                self.clip_param,
            )
            value_losses = (values - return_batch).pow(2)
            value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
            value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = 0.5 * (return_batch - values).pow(2).mean()

        entropy_loss = dist_entropy.mean()
        return value_loss, action_loss, entropy_loss

    def before_backward(self, loss: torch.Tensor) -> None:
        pass

    def after_backward(self, loss: torch.Tensor) -> None:
        pass

    def before_step(self) -> None:
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

    def after_step(self) -> None:
        pass

    def update(self, rollouts):
        advantages = self.get_advantages(rollouts)

        value_loss_epoch = 0.0
        action_loss_epoch = 0.0
        dist_entropy_epoch = 0.0

        for _ in range(self.ppo_epoch):
            data_generator = rollouts.recurrent_generator(advantages, self.num_mini_batch)
            for sample in data_generator:
                (
                    obs_batch,
                    actions_batch,
                    prev_actions_batch,
                    value_preds_batch,
                    return_batch,
                    masks_batch,
                    old_action_log_probs_batch,
                    adv_targ,
                    memory_tokens_batch,
                    memory_mask_batch,
                ) = sample

                (
                    values,
                    action_log_probs,
                    dist_entropy,
                    _,
                    _,
                    _,
                ) = self.actor_critic.evaluate_actions(
                    observations=obs_batch,
                    action=actions_batch,
                    prev_actions=prev_actions_batch,
                    masks=masks_batch,
                    memory_tokens=memory_tokens_batch,
                    memory_mask=memory_mask_batch,
                )

                value_loss, action_loss, entropy_loss = self._compute_losses(
                    values=values,
                    action_log_probs=action_log_probs,
                    dist_entropy=dist_entropy,
                    old_action_log_probs_batch=old_action_log_probs_batch,
                    adv_targ=adv_targ,
                    value_preds_batch=value_preds_batch,
                    return_batch=return_batch,
                )

                total_loss = (
                    self.value_loss_coef * value_loss
                    + action_loss
                    - self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                self.before_backward(total_loss)
                total_loss.backward()
                self.after_backward(total_loss)
                self.before_step()
                self.optimizer.step()
                self.after_step()

                value_loss_epoch += float(value_loss.item())
                action_loss_epoch += float(action_loss.item())
                dist_entropy_epoch += float(entropy_loss.item())

        num_updates = max(1, self.ppo_epoch * self.num_mini_batch)
        value_loss_epoch /= num_updates
        action_loss_epoch /= num_updates
        dist_entropy_epoch /= num_updates
        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch


__all__ = ["PPO"]
