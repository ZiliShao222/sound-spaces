#!/usr/bin/env python3

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class BehaviorCloning(nn.Module):
    def __init__(
        self,
        actor_critic: nn.Module,
        lr: float,
        eps: float,
        max_grad_norm: float,
        bc_epoch: int,
        num_mini_batch: int,
    ) -> None:
        super().__init__()
        self.actor_critic = actor_critic
        self.max_grad_norm = float(max_grad_norm)
        self.bc_epoch = int(bc_epoch)
        self.num_mini_batch = int(num_mini_batch)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=float(lr), eps=float(eps))

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def _compute_losses(
        self,
        logits: torch.Tensor,
        expert_actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        target = expert_actions.squeeze(-1).long()
        ce_loss = F.cross_entropy(logits, target)
        entropy = torch.distributions.Categorical(logits=logits).entropy().mean()
        return ce_loss, ce_loss, entropy

    def update(self, rollouts) -> Tuple[float, float, float]:
        mean_loss = 0.0
        mean_ce_loss = 0.0
        mean_entropy = 0.0
        num_updates = 0

        for _ in range(self.bc_epoch):
            for sample in rollouts.feed_forward_generator(self.num_mini_batch):
                (
                    observations_batch,
                    prev_actions_batch,
                    masks_batch,
                    expert_actions_batch,
                    memory_tokens_batch,
                    memory_mask_batch,
                ) = sample

                output = self.actor_critic(
                    observations=observations_batch,
                    prev_actions=prev_actions_batch,
                    masks=masks_batch,
                    memory_tokens=memory_tokens_batch,
                    memory_mask=memory_mask_batch,
                )
                loss, ce_loss, entropy = self._compute_losses(output.logits, expert_actions_batch)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_loss += float(loss.item())
                mean_ce_loss += float(ce_loss.item())
                mean_entropy += float(entropy.item())
                num_updates += 1

        num_updates = max(1, num_updates)
        return (
            mean_loss / num_updates,
            mean_ce_loss / num_updates,
            mean_entropy / num_updates,
        )


__all__ = ["BehaviorCloning"]
