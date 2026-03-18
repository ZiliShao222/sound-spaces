#!/usr/bin/env python3

from __future__ import annotations

from typing import Tuple

import torch
import torch.distributed as distrib

from ss_baselines.omni_long.ppo.ppo import EPS_PPO, PPO


def distributed_mean_and_var(values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert distrib.is_initialized(), "Distributed training must be initialized"

    world_size = distrib.get_world_size()
    mean = values.mean()
    distrib.all_reduce(mean)
    mean /= world_size

    sq_diff = (values - mean).pow(2).mean()
    distrib.all_reduce(sq_diff)
    var = sq_diff / world_size

    return mean, var


class DecentralizedDistributedMixin:
    def _get_advantages_distributed(self, rollouts) -> torch.Tensor:
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        if not self.use_normalized_advantage:
            return advantages

        mean, var = distributed_mean_and_var(advantages)
        return (advantages - mean) / (var.sqrt() + EPS_PPO)

    def init_distributed(self, find_unused_params: bool = True) -> None:
        class Guard:
            def __init__(self, model, device):
                if torch.cuda.is_available():
                    self.ddp = torch.nn.parallel.DistributedDataParallel(
                        model,
                        device_ids=[device],
                        output_device=device,
                    )
                else:
                    self.ddp = torch.nn.parallel.DistributedDataParallel(model)

        self._ddp_hooks = Guard(self.actor_critic, self.device)
        self.get_advantages = self._get_advantages_distributed
        self.reducer = self._ddp_hooks.ddp.reducer
        self.find_unused_params = bool(find_unused_params)

    def before_backward(self, loss: torch.Tensor) -> None:
        super().before_backward(loss)
        if self.find_unused_params:
            self.reducer.prepare_for_backward([loss])
        else:
            self.reducer.prepare_for_backward([])


class DDPPO(DecentralizedDistributedMixin, PPO):
    pass
