#!/usr/bin/env python3

__all__ = ["OmniLongBCTrainer", "OmniLongPPOTrainer", "OmniLongDDPPOTrainer"]


def __getattr__(name):
    if name == "OmniLongBCTrainer":
        from ss_baselines.omni_long.bc.bc_trainer import OmniLongBCTrainer

        return OmniLongBCTrainer
    if name == "OmniLongPPOTrainer":
        from ss_baselines.omni_long.ppo.ppo_trainer import OmniLongPPOTrainer

        return OmniLongPPOTrainer
    if name == "OmniLongDDPPOTrainer":
        from ss_baselines.omni_long.ppo.ddppo_trainer import OmniLongDDPPOTrainer

        return OmniLongDDPPOTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
