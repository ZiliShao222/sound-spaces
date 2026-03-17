#!/usr/bin/env python3

__all__ = ["OmniLongPPOTrainer"]


def __getattr__(name):
    if name == "OmniLongPPOTrainer":
        from ss_baselines.omni_long.ppo.ppo_trainer import OmniLongPPOTrainer

        return OmniLongPPOTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
