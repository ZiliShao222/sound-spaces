#!/usr/bin/env python3


_REMOVAL_MESSAGE = (
    "The previous omni-long PPO implementation has been removed. "
    "Rewrite `ss_baselines/omni_long/ppo/policy.py` before using OmniLongBaselinePolicy."
)


class OmniLongBaselinePolicy:
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_REMOVAL_MESSAGE)

    @classmethod
    def from_config(cls, *args, **kwargs):
        raise NotImplementedError(_REMOVAL_MESSAGE)
