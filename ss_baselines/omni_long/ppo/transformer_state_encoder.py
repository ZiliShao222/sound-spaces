#!/usr/bin/env python3

import torch.nn as nn


_REMOVAL_MESSAGE = (
    "The previous omni-long PPO implementation has been removed. "
    "Rewrite `ss_baselines/omni_long/ppo/transformer_state_encoder.py` before using TransformerStateEncoder."
)


class TransformerStateEncoder(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError(_REMOVAL_MESSAGE)
