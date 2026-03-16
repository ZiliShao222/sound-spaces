#!/usr/bin/env python3

import torch.nn as nn


_REMOVAL_MESSAGE = (
    "The previous omni-long PPO implementation has been removed. "
    "Rewrite `ss_baselines/omni_long/ppo/audio_cnn.py` before using OmniLongAudioCNN."
)


class OmniLongAudioCNN(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        raise NotImplementedError(_REMOVAL_MESSAGE)
