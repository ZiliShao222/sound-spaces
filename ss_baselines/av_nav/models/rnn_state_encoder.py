#!/usr/bin/env python3

import torch
import torch.nn as nn

from habitat_baselines.rl.models.rnn_state_encoder import build_rnn_state_encoder


class RNNStateEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        rnn_type: str = "GRU",
    ):
        super().__init__()
        self.encoder = build_rnn_state_encoder(
            input_size=input_size,
            hidden_size=hidden_size,
            rnn_type=rnn_type,
            num_layers=num_layers,
        )

    @property
    def num_recurrent_layers(self):
        return self.encoder.num_recurrent_layers

    def forward(self, x, hidden_states, masks):
        with torch.backends.cudnn.flags(enabled=False):
            return self.encoder(
                x,
                hidden_states.contiguous(),
                masks.bool(),
            )
