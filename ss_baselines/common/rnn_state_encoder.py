#!/usr/bin/env python3

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class RNNStateEncoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        rnn_type: str = "GRU",
    ) -> None:
        super().__init__()
        self._hidden_size = int(hidden_size)
        self._num_layers = int(num_layers)
        self._rnn_type = str(rnn_type).upper()

        if self._rnn_type not in {"GRU", "LSTM"}:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

        rnn_cls = nn.GRU if self._rnn_type == "GRU" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=int(input_size),
            hidden_size=self._hidden_size,
            num_layers=self._num_layers,
        )

        for name, param in self.rnn.named_parameters():
            if "weight" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0)

    @property
    def num_recurrent_layers(self) -> int:
        if self._rnn_type == "LSTM":
            return int(self._num_layers * 2)
        return int(self._num_layers)

    def _unpack_hidden(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if hidden_states.dim() != 3:
            raise RuntimeError(
                "Expected hidden_states with shape (N, L, H), "
                f"got shape={tuple(hidden_states.shape)}"
            )

        hidden_states = hidden_states.transpose(0, 1).contiguous()
        if self._rnn_type == "LSTM":
            expected_layers = self._num_layers * 2
            if hidden_states.size(0) != expected_layers:
                raise RuntimeError(
                    "LSTM hidden_states layer mismatch: "
                    f"expected {expected_layers}, got {hidden_states.size(0)}"
                )
            h, c = torch.chunk(hidden_states, 2, dim=0)
            return h.contiguous(), c.contiguous()

        if hidden_states.size(0) != self._num_layers:
            raise RuntimeError(
                "GRU hidden_states layer mismatch: "
                f"expected {self._num_layers}, got {hidden_states.size(0)}"
            )
        return hidden_states, None

    def _pack_hidden(self, hidden: torch.Tensor, cell: Optional[torch.Tensor]) -> torch.Tensor:
        if self._rnn_type == "LSTM":
            assert cell is not None
            hidden = torch.cat([hidden, cell], dim=0)
        return hidden.transpose(0, 1).contiguous()

    def _mask_hidden(
        self,
        hidden: torch.Tensor,
        cell: Optional[torch.Tensor],
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        mask = masks.view(1, -1, 1)
        hidden = hidden * mask
        if cell is not None:
            cell = cell * mask
        return hidden, cell

    def single_forward(
        self,
        x: torch.Tensor,
        hidden_states: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden, cell = self._unpack_hidden(hidden_states)
        hidden, cell = self._mask_hidden(hidden, cell, masks)

        x = x.unsqueeze(0)
        if self._rnn_type == "LSTM":
            output, (hidden, cell) = self.rnn(x, (hidden, cell))
        else:
            output, hidden = self.rnn(x, hidden)

        output = output.squeeze(0)
        hidden_states = self._pack_hidden(hidden, cell)
        return output, hidden_states

    def seq_forward(
        self,
        x: torch.Tensor,
        hidden_states: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_envs = hidden_states.size(0)
        time_steps = int(x.size(0) // num_envs)

        x = x.view(time_steps, num_envs, x.size(1))
        masks = masks.view(time_steps, num_envs)

        hidden, cell = self._unpack_hidden(hidden_states)
        outputs = []
        for t in range(time_steps):
            hidden, cell = self._mask_hidden(hidden, cell, masks[t])
            step_input = x[t : t + 1]
            if self._rnn_type == "LSTM":
                step_output, (hidden, cell) = self.rnn(step_input, (hidden, cell))
            else:
                step_output, hidden = self.rnn(step_input, hidden)
            outputs.append(step_output)

        output = torch.cat(outputs, dim=0).view(time_steps * num_envs, self._hidden_size)
        hidden_states = self._pack_hidden(hidden, cell)
        return output, hidden_states

    def forward(
        self,
        x: torch.Tensor,
        hidden_states: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.size(0) == hidden_states.size(0):
            return self.single_forward(x, hidden_states, masks)
        return self.seq_forward(x, hidden_states, masks)
