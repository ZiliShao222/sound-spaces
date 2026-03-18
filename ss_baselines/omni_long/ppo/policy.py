#!/usr/bin/env python3

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from gym import spaces


def _to_2d_tensor(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 1:
        return x.unsqueeze(-1)
    if x.dim() > 2:
        return x.reshape(x.size(0), -1)
    return x


def _as_float_tensor(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if not torch.is_floating_point(x):
        x = x.float()
    return x


def _infer_space_shape(observation_space: spaces.Dict, key: str) -> Optional[Tuple[int, ...]]:
    if not hasattr(observation_space, "spaces"):
        return None
    space = observation_space.spaces.get(key)
    if space is None or not hasattr(space, "shape"):
        return None
    return tuple(int(dim) for dim in space.shape)


def _infer_first_available_shape(
    observation_space: spaces.Dict,
    keys: Iterable[str],
) -> Optional[Tuple[int, ...]]:
    for key in keys:
        shape = _infer_space_shape(observation_space, key)
        if shape is not None:
            return shape
    return None


def _copy_space_dict_without_keys(
    observation_space: spaces.Dict,
    excluded_keys: Iterable[str],
) -> spaces.Dict:
    excluded = {str(key) for key in excluded_keys}
    copied = {
        key: space
        for key, space in observation_space.spaces.items()
        if key not in excluded
    }
    return spaces.Dict(copied)


def _compute_torch_audio_spectrogram(
    audio_waveform: torch.Tensor,
    n_fft: int = 512,
    hop_length: int = 160,
    win_length: int = 400,
    pool_kernel: Tuple[int, int] = (4, 4),
) -> torch.Tensor:
    if audio_waveform.dim() == 2:
        audio_waveform = audio_waveform.unsqueeze(0)
    if audio_waveform.dim() != 3:
        raise RuntimeError(
            "audio waveform must have shape (B, C, T) or (C, T); "
            f"got shape={tuple(audio_waveform.shape)}"
        )

    if audio_waveform.size(1) > audio_waveform.size(2) and audio_waveform.size(-1) <= 4:
        audio_waveform = audio_waveform.transpose(1, 2).contiguous()

    batch_size, num_channels, num_samples = audio_waveform.shape
    waveform = audio_waveform.float().reshape(batch_size * num_channels, num_samples)
    window = torch.hann_window(win_length, device=waveform.device, dtype=waveform.dtype)
    stft = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    magnitude = torch.log1p(stft.abs())
    magnitude = F.avg_pool2d(
        magnitude.unsqueeze(1),
        kernel_size=pool_kernel,
        stride=pool_kernel,
    ).squeeze(1)
    freq_bins, time_bins = int(magnitude.size(-2)), int(magnitude.size(-1))
    magnitude = magnitude.reshape(batch_size, num_channels, freq_bins, time_bins)
    return magnitude.permute(0, 2, 3, 1).contiguous()


def _infer_processed_audio_shape(audio_shape: Tuple[int, ...]) -> Tuple[int, ...]:
    dummy_audio = torch.zeros((1, *audio_shape), dtype=torch.float32)
    processed = _compute_torch_audio_spectrogram(dummy_audio)
    return tuple(int(dim) for dim in processed.shape[1:])


def _masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = mask.float().unsqueeze(-1)
    denom = weight.sum(dim=1).clamp_min(1.0)
    return (tokens * weight).sum(dim=1) / denom


def _build_sinusoidal_position_encoding(
    seq_len: int,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    position = torch.arange(seq_len, device=device, dtype=dtype).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, hidden_size, 2, device=device, dtype=dtype)
        * (-torch.log(torch.tensor(10000.0, device=device, dtype=dtype)) / max(1, hidden_size))
    )
    pe = torch.zeros(seq_len, hidden_size, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


def _infer_num_actions(action_space: Any) -> int:
    if hasattr(action_space, "n"):
        return int(action_space.n)
    raise RuntimeError("OMAPolicyNet currently expects a discrete action space with attribute `n`.")


def _orthogonal_init(module: nn.Module, gain: float = 1.0) -> nn.Module:
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    return module


class MLPBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size),
        )
        self.net.apply(_orthogonal_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConvEncoder(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_size: int,
        base_channels: int = 32,
        normalize_uint8: bool = False,
    ) -> None:
        super().__init__()
        self.normalize_uint8 = bool(normalize_uint8)
        self.backbone = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(4, base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, base_channels * 2),
            nn.GELU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.GELU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Linear(base_channels * 4, output_size)
        self.backbone.apply(_orthogonal_init)
        _orthogonal_init(self.proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise RuntimeError(f"ConvEncoder expects 4D tensor, got shape={tuple(x.shape)}")
        if x.shape[-1] <= 4:
            x = x.permute(0, 3, 1, 2).contiguous()
        x = x.float()
        if self.normalize_uint8 and x.max().item() > 1.5:
            x = x / 255.0
        features = self.backbone(x).flatten(1)
        return self.proj(features)


class FrozenCLIPGoalEncoder(nn.Module):
    TEXT_MODALITY_INDEX = 0
    IMAGE_MODALITY_INDEX = 1

    def __init__(
        self,
        model_name: str,
        output_size: int,
    ) -> None:
        super().__init__()
        clip_model, _ = clip.load(model_name, device="cpu", jit=False)
        clip_model.eval()
        for parameter in clip_model.parameters():
            parameter.requires_grad_(False)

        visual_output_dim = int(getattr(clip_model.visual, "output_dim", 0))
        text_projection = getattr(clip_model, "text_projection", None)
        text_output_dim = int(text_projection.shape[-1]) if text_projection is not None else visual_output_dim
        clip_output_dim = max(visual_output_dim, text_output_dim)
        if clip_output_dim <= 0:
            raise RuntimeError("Failed to infer CLIP output dimension for OmniLong goal encoding.")
        if int(output_size) != clip_output_dim:
            raise RuntimeError(
                "FrozenCLIPGoalEncoder output dimension mismatch: "
                f"expected {int(output_size)}, got {clip_output_dim}."
            )

        self.clip_model = clip_model
        self.output_size = int(output_size)
        self.input_resolution = int(getattr(clip_model.visual, "input_resolution", 224))
        self.register_buffer(
            "clip_mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "clip_std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self._text_token_cache: Dict[str, torch.Tensor] = {}

    def _tokenize_goal_text_batch(
        self,
        goal_text_raw: torch.Tensor,
    ) -> torch.Tensor:
        if goal_text_raw.dim() != 2:
            raise RuntimeError(
                "goal_text_raw must have shape (M, T_bytes); "
                f"got shape={tuple(goal_text_raw.shape)}"
            )

        goal_text_raw = goal_text_raw.detach().round().clamp_(0, 255).to(dtype=torch.uint8)
        raw_bytes = goal_text_raw.cpu().numpy()
        token_rows = []
        for row in raw_bytes:
            end_index = int(np.argmax(row == 0)) if np.any(row == 0) else int(len(row))
            text = bytes(row[:end_index].tolist()).decode("utf-8", errors="ignore").strip()
            if text not in self._text_token_cache:
                self._text_token_cache[text] = clip.tokenize([text], truncate=True)[0].cpu()
            token_rows.append(self._text_token_cache[text])

        if not token_rows:
            return torch.zeros((0, 77), dtype=torch.long, device=goal_text_raw.device)
        return torch.stack(token_rows, dim=0).to(device=goal_text_raw.device, dtype=torch.long)

    def _preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        if images.dim() != 4:
            raise RuntimeError(
                f"FrozenCLIPGoalEncoder expects image tensor with 4 dims, got {tuple(images.shape)}"
            )
        if images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2).contiguous()
        images = images.float()
        if images.max().item() > 1.5:
            images = images / 255.0
        if images.shape[-1] != self.input_resolution or images.shape[-2] != self.input_resolution:
            images = F.interpolate(
                images,
                size=(self.input_resolution, self.input_resolution),
                mode="bilinear",
                align_corners=False,
            )
        images = (images - self.clip_mean) / self.clip_std
        return images

    def forward(
        self,
        goal_images: torch.Tensor,
        goal_text_raw: torch.Tensor,
        goal_modality: torch.Tensor,
    ) -> torch.Tensor:
        if goal_images.dim() != 5:
            raise RuntimeError(
                "goal_images must have shape (B, N, H, W, C); "
                f"got shape={tuple(goal_images.shape)}"
            )
        if goal_text_raw.dim() != 3:
            raise RuntimeError(
                "goal_text_raw must have shape (B, N, T_bytes); "
                f"got shape={tuple(goal_text_raw.shape)}"
            )
        if goal_modality.dim() != 3 or goal_modality.size(-1) < 2:
            raise RuntimeError(
                "goal_modality must have shape (B, N, 2); "
                f"got shape={tuple(goal_modality.shape)}"
            )

        self.clip_model.eval()

        batch_size, num_goals = int(goal_images.size(0)), int(goal_images.size(1))
        device = goal_images.device
        embeddings = torch.zeros(
            batch_size,
            num_goals,
            self.output_size,
            device=device,
            dtype=torch.float32,
        )

        text_mask = goal_modality[..., self.TEXT_MODALITY_INDEX] > 0.5
        image_mask = goal_modality[..., self.IMAGE_MODALITY_INDEX] > 0.5
        image_energy = goal_images.float().abs().sum(dim=(-1, -2, -3))
        image_available = image_energy > 1e-6
        image_mask = image_mask & image_available
        text_mask = text_mask | ((goal_modality[..., self.IMAGE_MODALITY_INDEX] > 0.5) & ~image_available)

        if text_mask.any():
            text_tokens = self._tokenize_goal_text_batch(goal_text_raw[text_mask])
            with torch.no_grad():
                text_embeddings = self.clip_model.encode_text(text_tokens).float()
                text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            embeddings[text_mask] = text_embeddings

        if image_mask.any():
            selected_images = self._preprocess_images(goal_images[image_mask])
            with torch.no_grad():
                image_embeddings = self.clip_model.encode_image(selected_images).float()
                image_embeddings = image_embeddings / image_embeddings.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            embeddings[image_mask] = image_embeddings

        return embeddings


class SoundEventDescriptorEncoder(nn.Module):
    def __init__(
        self,
        num_audio_classes: int,
        hidden_size: int,
        nhead: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_audio_classes = int(max(1, num_audio_classes))
        self.coord_proj = MLPBlock(4, hidden_size, hidden_size, dropout=dropout)
        self.class_embedding = nn.Embedding(self.num_audio_classes, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.output_proj.apply(_orthogonal_init)

    def forward(self, audio_doa: torch.Tensor) -> torch.Tensor:
        if audio_doa.dim() != 3 or audio_doa.size(-1) != 3:
            raise RuntimeError(
                "audio_doa must have shape (B, NUM_CLASSES, 3), "
                f"got shape={tuple(audio_doa.shape)}"
            )
        batch_size, num_classes, _ = audio_doa.shape
        if num_classes > self.num_audio_classes:
            raise RuntimeError(
                "audio_doa classes exceed configured encoder capacity: "
                f"num_classes={num_classes}, capacity={self.num_audio_classes}"
            )

        magnitude = torch.linalg.norm(audio_doa, dim=-1, keepdim=True)
        doa_features = torch.cat([audio_doa, magnitude], dim=-1)
        token_features = self.coord_proj(doa_features)

        class_indices = torch.arange(num_classes, device=audio_doa.device)
        class_features = self.class_embedding(class_indices).unsqueeze(0).expand(batch_size, -1, -1)
        tokens = token_features + class_features

        valid_mask = magnitude.squeeze(-1) > 1e-6
        safe_mask = valid_mask.clone()
        no_event_rows = ~safe_mask.any(dim=1)
        if no_event_rows.any():
            safe_mask[no_event_rows, 0] = True

        encoded = self.token_encoder(tokens, src_key_padding_mask=~safe_mask)
        pooled = _masked_mean(encoded, safe_mask)
        pooled = self.output_proj(pooled)
        if no_event_rows.any():
            pooled = pooled * (~no_event_rows).float().unsqueeze(-1)
        return pooled


class GoalContextEncoder(nn.Module):
    def __init__(
        self,
        goal_input_size: int,
        hidden_size: int,
        nhead: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.goal_proj = nn.Sequential(
            nn.LayerNorm(goal_input_size),
            nn.Linear(goal_input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.mode_embedding = nn.Embedding(2, hidden_size)
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.mode_film = nn.Linear(hidden_size, hidden_size * 2)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_size * 3 + 1),
            nn.Linear(hidden_size * 3 + 1, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.goal_proj.apply(_orthogonal_init)
        _orthogonal_init(self.query_proj)
        _orthogonal_init(self.mode_film)
        self.output_proj.apply(_orthogonal_init)

    def forward(
        self,
        scene_query: torch.Tensor,
        global_goals: torch.Tensor,
        goal_mask: Optional[torch.Tensor],
        task_mode_flag: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if global_goals.dim() != 3:
            raise RuntimeError(
                "global_goals must have shape (B, N, D), "
                f"got shape={tuple(global_goals.shape)}"
            )

        batch_size, num_goals, _ = global_goals.shape
        if goal_mask is None:
            active_mask = torch.ones(batch_size, num_goals, device=global_goals.device, dtype=torch.bool)
        else:
            if goal_mask.dim() == 3:
                goal_mask = goal_mask.squeeze(-1)
            active_mask = goal_mask > 0.5

        safe_mask = active_mask.clone()
        no_goal_rows = ~safe_mask.any(dim=1)
        if no_goal_rows.any():
            safe_mask[no_goal_rows, 0] = True

        if task_mode_flag is None:
            task_mode_flag = torch.zeros(batch_size, device=global_goals.device, dtype=torch.long)
        task_mode_flag = _to_2d_tensor(task_mode_flag).squeeze(-1).long().clamp_(0, 1)

        goal_tokens = self.goal_proj(global_goals)
        mode_token = self.mode_embedding(task_mode_flag)

        query = self.query_proj(scene_query + mode_token).unsqueeze(1)
        attended, attn_weights = self.cross_attention(
            query=query,
            key=goal_tokens,
            value=goal_tokens,
            key_padding_mask=~safe_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        attended = attended.squeeze(1)
        pooled = _masked_mean(goal_tokens, safe_mask)
        active_ratio = active_mask.float().mean(dim=1, keepdim=True)

        goal_context = self.output_proj(
            torch.cat([scene_query, attended, pooled, active_ratio], dim=-1)
        )
        gamma, beta = self.mode_film(mode_token).chunk(2, dim=-1)
        goal_context = goal_context * (1.0 + torch.tanh(gamma)) + beta
        if no_goal_rows.any():
            goal_context = goal_context * (~no_goal_rows).float().unsqueeze(-1)
        return goal_context, attn_weights


class MultiScaleSceneMemory(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        memory_size: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        dim_feedforward: int,
        short_window: int = 8,
        long_stride: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.memory_size = int(max(1, memory_size))
        self.short_window = int(max(1, short_window))
        self.long_stride = int(max(1, long_stride))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.token_norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.output_proj.apply(_orthogonal_init)

    def init_memory(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = torch.zeros(batch_size, self.memory_size, self.hidden_size, device=device, dtype=dtype)
        mask = torch.zeros(batch_size, self.memory_size, device=device, dtype=torch.bool)
        return tokens, mask

    def _select_context(self, memory_tokens: torch.Tensor, memory_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        current_len = int(memory_tokens.size(1))
        if current_len <= self.short_window:
            indices = torch.arange(current_len, device=memory_tokens.device)
        else:
            short_start = max(0, current_len - self.short_window)
            short_indices = torch.arange(short_start, current_len, device=memory_tokens.device)
            long_indices = torch.arange(0, short_start, self.long_stride, device=memory_tokens.device)
            indices = torch.cat([long_indices, short_indices], dim=0)
        return memory_tokens.index_select(1, indices), memory_mask.index_select(1, indices)

    def forward(
        self,
        current_token: torch.Tensor,
        memory_tokens: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = int(current_token.size(0))
        device = current_token.device
        dtype = current_token.dtype

        if memory_tokens is None or memory_mask is None:
            memory_tokens, memory_mask = self.init_memory(batch_size, device, dtype)
        else:
            memory_tokens = memory_tokens.to(device=device, dtype=dtype)
            memory_mask = memory_mask.to(device=device)

        if masks is not None:
            reset_mask = _to_2d_tensor(masks).squeeze(-1) > 0.5
            memory_tokens = memory_tokens * reset_mask[:, None, None].float()
            memory_mask = memory_mask & reset_mask[:, None]

        context_tokens, context_mask = self._select_context(memory_tokens, memory_mask)
        current_token = self.token_norm(current_token)
        sequence = torch.cat([context_tokens, current_token.unsqueeze(1)], dim=1)
        sequence_mask = torch.cat(
            [context_mask, torch.ones(batch_size, 1, device=device, dtype=torch.bool)],
            dim=1,
        )
        sequence = sequence + _build_sinusoidal_position_encoding(
            sequence.size(1),
            self.hidden_size,
            device=device,
            dtype=dtype,
        )
        encoded = self.encoder(sequence, src_key_padding_mask=~sequence_mask)
        contextual_token = self.output_proj(
            torch.cat([current_token, encoded[:, -1]], dim=-1)
        )

        updated_memory = torch.cat([memory_tokens, contextual_token.unsqueeze(1)], dim=1)
        updated_mask = torch.cat(
            [memory_mask, torch.ones(batch_size, 1, device=device, dtype=torch.bool)],
            dim=1,
        )
        updated_memory = updated_memory[:, -self.memory_size :]
        updated_mask = updated_mask[:, -self.memory_size :]
        return contextual_token, updated_memory, updated_mask


@dataclass
class OMAPolicyOutput:
    logits: torch.Tensor
    value: torch.Tensor
    features: torch.Tensor
    updated_memory_tokens: torch.Tensor
    updated_memory_mask: torch.Tensor
    goal_attention_weights: Optional[torch.Tensor]


class OMAPolicyNet(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space: Any,
        hidden_size: int = 512,
        goal_sensor_uuid: str = "global_goals",
        goal_image_uuid: str = "omni_long_goal_image",
        goal_text_uuid: str = "omni_long_goal_text",
        goal_modality_uuid: str = "omni_long_goal_modality",
        goal_mask_uuid: str = "goal_mask",
        task_mode_uuid: str = "task_mode_flag",
        rgb_uuid: str = "rgb",
        depth_uuid: str = "depth",
        spectrogram_uuid: str = "spectrogram",
        audio_doa_uuid: str = "audio_doa",
        pointgoal_uuid: str = "pointgoal_with_gps_compass",
        gps_uuid: str = "gps",
        compass_uuid: str = "compass",
        prev_action_embedding_size: int = 32,
        rgb_embedding_size: int = 128,
        depth_embedding_size: int = 64,
        audio_embedding_size: int = 128,
        vector_embedding_size: int = 32,
        transformer_memory_size: int = 32,
        transformer_nhead: int = 8,
        transformer_num_layers: int = 2,
        transformer_dropout: float = 0.1,
        transformer_dim_feedforward: int = 1024,
        short_memory_window: int = 8,
        long_memory_stride: int = 4,
        goal_clip_model: str = "ViT-B/32",
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.goal_sensor_uuid = str(goal_sensor_uuid)
        self.goal_image_uuid = str(goal_image_uuid)
        self.goal_text_uuid = str(goal_text_uuid)
        self.goal_modality_uuid = str(goal_modality_uuid)
        self.goal_mask_uuid = str(goal_mask_uuid)
        self.task_mode_uuid = str(task_mode_uuid)
        self.rgb_uuid = str(rgb_uuid)
        self.depth_uuid = str(depth_uuid)
        self.spectrogram_uuid = str(spectrogram_uuid)
        self.audio_doa_uuid = str(audio_doa_uuid)
        self.pointgoal_uuid = str(pointgoal_uuid)
        self.gps_uuid = str(gps_uuid)
        self.compass_uuid = str(compass_uuid)
        self.num_actions = _infer_num_actions(action_space)
        self.goal_clip_model = str(goal_clip_model)
        self._raw_goal_sensor_keys = (
            self.goal_image_uuid,
            self.goal_text_uuid,
            self.goal_modality_uuid,
        )

        self._goal_sensor_candidates = (
            self.goal_sensor_uuid,
            "global_goals",
            "omni_long_goal",
        )

        goal_shape = _infer_first_available_shape(observation_space, self._goal_sensor_candidates)
        if goal_shape is None or len(goal_shape) != 2:
            raise RuntimeError(
                "OMAPolicyNet requires a goal tensor space with shape (N_MAX, D). "
                f"Tried keys={self._goal_sensor_candidates}"
            )
        self.max_goals = int(goal_shape[0])
        self.goal_input_size = int(goal_shape[1])
        self.goal_raw_encoder = FrozenCLIPGoalEncoder(
            model_name=self.goal_clip_model,
            output_size=self.goal_input_size,
        )
        self._goal_cache_embeddings: Optional[torch.Tensor] = None
        self._goal_cache_valid: Optional[torch.Tensor] = None

        rgb_shape = _infer_space_shape(observation_space, self.rgb_uuid)
        self.rgb_encoder = None
        if rgb_shape is not None and len(rgb_shape) == 3:
            rgb_channels = int(rgb_shape[-1] if rgb_shape[-1] <= 4 else rgb_shape[0])
            self.rgb_encoder = ConvEncoder(
                input_channels=rgb_channels,
                output_size=rgb_embedding_size,
                base_channels=32,
                normalize_uint8=True,
            )

        depth_shape = _infer_space_shape(observation_space, self.depth_uuid)
        self.depth_encoder = None
        if depth_shape is not None and len(depth_shape) == 3:
            depth_channels = int(depth_shape[-1] if depth_shape[-1] <= 4 else depth_shape[0])
            self.depth_encoder = ConvEncoder(
                input_channels=depth_channels,
                output_size=depth_embedding_size,
                base_channels=16,
                normalize_uint8=False,
            )

        spec_shape = _infer_space_shape(observation_space, self.spectrogram_uuid)
        self.spectrogram_encoder = None
        if spec_shape is not None and len(spec_shape) == 3:
            spec_channels = int(spec_shape[-1] if spec_shape[-1] <= 4 else spec_shape[0])
            self.spectrogram_encoder = ConvEncoder(
                input_channels=spec_channels,
                output_size=audio_embedding_size,
                base_channels=16,
                normalize_uint8=False,
            )

        doa_shape = _infer_space_shape(observation_space, self.audio_doa_uuid)
        self.audio_doa_encoder = None
        if doa_shape is not None and len(doa_shape) == 2 and doa_shape[-1] == 3:
            self.audio_doa_encoder = SoundEventDescriptorEncoder(
                num_audio_classes=int(doa_shape[0]),
                hidden_size=audio_embedding_size,
                nhead=max(1, min(transformer_nhead, audio_embedding_size // 32 or 1)),
                num_layers=1,
                dropout=transformer_dropout,
            )

        self.pointgoal_encoder = None
        pointgoal_shape = _infer_space_shape(observation_space, self.pointgoal_uuid)
        if pointgoal_shape is not None:
            self.pointgoal_encoder = MLPBlock(
                input_size=int(torch.tensor(pointgoal_shape).prod().item()),
                hidden_size=vector_embedding_size,
                output_size=vector_embedding_size,
                dropout=transformer_dropout,
            )

        self.gps_encoder = None
        gps_shape = _infer_space_shape(observation_space, self.gps_uuid)
        if gps_shape is not None:
            self.gps_encoder = MLPBlock(
                input_size=int(torch.tensor(gps_shape).prod().item()),
                hidden_size=vector_embedding_size,
                output_size=vector_embedding_size,
                dropout=transformer_dropout,
            )

        self.compass_encoder = None
        compass_shape = _infer_space_shape(observation_space, self.compass_uuid)
        if compass_shape is not None:
            self.compass_encoder = MLPBlock(
                input_size=2,
                hidden_size=vector_embedding_size,
                output_size=vector_embedding_size,
                dropout=transformer_dropout,
            )

        self.prev_action_embedding = nn.Embedding(self.num_actions + 1, prev_action_embedding_size)
        nn.init.normal_(self.prev_action_embedding.weight, mean=0.0, std=0.02)

        scene_input_size = prev_action_embedding_size
        if self.rgb_encoder is not None:
            scene_input_size += rgb_embedding_size
        if self.depth_encoder is not None:
            scene_input_size += depth_embedding_size
        if self.spectrogram_encoder is not None or self.audio_doa_encoder is not None:
            scene_input_size += audio_embedding_size
        if self.pointgoal_encoder is not None:
            scene_input_size += vector_embedding_size
        if self.gps_encoder is not None:
            scene_input_size += vector_embedding_size
        if self.compass_encoder is not None:
            scene_input_size += vector_embedding_size

        self.scene_fuser = nn.Sequential(
            nn.LayerNorm(scene_input_size),
            nn.Linear(scene_input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(transformer_dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.scene_fuser.apply(_orthogonal_init)

        self.goal_encoder = GoalContextEncoder(
            goal_input_size=self.goal_input_size,
            hidden_size=hidden_size,
            nhead=transformer_nhead,
            dropout=transformer_dropout,
        )
        self.pre_memory_fuser = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Dropout(transformer_dropout),
            nn.Linear(hidden_size, hidden_size),
        )
        self.pre_memory_fuser.apply(_orthogonal_init)

        self.scene_memory = MultiScaleSceneMemory(
            hidden_size=hidden_size,
            memory_size=transformer_memory_size,
            nhead=transformer_nhead,
            num_layers=transformer_num_layers,
            dropout=transformer_dropout,
            dim_feedforward=transformer_dim_feedforward,
            short_window=short_memory_window,
            long_stride=long_memory_stride,
        )

        self.policy_head = nn.Linear(hidden_size, self.num_actions)
        self.value_head = nn.Linear(hidden_size, 1)
        _orthogonal_init(self.policy_head, gain=0.01)
        _orthogonal_init(self.value_head, gain=1.0)

    @property
    def output_size(self) -> int:
        return self.hidden_size

    @classmethod
    def build_policy_observation_space(
        cls,
        config: Any,
        env_observation_space: spaces.Dict,
    ) -> spaces.Dict:
        ppo_cfg = config.RL.PPO if hasattr(config, "RL") else config
        task_cfg = getattr(config, "TASK_CONFIG", None)
        if task_cfg is not None:
            task = getattr(task_cfg, "TASK", task_cfg)
            goal_sensor_uuid = str(getattr(task, "GOAL_SENSOR_UUID", "omni_long_goal"))
        else:
            goal_sensor_uuid = "omni_long_goal"

        goal_image_uuid = str(getattr(ppo_cfg, "goal_image_uuid", "omni_long_goal_image"))
        goal_text_uuid = str(getattr(ppo_cfg, "goal_text_uuid", "omni_long_goal_text"))
        goal_modality_uuid = str(getattr(ppo_cfg, "goal_modality_uuid", "omni_long_goal_modality"))
        audio_uuid = str(getattr(ppo_cfg, "audio_sensor_uuid", "audiogoal"))
        excluded_keys = {goal_image_uuid, goal_text_uuid, goal_modality_uuid}
        policy_space = _copy_space_dict_without_keys(env_observation_space, excluded_keys)

        audio_shape = _infer_space_shape(env_observation_space, audio_uuid)
        if audio_shape is not None and len(audio_shape) == 2:
            processed_audio_shape = _infer_processed_audio_shape(audio_shape)
            policy_space.spaces[audio_uuid] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=processed_audio_shape,
                dtype=np.float32,
            )

        if goal_sensor_uuid not in policy_space.spaces:
            goal_image_shape = _infer_space_shape(env_observation_space, goal_image_uuid)
            goal_text_shape = _infer_space_shape(env_observation_space, goal_text_uuid)
            max_goals = 0
            if goal_image_shape is not None and len(goal_image_shape) >= 1:
                max_goals = int(goal_image_shape[0])
            elif goal_text_shape is not None and len(goal_text_shape) >= 1:
                max_goals = int(goal_text_shape[0])
            if max_goals > 0:
                goal_embedding_size = int(getattr(ppo_cfg, "goal_embedding_size", 512))
                policy_space.spaces[goal_sensor_uuid] = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(max_goals, goal_embedding_size),
                    dtype=np.float32,
                )

        return policy_space

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "goal_raw_encoder") and self.goal_raw_encoder is not None:
            self.goal_raw_encoder.eval()
        return self

    def init_memory(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.scene_memory.init_memory(batch_size, device, dtype)

    def _reset_goal_cache(self) -> None:
        self._goal_cache_embeddings = None
        self._goal_cache_valid = None

    def _goal_cache_refresh_mask(
        self,
        batch_size: int,
        device: torch.device,
        masks: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if (
            self._goal_cache_embeddings is None
            or self._goal_cache_valid is None
            or int(self._goal_cache_embeddings.size(0)) != int(batch_size)
            or self._goal_cache_embeddings.device != device
        ):
            self._goal_cache_embeddings = torch.zeros(
                batch_size,
                self.max_goals,
                self.goal_input_size,
                device=device,
                dtype=torch.float32,
            )
            self._goal_cache_valid = torch.zeros(batch_size, device=device, dtype=torch.bool)
            return torch.ones(batch_size, device=device, dtype=torch.bool)

        refresh_mask = ~self._goal_cache_valid
        if masks is None:
            return torch.ones(batch_size, device=device, dtype=torch.bool)

        if masks.dim() == 2 and masks.size(-1) == 1:
            masks = masks.squeeze(-1)
        refresh_mask |= masks.to(device=device).float() < 0.5
        return refresh_mask

    def _encode_goal_batch_with_cache(
        self,
        observations: Dict[str, torch.Tensor],
        masks: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        goal_images = observations.get(self.goal_image_uuid)
        goal_text = observations.get(self.goal_text_uuid)
        goal_modality = observations.get(self.goal_modality_uuid)
        if goal_images is None or goal_text is None or goal_modality is None:
            return observations.get(self.goal_sensor_uuid)

        batch_size = int(goal_images.size(0))
        device = goal_images.device
        refresh_mask = self._goal_cache_refresh_mask(batch_size, device, masks)
        if refresh_mask.any():
            refreshed_embeddings = self.goal_raw_encoder(
                goal_images=goal_images[refresh_mask],
                goal_text_raw=goal_text[refresh_mask],
                goal_modality=goal_modality[refresh_mask],
            )
            self._goal_cache_embeddings[refresh_mask] = refreshed_embeddings
            self._goal_cache_valid[refresh_mask] = True

        return self._goal_cache_embeddings

    def prepare_observations(
        self,
        observations: Dict[str, torch.Tensor],
        masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        prepared = dict(observations)
        raw_audio = prepared.get(self.spectrogram_uuid)
        if raw_audio is not None and raw_audio.dim() == 3:
            prepared[self.spectrogram_uuid] = _compute_torch_audio_spectrogram(raw_audio)

        if self.goal_sensor_uuid not in prepared:
            goal_embeddings = self._encode_goal_batch_with_cache(prepared, masks=masks)
            if goal_embeddings is not None:
                prepared[self.goal_sensor_uuid] = goal_embeddings

        for raw_key in self._raw_goal_sensor_keys:
            prepared.pop(raw_key, None)
        return prepared

    def _find_observation(self, observations: Dict[str, torch.Tensor], keys: Sequence[str]) -> Optional[torch.Tensor]:
        for key in keys:
            value = observations.get(key)
            if value is not None:
                return value
        return None

    def _encode_rgb(self, observations: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if self.rgb_encoder is None:
            return None
        rgb = observations.get(self.rgb_uuid)
        if rgb is None:
            return None
        return self.rgb_encoder(rgb)

    def _encode_depth(self, observations: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if self.depth_encoder is None:
            return None
        depth = observations.get(self.depth_uuid)
        if depth is None:
            return None
        return self.depth_encoder(depth)

    def _encode_audio(self, observations: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        audio_features = []
        if self.spectrogram_encoder is not None:
            spectrogram = observations.get(self.spectrogram_uuid)
            if spectrogram is not None:
                audio_features.append(self.spectrogram_encoder(spectrogram))
        if self.audio_doa_encoder is not None:
            audio_doa = observations.get(self.audio_doa_uuid)
            if audio_doa is not None:
                audio_features.append(self.audio_doa_encoder(audio_doa))
        if not audio_features:
            return None
        if len(audio_features) == 1:
            return audio_features[0]
        return torch.stack(audio_features, dim=0).mean(dim=0)

    def _encode_pointgoal(self, observations: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if self.pointgoal_encoder is None:
            return None
        pointgoal = observations.get(self.pointgoal_uuid)
        if pointgoal is None:
            return None
        return self.pointgoal_encoder(_to_2d_tensor(pointgoal.float()))

    def _encode_gps(self, observations: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if self.gps_encoder is None:
            return None
        gps = observations.get(self.gps_uuid)
        if gps is None:
            return None
        return self.gps_encoder(_to_2d_tensor(gps.float()))

    def _encode_compass(self, observations: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        if self.compass_encoder is None:
            return None
        compass = observations.get(self.compass_uuid)
        if compass is None:
            return None
        compass = _to_2d_tensor(compass.float())
        if compass.size(-1) == 1:
            compass = torch.cat([torch.sin(compass), torch.cos(compass)], dim=-1)
        return self.compass_encoder(compass)

    def _encode_prev_actions(
        self,
        prev_actions: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if prev_actions is None:
            prev_actions = torch.zeros(batch_size, 1, device=device, dtype=torch.long)
        prev_actions = _to_2d_tensor(prev_actions).squeeze(-1).long().clamp(min=0)
        if masks is not None:
            keep_mask = (_to_2d_tensor(masks).squeeze(-1) > 0.5).long()
            prev_actions = prev_actions * keep_mask
        prev_actions = torch.clamp(prev_actions, max=self.num_actions)
        return self.prev_action_embedding(prev_actions)

    def _gather_goal_inputs(
        self,
        observations: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        global_goals = self._find_observation(observations, self._goal_sensor_candidates)
        if global_goals is None:
            raise RuntimeError(
                "OMAPolicyNet forward requires goal embeddings in observations under one of: "
                f"{self._goal_sensor_candidates}"
            )
        goal_mask = observations.get(self.goal_mask_uuid)
        task_mode_flag = observations.get(self.task_mode_uuid)
        return global_goals.float(), goal_mask, task_mode_flag

    def _encode_scene(
        self,
        observations: Dict[str, torch.Tensor],
        prev_actions: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> torch.Tensor:
        any_tensor = next(iter(observations.values()))
        batch_size = int(any_tensor.size(0))
        device = any_tensor.device
        features = [self._encode_prev_actions(prev_actions, batch_size, device, masks=masks)]

        for encoder_output in (
            self._encode_rgb(observations),
            self._encode_depth(observations),
            self._encode_audio(observations),
            self._encode_pointgoal(observations),
            self._encode_gps(observations),
            self._encode_compass(observations),
        ):
            if encoder_output is not None:
                features.append(encoder_output)

        return self.scene_fuser(torch.cat(features, dim=-1))

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        prev_actions: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        memory_tokens: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> OMAPolicyOutput:
        scene_token = self._encode_scene(observations, prev_actions, masks)
        global_goals, goal_mask, task_mode_flag = self._gather_goal_inputs(observations)
        goal_context, goal_attention_weights = self.goal_encoder(
            scene_query=scene_token,
            global_goals=global_goals,
            goal_mask=goal_mask,
            task_mode_flag=task_mode_flag,
        )

        current_token = self.pre_memory_fuser(torch.cat([scene_token, goal_context], dim=-1))
        contextual_token, updated_memory_tokens, updated_memory_mask = self.scene_memory(
            current_token=current_token,
            memory_tokens=memory_tokens,
            memory_mask=memory_mask,
            masks=masks,
        )
        logits = self.policy_head(contextual_token)
        value = self.value_head(contextual_token)
        return OMAPolicyOutput(
            logits=logits,
            value=value,
            features=contextual_token,
            updated_memory_tokens=updated_memory_tokens,
            updated_memory_mask=updated_memory_mask,
            goal_attention_weights=goal_attention_weights,
        )

    def act(
        self,
        observations: Dict[str, torch.Tensor],
        prev_actions: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        memory_tokens: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, OMAPolicyOutput]:
        output = self.forward(
            observations=observations,
            prev_actions=prev_actions,
            masks=masks,
            memory_tokens=memory_tokens,
            memory_mask=memory_mask,
        )
        distribution = torch.distributions.Categorical(logits=output.logits)
        if deterministic:
            action = distribution.probs.argmax(dim=-1)
        else:
            action = distribution.sample()
        action_log_probs = distribution.log_prob(action).unsqueeze(-1)
        return (
            output.value,
            action.unsqueeze(-1),
            action_log_probs,
            output.updated_memory_tokens,
            output.updated_memory_mask,
            output,
        )

    def get_value(
        self,
        observations: Dict[str, torch.Tensor],
        prev_actions: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        memory_tokens: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output = self.forward(
            observations=observations,
            prev_actions=prev_actions,
            masks=masks,
            memory_tokens=memory_tokens,
            memory_mask=memory_mask,
        )
        return output.value

    def evaluate_actions(
        self,
        observations: Dict[str, torch.Tensor],
        action: torch.Tensor,
        prev_actions: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        memory_tokens: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, OMAPolicyOutput]:
        output = self.forward(
            observations=observations,
            prev_actions=prev_actions,
            masks=masks,
            memory_tokens=memory_tokens,
            memory_mask=memory_mask,
        )
        distribution = torch.distributions.Categorical(logits=output.logits)
        action = _to_2d_tensor(action).squeeze(-1).long()
        action_log_probs = distribution.log_prob(action).unsqueeze(-1)
        entropy = distribution.entropy().mean().unsqueeze(0)
        return (
            output.value,
            action_log_probs,
            entropy,
            output.updated_memory_tokens,
            output.updated_memory_mask,
            output,
        )


class OmniLongBaselinePolicy(OMAPolicyNet):
    @classmethod
    def from_config(
        cls,
        config: Any,
        observation_space: spaces.Dict,
        action_space: Any,
    ) -> "OmniLongBaselinePolicy":
        ppo_cfg = config.RL.PPO if hasattr(config, "RL") else config
        task_cfg = getattr(config, "TASK_CONFIG", None)
        if task_cfg is not None:
            task = getattr(task_cfg, "TASK", task_cfg)
            goal_sensor_uuid = str(getattr(task, "GOAL_SENSOR_UUID", "omni_long_goal"))
        else:
            goal_sensor_uuid = "omni_long_goal"

        return cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=int(getattr(ppo_cfg, "hidden_size", 512)),
            goal_sensor_uuid=goal_sensor_uuid,
            goal_image_uuid=str(getattr(ppo_cfg, "goal_image_uuid", "omni_long_goal_image")),
            goal_text_uuid=str(getattr(ppo_cfg, "goal_text_uuid", "omni_long_goal_text")),
            goal_modality_uuid=str(getattr(ppo_cfg, "goal_modality_uuid", "omni_long_goal_modality")),
            goal_mask_uuid=str(getattr(ppo_cfg, "goal_mask_uuid", "goal_mask")),
            task_mode_uuid=str(getattr(ppo_cfg, "task_mode_uuid", "task_mode_flag")),
            rgb_uuid=str(getattr(ppo_cfg, "rgb_sensor_uuid", "rgb")),
            depth_uuid=str(getattr(ppo_cfg, "depth_sensor_uuid", "depth")),
            spectrogram_uuid=str(getattr(ppo_cfg, "audio_sensor_uuid", "spectrogram")),
            audio_doa_uuid=str(getattr(ppo_cfg, "audio_doa_uuid", "audio_doa")),
            pointgoal_uuid=str(getattr(ppo_cfg, "pointgoal_sensor_uuid", "pointgoal_with_gps_compass")),
            gps_uuid=str(getattr(ppo_cfg, "gps_sensor_uuid", "gps")),
            compass_uuid=str(getattr(ppo_cfg, "compass_sensor_uuid", "compass")),
            prev_action_embedding_size=int(getattr(ppo_cfg, "prev_action_embedding_size", 32)),
            rgb_embedding_size=int(getattr(ppo_cfg, "rgb_embedding_size", 128)),
            depth_embedding_size=int(getattr(ppo_cfg, "depth_embedding_size", 128)),
            audio_embedding_size=int(getattr(ppo_cfg, "audio_embedding_size", 128)),
            vector_embedding_size=int(getattr(ppo_cfg, "gps_embedding_size", 32)),
            transformer_memory_size=int(getattr(ppo_cfg, "transformer_memory_size", 32)),
            transformer_nhead=int(getattr(ppo_cfg, "transformer_nhead", 8)),
            transformer_num_layers=int(getattr(ppo_cfg, "transformer_num_layers", 2)),
            transformer_dropout=float(getattr(ppo_cfg, "transformer_dropout", 0.1)),
            transformer_dim_feedforward=int(getattr(ppo_cfg, "transformer_dim_feedforward", 1024)),
            goal_clip_model=str(getattr(ppo_cfg, "goal_clip_model", "ViT-B/32")),
        )


__all__ = ["OMAPolicyNet", "OMAPolicyOutput", "OmniLongBaselinePolicy"]
