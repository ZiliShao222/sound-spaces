#!/usr/bin/env python3

from __future__ import annotations

from typing import Any, Dict, Sequence

import numpy as np
import torch


class MiniCPMGoalFeatureEncoder:
    name = "MiniCPM-o-2.6"

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        batch_size: int = 16,
    ) -> None:
        if not model_path:
            raise ValueError("`model_path` is required for MiniCPMGoalFeatureEncoder.")
        try:
            from transformers import AutoModel, AutoProcessor, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "`transformers` is required for MiniCPMGoalFeatureEncoder. "
                "Install it in your training environment first."
            ) from exc

        self._device = torch.device(device)
        self._batch_size = int(batch_size)
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        self._model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
        ).eval().to(self._device)

    @classmethod
    def from_args(cls, args: Any) -> "MiniCPMGoalFeatureEncoder":
        return cls(
            model_path=str(args.model_path),
            device=str(args.device),
            batch_size=int(args.batch_size),
        )

    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        moved: Dict[str, Any] = {}
        for key, value in batch.items():
            if hasattr(value, "to"):
                moved[key] = value.to(self._device)
            else:
                moved[key] = value
        return moved

    def _normalize(self, tensor: torch.Tensor) -> np.ndarray:
        tensor = tensor.float()
        tensor = tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return tensor.detach().cpu().numpy().astype(np.float32)

    def _pool_hidden_states(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            return hidden_states[:, 0]
        mask = attention_mask.unsqueeze(-1).float()
        numerator = (hidden_states * mask).sum(dim=1)
        denominator = mask.sum(dim=1).clamp_min(1.0)
        return numerator / denominator

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        with torch.no_grad():
            if hasattr(self._model, "encode_text"):
                outputs = self._model.encode_text(list(texts))
                if not isinstance(outputs, torch.Tensor):
                    outputs = torch.as_tensor(outputs)
                return self._normalize(outputs)

            if hasattr(self._model, "get_text_features"):
                batch = self._tokenizer(
                    list(texts),
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                outputs = self._model.get_text_features(**self._move_to_device(batch))
                return self._normalize(outputs)

            text_model = getattr(self._model, "text_model", None)
            if text_model is not None:
                batch = self._tokenizer(
                    list(texts),
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                batch = self._move_to_device(batch)
                outputs = text_model(**batch)
                pooled = self._pool_hidden_states(
                    outputs.last_hidden_state,
                    attention_mask=batch.get("attention_mask"),
                )
                return self._normalize(pooled)

        raise NotImplementedError(
            "This MiniCPM backend could not find a text embedding API on the loaded model. "
            "Customize `scripts/goal_feature_encoders/minicpm_o26.py` for your local checkpoint."
        )

    def encode_images(self, images: Sequence[Any]) -> np.ndarray:
        if not images:
            return np.zeros((0, 0), dtype=np.float32)
        with torch.no_grad():
            if hasattr(self._model, "encode_image"):
                outputs = self._model.encode_image(list(images))
                if not isinstance(outputs, torch.Tensor):
                    outputs = torch.as_tensor(outputs)
                return self._normalize(outputs)

            processor_batch = self._processor(images=list(images), return_tensors="pt")
            processor_batch = self._move_to_device(processor_batch)

            if hasattr(self._model, "get_image_features"):
                outputs = self._model.get_image_features(**processor_batch)
                return self._normalize(outputs)

            vision_model = getattr(self._model, "vision_model", None)
            if vision_model is not None:
                outputs = vision_model(**processor_batch)
                pooled = self._pool_hidden_states(outputs.last_hidden_state)
                return self._normalize(pooled)

        raise NotImplementedError(
            "This MiniCPM backend could not find an image embedding API on the loaded model. "
            "Customize `scripts/goal_feature_encoders/minicpm_o26.py` for your local checkpoint."
        )
