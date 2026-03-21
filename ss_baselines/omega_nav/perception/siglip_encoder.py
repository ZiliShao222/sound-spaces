from __future__ import annotations

from typing import Any, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class SigLIPEncoder:
    def __init__(self, model_name: str, device: torch.device) -> None:
        self._model_name = str(model_name)
        self._device = torch.device(device)
        self._model: Any = None
        self._processor: Any = None

    def _ensure_model(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        from transformers import AutoModel, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self._model_name)
        self._model = AutoModel.from_pretrained(self._model_name).to(self._device)
        self._model.eval()

    def _to_pil(self, image: np.ndarray) -> Image.Image:
        array = np.asarray(image, dtype=np.uint8)
        if array.ndim == 2:
            array = np.repeat(array[:, :, None], 3, axis=2)
        if array.ndim == 3 and array.shape[-1] == 4:
            array = array[:, :, :3]
        return Image.fromarray(array)

    def encode_text(self, text: str) -> np.ndarray:
        self._ensure_model()
        inputs = self._processor(text=[str(text)], return_tensors="pt", padding=True)
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.inference_mode():
            features = self._model.get_text_features(**inputs)
            features = F.normalize(features, dim=-1)
        return features[0].detach().cpu().numpy().astype(np.float32)

    def encode_image(self, image: np.ndarray) -> np.ndarray:
        return self.encode_images([image])[0]

    def encode_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        self._ensure_model()
        pil_images: List[Image.Image] = [self._to_pil(image) for image in images]
        inputs = self._processor(images=pil_images, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with torch.inference_mode():
            features = self._model.get_image_features(**inputs)
            features = F.normalize(features, dim=-1)
        return features.detach().cpu().numpy().astype(np.float32)
