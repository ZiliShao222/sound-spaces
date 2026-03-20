from __future__ import annotations

from typing import Any, Dict, Optional
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ss_baselines.common.omni_long_eval_policy import (
    LifelongEvalContext,
    LifelongEvalPolicy,
    register_lifelong_eval_policy,
)


@register_lifelong_eval_policy("omega_nav")
@register_lifelong_eval_policy("omega_nav_policy")
@register_lifelong_eval_policy("omega_oracle")
@register_lifelong_eval_policy("omega_nav_oracle")
class OmegaNavPolicy(LifelongEvalPolicy):
    def __init__(
        self,
        seed: int = 0,
        goal_order_mode: str = "ordered",
        siglip_model_name: str = "google/siglip-base-patch16-224",
        device: Optional[str] = None,
        **_: Any,
    ):
        self._rng = np.random.default_rng(int(seed))
        self._goal_order_mode = str(goal_order_mode)
        self._siglip_model_name = str(siglip_model_name)
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._siglip_processor = None
        self._siglip_model = None
        self._goal_embeddings: Optional[np.ndarray] = None
        self._goal_mask: Optional[np.ndarray] = None
        self._active_goal_embeddings: Optional[np.ndarray] = None
        self._last_pose: Any = None
        self._last_info: Optional[Dict[str, Any]] = None
        self._last_action: Optional[int] = None

    def reset(self, *, env: Any, episode: Any, observations: Dict[str, Any]) -> None:
        self._goal_embeddings = None
        self._goal_mask = None
        self._active_goal_embeddings = None
        self._last_pose = None
        self._last_info = None
        self._last_action = None

    def start_episode(
        self,
        *,
        env: Any,
        episode: Any,
        observations: Dict[str, Any],
        goal_payloads,
        order_mode: Optional[Any] = None,
    ) -> None:
        if self._siglip_processor is None or self._siglip_model is None:
            from transformers import AutoModel, AutoProcessor

            self._siglip_processor = AutoProcessor.from_pretrained(self._siglip_model_name)
            self._siglip_model = AutoModel.from_pretrained(self._siglip_model_name)
            self._siglip_model.eval()
            self._siglip_model.to(self._device)

        goal_count = len(goal_payloads)

        embeddings = [None] * goal_count
        text_indices = []
        text_inputs = []
        image_indices = []
        image_inputs = []

        for goal_index, payload in enumerate(goal_payloads):
            modality = str(payload.get("modality", "object"))
            if modality == "image":
                image = np.asarray(payload.get("image"), dtype=np.uint8)
                image_inputs.append(Image.fromarray(image))
                image_indices.append(goal_index)
            elif modality == "description":
                text_inputs.append(str(payload.get("text", "")))
                text_indices.append(goal_index)
            else:
                text_inputs.append(str(payload.get("category", "")))
                text_indices.append(goal_index)

        with torch.inference_mode():
            if text_inputs:
                text_batch = self._siglip_processor(text=text_inputs, padding=True, truncation=True, return_tensors="pt")
                text_batch = {key: value.to(self._device) for key, value in text_batch.items()}
                text_features = self._siglip_model.get_text_features(**text_batch)
                if not isinstance(text_features, torch.Tensor):
                    text_features = text_features.pooler_output
                text_features = F.normalize(text_features, dim=-1).detach().cpu().numpy().astype(np.float32)
                for row_index, goal_index in enumerate(text_indices):
                    embeddings[goal_index] = text_features[row_index]

            if image_inputs:
                image_batch = self._siglip_processor(images=image_inputs, return_tensors="pt")
                image_batch = {key: value.to(self._device) for key, value in image_batch.items()}
                image_features = self._siglip_model.get_image_features(**image_batch)
                if not isinstance(image_features, torch.Tensor):
                    image_features = image_features.pooler_output
                image_features = F.normalize(image_features, dim=-1).detach().cpu().numpy().astype(np.float32)
                for row_index, goal_index in enumerate(image_indices):
                    embeddings[goal_index] = image_features[row_index]

        self._goal_embeddings = np.stack(embeddings, axis=0)
        self._goal_mask = np.ones((goal_count,), dtype=np.float32)
        if self._goal_order_mode == "ordered":
            self._goal_mask[1:] = 0.0
        self._active_goal_embeddings = self._goal_embeddings * self._goal_mask[:, None]
        self._last_pose = observations.get("pose")
        self._last_info = None
        self._last_action = None

    def act(
        self,
        *,
        env: Any,
        episode: Any,
        observations: Dict[str, Any],
        context: LifelongEvalContext,
    ) -> Any:
        self._last_pose = observations.get("pose")
        self._last_info = context.info
        self._last_action = int(self._rng.integers(0, 5))
        return self._last_action

    def get_debug_state(self) -> Dict[str, Any]:
        return {
            "goal_order_mode": self._goal_order_mode,
            "goal_embeddings_shape": None if self._goal_embeddings is None else list(self._goal_embeddings.shape),
            "goal_mask": None if self._goal_mask is None else self._goal_mask.tolist(),
            "active_goal_embeddings_shape": None if self._active_goal_embeddings is None else list(self._active_goal_embeddings.shape),
            "pose": self._last_pose,
            "info": self._last_info,
            "action": self._last_action,
        }
