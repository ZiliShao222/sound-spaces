from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ss_baselines.common.omni_long_eval_policy import (
    LifelongEvalContext,
    LifelongEvalPolicy,
    TURN_RIGHT_ACTION_ID,
    register_lifelong_eval_policy,
)
from ss_baselines.omega_nav.audio_prototypes import OfflineAcousticPrototypeLibrary
from ss_baselines.omega_nav.perception import PerceptionEncoder, PerceptionOutput
from ss_baselines.omega_nav.qwen_service import chat_completion, extract_json_object, extract_message_text, image_array_to_data_url
from ss_baselines.omega_nav.utils import load_omega_config, normalize_text


@register_lifelong_eval_policy("omega_nav")
@register_lifelong_eval_policy("omega_nav_policy")
@register_lifelong_eval_policy("omega_oracle")
@register_lifelong_eval_policy("omega_nav_oracle")
class OmegaNavPolicy(LifelongEvalPolicy):
    def __init__(
        self,
        device: Optional[str] = None,
        look_around_steps: int = 11,
        audio_clean_sound_dir: str = "data/sounds/semantic_splits/val",
        audio_encoder_model_name: str = "laion/clap-htsat-unfused",
        audio_sampling_rate_hz: int = 16000,
        audio_embedding_sampling_rate_hz: int = 16000,
        audio_prototype_window_sec: float = 1.0,
        audio_prototype_hop_sec: float = 0.5,
        config_path: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        goal_encoder_model_name: str = "google/siglip-base-patch16-224",
        **_: Any,
    ) -> None:
        cfg = load_omega_config(config_path=config_path, overrides=config_overrides) if config_path or config_overrides else load_omega_config()
        qwen_cfg = dict(cfg.get("qwen", {}))
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._look_around_steps = max(int(look_around_steps), 0)
        self._qwen_enabled = bool(qwen_cfg.get("enabled", False))
        self._qwen_api_base = str(qwen_cfg.get("api_base", "http://127.0.0.1:8000/v1"))
        self._qwen_api_key = str(qwen_cfg.get("api_key") or os.environ.get("QWEN_API_KEY") or os.environ.get("OPENAI_API_KEY") or "EMPTY")
        self._qwen_model = str(qwen_cfg.get("model", "Qwen2.5-VL-7B-Instruct"))
        self._qwen_timeout = int(qwen_cfg.get("timeout_sec", 8))
        self._perception = PerceptionEncoder(cfg, self._device, goal_encoder_model_name)
        self._audio_prototype_library = OfflineAcousticPrototypeLibrary(
            clean_sound_dir=str(audio_clean_sound_dir),
            observation_sampling_rate_hz=max(int(audio_sampling_rate_hz), 1),
            encoder_model_name=str(audio_encoder_model_name),
            encoder_sampling_rate_hz=max(int(audio_embedding_sampling_rate_hz), 1),
            prototype_window_sec=max(float(audio_prototype_window_sec), 0.1),
            prototype_hop_sec=max(float(audio_prototype_hop_sec), 0.05),
            device=self._device,
        )
        self._audio_prototype_library.ensure_loaded()
        labels = tuple(self._audio_prototype_library.prototype_labels)
        matrix = np.asarray(self._audio_prototype_library.prototype_matrix, dtype=np.float32)
        self._audio_global = {str(label): np.asarray(matrix[i], dtype=np.float32) for i, label in enumerate(labels)}
        self.reset(env=None, episode=None, observations={})

    def reset(self, *, env: Any, episode: Any, observations: Dict[str, Any]) -> None:
        del env, episode, observations
        self._goal_ids: Tuple[str, ...] = ()
        self._goal_modalities: Dict[str, str] = {}
        self._goal_embeddings: Dict[str, np.ndarray] = {}
        self._audio_goal_labels: Dict[str, Tuple[str, ...]] = {}
        self._audio_episode_dict: Dict[str, np.ndarray] = {}
        self._unsupported_audio_goals: List[Dict[str, Any]] = []
        self._trajectory_step_index = 0
        self._last_action: Optional[int] = None
        self._last_info: Optional[Dict[str, Any]] = None
        self._last_perception: Optional[PerceptionOutput] = None

    def start_episode(
        self,
        *,
        env: Any,
        episode: Any,
        observations: Dict[str, Any],
        goal_payloads: Sequence[Dict[str, Any]],
        order_mode: Optional[Any] = None,
    ) -> None:
        del env, episode, order_mode
        self.reset(env=None, episode=None, observations={})
        self._perception.reset(observations)
        self._goal_ids = tuple(f"goal_{i:03d}" for i in range(len(goal_payloads)))
        self._goal_embeddings = self._perception.encode_goals(self._goal_ids, goal_payloads)
        for goal_id, payload in zip(self._goal_ids, goal_payloads):
            modality = str(payload.get("modality", "")).lower() or "object"
            self._goal_modalities[goal_id] = modality
            inferred_nodes = self._infer_graph_nodes(payload) if self._qwen_enabled and modality in {"description", "text", "image"} else []
            labels = self._resolve_labels(payload, inferred_nodes=inferred_nodes)
            if not labels:
                self._unsupported_audio_goals.append(
                    {
                        "goal_id": goal_id,
                        "modality": modality,
                        "has_embedding": goal_id in self._goal_embeddings,
                    }
                )
                continue
            self._audio_goal_labels[goal_id] = (labels[0],)
            episode_labels = labels if modality == "image" else (labels[0],)
            for label in episode_labels:
                self._audio_episode_dict.setdefault(label, self._audio_global[label])
        print(
            "[omega_nav][start_episode] "
            + json.dumps(
                {
                    "goal_modalities": dict(self._goal_modalities),
                    "goal_embedding_dims": self._perception.goal_embedding_dims(),
                    "audio_goal_labels": {k: list(v) for k, v in self._audio_goal_labels.items()},
                },
                ensure_ascii=False,
            )
        )

    def act(self, *, env: Any, episode: Any, observations: Dict[str, Any], context: LifelongEvalContext) -> Any:
        del env, episode
        self._last_info = context.info
        self._last_perception = self._perception.percept(observations, context.step_index)
        self._last_action = int(TURN_RIGHT_ACTION_ID)
        self._trajectory_step_index += 1
        return self._last_action

    def get_debug_state(self) -> Dict[str, Any]:
        return {
            "goal_ids": list(self._goal_ids),
            "goal_modalities": dict(self._goal_modalities),
            "goal_embedding_dims": self._perception.goal_embedding_dims(),
            "audio_goal_labels": {k: list(v) for k, v in self._audio_goal_labels.items()},
            "audio_episode_labels": list(self._audio_episode_dict.keys()),
            "unsupported_audio_goals": list(self._unsupported_audio_goals),
            "map_summary": self._last_perception.map_state.summary() if self._last_perception is not None else {},
            "qwen_enabled": bool(self._qwen_enabled),
            "action": self._last_action,
            "info": self._last_info,
        }

    def _resolve_labels(self, payload: Dict[str, Any], inferred_nodes: Optional[Sequence[str]] = None) -> Tuple[str, ...]:
        modality = str(payload.get("modality", "")).lower()
        if modality == "object":
            return self._match_labels([payload.get("category")])
        if self._qwen_enabled and modality in {"description", "text", "image"}:
            labels = self._match_labels(inferred_nodes if inferred_nodes is not None else self._infer_graph_nodes(payload))
            if labels:
                return labels
        if modality in {"description", "text"}:
            return self._match_labels([payload.get("text")])
        return ()

    def _match_labels(self, values: Sequence[Any]) -> Tuple[str, ...]:
        hits: List[str] = []
        for value in values:
            label = self._match_label(value)
            if label and label not in hits:
                hits.append(label)
            if len(hits) >= 4:
                break
        return tuple(hits)

    def _match_label(self, value: Any) -> Optional[str]:
        text = normalize_text(value).lower().replace("_", " ")
        if not text:
            return None
        for alias, label in self._audio_prototype_library.alias_to_category.items():
            token = normalize_text(alias).lower().replace("_", " ")
            if token and token in text and label in self._audio_global:
                return str(label)
        label = self._audio_prototype_library.resolve_category(text)
        return str(label) if label in self._audio_global else None

    def _infer_graph_nodes(self, payload: Dict[str, Any]) -> List[str]:
        modality = str(payload.get("modality", "")).lower()
        prompt = (
            'Return JSON only as {"nodes": []}. '
            'Build a text scene graph using only the relation "next to". '
            'Put the most likely goal object in nodes[0]. '
            'Put nearby context objects in later nodes. '
            'Each node must be an object noun only, with no attributes, rooms, or explanations. '
            'If uncertain, still place the most likely goal first. '
            'If nothing is identifiable, return {"nodes": []}. '
            + (
                'For images, only use objects that are clearly visible and salient in the foreground. '
                'Prioritize movable, interactable, or furniture-like objects that would matter for navigation. '
                'Ignore background structure and scene layout elements such as wall, floor, ceiling, window, door, doorway, hallway, stairs, room, shadow, and lighting unless one of them is the main target. '
                'Prefer concrete foreground objects over structural context. '
                if modality == "image"
                else 'For text, infer the goal and nearby context objects from the description. '
            )
            + f"modality={payload.get('modality', '')}"
        )
        if modality in {"description", "text"}:
            prompt += f"\ntext={payload.get('text', '')}"
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        if modality == "image" and isinstance(payload.get("image"), np.ndarray):
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": image_array_to_data_url(np.asarray(payload["image"], dtype=np.uint8))},
                }
            )
        resp = chat_completion(
            api_base=self._qwen_api_base,
            api_key=self._qwen_api_key,
            model=self._qwen_model,
            messages=[{"role": "user", "content": content}],
            timeout_sec=self._qwen_timeout,
        )
        data = extract_json_object(extract_message_text(resp))
        nodes = data.get("nodes", []) if isinstance(data, dict) else []
        return [str(node) for node in nodes[:4]] if isinstance(nodes, list) else []
