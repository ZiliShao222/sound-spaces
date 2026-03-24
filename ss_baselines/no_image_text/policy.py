from __future__ import annotations
import json
from pathlib import Path
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
from ss_baselines.omega_nav.utils import load_omega_config, normalize_text

@register_lifelong_eval_policy("no_image_text_omega_nav")
@register_lifelong_eval_policy("no_image_text_omega_nav_policy")
@register_lifelong_eval_policy("no_image_text_omega_oracle")
@register_lifelong_eval_policy("no_image_text_omega_nav_oracle")
class NoImageTextOmegaNavPolicy(LifelongEvalPolicy):
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
        self._device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self._look_around_steps = max(int(look_around_steps), 0)
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
        self._perception.set_audio_runtime(self._audio_prototype_library)
        labels = tuple(self._audio_prototype_library.prototype_labels)
        matrix = np.asarray(self._audio_prototype_library.prototype_matrix, dtype=np.float32)
        self._audio_global = {str(label): np.asarray(matrix[i], dtype=np.float32) for i, label in enumerate(labels)}
        self._audio_episode_dict: Dict[str, Any] = {}
        self._unsupported_audio_goals: List[Dict[str, Any]] = []
        self.reset(env=None, episode=None, observations={})
        
    def reset(self, *, env: Any, episode: Any, observations: Dict[str, Any]) -> None:
        del env, episode, observations
        self._goal_ids: Tuple[str, ...] = ()
        self._goal_modalities: Dict[str, str] = {}
        self._goal_jsons: Dict[str, Dict[str, Any]] = {}
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
        del env, order_mode
        self.reset(env=None, episode=None, observations={})
        self._perception.reset(observations)
        self._goal_ids = tuple(f"goal_{i:03d}" for i in range(len(goal_payloads)))
        self._goal_embeddings = self._perception.encode_goals(self._goal_ids, goal_payloads)

        def build_goal_json(payload: Dict[str, Any]) -> Dict[str, Any]:
            modality = str(payload.get("modality", "")).lower() or "object"

            # no_image_text 只处理 object 模态
            labels = (
                tuple(str(payload.get("category", "")).strip() for _ in [0] if str(payload.get("category", "")).strip())
                if modality == "object"
                else ()
            )

            goal_category = labels[0] if labels else None
            other_objects = []

            return {
                "modality": modality,
                "goal_category": goal_category,
                "other_objects": other_objects,
            }

        for goal_id, payload in zip(self._goal_ids, goal_payloads):
            modality = str(payload.get("modality", "")).lower() or "object"
            self._goal_modalities[goal_id] = modality
            goal_json = build_goal_json(payload)
            self._goal_jsons[goal_id] = goal_json
            goal_category = goal_json.get("goal_category")
            other_objects = goal_json.get("other_objects", [])
            raw_audio_values = [str(goal_category)] + [
                str(item.get("category")) for item in other_objects if item.get("category")
            ]
            audio_labels = self._match_labels(raw_audio_values)
            if not goal_category:
                self._unsupported_audio_goals.append(
                    {
                        "goal_id": goal_id,
                        "modality": modality,
                        "has_embedding": goal_id in self._goal_embeddings,
                    }
                )
                continue
            if not audio_labels:
                self._unsupported_audio_goals.append(
                    {
                        "goal_id": goal_id,
                        "modality": modality,
                        "has_embedding": goal_id in self._goal_embeddings,
                    }
                )
                continue

            self._audio_goal_labels[goal_id] = (str(audio_labels[0]),)
            episode_labels = list(audio_labels)
            for label in episode_labels:
                self._audio_episode_dict.setdefault(label, self._audio_global[label])

        for label in self._resolve_episode_audio_labels(episode):
            self._audio_episode_dict.setdefault(label, self._audio_global[label])

        self._perception.set_audio_candidates(self._audio_episode_dict)
        self._perception.set_audio_goal_cycle(self._goal_ids, self._audio_goal_labels)
        
        print(
            "[no_image_text_omega_nav][start_episode] "
            + json.dumps(
                {
                    "audio_episode_labels": list(self._audio_episode_dict.keys()),
                },
                ensure_ascii=False,
            )
        )
        
    def act(self, *, env: Any, episode: Any, observations: Dict[str, Any], context: LifelongEvalContext) -> Any:
        del env, episode
        self._last_info = context.info
        self._last_perception = self._perception.percept(observations, context.step_index)

        # 使用audio_bearing进行导航决策
        bearing_estimate = self._last_perception.audio_state.bearing
        # 根据音频方位估计选择动作
        self._last_action = self._select_action_from_bearing(bearing_estimate)

        self._trajectory_step_index += 1
        return self._last_action

    def _select_action_from_bearing(self, bearing_estimate: Any) -> int:
        if bearing_estimate.is_valid and bearing_estimate.confidence > 0.3:
            relative_bearing_deg = float(bearing_estimate.relative_bearing_deg)
            return self._direction_to_action(relative_bearing_deg)

        return int(TURN_RIGHT_ACTION_ID)

    def _direction_to_action(self, relative_bearing_deg: float) -> int:
        """Convert relative bearing (degrees) to action ID."""
        # 使用omega_nav/utils.py中的coarse_direction_from_angle函数来将相对方位角转换为粗略的方向类别
        from ss_baselines.omega_nav.utils import coarse_direction_from_angle

        direction = coarse_direction_from_angle(relative_bearing_deg)

        if direction == "forward":
            return 1  # MOVE_FORWARD
        elif direction == "left":
            return 2  # TURN_LEFT
        elif direction == "right":
            return 3  # TURN_RIGHT
        elif direction == "behind":
            return 3  # TURN_RIGHT

        return 1  # MOVE_FORWARD

    def get_debug_state(self) -> Dict[str, Any]:
        return {
            "goal_ids": list(self._goal_ids),
            "goal_modalities": dict(self._goal_modalities),
            "goal_jsons": dict(self._goal_jsons),
            "goal_embedding_dims": self._perception.goal_embedding_dims(),
            "audio_goal_labels": {k: list(v) for k, v in self._audio_goal_labels.items()},
            "audio_episode_labels": list(self._audio_episode_dict.keys()),
            "unsupported_audio_goals": list(self._unsupported_audio_goals),
            "map_summary": self._last_perception.map_state.summary() if self._last_perception is not None else {},
            "audio_summary": self._last_perception.audio_state.summary() if self._last_perception is not None else {},
            "audio_tracker": self._perception.audio_debug_state(),
            "action": self._last_action,
            "info": self._last_info,
        }
    
    def _resolve_episode_audio_labels(self, episode: Any) -> Tuple[str, ...]:
        sound_sources = getattr(episode, "sound_sources", None)
        if not isinstance(sound_sources, list):
            return ()
        values: List[Any] = []
        for source in sound_sources:
            if not isinstance(source, dict):
                continue
            sound_id = source.get("sound_id")
            if isinstance(sound_id, str) and sound_id.strip():
                values.append(Path(sound_id).stem)
            values.append(source.get("category"))
            values.append(source.get("object_category"))
        return self._match_labels(values)
    
    def _resolve_labels(self, payload: Dict[str, Any]) -> Tuple[str, ...]:
        modality = str(payload.get("modality", "")).lower()
        if modality == "object":
            return self._match_labels([payload.get("category")])
        if modality == "image":
            return self._match_labels(payload.get("detected_objects", []))
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
