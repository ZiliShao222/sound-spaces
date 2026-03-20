from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ss_baselines.omega_nav.perception.base import GoalSpec, PerceptionOutput
from ss_baselines.omega_nav.perception.clap_module import OracleCLAPMatcher
from ss_baselines.omega_nav.perception.clip_module import OracleCLIPDetector
from ss_baselines.omega_nav.perception.depth_module import OracleDepthProcessor
from ss_baselines.omega_nav.perception.vlm_module import OracleVLMDescriptor
from ss_baselines.omega_nav.utils import extract_pose, image_histogram_embedding, normalize_text, pose_to_position


class PerceptionEncoder:
    def __init__(self, config: Dict[str, Any]):
        self._config = config
        self._goal_encoder_cfg = dict(config.get("goal_encoder", {}))
        self._clip = OracleCLIPDetector({**config.get("visual", {}), **config.get("goal_encoder", {})})
        self._clap = OracleCLAPMatcher(config.get("audio", {}))
        self._depth = OracleDepthProcessor(config.get("depth", {}))
        self._vlm = OracleVLMDescriptor(config.get("visual", {}))
        self._prepared_goal_ids: Tuple[str, ...] = ()

    def reset(
        self,
        env: Optional[Any] = None,
        goal_specs: Optional[Sequence[GoalSpec]] = None,
        observations: Optional[Dict[str, Any]] = None,
    ) -> None:
        agent_position = None
        if env is not None and hasattr(env, "sim"):
            agent_position = np.asarray(env.sim.get_agent_state().position, dtype=np.float32)
        elif observations is not None:
            agent_position = pose_to_position(extract_pose(observations))
        self._depth.reset(agent_position)
        self._vlm.reset()
        self._prepared_goal_ids = ()
        self._clap.reset(goal_specs or ())
        if goal_specs:
            self._prepared_goal_ids = tuple(goal.goal_id for goal in goal_specs)

    def build_goal_specs(
        self,
        episode: Any,
        goal_payloads: Sequence[Dict[str, Any]],
    ) -> List[GoalSpec]:
        goal_specs: List[GoalSpec] = []
        task_specs = list(getattr(episode, "tasks", []) or [])
        if hasattr(episode, "goal_count"):
            episode_goal_count = int(getattr(episode, "goal_count") or 0)
        else:
            episode_goal_count = len(list(getattr(episode, "goals", []) or []))
        embedding_bins = max(int(self._goal_encoder_cfg.get("image_histogram_bins", 8)), 2)

        total = max(len(goal_payloads), len(task_specs), episode_goal_count)
        for goal_index in range(total):
            payload = goal_payloads[goal_index] if goal_index < len(goal_payloads) else {}
            task_spec = task_specs[goal_index] if goal_index < len(task_specs) else None

            modality = str(
                payload.get("modality")
                or (task_spec[1] if isinstance(task_spec, (list, tuple)) and len(task_spec) >= 2 else "object")
            )
            fallback_category = (
                task_spec[0]
                if isinstance(task_spec, (list, tuple)) and len(task_spec) >= 1
                else f"goal_{goal_index:03d}"
            )
            category = normalize_text(payload.get("category") or fallback_category)
            text_query = normalize_text(payload.get("text") or category)
            reference_image = payload.get("image") if isinstance(payload.get("image"), np.ndarray) else None
            image_description = normalize_text(
                payload.get("image_description")
                or (f"参考图像中的 {category}" if reference_image is not None else "")
            )
            image_embedding = (
                image_histogram_embedding(reference_image, embedding_bins)
                if reference_image is not None
                else None
            )

            goal_specs.append(
                GoalSpec(
                    goal_id=f"goal_{goal_index:03d}",
                    goal_index=int(goal_index),
                    modality=modality,
                    category=category or f"goal_{goal_index:03d}",
                    text_query=text_query or category or f"goal_{goal_index:03d}",
                    image_description=image_description,
                    image_embedding=image_embedding,
                    reference_image=reference_image,
                    semantic_id=None,
                    room_name=normalize_text(payload.get("room_name", "")),
                    sound_id=normalize_text(payload.get("sound_id", "")),
                    object_position=None,
                    sound_position=None,
                    view_positions=(),
                    metadata={
                        "instance_key": (
                            str(task_spec[0])
                            if isinstance(task_spec, (list, tuple)) and len(task_spec) >= 1
                            else ""
                        ),
                    },
                )
            )
        return goal_specs

    def encode(
        self,
        *,
        step_index: int,
        env: Any,
        episode: Any,
        observations: Dict[str, Any],
        goal_specs: Sequence[GoalSpec],
        pending_goal_ids: Optional[Sequence[str]] = None,
        order_mode: str = "ordered",
    ) -> PerceptionOutput:
        goal_ids = tuple(goal.goal_id for goal in goal_specs)
        if goal_ids != self._prepared_goal_ids:
            self._clap.reset(goal_specs)
            self._prepared_goal_ids = goal_ids

        semantic_map = self._depth.update(env, observations)
        visual_matches, top_clip_matches = self._clip.detect(observations, goal_specs)
        audio_matches = self._clap.match(
            env,
            episode,
            observations,
            goal_specs,
            pending_goal_ids=pending_goal_ids,
            order_mode=order_mode,
        )
        scene_description = self._vlm.describe(
            step_index=int(step_index),
            goal_specs=goal_specs,
            visual_matches=visual_matches,
            semantic_map=semantic_map,
        )

        clip_summary = ", ".join(
            f"{match.category}:{match.similarity:.2f}"
            for match in top_clip_matches
            if match.similarity > 0.0
        ) or "none"
        audio_detected = [match for match in audio_matches.values() if match.detected]
        audio_summary = ", ".join(
            f"{match.category}:{match.aggregated_similarity:.2f}@{match.direction_text}"
            for match in sorted(audio_detected, key=lambda item: float(item.aggregated_similarity), reverse=True)[:3]
        ) or "none"
        observation_summary = (
            f"visual={clip_summary} | audio={audio_summary} | explored={semantic_map.explored_ratio * 100.0:.1f}%"
        )

        return PerceptionOutput(
            step_index=int(step_index),
            scene_description=scene_description,
            visual_matches=visual_matches,
            audio_matches=audio_matches,
            top_clip_matches=top_clip_matches,
            semantic_map=semantic_map,
            observation_summary=observation_summary,
        )
