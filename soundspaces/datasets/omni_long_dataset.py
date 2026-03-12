#!/usr/bin/env python3

from __future__ import annotations

import gzip
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import attr

from habitat.config import Config
from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.core.simulator import AgentState, ShortestPathPoint

from soundspaces.tasks.omni_long_task import OmniLongEpisode
from soundspaces.tasks.semantic_audionav_task import ObjectViewLocation, SemanticAudioGoal


DEFAULT_SCENE_PATH_PREFIX = "data/scene_dataset/"


def _is_vec3(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(v, (int, float)) for v in value)
    )


def _safe_int_from_instance_key(instance_key: str) -> Optional[int]:
    if not isinstance(instance_key, str):
        return None
    token = instance_key.strip()
    if "_" not in token:
        return None
    suffix = token.split("_")[-1]
    if suffix.isdigit():
        return int(suffix)
    return None


def _normalize_task_specs(raw_tasks: Any, fallback_len: int) -> List[List[str]]:
    specs: List[List[str]] = []
    if isinstance(raw_tasks, list):
        for idx, item in enumerate(raw_tasks):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                specs.append([str(item[0]), str(item[1])])
                continue
            if isinstance(item, dict):
                instance_key = item.get("instance_key", item.get("goal", idx))
                modality = item.get("modality", "object")
                specs.append([str(instance_key), str(modality)])
                continue
            specs.append([str(idx), "object"])

    if fallback_len <= 0:
        return specs
    while len(specs) < fallback_len:
        specs.append([str(len(specs)), "object"])
    return specs[:fallback_len]


def _flatten_instances(raw_instances: Any) -> Dict[str, Dict[str, Any]]:
    flattened: Dict[str, Dict[str, Any]] = {}
    if not isinstance(raw_instances, dict):
        return flattened

    for key, value in raw_instances.items():
        if isinstance(value, dict) and "semantic_id" in value:
            flattened[str(key)] = value
            continue
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, dict):
                    flattened[str(nested_key)] = nested_value
    return flattened


def _goal_position_from_instance(record: Dict[str, Any]) -> List[float]:
    center = record.get("center")
    if _is_vec3(center):
        return [float(v) for v in center]

    nav_position = record.get("nav_position")
    if _is_vec3(nav_position):
        return [float(v) for v in nav_position]

    image_payload = record.get("image")
    if isinstance(image_payload, dict):
        render_views = image_payload.get("render_views")
        if isinstance(render_views, list) and render_views:
            first = render_views[0]
            if isinstance(first, dict):
                candidate = first.get("position")
                if _is_vec3(candidate):
                    return [float(v) for v in candidate]

    return [0.0, 0.0, 0.0]


def _view_point_from_instance(record: Dict[str, Any], goal_position: List[float]) -> List[float]:
    image_payload = record.get("image")
    if isinstance(image_payload, dict):
        render_views = image_payload.get("render_views")
        if isinstance(render_views, list) and render_views:
            first = render_views[0]
            if isinstance(first, dict):
                candidate = first.get("agent_base_position")
                if _is_vec3(candidate):
                    return [float(v) for v in candidate]
                candidate = first.get("position")
                if _is_vec3(candidate):
                    return [float(v) for v in candidate]

    nav_position = record.get("nav_position")
    if _is_vec3(nav_position):
        return [float(v) for v in nav_position]

    return [float(v) for v in goal_position]


def _build_goal_dict_from_instance(instance_key: str, record: Dict[str, Any]) -> Dict[str, Any]:
    category = record.get("category")
    if not isinstance(category, str) or not category.strip():
        category = "unknown"

    semantic_id = record.get("semantic_id")
    if not isinstance(semantic_id, int):
        semantic_id = _safe_int_from_instance_key(instance_key)
    if semantic_id is None:
        semantic_id = 0

    goal_position = _goal_position_from_instance(record)
    view_point = _view_point_from_instance(record, goal_position)

    return {
        "position": [float(v) for v in goal_position],
        "radius": 1e-5,
        "object_id": str(semantic_id),
        "object_id_raw": str(semantic_id),
        "object_name": None,
        "object_category": category,
        "room_id": None,
        "room_name": None,
        "view_points": [view_point],
    }


def _coerce_float(value: Any, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except Exception:
        return float(default)


def _coerce_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    try:
        return int(value)
    except Exception:
        return int(default)


@registry.register_dataset(name="OmniLongNav")
class OmniLongNavDataset(Dataset):
    episodes: List[OmniLongEpisode]

    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []
        self._config = config

        if config is None:
            return

        dataset_path = self._resolve_dataset_path(config)
        with gzip.open(dataset_path, "rt") as handle:
            self.from_json(handle.read(), scenes_dir=config.SCENES_DIR, scene_filename=dataset_path)

    @staticmethod
    def _resolve_dataset_path(config: Config) -> str:
        data_path = str(config.DATA_PATH)
        try:
            return data_path.format(version=config.VERSION, split=config.SPLIT)
        except Exception:
            return data_path

    @staticmethod
    def __deserialize_goal(serialized_goal: Dict[str, Any]) -> SemanticAudioGoal:
        goal = SemanticAudioGoal(**serialized_goal)
        for view_idx, view in enumerate(goal.view_points or []):
            view_location = ObjectViewLocation(view, iou=0)
            view_location.agent_state = AgentState(view_location.agent_state)
            goal.view_points[view_idx] = view_location
        return goal

    def _build_goals_and_tasks(
        self,
        raw_episode: Dict[str, Any],
        instances: Dict[str, Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], List[List[str]]]:
        raw_tasks = raw_episode.get("tasks")
        raw_goals = raw_episode.get("goals")

        if isinstance(raw_tasks, list):
            task_specs = _normalize_task_specs(raw_tasks, fallback_len=0)
        elif isinstance(raw_goals, list) and raw_goals and isinstance(raw_goals[0], (list, tuple, dict)):
            task_specs = _normalize_task_specs(raw_goals, fallback_len=0)
        else:
            task_specs = []

        goal_dicts: List[Dict[str, Any]] = []
        if task_specs:
            aligned_task_specs: List[List[str]] = []
            for spec in task_specs:
                instance_key = str(spec[0])
                record = instances.get(instance_key)
                if not isinstance(record, dict):
                    continue
                goal_dicts.append(_build_goal_dict_from_instance(instance_key, record))
                aligned_task_specs.append([str(spec[0]), str(spec[1])])
            task_specs = aligned_task_specs

        if not goal_dicts and isinstance(raw_goals, list) and raw_goals and isinstance(raw_goals[0], dict):
            for raw_goal in raw_goals:
                if isinstance(raw_goal, dict):
                    goal_dicts.append(raw_goal)
            if goal_dicts:
                task_specs = _normalize_task_specs(
                    [[g.get("object_id", idx), "object"] for idx, g in enumerate(goal_dicts)],
                    fallback_len=len(goal_dicts),
                )

        task_specs = _normalize_task_specs(task_specs, fallback_len=len(goal_dicts))
        return goal_dicts, task_specs

    def from_json(
        self,
        json_str: str,
        scenes_dir: Optional[str] = None,
        scene_filename: Optional[str] = None,
    ) -> None:
        deserialized = json.loads(json_str)
        if not isinstance(deserialized, dict):
            return

        raw_episodes = deserialized.get("episodes")
        if not isinstance(raw_episodes, list) or len(raw_episodes) == 0:
            return

        instances = _flatten_instances(deserialized.get("instances"))
        allowed_episode_keys = set(attr.fields_dict(OmniLongEpisode).keys())

        for idx, raw_episode in enumerate(raw_episodes):
            if not isinstance(raw_episode, dict):
                continue

            goal_dicts, task_specs = self._build_goals_and_tasks(raw_episode, instances)
            if not goal_dicts:
                continue

            episode_dict = dict(raw_episode)
            episode_dict["goals"] = goal_dicts
            episode_dict["tasks"] = task_specs
            episode_dict["num_goals"] = int(episode_dict.get("num_goals", len(goal_dicts)))

            if not isinstance(episode_dict.get("object_category"), str) or not episode_dict.get("object_category"):
                episode_dict["object_category"] = str(goal_dicts[0].get("object_category", "unknown"))

            if not isinstance(episode_dict.get("sound_id"), str) or not episode_dict.get("sound_id"):
                episode_dict["sound_id"] = f"val/{episode_dict['object_category']}.wav"

            if not isinstance(episode_dict.get("offset"), (int, str)):
                episode_dict["offset"] = 0
            if not isinstance(episode_dict.get("duration"), (int, str)):
                episode_dict["duration"] = 25

            if not isinstance(episode_dict.get("sound_sources"), list) or not episode_dict.get("sound_sources"):
                sound_id = str(episode_dict["sound_id"])
                episode_dict["sound_sources"] = [{"sound_id": sound_id} for _ in range(len(goal_dicts))]
            if not isinstance(episode_dict.get("sound_source_schedule"), list):
                episode_dict["sound_source_schedule"] = ["round_robin", 25]

            info = episode_dict.get("info")
            if not isinstance(info, dict):
                info = {}

            geod = _coerce_float(
                info.get("geodesic_distance", episode_dict.get("ordered_total_geodesic_distance", 0.0)),
                default=0.0,
            )
            if geod <= 1e-6:
                geod = max(
                    _coerce_float(episode_dict.get("ordered_total_geodesic_distance"), 0.0),
                    _coerce_float(episode_dict.get("unordered_total_geodesic_distance"), 0.0),
                    1e-6,
                )

            info["geodesic_distance"] = float(geod)
            info["num_action"] = _coerce_int(info.get("num_action", 0), default=0)

            per_goal = info.get("per_goal")
            if not isinstance(per_goal, list) or len(per_goal) == 0:
                info["per_goal"] = [
                    {
                        "object_id": str(goal.get("object_id", gi)),
                        "object_category": str(goal.get("object_category", "unknown")),
                        "geodesic_distance": float(geod),
                        "num_action": 0,
                    }
                    for gi, goal in enumerate(goal_dicts)
                ]

            episode_dict["info"] = info

            filtered_episode = {
                key: value for key, value in episode_dict.items() if key in allowed_episode_keys
            }

            episode = OmniLongEpisode(**filtered_episode)

            if self._config is not None and hasattr(self._config, "SCENES_DIR"):
                episode.scene_dataset_config = str(self._config.SCENES_DIR).split("/")[-1]

            if scenes_dir is not None:
                scene_id = str(episode.scene_id)
                if scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    scene_id = scene_id[len(DEFAULT_SCENE_PATH_PREFIX) :]
                if not os.path.isabs(scene_id):
                    scene_id = os.path.join(scenes_dir, scene_id)
                episode.scene_id = scene_id

            for goal_idx, goal in enumerate(episode.goals):
                if isinstance(goal, SemanticAudioGoal):
                    continue
                if isinstance(goal, dict):
                    episode.goals[goal_idx] = self.__deserialize_goal(goal)

            if episode.shortest_paths is not None:
                for path in episode.shortest_paths:
                    for point_idx, point in enumerate(path):
                        path[point_idx] = ShortestPathPoint(**point)

            self.episodes.append(episode)
