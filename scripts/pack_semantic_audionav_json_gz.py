#!/usr/bin/env python3

"""Normalize episode JSON and pack into .json.gz for SemanticAudioNav.

Supported input formats:
1) trajectory_dataset.json produced by build_trajectories_from_valid_instances.py
2) already episode-style JSON with top-level "episodes"
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _is_vec3(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(v, (int, float)) for v in value)
    )


def _is_quat(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 4
        and all(isinstance(v, (int, float)) for v in value)
    )


def _infer_scene_split(scene_id: Any, fallback: str = "val") -> str:
    if not isinstance(scene_id, str) or not scene_id.strip():
        return fallback
    parts = [part.strip() for part in scene_id.split("/") if part.strip()]
    for split in ("train", "val", "test", "minival"):
        if split in parts:
            return split
    return fallback


def _normalize_scene_id(scene_id: Any) -> str:
    if not isinstance(scene_id, str):
        return ""
    if scene_id.startswith("mp3d/"):
        return scene_id[len("mp3d/") :]
    return scene_id


def _safe_int_from_key(instance_key: str) -> Optional[int]:
    match = re.search(r"_(\d+)$", instance_key)
    if not match:
        return None
    return int(match.group(1))


def _goal_position_from_instance(instance_record: Dict[str, Any]) -> List[float]:
    center = instance_record.get("center")
    if _is_vec3(center):
        return [float(v) for v in center]

    nav_position = instance_record.get("nav_position")
    if _is_vec3(nav_position):
        return [float(v) for v in nav_position]

    image_payload = instance_record.get("image")
    if isinstance(image_payload, dict):
        render_views = image_payload.get("render_views")
        if isinstance(render_views, list) and render_views:
            first = render_views[0]
            if isinstance(first, dict):
                for key in ("agent_base_position", "position"):
                    value = first.get(key)
                    if _is_vec3(value):
                        return [float(v) for v in value]

    return [0.0, 0.0, 0.0]


def _view_point_from_instance(instance_record: Dict[str, Any]) -> List[float]:
    image_payload = instance_record.get("image")
    if isinstance(image_payload, dict):
        render_views = image_payload.get("render_views")
        if isinstance(render_views, list) and render_views:
            first = render_views[0]
            if isinstance(first, dict):
                for key in ("agent_base_position", "position"):
                    value = first.get(key)
                    if _is_vec3(value):
                        return [float(v) for v in value]

    nav_position = instance_record.get("nav_position")
    if _is_vec3(nav_position):
        return [float(v) for v in nav_position]

    center = instance_record.get("center")
    if _is_vec3(center):
        return [float(v) for v in center]

    return [0.0, 0.0, 0.0]


def _build_sound_sources(
    trajectory: Dict[str, Any],
    categories: List[str],
    scene_split: str,
) -> List[Dict[str, str]]:
    sound_sources = trajectory.get("sound_sources")
    if isinstance(sound_sources, list) and sound_sources:
        normalized: List[Dict[str, str]] = []
        for idx, source in enumerate(sound_sources):
            if not isinstance(source, dict):
                continue
            sound_id = source.get("sound_id")
            if isinstance(sound_id, str) and sound_id.strip():
                normalized.append({"sound_id": sound_id})
            elif idx < len(categories):
                normalized.append({"sound_id": f"{scene_split}/{categories[idx]}.wav"})
        if normalized:
            return normalized

    return [{"sound_id": f"{scene_split}/{category}.wav"} for category in categories]


def _normalize_episode_dict(
    episode: Dict[str, Any],
    idx: int,
    default_scene_id: str,
    default_scene_split: str,
) -> Dict[str, Any]:
    goals: List[Dict[str, Any]] = []
    raw_goals = episode.get("goals")
    if isinstance(raw_goals, list):
        for goal in raw_goals:
            if not isinstance(goal, dict):
                continue
            position = goal.get("position")
            if not _is_vec3(position):
                position = [0.0, 0.0, 0.0]
            view_points = goal.get("view_points")
            if not (
                isinstance(view_points, list)
                and view_points
                and all(_is_vec3(v) for v in view_points)
            ):
                view_points = [[float(v) for v in position]]
            object_category = goal.get("object_category")
            if not isinstance(object_category, str) or not object_category:
                object_category = "unknown"
            object_id = goal.get("object_id")
            if object_id is None:
                object_id = str(idx)

            goals.append(
                {
                    "position": [float(v) for v in position],
                    "radius": float(goal.get("radius", 1e-5)),
                    "object_id": str(object_id),
                    "object_id_raw": str(goal.get("object_id_raw", object_id)),
                    "object_name": goal.get("object_name"),
                    "object_category": object_category,
                    "room_id": goal.get("room_id"),
                    "room_name": goal.get("room_name"),
                    "view_points": [[float(v) for v in view_points[0]]],
                }
            )

    start_position = episode.get("start_position")
    if not _is_vec3(start_position):
        start_position = [0.0, 0.0, 0.0]

    start_rotation = episode.get("start_rotation")
    if not _is_quat(start_rotation):
        start_rotation = [0.0, 0.0, 0.0, 1.0]

    if goals:
        first_category = goals[0]["object_category"]
    else:
        first_category = "unknown"

    sound_sources = episode.get("sound_sources")
    if not isinstance(sound_sources, list) or not sound_sources:
        sound_sources = [{"sound_id": f"{default_scene_split}/{first_category}.wav"}]
    normalized_sources: List[Dict[str, str]] = []
    for source in sound_sources:
        if isinstance(source, dict) and isinstance(source.get("sound_id"), str):
            normalized_sources.append({"sound_id": source["sound_id"]})
    if not normalized_sources:
        normalized_sources = [{"sound_id": f"{default_scene_split}/{first_category}.wav"}]

    info = episode.get("info")
    geodesic_distance = 0.0
    if isinstance(info, dict) and isinstance(info.get("geodesic_distance"), (int, float)):
        geodesic_distance = float(info["geodesic_distance"])

    per_goal = []
    if isinstance(info, dict) and isinstance(info.get("per_goal"), list):
        for row in info["per_goal"]:
            if isinstance(row, dict):
                per_goal.append(
                    {
                        "object_id": str(row.get("object_id", "0")),
                        "object_category": str(row.get("object_category", "unknown")),
                        "geodesic_distance": float(row.get("geodesic_distance", 0.0)),
                        "num_action": int(row.get("num_action", 0)),
                    }
                )

    if not per_goal and goals:
        per_goal = [
            {
                "object_id": str(goal["object_id"]),
                "object_category": str(goal["object_category"]),
                "geodesic_distance": geodesic_distance,
                "num_action": 0,
            }
            for goal in goals
        ]

    episode_id = str(episode.get("episode_id", idx))
    if not episode_id.isdigit():
        episode_id = str(idx)

    normalized = {
        "episode_id": episode_id,
        "scene_id": _normalize_scene_id(episode.get("scene_id") or default_scene_id),
        "start_position": [float(v) for v in start_position],
        "start_rotation": [float(v) for v in start_rotation],
        "num_goals": int(episode.get("num_goals", len(goals))),
        "info": {
            "geodesic_distance": geodesic_distance,
            "num_action": int(info.get("num_action", 0)) if isinstance(info, dict) else 0,
            "per_goal": per_goal,
        },
        "goals": goals,
        "start_room": episode.get("start_room"),
        "shortest_paths": episode.get("shortest_paths"),
        "object_category": str(episode.get("object_category", first_category)),
        "sound_id": str(
            episode.get("sound_id", normalized_sources[0]["sound_id"])
        ),
        "offset": str(episode.get("offset", "0")),
        "duration": str(episode.get("duration", "25")),
        "sound_sources": normalized_sources,
        "sound_source_schedule": episode.get("sound_source_schedule", ["round_robin", 25]),
    }
    return normalized


def _convert_trajectory_dataset(
    payload: Dict[str, Any],
    force_scene_split: Optional[str],
) -> Tuple[List[Dict[str, Any]], str]:
    trajectories = payload.get("trajectories")
    instances = payload.get("instances")
    if not isinstance(trajectories, list) or not isinstance(instances, dict):
        raise RuntimeError(
            "Trajectory input must contain top-level 'trajectories' (list) and 'instances' (dict)."
        )

    raw_scene_id = payload.get("scene_id")
    normalized_scene_id = _normalize_scene_id(raw_scene_id)
    scene_split = force_scene_split or _infer_scene_split(normalized_scene_id)

    episodes: List[Dict[str, Any]] = []
    for idx, trajectory in enumerate(trajectories):
        if not isinstance(trajectory, dict):
            continue

        start_state = trajectory.get("start_state")
        start_position = [0.0, 0.0, 0.0]
        start_rotation = [0.0, 0.0, 0.0, 1.0]
        if isinstance(start_state, dict):
            if _is_vec3(start_state.get("position")):
                start_position = [float(v) for v in start_state["position"]]
            if _is_quat(start_state.get("rotation")):
                start_rotation = [float(v) for v in start_state["rotation"]]

        goal_keys = trajectory.get("goals")
        if not isinstance(goal_keys, list):
            goal_keys = []

        per_goal_distances = trajectory.get("start_to_goal_ground_distances")
        if not isinstance(per_goal_distances, list):
            per_goal_distances = []

        goals: List[Dict[str, Any]] = []
        per_goal_info: List[Dict[str, Any]] = []
        categories: List[str] = []

        for goal_i, key in enumerate(goal_keys):
            if not isinstance(key, str):
                continue

            instance_record = instances.get(key)
            if not isinstance(instance_record, dict):
                continue

            category = instance_record.get("category")
            if not isinstance(category, str) or not category:
                category = "unknown"
            categories.append(category)

            semantic_id = instance_record.get("semantic_id")
            if not isinstance(semantic_id, int):
                parsed_id = _safe_int_from_key(key)
                semantic_id = parsed_id if parsed_id is not None else goal_i

            goal_position = _goal_position_from_instance(instance_record)
            view_point = _view_point_from_instance(instance_record)

            goals.append(
                {
                    "position": goal_position,
                    "radius": 1e-5,
                    "object_id": str(semantic_id),
                    "object_id_raw": str(semantic_id),
                    "object_name": None,
                    "object_category": category,
                    "room_id": None,
                    "room_name": None,
                    "view_points": [view_point],
                }
            )

            geod = (
                float(per_goal_distances[goal_i])
                if goal_i < len(per_goal_distances)
                and isinstance(per_goal_distances[goal_i], (int, float))
                else 0.0
            )
            per_goal_info.append(
                {
                    "object_id": str(semantic_id),
                    "object_category": category,
                    "geodesic_distance": geod,
                    "num_action": 0,
                }
            )

        if not goals:
            continue

        sound_sources = _build_sound_sources(
            trajectory=trajectory,
            categories=categories,
            scene_split=scene_split,
        )
        sound_id = trajectory.get("sound_id")
        if not isinstance(sound_id, str) or not sound_id.strip():
            sound_id = sound_sources[0]["sound_id"]

        episode = {
            "episode_id": str(idx),
            "scene_id": normalized_scene_id,
            "start_position": start_position,
            "start_rotation": start_rotation,
            "num_goals": int(trajectory.get("num_goals", len(goals))),
            "info": {
                "geodesic_distance": float(per_goal_info[0]["geodesic_distance"]),
                "num_action": 0,
                "per_goal": per_goal_info,
            },
            "goals": goals,
            "start_room": None,
            "shortest_paths": None,
            "object_category": str(categories[0]),
            "sound_id": str(sound_id),
            "offset": str(trajectory.get("offset", "0")),
            "duration": str(trajectory.get("duration", "25")),
            "sound_sources": sound_sources,
            "sound_source_schedule": trajectory.get("sound_source_schedule", ["round_robin", 25]),
        }
        episodes.append(episode)

    return episodes, normalized_scene_id


def _normalize_episode_payload(
    payload: Dict[str, Any],
    force_scene_split: Optional[str],
) -> Tuple[List[Dict[str, Any]], str]:
    raw_episodes = payload.get("episodes")
    if not isinstance(raw_episodes, list):
        raise RuntimeError("Episode input must contain top-level 'episodes' list.")

    raw_scene_id = payload.get("scene_id")
    normalized_scene_id = _normalize_scene_id(raw_scene_id)
    if not normalized_scene_id and isinstance(payload.get("scene"), str):
        scene = str(payload.get("scene"))
        split = force_scene_split or "val"
        normalized_scene_id = f"{split}/{scene}/{scene}.glb"

    scene_split = force_scene_split or _infer_scene_split(normalized_scene_id)

    episodes = [
        _normalize_episode_dict(
            episode=episode,
            idx=idx,
            default_scene_id=normalized_scene_id,
            default_scene_split=scene_split,
        )
        for idx, episode in enumerate(raw_episodes)
        if isinstance(episode, dict)
    ]
    return episodes, normalized_scene_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize JSON and pack to SemanticAudioNav .json.gz"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSON path (trajectory_dataset.json or episodes JSON)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .json.gz path (default: <input_stem>.json.gz)",
    )
    parser.add_argument(
        "--scene-split",
        type=str,
        default=None,
        choices=["train", "val", "test", "minival"],
        help="Force sound split prefix and fallback scene split",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty JSON indentation in gzip output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise RuntimeError(f"Input JSON not found: {input_path}")

    if args.output is not None:
        output_path = args.output.expanduser().resolve()
    else:
        if input_path.suffix.lower() == ".json":
            output_path = input_path.with_suffix(".json.gz")
        else:
            output_path = input_path.parent / f"{input_path.name}.json.gz"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Input root must be a JSON object.")

    if isinstance(payload.get("trajectories"), list) and isinstance(
        payload.get("instances"), dict
    ):
        episodes, normalized_scene_id = _convert_trajectory_dataset(
            payload=payload,
            force_scene_split=args.scene_split,
        )
    elif isinstance(payload.get("episodes"), list):
        episodes, normalized_scene_id = _normalize_episode_payload(
            payload=payload,
            force_scene_split=args.scene_split,
        )
    else:
        raise RuntimeError(
            "Unrecognized input format. Expected either trajectory dataset or episodes dataset."
        )

    scene_name = "unknown_scene"
    if normalized_scene_id:
        scene_name = Path(normalized_scene_id).parent.name or scene_name
    elif isinstance(payload.get("scene"), str):
        scene_name = str(payload.get("scene"))

    out_payload = {
        "episodes": episodes,
        "scene": scene_name,
    }

    with gzip.open(output_path, "wt", encoding="utf-8") as handle:
        json.dump(
            out_payload,
            handle,
            ensure_ascii=False,
            indent=2 if args.pretty else None,
        )

    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    print(f"Episodes: {len(episodes)}")
    if episodes:
        print(f"Scene id example: {episodes[0].get('scene_id')}")


if __name__ == "__main__":
    main()

