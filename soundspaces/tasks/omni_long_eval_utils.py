#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import habitat
from habitat.config import Config
import numpy as np
from PIL import Image
import soundspaces  # noqa: F401 - register datasets/tasks/sims
import yaml
from habitat.utils.visualizations.utils import images_to_video

from ss_baselines.omni_long.config.default import get_task_config
from ss_baselines.common.omni_long_eval_policy import (
    build_lifelong_eval_context,
    build_lifelong_eval_policy,
    list_lifelong_eval_policies,
)
from ss_baselines.common.utils import observations_to_image, images_to_video_with_audio


DEFAULT_CONFIG = "configs/omni-long/mp3d/omni-long_semantic_audio.yaml"
DEFAULT_TASK_TYPE = "OmniLongSemanticAudioNav"
DEFAULT_EXP_NAME = "exp_eval_omni-long"
DEFAULT_OUTPUT_PARENT_DIR = "results"


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)
        if np.isfinite(value):
            return value
    return None


def _is_vec3(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 3
        and all(isinstance(v, (int, float)) for v in value)
    )


def _is_quat4(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(isinstance(v, (int, float)) for v in value)
    )


def _load_dataset_payload(dataset_path: str) -> Dict[str, Any]:
    p = Path(dataset_path)
    if p.suffix.lower() == ".gz":
        with gzip.open(p, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        with open(p, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    if not isinstance(payload, dict):
        return {}
    return payload


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


def _parse_image_modality_index(modality: str) -> Optional[int]:
    token = str(modality).strip().lower()
    if token == "image":
        return 0
    if token.startswith("image_"):
        suffix = token[len("image_") :]
        if suffix.isdigit():
            return int(suffix)
    return None


def _extract_text_description(instance_record: Dict[str, Any]) -> Optional[str]:
    for key in ("text_description", "description", "text", "caption"):
        value = instance_record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_rgb_from_observations(observations: Dict[str, Any]) -> Optional[np.ndarray]:
    for key in ("rgb", "RGB_SENSOR", "rgb_sensor"):
        if key in observations:
            candidate = observations[key]
            arr = np.asarray(candidate)
            if arr.ndim == 3:
                if arr.shape[2] == 4:
                    arr = arr[:, :, :3]
                elif arr.shape[2] == 1:
                    arr = np.repeat(arr, 3, axis=2)
                if arr.dtype != np.uint8:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                return arr
    for value in observations.values():
        arr = np.asarray(value)
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            if arr.shape[2] == 4:
                arr = arr[:, :, :3]
            if arr.dtype != np.uint8:
                arr = np.clip(arr, 0, 255).astype(np.uint8)
            return arr
    return None


def _render_reference_image(
    env: habitat.Env,
    instance_record: Dict[str, Any],
    image_index: int,
) -> Optional[np.ndarray]:
    image_payload = instance_record.get("image")
    render_views = None
    if isinstance(image_payload, dict):
        render_views = image_payload.get("render_views")
    if not isinstance(render_views, list) or len(render_views) == 0:
        render_views = instance_record.get("render_view_points")
    if not isinstance(render_views, list) or len(render_views) == 0:
        return None
    if image_index < 0 or image_index >= len(render_views):
        return None

    render_view = render_views[image_index]
    if not isinstance(render_view, dict):
        return None

    position = render_view.get("agent_base_position")
    if not _is_vec3(position):
        position = render_view.get("position")
    rotation = render_view.get("rotation")
    agent_state = render_view.get("agent_state")
    if isinstance(agent_state, dict):
        if not _is_vec3(position):
            position = agent_state.get("position")
        if not _is_quat4(rotation):
            rotation = agent_state.get("rotation")
    if not _is_vec3(position) or not _is_quat4(rotation):
        return None

    observations = env.sim.get_observations_at(
        position=[float(v) for v in position],
        rotation=[float(v) for v in rotation],
        keep_agent_at_new_pose=False,
    )

    if observations is None:
        return None
    return _extract_rgb_from_observations(observations)


def _sound_id_for_goal(episode: Any, goal_index: int) -> Optional[str]:
    sound_sources = getattr(episode, "sound_sources", None)
    if isinstance(sound_sources, list) and 0 <= goal_index < len(sound_sources):
        src = sound_sources[goal_index]
        if isinstance(src, dict):
            sid = src.get("sound_id")
            if isinstance(sid, str) and sid.strip():
                return sid.strip()
    sid = getattr(episode, "sound_id", None)
    if isinstance(sid, str) and sid.strip():
        return sid.strip()
    return None


def _normalize_goal_input_modality(modality: str) -> str:
    token = str(modality).strip().lower()
    if token.startswith("image"):
        return "image"
    if token in {"description", "text", "text_description"}:
        return "description"
    return "object"


def _fallback_object_category(instance_key: str) -> str:
    token = str(instance_key).strip()
    if "_" in token:
        prefix, suffix = token.rsplit("_", 1)
        if suffix.isdigit() and prefix:
            return prefix
    return token or "object"


def _build_goal_input_payload(
    env: habitat.Env,
    instance_key: str,
    modality: str,
    instance_record: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    normalized_modality = _normalize_goal_input_modality(modality)
    if normalized_modality == "image":
        image_index = _parse_image_modality_index(modality)
        rgb = None
        if isinstance(instance_record, dict):
            rgb = _render_reference_image(
                env,
                instance_record,
                0 if image_index is None else int(image_index),
            )
        return {
            "modality": "image",
            "image": np.asarray(rgb, dtype=np.uint8).copy() if rgb is not None else None,
        }

    if normalized_modality == "description":
        description = None
        if isinstance(instance_record, dict):
            description = _extract_text_description(instance_record)
        return {
            "modality": "description",
            "text": str(description).strip() if isinstance(description, str) else "",
        }

    category = None
    if isinstance(instance_record, dict):
        category = instance_record.get("category")
    if not isinstance(category, str) or not category.strip():
        category = _fallback_object_category(instance_key)

    return {
        "modality": "object",
        "category": str(category).strip(),
    }


def _save_goal_input_image_if_needed(
    payload: Dict[str, Any],
    prompt_image_dir: Optional[Path],
    episode_id: str,
    goal_index: int,
    instance_key: str,
    modality: str,
) -> Optional[str]:
    if prompt_image_dir is None:
        return None

    image = payload.get("image")
    if not isinstance(image, np.ndarray):
        return None

    prompt_image_dir.mkdir(parents=True, exist_ok=True)
    file_path = prompt_image_dir / (
        f"episode_{episode_id}_goal_{goal_index:02d}_{instance_key}_{modality}.png"
    )
    Image.fromarray(np.asarray(image, dtype=np.uint8)).save(file_path)
    return str(file_path)


def _summarize_goal_input_payload(goal_index: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    modality = str(payload.get("modality", "object"))
    summary: Dict[str, Any] = {"goal_index": int(goal_index), "modality": modality}
    if modality == "image":
        image = payload.get("image")
        summary["image_shape"] = list(image.shape) if isinstance(image, np.ndarray) else None
        summary["image_path"] = None
        return summary
    if modality == "description":
        summary["text"] = str(payload.get("text", ""))
        return summary
    return summary


def _format_goal_input_summary(summary: Dict[str, Any]) -> str:
    modality = str(summary.get("modality", "object"))
    if modality == "image":
        return "modality=image | image_shape={shape} | image_path={path}".format(
            shape=summary.get("image_shape", None),
            path=summary.get("image_path", None),
        )
    if modality == "description":
        return "modality=description | text={text}".format(text=summary.get("text", ""))
    return "modality=object | category={category}".format(
        category=summary.get("category", "")
    )


def _normalize_task_specs(raw_tasks: Any) -> List[Tuple[str, str]]:
    specs: List[Tuple[str, str]] = []
    if not isinstance(raw_tasks, list):
        return specs
    for item in raw_tasks:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            specs.append((str(item[0]), str(item[1])))
            continue
        if isinstance(item, dict):
            instance_key = item.get("instance_key", item.get("goal", ""))
            modality = item.get("modality", "object")
            specs.append((str(instance_key), str(modality)))
    return specs


def _episode_goal_count(episode: Any, goal_specs: List[Tuple[str, str]]) -> int:
    raw_count = getattr(episode, "num_goals", None)
    goal_count = _optional_float(raw_count)
    if goal_count is None:
        goal_count = float(len(goal_specs))
    if goal_count <= 0:
        goal_count = float(len(goal_specs))
    return max(1, int(goal_count))


def _format_episode_metrics(metrics: Dict[str, Any], num_goals: int) -> Dict[str, float]:
    formatted: Dict[str, float] = {}
    for key, value in metrics.items():
        token = str(key)
        if token == "lifelong_goal_completion":
            numeric_value = float(value)
            if np.isfinite(numeric_value):
                formatted["success"] = numeric_value
            continue
        if token == "lifelong_goals_found":
            numeric_value = float(value)
            if np.isfinite(numeric_value):
                formatted["found_goals"] = numeric_value
            continue
        if token == "success":
            continue
        numeric_value = _optional_float(value)
        if numeric_value is None:
            continue
        formatted[token] = numeric_value
    formatted["num_goals"] = float(max(1, int(num_goals)))
    return formatted


def _safe_json_load(raw: str, fallback: Any) -> Any:
    token = str(raw).strip()
    if not token:
        return fallback
    return json.loads(token)


def _safe_yaml_or_json_load(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Eval config not found: {path}")

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))

    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise RuntimeError("Eval config root must be a mapping/object.")
    return payload


def _cfg_get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _first_cfg_value(cfg: Dict[str, Any], *paths: str) -> Any:
    for path in paths:
        value = _cfg_get(cfg, path, None)
        if value is not None:
            return value
    return None


def _set_arg_if_default(args: argparse.Namespace, name: str, default: Any, value: Any) -> None:
    if value is None:
        return
    if getattr(args, name) == default:
        setattr(args, name, value)


def apply_eval_config(args: argparse.Namespace) -> argparse.Namespace:
    if not hasattr(args, "dataset_path"):
        args.dataset_path = None
    if args.eval_config is None:
        return args

    eval_config_path = Path(args.eval_config).expanduser().resolve()
    cfg = _safe_yaml_or_json_load(eval_config_path)

    _set_arg_if_default(
        args,
        "exp_config",
        DEFAULT_CONFIG,
        _first_cfg_value(cfg, "habitat_config_path", "exp_config", "habitat.exp_config"),
    )
    _set_arg_if_default(
        args,
        "dataset_path",
        None,
        _first_cfg_value(cfg, "dataset_path", "test_data_dir", "dataset.path"),
    )
    _set_arg_if_default(
        args,
        "split",
        "val",
        _first_cfg_value(cfg, "split", "dataset.split"),
    )
    _set_arg_if_default(
        args,
        "scenes_dir",
        None,
        _first_cfg_value(cfg, "scene_data_path", "scenes_dir", "dataset.scenes_dir"),
    )
    _set_arg_if_default(
        args,
        "scene_dataset_config",
        None,
        _first_cfg_value(cfg, "scene_dataset_config_path", "scene_dataset_config", "simulator.scene_dataset_config"),
    )
    _set_arg_if_default(
        args,
        "task_type",
        DEFAULT_TASK_TYPE,
        _first_cfg_value(cfg, "task_type", "task.type"),
    )
    _set_arg_if_default(
        args,
        "goal_order_mode",
        None,
        _first_cfg_value(cfg, "goal_order_mode", "task.goal_order_mode"),
    )
    _set_arg_if_default(
        args,
        "scene",
        None,
        _first_cfg_value(cfg, "scene", "dataset.scene_filter"),
    )
    _set_arg_if_default(
        args,
        "scene_start_index",
        None,
        _first_cfg_value(
            cfg,
            "scene_start_index",
            "eval.scene_start_index",
            "dataset.scene_start_index",
        ),
    )
    _set_arg_if_default(
        args,
        "scene_end_index",
        None,
        _first_cfg_value(
            cfg,
            "scene_end_index",
            "eval.scene_end_index",
            "dataset.scene_end_index",
        ),
    )
    _set_arg_if_default(
        args,
        "num_episodes",
        None,
        _first_cfg_value(cfg, "num_episodes", "eval.num_episodes"),
    )
    _set_arg_if_default(
        args,
        "policy",
        "distance_submit",
        _first_cfg_value(cfg, "policy_name", "policy.name", "policy"),
    )

    if args.policy_kwargs == "{}":
        policy_kwargs = _first_cfg_value(cfg, "policy_kwargs", "policy.kwargs")
        if isinstance(policy_kwargs, dict):
            args.policy_kwargs = json.dumps(policy_kwargs)
        elif isinstance(policy_kwargs, str) and policy_kwargs.strip():
            args.policy_kwargs = policy_kwargs

    _set_arg_if_default(
        args,
        "submit_action_name",
        "LIFELONG_SUBMIT",
        _first_cfg_value(cfg, "submit_action_name", "policy.submit_action_name"),
    )
    _set_arg_if_default(
        args,
        "distance_submit_threshold",
        1.0,
        _first_cfg_value(cfg, "distance_submit_threshold", "policy.distance_submit_threshold"),
    )
    _set_arg_if_default(
        args,
        "seed",
        0,
        _first_cfg_value(cfg, "seed", "eval.seed"),
    )
    _set_arg_if_default(
        args,
        "print_every",
        1,
        _first_cfg_value(cfg, "print_every", "eval.print_every"),
    )
    _set_arg_if_default(
        args,
        "video",
        True,
        _first_cfg_value(cfg, "save_visualization", "video", "eval.video"),
    )
    _set_arg_if_default(
        args,
        "video_dir",
        None,
        _first_cfg_value(cfg, "video_dir", "eval.video_dir"),
    )
    _set_arg_if_default(
        args,
        "video_fps",
        None,
        _first_cfg_value(cfg, "video_fps", "eval.video_fps"),
    )
    _set_arg_if_default(
        args,
        "video_audio",
        True,
        _first_cfg_value(cfg, "video_audio", "eval.video_audio"),
    )
    _set_arg_if_default(
        args,
        "audio_active_threshold",
        1e-6,
        _first_cfg_value(cfg, "audio_active_threshold", "eval.audio_active_threshold"),
    )
    _set_arg_if_default(
        args,
        "video_audio_normalize",
        True,
        _first_cfg_value(cfg, "video_audio_normalize", "eval.video_audio_normalize"),
    )
    _set_arg_if_default(
        args,
        "video_audio_max_gain",
        200.0,
        _first_cfg_value(cfg, "video_audio_max_gain", "eval.video_audio_max_gain"),
    )
    _set_arg_if_default(
        args,
        "save_action_observations",
        True,
        _first_cfg_value(
            cfg,
            "save_action_observations",
            "eval.save_action_observations",
        ),
    )

    if args.action_observation_dir is None:
        action_observation_dir = _first_cfg_value(
            cfg,
            "action_observation_dir",
            "eval.action_observation_dir",
        )
        if isinstance(action_observation_dir, str) and action_observation_dir.strip():
            args.action_observation_dir = action_observation_dir

    if args.disable_content_scenes is None:
        disable_content = _first_cfg_value(cfg, "disable_content_scenes", "dataset.disable_content_scenes")
        if isinstance(disable_content, bool):
            args.disable_content_scenes = disable_content

    _set_arg_if_default(
        args,
        "exp_name",
        DEFAULT_EXP_NAME,
        _first_cfg_value(cfg, "exp_name", "eval.exp_name"),
    )
    _set_arg_if_default(
        args,
        "output_parent_dir",
        DEFAULT_OUTPUT_PARENT_DIR,
        _first_cfg_value(cfg, "output_parent_dir", "eval.output_parent_dir"),
    )

    if args.episode_log is None:
        episode_log = _first_cfg_value(cfg, "episode_log", "eval.episode_log")
        if isinstance(episode_log, str) and episode_log.strip():
            args.episode_log = episode_log

    if args.mean_log is None:
        mean_log = _first_cfg_value(cfg, "mean_log", "eval.mean_log")
        if isinstance(mean_log, str) and mean_log.strip():
            args.mean_log = mean_log

    if args.prompt_image_dir is None:
        prompt_image_dir = _first_cfg_value(cfg, "prompt_image_dir", "eval.prompt_image_dir")
        if isinstance(prompt_image_dir, str) and prompt_image_dir.strip():
            args.prompt_image_dir = prompt_image_dir

    print(f"Loaded eval config: {eval_config_path}")
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Full Modal Lifelong Semantic Audio task with pluggable policies.",
    )
    parser.add_argument(
        "--eval-config",
        type=str,
        default=None,
        help="Optional eval config file (.yaml/.yml/.json), goat-bench style supported.",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Task config path.",
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default=DEFAULT_TASK_TYPE,
        help="Habitat TASK.TYPE to evaluate.",
    )
    parser.add_argument(
        "--goal-order-mode",
        type=str,
        default=None,
        choices=["ordered", "unordered"],
        help="Override TASK.GOAL_ORDER_MODE.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        help="Override DATASET.SPLIT.",
    )
    parser.add_argument(
        "--scenes-dir",
        type=str,
        default=None,
        help="Override DATASET.SCENES_DIR.",
    )
    parser.add_argument(
        "--scene-dataset-config",
        type=str,
        default=None,
        help="Override SIMULATOR.SCENE_DATASET.",
    )
    parser.add_argument(
        "--disable-content-scenes",
        default=None,
        action=argparse.BooleanOptionalAction,
        help="Disable per-scene content loading.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Optional scene filter by scene basename or parent dir.",
    )
    parser.add_argument(
        "--scene-start-index",
        type=int,
        default=None,
        help="Inclusive scene index in sorted scene list.",
    )
    parser.add_argument(
        "--scene-end-index",
        type=int,
        default=None,
        help="Exclusive scene index in sorted scene list.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Number of episodes to evaluate (default: all).",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="distance_submit",
        help=(
            "Policy name from registry or '<module>:<Class>'. "
            f"Built-ins: {', '.join(list_lifelong_eval_policies())}"
        ),
    )
    parser.add_argument(
        "--policy-kwargs",
        type=str,
        default="{}",
        help="JSON dict passed to policy constructor.",
    )
    parser.add_argument(
        "--submit-action-name",
        type=str,
        default="LIFELONG_SUBMIT",
        help="Action name used for submit.",
    )
    parser.add_argument(
        "--distance-submit-threshold",
        type=float,
        default=1.0,
        help="Used by built-in distance_submit policy.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Global random seed.",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Print every N episodes.",
    )
    parser.add_argument(
        "--episode-log",
        type=str,
        default=None,
        help="Optional jsonl file for per-episode output.",
    )
    parser.add_argument(
        "--mean-log",
        type=str,
        default=None,
        help="Optional json file for final mean metrics.",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default=DEFAULT_EXP_NAME,
        help="Experiment name used in default log directory naming.",
    )
    parser.add_argument(
        "--output-parent-dir",
        type=str,
        default=DEFAULT_OUTPUT_PARENT_DIR,
        help="Parent directory for default log outputs.",
    )
    parser.add_argument(
        "--video",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Save evaluation video per episode.",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default=None,
        help="Directory for saved videos (default: <run_dir>/videos).",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=None,
        help="FPS for output videos (default: 1 / STEP_TIME).",
    )
    parser.add_argument(
        "--video-audio",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Attach audiogoal waveform into saved video.",
    )
    parser.add_argument(
        "--audio-active-threshold",
        type=float,
        default=1e-6,
        help="Amplitude threshold used for audio activity diagnostics.",
    )
    parser.add_argument(
        "--video-audio-normalize",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Normalize per-episode audiogoal before muxing into video.",
    )
    parser.add_argument(
        "--video-audio-max-gain",
        type=float,
        default=200.0,
        help="Max gain when normalizing audio for video export.",
    )
    parser.add_argument(
        "--prompt-image-dir",
        type=str,
        default=None,
        help="Optional directory for saved image-goal prompt references.",
    )
    parser.add_argument(
        "--save-action-observations",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Save RGB and semantic snapshots when policy executes submit/stop.",
    )
    parser.add_argument(
        "--action-observation-dir",
        type=str,
        default=None,
        help="Directory for submit/stop RGB+semantic snapshots (default: <run_dir>/action_observations).",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> habitat.Config:
    cfg = get_task_config(config_paths=[args.exp_config])
    cfg.defrost()

    cfg.DATASET.DATA_PATH = args.dataset_path
    cfg.DATASET.SPLIT = args.split
    if args.scenes_dir:
        cfg.DATASET.SCENES_DIR = args.scenes_dir
    if args.disable_content_scenes:
        cfg.DATASET.CONTENT_SCENES = []

    if args.scene_dataset_config:
        cfg.SIMULATOR.SCENE_DATASET = args.scene_dataset_config

    cfg.TASK.TYPE = args.task_type
    if args.goal_order_mode is not None:
        cfg.TASK.GOAL_ORDER_MODE = args.goal_order_mode
    # Ensure RGB+Depth are both enabled so video can include depth panel.
    default_agent_id = getattr(cfg.SIMULATOR, "DEFAULT_AGENT_ID", 0)
    agent_names = list(getattr(cfg.SIMULATOR, "AGENTS", []))
    if 0 <= int(default_agent_id) < len(agent_names):
        agent_cfg = getattr(cfg.SIMULATOR, agent_names[int(default_agent_id)])
    else:
        agent_cfg = getattr(cfg.SIMULATOR, "AGENT_0")

    agent_sensors = [str(sensor) for sensor in list(getattr(agent_cfg, "SENSORS", []))]
    if "RGB_SENSOR" not in agent_sensors:
        agent_sensors.append("RGB_SENSOR")
    if "DEPTH_SENSOR" not in agent_sensors:
        agent_sensors.append("DEPTH_SENSOR")
    if "SEMANTIC_SENSOR" not in agent_sensors:
        agent_sensors.append("SEMANTIC_SENSOR")
    agent_cfg.SENSORS = agent_sensors

    if hasattr(cfg.SIMULATOR, "RGB_SENSOR") and hasattr(cfg.SIMULATOR.RGB_SENSOR, "UUID"):
        cfg.SIMULATOR.RGB_SENSOR.UUID = "rgb"
    if hasattr(cfg.SIMULATOR, "DEPTH_SENSOR") and hasattr(cfg.SIMULATOR.DEPTH_SENSOR, "UUID"):
        cfg.SIMULATOR.DEPTH_SENSOR.UUID = "depth"
    if hasattr(cfg.SIMULATOR, "SEMANTIC_SENSOR") and hasattr(cfg.SIMULATOR.SEMANTIC_SENSOR, "UUID"):
        cfg.SIMULATOR.SEMANTIC_SENSOR.UUID = "semantic"

    if args.video_audio and "AUDIOGOAL_SENSOR" not in cfg.TASK.SENSORS:
        cfg.TASK.SENSORS.append("AUDIOGOAL_SENSOR")
    if hasattr(cfg.TASK, "AUDIOGOAL_SENSOR") and hasattr(cfg.TASK.AUDIOGOAL_SENSOR, "UUID"):
        cfg.TASK.AUDIOGOAL_SENSOR.UUID = "audiogoal"

    sensor_type_overrides = {
        "AUDIOGOAL_SENSOR": "AudioGoalSensor",
        "SPECTROGRAM_SENSOR": "SpectrogramSensor",
        "CATEGORY": "Category",
        "CATEGORY_BELIEF": "CategoryBelief",
        "LOCATION_BELIEF": "LocationBelief",
        "POSE_SENSOR": "FullPoseSensor",
        "FULL_POSE_SENSOR": "FullPoseSensor",
        "POINTGOAL_WITH_GPS_COMPASS_SENSOR": "PointGoalWithGPSCompassSensor",
    }

    normalized_sensors: List[str] = []
    for sensor_name in list(cfg.TASK.SENSORS):
        token = "FULL_POSE_SENSOR" if str(sensor_name) == "POSE_SENSOR" else str(sensor_name)
        if token not in normalized_sensors:
            normalized_sensors.append(token)
        if token == "FULL_POSE_SENSOR" and hasattr(cfg.TASK, "POSE_SENSOR") and not hasattr(cfg.TASK, "FULL_POSE_SENSOR"):
            cfg.TASK.FULL_POSE_SENSOR = cfg.TASK.POSE_SENSOR
        if not hasattr(cfg.TASK, token):
            cfg.TASK[token] = Config()
        if hasattr(cfg.TASK[token], "TYPE") and str(getattr(cfg.TASK[token], "TYPE")):
            continue
        inferred_type = sensor_type_overrides.get(token)
        if inferred_type is not None:
            cfg.TASK[token].TYPE = inferred_type
    cfg.TASK.SENSORS = normalized_sensors

    if args.submit_action_name not in cfg.TASK.POSSIBLE_ACTIONS:
        cfg.TASK.POSSIBLE_ACTIONS.append(args.submit_action_name)

    if not hasattr(cfg.TASK, "ACTIONS"):
        cfg.TASK.ACTIONS = Config()
    if not hasattr(cfg.TASK.ACTIONS, args.submit_action_name):
        cfg.TASK.ACTIONS[args.submit_action_name] = Config()
    cfg.TASK.ACTIONS[args.submit_action_name].TYPE = "LifelongSubmitAction"

    cfg.TASK.MEASUREMENTS = [
        "SUCCESS",
        "SPL",
        "SOFT_SPL",
        "NUM_ACTION",
        "LIFELONG_GOALS_FOUND",
        "LIFELONG_GOAL_COMPLETION",
    ]
    if args.video:
        cfg.TASK.MEASUREMENTS.extend(["TOP_DOWN_MAP", "COLLISIONS"])

    if not hasattr(cfg.TASK, "LIFELONG_GOALS_FOUND"):
        cfg.TASK.LIFELONG_GOALS_FOUND = Config()
    cfg.TASK.LIFELONG_GOALS_FOUND.TYPE = "LifelongGoalsFound"

    if not hasattr(cfg.TASK, "LIFELONG_GOAL_COMPLETION"):
        cfg.TASK.LIFELONG_GOAL_COMPLETION = Config()
    cfg.TASK.LIFELONG_GOAL_COMPLETION.TYPE = "LifelongGoalCompletion"

    if not hasattr(cfg.TASK, "SUCCESS"):
        cfg.TASK.SUCCESS = Config()
    cfg.TASK.SUCCESS.TYPE = "OmniLongSuccess"

    if not hasattr(cfg.TASK, "SPL"):
        cfg.TASK.SPL = Config()
    cfg.TASK.SPL.TYPE = "OmniLongSPL"

    if not hasattr(cfg.TASK, "SOFT_SPL"):
        cfg.TASK.SOFT_SPL = Config()
    cfg.TASK.SOFT_SPL.TYPE = "OmniLongSoftSPL"

    cfg.freeze()
    return cfg


def _scene_key(scene_id: str) -> str:
    scene_base = os.path.basename(scene_id).split(".")[0]
    scene_parent = os.path.basename(os.path.dirname(scene_id))
    return scene_parent or scene_base


def _all_scene_names(episodes: List[Any]) -> List[str]:
    scene_names = {
        _scene_key(str(getattr(ep, "scene_id", "")))
        for ep in episodes
        if str(getattr(ep, "scene_id", "")).strip()
    }
    return sorted(scene_names)


def _resolve_scene_subset(
    all_scene_names: List[str],
    args: argparse.Namespace,
) -> Tuple[int, int, List[str]]:
    if args.scene and (
        args.scene_start_index is not None or args.scene_end_index is not None
    ):
        raise RuntimeError(
            "Use either --scene or --scene-start-index/--scene-end-index, not both."
        )

    total_scenes = len(all_scene_names)
    if args.scene:
        scene_name = str(args.scene)
        if scene_name not in all_scene_names:
            return 0, 0, []
        scene_index = all_scene_names.index(scene_name)
        return scene_index, scene_index + 1, [scene_name]

    start_index = 0 if args.scene_start_index is None else int(args.scene_start_index)
    end_index = (
        total_scenes if args.scene_end_index is None else int(args.scene_end_index)
    )

    if start_index < 0 or start_index > total_scenes:
        raise RuntimeError(
            f"scene_start_index={start_index} out of range for {total_scenes} scenes"
        )
    if end_index < 0 or end_index > total_scenes:
        raise RuntimeError(
            f"scene_end_index={end_index} out of range for {total_scenes} scenes"
        )
    if end_index < start_index:
        raise RuntimeError(
            f"scene_end_index={end_index} must be >= scene_start_index={start_index}"
        )

    return start_index, end_index, all_scene_names[start_index:end_index]


def _scene_subset_label(
    all_scene_names: List[str],
    args: argparse.Namespace,
) -> str:
    start_index, end_index, _ = _resolve_scene_subset(all_scene_names, args)
    return f"[{start_index}:{end_index}]"


def _prepare_episode_list(env: habitat.Env, args: argparse.Namespace) -> List[Any]:
    episodes = list(env.episodes)
    episodes.sort(
        key=lambda ep: (
            _scene_key(str(getattr(ep, "scene_id", ""))),
            int(getattr(ep, "episode_id", 0)),
        )
    )

    all_scene_names = _all_scene_names(episodes)
    _, _, selected_scene_names = _resolve_scene_subset(all_scene_names, args)
    if args.scene:
        scene_name = str(args.scene)
        episodes = [
            ep
            for ep in episodes
            if _scene_key(ep.scene_id) == scene_name
            or os.path.basename(ep.scene_id).split(".")[0] == scene_name
        ]
    else:
        selected_scene_name_set = set(selected_scene_names)
        episodes = [
            ep
            for ep in episodes
            if _scene_key(str(getattr(ep, "scene_id", ""))) in selected_scene_name_set
        ]

    if args.num_episodes is not None and args.num_episodes >= 0:
        episodes = episodes[: args.num_episodes]
    return episodes


def _build_policy(args: argparse.Namespace):
    raw_kwargs = _safe_json_load(args.policy_kwargs, fallback={})
    if not isinstance(raw_kwargs, dict):
        raw_kwargs = {}

    policy_kwargs: Dict[str, Any] = dict(raw_kwargs)
    policy_kwargs.setdefault("seed", int(args.seed))
    policy_kwargs.setdefault("submit_action_name", str(args.submit_action_name))
    if str(args.policy).strip().lower() in {"distance_submit", "oracle", "oracle_shortest_submit"}:
        policy_kwargs.setdefault("submit_distance", float(args.distance_submit_threshold))
    if str(args.policy).strip().lower() in {"oracle", "oracle_shortest_submit"}:
        policy_kwargs.setdefault("goal_order_mode", args.goal_order_mode)

    return build_lifelong_eval_policy(args.policy, **policy_kwargs)
