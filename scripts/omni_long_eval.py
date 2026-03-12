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
from habitat.utils.visualizations.utils import images_to_video

from ss_baselines.av_nav.config.default import get_task_config
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
    if not isinstance(image_payload, dict):
        return None
    render_views = image_payload.get("render_views")
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
    if not _is_vec3(position) or not _is_quat4(rotation):
        return None

    try:
        observations = env.sim.get_observations_at(
            position=[float(v) for v in position],
            rotation=[float(v) for v in rotation],
            keep_agent_at_new_pose=False,
        )
    except Exception:
        return None

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


def _build_goal_prompt(
    instance_key: str,
    modality: str,
    instance_record: Optional[Dict[str, Any]],
    sound_id: Optional[str],
    image_path: Optional[str],
) -> str:
    category = None
    semantic_id = None
    description = None
    if isinstance(instance_record, dict):
        category = instance_record.get("category")
        semantic_id = instance_record.get("semantic_id")
        description = _extract_text_description(instance_record)

    prompt_parts: List[str] = [
        f"instance_key={instance_key}",
        f"modality={modality}",
    ]
    if category is not None:
        prompt_parts.append(f"category={category}")
    if semantic_id is not None:
        prompt_parts.append(f"semantic_id={semantic_id}")
    if sound_id:
        prompt_parts.append(f"audio={sound_id}")

    m = str(modality).strip().lower()
    if m.startswith("image"):
        if image_path:
            prompt_parts.append(f"image_ref={image_path}")
        else:
            prompt_parts.append("image_ref=<missing>")
    elif m in {"description", "text", "text_description"}:
        if description:
            prompt_parts.append(f"text={description}")
        else:
            prompt_parts.append("text=<missing>")

    return " | ".join(prompt_parts)


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


def _save_prompt_image_if_needed(
    env: habitat.Env,
    instance_record: Optional[Dict[str, Any]],
    modality: str,
    prompt_image_dir: Optional[Path],
    episode_id: str,
    goal_index: int,
    instance_key: str,
) -> Optional[str]:
    if prompt_image_dir is None:
        return None
    if not isinstance(instance_record, dict):
        return None

    image_index = _parse_image_modality_index(modality)
    if image_index is None:
        return None

    rgb = _render_reference_image(env, instance_record, image_index)
    if rgb is None:
        return None

    prompt_image_dir.mkdir(parents=True, exist_ok=True)
    file_path = prompt_image_dir / (
        f"episode_{episode_id}_goal_{goal_index:02d}_{instance_key}_{modality}.png"
    )
    Image.fromarray(rgb).save(file_path)
    return str(file_path)


def _filter_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            out[str(key)] = float(value)
    return out


def _safe_json_load(raw: str, fallback: Any) -> Any:
    token = str(raw).strip()
    if not token:
        return fallback
    try:
        return json.loads(token)
    except Exception:
        return fallback


def _safe_yaml_or_json_load(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Eval config not found: {path}")

    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError(
                "Reading YAML eval config requires PyYAML. Install with `pip install pyyaml`."
            ) from exc
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
        "num_episodes",
        None,
        _first_cfg_value(cfg, "num_episodes", "eval.num_episodes"),
    )
    _set_arg_if_default(
        args,
        "max_steps",
        500,
        _first_cfg_value(cfg, "max_steps", "eval.max_steps"),
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

    if args.disable_content_scenes is False:
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
        "--dataset-path",
        type=str,
        default=None,
        help="Path to dataset .json.gz.",
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
        action="store_true",
        help="Disable per-scene content loading.",
    )
    parser.add_argument(
        "--scene",
        type=str,
        default=None,
        help="Optional scene filter by scene basename or parent dir.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Number of episodes to evaluate (default: all).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Step cap per episode.",
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
    cfg.ENVIRONMENT.MAX_EPISODE_STEPS = int(args.max_steps)

    # Ensure RGB+Depth are both enabled so video can include depth panel.
    try:
        default_agent_id = cfg.SIMULATOR.DEFAULT_AGENT_ID
        agent_name = cfg.SIMULATOR.AGENTS[default_agent_id]
        agent_cfg = getattr(cfg.SIMULATOR, agent_name)
    except Exception:
        agent_cfg = getattr(cfg.SIMULATOR, "AGENT_0")

    agent_sensors = [str(sensor) for sensor in list(getattr(agent_cfg, "SENSORS", []))]
    if "RGB_SENSOR" not in agent_sensors:
        agent_sensors.append("RGB_SENSOR")
    if "DEPTH_SENSOR" not in agent_sensors:
        agent_sensors.append("DEPTH_SENSOR")
    agent_cfg.SENSORS = agent_sensors

    if hasattr(cfg.SIMULATOR, "RGB_SENSOR") and hasattr(cfg.SIMULATOR.RGB_SENSOR, "UUID"):
        cfg.SIMULATOR.RGB_SENSOR.UUID = "rgb"
    if hasattr(cfg.SIMULATOR, "DEPTH_SENSOR") and hasattr(cfg.SIMULATOR.DEPTH_SENSOR, "UUID"):
        cfg.SIMULATOR.DEPTH_SENSOR.UUID = "depth"

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
        "POSE_SENSOR": "PoseSensor",
        "POINTGOAL_WITH_GPS_COMPASS_SENSOR": "PointGoalWithGPSCompassSensor",
    }

    normalized_sensors: List[str] = []
    for sensor_name in list(cfg.TASK.SENSORS):
        token = str(sensor_name)
        if token not in normalized_sensors:
            normalized_sensors.append(token)
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
        "DISTANCE_TO_GOAL",
        "SUCCESS",
        "SPL",
        "SOFT_SPL",
        "NUM_ACTION",
        "LIFELONG_GOALS_FOUND",
        "LIFELONG_GOAL_COMPLETION",
        "LIFELONG_TASK_SUCCESS",
    ]
    if args.video:
        cfg.TASK.MEASUREMENTS.extend(["TOP_DOWN_MAP", "COLLISIONS"])

    if not hasattr(cfg.TASK, "LIFELONG_GOALS_FOUND"):
        cfg.TASK.LIFELONG_GOALS_FOUND = Config()
    cfg.TASK.LIFELONG_GOALS_FOUND.TYPE = "LifelongGoalsFound"

    if not hasattr(cfg.TASK, "LIFELONG_GOAL_COMPLETION"):
        cfg.TASK.LIFELONG_GOAL_COMPLETION = Config()
    cfg.TASK.LIFELONG_GOAL_COMPLETION.TYPE = "LifelongGoalCompletion"

    if not hasattr(cfg.TASK, "LIFELONG_TASK_SUCCESS"):
        cfg.TASK.LIFELONG_TASK_SUCCESS = Config()
    cfg.TASK.LIFELONG_TASK_SUCCESS.TYPE = "LifelongTaskSuccess"

    cfg.freeze()
    return cfg


def _scene_key(scene_id: str) -> str:
    scene_base = os.path.basename(scene_id).split(".")[0]
    scene_parent = os.path.basename(os.path.dirname(scene_id))
    return scene_parent or scene_base


def _prepare_episode_list(env: habitat.Env, args: argparse.Namespace) -> List[Any]:
    episodes = list(env.episodes)
    if args.scene:
        episodes = [
            ep
            for ep in episodes
            if _scene_key(ep.scene_id) == args.scene
            or os.path.basename(ep.scene_id).split(".")[0] == args.scene
        ]
    episodes.sort(key=lambda ep: int(getattr(ep, "episode_id", 0)))
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
    if str(args.policy).strip().lower() == "distance_submit":
        policy_kwargs.setdefault("submit_distance", float(args.distance_submit_threshold))

    return build_lifelong_eval_policy(args.policy, **policy_kwargs)


def main() -> None:
    args = parse_args()
    args = apply_eval_config(args)

    if args.dataset_path is None or not str(args.dataset_path).strip():
        raise RuntimeError(
            "dataset_path is required. Pass --dataset-path or provide it in --eval-config."
        )

    cfg = build_config(args)

    print("dataset_path_in_use={}".format(cfg.DATASET.DATA_PATH))
    print("task_type_in_use={}".format(cfg.TASK.TYPE))
    if hasattr(cfg.TASK, "GOAL_ORDER_MODE"):
        print("goal_order_mode_in_use={}".format(cfg.TASK.GOAL_ORDER_MODE))

    dataset_payload = _load_dataset_payload(str(cfg.DATASET.DATA_PATH))
    instance_index = _flatten_instances(dataset_payload.get("instances"))
    print("instance_count_in_dataset={}".format(len(instance_index)))

    env = habitat.Env(config=cfg)
    policy = _build_policy(args)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.episode_log is None or args.mean_log is None:
        run_base_dir = os.path.join(
            str(args.output_parent_dir),
            str(args.exp_name),
            timestamp,
        )
        os.makedirs(run_base_dir, exist_ok=True)
        if args.episode_log is None:
            args.episode_log = os.path.join(run_base_dir, "episode.jsonl")
        if args.mean_log is None:
            args.mean_log = os.path.join(run_base_dir, "mean.json")

    run_base_dir = str(Path(args.episode_log).resolve().parent)

    if args.video and args.video_dir is None:
        default_video_dir = os.path.join(run_base_dir, "videos")
        args.video_dir = default_video_dir
    if args.prompt_image_dir is None:
        args.prompt_image_dir = os.path.join(run_base_dir, "prompt_images")

    os.makedirs(os.path.dirname(args.episode_log) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.mean_log) or ".", exist_ok=True)
    if args.video:
        os.makedirs(str(args.video_dir), exist_ok=True)
    os.makedirs(str(args.prompt_image_dir), exist_ok=True)
    prompt_image_dir = Path(args.prompt_image_dir)

    episode_writer = open(args.episode_log, "w", encoding="utf-8")

    episodes = _prepare_episode_list(env, args)
    if not episodes:
        episode_writer.close()
        env.close()
        print("No episodes selected.")
        return

    episodes_by_id = {int(getattr(ep, "episode_id", 0)): ep for ep in episodes}
    ordered_episode_ids = sorted(episodes_by_id.keys())

    sum_metrics: Dict[str, float] = {}
    episodes_done = 0
    episodes_truncated = 0

    try:
        for episode_index, episode_id in enumerate(ordered_episode_ids):
            target_episode = episodes_by_id[episode_id]
            env.current_episode = target_episode
            observations = env.reset()
            episode = env.current_episode
            policy.reset(env=env, episode=episode, observations=observations)

            episode_id_text = str(getattr(episode, "episode_id", ""))
            goal_specs = _normalize_task_specs(getattr(episode, "tasks", None))
            goal_summary = [f"{instance_key}:{modality}" for instance_key, modality in goal_specs]

            prompt_entries: List[Dict[str, Any]] = []
            for goal_idx, (instance_key, modality) in enumerate(goal_specs):
                instance_record = instance_index.get(instance_key)
                sound_id = _sound_id_for_goal(episode, goal_idx)
                image_path = _save_prompt_image_if_needed(
                    env=env,
                    instance_record=instance_record,
                    modality=modality,
                    prompt_image_dir=prompt_image_dir,
                    episode_id=episode_id_text,
                    goal_index=goal_idx,
                    instance_key=instance_key,
                )
                prompt_text = _build_goal_prompt(
                    instance_key=instance_key,
                    modality=modality,
                    instance_record=instance_record,
                    sound_id=sound_id,
                    image_path=image_path,
                )
                prompt_entries.append(
                    {
                        "goal_index": int(goal_idx),
                        "instance_key": instance_key,
                        "modality": modality,
                        "prompt": prompt_text,
                        "image_path": image_path,
                    }
                )

            print(
                "[episode {idx}] id={eid} goals={goals}".format(
                    idx=episode_index,
                    eid=episode_id_text,
                    goals=goal_summary,
                )
            )
            for prompt_entry in prompt_entries:
                image_tag = prompt_entry.get("image_path") or "<none>"
                print(
                    "  [goal {gi}] prompt={prompt} image_saved={img}".format(
                        gi=prompt_entry["goal_index"],
                        prompt=prompt_entry["prompt"],
                        img=image_tag,
                    )
                )

            task = getattr(env, "_task", None)
            last_task_token = getattr(task, "current_task_token", None)
            print(
                "  [audio_state] active_sound_idx={idx} active_sound={sound}".format(
                    idx=getattr(env.sim, "_active_sound_idx", None),
                    sound=getattr(env.sim, "_current_sound", None),
                )
            )
            schedule = getattr(episode, "sound_source_schedule", None)
            if isinstance(schedule, list) and len(schedule) >= 2 and str(schedule[0]).lower() == "round_robin":
                print(
                    "  [audio_note] round_robin schedule rotates source every {} simulator steps; "
                    "active sound may differ from current goal token.".format(schedule[1])
                )

            done = False
            step_idx = 0
            frames = []
            audios = []
            audio_peak_max = 0.0
            audio_rms_acc = 0.0
            audio_active_steps = 0

            while not done and step_idx < args.max_steps:
                context = build_lifelong_eval_context(
                    env=env,
                    episode=episode,
                    episode_index=episode_index,
                    step_index=step_idx,
                )
                action = policy.act(
                    env=env,
                    episode=episode,
                    observations=observations,
                    context=context,
                )

                observations = env.step(action)
                done = bool(env.episode_over)

                if args.video:
                    frame = observations_to_image(observations, env.get_metrics())
                    frames.append(frame)
                if args.video_audio:
                    if "audiogoal" not in observations:
                        raise RuntimeError(
                            "video_audio enabled but observation missing 'audiogoal'. "
                            "Ensure AUDIOGOAL_SENSOR is in TASK.SENSORS."
                        )
                    audio_obs = np.asarray(observations["audiogoal"], dtype=np.float32)
                    audios.append(audio_obs)
                    if audio_obs.size > 0:
                        peak = float(np.max(np.abs(audio_obs)))
                        rms = float(np.sqrt(np.mean(np.square(audio_obs))))
                        audio_peak_max = max(audio_peak_max, peak)
                        audio_rms_acc += rms
                        if peak > float(args.audio_active_threshold):
                            audio_active_steps += 1

                policy.observe(
                    env=env,
                    episode=episode,
                    observations=observations,
                    reward=None,
                    done=done,
                    info=None,
                )

                if task is not None:
                    current_task_token = getattr(task, "current_task_token", None)
                    if current_task_token != last_task_token:
                        print(
                            "  [goal_switch] step={step} from={src} to={dst} active_sound_idx={idx} active_sound={sound}".format(
                                step=step_idx,
                                src=last_task_token,
                                dst=current_task_token,
                                idx=getattr(env.sim, "_active_sound_idx", None),
                                sound=getattr(env.sim, "_current_sound", None),
                            )
                        )
                        last_task_token = current_task_token
                step_idx += 1

            if not done:
                episodes_truncated += 1

            if args.video_audio:
                audio_frame_count = len(audios)
                audio_mean_rms = audio_rms_acc / float(audio_frame_count) if audio_frame_count > 0 else 0.0
                print(
                    "  [audio_diag] frames={frames} active_frames={active} peak_max={peak:.6f} mean_rms={rms:.6f} "
                    "offset={offset} duration={duration} schedule={schedule} end_active_sound={sound}".format(
                        frames=audio_frame_count,
                        active=audio_active_steps,
                        peak=audio_peak_max,
                        rms=audio_mean_rms,
                        offset=getattr(episode, "offset", None),
                        duration=getattr(episode, "duration", None),
                        schedule=getattr(episode, "sound_source_schedule", None),
                        sound=getattr(env.sim, "_current_sound", None),
                    )
                )
                if audio_active_steps == 0:
                    print(
                        "  [audio_diag] warning: no active audiogoal frame above threshold; "
                        "check duration/schedule semantics and source alignment."
                    )

            metrics = _filter_metrics(env.get_metrics())
            for key, value in metrics.items():
                sum_metrics[key] = sum_metrics.get(key, 0.0) + float(value)

            task = getattr(env, "_task", None)
            row = {
                "episode_index": episode_index,
                "episode_id": str(getattr(episode, "episode_id", "")),
                "scene_id": str(getattr(episode, "scene_id", "")),
                "steps": int(step_idx),
                "truncated": bool(not done),
                "metrics": metrics,
                "goal_specs": goal_summary,
                "goal_prompts": prompt_entries,
                "goal_order_mode": str(getattr(episode, "goal_order_mode", getattr(cfg.TASK, "GOAL_ORDER_MODE", "ordered"))),
                "completed_goal_indices": list(getattr(task, "completed_goal_indices", ())),
                "remaining_goal_indices": list(getattr(task, "remaining_goal_indices", ())),
                "current_task_token": getattr(task, "current_task_token", None),
                "active_sound_idx": getattr(env.sim, "_active_sound_idx", None),
                "active_sound": getattr(env.sim, "_current_sound", None),
            }
            episode_writer.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            episode_writer.flush()

            episodes_done += 1
            if episode_index % max(1, args.print_every) == 0:
                print(
                    "[episode {idx}] id={eid} steps={steps} truncated={tr} goals={goals} metrics={metrics}".format(
                        idx=episode_index,
                        eid=row["episode_id"],
                        steps=row["steps"],
                        tr=row["truncated"],
                        goals=goal_summary,
                        metrics=json.dumps(metrics, sort_keys=True),
                    )
                )

            if args.video and len(frames) > 0:
                fps = args.video_fps
                if fps is None:
                    step_time = float(getattr(cfg.SIMULATOR, "STEP_TIME", 0.25))
                    fps = int(round(1.0 / step_time)) if step_time > 1e-6 else 4
                    fps = max(1, fps)
                video_name = "episode_{}_id_{}".format(episode_index, row["episode_id"])
                if args.video_audio:
                    if len(audios) == 0:
                        raise RuntimeError(
                            "video_audio enabled but no audio frames were collected."
                        )
                    muxed_audios = audios
                    raw_peak = 0.0
                    for clip in audios:
                        if clip.size > 0:
                            raw_peak = max(raw_peak, float(np.max(np.abs(clip))))
                    gain = 1.0
                    if args.video_audio_normalize and raw_peak > 0.0:
                        gain = min(float(args.video_audio_max_gain), 0.8 / raw_peak)
                        if abs(gain - 1.0) > 1e-6:
                            muxed_audios = [
                                np.clip(np.asarray(clip, dtype=np.float32) * gain, -1.0, 1.0)
                                for clip in audios
                            ]
                    print(
                        "  [video_audio] raw_peak={peak:.6f} gain={gain:.3f} normalized={norm}".format(
                            peak=raw_peak,
                            gain=gain,
                            norm=bool(args.video_audio_normalize),
                        )
                    )
                    images_to_video_with_audio(
                        frames,
                        str(args.video_dir),
                        video_name,
                        muxed_audios,
                        cfg.SIMULATOR.AUDIO.RIR_SAMPLING_RATE,
                        fps=int(fps),
                    )
                else:
                    images_to_video(frames, str(args.video_dir), video_name, fps=int(fps))

    finally:
        episode_writer.close()
        try:
            policy.close()
        finally:
            env.close()

    if episodes_done == 0:
        print("No episodes evaluated.")
        return

    mean_metrics = {k: v / float(episodes_done) for k, v in sum_metrics.items()}
    summary = {
        "episodes": int(episodes_done),
        "truncated_episodes": int(episodes_truncated),
        "task_type": str(cfg.TASK.TYPE),
        "dataset_path": str(cfg.DATASET.DATA_PATH),
        "policy": str(args.policy),
        "mean_metrics": mean_metrics,
    }

    with open(args.mean_log, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)

    print("Completed {} episodes".format(episodes_done))
    print("Truncated episodes: {}".format(episodes_truncated))
    print("Mean metrics: {}".format(json.dumps(mean_metrics, sort_keys=True)))
    print("Episode log: {}".format(args.episode_log))
    print("Mean log: {}".format(args.mean_log))


if __name__ == "__main__":
    main()
