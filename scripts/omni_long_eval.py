#!/usr/bin/env python3

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import habitat
import numpy as np
import soundspaces  # noqa: F401 - register datasets/tasks/sims
from habitat.utils.visualizations.utils import images_to_video
from PIL import Image

from ss_baselines.common.omni_long_eval_policy import build_lifelong_eval_context
from ss_baselines.common.utils import observations_to_image, images_to_video_with_audio
from soundspaces.tasks.omni_long_eval_utils import (
    _all_scene_names,
    _build_goal_input_payload,
    _build_policy,
    _episode_goal_count,
    _format_goal_input_summary,
    _format_episode_metrics,
    _flatten_instances,
    _load_dataset_payload,
    _normalize_task_specs,
    _prepare_episode_list,
    _resolve_scene_subset,
    _save_goal_input_image_if_needed,
    _scene_key,
    _scene_subset_label,
    _summarize_goal_input_payload,
    apply_eval_config,
    build_config,
    parse_args,
)


WEIGHTED_METRIC_NAMES = ("success", "spl", "softspl")
RUNNING_MEAN_METRIC_NAMES = ("na",)


def _new_metric_accumulator() -> Dict[str, Any]:
    return {
        "sum_metrics": {},
        "weighted_metric_sums": {name: 0.0 for name in WEIGHTED_METRIC_NAMES},
        "total_goal_weight": 0.0,
        "episodes_done": 0,
        "episodes_truncated": 0,
    }


def _update_metric_accumulator(
    accumulator: Dict[str, Any],
    metrics: Dict[str, float],
    goal_weight: float,
    truncated: bool,
) -> None:
    sum_metrics = accumulator["sum_metrics"]
    for key, value in metrics.items():
        sum_metrics[key] = sum_metrics.get(key, 0.0) + float(value)

    accumulator["total_goal_weight"] += float(goal_weight)
    weighted_metric_sums = accumulator["weighted_metric_sums"]
    for key in WEIGHTED_METRIC_NAMES:
        weighted_metric_sums[key] += float(metrics.get(key, 0.0)) * float(goal_weight)

    accumulator["episodes_done"] += 1
    if truncated:
        accumulator["episodes_truncated"] += 1


def _mean_metrics_from_accumulator(accumulator: Dict[str, Any]) -> Dict[str, float]:
    episodes_done = int(accumulator["episodes_done"])
    if episodes_done <= 0:
        return {}
    return {
        key: value / float(episodes_done)
        for key, value in accumulator["sum_metrics"].items()
    }


def _weighted_metrics_from_accumulator(accumulator: Dict[str, Any]) -> Dict[str, float]:
    total_goal_weight = float(accumulator["total_goal_weight"])
    if total_goal_weight <= 0.0:
        return {name: 0.0 for name in WEIGHTED_METRIC_NAMES}
    return {
        key: accumulator["weighted_metric_sums"][key] / total_goal_weight
        for key in WEIGHTED_METRIC_NAMES
    }


def _optional_metric_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)
        if np.isfinite(value):
            return value
    return None


def _spl_diagnostics(env: habitat.Env) -> Tuple[Optional[float], Optional[float]]:
    task = getattr(env, "_task", None)
    if task is None:
        task = getattr(env, "task", None)

    measurements = getattr(task, "measurements", None)
    measures = getattr(measurements, "measures", None)
    if not isinstance(measures, dict):
        return None, None

    spl_measure = measures.get("spl")
    if spl_measure is None:
        for measure in measures.values():
            if hasattr(measure, "_agent_episode_distance") or hasattr(measure, "_reference_distance"):
                spl_measure = measure
                break
    if spl_measure is None:
        return None, None

    agent_episode_distance_m = _optional_metric_float(
        getattr(spl_measure, "_agent_episode_distance", None)
    )
    reference_distance_m = _optional_metric_float(
        getattr(spl_measure, "_reference_distance", None)
    )
    return agent_episode_distance_m, reference_distance_m


def _is_submit_action(action: Any, submit_action_name: str) -> bool:
    action_name = str(submit_action_name).strip().upper()
    if isinstance(action, dict):
        value = action.get("action")
        return isinstance(value, str) and value.strip().upper() == action_name
    if isinstance(action, str):
        return action.strip().upper() == action_name
    return False


def _is_stop_action(action: Any, stop_action_name: str) -> bool:
    action_name = str(stop_action_name).strip().upper()
    if isinstance(action, dict):
        value = action.get("action")
        return isinstance(value, str) and value.strip().upper() == action_name
    if isinstance(action, str):
        return action.strip().upper() == action_name
    return False


def _observation_image(observations: Optional[Dict[str, Any]], *keys: str) -> Optional[np.ndarray]:
    if not isinstance(observations, dict):
        return None
    for key in keys:
        value = observations.get(key)
        if value is None:
            continue
        image = np.asarray(value)
        if image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
        if image.ndim in {2, 3}:
            return image
    return None


def _semantic_color(semantic_id: int) -> Tuple[int, int, int]:
    sid = int(semantic_id)
    if sid == 0:
        return (0, 0, 0)
    return (
        (sid * 37) % 256,
        (sid * 67) % 256,
        (sid * 97) % 256,
    )


def _semantic_to_rgb(semantic: np.ndarray) -> np.ndarray:
    semantic_ids = np.asarray(semantic).astype(np.int64, copy=False)
    if semantic_ids.ndim == 3:
        semantic_ids = semantic_ids[:, :, 0]
    semantic_rgb = np.zeros((*semantic_ids.shape, 3), dtype=np.uint8)
    for semantic_id in np.unique(semantic_ids):
        semantic_rgb[semantic_ids == semantic_id] = _semantic_color(int(semantic_id))
    return semantic_rgb


def _save_observation_png(image: np.ndarray, output_path: Path) -> Optional[str]:
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.clip(array, 0, 255).astype(np.uint8)
    elif array.ndim == 3:
        if array.shape[2] > 3:
            array = array[:, :, :3]
        array = np.clip(array, 0, 255).astype(np.uint8)
    else:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(output_path)
    return str(output_path)


def _save_action_observation_bundle(
    root_dir: Path,
    *,
    scene_name: str,
    episode_index: int,
    episode_id: str,
    step_idx: int,
    action_name: str,
    phase: str,
    observations: Optional[Dict[str, Any]],
) -> Dict[str, Optional[str]]:
    phase_token = str(phase).strip().lower()
    action_dir = root_dir / scene_name / f"episode_{episode_index}_id_{episode_id}" / (
        f"step_{int(step_idx):04d}_{str(action_name).strip().lower()}"
    )

    rgb = _observation_image(observations, "rgb", "RGB_SENSOR", "rgb_sensor")
    semantic = _observation_image(observations, "semantic", "SEMANTIC_SENSOR", "semantic_sensor")

    payload: Dict[str, Optional[str]] = {"rgb": None, "semantic": None}
    if rgb is not None:
        payload["rgb"] = _save_observation_png(rgb, action_dir / f"{phase_token}_rgb.png")
    if semantic is not None:
        payload["semantic"] = _save_observation_png(
            _semantic_to_rgb(semantic),
            action_dir / f"{phase_token}_semantic.png",
        )
    return payload

def main() -> None:
    args = parse_args()
    args = apply_eval_config(args)

    if args.dataset_path is None or not str(args.dataset_path).strip():
        raise RuntimeError(
            "dataset_path is required. Pass --dataset-path or provide it in --eval-config."
        )

    cfg = build_config(args)
    max_steps = int(cfg.ENVIRONMENT.MAX_EPISODE_STEPS)
    dataset_payload = _load_dataset_payload(str(args.dataset_path))
    instance_index = _flatten_instances(dataset_payload.get("instances", {}))

    print("dataset_path_in_use={}".format(cfg.DATASET.DATA_PATH))
    print("task_type_in_use={}".format(cfg.TASK.TYPE))
    if hasattr(cfg.TASK, "GOAL_ORDER_MODE"):
        print("goal_order_mode_in_use={}".format(cfg.TASK.GOAL_ORDER_MODE))
    print("max_episode_steps_in_use={}".format(max_steps))

    env = habitat.Env(config=cfg)
    policy = _build_policy(args)

    all_scene_names = _all_scene_names(list(env.episodes))
    scene_range_label = _scene_subset_label(all_scene_names, args)
    scene_range_start, scene_range_end, selected_scene_names = _resolve_scene_subset(
        all_scene_names,
        args,
    )
    print("scene_range_in_use={}".format(scene_range_label))
    print("selected_scenes_in_use={}".format(selected_scene_names))

    episodes = _prepare_episode_list(env, args)
    if not episodes:
        policy.close()
        env.close()
        print("No episodes selected.")
        return

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.episode_log is None or args.mean_log is None:
        run_base_dir = os.path.join(
            str(args.output_parent_dir),
            str(args.exp_name),
            scene_range_label,
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
    if args.save_action_observations and args.action_observation_dir is None:
        args.action_observation_dir = os.path.join(run_base_dir, "action_observations")

    os.makedirs(os.path.dirname(args.episode_log) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.mean_log) or ".", exist_ok=True)
    if args.video:
        os.makedirs(str(args.video_dir), exist_ok=True)
    os.makedirs(str(args.prompt_image_dir), exist_ok=True)
    if args.save_action_observations and args.action_observation_dir is not None:
        os.makedirs(str(args.action_observation_dir), exist_ok=True)
    prompt_image_dir = Path(args.prompt_image_dir)
    action_observation_dir = Path(args.action_observation_dir) if args.action_observation_dir is not None else None
    if args.video_dir is not None:
        metric_log_dir = str(Path(args.video_dir).resolve().parent / "logs")
    else:
        metric_log_dir = os.path.join(run_base_dir, "logs")
    os.makedirs(metric_log_dir, exist_ok=True)
    episode_metric_txt = os.path.join(metric_log_dir, "episode_metrics.txt")
    running_metric_txt = os.path.join(metric_log_dir, "running_weighted_mean.txt")
    scene_mean_log = os.path.join(run_base_dir, "scene_mean.json")
    scene_metric_txt = os.path.join(metric_log_dir, "scene_metrics.txt")

    episode_writer = open(args.episode_log, "w", encoding="utf-8")
    episode_metric_writer = open(episode_metric_txt, "w", encoding="utf-8")
    running_metric_writer = open(running_metric_txt, "w", encoding="utf-8")

    all_metrics = _new_metric_accumulator()
    scene_metrics: Dict[str, Dict[str, Any]] = {}
    scene_order: List[str] = []
    scene_episode_budget: Dict[str, int] = {}
    for episode in episodes:
        scene_name = _scene_key(str(getattr(episode, "scene_id", "")))
        if scene_name not in scene_episode_budget:
            scene_order.append(scene_name)
            scene_episode_budget[scene_name] = 0
        scene_episode_budget[scene_name] += 1

    current_scene_name: Optional[str] = None

    if True:
        for episode_index, target_episode in enumerate(episodes):
            env.current_episode = target_episode
            observations = env.reset()
            episode = env.current_episode

            episode_id_text = str(getattr(episode, "episode_id", ""))
            scene_name = _scene_key(str(getattr(episode, "scene_id", "")))
            episode_uid = f"{scene_name}:{episode_id_text}"
            if scene_name != current_scene_name:
                current_scene_name = scene_name
                print(
                    "[scene] {scene} ({count} episodes)".format(
                        scene=scene_name,
                        count=scene_episode_budget.get(scene_name, 0),
                    )
                )
            goal_specs = _normalize_task_specs(getattr(episode, "tasks", None))
            goal_summary = [f"{instance_key}:{modality}" for instance_key, modality in goal_specs]

            goal_payloads: List[Dict[str, Any]] = []
            goal_input_summaries: List[Dict[str, Any]] = []
            scene_prompt_image_dir = prompt_image_dir / scene_name
            for goal_idx, (instance_key, modality) in enumerate(goal_specs):
                instance_record = instance_index.get(instance_key)
                goal_payload = _build_goal_input_payload(
                    env=env,
                    instance_key=instance_key,
                    modality=modality,
                    instance_record=instance_record,
                )
                goal_payloads.append(goal_payload)
                goal_input_summary = _summarize_goal_input_payload(goal_idx, goal_payload)
                image_path = _save_goal_input_image_if_needed(
                    payload=goal_payload,
                    prompt_image_dir=scene_prompt_image_dir,
                    episode_id=episode_id_text,
                    goal_index=goal_idx,
                    instance_key=instance_key,
                    modality=modality,
                )
                if image_path is not None:
                    goal_input_summary["image_path"] = image_path
                goal_input_summaries.append(goal_input_summary)

            context_goal_payloads = tuple(goal_payloads)
            policy.reset(env=env, episode=episode, observations=observations)

            print(
                "[policy_task] episode={idx} id={eid} task={task}".format(
                    idx=episode_index,
                    eid=episode_id_text,
                    task=goal_summary,
                )
            )

            task = getattr(env, "_task", None)

            done = False
            step_idx = 0
            frames = []
            audios = []
            audio_peak_max = 0.0
            audio_rms_acc = 0.0
            audio_active_steps = 0

            while not done and step_idx < max_steps:
                context = build_lifelong_eval_context(
                    step_idx,
                    goal_payloads=context_goal_payloads,
                )
                action = policy.act(
                    env=env,
                    episode=episode,
                    observations=observations,
                    context=context,
                )

                should_dump_action_obs = bool(
                    args.save_action_observations
                    and action_observation_dir is not None
                    and (
                        _is_submit_action(action, args.submit_action_name)
                        or _is_stop_action(action, "STOP")
                    )
                )
                action_name = None
                if _is_submit_action(action, args.submit_action_name):
                    action_name = str(args.submit_action_name)
                elif _is_stop_action(action, "STOP"):
                    action_name = "STOP"

                action_snapshot: Dict[str, Any] = {}
                if should_dump_action_obs and action_name is not None:
                    action_snapshot["before"] = _save_action_observation_bundle(
                        action_observation_dir,
                        scene_name=scene_name,
                        episode_index=episode_index,
                        episode_id=episode_id_text,
                        step_idx=step_idx,
                        action_name=action_name,
                        phase="before",
                        observations=observations,
                    )

                observations = env.step(action)
                done = bool(env.episode_over)

                if should_dump_action_obs and action_name is not None:
                    action_snapshot["after"] = _save_action_observation_bundle(
                        action_observation_dir,
                        scene_name=scene_name,
                        episode_index=episode_index,
                        episode_id=episode_id_text,
                        step_idx=step_idx,
                        action_name=action_name,
                        phase="after",
                        observations=observations,
                    )
                    feedback = {}
                    if task is not None and hasattr(task, "get_last_action_feedback"):
                        feedback = task.get_last_action_feedback()
                    snapshot_dir = (
                        action_observation_dir
                        / scene_name
                        / f"episode_{episode_index}_id_{episode_id_text}"
                        / f"step_{int(step_idx):04d}_{str(action_name).strip().lower()}"
                    )
                    metadata_path = snapshot_dir / "metadata.json"
                    metadata_path.parent.mkdir(parents=True, exist_ok=True)
                    metadata = {
                        "scene": scene_name,
                        "episode_index": int(episode_index),
                        "episode_id": str(episode_id_text),
                        "step": int(step_idx),
                        "action": str(action_name),
                        "done_after_action": bool(done),
                        "before": action_snapshot.get("before", {}),
                        "after": action_snapshot.get("after", {}),
                        "feedback": feedback,
                    }
                    metadata_path.write_text(
                        json.dumps(metadata, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )

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
                step_idx += 1

            if args.video_audio:
                audio_frame_count = len(audios)
                audio_mean_rms = audio_rms_acc / float(audio_frame_count) if audio_frame_count > 0 else 0.0
                print(
                    "  [audio_diag] frames={frames} active_frames={active} peak_max={peak:.6f} mean_rms={rms:.6f} "
                    "offset={offset} duration={duration} schedule={schedule}".format(
                        frames=audio_frame_count,
                        active=audio_active_steps,
                        peak=audio_peak_max,
                        rms=audio_mean_rms,
                        offset=getattr(episode, "offset", None),
                        duration=getattr(episode, "duration", None),
                        schedule=getattr(episode, "sound_source_schedule", None),
                    )
                )
                if audio_active_steps == 0:
                    print(
                        "  [audio_diag] warning: no active audiogoal frame above threshold; "
                        "check duration/schedule semantics and source alignment."
                    )
            metrics = {
                str(key): float(value)
                for key, value in env.get_metrics().items()
                if isinstance(value, (int, float))
            }
            metrics = _format_episode_metrics(
                metrics,
                _episode_goal_count(episode, goal_specs),
            )
            agent_episode_distance_m, reference_distance_m = _spl_diagnostics(env)
            if agent_episode_distance_m is not None or reference_distance_m is not None:
                print(
                    "  [path_diag] agent_episode_distance_m={agent} reference_distance_m={ref}".format(
                        agent=(
                            "{:.3f}".format(agent_episode_distance_m)
                            if agent_episode_distance_m is not None
                            else "<unknown>"
                        ),
                        ref=(
                            "{:.3f}".format(reference_distance_m)
                            if reference_distance_m is not None
                            else "<unknown>"
                        ),
                    )
                )
            goal_weight = float(metrics.get("num_goals", 1.0))
            _update_metric_accumulator(
                all_metrics,
                metrics,
                goal_weight=goal_weight,
                truncated=bool(not done),
            )
            if scene_name not in scene_metrics:
                scene_metrics[scene_name] = _new_metric_accumulator()
            _update_metric_accumulator(
                scene_metrics[scene_name],
                metrics,
                goal_weight=goal_weight,
                truncated=bool(not done),
            )

            episodes_done = int(all_metrics["episodes_done"])
            episodes_truncated = int(all_metrics["episodes_truncated"])
            running_weighted_mean = _weighted_metrics_from_accumulator(all_metrics)
            running_logged_metrics = dict(running_weighted_mean)
            for key in RUNNING_MEAN_METRIC_NAMES:
                if key in all_metrics["sum_metrics"]:
                    running_logged_metrics[key] = all_metrics["sum_metrics"][key] / float(episodes_done)

            row = {
                "episode_index": episode_index,
                "episode_id": str(getattr(episode, "episode_id", "")),
                "episode_uid": episode_uid,
                "scene": scene_name,
                "scene_id": str(getattr(episode, "scene_id", "")),
                "steps": int(step_idx),
                "truncated": bool(not done),
                "agent_episode_distance_m": agent_episode_distance_m,
                "reference_distance_m": reference_distance_m,
                "metrics": metrics,
                "goal_specs": goal_summary,
                "goal_inputs": goal_input_summaries,
                "goal_order_mode": str(
                    getattr(episode, "goal_order_mode", getattr(cfg.TASK, "GOAL_ORDER_MODE", "ordered"))
                ),
            }
            episode_writer.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            episode_writer.flush()
            episode_metric_writer.write(
                "[episode {idx}] uid={uid} agent_episode_distance_m={agent} reference_distance_m={ref} metrics={metrics}\n".format(
                    idx=episode_index,
                    uid=row["episode_uid"],
                    agent=(
                        "{:.3f}".format(agent_episode_distance_m)
                        if agent_episode_distance_m is not None
                        else "<unknown>"
                    ),
                    ref=(
                        "{:.3f}".format(reference_distance_m)
                        if reference_distance_m is not None
                        else "<unknown>"
                    ),
                    metrics=json.dumps(metrics, sort_keys=True),
                )
            )
            episode_metric_writer.flush()
            running_metric_writer.write(
                "[episode {idx}] uid={uid} weighted_mean={metrics}\n".format(
                    idx=episode_index,
                    uid=row["episode_uid"],
                    metrics=json.dumps(running_logged_metrics, sort_keys=True),
                )
            )
            running_metric_writer.flush()

            if episode_index % max(1, args.print_every) == 0:
                print(
                    "[episode {idx}] uid={uid} steps={steps} truncated={tr} goals={goals} metrics={metrics}".format(
                        idx=episode_index,
                        uid=row["episode_uid"],
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
                scene_video_dir = os.path.join(str(args.video_dir), scene_name)
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
                        scene_video_dir,
                        video_name,
                        muxed_audios,
                        cfg.SIMULATOR.AUDIO.RIR_SAMPLING_RATE,
                        fps=int(fps),
                    )
                else:
                    images_to_video(frames, scene_video_dir, video_name, fps=int(fps))

    episode_writer.close()
    episode_metric_writer.close()
    running_metric_writer.close()
    policy.close()
    env.close()

    episodes_done = int(all_metrics["episodes_done"])
    episodes_truncated = int(all_metrics["episodes_truncated"])
    if episodes_done == 0:
        print("No episodes evaluated.")
        return

    mean_metrics = _mean_metrics_from_accumulator(all_metrics)
    weighted_mean_metrics = _weighted_metrics_from_accumulator(all_metrics)
    scene_summaries: List[Dict[str, Any]] = []
    for scene_name in scene_order:
        accumulator = scene_metrics.get(scene_name)
        if accumulator is None or int(accumulator["episodes_done"]) <= 0:
            continue
        scene_summary = {
            "scene": scene_name,
            "episodes": int(accumulator["episodes_done"]),
            "truncated_episodes": int(accumulator["episodes_truncated"]),
            "mean_metrics": _mean_metrics_from_accumulator(accumulator),
            "weighted_mean_metrics": _weighted_metrics_from_accumulator(accumulator),
        }
        scene_summaries.append(scene_summary)

    with open(scene_metric_txt, "w", encoding="utf-8") as handle:
        for scene_summary in scene_summaries:
            handle.write(json.dumps(scene_summary, ensure_ascii=False, sort_keys=True) + "\n")

    with open(scene_mean_log, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "task_type": str(cfg.TASK.TYPE),
                "dataset_path": str(cfg.DATASET.DATA_PATH),
                "policy": str(args.policy),
                "scene_range_label": scene_range_label,
                "scene_range_start": int(scene_range_start),
                "scene_range_end": int(scene_range_end),
                "selected_scenes": list(selected_scene_names),
                "scenes": scene_summaries,
            },
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

    summary = {
        "episodes": int(episodes_done),
        "truncated_episodes": int(episodes_truncated),
        "task_type": str(cfg.TASK.TYPE),
        "dataset_path": str(cfg.DATASET.DATA_PATH),
        "policy": str(args.policy),
        "scene_range_label": scene_range_label,
        "scene_range_start": int(scene_range_start),
        "scene_range_end": int(scene_range_end),
        "selected_scenes": list(selected_scene_names),
        "mean_metrics": mean_metrics,
        "weighted_mean_metrics": weighted_mean_metrics,
        "scene_mean_log": scene_mean_log,
    }

    with open(args.mean_log, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)

    print("Completed {} episodes".format(episodes_done))
    print("Truncated episodes: {}".format(episodes_truncated))
    print("Mean metrics: {}".format(json.dumps(mean_metrics, sort_keys=True)))
    print("Episode log: {}".format(args.episode_log))
    print("Mean log: {}".format(args.mean_log))
    print("Scene mean log: {}".format(scene_mean_log))
    print("Episode metric txt: {}".format(episode_metric_txt))
    print("Running mean txt: {}".format(running_metric_txt))
    print("Scene metric txt: {}".format(scene_metric_txt))


if __name__ == "__main__":
    main()
