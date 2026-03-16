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

from ss_baselines.common.omni_long_eval_policy import build_lifelong_eval_context
from ss_baselines.common.utils import observations_to_image, images_to_video_with_audio
from soundspaces.tasks.omni_long_eval_utils import (
    _build_goal_input_payload,
    _build_policy,
    _episode_goal_count,
    _format_goal_input_summary,
    _format_episode_metrics,
    _flatten_instances,
    _load_dataset_payload,
    _normalize_task_specs,
    _prepare_episode_list,
    _save_goal_input_image_if_needed,
    _summarize_goal_input_payload,
    apply_eval_config,
    build_config,
    parse_args,
)


WEIGHTED_METRIC_NAMES = ("success", "spl", "softspl")
RUNNING_MEAN_METRIC_NAMES = ("na",)


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

    agent_episode_distance_m: Optional[float] = None
    reference_distance_m: Optional[float] = None
    try:
        agent_episode_distance_m = float(getattr(spl_measure, "_agent_episode_distance"))
    except Exception:
        pass
    try:
        reference_distance_m = float(getattr(spl_measure, "_reference_distance"))
    except Exception:
        pass
    return agent_episode_distance_m, reference_distance_m


def _is_submit_action(action: Any, submit_action_name: str) -> bool:
    action_name = str(submit_action_name).strip().upper()
    if isinstance(action, dict):
        value = action.get("action")
        return isinstance(value, str) and value.strip().upper() == action_name
    if isinstance(action, str):
        return action.strip().upper() == action_name
    return False

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
    if args.video_dir is not None:
        metric_log_dir = str(Path(args.video_dir).resolve().parent / "logs")
    else:
        metric_log_dir = os.path.join(run_base_dir, "logs")
    os.makedirs(metric_log_dir, exist_ok=True)
    episode_metric_txt = os.path.join(metric_log_dir, "episode_metrics.txt")
    running_metric_txt = os.path.join(metric_log_dir, "running_weighted_mean.txt")

    episode_writer = open(args.episode_log, "w", encoding="utf-8")
    episode_metric_writer = open(episode_metric_txt, "w", encoding="utf-8")
    running_metric_writer = open(running_metric_txt, "w", encoding="utf-8")

    episodes = _prepare_episode_list(env, args)
    if not episodes:
        episode_writer.close()
        env.close()
        print("No episodes selected.")
        return

    episodes_by_id = {int(getattr(ep, "episode_id", 0)): ep for ep in episodes}
    ordered_episode_ids = sorted(episodes_by_id.keys())

    sum_metrics: Dict[str, float] = {}
    weighted_metric_sums = {name: 0.0 for name in WEIGHTED_METRIC_NAMES}
    total_goal_weight = 0.0
    episodes_done = 0
    episodes_truncated = 0

    try:
        for episode_index, episode_id in enumerate(ordered_episode_ids):
            target_episode = episodes_by_id[episode_id]
            env.current_episode = target_episode
            observations = env.reset()
            episode = env.current_episode

            episode_id_text = str(getattr(episode, "episode_id", ""))
            goal_specs = _normalize_task_specs(getattr(episode, "tasks", None))
            goal_summary = [f"{instance_key}:{modality}" for instance_key, modality in goal_specs]

            goal_payloads: List[Dict[str, Any]] = []
            goal_input_summaries: List[Dict[str, Any]] = []
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
                    prompt_image_dir=prompt_image_dir,
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
                step_idx += 1

            if not done:
                episodes_truncated += 1

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
            for key, value in metrics.items():
                sum_metrics[key] = sum_metrics.get(key, 0.0) + float(value)
            total_goal_weight += goal_weight
            for key in WEIGHTED_METRIC_NAMES:
                weighted_metric_sums[key] += float(metrics.get(key, 0.0)) * goal_weight
            running_weighted_mean = {
                key: (weighted_metric_sums[key] / total_goal_weight if total_goal_weight > 0 else 0.0)
                for key in WEIGHTED_METRIC_NAMES
            }
            episodes_seen = episodes_done + 1
            running_logged_metrics = dict(running_weighted_mean)
            for key in RUNNING_MEAN_METRIC_NAMES:
                if key in sum_metrics:
                    running_logged_metrics[key] = sum_metrics[key] / float(episodes_seen)

            row = {
                "episode_index": episode_index,
                "episode_id": str(getattr(episode, "episode_id", "")),
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
                "[episode {idx}] id={eid} agent_episode_distance_m={agent} reference_distance_m={ref} metrics={metrics}\n".format(
                    idx=episode_index,
                    eid=row["episode_id"],
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
                "[episode {idx}] id={eid} weighted_mean={metrics}\n".format(
                    idx=episode_index,
                    eid=row["episode_id"],
                    metrics=json.dumps(running_logged_metrics, sort_keys=True),
                )
            )
            running_metric_writer.flush()

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
        episode_metric_writer.close()
        running_metric_writer.close()
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
    print("Episode metric txt: {}".format(episode_metric_txt))
    print("Running mean txt: {}".format(running_metric_txt))


if __name__ == "__main__":
    main()
