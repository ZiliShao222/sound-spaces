#!/usr/bin/env python3

from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import habitat
import numpy as np
import soundspaces  # noqa: F401 - register datasets/tasks/sims
from habitat.utils.visualizations.utils import images_to_video

from ss_baselines.common.omni_long_eval_policy import build_lifelong_eval_context
from ss_baselines.common.utils import observations_to_image, images_to_video_with_audio
from soundspaces.tasks.omni_long_eval_utils import (
    _build_goal_prompt,
    _build_policy,
    _flatten_instances,
    _load_dataset_payload,
    _normalize_task_specs,
    _prepare_episode_list,
    _save_prompt_image_if_needed,
    _sound_id_for_goal,
    apply_eval_config,
    build_config,
    parse_args,
)

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
            metrics = {
                str(key): float(value)
                for key, value in env.get_metrics().items()
                if isinstance(value, (int, float))
            }
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
                "submit_count": int(getattr(task, "submit_count", 0)),
                "submit_limit": int(getattr(task, "submit_limit", 0)),
                "submit_rejected_count": int(getattr(task, "submit_rejected_count", 0)),
                "remaining_submit_quota": int(getattr(task, "remaining_submit_quota", 0)),
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
