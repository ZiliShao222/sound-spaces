#!/usr/bin/env python3

import argparse
import json
import os
import datetime as dt
from typing import Dict, Any

import habitat
import numpy as np
import soundspaces  # noqa: F401 - register SoundSpaces datasets/tasks
from ss_baselines.av_nav.config.default import get_task_config
from ss_baselines.common.utils import images_to_video_with_audio
from ss_baselines.common.utils import observations_to_image
from habitat.utils.visualizations.utils import images_to_video


DEFAULT_CONFIG = "configs/semantic_audionav/av_nav/mp3d/semantic_audiogoal.yaml"
DEFAULT_DISTRACTOR_DIR = "data/sounds/1s_all_distractor"
SPLIT_ALIASES = {
    "val": "val",
    "test": "test",
    "train": "train",
    "distractor": "test_distractor",
    "val_distractor": "val_distractor",
    "test_distractor": "test_distractor",
    "train_distractor": "train_distractor",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate policies on SemanticAudioNav with a selectable split."
    )
    parser.add_argument(
        "--exp-config",
        default=DEFAULT_CONFIG,
        help="Task config path for SemanticAudioNav.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=sorted(SPLIT_ALIASES.keys()),
        help="Dataset split alias (test/val/distractor/...).",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Number of episodes to run (default: all in split).",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="RGB/Depth resolution (square).",
    )
    parser.add_argument(
        "--policy",
        default="random",
        choices=["random"],
        help="Policy name to evaluate.",
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        default=True,
        help="Use ContinuousSoundSpacesSim (ss2.0) settings.",
    )
    parser.add_argument(
        "--no-stop-steps",
        type=int,
        default=100,
        help="Disallow STOP action for the first N steps of each episode.",
    )
    parser.add_argument(
        "--episode-log",
        default=None,
        help="Path to per-episode metrics log file (default: auto under ./logs).",
    )
    parser.add_argument(
        "--audio-activity-log",
        default=None,
        help="Path to per-episode audio activity log (JSONL).",
    )
    parser.add_argument(
        "--audio-activity-threshold",
        type=float,
        default=1e-6,
        help="Amplitude threshold to mark a timestep as audible.",
    )
    parser.add_argument(
        "--mean-log",
        default=None,
        help="Path to mean metrics log file (default: auto under ./logs).",
    )
    parser.add_argument(
        "--video",
        action="store_true",
        default=True,
        help="Save per-episode visualization videos.",
    )
    parser.add_argument(
        "--video-dir",
        default="video_dir",
        help="Directory to save videos when --video is set.",
    )
    parser.add_argument(
        "--video-fps",
        type=int,
        default=None,
        help="FPS for saved videos (default: 1/STEP_TIME).",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Print metrics every N episodes.",
    )
    parser.add_argument(
        "--split-l",
        type=int,
        default=0,
        help="Start scene index (inclusive) within the split.",
    )
    parser.add_argument(
        "--split-r",
        type=int,
        default=10,
        help="End scene index (exclusive) within the split.",
    )
    parser.add_argument(
        "--video-audio",
        action="store_true",
        default=True,
        help="Attach audiogoal audio to saved videos when available.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> habitat.Config:
    cfg = get_task_config(config_paths=[args.exp_config])
    split_name = SPLIT_ALIASES[args.split]

    cfg.defrost()
    cfg.DATASET.SPLIT = split_name

    cfg.SIMULATOR.RGB_SENSOR.WIDTH = args.resolution
    cfg.SIMULATOR.RGB_SENSOR.HEIGHT = args.resolution
    cfg.SIMULATOR.DEPTH_SENSOR.WIDTH = args.resolution
    cfg.SIMULATOR.DEPTH_SENSOR.HEIGHT = args.resolution

    cfg.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]

    if args.continuous:
        cfg.SIMULATOR.TYPE = "ContinuousSoundSpacesSim"
        cfg.SIMULATOR.USE_RENDERED_OBSERVATIONS = False
        cfg.SIMULATOR.STEP_TIME = 0.25
        cfg.SIMULATOR.FORWARD_STEP_SIZE = 0.25
        cfg.SIMULATOR.AUDIO.CROSSFADE = True
        cfg.DATASET.CONTINUOUS = True

    if args.video:
        if "TOP_DOWN_MAP" not in cfg.TASK.MEASUREMENTS:
            cfg.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
        if "COLLISIONS" not in cfg.TASK.MEASUREMENTS:
            cfg.TASK.MEASUREMENTS.append("COLLISIONS")

    if args.video_audio and "AUDIOGOAL_SENSOR" not in cfg.TASK.SENSORS:
        cfg.TASK.SENSORS.append("AUDIOGOAL_SENSOR")
    if args.audio_activity_log and "AUDIOGOAL_SENSOR" not in cfg.TASK.SENSORS:
        cfg.TASK.SENSORS.append("AUDIOGOAL_SENSOR")

    if "distractor" in split_name:
        cfg.SIMULATOR.AUDIO.HAS_DISTRACTOR_SOUND = True
        if not hasattr(cfg.SIMULATOR.AUDIO, "DISTRACTOR_SOUND_DIR"):
            cfg.SIMULATOR.AUDIO.DISTRACTOR_SOUND_DIR = DEFAULT_DISTRACTOR_DIR
        elif not cfg.SIMULATOR.AUDIO.DISTRACTOR_SOUND_DIR:
            cfg.SIMULATOR.AUDIO.DISTRACTOR_SOUND_DIR = DEFAULT_DISTRACTOR_DIR

    cfg.freeze()
    return cfg


def filter_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    filtered = {}
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            filtered[k] = float(v)
    return filtered


def is_stop_action(action: Any) -> bool:
    if isinstance(action, dict) and "action" in action:
        act = action["action"]
    else:
        act = action
    if isinstance(act, str):
        return act.upper() == "STOP"
    if isinstance(act, int):
        return act == 0
    return False


def get_goal_label(episode: Any) -> str:
    goal_category = ""
    if hasattr(episode, "object_category") and episode.object_category:
        goal_category = str(episode.object_category)
    goal_instance = ""
    if hasattr(episode, "goals") and episode.goals:
        goal = episode.goals[0]
        if hasattr(goal, "object_name") and goal.object_name:
            goal_instance = str(goal.object_name)
        elif hasattr(goal, "object_id") and goal.object_id:
            goal_instance = str(goal.object_id)
    if goal_category and goal_instance:
        return f"{goal_category}:{goal_instance}"
    if goal_category:
        return goal_category
    if goal_instance:
        return goal_instance
    if hasattr(episode, "sound_id") and episode.sound_id:
        return os.path.basename(str(episode.sound_id))
    return ""


def main() -> None:
    args = parse_args()
    cfg = build_config(args)

    env = habitat.Env(config=cfg)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    split_range = f"{args.split_l}:{args.split_r}"
    if args.episode_log is None or args.mean_log is None:
        base_dir = os.path.join("logs", f"[{split_range}]_{timestamp}")
        os.makedirs(base_dir, exist_ok=True)
        if args.episode_log is None:
            args.episode_log = os.path.join(base_dir, "semantic_episode.log")
        if args.mean_log is None:
            args.mean_log = os.path.join(base_dir, "semantic_mean.log")

    scene_names = []
    for ep in env.episodes:
        scene_base = os.path.basename(ep.scene_id).split(".")[0]
        if scene_base not in scene_names:
            scene_names.append(scene_base)
    scene_names.sort()
    split_l = max(0, args.split_l)
    split_r = min(len(scene_names), args.split_r)
    selected_scenes = set(scene_names[split_l:split_r])
    if selected_scenes:
        env.episodes = [
            ep
            for ep in env.episodes
            if os.path.basename(ep.scene_id).split(".")[0] in selected_scenes
        ]

    total_episodes = args.num_episodes or len(env.episodes)
    episode_count = 0
    sum_metrics: Dict[str, float] = {}
    printed_shape = False

    if args.video:
        if args.video_dir == "video_dir":
            args.video_dir = os.path.join(
                "output_video",
                f"[{split_range}]_{timestamp}",
            )
        os.makedirs(args.video_dir, exist_ok=True)
    if args.episode_log:
        os.makedirs(os.path.dirname(args.episode_log) or ".", exist_ok=True)
        episode_log_f = open(args.episode_log, "w", encoding="utf-8")
    else:
        episode_log_f = None
    if args.audio_activity_log:
        os.makedirs(os.path.dirname(args.audio_activity_log) or ".", exist_ok=True)
        audio_activity_log_f = open(args.audio_activity_log, "w", encoding="utf-8")
    else:
        audio_activity_log_f = None
    if args.mean_log:
        os.makedirs(os.path.dirname(args.mean_log) or ".", exist_ok=True)
        mean_log_f = open(args.mean_log, "w", encoding="utf-8")
    else:
        mean_log_f = None

    while episode_count < total_episodes:
        observations = env.reset()
        if not printed_shape and "spectrogram" in observations:
            print(f"spectrogram shape: {observations['spectrogram'].shape}")
            printed_shape = True

        frames = []
        audios = []
        audio_intervals = []
        audio_interval_start = None
        step_count = 0
        done = False
        while not done:
            if args.policy == "random":
                action = env.action_space.sample()
                if step_count < args.no_stop_steps:
                    while is_stop_action(action):
                        action = env.action_space.sample()
            else:
                raise NotImplementedError(f"Unsupported policy: {args.policy}")
            observations = env.step(action)
            done = env.episode_over
            if args.video:
                frame = observations_to_image(observations, env.get_metrics())
                frames.append(frame)
                if args.video_audio and "audiogoal" in observations:
                    audios.append(observations["audiogoal"])
            if audio_activity_log_f is not None:
                if "audiogoal" in observations:
                    audio = observations["audiogoal"]
                    audio_active = float(np.max(np.abs(audio))) > args.audio_activity_threshold
                elif hasattr(env.sim, "is_silent"):
                    audio_active = not env.sim.is_silent
                else:
                    audio_active = False
                if audio_active:
                    if audio_interval_start is None:
                        audio_interval_start = step_count
                elif audio_interval_start is not None:
                    audio_intervals.append([audio_interval_start, step_count - 1])
                    audio_interval_start = None
            step_count += 1
        if audio_activity_log_f is not None and audio_interval_start is not None:
            audio_intervals.append([audio_interval_start, step_count - 1])

        metrics = filter_metrics(env.get_metrics())
        for k, v in metrics.items():
            sum_metrics[k] = sum_metrics.get(k, 0.0) + v
        mean_metrics = {k: v / (episode_count + 1) for k, v in sum_metrics.items()}

        ep = env.current_episode
        goal_label = get_goal_label(ep)
        goal_suffix = f" goal={goal_label}" if goal_label else ""
        line = (
            f"[episode {episode_count}] scene={ep.scene_id} id={ep.episode_id}"
            f"{goal_suffix} metrics={json.dumps(metrics, sort_keys=True)}"
        )
        if episode_log_f is not None:
            episode_log_f.write(line + "\n")
            episode_log_f.flush()
        if audio_activity_log_f is not None:
            audio_activity_log_f.write(
                json.dumps(
                    {
                        "episode": episode_count,
                        "scene": ep.scene_id,
                        "episode_id": ep.episode_id,
                        "sound_intervals": audio_intervals,
                        "threshold": args.audio_activity_threshold,
                        "steps": step_count,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            audio_activity_log_f.flush()
        if mean_log_f is not None:
            mean_log_f.write(
                json.dumps(
                    {
                        "split": cfg.DATASET.SPLIT,
                        "episode": episode_count,
                        "episodes": episode_count + 1,
                        "mean_metrics": mean_metrics,
                    },
                    sort_keys=True,
                )
                + "\n"
            )
            mean_log_f.flush()

        if episode_count % args.print_every == 0:
            print(line)

        if args.video and len(frames) > 0:
            video_name = f"episode_{episode_count}"
            fps = args.video_fps or int(1 / cfg.SIMULATOR.STEP_TIME)
            if args.video_audio and len(audios) > 0:
                images_to_video_with_audio(
                    frames,
                    args.video_dir,
                    video_name,
                    audios,
                    cfg.SIMULATOR.AUDIO.RIR_SAMPLING_RATE,
                    fps=fps,
                )
            else:
                images_to_video(frames, args.video_dir, video_name, fps=fps)

        episode_count += 1

    if episode_count == 0:
        print("No episodes evaluated.")
        return

    mean_metrics = {k: v / episode_count for k, v in sum_metrics.items()}
    if episode_log_f is not None:
        episode_log_f.close()
    if audio_activity_log_f is not None:
        audio_activity_log_f.close()
    if mean_log_f is not None:
        mean_log_f.close()
    print(f"Completed {episode_count} episodes on split={cfg.DATASET.SPLIT}")
    print(f"Mean metrics: {json.dumps(mean_metrics, sort_keys=True)}")


if __name__ == "__main__":
    main()
