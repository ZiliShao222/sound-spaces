#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Random policy evaluation script for SoundSpaces AudioNav.

import argparse
import json
import logging
import os
from typing import Dict, List

import numpy as np

import soundspaces  # noqa: F401
from habitat.datasets import make_dataset
from ss_baselines.omni_long.config.default import get_config
from ss_baselines.common.environments import get_env_class
from ss_baselines.common.utils import observations_to_image
from habitat.utils.visualizations.utils import images_to_video


def _collect_episode(
    env,
    max_steps: int,
    save_video: bool,
    video_dir: str,
    fps: int,
) -> Dict:
    observations = env.reset()
    images: List[np.ndarray] = []
    steps = 0
    done = False

    while not done:
        info = env.get_info(observations)
        if save_video:
            images.append(observations_to_image(observations, info))

        action = env.action_space.sample()
        observations, _, done, info = env.step(action=action)
        steps += 1

        if max_steps > 0 and steps >= max_steps:
            done = True

    # Capture last frame after termination
    if save_video:
        images.append(observations_to_image(observations, info))

    episode_id = env.get_current_episode_id()
    scene = env.habitat_env.current_episode.scene_id.split("/")[-1].split(".")[0]
    sound_id = env.habitat_env.current_episode.info.get("sound_id", "sound")

    if save_video and len(images) > 0:
        video_name = f"{scene}_ep{episode_id}_random"
        images_to_video(images, video_dir, video_name, fps=fps)

    result = {
        "episode_id": episode_id,
        "scene_id": env.habitat_env.current_episode.scene_id,
        "sound_id": sound_id,
        "steps": steps,
    }
    result.update(info)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-config",
        type=str,
        default="configs/omni-long/mp3d/omni-long_semantic_audio.yaml",
    )
    parser.add_argument("--output-dir", type=str, default="data/random_eval/mp3d")
    parser.add_argument("--split", type=str, default="val_telephone")
    parser.add_argument("--max-episodes", type=int, default=-1)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--no-video", action="store_true", default=False)
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    video_dir = os.path.join(args.output_dir, "videos")
    os.makedirs(video_dir, exist_ok=True)

    config = get_config(args.exp_config, args.opts, None, "eval", False)
    config.defrost()
    config.TASK_CONFIG.defrost()

    # Ensure eval uses the current config (not checkpoint config).
    config.EVAL.USE_CKPT_CONFIG = False
    config.EVAL.SPLIT = args.split
    config.TASK_CONFIG.DATASET.SPLIT = args.split

    # Real-time rendering path.
    config.TASK_CONFIG.SIMULATOR.USE_RENDERED_OBSERVATIONS = False

    # Make sure visual sensors are enabled.
    config.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
    config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS = config.SENSORS

    # Single-process, single-env evaluation.
    config.NUM_PROCESSES = 1
    config.USE_VECENV = False
    config.USE_SYNC_VECENV = True

    config.TASK_CONFIG.freeze()
    config.freeze()

    level = logging.DEBUG if config.DEBUG else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s, %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    dataset = make_dataset(config.TASK_CONFIG.DATASET.TYPE, config=config.TASK_CONFIG.DATASET)
    env_class = get_env_class(config.ENV_NAME)
    env = env_class(config=config, dataset=dataset)

    num_episodes = env.habitat_env.number_of_episodes
    if args.max_episodes > 0:
        num_episodes = min(num_episodes, args.max_episodes)

    results: List[Dict] = []
    for _ in range(num_episodes):
        result = _collect_episode(
            env=env,
            max_steps=args.max_steps,
            save_video=not args.no_video,
            video_dir=video_dir,
            fps=args.fps,
        )
        results.append(result)
        logging.info(
            "Episode %s | SPL %.3f | Success %.3f",
            result.get("episode_id"),
            result.get("spl", 0.0),
            result.get("success", 0.0),
        )

    env.close()

    summary = {
        "episodes": len(results),
        "spl_mean": float(np.mean([r.get("spl", 0.0) for r in results])) if results else 0.0,
        "success_mean": float(np.mean([r.get("success", 0.0) for r in results])) if results else 0.0,
        "soft_spl_mean": float(np.mean([r.get("soft_spl", 0.0) for r in results])) if results else 0.0,
    }

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump({"summary": summary, "episodes": results}, f, indent=2)

    logging.info("Done. Results saved to %s", os.path.join(args.output_dir, "results.json"))


if __name__ == "__main__":
    main()
