#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json

import habitat
import soundspaces  # noqa: F401

from soundspaces.tasks.omni_long_eval_utils import (
    _build_goal_input_payload,
    _load_dataset_payload,
    _normalize_task_specs,
    _prepare_episode_list,
    apply_eval_config,
    build_config,
)
from ss_baselines.common.omni_long_eval_policy import build_lifelong_eval_context, filter_policy_observations
from ss_baselines.omega_nav.policy import OmegaNavPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OmegaNav perception/planning demo in Habitat.")
    parser.add_argument("--eval-config", type=str, default=None)
    parser.add_argument("--exp-config", type=str, default="configs/omni-long/mp3d/omni-long_semantic_audio.yaml")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--scenes-dir", type=str, default=None)
    parser.add_argument("--scene-dataset-config", type=str, default=None)
    parser.add_argument("--disable-content-scenes", action="store_true")
    parser.add_argument("--task-type", type=str, default="OmniLongSemanticAudioNav")
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--scene-start-index", type=int, default=None)
    parser.add_argument("--scene-end-index", type=int, default=None)
    parser.add_argument("--num-episodes", type=int, default=1)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--goal-order-mode", type=str, default=None)
    parser.add_argument("--submit-action-name", type=str, default="LIFELONG_SUBMIT")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-audio", action="store_true")
    parser.add_argument("--policy-config", type=str, default="ss_baselines/omega_nav/configs/perception.yaml")
    parser.add_argument("--policy-kwargs", type=str, default="{}")
    return apply_eval_config(parser.parse_args())


def main() -> None:
    args = parse_args()
    if args.dataset_path is None or not str(args.dataset_path).strip():
        raise RuntimeError("dataset_path is required. Provide it in --eval-config.")
    cfg = build_config(args)
    dataset_payload = _load_dataset_payload(args.dataset_path)
    instance_index = dataset_payload.get("instance_index", {}) if isinstance(dataset_payload, dict) else {}

    with habitat.Env(config=cfg) as env:
        episodes = _prepare_episode_list(env, args)
        if not episodes:
            return
        env.current_episode = episodes[0]
        observations = env.reset()
        observations = filter_policy_observations("omega_nav", observations)
        episode = env.current_episode

        goal_payloads = []
        for instance_key, modality in _normalize_task_specs(getattr(episode, "tasks", None)):
            goal_payloads.append(
                _build_goal_input_payload(
                    env=env,
                    instance_key=instance_key,
                    modality=modality,
                    instance_record=instance_index.get(instance_key),
                )
            )
        
        policy = OmegaNavPolicy(config_path=args.policy_config)
        policy.reset(env=env, episode=episode, observations=observations)
        policy.start_episode(
            env=env,
            episode=episode,
            observations=observations,
            goal_payloads=goal_payloads,
            order_mode=getattr(episode, "goal_order_mode", None),
        )
        done = False
        step_index = 0
        info = None
        while not done and step_index < int(args.steps):
            context = build_lifelong_eval_context(step_index, goal_payloads=goal_payloads, info=info)
            action = policy.act(env=env, episode=episode, observations=observations, context=context)
            debug_state = policy.get_debug_state()
            print(json.dumps({"step": step_index, "action": action, "debug": debug_state}, ensure_ascii=False, indent=2))
            observations = env.step({"action": {0: "STOP", 1: "MOVE_FORWARD", 2: "TURN_LEFT", 3: "TURN_RIGHT", 4: "LIFELONG_SUBMIT"}[int(action)]})
            observations = filter_policy_observations("omega_nav", observations)
            done = bool(env.episode_over)
            info = env.task.get_last_action_feedback() if hasattr(env.task, "get_last_action_feedback") else None
            step_index += 1


if __name__ == "__main__":
    main()
