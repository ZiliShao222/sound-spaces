#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import json
import os
import signal
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import msgpack
import numpy as np
import torch
import zmq

from ss_baselines.common.omni_long_eval_policy import (
    LifelongEvalPolicy,
    build_lifelong_eval_context,
    build_lifelong_eval_policy,
)
from ss_baselines.common.utils import images_to_video_with_audio, observations_to_image


ARRAY_MARKER = "__ndarray__"


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _normalize_scalar(value.item())
    return value


def _pack_tree(value: Any, frames: List[memoryview]) -> Any:
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.dtype == object:
            raise TypeError("Object dtype arrays are not supported in the fast ZMQ codec")
        contiguous = np.ascontiguousarray(array)
        frame_index = len(frames)
        frames.append(memoryview(contiguous))
        return {
            ARRAY_MARKER: True,
            "dtype": str(contiguous.dtype),
            "shape": list(contiguous.shape),
            "frame": int(frame_index),
        }
    if isinstance(value, dict):
        return {str(key): _pack_tree(item, frames) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pack_tree(item, frames) for item in value]
    return _normalize_scalar(value)


def _unpack_tree(value: Any, frames: Sequence[memoryview]) -> Any:
    if isinstance(value, dict) and value.get(ARRAY_MARKER):
        frame = frames[int(value["frame"])]
        dtype = np.dtype(str(value["dtype"]))
        shape = tuple(int(dim) for dim in value["shape"])
        return np.frombuffer(frame, dtype=dtype).reshape(shape)
    if isinstance(value, dict):
        return {key: _unpack_tree(item, frames) for key, item in value.items()}
    if isinstance(value, list):
        return [_unpack_tree(item, frames) for item in value]
    return value


def recv_message(
    socket: zmq.Socket,
    *,
    should_stop: Optional[Any] = None,
    request_timeout_ms: int = -1,
    poll_interval_ms: int = 200,
) -> Dict[str, Any]:
    deadline = None
    if int(request_timeout_ms) >= 0:
        deadline = time.monotonic() + float(request_timeout_ms) / 1000.0

    while True:
        if should_stop is not None and bool(should_stop()):
            return {"op": "interrupted"}

        timeout_ms = int(poll_interval_ms)
        if deadline is not None:
            remaining_ms = int(max(0.0, (deadline - time.monotonic()) * 1000.0))
            if remaining_ms <= 0:
                raise TimeoutError("Timed out waiting for env_server reply")
            timeout_ms = min(timeout_ms, remaining_ms)

        if socket.poll(timeout_ms, zmq.POLLIN):
            break

    parts = socket.recv_multipart(copy=False)
    header = msgpack.unpackb(parts[0].buffer, raw=False)
    frames = [memoryview(part.buffer) for part in parts[1:]]
    payload = _unpack_tree(header, frames)
    if not isinstance(payload, dict):
        raise RuntimeError("ZMQ reply payload must be a dict")
    return payload


def send_message(socket: zmq.Socket, payload: Dict[str, Any]) -> None:
    frames: List[memoryview] = []
    header = msgpack.packb(_pack_tree(payload, frames), use_bin_type=True)
    socket.send_multipart([header, *frames], copy=False)


def _load_object(spec: str) -> Any:
    token = str(spec).strip()
    if ":" in token:
        module_name, attr_name = token.split(":", 1)
    else:
        module_name, attr_name = token.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


def _obs_to_torch(value: Any, device: torch.device) -> Any:
    if isinstance(value, np.ndarray):
        return torch.from_numpy(np.asarray(value)).to(device=device, non_blocking=True)
    if isinstance(value, dict):
        return {key: _obs_to_torch(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_obs_to_torch(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_obs_to_torch(item, device) for item in value)
    return value


def _normalize_action(value: Any) -> Any:
    if isinstance(value, (dict, str, int, np.integer)):
        return value
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return int(value.reshape(-1)[0].item())
        return int(np.asarray(value).reshape(-1).argmax())
    if torch.is_tensor(value):
        tensor = value.detach()
        if tensor.numel() == 1:
            return int(tensor.item())
        return int(torch.argmax(tensor.reshape(-1)).item())
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _normalize_action(value[0])
    raise RuntimeError(f"Unsupported action output type: {type(value)!r}")


def _episode_object(payload: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**payload)


def _resolve_video_dir(args: argparse.Namespace, server_run_base_dir: Optional[str]) -> str:
    if args.video_dir is not None and str(args.video_dir).strip():
        return str(args.video_dir)
    if server_run_base_dir is not None and str(server_run_base_dir).strip():
        return os.path.join(str(server_run_base_dir), "videos")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join("results", "policy_videos", timestamp)


def _maybe_save_episode_video(
    *,
    args: argparse.Namespace,
    episode: Any,
    episode_index: int,
    frames: Sequence[np.ndarray],
    audios: Sequence[np.ndarray],
    audio_sample_rate: int,
    scene_name: str,
    server_run_base_dir: Optional[str],
    step_time: float,
) -> Optional[str]:
    if not bool(args.video) or len(frames) == 0:
        return None

    video_root = _resolve_video_dir(args, server_run_base_dir)
    scene_video_dir = os.path.join(video_root, str(scene_name))
    os.makedirs(scene_video_dir, exist_ok=True)
    fps = args.video_fps
    if fps is None:
        fps = int(round(1.0 / float(step_time))) if float(step_time) > 1e-6 else 4
        fps = max(1, int(fps))

    episode_id = str(getattr(episode, "episode_id", ""))
    video_name = f"episode_{int(episode_index)}_id_{episode_id}"
    if len(audios) == 0:
        raise RuntimeError("video enabled but no audio frames were collected")
    muxed_audios = list(audios)
    raw_peak = 0.0
    for clip in audios:
        if np.asarray(clip).size > 0:
            raw_peak = max(raw_peak, float(np.max(np.abs(np.asarray(clip, dtype=np.float32)))))
    if bool(args.video_audio_normalize) and raw_peak > 0.0:
        gain = min(float(args.video_audio_max_gain), 0.8 / raw_peak)
        if abs(gain - 1.0) > 1e-6:
            muxed_audios = [
                np.clip(np.asarray(clip, dtype=np.float32) * gain, -1.0, 1.0)
                for clip in audios
            ]
    images_to_video_with_audio(
        list(frames),
        scene_video_dir,
        video_name,
        list(muxed_audios),
        int(audio_sample_rate),
        fps=int(fps),
    )
    return os.path.join(scene_video_dir, f"{video_name}.mp4")


class PolicyAdapter:
    def reset(self, *, episode: Any, observations: Dict[str, Any], goal_payloads: Sequence[Dict[str, Any]]) -> None:
        return None

    def act(
        self,
        *,
        episode: Any,
        observations: Dict[str, Any],
        goal_payloads: Sequence[Dict[str, Any]],
        step_index: int,
        info: Optional[Dict[str, Any]],
    ) -> Any:
        raise NotImplementedError

    def observe(
        self,
        *,
        episode: Any,
        observations: Dict[str, Any],
        reward: Optional[float],
        done: bool,
        info: Optional[Dict[str, Any]],
    ) -> None:
        return None

    def close(self) -> None:
        return None


class LifelongPolicyAdapter(PolicyAdapter):
    def __init__(self, policy_name: str, policy_kwargs: Dict[str, Any]):
        self._policy = build_lifelong_eval_policy(policy_name, **policy_kwargs)

    def reset(self, *, episode: Any, observations: Dict[str, Any], goal_payloads: Sequence[Dict[str, Any]]) -> None:
        del goal_payloads
        self._policy.reset(env=None, episode=episode, observations=observations)

    def act(
        self,
        *,
        episode: Any,
        observations: Dict[str, Any],
        goal_payloads: Sequence[Dict[str, Any]],
        step_index: int,
        info: Optional[Dict[str, Any]],
    ) -> Any:
        del info
        context = build_lifelong_eval_context(step_index, goal_payloads=goal_payloads)
        return self._policy.act(env=None, episode=episode, observations=observations, context=context)

    def observe(
        self,
        *,
        episode: Any,
        observations: Dict[str, Any],
        reward: Optional[float],
        done: bool,
        info: Optional[Dict[str, Any]],
    ) -> None:
        self._policy.observe(
            env=None,
            episode=episode,
            observations=observations,
            reward=reward,
            done=done,
            info=info,
        )

    def close(self) -> None:
        self._policy.close()


class TorchModuleAdapter(PolicyAdapter):
    def __init__(
        self,
        module_class: str,
        checkpoint: Optional[str],
        module_kwargs: Dict[str, Any],
        device: torch.device,
    ):
        module_cls = _load_object(module_class)
        self._device = device
        self._model = module_cls(**module_kwargs)
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            self._model.load_state_dict(state)
        self._model.to(device=self._device)
        self._model.eval()

    def reset(self, *, episode: Any, observations: Dict[str, Any], goal_payloads: Sequence[Dict[str, Any]]) -> None:
        if hasattr(self._model, "reset"):
            self._model.reset(episode=episode, observations=observations, goal_payloads=goal_payloads)

    def act(
        self,
        *,
        episode: Any,
        observations: Dict[str, Any],
        goal_payloads: Sequence[Dict[str, Any]],
        step_index: int,
        info: Optional[Dict[str, Any]],
    ) -> Any:
        obs_torch = _obs_to_torch(observations, self._device)
        with torch.inference_mode():
            if hasattr(self._model, "act"):
                output = self._model.act(
                    observations=obs_torch,
                    raw_observations=observations,
                    episode=episode,
                    goal_payloads=goal_payloads,
                    step_index=int(step_index),
                    info=info,
                )
            else:
                output = self._model(obs_torch)
        return _normalize_action(output)

    def observe(
        self,
        *,
        episode: Any,
        observations: Dict[str, Any],
        reward: Optional[float],
        done: bool,
        info: Optional[Dict[str, Any]],
    ) -> None:
        if hasattr(self._model, "observe"):
            self._model.observe(
                episode=episode,
                observations=observations,
                reward=reward,
                done=done,
                info=info,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run policy in a separate process and talk to Habitat env via ZMQ.")
    parser.add_argument("--connect", type=str, default="tcp://127.0.0.1:5555")
    parser.add_argument("--policy", type=str, default="omega_nav")
    parser.add_argument("--policy-kwargs", type=str, default="{}")
    parser.add_argument("--module-class", type=str, default=None)
    parser.add_argument("--module-kwargs", type=str, default="{}")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--goal-order-mode", type=str, default=None, choices=["ordered", "unordered"])
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--request-timeout-ms", type=int, default=-1)
    parser.add_argument("--linger-ms", type=int, default=0)
    parser.add_argument("--video", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--video-dir", type=str, default=None)
    parser.add_argument("--video-fps", type=int, default=None)
    parser.add_argument("--video-audio-normalize", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--video-audio-max-gain", type=float, default=200.0)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def build_adapter(args: argparse.Namespace) -> PolicyAdapter:
    device = torch.device(args.device)
    if args.module_class:
        module_kwargs = json.loads(args.module_kwargs)
        if not isinstance(module_kwargs, dict):
            raise RuntimeError("--module-kwargs must be a JSON object")
        return TorchModuleAdapter(args.module_class, args.checkpoint, module_kwargs, device)

    policy_kwargs = json.loads(args.policy_kwargs)
    if not isinstance(policy_kwargs, dict):
        raise RuntimeError("--policy-kwargs must be a JSON object")
    policy_name = str(args.policy).strip().lower()
    if policy_name in {"omega_nav", "omega_oracle", "omega_nav_policy", "omega_nav_oracle"}:
        policy_kwargs.setdefault("atomic_actions_only", True)
        policy_kwargs.setdefault("use_local_navigator", False)
    if args.goal_order_mode is not None:
        policy_kwargs.setdefault("goal_order_mode", args.goal_order_mode)
    return LifelongPolicyAdapter(args.policy, policy_kwargs)


def main() -> None:
    args = parse_args()
    adapter = build_adapter(args)

    context = zmq.Context.instance()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, int(args.linger_ms))
    socket.connect(args.connect)

    should_stop = False

    def _handle_signal(signum: int, frame: Any) -> None:
        del signum, frame
        nonlocal should_stop
        should_stop = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    episodes_done = 0
    while not should_stop:
        send_message(socket, {"op": "reset"})
        reply = recv_message(
            socket,
            should_stop=lambda: should_stop,
            request_timeout_ms=int(args.request_timeout_ms),
        )
        op = str(reply.get("op", "")).strip().lower()
        if op == "interrupted":
            break
        if op == "run_end":
            print(json.dumps(reply.get("summary", {}), ensure_ascii=False, indent=2))
            send_message(socket, {"op": "close"})
            _ = recv_message(
                socket,
                should_stop=lambda: False,
                request_timeout_ms=int(args.request_timeout_ms),
            )
            break
        if op == "error":
            raise RuntimeError(reply.get("message", "Unknown server error"))
        if op != "episode_start":
            raise RuntimeError(f"Unexpected server op: {op}")

        episode = _episode_object(reply.get("episode", {}))
        observations = reply.get("observations")
        goal_payloads = reply.get("goal_payloads", [])
        server_run_base_dir = str(reply.get("run_base_dir", "")).strip() or None
        step_time = float(reply.get("step_time", 0.25))
        audio_sample_rate = int(reply.get("audio_sample_rate", 16000))
        if not isinstance(observations, dict):
            raise RuntimeError("episode_start observations must be a dict")

        adapter.reset(episode=episode, observations=observations, goal_payloads=goal_payloads)
        done = False
        reward = None
        info: Optional[Dict[str, Any]] = None
        step_index = 0
        step_reply: Dict[str, Any] = {}
        frames: List[np.ndarray] = []
        audios: List[np.ndarray] = []
        episode_interrupted = False
        while not done and not should_stop:
            action = adapter.act(
                episode=episode,
                observations=observations,
                goal_payloads=goal_payloads,
                step_index=step_index,
                info=info,
            )
            send_message(socket, {"op": "action", "action": _normalize_action(action)})
            step_reply = recv_message(
                socket,
                should_stop=lambda: should_stop,
                request_timeout_ms=int(args.request_timeout_ms),
            )
            step_op = str(step_reply.get("op", "")).strip().lower()
            if step_op == "interrupted":
                should_stop = True
                episode_interrupted = True
                break
            if step_op == "error":
                raise RuntimeError(step_reply.get("message", "Unknown server error"))
            if step_op != "step":
                raise RuntimeError(f"Unexpected server op during stepping: {step_op}")

            observations = step_reply.get("observations")
            reward = step_reply.get("reward")
            done = bool(step_reply.get("done", False))
            info = step_reply.get("info") if isinstance(step_reply.get("info"), dict) else None
            step_index = int(step_reply.get("step_index", step_index + 1))

            if bool(args.video):
                frame_info = dict(info or {})
                metrics = step_reply.get("metrics")
                if isinstance(metrics, dict):
                    frame_info.update(metrics)
                frames.append(observations_to_image(observations, frame_info))
                if "audiogoal" not in observations:
                    raise RuntimeError(
                        "video enabled but observation missing 'audiogoal'"
                    )
                audios.append(np.asarray(observations["audiogoal"], dtype=np.float32))

            adapter.observe(
                episode=episode,
                observations=observations,
                reward=reward,
                done=done,
                info=info,
            )

            if args.verbose:
                debug = {
                    "episode_index": step_reply.get("episode_index"),
                    "step_index": step_index,
                    "done": done,
                    "action": step_reply.get("action"),
                    "episode_metrics": step_reply.get("episode_metrics"),
                }
                print(json.dumps(debug, ensure_ascii=False))

        video_path = _maybe_save_episode_video(
            args=args,
            episode=episode,
            episode_index=int(reply.get("episode_index", episodes_done)),
            frames=frames,
            audios=audios,
            audio_sample_rate=audio_sample_rate,
            scene_name=str(reply.get("scene", getattr(episode, "scene_name", "scene"))),
            server_run_base_dir=server_run_base_dir,
            step_time=step_time,
        )
        if video_path is not None:
            print(f"[policy_client] saved video: {video_path}")

        if episode_interrupted:
            break

        episodes_done += 1
        if isinstance(step_reply.get("episode_metrics"), dict):
            print(json.dumps(step_reply["episode_metrics"], ensure_ascii=False, indent=2))
        if args.max_episodes is not None and episodes_done >= int(args.max_episodes):
            send_message(socket, {"op": "close"})
            _ = recv_message(
                socket,
                should_stop=lambda: False,
                request_timeout_ms=int(args.request_timeout_ms),
            )
            break

    adapter.close()
    socket.close(0)
    context.term()


if __name__ == "__main__":
    main()
