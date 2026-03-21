#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import json
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
    filter_policy_observations,
)


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


def _episode_object(payload: Dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(**payload)


def _episode_goal_inputs(episode: Any) -> Sequence[Dict[str, Any]]:
    goal_inputs = getattr(episode, "goal_inputs", ())
    return goal_inputs if isinstance(goal_inputs, (list, tuple)) else ()


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

    def close(self) -> None:
        return None


class LifelongPolicyAdapter(PolicyAdapter):
    def __init__(self, policy_name: str, policy_kwargs: Dict[str, Any]):
        self._policy = build_lifelong_eval_policy(policy_name, **policy_kwargs)

    def reset(self, *, episode: Any, observations: Dict[str, Any], goal_payloads: Sequence[Dict[str, Any]]) -> None:
        self._policy.reset(env=None, episode=episode, observations=observations)
        self._policy.start_episode(
            env=None,
            episode=episode,
            observations=observations,
            goal_payloads=goal_payloads,
            order_mode=getattr(episode, "goal_order_mode", None),
        )

    def act(
        self,
        *,
        episode: Any,
        observations: Dict[str, Any],
        goal_payloads: Sequence[Dict[str, Any]],
        step_index: int,
        info: Optional[Dict[str, Any]],
    ) -> Any:
        context = build_lifelong_eval_context(step_index, goal_payloads=goal_payloads, info=info)
        return self._policy.act(env=None, episode=episode, observations=observations, context=context)

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
        return output


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
    policy_name_for_filter = None if args.module_class else str(args.policy).strip().lower()

    context = zmq.Context.instance()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.LINGER, int(args.linger_ms))
    socket.connect(args.connect)

    should_stop = False

    def _handle_signal(signum: int, frame: Any) -> None:
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
        if not isinstance(observations, dict):
            raise RuntimeError("episode_start observations must be a dict")
        observations = filter_policy_observations(policy_name_for_filter, observations)
        goal_inputs = _episode_goal_inputs(episode)

        adapter.reset(episode=episode, observations=observations, goal_payloads=goal_inputs)
        done = False
        reward = None
        info: Optional[Dict[str, Any]] = None
        step_index = 0
        step_reply: Dict[str, Any] = {}
        episode_interrupted = False
        while not done and not should_stop:
            action = adapter.act(
                episode=episode,
                observations=observations,
                goal_payloads=goal_inputs,
                step_index=step_index,
                info=info,
            )
            send_message(socket, {"op": "action", "action": action})
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
            if not isinstance(observations, dict):
                raise RuntimeError("step observations must be a dict")
            observations = filter_policy_observations(policy_name_for_filter, observations)
            reward = step_reply.get("reward")
            done = bool(step_reply.get("done", False))
            info = step_reply.get("info") if isinstance(step_reply.get("info"), dict) else None
            step_index = int(step_reply.get("step_index", step_index + 1))

            if args.verbose:
                debug = {
                    "episode_index": step_reply.get("episode_index"),
                    "step_index": step_index,
                    "done": done,
                    "action": step_reply.get("action"),
                    "episode_metrics": step_reply.get("episode_metrics"),
                }
                print(json.dumps(debug, ensure_ascii=False))

        video_path = step_reply.get("video_path")
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
