#!/usr/bin/env python3

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import signal
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import habitat
from habitat.config import Config
import msgpack
import numpy as np
import soundspaces  # noqa: F401
import zmq

from soundspaces.tasks.omni_long_eval_utils import (
    DEFAULT_CONFIG,
    DEFAULT_EXP_NAME,
    DEFAULT_OUTPUT_PARENT_DIR,
    DEFAULT_TASK_TYPE,
    _all_scene_names,
    _build_goal_input_payload,
    _episode_goal_count,
    _flatten_instances,
    _format_episode_metrics,
    _load_dataset_payload,
    _normalize_task_specs,
    _prepare_episode_list,
    _resolve_scene_subset,
    _scene_key,
    _scene_subset_label,
    apply_eval_config,
    build_config,
)


ARRAY_MARKER = "__ndarray__"
PACKABLE_SCALAR_TYPES = (type(None), bool, int, float, str, bytes)
DEFAULT_ACTION_MAP = {
    0: "STOP",
    1: "MOVE_FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
}


def _quaternion_payload(value: Any) -> Optional[List[float]]:
    if all(hasattr(value, key) for key in ("x", "y", "z", "w")):
        return [float(value.x), float(value.y), float(value.z), float(value.w)]
    components = getattr(value, "components", None)
    if components is not None:
        array = np.asarray(components, dtype=np.float32).reshape(-1)
        if array.size >= 4:
            return [float(array[1]), float(array[2]), float(array[3]), float(array[0])]
    if hasattr(value, "imag") and hasattr(value, "real"):
        imag = np.asarray(value.imag, dtype=np.float32).reshape(-1)
        if imag.size >= 3:
            return [
                float(imag[0]),
                float(imag[1]),
                float(imag[2]),
                float(value.real),
            ]
    return None


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, PACKABLE_SCALAR_TYPES):
        return value
    if isinstance(value, Path):
        return str(value)
    quaternion_payload = _quaternion_payload(value)
    if quaternion_payload is not None:
        return quaternion_payload
    type_name = f"{type(value).__module__}.{type(value).__name__}".lower()
    if "quaternion" in type_name:
        return str(value)
    if isinstance(value, np.generic):
        item = value.item()
        if item is value:
            return str(value)
        return _normalize_scalar(item)
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _pack_tree(value: Any, frames: List[memoryview], path: str = "root") -> Any:
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.dtype == object:
            raise TypeError(f"Object dtype arrays are not supported in the fast ZMQ codec: {path}")
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
        return {
            str(key): _pack_tree(item, frames, f"{path}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_pack_tree(item, frames, f"{path}[{index}]") for index, item in enumerate(value)]
    normalized = _normalize_scalar(value)
    if normalized is value:
        if isinstance(value, PACKABLE_SCALAR_TYPES):
            return value
        raise RuntimeError(
            f"Unsupported wire payload at {path}: {type(value).__module__}.{type(value).__name__}"
        )
    return _pack_tree(normalized, frames, path)


def _unpack_tree(value: Any, frames: Sequence[memoryview]) -> Any:
    if isinstance(value, dict) and value.get(ARRAY_MARKER):
        frame = frames[int(value["frame"])]
        dtype = np.dtype(str(value["dtype"]))
        shape = tuple(int(dim) for dim in value["shape"])
        array = np.frombuffer(frame, dtype=dtype)
        return array.reshape(shape)
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
                return {"op": "interrupted"}
            timeout_ms = min(timeout_ms, remaining_ms)

        if socket.poll(timeout_ms, zmq.POLLIN):
            break

    parts = socket.recv_multipart(copy=False)
    header = msgpack.unpackb(parts[0].buffer, raw=False)
    frames = [memoryview(part.buffer) for part in parts[1:]]
    payload = _unpack_tree(header, frames)
    if not isinstance(payload, dict):
        raise RuntimeError("ZMQ request payload must be a dict")
    return payload


def send_message(socket: zmq.Socket, payload: Dict[str, Any]) -> None:
    frames: List[memoryview] = []
    packed = _pack_tree(payload, frames)
    header = msgpack.packb(packed, use_bin_type=True)
    socket.send_multipart([header, *frames], copy=False)


def _episode_payload(episode: Any) -> Dict[str, Any]:
    return {
        "episode_id": str(getattr(episode, "episode_id", "")),
        "scene_id": str(getattr(episode, "scene_id", "")),
        "scene_name": _scene_key(str(getattr(episode, "scene_id", ""))),
        "tasks": list(getattr(episode, "tasks", []) or []),
        "goal_count": len(list(getattr(episode, "goals", []) or [])),
        "start_position": getattr(episode, "start_position", None),
        "start_rotation": getattr(episode, "start_rotation", None),
    }


def _mean_metrics(records: Sequence[Dict[str, float]]) -> Dict[str, float]:
    if not records:
        return {}
    keys = sorted({key for record in records for key in record.keys()})
    result: Dict[str, float] = {}
    for key in keys:
        values = [float(record[key]) for record in records if key in record]
        if values:
            result[key] = float(np.mean(values))
    return result


def _scene_mean_summary(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, float]]] = {}
    for record in records:
        scene_name = str(record.get("scene", ""))
        metrics = record.get("metrics")
        if not scene_name or not isinstance(metrics, dict):
            continue
        grouped.setdefault(scene_name, []).append(metrics)

    summaries: List[Dict[str, Any]] = []
    for scene_name in sorted(grouped.keys()):
        metrics_list = grouped[scene_name]
        summaries.append(
            {
                "scene": scene_name,
                "episodes": len(metrics_list),
                "mean_metrics": _mean_metrics(metrics_list),
            }
        )
    return summaries


def _build_action_map(args: argparse.Namespace) -> Dict[int, str]:
    mapping = dict(DEFAULT_ACTION_MAP)
    mapping[4] = str(args.submit_action_name)
    if args.action_map is None:
        return mapping
    payload = json.loads(args.action_map)
    if not isinstance(payload, dict):
        raise RuntimeError("--action-map must be a JSON object")
    for key, value in payload.items():
        mapping[int(key)] = str(value)
    return mapping


def _normalize_action(action: Any, action_map: Dict[int, str]) -> Any:
    if isinstance(action, dict):
        return action
    if isinstance(action, str):
        return {"action": str(action)}
    if isinstance(action, np.ndarray):
        if action.size == 1:
            return _normalize_action(action.reshape(-1)[0].item(), action_map)
        raise RuntimeError("Array action must be scalar")
    if isinstance(action, (int, np.integer)):
        action_name = action_map.get(int(action))
        if action_name is not None:
            return {"action": action_name}
        return int(action)
    return action


def _force_pose_sensor(cfg: habitat.Config) -> habitat.Config:
    cfg.defrost()
    sensors = [str(sensor) for sensor in list(cfg.TASK.SENSORS)]
    if "POSE_SENSOR" not in sensors:
        sensors.append("POSE_SENSOR")
    cfg.TASK.SENSORS = sensors
    if not hasattr(cfg.TASK, "POSE_SENSOR"):
        cfg.TASK.POSE_SENSOR = Config()
    cfg.TASK.POSE_SENSOR.TYPE = "PoseSensor"
    cfg.freeze()
    return cfg


def _force_video_measurements(cfg: habitat.Config) -> habitat.Config:
    cfg.defrost()
    measurements = [str(name) for name in list(cfg.TASK.MEASUREMENTS)]
    if "TOP_DOWN_MAP" not in measurements:
        measurements.append("TOP_DOWN_MAP")
    if "COLLISIONS" not in measurements:
        measurements.append("COLLISIONS")
    cfg.TASK.MEASUREMENTS = measurements
    cfg.freeze()
    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OmniLong Habitat env as a local ZMQ server.")
    parser.add_argument("--eval-config", type=str, default=None)
    parser.add_argument("--exp-config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--scenes-dir", type=str, default=None)
    parser.add_argument("--scene-dataset-config", type=str, default=None)
    parser.add_argument("--disable-content-scenes", default=None, action=argparse.BooleanOptionalAction)
    parser.add_argument("--task-type", type=str, default=DEFAULT_TASK_TYPE)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument("--scene-start-index", type=int, default=None)
    parser.add_argument("--scene-end-index", type=int, default=None)
    parser.add_argument("--num-episodes", type=int, default=None)
    parser.add_argument("--goal-order-mode", type=str, default=None, choices=["ordered", "unordered"])
    parser.add_argument("--submit-action-name", type=str, default="LIFELONG_SUBMIT")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episode-log", type=str, default=None)
    parser.add_argument("--mean-log", type=str, default=None)
    parser.add_argument("--bind", type=str, default="tcp://*:5555")
    parser.add_argument("--action-map", type=str, default=None)
    parser.add_argument("--linger-ms", type=int, default=0)
    parser.add_argument("--rcvtimeo-ms", type=int, default=-1)
    args = parser.parse_args()

    injected_defaults = {
        "dataset_path": None,
        "policy": "distance_submit",
        "policy_kwargs": "{}",
        "distance_submit_threshold": 1.0,
        "print_every": 1,
        "video": True,
        "video_dir": None,
        "video_fps": None,
        "video_audio": True,
        "audio_active_threshold": 1e-6,
        "video_audio_normalize": True,
        "video_audio_max_gain": 200.0,
        "prompt_image_dir": None,
        "save_action_observations": False,
        "action_observation_dir": None,
        "exp_name": DEFAULT_EXP_NAME,
        "output_parent_dir": DEFAULT_OUTPUT_PARENT_DIR,
    }
    for name, value in injected_defaults.items():
        setattr(args, name, getattr(args, name, value))

    args = apply_eval_config(args)
    args.video = False
    args.video_audio = False
    args.save_action_observations = False

    if args.dataset_path is None or not str(args.dataset_path).strip():
        raise RuntimeError("dataset_path is required and must come from --eval-config")
    return args


class OmniLongEnvServer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.cfg = _force_video_measurements(_force_pose_sensor(build_config(args)))
        self.dataset_payload = _load_dataset_payload(args.dataset_path)
        self.instance_index = _flatten_instances(self.dataset_payload.get("instance_index", {}))
        self.action_map = _build_action_map(args)

        self.env = habitat.Env(config=self.cfg)
        all_scene_names = _all_scene_names(list(self.env.episodes))
        self.scene_range_label = _scene_subset_label(all_scene_names, args)
        self.scene_range_start, self.scene_range_end, self.selected_scene_names = _resolve_scene_subset(
            all_scene_names,
            args,
        )
        self.episodes = _prepare_episode_list(self.env, args)
        self.current_episode: Optional[Any] = None
        self.current_episode_index = -1
        self.current_observations: Optional[Dict[str, Any]] = None
        self.current_goal_payloads: List[Dict[str, Any]] = []
        self.current_step_index = 0
        self.current_done = True
        self.completed_records: List[Dict[str, Any]] = []
        self.run_base_dir = self._ensure_output_paths()
        self.scene_mean_log = os.path.join(self.run_base_dir, "scene_mean.json")

    def close(self) -> None:
        self.env.close()

    def _ensure_output_paths(self) -> str:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        if self.args.episode_log is None or self.args.mean_log is None:
            run_base_dir = os.path.join(
                str(self.args.output_parent_dir),
                str(self.args.exp_name),
                str(self.scene_range_label),
                timestamp,
            )
            os.makedirs(run_base_dir, exist_ok=True)
            if self.args.episode_log is None:
                self.args.episode_log = os.path.join(run_base_dir, "episode.jsonl")
            if self.args.mean_log is None:
                self.args.mean_log = os.path.join(run_base_dir, "mean.json")

        run_base_dir = str(Path(self.args.episode_log).resolve().parent)
        os.makedirs(run_base_dir, exist_ok=True)
        Path(self.args.episode_log).write_text("", encoding="utf-8")
        return run_base_dir

    def _summary(self) -> Dict[str, Any]:
        metrics = [record["metrics"] for record in self.completed_records if isinstance(record.get("metrics"), dict)]
        return {
            "episodes_completed": len(self.completed_records),
            "task_type": str(self.cfg.TASK.TYPE),
            "dataset_path": str(self.cfg.DATASET.DATA_PATH),
            "scene_range_label": str(self.scene_range_label),
            "scene_range_start": int(self.scene_range_start),
            "scene_range_end": int(self.scene_range_end),
            "selected_scenes": list(self.selected_scene_names),
            "mean_metrics": _mean_metrics(metrics),
            "scene_mean_log": str(self.scene_mean_log),
        }

    def _write_episode_record(self, record: Dict[str, Any]) -> None:
        if self.args.episode_log is None:
            return
        path = Path(self.args.episode_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def write_summary(self) -> None:
        scene_mean_path = Path(self.scene_mean_log)
        scene_mean_path.parent.mkdir(parents=True, exist_ok=True)
        scene_mean_payload = {
            "task_type": str(self.cfg.TASK.TYPE),
            "dataset_path": str(self.cfg.DATASET.DATA_PATH),
            "scene_range_label": str(self.scene_range_label),
            "scene_range_start": int(self.scene_range_start),
            "scene_range_end": int(self.scene_range_end),
            "selected_scenes": list(self.selected_scene_names),
            "scenes": _scene_mean_summary(self.completed_records),
        }
        scene_mean_path.write_text(
            json.dumps(scene_mean_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        if self.args.mean_log is None:
            return
        path = Path(self.args.mean_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self._summary(), ensure_ascii=False, indent=2), encoding="utf-8")

    def _episode_start_message(self) -> Dict[str, Any]:
        self.current_episode_index += 1
        if self.current_episode_index >= len(self.episodes):
            return {
                "op": "run_end",
                "episodes_total": len(self.episodes),
                "summary": self._summary(),
            }

        episode = self.episodes[self.current_episode_index]
        self.env.current_episode = episode
        observations = self.env.reset()
        goal_payloads: List[Dict[str, Any]] = []
        for instance_key, modality in _normalize_task_specs(getattr(episode, "tasks", None)):
            goal_payloads.append(
                _build_goal_input_payload(
                    env=self.env,
                    instance_key=instance_key,
                    modality=modality,
                    instance_record=self.instance_index.get(instance_key),
                )
            )

        self.current_episode = episode
        self.current_observations = observations
        self.current_goal_payloads = goal_payloads
        self.current_step_index = 0
        self.current_done = False
        return {
            "op": "episode_start",
            "episode_index": self.current_episode_index,
            "episode": _episode_payload(episode),
            "observations": observations,
            "goal_payloads": goal_payloads,
            "submit_action_name": str(self.args.submit_action_name),
            "scene": _scene_key(str(getattr(episode, "scene_id", ""))),
            "step_index": 0,
            "run_base_dir": str(self.run_base_dir),
            "step_time": float(getattr(self.cfg.SIMULATOR, "STEP_TIME", 0.25)),
            "audio_sample_rate": int(getattr(getattr(self.cfg.SIMULATOR, "AUDIO", None), "RIR_SAMPLING_RATE", 16000)),
        }

    def _step_message(self, action: Any) -> Dict[str, Any]:
        if self.current_episode is None or self.current_observations is None:
            raise RuntimeError("No active episode. Send op=reset first.")
        if self.current_done:
            raise RuntimeError("Current episode already finished. Send op=reset for next episode.")

        normalized_action = _normalize_action(action, self.action_map)
        step_result = self.env.step(normalized_action)
        reward = None
        info: Dict[str, Any] = {}
        if isinstance(step_result, tuple) and len(step_result) == 4:
            observations, reward, done, base_info = step_result
            if isinstance(base_info, dict):
                info.update(base_info)
        else:
            observations = step_result
            done = bool(self.env.episode_over)

        task = getattr(self.env, "_task", None) or getattr(self.env, "task", None)
        feedback = {}
        if task is not None and hasattr(task, "get_last_action_feedback"):
            feedback = task.get_last_action_feedback() or {}
        metrics = self.env.get_metrics() if hasattr(self.env, "get_metrics") else {}

        self.current_observations = observations
        self.current_step_index += 1
        self.current_done = bool(done)

        info.update(
            {
                "feedback": feedback,
                "step_index": int(self.current_step_index),
                "scene": _scene_key(str(getattr(self.current_episode, "scene_id", ""))),
                "episode_id": str(getattr(self.current_episode, "episode_id", "")),
            }
        )
        message: Dict[str, Any] = {
            "op": "step",
            "episode_index": self.current_episode_index,
            "step_index": int(self.current_step_index),
            "action": normalized_action,
            "observations": observations,
            "reward": reward,
            "done": bool(done),
            "info": info,
            "metrics": metrics,
        }

        if self.current_done:
            goal_specs = _normalize_task_specs(getattr(self.current_episode, "tasks", None))
            goal_count = _episode_goal_count(self.current_episode, goal_specs)
            episode_metrics = _format_episode_metrics(metrics, goal_count)
            record = {
                "episode_index": int(self.current_episode_index),
                "episode_id": str(getattr(self.current_episode, "episode_id", "")),
                "scene": _scene_key(str(getattr(self.current_episode, "scene_id", ""))),
                "metrics": episode_metrics,
            }
            self.completed_records.append(record)
            self._write_episode_record(record)
            message["episode"] = _episode_payload(self.current_episode)
            message["episode_metrics"] = episode_metrics
            message["summary"] = self._summary()

        return message

    def handle(self, request: Dict[str, Any]) -> Dict[str, Any]:
        op = str(request.get("op", "")).strip().lower()
        if op in {"reset", "start"}:
            return self._episode_start_message()
        if op == "action":
            return self._step_message(request.get("action"))
        if op == "close":
            return {"op": "closed", "summary": self._summary()}
        if op == "ping":
            return {"op": "pong", "episodes_total": len(self.episodes)}
        raise RuntimeError(f"Unknown request op: {op}")


def main() -> None:
    args = parse_args()
    server = OmniLongEnvServer(args)

    context = zmq.Context.instance()
    socket = context.socket(zmq.REP)
    socket.setsockopt(zmq.LINGER, int(args.linger_ms))
    if int(args.rcvtimeo_ms) >= 0:
        socket.setsockopt(zmq.RCVTIMEO, int(args.rcvtimeo_ms))
    socket.bind(args.bind)

    should_stop = False

    def _handle_signal(signum: int, frame: Any) -> None:
        del signum, frame
        nonlocal should_stop
        should_stop = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    print(f"[env_server] bound to {args.bind} | episodes={len(server.episodes)}")
    print(f"[env_server] episode_log={server.args.episode_log}")
    print(f"[env_server] mean_log={server.args.mean_log}")
    print(f"[env_server] scene_mean_log={server.scene_mean_log}")
    while not should_stop:
        request = recv_message(
            socket,
            should_stop=lambda: should_stop,
            request_timeout_ms=int(args.rcvtimeo_ms),
        )
        if str(request.get("op", "")).strip().lower() == "interrupted":
            break
        reply = server.handle(request)
        send_message(socket, reply)
        if str(reply.get("op", "")).lower() == "closed":
            break

    server.write_summary()
    server.close()
    socket.close(0)
    context.term()


if __name__ == "__main__":
    main()
