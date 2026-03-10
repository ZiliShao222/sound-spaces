#!/usr/bin/env python3

"""Render one RGB view from a saved pose-only ImageNav view entry."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _bootstrap_native_dependencies() -> None:
    """Load native deps before importing habitat-sim."""
    try:
        import quaternion  # noqa: F401
    except Exception:
        pass

    if os.environ.get("SOUNDSPACES_FORCE_SYSTEM_LIBZ", "0") != "1":
        return

    candidates = (
        "/lib/x86_64-linux-gnu/libz.so.1",
        "/usr/lib/x86_64-linux-gnu/libz.so.1",
    )
    for candidate in candidates:
        if not os.path.isfile(candidate):
            continue
        try:
            ctypes.CDLL(candidate, mode=ctypes.RTLD_GLOBAL)
            return
        except OSError:
            continue

    libz_path = ctypes.util.find_library("z")
    if libz_path is None:
        return
    try:
        ctypes.CDLL(libz_path, mode=ctypes.RTLD_GLOBAL)
    except OSError:
        pass


_bootstrap_native_dependencies()

import numpy as np
from PIL import Image

import soundspaces  # noqa: F401
from habitat.sims import make_sim
from habitat_sim.utils.common import quat_from_coeffs
from ss_baselines.av_nav.config.default import get_task_config


def _resolve_scene_path(scene_dir: Path, scene_name: str) -> Path:
    candidate = Path(scene_name)
    if candidate.suffix == ".glb":
        if candidate.is_absolute() and candidate.is_file():
            return candidate.resolve()
        local_candidate = (scene_dir / candidate).resolve()
        if local_candidate.is_file():
            return local_candidate

    direct = (scene_dir / scene_name / f"{scene_name}.glb").resolve()
    if direct.is_file():
        return direct

    recursive = sorted(scene_dir.glob(f"**/{scene_name}.glb"))
    if recursive:
        return recursive[0].resolve()

    raise RuntimeError(f"Failed to resolve scene '{scene_name}' under {scene_dir}")


def _build_simulator(
    scene_path: Path,
    scene_dataset_config: Path,
    exp_config: Path,
    width: int,
    height: int,
    hfov: float,
    sensor_height: float,
):
    cfg = get_task_config(config_paths=[str(exp_config)])
    cfg.defrost()
    cfg.SIMULATOR.TYPE = "ContinuousSoundSpacesSim"
    cfg.SIMULATOR.SCENE = str(scene_path)
    cfg.SIMULATOR.SCENE_DATASET = str(scene_dataset_config)
    cfg.SIMULATOR.AUDIO.ENABLED = False

    cfg.SIMULATOR.RGB_SENSOR.WIDTH = int(width)
    cfg.SIMULATOR.RGB_SENSOR.HEIGHT = int(height)
    cfg.SIMULATOR.SEMANTIC_SENSOR.WIDTH = int(width)
    cfg.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = int(height)

    if hasattr(cfg.SIMULATOR.RGB_SENSOR, "HFOV"):
        cfg.SIMULATOR.RGB_SENSOR.HFOV = float(hfov)
    if hasattr(cfg.SIMULATOR.SEMANTIC_SENSOR, "HFOV"):
        cfg.SIMULATOR.SEMANTIC_SENSOR.HFOV = float(hfov)

    if hasattr(cfg.SIMULATOR.RGB_SENSOR, "POSITION"):
        cfg.SIMULATOR.RGB_SENSOR.POSITION = [0.0, float(sensor_height), 0.0]
    if hasattr(cfg.SIMULATOR.SEMANTIC_SENSOR, "POSITION"):
        cfg.SIMULATOR.SEMANTIC_SENSOR.POSITION = [0.0, float(sensor_height), 0.0]

    sensors = list(getattr(cfg.SIMULATOR.AGENT_0, "SENSORS", []))
    for sensor_name in ("RGB_SENSOR", "SEMANTIC_SENSOR"):
        if sensor_name not in sensors:
            sensors.append(sensor_name)
    cfg.SIMULATOR.AGENT_0.SENSORS = sensors

    cfg.freeze()
    sim = make_sim(cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR)
    sim.reset()
    return sim


def _get_sensor_uuid(sim: Any, sensor_type: str) -> str:
    if sensor_type == "rgb" and hasattr(sim, "_get_rgb_uuid"):
        rgb_uuid = sim._get_rgb_uuid()
        if rgb_uuid is not None:
            return rgb_uuid
    if sensor_type == "semantic" and hasattr(sim, "_get_semantic_uuid"):
        semantic_uuid = sim._get_semantic_uuid()
        if semantic_uuid is not None:
            return semantic_uuid
    return "rgb" if sensor_type == "rgb" else "semantic"


def _normalized_rotation_from_xyzw(coeffs: List[float]) -> Any:
    """Convert stored `[x, y, z, w]` coefficients into a unit quaternion."""
    quat_xyzw = np.array(coeffs, dtype=np.float64)
    if quat_xyzw.shape != (4,):
        raise RuntimeError(f"Invalid rotation coefficients: {coeffs}")
    norm = float(np.linalg.norm(quat_xyzw))
    if norm <= 1e-8:
        raise RuntimeError(f"Rotation quaternion has near-zero norm: {coeffs}")
    quat_xyzw /= norm
    return quat_from_coeffs(quat_xyzw.astype(np.float32))


def _load_json_source(args: argparse.Namespace) -> Dict[str, Any]:
    if args.view is not None:
        return json.loads(args.view)
    if args.view_json is None:
        raise RuntimeError("Provide either --view or --view-json.")
    with args.view_json.open("r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _extract_views(payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if "position" in payload and "rotation" in payload:
        return [payload], {}

    if isinstance(payload.get("views"), list):
        metadata = payload.get("metadata", {}) if isinstance(payload, dict) else {}
        return list(payload["views"]), metadata

    goals = payload.get("goals")
    if isinstance(goals, dict):
        gathered: List[Dict[str, Any]] = []
        for goal in goals.values():
            if isinstance(goal, dict) and isinstance(goal.get("render_views"), list):
                gathered.extend(goal["render_views"])
        return gathered, payload.get("metadata", {})

    raise RuntimeError("Unsupported JSON structure for pose rendering.")


def _select_view(
    views: List[Dict[str, Any]],
    semantic_id: Optional[int],
    view_index: int,
) -> Dict[str, Any]:
    filtered = views
    if semantic_id is not None:
        filtered = [
            view for view in views if int(view.get("semantic_id", -1)) == int(semantic_id)
        ]
    if not filtered:
        raise RuntimeError("No matching view found for the requested semantic id.")
    if view_index < 0 or view_index >= len(filtered):
        raise RuntimeError(
            f"view_index={view_index} is out of range for {len(filtered)} matching views."
        )
    return filtered[view_index]


def _resolve_scene_name(args: argparse.Namespace, metadata: Dict[str, Any]) -> str:
    if args.scene_name is not None:
        return args.scene_name
    scene_name = metadata.get("scene_name")
    if isinstance(scene_name, str) and scene_name:
        return scene_name
    raise RuntimeError("Scene name is required; pass it explicitly or via manifest metadata.")


def _resolve_render_value(
    override: Optional[Any],
    view_value: Optional[Any],
    metadata_value: Optional[Any],
    fallback: Any,
) -> Any:
    if override is not None:
        return override
    if view_value is not None:
        return view_value
    if metadata_value is not None:
        return metadata_value
    return fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render one RGB image from a saved pose-only ImageNav view entry."
    )
    parser.add_argument(
        "scene_name",
        type=str,
        nargs="?",
        help="Scene id like QUCTc6BB5sX; can be omitted if present in manifest metadata",
    )
    parser.add_argument(
        "--view-json",
        type=Path,
        default=None,
        help="Path to a single-view JSON, per-object views.json, or imagenav_eval_episodes.json",
    )
    parser.add_argument(
        "--view",
        type=str,
        default=None,
        help="Inline JSON string for a single render view",
    )
    parser.add_argument(
        "--semantic-id",
        type=int,
        default=None,
        help="Optional semantic id filter when the JSON contains many views",
    )
    parser.add_argument(
        "--view-index",
        type=int,
        default=0,
        help="Index among matching views",
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=Path("data/scene_datasets/mp3d"),
        help="Root directory containing MP3D scenes",
    )
    parser.add_argument(
        "--scene-dataset-config",
        type=Path,
        default=Path("data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"),
        help="Path to MP3D scene_dataset_config JSON",
    )
    parser.add_argument(
        "--exp-config",
        type=Path,
        default=Path("configs/semantic_audionav/av_nav/mp3d/semantic_audiogoal.yaml"),
        help="Habitat task config used to instantiate simulator",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Optional width override",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Optional height override",
    )
    parser.add_argument(
        "--hfov",
        type=float,
        default=None,
        help="Optional horizontal FOV override",
    )
    parser.add_argument(
        "--sensor-height",
        type=float,
        default=None,
        help="Optional sensor height override used if agent_base_position is absent",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rendered_pose_view.png"),
        help="Output RGB image path",
    )
    parser.add_argument(
        "--semantic-output",
        type=Path,
        default=None,
        help="Optional semantic visualization output path",
    )
    return parser.parse_args()


def _semantic_to_image(semantic: np.ndarray) -> Image.Image:
    semantic_ids = semantic.astype(np.int64, copy=False)
    semantic_rgb = np.zeros((*semantic_ids.shape, 3), dtype=np.uint8)
    for semantic_id in np.unique(semantic_ids):
        sid = int(semantic_id)
        if sid == 0:
            color = (0, 0, 0)
        else:
            color = ((sid * 37) % 256, (sid * 67) % 256, (sid * 97) % 256)
        semantic_rgb[semantic_ids == semantic_id] = color
    return Image.fromarray(semantic_rgb)


def main() -> None:
    args = parse_args()
    payload = _load_json_source(args)
    views, metadata = _extract_views(payload)
    view = _select_view(views, semantic_id=args.semantic_id, view_index=args.view_index)

    scene_name = _resolve_scene_name(args, metadata)
    render_sensor = metadata.get("render_sensor", {}) if isinstance(metadata, dict) else {}
    view_resolution = view.get("resolution", {}) if isinstance(view, dict) else {}

    width = int(
        _resolve_render_value(
            args.width,
            view_resolution.get("width"),
            render_sensor.get("width"),
            512,
        )
    )
    height = int(
        _resolve_render_value(
            args.height,
            view_resolution.get("height"),
            render_sensor.get("height"),
            512,
        )
    )
    hfov = float(
        _resolve_render_value(
            args.hfov,
            view.get("hfov"),
            render_sensor.get("hfov"),
            90.0,
        )
    )

    agent_base_position = view.get("agent_base_position")
    if agent_base_position is not None:
        base_position = np.array(agent_base_position, dtype=np.float32)
        sensor_height = float(
            _resolve_render_value(
                args.sensor_height,
                None,
                render_sensor.get("sensor_height"),
                float(view["position"][1]) - float(base_position[1]),
            )
        )
    else:
        sensor_position = np.array(view["position"], dtype=np.float32)
        sensor_height = float(
            _resolve_render_value(
                args.sensor_height,
                None,
                render_sensor.get("sensor_height"),
                1.25,
            )
        )
        base_position = np.array(sensor_position, dtype=np.float32)
        base_position[1] -= sensor_height

    rotation = _normalized_rotation_from_xyzw(view["rotation"])

    scene_dir = args.scene_dir.expanduser().resolve()
    scene_dataset_config = args.scene_dataset_config.expanduser().resolve()
    exp_config = args.exp_config.expanduser().resolve()
    scene_path = _resolve_scene_path(scene_dir, scene_name)

    sim = _build_simulator(
        scene_path=scene_path,
        scene_dataset_config=scene_dataset_config,
        exp_config=exp_config,
        width=width,
        height=height,
        hfov=hfov,
        sensor_height=sensor_height,
    )
    try:
        inner_sim = getattr(sim, "_sim", sim)
        rgb_uuid = _get_sensor_uuid(sim, "rgb")
        semantic_uuid = _get_sensor_uuid(sim, "semantic")

        sim.set_agent_state(base_position, rotation, reset_sensors=False)
        observations = inner_sim.get_sensor_observations()
        rgb = observations.get(rgb_uuid)
        semantic = observations.get(semantic_uuid)
        if rgb is None:
            raise RuntimeError("Missing RGB observation during rendering.")
        if semantic is not None and semantic.ndim == 3:
            semantic = semantic[:, :, 0]

        args.output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(args.output)
        print(f"Saved RGB render to: {args.output.resolve()}")

        if args.semantic_output is not None:
            if semantic is None:
                raise RuntimeError("Missing semantic observation during semantic render.")
            args.semantic_output.parent.mkdir(parents=True, exist_ok=True)
            _semantic_to_image(semantic).save(args.semantic_output)
            print(f"Saved semantic render to: {args.semantic_output.resolve()}")

        print(
            json.dumps(
                {
                    "scene_name": scene_name,
                    "scene_path": str(scene_path),
                    "width": width,
                    "height": height,
                    "hfov": hfov,
                    "sensor_height": sensor_height,
                    "agent_base_position": [round(float(v), 3) for v in base_position],
                    "rotation": [round(float(v), 3) for v in view["rotation"]],
                },
                indent=2,
            )
        )
    finally:
        sim.close()


if __name__ == "__main__":
    main()
