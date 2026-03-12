#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


def _is_vec3(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(v, (int, float)) for v in value)
    )


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _infer_output_root(valid_instances_path: Path) -> Path:
    return valid_instances_path.parent.parent


def _resolve_relative_path(base_root: Path, raw_path: Optional[str]) -> Optional[Path]:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None
    p = Path(raw_path)
    if p.is_absolute():
        return p
    return (base_root / p).resolve()


def _build_reference_index(
    valid_instances_payload: Dict[str, Any],
    output_root: Path,
) -> Tuple[Dict[Tuple[str, int], Path], Dict[str, List[Optional[Path]]]]:
    by_instance_and_index: Dict[Tuple[str, int], Path] = {}
    all_by_instance: Dict[str, List[Optional[Path]]] = {}

    nested_instances = valid_instances_payload.get("instances")
    if not isinstance(nested_instances, dict):
        return by_instance_and_index, all_by_instance

    for _, per_category in nested_instances.items():
        if not isinstance(per_category, dict):
            continue
        for instance_key, record in per_category.items():
            if not isinstance(record, dict):
                continue
            key = str(instance_key)
            image_payload = record.get("image")
            if not isinstance(image_payload, dict):
                continue

            views_json_path = _resolve_relative_path(
                output_root,
                image_payload.get("views_json"),
            )
            if views_json_path is None or not views_json_path.is_file():
                continue

            try:
                views_payload = _load_json(views_json_path)
            except Exception:
                continue

            views = views_payload.get("views")
            if not isinstance(views, list):
                continue

            instance_paths: List[Optional[Path]] = []
            for idx, view in enumerate(views):
                if not isinstance(view, dict):
                    instance_paths.append(None)
                    continue
                goal_image_rel = view.get("goal_image")
                goal_image_path = _resolve_relative_path(output_root, goal_image_rel)
                if goal_image_path is not None and goal_image_path.is_file():
                    by_instance_and_index[(key, int(idx))] = goal_image_path
                    instance_paths.append(goal_image_path)
                else:
                    instance_paths.append(None)
            all_by_instance[key] = instance_paths

    return by_instance_and_index, all_by_instance


def _parse_image_index(modality_token: str) -> Optional[int]:
    if not isinstance(modality_token, str):
        return None
    token = modality_token.strip().lower()
    if token == "image":
        return 0
    if token.startswith("image_"):
        suffix = token[len("image_") :]
        if suffix.isdigit():
            return int(suffix)
    return None


def _collect_samples_from_instances(
    trajectory_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    instances = trajectory_payload.get("instances")
    if not isinstance(instances, dict):
        return samples

    for instance_key, record in instances.items():
        if not isinstance(record, dict):
            continue
        image_payload = record.get("image")
        if not isinstance(image_payload, dict):
            continue
        render_views = image_payload.get("render_views")
        if not isinstance(render_views, list):
            continue
        for view_index, render_view in enumerate(render_views):
            if not isinstance(render_view, dict):
                continue
            samples.append(
                {
                    "source": "instances",
                    "instance_key": str(instance_key),
                    "view_index": int(view_index),
                    "render_view": render_view,
                }
            )
    return samples


def _collect_samples_from_episodes(
    trajectory_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    samples: List[Dict[str, Any]] = []
    episodes = trajectory_payload.get("episodes")
    instances = trajectory_payload.get("instances")
    if not isinstance(episodes, list) or not isinstance(instances, dict):
        return samples

    for episode in episodes:
        if not isinstance(episode, dict):
            continue
        episode_id = str(episode.get("episode_id", ""))
        goals = episode.get("goals")
        if not isinstance(goals, list):
            continue
        for goal_index, goal in enumerate(goals):
            if not (isinstance(goal, list) and len(goal) >= 2):
                continue
            instance_key = str(goal[0])
            modality = str(goal[1])
            image_index = _parse_image_index(modality)
            if image_index is None:
                continue

            instance_record = instances.get(instance_key)
            if not isinstance(instance_record, dict):
                continue
            image_payload = instance_record.get("image")
            if not isinstance(image_payload, dict):
                continue
            render_views = image_payload.get("render_views")
            if not isinstance(render_views, list):
                continue
            if image_index < 0 or image_index >= len(render_views):
                continue
            render_view = render_views[image_index]
            if not isinstance(render_view, dict):
                continue

            samples.append(
                {
                    "source": "episodes",
                    "episode_id": episode_id,
                    "goal_index": int(goal_index),
                    "instance_key": instance_key,
                    "modality": modality,
                    "view_index": int(image_index),
                    "render_view": render_view,
                }
            )

    return samples


def _resolve_scene_path(
    trajectory_payload: Dict[str, Any],
    scene_dir: Path,
) -> Path:
    sampling = trajectory_payload.get("sampling")
    if isinstance(sampling, dict):
        start_sampling = sampling.get("start_sampling")
        if isinstance(start_sampling, dict):
            scene_path = start_sampling.get("scene_path")
            if isinstance(scene_path, str) and scene_path.strip():
                p = Path(scene_path)
                if p.is_file():
                    return p.resolve()

    scene_id = trajectory_payload.get("scene_id")
    if isinstance(scene_id, str) and scene_id.strip():
        normalized = scene_id.strip()
        if normalized.startswith("mp3d/"):
            normalized = normalized[len("mp3d/") :]
        candidate = (scene_dir / normalized).resolve()
        if candidate.is_file():
            return candidate

    scene_name = trajectory_payload.get("scene_name")
    if isinstance(scene_name, str) and scene_name.strip():
        candidate = (scene_dir / "val" / scene_name / f"{scene_name}.glb").resolve()
        if candidate.is_file():
            return candidate

    raise RuntimeError("Failed to resolve scene GLB path from trajectory dataset.")


def _extract_sensor_profile(
    samples: List[Dict[str, Any]],
    fallback_width: int,
    fallback_height: int,
    fallback_hfov: float,
    fallback_sensor_height: float,
) -> Dict[str, float]:
    width = int(fallback_width)
    height = int(fallback_height)
    hfov = float(fallback_hfov)
    sensor_height = float(fallback_sensor_height)

    for sample in samples:
        view = sample.get("render_view")
        if not isinstance(view, dict):
            continue
        resolution = view.get("resolution")
        if isinstance(resolution, dict):
            w = resolution.get("width")
            h = resolution.get("height")
            if isinstance(w, int) and w > 0:
                width = int(w)
            if isinstance(h, int) and h > 0:
                height = int(h)
        if isinstance(view.get("hfov"), (int, float)):
            hfov = float(view.get("hfov"))

        position = view.get("position")
        base = view.get("agent_base_position")
        if _is_vec3(position) and _is_vec3(base):
            sensor_height = float(position[1]) - float(base[1])
        break

    return {
        "width": float(width),
        "height": float(height),
        "hfov": float(hfov),
        "sensor_height": float(sensor_height),
    }


def _build_rgb_simulator(
    scene_path: Path,
    scene_dataset_config: Path,
    exp_config: Path,
    width: int,
    height: int,
    hfov: float,
    sensor_height: float,
    gpu_device_id: int,
):
    from ss_baselines.av_nav.config.default import get_task_config
    from habitat.sims import make_sim

    cfg = get_task_config(config_paths=[str(exp_config)])
    cfg.defrost()
    cfg.SIMULATOR.TYPE = "ContinuousSoundSpacesSim"
    cfg.SIMULATOR.SCENE = str(scene_path)
    cfg.SIMULATOR.SCENE_DATASET = str(scene_dataset_config)
    cfg.SIMULATOR.AUDIO.ENABLED = False
    cfg.SIMULATOR.CREATE_RENDERER = True
    if hasattr(cfg.SIMULATOR, "HABITAT_SIM_V0") and hasattr(
        cfg.SIMULATOR.HABITAT_SIM_V0,
        "GPU_DEVICE_ID",
    ):
        cfg.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = int(gpu_device_id)

    cfg.SIMULATOR.RGB_SENSOR.WIDTH = int(width)
    cfg.SIMULATOR.RGB_SENSOR.HEIGHT = int(height)
    if hasattr(cfg.SIMULATOR.RGB_SENSOR, "HFOV"):
        cfg.SIMULATOR.RGB_SENSOR.HFOV = float(hfov)
    if hasattr(cfg.SIMULATOR.RGB_SENSOR, "POSITION"):
        cfg.SIMULATOR.RGB_SENSOR.POSITION = [0.0, float(sensor_height), 0.0]
    cfg.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR"]
    cfg.freeze()

    sim = make_sim(cfg.SIMULATOR.TYPE, config=cfg.SIMULATOR)
    sim.reset()
    inner_sim = sim._sim if hasattr(sim, "_sim") else sim
    rgb_uuid = "rgb"
    if hasattr(cfg.SIMULATOR, "RGB_SENSOR") and hasattr(cfg.SIMULATOR.RGB_SENSOR, "UUID"):
        rgb_uuid = str(cfg.SIMULATOR.RGB_SENSOR.UUID)
    return sim, inner_sim, rgb_uuid


def _extract_rgb_from_obs(obs: Dict[str, Any], preferred_uuid: str) -> Optional[np.ndarray]:
    candidate = obs.get(preferred_uuid)
    if candidate is None and preferred_uuid != "rgb":
        candidate = obs.get("rgb")

    if candidate is None:
        for value in obs.values():
            array = np.asarray(value)
            if array.ndim == 3 and array.shape[2] in (1, 3, 4):
                candidate = array
                break

    if candidate is None:
        return None

    rgb = np.asarray(candidate)
    if rgb.ndim != 3:
        return None
    if rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    elif rgb.shape[2] == 1:
        rgb = np.repeat(rgb, 3, axis=2)
    elif rgb.shape[2] != 3:
        return None
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return rgb


def _render_rgb(
    sim,
    inner_sim,
    rgb_uuid: str,
    render_view: Dict[str, Any],
    sensor_height: float,
) -> np.ndarray:
    position = render_view.get("position")
    base_position = render_view.get("agent_base_position")
    rotation = render_view.get("rotation")

    if _is_vec3(base_position):
        base = [float(v) for v in base_position]
    elif _is_vec3(position):
        base = [float(position[0]), float(position[1]) - float(sensor_height), float(position[2])]
    else:
        raise RuntimeError("render_view missing position/agent_base_position")

    if not (isinstance(rotation, list) and len(rotation) == 4):
        raise RuntimeError("render_view missing quaternion rotation")
    quat = [float(v) for v in rotation]

    sim.set_agent_state(base, quat, reset_sensors=False)
    obs = inner_sim.get_sensor_observations()
    rgb = _extract_rgb_from_obs(obs, preferred_uuid=rgb_uuid)
    if rgb is None:
        raise RuntimeError("Failed to extract RGB observation from simulator output")
    return rgb


def _image_similarity(rendered: np.ndarray, reference: np.ndarray) -> float:
    if rendered.shape != reference.shape:
        return 0.0
    diff = np.abs(rendered.astype(np.float32) - reference.astype(np.float32))
    mae = float(np.mean(diff))
    similarity = 1.0 - (mae / 255.0)
    return max(0.0, min(1.0, similarity))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate trajectory image replay consistency against offline saved images",
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to trajectory_dataset.json",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["instances", "episodes", "both"],
        default="both",
        help="Validation source mode",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output report JSON path (default: <input_dir>/image_replay_validation.json)",
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=Path("data/scene_datasets/mp3d"),
        help="Scene root directory",
    )
    parser.add_argument(
        "--scene-dataset-config",
        type=Path,
        default=Path("data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"),
        help="Scene dataset config JSON",
    )
    parser.add_argument(
        "--exp-config",
        type=Path,
        default=Path("configs/semantic_audionav/av_nav/mp3d/semantic_audiogoal.yaml"),
        help="Habitat task config used to build simulator",
    )
    parser.add_argument(
        "--gpu-device-id",
        type=int,
        default=0,
        help="Habitat simulator GPU device id",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.99,
        help="Minimum similarity to count as pass",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on total validated samples",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="Fallback sensor width if not available in render view",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="Fallback sensor height if not available in render view",
    )
    parser.add_argument(
        "--hfov",
        type=float,
        default=90.0,
        help="Fallback sensor HFOV if not available in render view",
    )
    parser.add_argument(
        "--sensor-height",
        type=float,
        default=1.25,
        help="Fallback sensor height above agent base",
    )
    parser.add_argument(
        "--episode-check-best-match",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="In episode mode, also compare rendered image against all views of the same instance",
    )
    parser.add_argument(
        "--print-reference-paths",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Print offline reference image paths while validating",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise RuntimeError(f"Input trajectory dataset not found: {input_path}")

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else input_path.parent / "image_replay_validation.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    trajectory_payload = _load_json(input_path)
    source_valid_instances = trajectory_payload.get("source_valid_instances")
    if not isinstance(source_valid_instances, str) or not source_valid_instances.strip():
        raise RuntimeError("trajectory_dataset.json missing source_valid_instances")

    valid_instances_path = Path(source_valid_instances).expanduser().resolve()
    if not valid_instances_path.is_file():
        raise RuntimeError(f"source_valid_instances not found: {valid_instances_path}")
    valid_instances_payload = _load_json(valid_instances_path)
    output_root = _infer_output_root(valid_instances_path)

    ref_by_key_idx, ref_all_by_instance = _build_reference_index(
        valid_instances_payload,
        output_root,
    )

    samples: List[Dict[str, Any]] = []
    if args.mode in ("instances", "both"):
        samples.extend(_collect_samples_from_instances(trajectory_payload))
    if args.mode in ("episodes", "both"):
        samples.extend(_collect_samples_from_episodes(trajectory_payload))

    if args.max_samples is not None and int(args.max_samples) >= 0:
        samples = samples[: int(args.max_samples)]

    if not samples:
        raise RuntimeError("No validation samples found for requested mode")

    sensor_profile = _extract_sensor_profile(
        samples,
        fallback_width=int(args.width),
        fallback_height=int(args.height),
        fallback_hfov=float(args.hfov),
        fallback_sensor_height=float(args.sensor_height),
    )

    scene_path = _resolve_scene_path(
        trajectory_payload,
        scene_dir=args.scene_dir.expanduser().resolve(),
    )
    scene_dataset_config = args.scene_dataset_config.expanduser().resolve()
    exp_config = args.exp_config.expanduser().resolve()

    sim, inner_sim, rgb_uuid = _build_rgb_simulator(
        scene_path=scene_path,
        scene_dataset_config=scene_dataset_config,
        exp_config=exp_config,
        width=int(sensor_profile["width"]),
        height=int(sensor_profile["height"]),
        hfov=float(sensor_profile["hfov"]),
        sensor_height=float(sensor_profile["sensor_height"]),
        gpu_device_id=int(args.gpu_device_id),
    )

    results: List[Dict[str, Any]] = []
    compared = 0
    passed = 0
    missing_ref = 0
    render_error = 0

    try:
        for sample in samples:
            instance_key = str(sample["instance_key"])
            view_index = int(sample["view_index"])
            render_view = sample["render_view"]
            ref_path = ref_by_key_idx.get((instance_key, view_index))

            row: Dict[str, Any] = {
                "source": sample.get("source"),
                "instance_key": instance_key,
                "view_index": view_index,
                "episode_id": sample.get("episode_id"),
                "goal_index": sample.get("goal_index"),
                "modality": sample.get("modality"),
                "reference_image": str(ref_path) if ref_path is not None else None,
                "similarity": None,
                "threshold": float(args.similarity_threshold),
                "pass": False,
                "status": "ok",
            }

            if ref_path is None or not ref_path.is_file():
                row["status"] = "missing_reference"
                if bool(args.print_reference_paths):
                    print(
                        "[{}] episode={} goal={} instance={} view={} offline_ref=MISSING".format(
                            sample.get("source"),
                            sample.get("episode_id"),
                            sample.get("goal_index"),
                            instance_key,
                            view_index,
                        )
                    )
                missing_ref += 1
                results.append(row)
                continue

            if bool(args.print_reference_paths):
                print(
                    "[{}] episode={} goal={} instance={} view={} offline_ref={}".format(
                        sample.get("source"),
                        sample.get("episode_id"),
                        sample.get("goal_index"),
                        instance_key,
                        view_index,
                        ref_path,
                    )
                )

            try:
                rendered = _render_rgb(
                    sim=sim,
                    inner_sim=inner_sim,
                    rgb_uuid=rgb_uuid,
                    render_view=render_view,
                    sensor_height=float(sensor_profile["sensor_height"]),
                )
            except Exception as exc:
                row["status"] = "render_error"
                row["error"] = str(exc)
                render_error += 1
                results.append(row)
                continue

            ref_img = np.asarray(Image.open(ref_path).convert("RGB"), dtype=np.uint8)
            similarity = _image_similarity(rendered, ref_img)
            print(f'similarity: {similarity}')
            row["similarity"] = round(float(similarity), 6)
            row["pass"] = bool(float(similarity) >= float(args.similarity_threshold))
            compared += 1
            if row["pass"]:
                passed += 1

            if (
                bool(args.episode_check_best_match)
                and str(sample.get("source")) == "episodes"
            ):
                ref_list = ref_all_by_instance.get(instance_key, [])
                best_index = None
                best_similarity = None
                for idx, candidate_path in enumerate(ref_list):
                    if candidate_path is None or not candidate_path.is_file():
                        continue
                    candidate_img = np.asarray(
                        Image.open(candidate_path).convert("RGB"),
                        dtype=np.uint8,
                    )
                    candidate_similarity = _image_similarity(rendered, candidate_img)
                    if best_similarity is None or float(candidate_similarity) > float(best_similarity):
                        best_similarity = float(candidate_similarity)
                        best_index = int(idx)
                row["best_match_view_index"] = best_index
                row["best_match_similarity"] = (
                    round(float(best_similarity), 6)
                    if best_similarity is not None
                    else None
                )
                row["expected_index_is_best"] = (
                    bool(best_index == view_index) if best_index is not None else None
                )

            results.append(row)
    finally:
        if hasattr(sim, "close"):
            sim.close()

    summary = {
        "input": str(input_path),
        "source_valid_instances": str(valid_instances_path),
        "scene_path": str(scene_path),
        "mode": str(args.mode),
        "sensor_profile": {
            "width": int(sensor_profile["width"]),
            "height": int(sensor_profile["height"]),
            "hfov": float(sensor_profile["hfov"]),
            "sensor_height": float(sensor_profile["sensor_height"]),
        },
        "threshold": float(args.similarity_threshold),
        "total_samples": int(len(samples)),
        "compared": int(compared),
        "passed": int(passed),
        "failed": int(max(0, compared - passed)),
        "missing_reference": int(missing_ref),
        "render_error": int(render_error),
        "pass_rate_over_compared": (
            round(float(passed) / float(compared), 6) if compared > 0 else None
        ),
    }

    report = {
        "summary": summary,
        "results": results,
    }
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Input: {input_path}")
    print(f"Report: {output_path}")
    print(
        "Compared={compared} Passed={passed} MissingRef={missing_ref} RenderError={render_error}".format(
            compared=compared,
            passed=passed,
            missing_ref=missing_ref,
            render_error=render_error,
        )
    )
    if compared > 0:
        print(f"Pass rate: {summary['pass_rate_over_compared']:.4f}")


if __name__ == "__main__":
    main()
