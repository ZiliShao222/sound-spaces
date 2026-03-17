#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm import tqdm


def _load_goal_feature_utils() -> Any:
    module_path = Path(__file__).resolve().parents[1] / "soundspaces" / "tasks" / "omni_long_goal_features.py"
    spec = importlib.util.spec_from_file_location("omni_long_goal_features", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load goal feature utils from `{module_path}`.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


goal_feature_utils = _load_goal_feature_utils()
DEFAULT_GOAL_FEATURES_FILENAME = goal_feature_utils.DEFAULT_GOAL_FEATURES_FILENAME
collect_instance_feature_inputs = goal_feature_utils.collect_instance_feature_inputs
flatten_instances_payload = goal_feature_utils.flatten_instances_payload
l2_normalize_feature = goal_feature_utils.l2_normalize_feature
save_goal_feature_cache = goal_feature_utils.save_goal_feature_cache


def _load_multimodal_module() -> Any:
    module_path = Path(__file__).with_name("generate_multimodal_starts.py")
    spec = importlib.util.spec_from_file_location("generate_multimodal_starts", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load generate_multimodal_starts.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute static omni-long goal features for object/text/image modalities.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Scene names or valid_instances.json paths.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Root folder used to resolve scene outputs and scene-local valid_instances files.",
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=Path("data/scene_datasets/mp3d"),
        help="Root directory containing MP3D scenes.",
    )
    parser.add_argument(
        "--scene-dataset-config",
        type=Path,
        default=Path("data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"),
        help="Path to MP3D scene_dataset_config JSON.",
    )
    parser.add_argument(
        "--exp-config",
        type=Path,
        default=Path("configs/omni-long/mp3d/omni-long_semantic_audio.yaml"),
        help="Habitat task config used to instantiate simulator.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="RGB render width for simulator re-rendering.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="RGB render height for simulator re-rendering.",
    )
    parser.add_argument(
        "--hfov",
        type=float,
        default=90.0,
        help="RGB sensor HFOV used during simulator re-rendering.",
    )
    parser.add_argument(
        "--sensor-height",
        type=float,
        default=1.25,
        help="RGB sensor height above agent base for simulator re-rendering.",
    )
    parser.add_argument(
        "--gpu-device-id",
        type=int,
        default=0,
        help="GPU id used by Habitat renderer.",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default=DEFAULT_GOAL_FEATURES_FILENAME,
        help="Per-scene cache filename.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        required=True,
        help=(
            "Encoder spec in `module.path:Symbol` or `path/to/file.py:Symbol` form. "
            "The encoder must provide `encode_texts()` and `encode_images()`."
        ),
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="",
        help="Optional model path passed to the encoder.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device string passed to the encoder.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size used for text/image encoding.",
    )
    parser.add_argument(
        "--category-prompt-template",
        type=str,
        default="Find the {category}.",
        help="Prompt template used for object/category goals.",
    )
    parser.add_argument(
        "--max-image-views",
        type=int,
        default=None,
        help="Optional cap on how many rendered views to encode per instance.",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Overwrite existing cache files.",
    )
    return parser.parse_args()


def _resolve_valid_instances_path(token: str, output_root: Path) -> Path:
    candidate = Path(token)
    if candidate.suffix.lower() == ".json":
        return candidate
    return output_root / str(token) / "valid_instances.json"


def _load_symbol(spec: str) -> Any:
    if ":" not in spec:
        raise ValueError("Encoder spec must be `module:Symbol` or `path.py:Symbol`.")
    module_ref, symbol_name = spec.rsplit(":", 1)
    if module_ref.endswith(".py") or Path(module_ref).exists():
        module_path = Path(module_ref)
        loader_spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
        if loader_spec is None or loader_spec.loader is None:
            raise RuntimeError(f"Failed to load encoder module from `{module_path}`.")
        module = importlib.util.module_from_spec(loader_spec)
        loader_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_ref)
    if not hasattr(module, symbol_name):
        raise AttributeError(f"Encoder symbol `{symbol_name}` not found in `{module_ref}`.")
    return getattr(module, symbol_name)


def _instantiate_encoder(factory: Any, args: argparse.Namespace) -> Any:
    if hasattr(factory, "from_args") and callable(getattr(factory, "from_args")):
        return factory.from_args(args)

    if inspect.isclass(factory):
        signature = inspect.signature(factory.__init__)
    else:
        signature = inspect.signature(factory)
    kwargs: Dict[str, Any] = {}
    for key, value in {
        "model_path": args.model_path,
        "device": args.device,
        "batch_size": args.batch_size,
    }.items():
        if key in signature.parameters:
            kwargs[key] = value
    return factory(**kwargs)


def _batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    stride = max(1, int(batch_size))
    for start in range(0, len(items), stride):
        yield items[start : start + stride]


def _ensure_encoder_contract(encoder: Any) -> None:
    for method_name in ("encode_texts", "encode_images"):
        if not hasattr(encoder, method_name) or not callable(getattr(encoder, method_name)):
            raise TypeError(f"Encoder must implement `{method_name}()`.")


def _encode_text_jobs(
    encoder: Any,
    jobs: Sequence[Dict[str, Any]],
    batch_size: int,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], int]:
    encoded: Dict[str, Dict[str, np.ndarray]] = {}
    feature_dim = 0
    for batch in tqdm(list(_batched(list(jobs), batch_size)), desc="text", leave=False):
        texts = [str(item["text"]) for item in batch]
        vectors = np.asarray(encoder.encode_texts(texts), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(batch):
            raise RuntimeError(
                "Encoder `encode_texts()` must return a 2D array with one row per input text."
            )
        if feature_dim > 0 and int(vectors.shape[1]) != int(feature_dim):
            raise RuntimeError(
                "Encoder `encode_texts()` returned inconsistent feature dimensions across batches."
            )
        feature_dim = int(vectors.shape[1])
        for item, vector in zip(batch, vectors):
            encoded.setdefault(str(item["instance_key"]), {})[str(item["modality"])] = l2_normalize_feature(vector)
    return encoded, feature_dim


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
    if rgb.shape[2] == 1:
        rgb = np.repeat(rgb, 3, axis=2)
    elif rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    return np.asarray(rgb, dtype=np.uint8)


def _build_renderer_simulator(
    scene_path: Path,
    scene_dataset_config: Optional[Path],
    exp_config: Path,
    width: int,
    height: int,
    hfov: float,
    sensor_height: float,
    gpu_device_id: int,
):
    try:
        from ss_baselines.omni_long.config.default import get_task_config
        from habitat.sims import make_sim
    except Exception as exc:
        raise RuntimeError(
            "Failed to import Habitat/SoundSpaces dependencies needed for simulator rendering."
        ) from exc

    cfg = get_task_config(config_paths=[str(exp_config)])
    cfg.defrost()
    cfg.SIMULATOR.TYPE = "ContinuousSoundSpacesSim"
    cfg.SIMULATOR.SCENE = str(scene_path)
    if scene_dataset_config is not None:
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
    rgb_uuid = "rgb"
    if hasattr(cfg.SIMULATOR, "RGB_SENSOR") and hasattr(cfg.SIMULATOR.RGB_SENSOR, "UUID"):
        rgb_uuid = str(cfg.SIMULATOR.RGB_SENSOR.UUID)
    return sim, rgb_uuid


def _render_rgb_image(
    sim: Any,
    rgb_uuid: str,
    render_view: Dict[str, Any],
) -> Image.Image:
    position = render_view.get("agent_base_position")
    if not isinstance(position, list) or len(position) != 3:
        position = render_view.get("position")
    rotation = render_view.get("rotation")
    if not isinstance(position, list) or len(position) != 3:
        raise RuntimeError(f"Missing valid render position in view payload: {render_view}")
    if not isinstance(rotation, list) or len(rotation) != 4:
        raise RuntimeError(f"Missing valid render rotation in view payload: {render_view}")

    observations = sim.get_observations_at(
        position=[float(v) for v in position],
        rotation=[float(v) for v in rotation],
        keep_agent_at_new_pose=False,
    )
    if observations is None:
        raise RuntimeError("Simulator returned no observations for render pose.")
    rgb = _extract_rgb_from_obs(observations, rgb_uuid)
    if rgb is None:
        raise RuntimeError("Failed to extract RGB observation from simulator output.")
    return Image.fromarray(rgb, mode="RGB")


def _encode_image_jobs(
    encoder: Any,
    jobs: Sequence[Dict[str, Any]],
    batch_size: int,
    sim: Any,
    rgb_uuid: str,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], int]:
    encoded: Dict[str, Dict[str, np.ndarray]] = {}
    feature_dim = 0
    for batch in tqdm(list(_batched(list(jobs), batch_size)), desc="image", leave=False):
        images: List[Image.Image] = []
        for item in batch:
            images.append(_render_rgb_image(sim, rgb_uuid, item["render_view"]))
        vectors = np.asarray(encoder.encode_images(images), dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[0] != len(batch):
            raise RuntimeError(
                "Encoder `encode_images()` must return a 2D array with one row per input image."
            )
        if feature_dim > 0 and int(vectors.shape[1]) != int(feature_dim):
            raise RuntimeError(
                "Encoder `encode_images()` returned inconsistent feature dimensions across batches."
            )
        feature_dim = int(vectors.shape[1])
        for item, vector in zip(batch, vectors):
            encoded.setdefault(str(item["instance_key"]), {})[str(item["modality"])] = l2_normalize_feature(vector)
    return encoded, feature_dim


def _merge_features(
    target: Dict[str, Dict[str, np.ndarray]],
    source: Dict[str, Dict[str, np.ndarray]],
) -> None:
    for instance_key, modalities in source.items():
        instance_payload = target.setdefault(str(instance_key), {})
        for modality, vector in modalities.items():
            instance_payload[str(modality)] = np.asarray(vector, dtype=np.float32)


def _scene_name_from_payload(input_path: Path, payload: Dict[str, Any]) -> str:
    scene_name = payload.get("scene_name")
    if isinstance(scene_name, str) and scene_name.strip():
        return scene_name.strip()
    return input_path.parent.name


def _build_jobs_for_scene(
    payload: Dict[str, Any],
    category_prompt_template: str,
    max_image_views: Optional[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    instances = flatten_instances_payload(payload)
    text_jobs: List[Dict[str, Any]] = []
    image_jobs: List[Dict[str, Any]] = []
    for instance_key, instance_record in tqdm(
        list(instances.items()),
        desc="collect",
        leave=False,
    ):
        feature_inputs = collect_instance_feature_inputs(
            instance_key,
            instance_record,
            category_prompt_template=category_prompt_template,
        )
        text_jobs.extend(feature_inputs["text_inputs"])
        candidate_images = feature_inputs["image_inputs"]
        if max_image_views is not None:
            candidate_images = candidate_images[: max(0, int(max_image_views))]
        image_jobs.extend(candidate_images)
    return text_jobs, image_jobs


def _resolve_scene_path(scene_name: str, args: argparse.Namespace) -> Path:
    multimodal_module = _load_multimodal_module()
    return multimodal_module._resolve_scene_path(args.scene_dir, scene_name)


def main() -> None:
    args = parse_args()
    encoder_factory = _load_symbol(args.encoder)
    encoder = _instantiate_encoder(encoder_factory, args)
    _ensure_encoder_contract(encoder)

    for input_token in args.inputs:
        input_path = _resolve_valid_instances_path(input_token, args.output_root)
        if not input_path.exists():
            raise FileNotFoundError(f"Could not find valid_instances file `{input_path}`.")

        payload = json.loads(input_path.read_text())
        scene_name = _scene_name_from_payload(input_path, payload)
        output_path = input_path.with_name(args.output_filename)
        if output_path.exists() and not args.overwrite:
            print(f"[skip] scene={scene_name} cache exists: {output_path}")
            continue

        print(f"[scene] {scene_name}")
        text_jobs, image_jobs = _build_jobs_for_scene(
            payload,
            category_prompt_template=args.category_prompt_template,
            max_image_views=args.max_image_views,
        )

        features: Dict[str, Dict[str, np.ndarray]] = {}
        feature_dim = 0
        text_features, text_dim = _encode_text_jobs(encoder, text_jobs, args.batch_size)
        _merge_features(features, text_features)
        feature_dim = max(feature_dim, int(text_dim))

        image_features: Dict[str, Dict[str, np.ndarray]] = {}
        image_dim = 0
        if image_jobs:
            scene_path = _resolve_scene_path(scene_name, args)
            scene_dataset_config = (
                args.scene_dataset_config.expanduser().resolve()
                if args.scene_dataset_config is not None
                else None
            )
            sim, rgb_uuid = _build_renderer_simulator(
                scene_path=scene_path,
                scene_dataset_config=scene_dataset_config,
                exp_config=args.exp_config.expanduser().resolve(),
                width=int(args.width),
                height=int(args.height),
                hfov=float(args.hfov),
                sensor_height=float(args.sensor_height),
                gpu_device_id=int(args.gpu_device_id),
            )
            try:
                image_features, image_dim = _encode_image_jobs(
                    encoder,
                    image_jobs,
                    args.batch_size,
                    sim=sim,
                    rgb_uuid=rgb_uuid,
                )
            finally:
                if hasattr(sim, "close"):
                    sim.close()
        _merge_features(features, image_features)
        feature_dim = max(feature_dim, int(image_dim))

        if text_dim > 0 and image_dim > 0 and int(text_dim) != int(image_dim):
            raise RuntimeError(
                "Text/image feature dimensions do not match: "
                f"text_dim={text_dim}, image_dim={image_dim}."
            )

        if feature_dim <= 0:
            raise RuntimeError(f"No features were produced for scene `{scene_name}`.")

        encoder_name = getattr(encoder, "name", encoder.__class__.__name__)
        save_goal_feature_cache(
            output_path,
            feature_dim=feature_dim,
            entries=features,
            encoder_name=str(encoder_name),
            encoder_kwargs={
                "model_path": str(args.model_path),
                "device": str(args.device),
                "batch_size": int(args.batch_size),
            },
            source_path=str(input_path),
            scene_name=scene_name,
            category_prompt_template=str(args.category_prompt_template),
        )
        print(
            "[done] scene={scene} instances={instances} text_jobs={text_jobs} image_jobs={image_jobs} -> {path}".format(
                scene=scene_name,
                instances=len(features),
                text_jobs=len(text_jobs),
                image_jobs=len(image_jobs),
                path=output_path,
            )
        )


if __name__ == "__main__":
    main()
