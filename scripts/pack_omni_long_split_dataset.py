#!/usr/bin/env python3

"""Pack OmniLong trajectory dataset(s) into GOAT-style split/content layout.

This script keeps each scene dataset file unchanged, and only rearranges files
into a layout like:

data/datasets/omni_long/mp3d/v1/train/
  train.json.gz
  content/
    <scene>.json.gz

The root ``train.json.gz`` is a lightweight index file used by the dataset
loader. The actual episodes remain in ``content/<scene>.json.gz``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List


CATEGORY_TO_TASK_CATEGORY_ID: Dict[str, int] = {
    "chair": 0,
    "table": 1,
    "picture": 2,
    "cabinet": 3,
    "cushion": 4,
    "sofa": 5,
    "bed": 6,
    "chest_of_drawers": 7,
    "plant": 8,
    "sink": 9,
    "toilet": 10,
    "stool": 11,
    "towel": 12,
    "tv_monitor": 13,
    "shower": 14,
    "bathtub": 15,
    "counter": 16,
    "fireplace": 17,
    "gym_equipment": 18,
    "seating": 19,
    "clothes": 20,
}


CATEGORY_TO_MP3D_CATEGORY_ID: Dict[str, int] = {
    "chair": 3,
    "table": 5,
    "picture": 6,
    "cabinet": 7,
    "cushion": 8,
    "sofa": 10,
    "bed": 11,
    "chest_of_drawers": 13,
    "plant": 14,
    "sink": 15,
    "toilet": 18,
    "stool": 19,
    "towel": 20,
    "tv_monitor": 22,
    "shower": 23,
    "bathtub": 25,
    "counter": 26,
    "fireplace": 27,
    "gym_equipment": 33,
    "seating": 34,
    "clothes": 38,
}


def _read_json(path: Path) -> Any:
    name = path.name.lower()
    if name.endswith(".json.gz") or name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_gz(path: Path, payload: Any, pretty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2 if pretty else None,
        )


def _infer_scene_name(payload: Dict[str, Any], input_path: Path) -> str:
    scene_name = payload.get("scene_name")
    if isinstance(scene_name, str) and scene_name.strip():
        return scene_name.strip()

    scene_id = payload.get("scene_id")
    if isinstance(scene_id, str) and scene_id.strip():
        return Path(scene_id).stem

    episodes = payload.get("episodes")
    if isinstance(episodes, list) and episodes:
        episode_scene_id = episodes[0].get("scene_id")
        if isinstance(episode_scene_id, str) and episode_scene_id.strip():
            return Path(episode_scene_id).stem

    name = input_path.name
    if name.lower().endswith(".json.gz"):
        return name[: -len(".json.gz")]
    if name.lower().endswith(".json"):
        return input_path.stem
    if name.lower().endswith(".gz"):
        return name[: -len(".gz")]
    return input_path.stem


def _normalize_inputs(values: Iterable[Path]) -> List[Path]:
    paths: List[Path] = []
    for value in values:
        path = value.expanduser().resolve()
        if not path.is_file():
            raise RuntimeError(f"Input dataset not found: {path}")
        paths.append(path)
    if not paths:
        raise RuntimeError("At least one --input path is required")
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pack OmniLong dataset(s) into GOAT-style split/content layout",
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="Input OmniLong dataset path(s) (.json or .json.gz)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/datasets/omni_long/mp3d/v1"),
        help="Dataset root directory that contains train/ val/ ... splits",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Target split name, such as train or val",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON in gzip output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_paths = _normalize_inputs(args.input)
    split_name = str(args.split).strip()
    if not split_name:
        raise RuntimeError("--split must be a non-empty string")

    output_root = args.output_root.expanduser().resolve()
    split_dir = output_root / split_name
    content_dir = split_dir / "content"
    split_index_path = split_dir / f"{split_name}.json.gz"

    packed_scenes: List[str] = []
    for input_path in input_paths:
        payload = _read_json(input_path)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Dataset root must be a JSON object: {input_path}")

        scene_name = _infer_scene_name(payload, input_path)
        if not scene_name:
            raise RuntimeError(f"Failed to infer scene name from: {input_path}")

        scene_output_path = content_dir / f"{scene_name}.json.gz"
        _write_json_gz(scene_output_path, payload, pretty=args.pretty)
        packed_scenes.append(scene_name)
        print(f"Packed scene dataset: {input_path} -> {scene_output_path}")

    split_index_payload = {
        "dataset": "omni_long_nav",
        "split": split_name,
        "episodes": [],
        "content_scenes_path": "{data_path}/content/{scene}.json.gz",
        "category_to_task_category_id": CATEGORY_TO_TASK_CATEGORY_ID,
        "category_to_mp3d_category_id": CATEGORY_TO_MP3D_CATEGORY_ID,
    }
    _write_json_gz(split_index_path, split_index_payload, pretty=args.pretty)

    existing_scene_files = sorted(
        path.name for path in content_dir.glob("*.json.gz") if path.is_file()
    )

    print(f"Split index: {split_index_path}")
    print(f"Content dir: {content_dir}")
    print(f"Packed this run: {len(packed_scenes)} scene(s)")
    print(f"Scenes now available: {len(existing_scene_files)}")
    for filename in existing_scene_files:
        print(f"  - {filename}")


if __name__ == "__main__":
    main()
