#!/usr/bin/env python3

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch


GOAL_FEATURE_CACHE_VERSION = 1
DEFAULT_GOAL_FEATURES_FILENAME = "goal_features.pt"


def scene_name_from_scene_id(scene_id: Any) -> str:
    token = os.path.basename(str(scene_id))
    stem, ext = os.path.splitext(token)
    if ext.lower() in {".glb", ".ply", ".obj"}:
        return stem
    return token


def goal_feature_cache_path(
    features_root: Any,
    scene_id: Any,
    filename: str = DEFAULT_GOAL_FEATURES_FILENAME,
) -> Path:
    return Path(features_root) / scene_name_from_scene_id(scene_id) / str(filename)


def normalize_goal_modality(modality: Any) -> str:
    token = str(modality or "").strip().lower()
    if not token:
        return "object"
    if token in {"object", "category"}:
        return "object"
    if token in {"description", "text", "text_description"} or token.startswith("text"):
        return "description"
    if token == "image":
        return "image_0"
    if token.startswith("image_"):
        suffix = token.split("_")[-1]
        if suffix.isdigit():
            return f"image_{int(suffix)}"
    return token


def goal_modality_lookup_order(modality: Any) -> Tuple[str, ...]:
    token = normalize_goal_modality(modality)
    ordered: List[str] = [token]
    if token == "object":
        ordered.append("category")
    elif token == "description":
        ordered.extend(["text", "text_description", "object", "category"])
    deduped: List[str] = []
    for candidate in ordered:
        if candidate not in deduped:
            deduped.append(candidate)
    return tuple(deduped)


def l2_normalize_feature(vector: Any) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm <= 1e-6:
        return np.zeros_like(array, dtype=np.float32)
    return (array / norm).astype(np.float32)


def save_goal_feature_cache(
    path: Any,
    *,
    feature_dim: int,
    entries: Dict[str, Dict[str, np.ndarray]],
    encoder_name: str,
    encoder_kwargs: Optional[Dict[str, Any]] = None,
    source_path: Optional[str] = None,
    scene_name: Optional[str] = None,
    category_prompt_template: Optional[str] = None,
) -> None:
    serialized_entries: Dict[str, Dict[str, torch.Tensor]] = {}
    for instance_key, modalities in entries.items():
        instance_payload: Dict[str, torch.Tensor] = {}
        for modality, vector in modalities.items():
            normalized = l2_normalize_feature(vector)
            instance_payload[str(modality)] = torch.from_numpy(normalized)
        serialized_entries[str(instance_key)] = instance_payload

    payload = {
        "version": int(GOAL_FEATURE_CACHE_VERSION),
        "feature_dim": int(feature_dim),
        "scene_name": str(scene_name or ""),
        "encoder": {
            "name": str(encoder_name),
            "kwargs": dict(encoder_kwargs or {}),
        },
        "source_path": str(source_path or ""),
        "category_prompt_template": str(category_prompt_template or ""),
        "entries": serialized_entries,
    }
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def load_goal_feature_cache(path: Any) -> Dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu")
    raw_entries = payload.get("entries", {})
    entries: Dict[str, Dict[str, np.ndarray]] = {}
    for instance_key, modalities in raw_entries.items():
        if not isinstance(modalities, dict):
            continue
        entry_payload: Dict[str, np.ndarray] = {}
        for modality, value in modalities.items():
            if isinstance(value, torch.Tensor):
                array = value.detach().cpu().numpy().astype(np.float32)
            else:
                array = np.asarray(value, dtype=np.float32)
            entry_payload[str(modality)] = array.reshape(-1)
        entries[str(instance_key)] = entry_payload
    payload["entries"] = entries
    payload["feature_dim"] = int(payload.get("feature_dim", 0))
    return payload


def lookup_goal_feature(
    cache_payload: Dict[str, Any],
    instance_key: Any,
    modality: Any,
) -> Optional[np.ndarray]:
    entries = cache_payload.get("entries", {})
    if not isinstance(entries, dict):
        return None
    instance_payload = entries.get(str(instance_key))
    if not isinstance(instance_payload, dict):
        return None
    for candidate in goal_modality_lookup_order(modality):
        value = instance_payload.get(candidate)
        if value is None:
            continue
        return np.asarray(value, dtype=np.float32).reshape(-1)
    return None


def extract_text_description(instance_record: Dict[str, Any]) -> str:
    for key in ("description", "text_description", "text", "caption"):
        value = instance_record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def build_category_prompt(instance_record: Dict[str, Any], prompt_template: str) -> str:
    category = str(instance_record.get("category", "object")).strip() or "object"
    return str(prompt_template).format(category=category)


def resolve_instance_render_views(
    instance_record: Dict[str, Any],
) -> List[Tuple[str, Dict[str, Any]]]:
    image_payload = instance_record.get("image")
    if not isinstance(image_payload, dict):
        return []

    render_views = image_payload.get("render_views")
    if not isinstance(render_views, list):
        return []

    resolved: List[Tuple[str, Dict[str, Any]]] = []
    for index, render_view in enumerate(render_views):
        if not isinstance(render_view, dict):
            continue
        resolved.append((f"image_{int(index)}", dict(render_view)))
    return resolved


def collect_instance_feature_inputs(
    instance_key: str,
    instance_record: Dict[str, Any],
    category_prompt_template: str,
) -> Dict[str, List[Dict[str, Any]]]:
    text_inputs: List[Dict[str, Any]] = [
        {
            "instance_key": str(instance_key),
            "modality": "object",
            "text": build_category_prompt(instance_record, category_prompt_template),
        }
    ]
    description = extract_text_description(instance_record)
    if description:
        text_inputs.append(
            {
                "instance_key": str(instance_key),
                "modality": "description",
                "text": description,
            }
        )

    image_inputs: List[Dict[str, Any]] = []
    for modality, render_view in resolve_instance_render_views(instance_record):
        image_inputs.append(
            {
                "instance_key": str(instance_key),
                "modality": str(modality),
                "render_view": render_view,
            }
        )

    return {
        "text_inputs": text_inputs,
        "image_inputs": image_inputs,
    }


def flatten_instances_payload(payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    raw_instances = payload.get("instances")
    flattened: Dict[str, Dict[str, Any]] = {}
    if not isinstance(raw_instances, dict):
        return flattened
    for key, value in raw_instances.items():
        if isinstance(value, dict) and "semantic_id" in value:
            flattened[str(key)] = value
            continue
        if isinstance(value, dict):
            for nested_key, nested_value in value.items():
                if isinstance(nested_value, dict):
                    flattened[str(nested_key)] = nested_value
    return flattened


def infer_feature_dim(cache_payload: Dict[str, Any]) -> int:
    feature_dim = int(cache_payload.get("feature_dim", 0))
    if feature_dim > 0:
        return feature_dim
    entries = cache_payload.get("entries", {})
    if not isinstance(entries, dict):
        return 0
    for modalities in entries.values():
        if not isinstance(modalities, dict):
            continue
        for value in modalities.values():
            return int(np.asarray(value).reshape(-1).shape[0])
    return 0
