#!/usr/bin/env python3

"""Render all valid instances from new-format valid_instances.json into one GLB.

This script is designed for JSON files like:
  output/<scene>/valid_instances.json

It adds one colored bounding box per instance to the source scene GLB, and uses
the same color for all instances in the same category.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pygltflib


# Distinct colors (RGB in [0,1]); alpha is added at runtime.
CATEGORY_PALETTE: List[Tuple[float, float, float]] = [
    (0.121, 0.466, 0.705),
    (1.000, 0.498, 0.054),
    (0.172, 0.627, 0.172),
    (0.839, 0.153, 0.157),
    (0.580, 0.404, 0.741),
    (0.549, 0.337, 0.294),
    (0.890, 0.467, 0.761),
    (0.498, 0.498, 0.498),
    (0.737, 0.741, 0.133),
    (0.090, 0.745, 0.811),
    (0.682, 0.780, 0.910),
    (1.000, 0.733, 0.471),
    (0.596, 0.875, 0.541),
    (1.000, 0.596, 0.588),
    (0.773, 0.690, 0.835),
    (0.769, 0.612, 0.580),
    (0.969, 0.714, 0.824),
    (0.780, 0.780, 0.780),
    (0.859, 0.859, 0.553),
    (0.620, 0.855, 0.898),
]


def _normalize_axis(axis: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-8:
        raise RuntimeError(f"Invalid axis with near-zero norm: {axis}")
    return axis / norm


def _asset_to_world_rotation(up: np.ndarray, front: np.ndarray) -> np.ndarray:
    up_axis = _normalize_axis(np.array(up, dtype=np.float32))
    front_axis = _normalize_axis(np.array(front, dtype=np.float32))
    right_axis = _normalize_axis(np.cross(front_axis, up_axis))
    asset_basis = np.stack([right_axis, up_axis, front_axis], axis=1)

    world_right = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    world_front = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    world_basis = np.stack([world_right, world_up, world_front], axis=1)
    return world_basis @ np.linalg.inv(asset_basis)


def _load_world_to_stage_rotation(scene_dataset_config: Optional[Path]) -> np.ndarray:
    if scene_dataset_config is None or not scene_dataset_config.is_file():
        return np.eye(3, dtype=np.float32)
    try:
        config = json.loads(scene_dataset_config.read_text())
        stage_defaults = config.get("stages", {}).get("default_attributes", {})
        up = np.array(stage_defaults.get("up", [0.0, 1.0, 0.0]), dtype=np.float32)
        front = np.array(
            stage_defaults.get("front", [0.0, 0.0, -1.0]), dtype=np.float32
        )
        stage_to_world = _asset_to_world_rotation(up=up, front=front)
        return np.linalg.inv(stage_to_world)
    except Exception:
        return np.eye(3, dtype=np.float32)


def _resolve_scene_name(payload: Dict[str, Any], override: Optional[str]) -> str:
    if override:
        return override

    scene_name = payload.get("scene_name")
    if isinstance(scene_name, str) and scene_name:
        return scene_name

    scene_id = payload.get("scene_id")
    if isinstance(scene_id, str) and scene_id:
        return Path(scene_id).stem

    raise RuntimeError("Unable to determine scene name from valid_instances.json")


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

    scene_folder = (scene_dir / scene_name).resolve()
    if scene_folder.is_dir():
        basis_glbs = sorted(scene_folder.glob("*.basis.glb"))
        if basis_glbs:
            return basis_glbs[0].resolve()
        glbs = sorted(scene_folder.glob("*.glb"))
        if glbs:
            return glbs[0].resolve()

    raise RuntimeError(f"Failed to resolve scene '{scene_name}' under {scene_dir}")


def _apply_rotation(position: List[float], rotation: np.ndarray) -> np.ndarray:
    return rotation @ np.array(position, dtype=np.float32)


def _generate_box(extents: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    sx, sy, sz = [float(e) / 2.0 for e in extents]
    vertices = np.array(
        [
            [-sx, -sy, -sz],
            [sx, -sy, -sz],
            [sx, sy, -sz],
            [-sx, sy, -sz],
            [-sx, -sy, sz],
            [sx, -sy, sz],
            [sx, sy, sz],
            [-sx, sy, sz],
            [-sx, -sy, -sz],
            [-sx, sy, -sz],
            [-sx, sy, sz],
            [-sx, -sy, sz],
            [sx, -sy, -sz],
            [sx, sy, -sz],
            [sx, sy, sz],
            [sx, -sy, sz],
            [-sx, -sy, -sz],
            [sx, -sy, -sz],
            [sx, -sy, sz],
            [-sx, -sy, sz],
            [-sx, sy, -sz],
            [sx, sy, -sz],
            [sx, sy, sz],
            [-sx, sy, sz],
        ],
        dtype=np.float32,
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [8, 9, 10],
            [8, 10, 11],
            [12, 14, 13],
            [12, 15, 14],
            [16, 17, 18],
            [16, 18, 19],
            [20, 22, 21],
            [20, 23, 22],
        ],
        dtype=np.uint32,
    )
    return vertices, faces


def _generate_wireframe_box(
    extents: List[float], radius: float = 0.02, segments: int = 6
) -> Tuple[np.ndarray, np.ndarray]:
    sx, sy, sz = [float(e) / 2.0 for e in extents]
    corners = np.array(
        [
            [-sx, -sy, -sz],
            [sx, -sy, -sz],
            [sx, sy, -sz],
            [-sx, sy, -sz],
            [-sx, -sy, sz],
            [sx, -sy, sz],
            [sx, sy, sz],
            [-sx, sy, sz],
        ],
        dtype=np.float32,
    )
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    all_vertices: List[np.ndarray] = []
    all_faces: List[np.ndarray] = []
    vertex_offset = 0

    for start_idx, end_idx in edges:
        point0 = corners[start_idx]
        point1 = corners[end_idx]
        direction = point1 - point0
        length = float(np.linalg.norm(direction))
        if length < 1e-8:
            continue
        direction = direction / length

        if abs(direction[0]) < 0.9:
            perpendicular = np.cross(direction, np.array([1, 0, 0], dtype=np.float32))
        else:
            perpendicular = np.cross(direction, np.array([0, 1, 0], dtype=np.float32))
        perpendicular = perpendicular / np.linalg.norm(perpendicular)
        perpendicular2 = np.cross(direction, perpendicular)

        angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        ring = np.zeros((segments, 3), dtype=np.float32)
        for ring_index, angle in enumerate(angles):
            ring[ring_index] = (
                np.cos(angle) * perpendicular + np.sin(angle) * perpendicular2
            )
        ring *= radius

        bottom_ring = ring + point0
        top_ring = ring + point1
        cylinder_vertices = np.vstack([bottom_ring, top_ring])

        cylinder_faces: List[List[int]] = []
        for ring_index in range(segments):
            next_index = (ring_index + 1) % segments
            cylinder_faces.append([ring_index, next_index, segments + next_index])
            cylinder_faces.append([ring_index, segments + next_index, segments + ring_index])

        all_vertices.append(cylinder_vertices)
        all_faces.append(np.array(cylinder_faces, dtype=np.uint32) + vertex_offset)
        vertex_offset += len(cylinder_vertices)

    if not all_vertices:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint32)

    return np.vstack(all_vertices), np.vstack(all_faces)


def _append_mesh_to_gltf(
    gltf: pygltflib.GLTF2,
    blob: bytearray,
    vertices: np.ndarray,
    indices: np.ndarray,
    translation: List[float],
    material_idx: int,
    node_name: str,
) -> None:
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    indices = np.ascontiguousarray(indices.flatten(), dtype=np.uint32)

    if len(indices) == 0 or len(vertices) == 0:
        return

    vertex_bytes = vertices.tobytes()
    index_bytes = indices.tobytes()

    while len(blob) % 4 != 0:
        blob.append(0)
    vertex_offset = len(blob)
    blob.extend(vertex_bytes)

    while len(blob) % 4 != 0:
        blob.append(0)
    index_offset = len(blob)
    blob.extend(index_bytes)

    vertex_bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=vertex_offset,
            byteLength=len(vertex_bytes),
            target=pygltflib.ARRAY_BUFFER,
        )
    )
    index_bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(
        pygltflib.BufferView(
            buffer=0,
            byteOffset=index_offset,
            byteLength=len(index_bytes),
            target=pygltflib.ELEMENT_ARRAY_BUFFER,
        )
    )

    vmin = vertices.min(axis=0).tolist()
    vmax = vertices.max(axis=0).tolist()

    vertex_acc_idx = len(gltf.accessors)
    gltf.accessors.append(
        pygltflib.Accessor(
            bufferView=vertex_bv_idx,
            byteOffset=0,
            componentType=pygltflib.FLOAT,
            count=len(vertices),
            type=pygltflib.VEC3,
            min=vmin,
            max=vmax,
        )
    )

    index_acc_idx = len(gltf.accessors)
    gltf.accessors.append(
        pygltflib.Accessor(
            bufferView=index_bv_idx,
            byteOffset=0,
            componentType=pygltflib.UNSIGNED_INT,
            count=len(indices),
            type=pygltflib.SCALAR,
            min=[int(indices.min())],
            max=[int(indices.max())],
        )
    )

    mesh_idx = len(gltf.meshes)
    gltf.meshes.append(
        pygltflib.Mesh(
            name=node_name,
            primitives=[
                pygltflib.Primitive(
                    attributes=pygltflib.Attributes(POSITION=vertex_acc_idx),
                    indices=index_acc_idx,
                    material=material_idx,
                )
            ],
        )
    )

    node_idx = len(gltf.nodes)
    gltf.nodes.append(
        pygltflib.Node(
            name=node_name,
            mesh=mesh_idx,
            translation=translation,
        )
    )

    scene = gltf.scenes[gltf.scene]
    if scene.nodes is None:
        scene.nodes = []
    scene.nodes.append(node_idx)


def _build_category_materials(
    gltf: pygltflib.GLTF2,
    categories: List[str],
    alpha: float,
) -> Dict[str, int]:
    category_to_material: Dict[str, int] = {}
    for category_index, category in enumerate(categories):
        rgb = CATEGORY_PALETTE[category_index % len(CATEGORY_PALETTE)]
        rgba = [rgb[0], rgb[1], rgb[2], float(alpha)]
        material_idx = len(gltf.materials)
        material = pygltflib.Material(
            name=f"instance_{category}",
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                baseColorFactor=rgba,
                metallicFactor=0.0,
                roughnessFactor=0.9,
            ),
            doubleSided=True,
        )
        if alpha < 1.0:
            material.alphaMode = "BLEND"
        gltf.materials.append(material)
        category_to_material[category] = material_idx
    return category_to_material


def _build_wire_materials(
    gltf: pygltflib.GLTF2,
    categories: List[str],
) -> Dict[str, int]:
    category_to_material: Dict[str, int] = {}
    for category_index, category in enumerate(categories):
        rgb = CATEGORY_PALETTE[category_index % len(CATEGORY_PALETTE)]
        rgba = [rgb[0], rgb[1], rgb[2], 1.0]
        material_idx = len(gltf.materials)
        material = pygltflib.Material(
            name=f"instance_{category}_wire",
            pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
                baseColorFactor=rgba,
                metallicFactor=0.0,
                roughnessFactor=0.9,
            ),
            doubleSided=True,
        )
        gltf.materials.append(material)
        category_to_material[category] = material_idx
    return category_to_material


def _validate_vec3(value: Any) -> Optional[List[float]]:
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        return [float(value[0]), float(value[1]), float(value[2])]
    except Exception:
        return None


def _center_and_size(instance_payload: Dict[str, Any]) -> Optional[Tuple[List[float], List[float]]]:
    center = _validate_vec3(instance_payload.get("center"))
    size = _validate_vec3(instance_payload.get("bbox_size"))
    if center is not None and size is not None:
        return center, size

    bbox_min = _validate_vec3(instance_payload.get("bbox_min"))
    bbox_max = _validate_vec3(instance_payload.get("bbox_max"))
    if bbox_min is None or bbox_max is None:
        return None

    center_from_bounds = [
        (bbox_min[0] + bbox_max[0]) / 2.0,
        (bbox_min[1] + bbox_max[1]) / 2.0,
        (bbox_min[2] + bbox_max[2]) / 2.0,
    ]
    size_from_bounds = [
        max(1e-4, bbox_max[0] - bbox_min[0]),
        max(1e-4, bbox_max[1] - bbox_min[1]),
        max(1e-4, bbox_max[2] - bbox_min[2]),
    ]
    return center_from_bounds, size_from_bounds


def _iter_instances(payload: Dict[str, Any]) -> Iterable[Tuple[str, str, Dict[str, Any]]]:
    instances = payload.get("instances")
    if not isinstance(instances, dict):
        return

    for category, category_instances in instances.items():
        if not isinstance(category_instances, dict):
            continue
        for instance_key, instance_payload in category_instances.items():
            if not isinstance(instance_payload, dict):
                continue
            key = str(instance_key)
            yield str(category), key, instance_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render all instances from new-format valid_instances.json into one GLB "
            "with per-category box colors."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("output/QUCTc6BB5sX/valid_instances.json"),
        help="Path to valid_instances.json",
    )
    parser.add_argument(
        "--scene-name",
        type=str,
        default=None,
        help="Optional scene name override",
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=Path("data/scene_datasets/mp3d"),
        help="Scene dataset root directory",
    )
    parser.add_argument(
        "--scene-dataset-config",
        type=Path,
        default=Path("data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"),
        help="Scene dataset config for coordinate transform",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output GLB path (default: <input_dir>/valid_instances_bbox.glb)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Alpha for solid instance boxes",
    )
    parser.add_argument(
        "--wireframe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also render wireframe edges for each instance",
    )
    parser.add_argument(
        "--wire-radius",
        type=float,
        default=0.02,
        help="Cylinder radius for wireframe edges",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of instances to render",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text())

    scene_name = _resolve_scene_name(payload, args.scene_name)
    scene_dir = args.scene_dir.expanduser().resolve()
    scene_path = _resolve_scene_path(scene_dir, scene_name)
    output_path = (
        args.output
        if args.output is not None
        else args.input.parent / "valid_instances_bbox.glb"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rotation = _load_world_to_stage_rotation(args.scene_dataset_config)

    all_instances = list(_iter_instances(payload))
    if args.limit is not None:
        all_instances = all_instances[: args.limit]

    if not all_instances:
        raise RuntimeError("No instances found in input JSON.")

    categories = sorted({category for category, _, _ in all_instances})

    print(f"Loading base scene: {scene_path}")
    gltf = pygltflib.GLTF2.load(str(scene_path))
    blob = bytearray(gltf.binary_blob() or b"")

    solid_materials = _build_category_materials(gltf, categories, args.alpha)
    wire_materials = _build_wire_materials(gltf, categories) if args.wireframe else {}

    skipped = 0
    rendered = 0
    category_counts: Dict[str, int] = {category: 0 for category in categories}

    for category, instance_key, instance_payload in all_instances:
        parsed = _center_and_size(instance_payload)
        if parsed is None:
            skipped += 1
            continue
        center, bbox_size = parsed

        rotated_center = _apply_rotation(center, rotation)

        box_vertices, box_faces = _generate_box(bbox_size)
        _append_mesh_to_gltf(
            gltf=gltf,
            blob=blob,
            vertices=box_vertices,
            indices=box_faces,
            translation=rotated_center.tolist(),
            material_idx=solid_materials[category],
            node_name=f"instance_{instance_key}_box",
        )

        if args.wireframe:
            wire_vertices, wire_faces = _generate_wireframe_box(
                bbox_size, radius=args.wire_radius
            )
            _append_mesh_to_gltf(
                gltf=gltf,
                blob=blob,
                vertices=wire_vertices,
                indices=wire_faces,
                translation=rotated_center.tolist(),
                material_idx=wire_materials[category],
                node_name=f"instance_{instance_key}_wire",
            )

        rendered += 1
        category_counts[category] += 1

    if gltf.buffers:
        gltf.buffers[0].byteLength = len(blob)

    gltf.set_binary_blob(bytes(blob))
    gltf.save(str(output_path))

    print(f"Saved GLB with {rendered} instances to: {output_path}")
    print(f"Categories rendered: {len([c for c, n in category_counts.items() if n > 0])}")
    if skipped > 0:
        print(f"Skipped instances (invalid geometry): {skipped}")

    print("Per-category counts:")
    for category in sorted(category_counts):
        count = category_counts[category]
        if count > 0:
            print(f"  - {category}: {count}")


if __name__ == "__main__":
    main()
