#!/usr/bin/env python3

"""Export per-episode GLB scenes with start sphere and goal bounding boxes.

Uses pygltflib to directly manipulate the glTF structure, preserving all
original textures, materials, and scene graph from the source GLB.
"""

from __future__ import annotations

import argparse
import copy
import json
import struct
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pygltflib

# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

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


def _resolve_scene_name(payload: Dict[str, Any], override: Optional[str]) -> str:
    if override:
        return override
    episodes = payload.get("episodes", [])
    if episodes:
        first = episodes[0]
        if isinstance(first, dict):
            scene_name = first.get("scene_name")
            if isinstance(scene_name, str) and scene_name:
                return scene_name
            scene_id = first.get("scene_id")
            if isinstance(scene_id, str) and scene_id:
                return Path(scene_id).stem
    raise RuntimeError("Unable to determine scene name from JSON.")


def _apply_rotation(position: List[float], rotation: np.ndarray) -> np.ndarray:
    return rotation @ np.array(position, dtype=np.float32)


# ---------------------------------------------------------------------------
# Geometry generation (raw arrays for glTF injection)
# ---------------------------------------------------------------------------

def _generate_icosphere(subdivisions: int = 2, radius: float = 0.25) -> Tuple[np.ndarray, np.ndarray]:
    """Generate icosphere vertices and triangle indices."""
    # Golden ratio
    t = (1.0 + np.sqrt(5.0)) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=np.float32)
    faces = np.array([
        [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
        [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
        [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
        [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
    ], dtype=np.uint32)

    for _ in range(subdivisions):
        edge_midpoints = {}
        new_faces = []
        new_verts = list(verts)

        def get_mid(i, j):
            key = (min(i, j), max(i, j))
            if key in edge_midpoints:
                return edge_midpoints[key]
            mid = (new_verts[i] + new_verts[j]) / 2.0
            idx = len(new_verts)
            new_verts.append(mid)
            edge_midpoints[key] = idx
            return idx

        for f in faces:
            a, b, c = int(f[0]), int(f[1]), int(f[2])
            ab = get_mid(a, b)
            bc = get_mid(b, c)
            ca = get_mid(c, a)
            new_faces.extend([
                [a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]
            ])
        verts = np.array(new_verts, dtype=np.float32)
        faces = np.array(new_faces, dtype=np.uint32)

    # Normalize to sphere
    norms = np.linalg.norm(verts, axis=1, keepdims=True)
    verts = verts / norms * radius
    return verts, faces


def _generate_box(extents: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    """Generate box vertices and triangle indices with proper normals (24 verts)."""
    sx, sy, sz = [e / 2.0 for e in extents]
    # 24 vertices (4 per face, 6 faces) for proper flat shading
    verts = np.array([
        # -Z face
        [-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
        # +Z face
        [-sx, -sy, sz], [sx, -sy, sz], [sx, sy, sz], [-sx, sy, sz],
        # -X face
        [-sx, -sy, -sz], [-sx, sy, -sz], [-sx, sy, sz], [-sx, -sy, sz],
        # +X face
        [sx, -sy, -sz], [sx, sy, -sz], [sx, sy, sz], [sx, -sy, sz],
        # -Y face
        [-sx, -sy, -sz], [sx, -sy, -sz], [sx, -sy, sz], [-sx, -sy, sz],
        # +Y face
        [-sx, sy, -sz], [sx, sy, -sz], [sx, sy, sz], [-sx, sy, sz],
    ], dtype=np.float32)
    faces = np.array([
        [0,2,1],[0,3,2],     # -Z
        [4,5,6],[4,6,7],     # +Z
        [8,9,10],[8,10,11],  # -X
        [12,14,13],[12,15,14], # +X
        [16,17,18],[16,18,19], # -Y
        [20,22,21],[20,23,22], # +Y
    ], dtype=np.uint32)
    return verts, faces


def _generate_wireframe_box(extents: List[float], radius: float = 0.02, segments: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    """Generate wireframe box edges as cylinders."""
    sx, sy, sz = [e / 2.0 for e in extents]
    corners = np.array([
        [-sx, -sy, -sz], [sx, -sy, -sz], [sx, sy, -sz], [-sx, sy, -sz],
        [-sx, -sy,  sz], [sx, -sy,  sz], [sx, sy,  sz], [-sx, sy,  sz],
    ], dtype=np.float32)
    edges = [
        (0,1),(1,2),(2,3),(3,0),
        (4,5),(5,6),(6,7),(7,4),
        (0,4),(1,5),(2,6),(3,7),
    ]

    all_verts = []
    all_faces = []
    vert_offset = 0
    for i, j in edges:
        p0, p1 = corners[i], corners[j]
        direction = p1 - p0
        length = float(np.linalg.norm(direction))
        if length < 1e-8:
            continue
        d = direction / length

        # Build a perpendicular basis
        if abs(d[0]) < 0.9:
            perp = np.cross(d, np.array([1, 0, 0], dtype=np.float32))
        else:
            perp = np.cross(d, np.array([0, 1, 0], dtype=np.float32))
        perp = perp / np.linalg.norm(perp)
        perp2 = np.cross(d, perp)

        # Cylinder vertices: 2 rings of `segments` vertices
        angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        ring = np.zeros((segments, 3), dtype=np.float32)
        for k, a in enumerate(angles):
            ring[k] = np.cos(a) * perp + np.sin(a) * perp2
        ring *= radius

        bottom_ring = ring + p0
        top_ring = ring + p1
        cyl_verts = np.vstack([bottom_ring, top_ring])

        cyl_faces = []
        for k in range(segments):
            k_next = (k + 1) % segments
            # Two triangles per quad
            cyl_faces.append([k, k_next, segments + k_next])
            cyl_faces.append([k, segments + k_next, segments + k])

        all_verts.append(cyl_verts)
        all_faces.append(np.array(cyl_faces, dtype=np.uint32) + vert_offset)
        vert_offset += len(cyl_verts)

    if not all_verts:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint32)

    return np.vstack(all_verts), np.vstack(all_faces)


# ---------------------------------------------------------------------------
# glTF injection helpers
# ---------------------------------------------------------------------------
GOAL_COLORS_RGBA = [                                                                                  
   [0.13, 0.77, 0.37, 0.63],   # green                                                                 
#    [0.23, 0.51, 0.96, 0.63],   # blue                                                                  
#    [0.92, 0.70, 0.03, 0.63],   # yellow                                                                
#    [0.66, 0.33, 0.97, 0.63],   # purple                                                                
]   
# GOAL_COLOR_RGBA = [0.13, 0.77, 0.37, 0.63]   # green

START_COLOR_RGBA = [1.0, 0.23, 0.19, 1.0]  # red


def _append_mesh_to_gltf(
    gltf: pygltflib.GLTF2,
    blob: bytearray,
    vertices: np.ndarray,
    indices: np.ndarray,
    translation: List[float],
    color_rgba: List[float],
    node_name: str,
    parent_node_idx: Optional[int] = None,
) -> None:
    """Append a colored mesh (vertices + indices) to the glTF, adding it to the scene."""
    buf_idx = 0  # We work with the single binary blob

    # Ensure vertices are float32, indices are uint32
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    indices = np.ascontiguousarray(indices.flatten(), dtype=np.uint32)

    vert_bytes = vertices.tobytes()
    idx_bytes = indices.tobytes()

    # Pad blob to 4-byte alignment before appending
    while len(blob) % 4 != 0:
        blob.append(0)
    vert_offset = len(blob)
    blob.extend(vert_bytes)

    while len(blob) % 4 != 0:
        blob.append(0)
    idx_offset = len(blob)
    blob.extend(idx_bytes)

    # BufferViews
    vert_bv_idx = len(gltf.bufferViews)
    gltf.bufferViews.append(pygltflib.BufferView(
        buffer=buf_idx,
        byteOffset=vert_offset,
        byteLength=len(vert_bytes),
        target=pygltflib.ARRAY_BUFFER,
    ))
    idx_bv_idx = vert_bv_idx + 1
    gltf.bufferViews.append(pygltflib.BufferView(
        buffer=buf_idx,
        byteOffset=idx_offset,
        byteLength=len(idx_bytes),
        target=pygltflib.ELEMENT_ARRAY_BUFFER,
    ))

    # Compute bounds
    vmin = vertices.min(axis=0).tolist()
    vmax = vertices.max(axis=0).tolist()

    # Accessors
    vert_acc_idx = len(gltf.accessors)
    gltf.accessors.append(pygltflib.Accessor(
        bufferView=vert_bv_idx,
        byteOffset=0,
        componentType=pygltflib.FLOAT,
        count=len(vertices),
        type=pygltflib.VEC3,
        max=vmax,
        min=vmin,
    ))
    idx_acc_idx = vert_acc_idx + 1
    gltf.accessors.append(pygltflib.Accessor(
        bufferView=idx_bv_idx,
        byteOffset=0,
        componentType=pygltflib.UNSIGNED_INT,
        count=len(indices),
        type=pygltflib.SCALAR,
        max=[int(indices.max())],
        min=[int(indices.min())],
    ))

    # Material (unlit colored)
    is_transparent = color_rgba[3] < 1.0
    mat_idx = len(gltf.materials)
    mat = pygltflib.Material(
        name=f"mat_{node_name}",
        pbrMetallicRoughness=pygltflib.PbrMetallicRoughness(
            baseColorFactor=color_rgba,
            metallicFactor=0.0,
            roughnessFactor=0.9,
        ),
        doubleSided=True,
    )
    if is_transparent:
        mat.alphaMode = "BLEND"
    gltf.materials.append(mat)

    # Mesh
    mesh_idx = len(gltf.meshes)
    gltf.meshes.append(pygltflib.Mesh(
        name=node_name,
        primitives=[pygltflib.Primitive(
            attributes=pygltflib.Attributes(POSITION=vert_acc_idx),
            indices=idx_acc_idx,
            material=mat_idx,
        )],
    ))

    # Node
    node_idx = len(gltf.nodes)
    gltf.nodes.append(pygltflib.Node(
        name=node_name,
        mesh=mesh_idx,
        translation=translation,
    ))

    # Add node to scene's root or parent
    if parent_node_idx is not None:
        if gltf.nodes[parent_node_idx].children is None:
            gltf.nodes[parent_node_idx].children = []
        gltf.nodes[parent_node_idx].children.append(node_idx)
    else:
        scene = gltf.scenes[gltf.scene]
        if scene.nodes is None:
            scene.nodes = []
        scene.nodes.append(node_idx)


# ---------------------------------------------------------------------------
# Episode processing
# ---------------------------------------------------------------------------

def _add_episode_markers(
    gltf: pygltflib.GLTF2,
    blob: bytearray,
    episode: Dict[str, Any],
    rotation: np.ndarray,
) -> None:
    """Add start sphere and goal bounding boxes to the glTF."""
    # Start position - red sphere
    start_state = episode.get("start_state", {})
    start_pos = start_state.get("position")
    if isinstance(start_pos, list) and len(start_pos) == 3:
        pos = _apply_rotation(start_pos, rotation)
        verts, faces = _generate_icosphere(subdivisions=2, radius=0.25)
        _append_mesh_to_gltf(
            gltf, blob, verts, faces,
            translation=pos.tolist(),
            color_rgba=START_COLOR_RGBA,
            node_name="start_marker",
        )

    # Goals - bounding boxes
    goals = episode.get("goals", [])
    for idx, goal in enumerate(goals):
        goal_state = goal.get("goal_state", {})
        center = goal_state.get("object_center")
        bbox_size = goal_state.get("bbox_size")
        if not (isinstance(center, list) and len(center) == 3):
            continue
        if not (isinstance(bbox_size, list) and len(bbox_size) == 3):
            continue

        pos = _apply_rotation(center, rotation)
        category = goal.get("category", f"goal_{idx}")
        color = GOAL_COLORS_RGBA[idx % len(GOAL_COLORS_RGBA)]
        # color = GOAL_COLOR_RGBA
        wire_color = color[:3] + [1.0]  # opaque wireframe

        # Semi-transparent solid box
        box_verts, box_faces = _generate_box(bbox_size)
        _append_mesh_to_gltf(
            gltf, blob, box_verts, box_faces,
            translation=pos.tolist(),
            color_rgba=color,
            node_name=f"goal_{idx}_{category}_box",
        )

        # Wireframe edges
        wire_verts, wire_faces = _generate_wireframe_box(bbox_size)
        if len(wire_verts) > 0:
            _append_mesh_to_gltf(
                gltf, blob, wire_verts, wire_faces,
                translation=pos.tolist(),
                color_rgba=wire_color,
                node_name=f"goal_{idx}_{category}_wire",
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export GLB scenes with start sphere and goal bounding boxes."
    )
    parser.add_argument(
        "--input", type=Path,
        default=Path("output/QUCTc6BB5sX/multimodal_episodes.json"),
    )
    parser.add_argument("--scene-name", type=str, default=None)
    parser.add_argument(
        "--scene-dir", type=Path,
        default=Path("data/scene_datasets/mp3d"),
    )
    parser.add_argument(
        "--scene-dataset-config", type=Path,
        default=Path("data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text())
    episodes = payload.get("episodes", [])
    if not isinstance(episodes, list) or not episodes:
        raise RuntimeError("No episodes found in input JSON.")

    scene_name = _resolve_scene_name(payload, args.scene_name)
    scene_dir = args.scene_dir.expanduser().resolve()
    scene_path = _resolve_scene_path(scene_dir, scene_name)
    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else args.input.parent / "episode_bbox_glb"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    rotation = _load_world_to_stage_rotation(args.scene_dataset_config)

    print(f"Loading base scene: {scene_path}")
    base_gltf = pygltflib.GLTF2.load(str(scene_path))
    # Extract binary blob
    base_blob = bytearray(base_gltf.binary_blob() or b"")
    print(f"Base scene loaded: {len(base_gltf.meshes)} meshes, "
          f"{len(base_gltf.materials)} materials, "
          f"blob size {len(base_blob)} bytes")

    limit = args.limit if args.limit is not None else len(episodes)
    rendered = 0
    for idx, episode in enumerate(episodes[:limit]):
        if not isinstance(episode, dict):
            continue
        episode_id = episode.get("episode_id", f"episode_{idx:06d}")

        # Deep copy the glTF structure and blob for this episode
        gltf = copy.deepcopy(base_gltf)
        blob = bytearray(base_blob)

        _add_episode_markers(gltf, blob, episode, rotation)

        # Update buffer size to match the expanded blob
        if gltf.buffers:
            gltf.buffers[0].byteLength = len(blob)

        # Save
        out_path = output_dir / f"{episode_id}.glb"
        gltf.set_binary_blob(bytes(blob))
        gltf.save(str(out_path))

        rendered += 1
        print(f"[{rendered}/{min(limit, len(episodes))}] Saved: {out_path}")

    print(f"\nExported {rendered} episode GLBs to {output_dir}")


if __name__ == "__main__":
    main()
