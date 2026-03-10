#!/usr/bin/env python3

"""Export semantic scene overlays for target categories."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def _bootstrap_native_dependencies() -> None:
    """Load known native deps before importing habitat-sim."""
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

import habitat_sim
import numpy as np

try:
    import trimesh
except Exception:
    trimesh = None


@dataclass
class SemanticTarget:
    """Semantic object information for viewer export."""

    semantic_id: int
    category_name: str
    center: np.ndarray
    sizes: np.ndarray

    @property
    def aabb_min(self) -> np.ndarray:
        return self.center - 0.5 * self.sizes

    @property
    def aabb_max(self) -> np.ndarray:
        return self.center + 0.5 * self.sizes

    def to_payload(self) -> Dict[str, Any]:
        return {
            "semantic_id": int(self.semantic_id),
            "category": self.category_name,
            "center": [round(float(v), 5) for v in self.center.tolist()],
            "sizes": [round(float(v), 5) for v in self.sizes.tolist()],
            "aabb_min": [round(float(v), 5) for v in self.aabb_min.tolist()],
            "aabb_max": [round(float(v), 5) for v in self.aabb_max.tolist()],
        }


def _normalize_axis(axis: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-8:
        raise RuntimeError(f"Invalid axis with near-zero norm: {axis}")
    return axis / norm


def _asset_to_world_rotation(up: np.ndarray, front: np.ndarray) -> np.ndarray:
    """Build the Habitat stage asset-to-world rotation from config axes."""
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
    """Load world-to-stage rotation for overlaying Habitat coords onto raw stage GLB."""
    if scene_dataset_config is None or not scene_dataset_config.is_file():
        return np.eye(3, dtype=np.float32)

    try:
        config = json.loads(scene_dataset_config.read_text())
        stage_defaults = config.get("stages", {}).get("default_attributes", {})
        up = np.array(stage_defaults.get("up", [0.0, 1.0, 0.0]), dtype=np.float32)
        front = np.array(stage_defaults.get("front", [0.0, 0.0, -1.0]), dtype=np.float32)
        stage_to_world = _asset_to_world_rotation(up=up, front=front)
        return np.linalg.inv(stage_to_world)
    except Exception:
        return np.eye(3, dtype=np.float32)


def _slugify(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "target"


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


def _build_semantic_sim(
    scene_path: Path,
    scene_dataset_config: Optional[Path],
) -> habitat_sim.Simulator:
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene_path)
    sim_cfg.enable_physics = False
    if scene_dataset_config is not None:
        sim_cfg.scene_dataset_config_file = str(scene_dataset_config.resolve())

    agent_cfg = habitat_sim.AgentConfiguration()
    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


class SceneOverlayExporter:
    """Export HTML and GLB overlays for semantic targets."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.logger = logging.getLogger(self.__class__.__name__)

        self.scene_dir = args.scene_dir.expanduser().resolve()
        self.scene_path = _resolve_scene_path(self.scene_dir, args.scene_name)
        self.scene_name = args.scene_name
        self.target_category = args.target_category.strip()
        self.target_category_lc = self.target_category.lower()

        self.output_root = args.output_root.expanduser().resolve()
        self.scene_output_dir = self.output_root / self.scene_name
        self.overlay_dir = self.scene_output_dir / f"{_slugify(self.target_category)}_scene_overlay"
        self.overlay_dir.mkdir(parents=True, exist_ok=True)

        dataset_config = (
            args.scene_dataset_config.expanduser().resolve()
            if args.scene_dataset_config is not None
            else None
        )
        self.scene_dataset_config = dataset_config
        self.world_to_stage_rotation = _load_world_to_stage_rotation(dataset_config)
        self.sim = _build_semantic_sim(self.scene_path, dataset_config)
        self.inner_sim = getattr(self.sim, "_sim", self.sim)

    def close(self) -> None:
        if self.sim is not None:
            self.sim.close()

    def _semantic_objects(self) -> List[Any]:
        semantic_scene = getattr(self.inner_sim, "semantic_scene", None)
        if semantic_scene is None:
            return []
        try:
            return list(semantic_scene.objects)
        except Exception:
            return []

    def _safe_semantic_id(self, obj: Any) -> Optional[int]:
        for attr_name in ("semantic_id", "semanticID"):
            value = getattr(obj, attr_name, None)
            if value is None:
                continue
            try:
                return int(value)
            except Exception:
                continue
        return None

    def _safe_category_name(self, obj: Any) -> str:
        category = getattr(obj, "category", None)
        if category is None:
            return ""
        try:
            return str(category.name())
        except Exception:
            return ""

    def _collect_targets(self) -> List[SemanticTarget]:
        targets: List[SemanticTarget] = []
        seen_ids: set[int] = set()
        for obj in self._semantic_objects():
            semantic_id = self._safe_semantic_id(obj)
            if semantic_id is None or semantic_id in seen_ids:
                continue

            category_name = self._safe_category_name(obj)
            if self.target_category_lc not in category_name.strip().lower():
                continue

            aabb = getattr(obj, "aabb", None)
            if aabb is None:
                continue

            center = np.array(aabb.center, dtype=np.float32)
            sizes = np.array(aabb.sizes, dtype=np.float32)
            if sizes.size < 3 or not np.all(np.isfinite(sizes)):
                continue

            targets.append(
                SemanticTarget(
                    semantic_id=int(semantic_id),
                    category_name=category_name,
                    center=center,
                    sizes=sizes,
                )
            )
            seen_ids.add(int(semantic_id))

        targets.sort(key=lambda item: item.semantic_id)
        return targets

    def _transform_target_to_stage(self, target: SemanticTarget) -> SemanticTarget:
        """Map Habitat world-space target geometry into raw stage GLB coordinates."""
        rotation = np.array(self.world_to_stage_rotation, dtype=np.float32)
        transformed_center = rotation @ np.array(target.center, dtype=np.float32)
        transformed_sizes = np.abs(rotation) @ np.array(target.sizes, dtype=np.float32)
        return SemanticTarget(
            semantic_id=target.semantic_id,
            category_name=target.category_name,
            center=transformed_center,
            sizes=transformed_sizes,
        )

    def _build_payload(self, targets: List[SemanticTarget]) -> Dict[str, Any]:
        viewer_html = self.overlay_dir / "viewer.html"
        scene_rel = os.path.relpath(self.scene_path, start=viewer_html.parent)
        overlay_targets = [self._transform_target_to_stage(target) for target in targets]
        return {
            "scene_name": self.scene_name,
            "scene_glb": Path(scene_rel).as_posix(),
            "scene_glb_absolute": str(self.scene_path),
            "target_category": self.target_category,
            "num_targets": len(targets),
            "world_to_stage_rotation": [
                [round(float(v), 5) for v in row]
                for row in self.world_to_stage_rotation.tolist()
            ],
            "targets": [
                {
                    **overlay_target.to_payload(),
                    "world_center": [round(float(v), 5) for v in target.center.tolist()],
                    "world_sizes": [round(float(v), 5) for v in target.sizes.tolist()],
                }
                for target, overlay_target in zip(targets, overlay_targets)
            ],
        }

    def _write_payload(self, payload: Dict[str, Any]) -> Path:
        out_path = self.overlay_dir / "targets.json"
        with out_path.open("w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, indent=2, ensure_ascii=False)
        return out_path

    def _build_overlay_scene(self, targets: List[SemanticTarget]):
        """Build a trimesh scene containing only highlight geometry."""
        if trimesh is None:
            return None

        scene = trimesh.Scene()
        for target in [self._transform_target_to_stage(target) for target in targets]:
            box_transform = trimesh.transformations.translation_matrix(target.center)
            box_mesh = trimesh.creation.box(extents=target.sizes, transform=box_transform)
            box_mesh.visual.face_colors = np.tile(
                np.array([255, 215, 0, 110], dtype=np.uint8),
                (len(box_mesh.faces), 1),
            )
            scene.add_geometry(
                box_mesh,
                geom_name=f"target_box_{target.semantic_id}",
                node_name=f"target_box_{target.semantic_id}",
            )

            center_radius = max(0.03, float(np.min(target.sizes)) * 0.08)
            center_mesh = trimesh.creation.icosphere(subdivisions=2, radius=center_radius)
            center_mesh.apply_translation(target.center)
            center_mesh.visual.face_colors = np.tile(
                np.array([255, 48, 48, 255], dtype=np.uint8),
                (len(center_mesh.faces), 1),
            )
            scene.add_geometry(
                center_mesh,
                geom_name=f"target_center_{target.semantic_id}",
                node_name=f"target_center_{target.semantic_id}",
            )
        return scene

    def _export_overlay_glb(self, targets: List[SemanticTarget]) -> Optional[Path]:
        """Export a standalone GLB containing only overlay highlight geometry."""
        if trimesh is None:
            self.logger.warning("`trimesh` is unavailable; skipping GLB overlay export.")
            return None

        scene = self._build_overlay_scene(targets)
        if scene is None:
            return None

        out_path = self.overlay_dir / f"{_slugify(self.target_category)}_highlight_overlay.glb"
        glb_bytes = scene.export(file_type="glb")
        out_path.write_bytes(glb_bytes)
        return out_path

    def _viewer_html(self, payload: Dict[str, Any]) -> str:
        template = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Semantic Scene Overlay</title>
  <style>
    html, body { margin: 0; height: 100%; overflow: hidden; background: #101114; color: #f3f4f6; font-family: Inter, Arial, sans-serif; }
    #app { display: grid; grid-template-columns: 320px 1fr; height: 100%; }
    #sidebar { padding: 16px; border-right: 1px solid #27272a; overflow-y: auto; background: #17181c; }
    #viewer { position: relative; }
    #canvas-wrap { position: absolute; inset: 0; }
    h1 { font-size: 18px; margin: 0 0 8px; }
    .meta { font-size: 13px; color: #c4c7cf; margin-bottom: 12px; line-height: 1.5; }
    .hint { font-size: 12px; color: #9ca3af; margin-bottom: 14px; }
    .target-item { padding: 10px 12px; margin-bottom: 8px; border: 1px solid #30333a; border-radius: 10px; cursor: pointer; background: #1f2128; }
    .target-item:hover { background: #252833; }
    .target-item.active { border-color: #facc15; box-shadow: 0 0 0 1px #facc15 inset; }
    .target-title { font-size: 14px; font-weight: 600; }
    .target-sub { font-size: 12px; color: #9ca3af; margin-top: 4px; }
    .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; background: #312e81; color: #e0e7ff; font-size: 12px; margin-left: 8px; }
    #status { position: absolute; left: 12px; bottom: 12px; padding: 8px 10px; border-radius: 8px; background: rgba(15, 23, 42, 0.78); color: #e5e7eb; font-size: 12px; }
    a { color: #93c5fd; }
  </style>
</head>
<body>
  <div id="app">
    <div id="sidebar">
      <h1>Semantic Scene Overlay</h1>
      <div class="meta">
        <div><strong>Scene:</strong> __SCENE_NAME__</div>
        <div><strong>Target category:</strong> __TARGET_CATEGORY__</div>
        <div><strong>Matched objects:</strong> __TARGET_COUNT__</div>
      </div>
      <div class="hint">
        Use mouse drag to orbit, wheel to zoom, right-drag to pan.<br/>
        Click a target below to center the camera on it.
      </div>
      <div id="target-list"></div>
    </div>
    <div id="viewer">
      <div id="canvas-wrap"></div>
      <div id="status">Loading scene...</div>
    </div>
  </div>

  <script type="module">
    import * as THREE from 'https://unpkg.com/three@0.160.0/build/three.module.js';
    import { OrbitControls } from 'https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js';
    import { GLTFLoader } from 'https://unpkg.com/three@0.160.0/examples/jsm/loaders/GLTFLoader.js';

    const payload = __PAYLOAD_JSON__;
    const targets = payload.targets;
    const container = document.getElementById('canvas-wrap');
    const status = document.getElementById('status');
    const targetList = document.getElementById('target-list');

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0f1115);

    const camera = new THREE.PerspectiveCamera(60, container.clientWidth / container.clientHeight, 0.05, 2000);
    camera.position.set(0, 3, 8);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.target.set(0, 1.2, 0);

    scene.add(new THREE.AmbientLight(0xffffff, 1.35));
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.6);
    keyLight.position.set(4, 8, 6);
    scene.add(keyLight);

    const axesHelper = new THREE.AxesHelper(1.0);
    scene.add(axesHelper);

    const markerGroups = [];
    let activeSemanticId = null;

    function updateStatus(text) {
      status.textContent = text;
    }

    function makeLabelSprite(text) {
      const canvas = document.createElement('canvas');
      canvas.width = 512;
      canvas.height = 128;
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = 'rgba(14, 18, 28, 0.82)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.strokeStyle = 'rgba(250, 204, 21, 0.95)';
      ctx.lineWidth = 4;
      ctx.strokeRect(2, 2, canvas.width - 4, canvas.height - 4);
      ctx.fillStyle = '#f8fafc';
      ctx.font = 'bold 42px sans-serif';
      ctx.fillText(text, 24, 78);
      const texture = new THREE.CanvasTexture(canvas);
      const material = new THREE.SpriteMaterial({ map: texture, depthTest: false });
      const sprite = new THREE.Sprite(material);
      sprite.scale.set(1.6, 0.4, 1.0);
      return sprite;
    }

    function markerColor(isActive) {
      return isActive ? 0xff3b30 : 0xfacc15;
    }

    function setMarkerAppearance(group, isActive) {
      const fillMesh = group.userData.fillMesh;
      const edgeLines = group.userData.edgeLines;
      const centerSphere = group.userData.centerSphere;
      const label = group.userData.label;
      const color = markerColor(isActive);

      fillMesh.material.color.setHex(color);
      fillMesh.material.opacity = isActive ? 0.34 : 0.18;
      edgeLines.material.color.setHex(color);
      centerSphere.material.color.setHex(isActive ? 0xff2d55 : 0xff0000);
      label.material.opacity = isActive ? 1.0 : 0.88;
    }

    function frameTarget(group) {
      const target = group.userData.target;
      activeSemanticId = target.semantic_id;

      for (const other of markerGroups) {
        const isActive = other.userData.target.semantic_id === activeSemanticId;
        setMarkerAppearance(other, isActive);
      }

      const center = new THREE.Vector3(...target.center);
      const size = new THREE.Vector3(...target.sizes);
      const radius = Math.max(size.x, size.y, size.z, 0.5);
      controls.target.copy(center);
      camera.position.copy(center.clone().add(new THREE.Vector3(radius * 1.6, radius * 1.2, radius * 1.6)));
      controls.update();

      document.querySelectorAll('.target-item').forEach((element) => {
        element.classList.toggle('active', Number(element.dataset.semanticId) === activeSemanticId);
      });
      updateStatus(`Focused target ${target.semantic_id} (${target.category})`);
    }

    function buildMarker(target) {
      const group = new THREE.Group();
      group.userData.target = target;

      const boxGeometry = new THREE.BoxGeometry(target.sizes[0], target.sizes[1], target.sizes[2]);
      const fillMesh = new THREE.Mesh(
        boxGeometry,
        new THREE.MeshBasicMaterial({ color: markerColor(false), transparent: true, opacity: 0.18 })
      );
      fillMesh.position.set(...target.center);
      group.add(fillMesh);

      const edgeLines = new THREE.LineSegments(
        new THREE.EdgesGeometry(boxGeometry),
        new THREE.LineBasicMaterial({ color: markerColor(false) })
      );
      edgeLines.position.copy(fillMesh.position);
      group.add(edgeLines);

      const sphereRadius = Math.max(Math.min(...target.sizes) * 0.08, 0.04);
      const centerSphere = new THREE.Mesh(
        new THREE.SphereGeometry(sphereRadius, 18, 18),
        new THREE.MeshBasicMaterial({ color: 0xff0000 })
      );
      centerSphere.position.set(...target.center);
      group.add(centerSphere);

      const label = makeLabelSprite(`${target.semantic_id} · ${target.category}`);
      label.position.set(target.center[0], target.center[1] + target.sizes[1] * 0.65 + 0.1, target.center[2]);
      group.add(label);

      group.userData.fillMesh = fillMesh;
      group.userData.edgeLines = edgeLines;
      group.userData.centerSphere = centerSphere;
      group.userData.label = label;
      return group;
    }

    function populateSidebar() {
      if (targets.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'target-sub';
        empty.textContent = 'No matching semantic objects were found in this scene.';
        targetList.appendChild(empty);
        return;
      }

      for (const target of targets) {
        const item = document.createElement('div');
        item.className = 'target-item';
        item.dataset.semanticId = String(target.semantic_id);
        item.innerHTML = `
          <div class="target-title">${target.semantic_id}<span class="badge">${target.category}</span></div>
          <div class="target-sub">center = [${target.center.map(v => v.toFixed(2)).join(', ')}]</div>
          <div class="target-sub">size = [${target.sizes.map(v => v.toFixed(2)).join(', ')}]</div>
        `;
        item.addEventListener('click', () => {
          const group = markerGroups.find((marker) => marker.userData.target.semantic_id === target.semantic_id);
          if (group) {
            frameTarget(group);
          }
        });
        targetList.appendChild(item);
      }
    }

    function fitInitialCamera() {
      if (targets.length === 0) {
        camera.position.set(0, 4, 8);
        controls.target.set(0, 1, 0);
        controls.update();
        return;
      }

      const box = new THREE.Box3();
      for (const target of targets) {
        const half = new THREE.Vector3(target.sizes[0] / 2, target.sizes[1] / 2, target.sizes[2] / 2);
        box.expandByPoint(new THREE.Vector3(...target.center).sub(half));
        box.expandByPoint(new THREE.Vector3(...target.center).add(half));
      }

      const center = box.getCenter(new THREE.Vector3());
      const size = box.getSize(new THREE.Vector3());
      const radius = Math.max(size.x, size.y, size.z, 2.0);
      controls.target.copy(center);
      camera.position.copy(center.clone().add(new THREE.Vector3(radius * 0.9, radius * 0.7, radius * 1.1)));
      controls.update();
    }

    const sceneLoader = new GLTFLoader();
    sceneLoader.load(
      payload.scene_glb,
      (gltf) => {
        scene.add(gltf.scene);
        for (const target of targets) {
          const marker = buildMarker(target);
          markerGroups.push(marker);
          scene.add(marker);
        }
        populateSidebar();
        fitInitialCamera();
        if (markerGroups.length > 0) {
          frameTarget(markerGroups[0]);
        } else {
          updateStatus('Scene loaded. No matching targets found.');
        }
      },
      undefined,
      (error) => {
        console.error(error);
        updateStatus('Failed to load scene. Serve this file with `python -m http.server` from the repo root.');
      }
    );

    function onResize() {
      const width = container.clientWidth;
      const height = container.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    }
    window.addEventListener('resize', onResize);

    renderer.setAnimationLoop(() => {
      controls.update();
      renderer.render(scene, camera);
    });
  </script>
</body>
</html>
"""
        html = template.replace("__SCENE_NAME__", self.scene_name)
        html = html.replace("__TARGET_CATEGORY__", self.target_category)
        html = html.replace("__TARGET_COUNT__", str(payload["num_targets"]))
        html = html.replace("__PAYLOAD_JSON__", json.dumps(payload, ensure_ascii=False))
        return html

    def _write_viewer(self, payload: Dict[str, Any]) -> Path:
        out_path = self.overlay_dir / "viewer.html"
        out_path.write_text(self._viewer_html(payload), encoding="utf-8")
        return out_path

    def run(self) -> Dict[str, Path]:
        targets = self._collect_targets()
        payload = self._build_payload(targets)
        json_path = self._write_payload(payload)
        html_path = self._write_viewer(payload)
        glb_path = self._export_overlay_glb(targets)
        self.logger.info(
            "Exported overlay viewer for %d '%s' objects: %s",
            len(targets),
            self.target_category,
            html_path,
        )
        outputs: Dict[str, Path] = {
            "json": json_path,
            "html": html_path,
        }
        if glb_path is not None:
            outputs["glb"] = glb_path
        return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export an HTML scene viewer that loads the original glb scene and "
            "highlights semantic objects whose category matches --target-category."
        )
    )
    parser.add_argument("scene_name", type=str, help="Scene id or .glb path")
    parser.add_argument(
        "--target-category",
        type=str,
        default="chair",
        help="Semantic category substring to highlight",
    )
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=Path("data/scene_datasets/mp3d"),
        help="Root directory containing scene assets",
    )
    parser.add_argument(
        "--scene-dataset-config",
        type=Path,
        default=Path("data/scene_datasets/mp3d/mp3d.scene_dataset_config.json"),
        help="Path to the scene_dataset_config JSON",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Root output directory",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Python logging level",
    )
    return parser.parse_args()


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    args = parse_args()
    configure_logging(args.log_level)

    exporter = SceneOverlayExporter(args)
    try:
        outputs = exporter.run()
        print(f"Overlay JSON saved at: {outputs['json']}")
        print(f"Overlay viewer saved at: {outputs['html']}")
        if 'glb' in outputs:
            print(f"Overlay GLB saved at: {outputs['glb']}")
            print("Open the original scene GLB and this overlay GLB together in Blender or another GLB viewer.")
        print("Serve the repo root with `python -m http.server` and open the viewer in your browser.")
    finally:
        exporter.close()


if __name__ == "__main__":
    main()
