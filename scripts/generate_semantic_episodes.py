#!/usr/bin/env python3

import argparse
import gzip
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    # Workaround for habitat-sim invalid pointer issues on some systems.
    import quaternion  # noqa: F401
except Exception:
    pass

import habitat
import habitat_sim
import soundspaces  # noqa: F401
from ss_baselines.omni_long.config.default import get_task_config
import numpy as np
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from soundspaces.tasks.shortest_path_follower import ShortestPathFollower


def _load_sound_map(sounds_dir: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not sounds_dir.exists():
        return mapping
    for wav in sounds_dir.glob("*.wav"):
        key = wav.stem
        mapping[key] = str(wav)
    return mapping


def _resolve_scene_path(scene_dir: Path, scene: str) -> Path:
    # If scene is already a .glb path (absolute or relative to scene_dir)
    p = Path(scene)
    if p.suffix.endswith("glb"):
        if p.is_absolute() and p.is_file():
            return p
        candidate = scene_dir / p
        if candidate.is_file():
            return candidate
    # Default mp3d layout: scene_dir/scene/scene.glb
    candidate = scene_dir / scene / f"{scene}.glb"
    if candidate.is_file():
        return candidate
    # HM3D layout: scene_dir/scene/*.basis.glb
    scene_folder = scene_dir / scene
    if scene_folder.is_dir():
        glbs = sorted(scene_folder.glob("*.basis.glb"))
        if not glbs:
            glbs = sorted(scene_folder.glob("*.glb"))
        if len(glbs) == 1:
            return glbs[0]
        if len(glbs) > 1:
            basis = [g for g in glbs if g.name.endswith(".basis.glb")]
            if len(basis) == 1:
                return basis[0]
    raise RuntimeError(f"Unable to resolve scene glb for scene={scene} under {scene_dir}")


def _scene_id_from_path(
    scene_path: Path, scene_dir: Path, scene_name: Optional[str] = None
) -> str:
    try:
        rel = scene_path.resolve().relative_to(scene_dir.resolve())
        return rel.as_posix()
    except Exception:
        pass
    if scene_name:
        return f"{scene_name}/{scene_path.name}"
    parent = scene_path.parent.name
    if parent:
        return f"{parent}/{scene_path.name}"
    return scene_path.name or str(scene_path)


def _make_sim(
    scene_path: Path,
    scene_dataset_config: Optional[Path],
    width: int,
    height: int,
) -> habitat_sim.Simulator:
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene_path)
    if scene_dataset_config is not None:
        sim_cfg.scene_dataset_config_file = str(scene_dataset_config.resolve())
    sim_cfg.enable_physics = False

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [height, width]

    sem_spec = habitat_sim.CameraSensorSpec()
    sem_spec.uuid = "semantic"
    sem_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    sem_spec.resolution = [height, width]

    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec, sem_spec]

    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def _make_env(
    scene_dir: Path,
    scene: str,
    width: int,
    height: int,
    scene_dataset: Optional[str],
    scene_dataset_config: Optional[Path],
) -> Tuple[Optional[habitat.Env], habitat_sim.Simulator, Path]:
    cfg = get_task_config(
        config_paths=["configs/semantic_audionav/av_nav/mp3d/semantic_audiogoal.yaml"]
    )
    cfg.defrost()
    sim_type = "ContinuousSoundSpacesSim"
    dataset_hint = ""
    if scene_dataset_config is not None:
        dataset_hint = str(scene_dataset_config)
        cfg.SIMULATOR.SCENE_DATASET = str(scene_dataset_config.resolve())
    elif scene_dataset is not None:
        dataset_hint = scene_dataset
        cfg.SIMULATOR.SCENE_DATASET = scene_dataset
    use_env = True
    if "hm3d" in dataset_hint.lower():
        sim_type = "Sim-v0"
        use_env = False
    cfg.SIMULATOR.TYPE = sim_type
    cfg.SIMULATOR.AUDIO.ENABLED = False
    cfg.SIMULATOR.RGB_SENSOR.WIDTH = width
    cfg.SIMULATOR.RGB_SENSOR.HEIGHT = height
    cfg.SIMULATOR.SEMANTIC_SENSOR.WIDTH = width
    cfg.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = height
    sensors = list(getattr(cfg.SIMULATOR.AGENT_0, "SENSORS", []))
    for sensor_name in ["RGB_SENSOR", "SEMANTIC_SENSOR"]:
        if sensor_name not in sensors:
            sensors.append(sensor_name)
    cfg.SIMULATOR.AGENT_0.SENSORS = sensors
    cfg.DATASET.SPLIT = "test"
    cfg.DATASET.SCENES_DIR = str(scene_dir)
    scene_path = _resolve_scene_path(scene_dir, scene).resolve()
    if not scene_path.is_file():
        raise RuntimeError(f"Resolved scene_path does not exist: {scene_path}")
    cfg.SIMULATOR.SCENE = str(scene_path)
    cfg.freeze()
    print(f"scene_path_in_use={scene_path}", flush=True)
    if use_env:
        env = habitat.Env(config=cfg)
        env.sim._current_scene = str(scene_path)
        return env, env.sim, scene_path
    sim = _make_sim(scene_path, scene_dataset_config, width, height)
    return None, sim, scene_path


def _collect_objects(
    sim: habitat_sim.Simulator,
) -> Dict[str, List[Tuple[int, str, np.ndarray]]]:
    objects_by_cat: Dict[str, List[Tuple[int, str, np.ndarray]]] = {}
    scene = sim.semantic_scene
    if scene is None and hasattr(sim, "_sim"):
        scene = getattr(sim._sim, "semantic_scene", None)
    if scene is None:
        print("semantic_scene=None", flush=True)
        return objects_by_cat
    for obj in scene.objects:
        cat = getattr(obj, "category", None)
        if cat is None:
            continue
        try:
            name = cat.name()
        except Exception:
            continue
        aabb = getattr(obj, "aabb", None)
        if aabb is None:
            continue
        center = np.array(aabb.center, dtype=np.float32)
        sem_id = getattr(obj, "semantic_id", None)
        if sem_id is None:
            # Fallback for older bindings (if any)
            sem_id = getattr(obj, "semanticID", None)
        if sem_id is None:
            continue
        try:
            sem_id_int = int(sem_id)
        except Exception:
            # HM3D sometimes uses string ids like "Unknown_0"
            continue
        obj_id_raw: str
        try:
            obj_id_raw = str(int(obj.id))
        except Exception:
            obj_id_raw = str(obj.id)
        objects_by_cat.setdefault(name, []).append(
            (sem_id_int, obj_id_raw, center)
        )
    return objects_by_cat


def _pick_goals(
    objects_by_cat: Dict[str, List[Tuple[int, int, np.ndarray]]],
    sound_map: Dict[str, str],
    num_goals: int,
    rng: random.Random,
    min_goal_sep: float,
    sim: habitat_sim.Simulator,
) -> List[Dict]:
    categories = [c for c in objects_by_cat.keys() if c in sound_map]
    if len(categories) < num_goals:
        raise RuntimeError("Not enough categories with sounds to sample goals.")
    chosen_cats = rng.sample(categories, num_goals)
    goals: List[Dict] = []
    for cat in chosen_cats:
        tries = 0
        while tries < 50:
            sem_id, obj_id_raw, pos = rng.choice(objects_by_cat[cat])
            view_pos = _snap_to_navmesh(sim, pos)
            if not _is_navigable(sim, view_pos):
                alt = _sample_nearby_navigable(sim, pos, rng)
                if alt is None:
                    tries += 1
                    continue
                view_pos = alt
            if not goals:
                break
            ok = True
            for g in goals:
                g_view = g["view_points"][0] if g.get("view_points") else g["position"]
                d = _geodesic_distance(
                    sim,
                    view_pos,
                    np.array(g_view, dtype=np.float32),
                )
                if not np.isfinite(d) or d < min_goal_sep:
                    ok = False
                    break
            if ok:
                break
            tries += 1
        goals.append(
            {
                "position": pos.tolist(),
                "radius": 1e-5,
                # Use semantic sensor id for matching semantic observations
                "object_id": sem_id,
                # Preserve raw scene object id for traceability
                "object_id_raw": obj_id_raw,
                "object_name": None,
                "object_category": cat,
                "room_id": None,
                "room_name": None,
                "view_points": [view_pos.tolist()],
            }
        )
    return goals


def _geodesic_distance(sim: habitat_sim.Simulator, start: np.ndarray, goal: np.ndarray) -> float:
    path = habitat_sim.ShortestPath()
    path.requested_start = start
    path.requested_end = goal
    sim.pathfinder.find_path(path)
    return float(path.geodesic_distance)


def _navmesh_loaded(sim: habitat_sim.Simulator) -> Optional[bool]:
    pf = getattr(sim, "pathfinder", None)
    if pf is None:
        return None
    for attr in ("is_loaded", "isLoaded"):
        if hasattr(pf, attr):
            try:
                value = getattr(pf, attr)
                return bool(value() if callable(value) else value)
            except Exception:
                return None
    return None


def _snap_to_navmesh(sim: habitat_sim.Simulator, position: np.ndarray) -> np.ndarray:
    try:
        pf = sim.pathfinder
    except Exception:
        return np.array(position, dtype=np.float32)
    try:
        if hasattr(pf, "snap_point"):
            return np.array(pf.snap_point(position), dtype=np.float32)
    except Exception:
        pass
    return np.array(position, dtype=np.float32)


def _is_navigable(sim: habitat_sim.Simulator, position: np.ndarray) -> bool:
    try:
        return bool(sim.pathfinder.is_navigable(position))
    except Exception:
        return True


def _sample_nearby_navigable(
    sim: habitat_sim.Simulator,
    center: np.ndarray,
    rng: random.Random,
    radius_min: float = 0.5,
    radius_max: float = 4.0,
    max_tries: int = 50,
) -> Optional[np.ndarray]:
    for _ in range(max_tries):
        angle = rng.uniform(0.0, 2 * math.pi)
        radius = rng.uniform(radius_min, radius_max)
        candidate = np.array(
            [
                center[0] + radius * math.cos(angle),
                center[1],
                center[2] + radius * math.sin(angle),
            ],
            dtype=np.float32,
        )
        candidate = _snap_to_navmesh(sim, candidate)
        if _is_navigable(sim, candidate):
            return candidate
    try:
        candidate = np.array(sim.pathfinder.get_random_navigable_point(), dtype=np.float32)
        candidate = _snap_to_navmesh(sim, candidate)
        if _is_navigable(sim, candidate):
            return candidate
    except Exception:
        pass
    return None


def _estimate_num_action(
    sim,
    start_pos: np.ndarray,
    start_rot: List[float],
    goal_pos: np.ndarray,
    goal_radius: float,
    max_steps: int,
    forward_step_size: Optional[float] = None,
) -> int:
    """Estimate minimal action count using greedy shortest-path follower."""
    # HabitatSim wrapper path
    if hasattr(sim, "_sim"):
        saved = sim.get_agent_state()
        sim.set_agent_state(start_pos.tolist(), start_rot, reset_sensors=False)
        follower = ShortestPathFollower(
            sim=sim,
            goal_radius=goal_radius,
            return_one_hot=False,
            stop_on_error=True,
        )
        steps = 0
        for _ in range(max_steps):
            action = follower.get_next_action(goal_pos)
            if action is None or action == HabitatSimActions.STOP:
                break
            sim.step(action)
            steps += 1
        sim.set_agent_state(saved.position, saved.rotation, reset_sensors=False)
        return int(steps)

    # Direct habitat_sim.Simulator fallback: approximate using geodesic distance
    if forward_step_size is None or forward_step_size <= 0:
        forward_step_size = 0.25
    geod = _geodesic_distance(sim, start_pos, goal_pos)
    if not np.isfinite(geod):
        return 0
    remaining = max(0.0, geod - goal_radius)
    steps = int(math.ceil(remaining / forward_step_size))
    return int(min(steps, max_steps))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate semantic AudioNav episodes.")
    parser.add_argument("--scene", default="yqstnuAEVhm", help="Scene name")
    parser.add_argument("--num-episodes", type=int, default=10)
    parser.add_argument("--min-goals", type=int, default=1)
    parser.add_argument("--max-goals", type=int, default=4)
    parser.add_argument(
        "--scene-dir",
        type=Path,
        default=Path("data/scene_datasets/mp3d"),
    )
    parser.add_argument(
        "--scene-dataset",
        type=str,
        default=None,
        help="Override SIMULATOR.SCENE_DATASET (e.g., 'hm3d').",
    )
    parser.add_argument(
        "--scene-dataset-config",
        type=Path,
        default=None,
        help="Path to scene_dataset_config.json (takes priority over --scene-dataset).",
    )
    parser.add_argument(
        "--sounds-dir",
        type=Path,
        default=Path("data/sounds/semantic_splits/test"),
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--out", type=Path, default=Path("yqstnuAEVhm.json.gz"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--schedule", type=int, default=25)
    parser.add_argument("--min-start-goal-dist", type=float, default=4.0)
    parser.add_argument(
        "--max-start-height-diff",
        type=float,
        default=1.5,
        help="Max allowed |start_y - goal_view_y| to keep start on the same floor.",
    )
    parser.add_argument("--min-goal-sep", type=float, default=2.0)
    parser.add_argument("--max-episode-tries", type=int, default=50)
    parser.add_argument("--max-start-tries", type=int, default=500)
    parser.add_argument(
        "--force-exit",
        action="store_true",
        help="Exit without simulator cleanup (workaround for habitat-sim crashes)",
    )
    args = parser.parse_args()

    if args.scene_dataset_config is not None and not args.scene_dataset_config.exists():
        raise RuntimeError(f"scene-dataset-config not found: {args.scene_dataset_config}")

    rng = random.Random(args.seed)
    sound_map = _load_sound_map(args.sounds_dir)
    if not sound_map:
        raise RuntimeError("No sound files found in sounds-dir.")

    env: Optional[habitat.Env] = None
    sim: Optional[habitat_sim.Simulator] = None
    env, sim, scene_path = _make_env(
        args.scene_dir,
        args.scene,
        args.width,
        args.height,
        args.scene_dataset,
        args.scene_dataset_config,
    )
    try:
        navmesh_loaded = _navmesh_loaded(sim)
        if navmesh_loaded is False:
            raise RuntimeError("Navmesh not loaded; check scene_dataset_config and navmesh files.")
        print("env_ready=True", flush=True)
        sim_sem = getattr(sim, "semantic_scene", None)
        sim_inner = getattr(sim, "_sim", None)
        inner_sem = getattr(sim_inner, "semantic_scene", None) if sim_inner is not None else None
        print(
            f"semantic_scene_outer_none={sim_sem is None} semantic_scene_inner_none={inner_sem is None}",
            flush=True,
        )
        objects_by_cat = _collect_objects(sim_inner if sim_inner is not None else sim)
        print(f"semantic_objects_by_cat={len(objects_by_cat)}", flush=True)
        if not objects_by_cat:
            raise RuntimeError("No semantic objects with semantic_id found in scene.")

        episodes: List[Dict] = []
        for ep_idx in range(args.num_episodes):
            episode_ok = False
            for _ in range(args.max_episode_tries):
                num_goals = rng.randint(args.min_goals, args.max_goals)
                goals = _pick_goals(
                    objects_by_cat, sound_map, num_goals, rng, args.min_goal_sep, sim
                )

                ref_height = None
                if goals and args.max_start_height_diff is not None:
                    try:
                        ref_height = float(goals[0]["view_points"][0][1])
                    except Exception:
                        ref_height = None

                start_pos = None
                for _ in range(args.max_start_tries):
                    cand = np.array(
                        sim.pathfinder.get_random_navigable_point(),
                        dtype=np.float32,
                    )
                    cand = _snap_to_navmesh(sim, cand)
                    if not _is_navigable(sim, cand):
                        continue
                    ok = True
                    for g in goals:
                        g_view = g["view_points"][0] if g.get("view_points") else g["position"]
                        d = _geodesic_distance(
                            sim, cand, np.array(g_view, dtype=np.float32)
                        )
                        if not np.isfinite(d) or d < args.min_start_goal_dist:
                            ok = False
                            break
                    if ok and ref_height is not None:
                        if abs(float(cand[1]) - ref_height) > args.max_start_height_diff:
                            ok = False
                    if ok:
                        start_pos = cand
                        break
                if start_pos is None:
                    continue
                episode_ok = True
                break
            if not episode_ok:
                raise RuntimeError(
                    "Failed to sample episode with min distance constraints."
                )
            start_rot = [0.0, 0.0, 0.0, 1.0]

            # Estimate per-goal geodesic distance and action count
            try:
                goal_radius = float(env._config.TASK.SUCCESS.SUCCESS_DISTANCE) if env else 1.0
            except Exception:
                goal_radius = 1.0
            try:
                max_steps = int(env._config.ENVIRONMENT.MAX_EPISODE_STEPS) if env else 500
            except Exception:
                max_steps = 500
            try:
                forward_step = float(env._config.SIMULATOR.FORWARD_STEP_SIZE) if env else None
            except Exception:
                forward_step = None

            per_goal_info: List[Dict] = []
            for g in goals:
                g_view = g["view_points"][0] if g.get("view_points") else g["position"]
                goal_pos = np.array(g_view, dtype=np.float32)
                geod = _geodesic_distance(sim, start_pos, goal_pos)
                if not np.isfinite(geod):
                    geod = 0.0
                try:
                    na = _estimate_num_action(
                        sim=sim,
                        start_pos=start_pos,
                        start_rot=start_rot,
                        goal_pos=goal_pos,
                        goal_radius=goal_radius,
                        max_steps=max_steps,
                        forward_step_size=forward_step,
                    )
                except Exception:
                    na = 0
                per_goal_info.append(
                    {
                        "object_id": g["object_id"],
                        "object_category": g["object_category"],
                        "geodesic_distance": geod,
                        "num_action": int(na),
                    }
                )
            # Keep first-goal summary fields for compatibility with existing metrics
            geod = per_goal_info[0]["geodesic_distance"] if per_goal_info else 0.0
            num_action = per_goal_info[0]["num_action"] if per_goal_info else 0
            sound_sources = []
            for g in goals:
                wav_path = Path(sound_map[g["object_category"]])
                # Store path relative to SOURCE_SOUND_DIR (e.g., test/chair.wav)
                rel = wav_path.relative_to(args.sounds_dir.parent)
                sound_sources.append({"sound_id": rel.as_posix()})
            episode = {
                "episode_id": str(ep_idx),
                "scene_id": _scene_id_from_path(scene_path, args.scene_dir, args.scene),
                "start_position": start_pos.tolist(),
                "start_rotation": start_rot,
                "num_goals": len(goals),
                "info": {
                    "geodesic_distance": geod,
                    "num_action": num_action,
                    "per_goal": per_goal_info,
                },
                "goals": goals,
                "start_room": None,
                "shortest_paths": None,
                "object_category": goals[0]["object_category"],
                "sound_id": sound_sources[0]["sound_id"],
                "offset": "0",
                "duration": str(args.schedule),
                "sound_sources": sound_sources,
                "sound_source_schedule": ["round_robin", args.schedule],
            }
            episodes.append(episode)

        out_data = {"episodes": episodes, "scene": args.scene}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(args.out, "wt") as f:
            json.dump(out_data, f, ensure_ascii=True)
        print(f"wrote {len(episodes)} episodes to {args.out}", flush=True)
    except Exception as exc:
        print(f"error={exc}", flush=True)
        if args.force_exit:
            sys.stdout.flush()
            os._exit(1)
        raise
    finally:
        if args.force_exit:
            sys.stdout.flush()
            os._exit(0)
        if env is not None:
            env.close()
        elif sim is not None:
            sim.close()


if __name__ == "__main__":
    main()
