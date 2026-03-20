from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from ss_baselines.omega_nav.memory import HierarchicalMemory
from ss_baselines.omega_nav.perception import GoalSpec, PerceptionEncoder
from ss_baselines.omega_nav.perception.base import AudioMatch, ObjectRegionMatch, PerceptionOutput, SemanticMapState
from ss_baselines.omega_nav.planner import OmegaLLMPlanner
from ss_baselines.omega_nav.utils import DEFAULT_OMEGA_CONFIG, hash_embedding, load_omega_config


class DummySim:
    def __init__(self):
        self._state = SimpleNamespace(
            position=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
            rotation=np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        )

    def get_agent_state(self):
        return self._state

    def geodesic_distance(self, position_a, position_bs, episode=None):
        del episode
        position_a = np.asarray(position_a, dtype=np.float32)
        target = np.asarray(position_bs[0], dtype=np.float32)
        return float(np.linalg.norm((target - position_a)[[0, 2]]))


class DummyEnv:
    def __init__(self):
        self.sim = DummySim()
        self.task = SimpleNamespace(goal_order_mode="ordered", order_enforced=True)


class DummyGoal:
    def __init__(self, *, category: str, semantic_id: int, position, room_name: str = "living room"):
        self.object_category = category
        self.object_name = category
        self.object_id = semantic_id
        self.object_id_raw = semantic_id
        self.position = position
        self.room_name = room_name
        self.view_points = [SimpleNamespace(agent_state=SimpleNamespace(position=position))]


def make_episode():
    return SimpleNamespace(
        goals=[
            DummyGoal(category="sofa", semantic_id=10, position=np.asarray([1.0, 0.0, -2.0], dtype=np.float32)),
            DummyGoal(category="lamp", semantic_id=11, position=np.asarray([2.0, 0.0, -3.0], dtype=np.float32)),
        ],
        tasks=[["sofa_01", "description"], ["lamp_02", "image_0"]],
        sound_sources=[
            {"sound_id": "bell.wav", "position": [2.0, 0.0, -2.0]},
            {"sound_id": "hum.wav", "position": [3.0, 0.0, -2.0]},
        ],
        sound_id="bell.wav",
    )


def make_observations(include_reference_match: bool = False):
    rgb = np.zeros((48, 48, 3), dtype=np.uint8)
    rgb[:, :, 1] = 40
    if include_reference_match:
        rgb[12:36, 16:32, 0] = 255
        rgb[12:36, 16:32, 2] = 80
    depth = np.ones((48, 48, 1), dtype=np.float32) * 0.4
    depth[12:36, 16:32, 0] = 0.15

    samples = np.linspace(0.0, 8.0 * np.pi, 1600, dtype=np.float32)
    left = 0.08 * np.sin(samples)
    right = 0.03 * np.sin(samples)
    obs = {
        "rgb": rgb,
        "depth": depth,
        "audiogoal": np.stack([left, right], axis=0),
    }
    return obs


def make_perception_output(goal_specs):
    visual_matches = {
        goal_specs[0].goal_id: ObjectRegionMatch(
            goal_id=goal_specs[0].goal_id,
            goal_index=0,
            category="sofa",
            similarity=0.95,
            visible=True,
            pixel_count=64,
            visible_ratio=0.25,
            bbox=(10, 8, 22, 20),
            estimated_distance_m=0.8,
            relative_angle_deg=0.0,
            relative_direction="forward",
            target_position=None,
        )
    }
    if len(goal_specs) > 1:
        visual_matches[goal_specs[1].goal_id] = ObjectRegionMatch(
            goal_id=goal_specs[1].goal_id,
            goal_index=1,
            category="lamp",
            similarity=0.1,
            visible=False,
        )
    audio_matches = {
        goal_specs[0].goal_id: AudioMatch(
            goal_id=goal_specs[0].goal_id,
            goal_index=0,
            category="sofa",
            similarity=0.9,
            aggregated_similarity=0.95,
            detected=True,
            direction_text="左前方 30°",
            relative_angle_deg=-30.0,
            itd_seconds=-0.0002,
            distance_m=0.9,
            sound_position=None,
        )
    }
    if len(goal_specs) > 1:
        audio_matches[goal_specs[1].goal_id] = AudioMatch(
            goal_id=goal_specs[1].goal_id,
            goal_index=1,
            category="lamp",
            similarity=0.02,
            aggregated_similarity=0.02,
            detected=False,
        )
    semantic_map = SemanticMapState(
        occupancy=np.zeros((8, 8), dtype=np.int8),
        visited=np.zeros((8, 8), dtype=np.uint8),
        frontier=np.zeros((8, 8), dtype=np.uint8),
        agent_cell=(4, 4),
        origin_world=np.zeros(3, dtype=np.float32),
        resolution_m=0.25,
        frontier_world_positions=(np.asarray([1.0, 0.0, -1.0], dtype=np.float32),),
        explored_ratio=0.25,
        free_space_by_direction={"left": 1.0, "forward": 3.0, "right": 0.8},
    )
    return PerceptionOutput(
        step_index=0,
        scene_description="视野中可见 sofa。",
        visual_matches=visual_matches,
        audio_matches=audio_matches,
        top_clip_matches=tuple(visual_matches.values()),
        semantic_map=semantic_map,
        observation_summary="visual=sofa:0.95 | audio=sofa:0.95@左前方 30° | explored=25.0%",
    )


def test_goal_spec_builds_text_and_image_routes_without_gt_geometry():
    encoder = PerceptionEncoder(load_omega_config(overrides={}))
    payloads = [
        {"modality": "description", "text": "a sofa near the wall"},
        {"modality": "image", "image": np.full((8, 8, 3), 255, dtype=np.uint8)},
    ]
    goal_specs = encoder.build_goal_specs(make_episode(), payloads)
    assert goal_specs[0].text_query == "a sofa near the wall"
    assert goal_specs[1].reference_image is not None
    assert goal_specs[1].image_embedding is not None
    assert goal_specs[0].semantic_id is None
    assert goal_specs[0].object_position is None
    assert goal_specs[0].sound_position is None
    assert goal_specs[0].view_positions == ()


def test_vlm_description_is_cached_between_refresh_steps():
    encoder = PerceptionEncoder(load_omega_config(overrides={"visual": {"describe_every": 5}}))
    env = DummyEnv()
    episode = make_episode()
    goal_specs = encoder.build_goal_specs(episode, [{"modality": "description", "text": "sofa"}, {"modality": "description", "text": "lamp"}])
    encoder.reset(env=env, goal_specs=goal_specs)
    obs = make_observations()
    first = encoder.encode(step_index=0, env=env, episode=episode, observations=obs, goal_specs=goal_specs, pending_goal_ids=[goal_specs[0].goal_id], order_mode="ordered")
    second = encoder.encode(step_index=1, env=env, episode=episode, observations=obs, goal_specs=goal_specs, pending_goal_ids=[goal_specs[0].goal_id], order_mode="ordered")
    assert first.scene_description == second.scene_description


def test_clip_detector_uses_reference_image_not_semantic_gt():
    encoder = PerceptionEncoder(load_omega_config(overrides={}))
    goal_specs = encoder.build_goal_specs(make_episode(), [{"modality": "image", "image": np.full((16, 16, 3), [255, 0, 80], dtype=np.uint8)}, {"modality": "description", "text": "lamp"}])
    visual_matches, ranked = encoder._clip.detect(make_observations(include_reference_match=True), goal_specs)
    assert visual_matches[goal_specs[0].goal_id].visible is True
    assert ranked[0].goal_id == goal_specs[0].goal_id


def test_clip_distance_uses_depth_from_best_matching_region():
    encoder = PerceptionEncoder(load_omega_config(overrides={}))
    goal_specs = encoder.build_goal_specs(make_episode(), [{"modality": "image", "image": np.full((16, 16, 3), [255, 0, 80], dtype=np.uint8)}])
    visual_matches, _ = encoder._clip.detect(make_observations(include_reference_match=True), goal_specs)
    assert abs(float(visual_matches[goal_specs[0].goal_id].estimated_distance_m) - 0.9) < 1e-5


def test_clap_detector_tracks_pending_ordered_goal_from_audio_only():
    encoder = PerceptionEncoder(load_omega_config(overrides={}))
    env = DummyEnv()
    episode = make_episode()
    goal_specs = encoder.build_goal_specs(episode, [{"modality": "description", "text": "sofa"}, {"modality": "description", "text": "lamp"}])
    encoder._clap.reset(goal_specs)
    audio_matches = encoder._clap.match(env, episode, make_observations(), goal_specs, pending_goal_ids=[goal_specs[0].goal_id], order_mode="ordered")
    assert audio_matches[goal_specs[0].goal_id].detected is True
    assert audio_matches[goal_specs[1].goal_id].detected is False
    assert "方" in audio_matches[goal_specs[0].goal_id].direction_text


def test_clap_temporal_median_crosses_threshold():
    encoder = PerceptionEncoder(load_omega_config(overrides={"audio": {"aggregation_window": 3, "detection_threshold": 0.6}}))
    env = DummyEnv()
    episode = make_episode()
    goal_specs = encoder.build_goal_specs(episode, [{"modality": "description", "text": "sofa"}, {"modality": "description", "text": "lamp"}])
    encoder._clap.reset(goal_specs)
    obs = make_observations()
    for _ in range(3):
        matches = encoder._clap.match(env, episode, obs, goal_specs, pending_goal_ids=[goal_specs[0].goal_id], order_mode="ordered")
    assert matches[goal_specs[0].goal_id].aggregated_similarity >= 0.6


def test_depth_processor_updates_frontier_and_exploration():
    encoder = PerceptionEncoder(load_omega_config(overrides={"depth": {"map_size_cells": 64, "sample_stride": 8}}))
    env = DummyEnv()
    episode = make_episode()
    goal_specs = encoder.build_goal_specs(episode, [{"modality": "description", "text": "sofa"}, {"modality": "description", "text": "lamp"}])
    encoder.reset(env=env, goal_specs=goal_specs)
    output = encoder.encode(step_index=0, env=env, episode=episode, observations=make_observations(), goal_specs=goal_specs, pending_goal_ids=[goal_specs[0].goal_id], order_mode="ordered")
    assert output.semantic_map.explored_ratio > 0.0
    assert len(output.semantic_map.frontier_world_positions) > 0


def test_working_memory_slides_window():
    memory = HierarchicalMemory({"working_window": 2, "episodic_decay": 0.0, "episodic_prune_threshold": 0.0, "revisit_radius_m": 0.0})
    goals = [GoalSpec(goal_id="goal_000", goal_index=0, modality="text", category="sofa", text_query="sofa")]
    memory.reset(goals)
    env = DummyEnv()
    semantic_map = SemanticMapState(
        occupancy=np.zeros((4, 4), dtype=np.int8),
        visited=np.zeros((4, 4), dtype=np.uint8),
        frontier=np.zeros((4, 4), dtype=np.uint8),
        agent_cell=(1, 1),
        origin_world=np.zeros(3, dtype=np.float32),
        resolution_m=0.25,
        explored_ratio=0.1,
    )
    for idx in range(3):
        perception = PerceptionOutput(
            step_index=idx,
            scene_description="scene",
            visual_matches={goals[0].goal_id: ObjectRegionMatch(goal_id=goals[0].goal_id, goal_index=0, category="sofa", similarity=0.0, visible=False)},
            audio_matches={goals[0].goal_id: AudioMatch(goal_id=goals[0].goal_id, goal_index=0, category="sofa", similarity=0.0, aggregated_similarity=0.0, detected=False)},
            top_clip_matches=(),
            semantic_map=semantic_map,
            observation_summary=f"obs-{idx}",
        )
        memory.update(step_index=idx, env=env, perception=perception, pending_goal_ids=[goals[0].goal_id], order_mode="ordered")
    assert memory.working_memory() == ["obs-1", "obs-2"]


def test_episodic_memory_decays_and_prunes():
    memory = HierarchicalMemory({"working_window": 2, "episodic_decay": 1.0, "episodic_prune_threshold": 0.2, "revisit_radius_m": 5.0})
    goals = [GoalSpec(goal_id="goal_000", goal_index=0, modality="text", category="sofa", text_query="sofa")]
    memory.reset(goals)
    env = DummyEnv()
    perception = make_perception_output(goals)
    memory.update(step_index=0, env=env, perception=perception, pending_goal_ids=[goals[0].goal_id], order_mode="ordered")
    assert memory.best_hint(goals[0].goal_id) is not None
    memory.update(step_index=1, env=env, perception=PerceptionOutput(step_index=1, scene_description="scene", visual_matches={goals[0].goal_id: ObjectRegionMatch(goal_id=goals[0].goal_id, goal_index=0, category="sofa", similarity=0.0, visible=False)}, audio_matches={goals[0].goal_id: AudioMatch(goal_id=goals[0].goal_id, goal_index=0, category="sofa", similarity=0.0, aggregated_similarity=0.0, detected=False)}, top_clip_matches=(), semantic_map=perception.semantic_map, observation_summary="none"), pending_goal_ids=[goals[0].goal_id], order_mode="ordered")
    assert memory.best_hint(goals[0].goal_id) is None


def test_planner_ordered_picks_queue_head():
    env = DummyEnv()
    episode = make_episode()
    encoder = PerceptionEncoder(load_omega_config(overrides={}))
    goal_specs = encoder.build_goal_specs(episode, [{"modality": "description", "text": "sofa"}, {"modality": "description", "text": "lamp"}])
    memory = HierarchicalMemory(DEFAULT_OMEGA_CONFIG["memory"])
    memory.reset(goal_specs)
    planner = OmegaLLMPlanner(DEFAULT_OMEGA_CONFIG["planner"])
    decision = planner.decide(env=env, episode=episode, goal_specs=goal_specs, perception=make_perception_output(goal_specs), memory=memory, pending_goal_ids=[goal.goal_id for goal in goal_specs], order_mode="ordered", step_index=0, submit_distance_m=0.1)
    assert decision.next_goal == goal_specs[0].goal_id


def test_planner_unordered_prefers_visible_goal():
    env = DummyEnv()
    env.task.goal_order_mode = "unordered"
    env.task.order_enforced = False
    episode = make_episode()
    encoder = PerceptionEncoder(load_omega_config(overrides={}))
    goal_specs = encoder.build_goal_specs(episode, [{"modality": "description", "text": "sofa"}, {"modality": "description", "text": "lamp"}])
    memory = HierarchicalMemory(DEFAULT_OMEGA_CONFIG["memory"])
    memory.reset(goal_specs)
    planner = OmegaLLMPlanner(DEFAULT_OMEGA_CONFIG["planner"])
    decision = planner.decide(env=env, episode=episode, goal_specs=goal_specs, perception=make_perception_output(goal_specs), memory=memory, pending_goal_ids=[goal.goal_id for goal in goal_specs], order_mode="unordered", step_index=0, submit_distance_m=0.1)
    assert decision.next_goal == goal_specs[0].goal_id


def test_hash_embedding_is_deterministic():
    assert np.allclose(hash_embedding("hello", 8), hash_embedding("hello", 8))
