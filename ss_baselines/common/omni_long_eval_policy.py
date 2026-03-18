#!/usr/bin/env python3

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type

import numpy as np
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.utils.geometry_utils import quaternion_rotate_vector

from soundspaces.tasks.shortest_path_follower import ShortestPathFollower


def _as_int_or_none(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, str):
        token = value.strip()
        if token.startswith(("+", "-")):
            sign = token[0]
            digits = token[1:]
            if digits.isdigit():
                return int(sign + digits)
        if token.isdigit():
            return int(token)
    return None


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        value = float(value)
        if np.isfinite(value):
            return value
    return None


def _get_task(env: Any) -> Any:
    task = getattr(env, "task", None)
    if task is not None:
        return task
    return getattr(env, "_task", None)


def _extract_action_names(env: Any) -> List[str]:
    task = _get_task(env)
    if task is None:
        return []
    actions = getattr(task, "actions", None)
    if isinstance(actions, dict):
        return [str(name) for name in actions.keys()]
    return []


def _is_named_action(action: Any, action_name: str) -> bool:
    if isinstance(action, dict):
        value = action.get("action")
        return isinstance(value, str) and value.upper() == action_name.upper()
    if isinstance(action, str):
        return action.upper() == action_name.upper()
    return False


def _build_named_action(action_name: str) -> Dict[str, str]:
    return {"action": action_name}


def _sim_action_to_named_action(action: Any) -> Any:
    action_id = _as_int_or_none(action)
    if action_id is None:
        return action

    if action_id == int(HabitatSimActions.STOP):
        return _build_named_action("STOP")
    if action_id == int(HabitatSimActions.MOVE_FORWARD):
        return _build_named_action("MOVE_FORWARD")
    if action_id == int(HabitatSimActions.TURN_LEFT):
        return _build_named_action("TURN_LEFT")
    if action_id == int(HabitatSimActions.TURN_RIGHT):
        return _build_named_action("TURN_RIGHT")
    return action


def _distance_to_current_goal(env: Any, episode: Any) -> Optional[float]:
    goals = getattr(episode, "goals", None)
    if not isinstance(goals, list) or len(goals) == 0:
        return None
    return _distance_to_specific_goal(env, episode, goals[0])


def _distance_to_specific_goal(env: Any, episode: Any, goal: Any) -> Optional[float]:
    task = _get_task(env)
    if task is not None and hasattr(task, "_distance_to_goal"):
        return _optional_float(task._distance_to_goal(goal, episode))

    goal_pos = getattr(goal, "position", None)
    if goal_pos is None:
        return None
    current_pos = env.sim.get_agent_state().position
    distance = env.sim.geodesic_distance(current_pos, [goal_pos], episode)
    return _optional_float(distance)


def _normalize_goal_order_mode(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bool):
        return "ordered" if value else "unordered"

    token = str(value).strip().lower()
    if token in {"ordered", "order", "true", "1", "yes", "y"}:
        return "ordered"
    if token in {"unordered", "unorder", "false", "0", "no", "n"}:
        return "unordered"
    return None


def _order_enforced(env: Any, policy: Optional["LifelongEvalPolicy"] = None) -> bool:
    if policy is not None:
        override_mode = _normalize_goal_order_mode(getattr(policy, "_goal_order_mode", None))
        if override_mode is not None:
            return override_mode == "ordered"

    task = _get_task(env)
    if task is None:
        return True
    if hasattr(task, "order_enforced"):
        return bool(task.order_enforced)
    mode = getattr(task, "_mode", None)
    if mode is None:
        return True
    return str(mode).strip().lower() == "ordered"


def _navigation_goal(
    env: Any,
    episode: Any,
    policy: Optional["LifelongEvalPolicy"] = None,
) -> Tuple[Optional[int], Optional[Any]]:
    goals = tuple(getattr(policy, "_episode_goals", ())) if policy is not None else ()
    if not goals:
        episode_goals = getattr(episode, "goals", None)
        if isinstance(episode_goals, list):
            goals = tuple(episode_goals)

    if not goals:
        return None, None

    order_enforced = _order_enforced(env, policy=policy)
    if order_enforced:
        goal_index = int(getattr(policy, "_ordered_goal_index", 0)) if policy is not None else 0
        if goal_index < 0 or goal_index >= len(goals):
            return None, None
        return goal_index, goals[goal_index]

    completed_goal_indices = set(getattr(policy, "_submitted_goal_indices", ())) if policy is not None else set()
    winner: Optional[Tuple[int, Any]] = None
    winner_distance: Optional[float] = None
    for goal_index, goal in enumerate(goals):
        if goal_index in completed_goal_indices:
            continue
        distance = _distance_to_specific_goal(env, episode, goal)
        if distance is None:
            continue
        if winner_distance is None or float(distance) < float(winner_distance):
            winner = (goal_index, goal)
            winner_distance = float(distance)

    if winner is not None:
        return winner
    return None, None


def _goal_signature(goal_index: Optional[int], goal: Any) -> Tuple[Any, Any]:
    return (
        int(goal_index) if goal_index is not None else None,
        getattr(goal, "object_id", None),
    )


def _goal_target_positions(goal: Any, follow_view_points: bool) -> List[np.ndarray]:
    targets: List[np.ndarray] = []
    if follow_view_points:
        for view in getattr(goal, "view_points", None) or []:
            agent_state = getattr(view, "agent_state", None)
            position = getattr(agent_state, "position", None)
            if position is None:
                continue
            targets.append(np.asarray(position, dtype=np.float32))
    if targets:
        return targets

    position = getattr(goal, "position", None)
    if position is None:
        return []
    return [np.asarray(position, dtype=np.float32)]


def _nearest_target_position(
    env: Any,
    episode: Any,
    target_positions: Sequence[np.ndarray],
) -> Optional[np.ndarray]:
    if not target_positions:
        return None

    current_position = np.asarray(env.sim.get_agent_state().position, dtype=np.float32)
    best_target: Optional[np.ndarray] = None
    best_distance: Optional[float] = None
    for target_position in target_positions:
        candidate_target = np.asarray(target_position, dtype=np.float32)
        if hasattr(env.sim, "_snap_to_navmesh"):
            candidate_target = np.asarray(env.sim._snap_to_navmesh(candidate_target), dtype=np.float32)
        elif hasattr(env.sim, "pathfinder") and hasattr(env.sim.pathfinder, "snap_point"):
            candidate_target = np.asarray(env.sim.pathfinder.snap_point(candidate_target), dtype=np.float32)
        distance = env.sim.geodesic_distance(current_position, [candidate_target], episode)
        if distance is None or not np.isfinite(distance):
            continue
        if best_distance is None or float(distance) < float(best_distance):
            best_distance = float(distance)
            best_target = np.asarray(candidate_target, dtype=np.float32)

    if best_target is not None:
        return best_target
    return np.asarray(target_positions[0], dtype=np.float32)


def _heading_fallback_action(env: Any, target_position: np.ndarray) -> Any:
    state = env.sim.get_agent_state()
    current_position = np.asarray(state.position, dtype=np.float32)
    target_vector = np.asarray(target_position, dtype=np.float32) - current_position
    target_vector[1] = 0.0
    target_norm = float(np.linalg.norm(target_vector[[0, 2]]))
    if target_norm <= 1e-6:
        return _build_named_action("MOVE_FORWARD")
    target_vector = target_vector / max(target_norm, 1e-6)

    forward_vector = np.asarray(
        quaternion_rotate_vector(state.rotation, np.array([0.0, 0.0, -1.0], dtype=np.float32)),
        dtype=np.float32,
    )
    forward_vector[1] = 0.0
    forward_norm = float(np.linalg.norm(forward_vector[[0, 2]]))
    if forward_norm <= 1e-6:
        return _build_named_action("MOVE_FORWARD")
    forward_vector = forward_vector / max(forward_norm, 1e-6)

    cross = float(forward_vector[0] * target_vector[2] - forward_vector[2] * target_vector[0])
    dot = float(forward_vector[0] * target_vector[0] + forward_vector[2] * target_vector[2])
    angle = float(np.arctan2(cross, dot))
    if abs(angle) > np.deg2rad(10.0):
        if angle < 0.0:
            return _build_named_action("TURN_LEFT")
        return _build_named_action("TURN_RIGHT")
    return _build_named_action("MOVE_FORWARD")


def _task_success_distance(env: Any, default: float = 1.0) -> float:
    task = _get_task(env)
    task_config = getattr(task, "_config", None)
    success_cfg = getattr(task_config, "SUCCESS", None)
    value = getattr(success_cfg, "SUCCESS_DISTANCE", default)
    parsed = _optional_float(value)
    if parsed is not None:
        return parsed
    return float(default)


@dataclass
class LifelongEvalContext:
    step_index: int
    goal_payloads: Tuple[Dict[str, Any], ...] = ()


class LifelongEvalPolicy:
    def reset(self, *, env: Any, episode: Any, observations: Dict[str, Any]) -> None:
        return None

    def act(
        self,
        *,
        env: Any,
        episode: Any,
        observations: Dict[str, Any],
        context: LifelongEvalContext,
    ) -> Any:
        raise NotImplementedError

    def observe(
        self,
        *,
        env: Any,
        episode: Any,
        observations: Dict[str, Any],
        reward: Optional[float],
        done: bool,
        info: Optional[Dict[str, Any]],
    ) -> None:
        return None

    def close(self) -> None:
        return None


_POLICY_REGISTRY: Dict[str, Type[LifelongEvalPolicy]] = {}


def register_lifelong_eval_policy(name: str) -> Callable[[Type[LifelongEvalPolicy]], Type[LifelongEvalPolicy]]:
    policy_name = str(name).strip().lower()

    def _decorator(policy_cls: Type[LifelongEvalPolicy]) -> Type[LifelongEvalPolicy]:
        _POLICY_REGISTRY[policy_name] = policy_cls
        return policy_cls

    return _decorator


def list_lifelong_eval_policies() -> List[str]:
    return sorted(_POLICY_REGISTRY.keys())


def _load_external_policy(policy_spec: str) -> Type[LifelongEvalPolicy]:
    token = str(policy_spec).strip()
    if ":" in token:
        module_name, class_name = token.split(":", 1)
    elif "." in token:
        module_name, class_name = token.rsplit(".", 1)
    else:
        raise RuntimeError(
            "Unknown policy name '{}'. Use registered name or '<module>:<Class>'.".format(token)
        )

    module = importlib.import_module(module_name)
    policy_cls = getattr(module, class_name)
    if not isinstance(policy_cls, type) or not issubclass(policy_cls, LifelongEvalPolicy):
        raise RuntimeError(
            "Policy class '{}.{}' must inherit LifelongEvalPolicy".format(module_name, class_name)
        )
    return policy_cls


def build_lifelong_eval_policy(name: str, **kwargs: Any) -> LifelongEvalPolicy:
    key = str(name).strip().lower()
    if key in _POLICY_REGISTRY:
        return _POLICY_REGISTRY[key](**kwargs)
    policy_cls = _load_external_policy(name)
    return policy_cls(**kwargs)


@register_lifelong_eval_policy("random")
class RandomLifelongEvalPolicy(LifelongEvalPolicy):
    def __init__(
        self,
        submit_action_name: str = "LIFELONG_SUBMIT",
        stop_action_name: str = "STOP",
        min_submit_steps: int = 0,
        submit_probability: float = 0.0,
        seed: int = 0,
    ):
        self._submit_action_name = str(submit_action_name)
        self._stop_action_name = str(stop_action_name)
        self._min_submit_steps = int(min_submit_steps)
        self._submit_probability = float(submit_probability)
        self._rng = np.random.default_rng(int(seed))

    def _sample_named_navigation_action(self, env: Any) -> Optional[Any]:
        action_names = _extract_action_names(env)
        if not action_names:
            return None
        candidates = [
            name
            for name in action_names
            if name.upper() not in {self._submit_action_name.upper(), self._stop_action_name.upper()}
        ]
        if not candidates:
            candidates = action_names
        choice = str(self._rng.choice(candidates))
        return _build_named_action(choice)

    def _sample_generic_action(self, env: Any) -> Any:
        for _ in range(32):
            action = env.action_space.sample()
            if _is_named_action(action, self._submit_action_name):
                continue
            if _is_named_action(action, self._stop_action_name):
                continue
            if isinstance(action, int) and action == 0:
                continue
            return action
        return env.action_space.sample()

    def _should_submit(self, context: LifelongEvalContext) -> bool:
        if context.step_index < self._min_submit_steps:
            return False
        if self._submit_probability <= 0:
            return False
        return bool(self._rng.random() < self._submit_probability)

    def act(
        self,
        *,
        env: Any,
        episode: Any,
        observations: Dict[str, Any],
        context: LifelongEvalContext,
    ) -> Any:
        if self._should_submit(context):
            return _build_named_action(self._submit_action_name)

        named_action = self._sample_named_navigation_action(env)
        if named_action is not None:
            return named_action
        return self._sample_generic_action(env)


@register_lifelong_eval_policy("distance_submit")
class DistanceSubmitLifelongEvalPolicy(RandomLifelongEvalPolicy):
    def __init__(
        self,
        submit_distance: float = 1.0,
        submit_action_name: str = "LIFELONG_SUBMIT",
        stop_action_name: str = "STOP",
        min_submit_steps: int = 0,
        submit_probability: float = 0.0,
        seed: int = 0,
    ):
        super().__init__(
            submit_action_name=submit_action_name,
            stop_action_name=stop_action_name,
            min_submit_steps=min_submit_steps,
            submit_probability=submit_probability,
            seed=seed,
        )
        self._submit_distance = float(submit_distance)

    def act(
        self,
        *,
        env: Any,
        episode: Any,
        observations: Dict[str, Any],
        context: LifelongEvalContext,
    ) -> Any:
        if context.step_index >= self._min_submit_steps:
            distance = _distance_to_current_goal(env, episode)
            if distance is not None and float(distance) <= self._submit_distance:
                return _build_named_action(self._submit_action_name)
        return super().act(env=env, episode=episode, observations=observations, context=context)


@register_lifelong_eval_policy("oracle")
@register_lifelong_eval_policy("oracle_shortest_submit")
class OracleShortestSubmitLifelongEvalPolicy(LifelongEvalPolicy):
    def __init__(
        self,
        submit_action_name: str = "LIFELONG_SUBMIT",
        stop_action_name: str = "STOP",
        submit_distance: float = 1.0,
        follow_view_points: bool = True,
        follower_goal_radius: float = 1e-3,
        stop_on_error: bool = True,
        min_submit_steps: int = 0,
        submit_probability: float = 0.0,
        seed: int = 0,
        goal_order_mode: Optional[Any] = None,
        **_: Any,
    ):
        self._submit_action_name = str(submit_action_name)
        self._stop_action_name = str(stop_action_name)
        self._submit_distance = float(submit_distance)
        self._follow_view_points = bool(follow_view_points)
        self._follower_goal_radius = max(float(follower_goal_radius), 1e-5)
        self._stop_on_error = bool(stop_on_error)
        self._min_submit_steps = int(min_submit_steps)
        self._submit_probability = float(submit_probability)
        self._seed = int(seed)
        self._goal_order_mode = _normalize_goal_order_mode(goal_order_mode)
        self._follower: Optional[ShortestPathFollower] = None
        self._goal_signature: Optional[Any] = None

    def _reset_follower(self, env: Any) -> None:
        self._follower = ShortestPathFollower(
            sim=env.sim,
            goal_radius=self._follower_goal_radius,
            return_one_hot=False,
            stop_on_error=self._stop_on_error,
        )
        self._goal_signature = None

    def _mark_goal_submitted(self, env: Any, goal_index: Optional[int]) -> None:
        if goal_index is None:
            return
        if _order_enforced(env, policy=self):
            if int(goal_index) == int(self._ordered_goal_index):
                self._ordered_goal_index = min(
                    int(self._ordered_goal_index) + 1,
                    max(len(self._episode_goals) - 1, 0),
                )
        else:
            self._submitted_goal_indices.add(int(goal_index))
        self._goal_signature = None

    def reset(self, *, env: Any, episode: Any, observations: Dict[str, Any]) -> None:
        self._reset_follower(env)
        task = _get_task(env)
        goals = getattr(task, "all_goals", None) if task is not None else None
        if not goals:
            episode_goals = getattr(episode, "goals", None)
            goals = tuple(episode_goals) if isinstance(episode_goals, list) else ()
        self._episode_goals = tuple(goals)
        self._ordered_goal_index = 0
        self._submitted_goal_indices = set()

    def _reached_goal(
        self,
        env: Any,
        episode: Any,
        goal: Any,
    ) -> bool:
        distance = _distance_to_specific_goal(env, episode, goal)
        if distance is None:
            return False
        threshold = self._submit_distance
        if threshold <= 0.0:
            threshold = _task_success_distance(env, default=1.0)
        return float(distance) <= float(threshold)

    def act(
        self,
        *,
        env: Any,
        episode: Any,
        observations: Dict[str, Any],
        context: LifelongEvalContext,
    ) -> Any:
        goal_index, goal = _navigation_goal(env, episode, policy=self)
        if goal is None:
            return _build_named_action(self._stop_action_name)

        if self._reached_goal(env, episode, goal):
            self._mark_goal_submitted(env, goal_index)
            return _build_named_action(self._submit_action_name)

        target_positions = _goal_target_positions(goal, self._follow_view_points)
        target_position = _nearest_target_position(env, episode, target_positions)
        if target_position is None:
            return _build_named_action(self._stop_action_name)

        goal_signature = _goal_signature(goal_index, goal)
        if self._follower is None:
            self._reset_follower(env)
        elif goal_signature != self._goal_signature:
            self._reset_follower(env)

        assert self._follower is not None
        self._goal_signature = goal_signature
        action = self._follower.get_next_action(target_position)
        if action is None:
            return _heading_fallback_action(env, target_position)
        if int(action) == int(HabitatSimActions.STOP):
            if self._reached_goal(env, episode, goal):
                self._mark_goal_submitted(env, goal_index)
                return _build_named_action(self._submit_action_name)
            self._reset_follower(env)
            assert self._follower is not None
            self._goal_signature = goal_signature
            retry_action = self._follower.get_next_action(target_position)
            if retry_action is not None and int(retry_action) != int(HabitatSimActions.STOP):
                return _sim_action_to_named_action(retry_action)
            return _heading_fallback_action(env, target_position)
        return _sim_action_to_named_action(action)


def build_lifelong_eval_context(
    step_index: int,
    goal_payloads: Optional[Sequence[Dict[str, Any]]] = None,
) -> LifelongEvalContext:
    return LifelongEvalContext(
        step_index=int(step_index),
        goal_payloads=tuple(goal_payloads or ()),
    )
