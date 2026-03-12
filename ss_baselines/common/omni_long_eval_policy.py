#!/usr/bin/env python3

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Type

import numpy as np


def _get_task(env: Any) -> Any:
    task = getattr(env, "task", None)
    if task is not None:
        return task
    return getattr(env, "_task", None)


def _get_task_attr(env: Any, name: str, default: Any = None) -> Any:
    task = _get_task(env)
    if task is None:
        return default
    return getattr(task, name, default)


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


def _distance_to_current_goal(env: Any, episode: Any) -> Optional[float]:
    goals = getattr(episode, "goals", None)
    if not isinstance(goals, list) or len(goals) == 0:
        return None

    goal = goals[0]
    task = _get_task(env)
    if task is not None and hasattr(task, "_distance_to_goal"):
        try:
            distance = task._distance_to_goal(goal, episode)
            if distance is None:
                return None
            return float(distance)
        except Exception:
            pass

    try:
        goal_pos = getattr(goal, "position", None)
        if goal_pos is None:
            return None
        current_pos = env.sim.get_agent_state().position
        distance = env.sim.geodesic_distance(current_pos, [goal_pos], episode)
        return float(distance)
    except Exception:
        return None


@dataclass
class LifelongEvalContext:
    episode_index: int
    step_index: int
    episode_id: str
    scene_id: str
    current_goal_index: Optional[int]
    current_task_token: Optional[List[str]]
    completed_goal_indices: Sequence[int]
    remaining_goal_indices: Sequence[int]


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


def build_lifelong_eval_context(
    env: Any,
    episode: Any,
    episode_index: int,
    step_index: int,
) -> LifelongEvalContext:
    current_goal_index = _get_task_attr(env, "_current_goal_index", None)
    current_task_token = _get_task_attr(env, "current_task_token", None)
    completed_goal_indices = _get_task_attr(env, "completed_goal_indices", ())
    remaining_goal_indices = _get_task_attr(env, "remaining_goal_indices", ())

    return LifelongEvalContext(
        episode_index=int(episode_index),
        step_index=int(step_index),
        episode_id=str(getattr(episode, "episode_id", "")),
        scene_id=str(getattr(episode, "scene_id", "")),
        current_goal_index=int(current_goal_index) if isinstance(current_goal_index, int) else None,
        current_task_token=list(current_task_token) if isinstance(current_task_token, list) else None,
        completed_goal_indices=tuple(int(x) for x in completed_goal_indices),
        remaining_goal_indices=tuple(int(x) for x in remaining_goal_indices),
    )
