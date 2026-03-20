from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ss_baselines.common.omni_long_eval_policy import LifelongEvalContext, LifelongEvalPolicy, register_lifelong_eval_policy
from ss_baselines.omega_nav.memory import HierarchicalMemory
from ss_baselines.omega_nav.navigator import LocalNavigator
from ss_baselines.omega_nav.perception import GoalSpec, PerceptionEncoder
from ss_baselines.omega_nav.planner import OmegaLLMPlanner, PlannerDecision
from ss_baselines.omega_nav.utils import as_serializable, load_omega_config


def _named_action(action_name: str) -> Dict[str, str]:
    return {"action": str(action_name)}


def _normalize_goal_order_mode(value: Any) -> Optional[str]:
    if value is None:
        return None
    token = str(value).strip().lower()
    if token in {"ordered", "order", "true", "1", "yes", "y"}:
        return "ordered"
    if token in {"unordered", "unorder", "false", "0", "no", "n"}:
        return "unordered"
    return None


def _task_order_mode(env: Any, override: Optional[str] = None) -> str:
    normalized = _normalize_goal_order_mode(override)
    if normalized is not None:
        return normalized
    task = getattr(env, "task", None) or getattr(env, "_task", None)
    if task is None:
        return "ordered"
    if hasattr(task, "goal_order_mode"):
        normalized = _normalize_goal_order_mode(getattr(task, "goal_order_mode"))
        if normalized is not None:
            return normalized
    if hasattr(task, "order_enforced"):
        return "ordered" if bool(task.order_enforced) else "unordered"
    return "ordered"


def _goal_submit_distance(env: Any, default: float) -> float:
    task = getattr(env, "task", None) or getattr(env, "_task", None)
    task_cfg = getattr(task, "_config", None)
    success_cfg = getattr(task_cfg, "SUCCESS", None)
    value = getattr(success_cfg, "SUCCESS_DISTANCE", default)
    if isinstance(value, (int, float, np.integer, np.floating)):
        parsed = float(value)
        if np.isfinite(parsed):
            return float(parsed)
    return float(default)


@register_lifelong_eval_policy("omega_nav")
@register_lifelong_eval_policy("omega_nav_policy")
@register_lifelong_eval_policy("omega_oracle")
@register_lifelong_eval_policy("omega_nav_oracle")
class OmegaNavPolicy(LifelongEvalPolicy):
    def __init__(
        self,
        submit_action_name: str = "LIFELONG_SUBMIT",
        stop_action_name: str = "STOP",
        submit_distance: float = 0.0,
        goal_order_mode: Optional[Any] = None,
        config_path: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
        **_: Any,
    ):
        default_config_path = Path(__file__).resolve().parent / "configs" / "perception.yaml"
        self._config = load_omega_config(
            str(config_path or default_config_path),
            overrides=config_overrides,
        )
        self._submit_action_name = str(submit_action_name or self._config.get("policy", {}).get("submit_action_name", "LIFELONG_SUBMIT"))
        self._stop_action_name = str(stop_action_name or self._config.get("policy", {}).get("stop_action_name", "STOP"))
        self._submit_distance = float(submit_distance)
        self._goal_order_mode = _normalize_goal_order_mode(goal_order_mode)

        self._encoder = PerceptionEncoder(self._config)
        self._memory = HierarchicalMemory(self._config.get("memory", {}))
        self._planner = OmegaLLMPlanner(self._config.get("planner", {}))
        self._navigator = LocalNavigator(self._config.get("navigator", {}))

        self._goal_specs: List[GoalSpec] = []
        self._goal_specs_by_id: Dict[str, GoalSpec] = {}
        self._submitted_goal_ids: set[str] = set()
        self._ordered_goal_index = 0
        self._last_action_name: Optional[str] = None
        self._last_submit_goal_id: Optional[str] = None
        self._last_perception = None
        self._last_decision: Optional[PlannerDecision] = None
        self._last_order_mode = "ordered"

    def _sync_goal_specs(self, env: Any, episode: Any, context: LifelongEvalContext) -> None:
        if self._goal_specs:
            return
        self._goal_specs = self._encoder.build_goal_specs(episode, context.goal_payloads)
        self._goal_specs_by_id = {goal.goal_id: goal for goal in self._goal_specs}
        self._encoder.reset(env=env, goal_specs=self._goal_specs)
        self._memory.reset(self._goal_specs)

    def _pending_goal_ids(self, order_mode: str) -> List[str]:
        if str(order_mode) == "ordered":
            return [goal.goal_id for goal in self._goal_specs[self._ordered_goal_index :]]
        return [goal.goal_id for goal in self._goal_specs if goal.goal_id not in self._submitted_goal_ids]

    def _advance_goal(self, goal_id: str, order_mode: str) -> None:
        if str(order_mode) == "ordered":
            goal = self._goal_specs_by_id.get(goal_id)
            if goal is not None and int(goal.goal_index) == int(self._ordered_goal_index):
                self._ordered_goal_index = min(int(self._ordered_goal_index) + 1, len(self._goal_specs))
        else:
            self._submitted_goal_ids.add(str(goal_id))
        self._memory.confirm_goal(goal_id)

    def reset(self, *, env: Any, episode: Any, observations: Dict[str, Any]) -> None:
        self._goal_specs = []
        self._goal_specs_by_id = {}
        self._submitted_goal_ids = set()
        self._ordered_goal_index = 0
        self._last_action_name = None
        self._last_submit_goal_id = None
        self._last_perception = None
        self._last_decision = None
        self._last_order_mode = _task_order_mode(env, override=self._goal_order_mode)
        self._encoder.reset(env=env)
        self._navigator.reset()

    def act(
        self,
        *,
        env: Any,
        episode: Any,
        observations: Dict[str, Any],
        context: LifelongEvalContext,
    ) -> Any:
        self._sync_goal_specs(env, episode, context)
        order_mode = _task_order_mode(env, override=self._goal_order_mode)
        self._last_order_mode = order_mode
        pending_goal_ids = self._pending_goal_ids(order_mode)
        if not pending_goal_ids:
            self._last_action_name = self._stop_action_name
            return _named_action(self._stop_action_name)

        perception = self._encoder.encode(
            step_index=int(context.step_index),
            env=env,
            episode=episode,
            observations=observations,
            goal_specs=self._goal_specs,
            pending_goal_ids=pending_goal_ids,
            order_mode=order_mode,
        )
        self._memory.update(
            step_index=int(context.step_index),
            env=env,
            perception=perception,
            pending_goal_ids=pending_goal_ids,
            order_mode=order_mode,
        )
        self._last_perception = perception

        effective_submit_distance = self._submit_distance if self._submit_distance > 0.0 else _goal_submit_distance(env, 1.0)
        decision = self._planner.decide(
            env=env,
            episode=episode,
            goal_specs=self._goal_specs,
            perception=perception,
            memory=self._memory,
            pending_goal_ids=pending_goal_ids,
            order_mode=order_mode,
            step_index=int(context.step_index),
            submit_distance_m=effective_submit_distance,
        )
        self._last_decision = decision

        if decision.mark_complete:
            self._last_submit_goal_id = decision.mark_complete[0]
            self._last_action_name = self._submit_action_name
            return _named_action(self._submit_action_name)

        if decision.next_goal is None:
            self._last_action_name = self._stop_action_name
            return _named_action(self._stop_action_name)

        action = self._navigator.act(
            env=env,
            episode=episode,
            target_positions=decision.target_positions,
            target_signature=(str(decision.action), str(decision.next_goal)),
            fallback_direction=decision.direction,
        )
        if isinstance(action, dict):
            self._last_action_name = str(action.get("action"))
        else:
            self._last_action_name = None
        self._last_submit_goal_id = None
        return action

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
        task = getattr(env, "task", None) or getattr(env, "_task", None)
        order_mode = _task_order_mode(env, override=self._goal_order_mode)
        if self._last_action_name == self._submit_action_name and task is not None and hasattr(task, "get_last_action_feedback"):
            feedback = task.get_last_action_feedback()
            found_goal = int(feedback.get("found_goal_index", -1)) if isinstance(feedback, dict) else -1
            if found_goal >= 0 and 0 <= found_goal < len(self._goal_specs):
                goal_id = self._goal_specs[found_goal].goal_id
                self._advance_goal(goal_id, order_mode)
            elif self._last_submit_goal_id is not None:
                self._memory.penalize_goal(self._last_submit_goal_id)
        if done:
            self._navigator.reset()

    def get_debug_state(self) -> Dict[str, Any]:
        return {
            "pending_goals": self._pending_goal_ids(self._last_order_mode) if self._goal_specs else [],
            "submitted_goals": sorted(self._submitted_goal_ids),
            "ordered_goal_index": int(self._ordered_goal_index),
            "perception": self._last_perception.to_dict() if self._last_perception is not None else None,
            "plan": self._last_decision.to_dict() if self._last_decision is not None else None,
            "memory": self._memory.to_prompt_summary() if self._goal_specs else None,
        }

    def close(self) -> None:
        self._navigator.reset()
