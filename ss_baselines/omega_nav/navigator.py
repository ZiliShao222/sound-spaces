from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np

from ss_baselines.omega_nav.utils import coarse_direction_from_angle, relative_bearing_deg


STOP_ACTION_ID = 0
MOVE_FORWARD_ACTION_ID = 1
TURN_LEFT_ACTION_ID = 2
TURN_RIGHT_ACTION_ID = 3


def _named_action(action_name: str) -> Dict[str, str]:
    return {"action": str(action_name)}


def _direction_action(direction: str) -> Dict[str, str]:
    token = str(direction).strip().lower()
    if token == "left":
        return _named_action("TURN_LEFT")
    if token == "right":
        return _named_action("TURN_RIGHT")
    return _named_action("MOVE_FORWARD")


def _sim_action_to_named(action: Any) -> Any:
    if not isinstance(action, (int, np.integer)):
        return action
    action_id = int(action)
    if action_id == int(MOVE_FORWARD_ACTION_ID):
        return _named_action("MOVE_FORWARD")
    if action_id == int(TURN_LEFT_ACTION_ID):
        return _named_action("TURN_LEFT")
    if action_id == int(TURN_RIGHT_ACTION_ID):
        return _named_action("TURN_RIGHT")
    if action_id == int(STOP_ACTION_ID):
        return _named_action("STOP")
    return action


class LocalNavigator:
    def __init__(self, config: Dict[str, Any]):
        self._follower_goal_radius = max(float(config.get("follower_goal_radius", 1e-3)), 1e-5)
        self._follower: Optional[Any] = None
        self._target_signature: Optional[Tuple[str, str]] = None

    def reset(self) -> None:
        self._follower = None
        self._target_signature = None

    def _ensure_follower(self, env: Any) -> None:
        from soundspaces.tasks.shortest_path_follower import ShortestPathFollower

        if self._follower is None:
            self._follower = ShortestPathFollower(
                sim=env.sim,
                goal_radius=self._follower_goal_radius,
                return_one_hot=False,
            )

    def _nearest_target(self, env: Any, episode: Any, targets: Sequence[np.ndarray]) -> Optional[np.ndarray]:
        if not targets:
            return None
        current = np.asarray(env.sim.get_agent_state().position, dtype=np.float32)
        best_target = None
        best_distance = None
        for target in targets:
            candidate = np.asarray(target, dtype=np.float32)
            distance = env.sim.geodesic_distance(current, [candidate], episode)
            if distance is None or not np.isfinite(distance):
                continue
            if best_distance is None or float(distance) < float(best_distance):
                best_target = candidate
                best_distance = float(distance)
        if best_target is not None:
            return best_target
        return np.asarray(targets[0], dtype=np.float32)

    def act(
        self,
        *,
        env: Any,
        episode: Any,
        target_positions: Sequence[np.ndarray],
        target_signature: Tuple[str, str],
        fallback_direction: str,
    ) -> Any:
        if not target_positions:
            return _direction_action(fallback_direction)

        if env is None or not hasattr(env, "sim"):
            return _direction_action(fallback_direction)

        self._ensure_follower(env)
        if self._target_signature != tuple(target_signature):
            self._follower = None
            self._ensure_follower(env)
            self._target_signature = tuple(target_signature)

        target = self._nearest_target(env, episode, target_positions)
        if target is None:
            return _direction_action(fallback_direction)

        assert self._follower is not None
        action = self._follower.get_next_action(target)
        if action is None:
            state = env.sim.get_agent_state()
            angle_deg = relative_bearing_deg(np.asarray(state.position, dtype=np.float32), state.rotation, target)
            return _direction_action(coarse_direction_from_angle(angle_deg))
        named = _sim_action_to_named(action)
        if isinstance(named, dict) and named.get("action") == "STOP":
            return _direction_action(fallback_direction)
        return named
