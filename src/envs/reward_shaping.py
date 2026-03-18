"""Configurable reward functions for EV corridor optimization.

Provides modular reward components that can be weighted and combined
to shape the learning signal for different objectives.
"""

from __future__ import annotations

from typing import Any


class RewardFunction:
    """Composable reward function for EV corridor environments.

    Each reward component is computed independently and then combined
    using configurable weights.  The :meth:`compute` method returns both
    the scalar total and a breakdown dict for logging / analysis.

    Parameters
    ----------
    weights : dict[str, float] | None
        Mapping from component name to scalar weight.  Missing keys
        fall back to the built-in defaults.
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "time_penalty": -1.0,
        "queue_penalty": -0.1,
        "intersection_bonus": 5.0,
        "arrival_bonus": 50.0,
        "timeout_penalty": -50.0,
        "ev_wait_penalty": -2.0,
        "green_corridor_bonus": 0.5,
    }

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self.weights = dict(self.DEFAULT_WEIGHTS)
        if weights is not None:
            self.weights.update(weights)

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def compute(self, env_state: dict[str, Any]) -> tuple[float, dict[str, float]]:
        """Compute total reward and per-component breakdown.

        Parameters
        ----------
        env_state : dict
            Snapshot of the environment state.  Expected keys:

            * ``ev_arrived`` (bool)
            * ``truncated`` (bool) — episode timed out
            * ``ev_passed_intersection`` (bool) — EV crossed an intersection this step
            * ``ev_was_blocked`` (bool) — EV was stopped by a red signal
            * ``total_queue`` (float) — aggregate queue proxy across the network
            * ``ev_on_green_streak`` (int) — consecutive intersections the EV
              passed without stopping (0 if it just stopped)
            * ``step`` (int) — current simulation step

        Returns
        -------
        total_reward : float
            Weighted sum of all components.
        components : dict[str, float]
            Each component's *weighted* contribution to the total.
        """
        components: dict[str, float] = {}

        # 1. Per-step time penalty (encourages faster completion)
        components["time_penalty"] = self.weights["time_penalty"]

        # 2. Network-wide queue penalty (discourages congestion build-up)
        total_queue = env_state.get("total_queue", 0.0)
        components["queue_penalty"] = self.weights["queue_penalty"] * total_queue

        # 3. Intersection traversal bonus (shaped progress signal)
        if env_state.get("ev_passed_intersection", False):
            components["intersection_bonus"] = self.weights["intersection_bonus"]
        else:
            components["intersection_bonus"] = 0.0

        # 4. Terminal arrival bonus
        if env_state.get("ev_arrived", False):
            components["arrival_bonus"] = self.weights["arrival_bonus"]
        else:
            components["arrival_bonus"] = 0.0

        # 5. Timeout penalty
        if env_state.get("truncated", False):
            components["timeout_penalty"] = self.weights["timeout_penalty"]
        else:
            components["timeout_penalty"] = 0.0

        # 6. EV waiting penalty (per step the EV is blocked at a red)
        if env_state.get("ev_was_blocked", False):
            components["ev_wait_penalty"] = self.weights["ev_wait_penalty"]
        else:
            components["ev_wait_penalty"] = 0.0

        # 7. Green corridor bonus — reward consecutive green passages
        streak = env_state.get("ev_on_green_streak", 0)
        if env_state.get("ev_passed_intersection", False) and streak > 1:
            # Quadratic scaling: longer streaks are disproportionately rewarded
            components["green_corridor_bonus"] = (
                self.weights["green_corridor_bonus"] * (streak ** 1.5)
            )
        else:
            components["green_corridor_bonus"] = 0.0

        total_reward = sum(components.values())
        return total_reward, components

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_preset(cls, preset_name: str) -> "RewardFunction":
        """Create a :class:`RewardFunction` from a named preset.

        Parameters
        ----------
        preset_name : str
            One of the keys in :data:`REWARD_PRESETS`.

        Raises
        ------
        KeyError
            If the preset name is not recognised.
        """
        if preset_name not in REWARD_PRESETS:
            available = ", ".join(sorted(REWARD_PRESETS.keys()))
            raise KeyError(
                f"Unknown reward preset {preset_name!r}. "
                f"Available: {available}"
            )
        return cls(weights=dict(REWARD_PRESETS[preset_name]))

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RewardFunction":
        """Create from an arbitrary config dict.

        The dict may contain a ``"preset"`` key (loaded first) and/or
        individual weight overrides under a ``"weights"`` key.
        """
        preset = config.get("preset")
        if preset is not None:
            rf = cls.from_preset(preset)
            if "weights" in config:
                rf.weights.update(config["weights"])
            return rf
        return cls(weights=config.get("weights"))

    def __repr__(self) -> str:
        return f"RewardFunction(weights={self.weights})"


# ======================================================================
# Pre-defined reward configurations
# ======================================================================

REWARD_PRESETS: dict[str, dict[str, float]] = {
    "default": {
        "time_penalty": -1.0,
        "queue_penalty": -0.1,
        "intersection_bonus": 5.0,
        "arrival_bonus": 50.0,
        "timeout_penalty": -50.0,
        "ev_wait_penalty": -2.0,
        "green_corridor_bonus": 0.5,
    },
    "ev_priority": {
        "time_penalty": -1.5,
        "queue_penalty": -0.02,
        "intersection_bonus": 8.0,
        "arrival_bonus": 80.0,
        "timeout_penalty": -80.0,
        "ev_wait_penalty": -5.0,
        "green_corridor_bonus": 2.0,
    },
    "balanced": {
        "time_penalty": -1.0,
        "queue_penalty": -0.3,
        "intersection_bonus": 4.0,
        "arrival_bonus": 40.0,
        "timeout_penalty": -40.0,
        "ev_wait_penalty": -1.5,
        "green_corridor_bonus": 0.5,
    },
    "minimal_disruption": {
        "time_penalty": -0.5,
        "queue_penalty": -0.5,
        "intersection_bonus": 3.0,
        "arrival_bonus": 30.0,
        "timeout_penalty": -30.0,
        "ev_wait_penalty": -1.0,
        "green_corridor_bonus": 0.3,
    },
}
