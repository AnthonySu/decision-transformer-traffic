"""Fixed-timing Emergency Vehicle Preemption (EVP) baseline.

When an EV is detected within a configurable distance of an intersection,
the controller immediately switches to the EV's phase, holds green until
the EV passes, then reverts to a fixed-timing plan with equal phase splits.
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class FixedTimeEVP:
    """Fixed-timing signal controller with naive EV preemption.

    Normal operation cycles through phases with equal durations.  When an
    emergency vehicle is detected within ``preemption_distance`` cells of the
    intersection, the controller overrides to the EV's requested phase and
    holds it until the EV clears the intersection.

    Parameters
    ----------
    preemption_distance : int
        Number of cells ahead at which the EV triggers preemption.
    phase_duration : int
        Duration (in decision steps) of each phase under the fixed plan.
    num_phases : int
        Total number of signal phases to cycle through.
    """

    def __init__(
        self,
        preemption_distance: int = 3,
        phase_duration: int = 30,
        num_phases: int = 4,
    ) -> None:
        self.preemption_distance = preemption_distance
        self.phase_duration = phase_duration
        self.num_phases = num_phases

        # Internal state
        self._current_phase: int = 0
        self._timer: int = 0
        self._preempting: bool = False
        self._preempt_phase: Optional[int] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select_action(self, obs: Any, ev_info: Dict[str, Any]) -> int:
        """Choose the next signal phase.

        Parameters
        ----------
        obs : array-like
            Observation from the environment (unused by this rule-based
            controller, kept for interface compatibility).
        ev_info : dict
            Emergency-vehicle information dict.  Expected keys:

            * ``"distance"`` – cells between EV and the intersection.
            * ``"phase"``    – phase index the EV needs (its approach direction).
            * ``"active"``   – bool, whether an EV is currently in the network.

        Returns
        -------
        int
            The signal phase to activate.
        """
        ev_active = ev_info.get("active", False)
        ev_distance = ev_info.get("distance", float("inf"))
        ev_phase = ev_info.get("phase", None)

        # ----- Preemption logic -----
        if ev_active and ev_distance <= self.preemption_distance and ev_phase is not None:
            self._preempting = True
            self._preempt_phase = ev_phase
            return self._preempt_phase

        # EV has passed – exit preemption and resume fixed plan
        if self._preempting:
            self._preempting = False
            self._preempt_phase = None
            # Reset the timer so the fixed plan restarts cleanly
            self._timer = 0

        # ----- Fixed-timing plan -----
        self._timer += 1
        if self._timer >= self.phase_duration:
            self._timer = 0
            self._current_phase = (self._current_phase + 1) % self.num_phases

        return self._current_phase

    def reset(self) -> None:
        """Reset internal state for a new episode."""
        self._current_phase = 0
        self._timer = 0
        self._preempting = False
        self._preempt_phase = None

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FixedTimeEVP(preemption_distance={self.preemption_distance}, "
            f"phase_duration={self.phase_duration}, "
            f"num_phases={self.num_phases})"
        )
