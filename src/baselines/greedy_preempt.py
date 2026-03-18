"""Greedy preemption 'expert' policy for data collection.

This policy always grants green to the emergency vehicle's approach
direction at intersections along the EV route.  At intersections *not*
on the EV path it falls back to MaxPressure control, which selects the
phase that relieves the greatest difference in queue pressure between
incoming and outgoing links.  Together this produces near-optimal
corridor clearing for offline demonstration collection.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np


class GreedyPreemptPolicy:
    """Near-optimal heuristic: green-wave for the EV, MaxPressure elsewhere.

    Parameters
    ----------
    network : object
        Network/topology object that exposes intersection and link metadata.
        Expected attributes / methods:

        * ``intersections`` – iterable of intersection ids.
        * ``get_incoming_counts(intersection, obs)`` – per-phase incoming
          queue counts.
        * ``get_outgoing_counts(intersection, obs)`` – per-phase outgoing
          queue counts.
        * ``num_phases(intersection)`` – number of phases at an intersection.
    route : Sequence[str]
        Ordered sequence of intersection ids along the EV planned route.
    """

    def __init__(self, network: Any, route: Sequence[str]) -> None:
        self.network = network
        self.route: List[str] = list(route)
        self._route_set: set = set(route)

    # ------------------------------------------------------------------
    # Single-agent interface
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, ev_info: Dict[str, Any]) -> int:
        """Select a phase for a single intersection.

        This is a convenience method when only one intersection is controlled.
        It delegates to :meth:`_phase_for_intersection` using the first
        intersection on the EV route.

        Parameters
        ----------
        obs : np.ndarray
            Observation vector for the intersection.
        ev_info : dict
            Emergency-vehicle state.  Expected keys:

            * ``"phase"``          – phase the EV needs at this intersection.
            * ``"active"``         – bool, EV is in the network.
            * ``"intersection_id"`` – id of the intersection being controlled.

        Returns
        -------
        int
            Phase index to activate.
        """
        intersection_id = ev_info.get("intersection_id", self.route[0])
        return self._phase_for_intersection(intersection_id, obs, ev_info)

    # ------------------------------------------------------------------
    # Multi-agent interface
    # ------------------------------------------------------------------

    def select_actions_multi_agent(
        self,
        obs_dict: Dict[str, np.ndarray],
        ev_info: Dict[str, Any],
    ) -> Dict[str, int]:
        """Select phases for every controlled intersection simultaneously.

        Parameters
        ----------
        obs_dict : dict[str, np.ndarray]
            Mapping from intersection id to its observation vector.
        ev_info : dict
            Global EV state.  Expected keys:

            * ``"active"``             – bool.
            * ``"current_intersection"`` – id of the intersection the EV is
              currently approaching (or ``None``).
            * ``"phase_map"``          – dict mapping intersection id to the
              phase the EV needs there.

        Returns
        -------
        dict[str, int]
            Mapping from intersection id to the chosen phase.
        """
        actions: Dict[str, int] = {}
        for intersection_id, obs in obs_dict.items():
            # Build a per-intersection ev_info view
            local_ev_info = self._local_ev_info(intersection_id, ev_info)
            actions[intersection_id] = self._phase_for_intersection(
                intersection_id, obs, local_ev_info
            )
        return actions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _phase_for_intersection(
        self,
        intersection_id: str,
        obs: np.ndarray,
        ev_info: Dict[str, Any],
    ) -> int:
        """Decide a phase for one intersection.

        On-route intersections with an active EV use the EV's requested
        phase.  All other intersections fall back to MaxPressure.
        """
        ev_active = ev_info.get("active", False)
        ev_phase: Optional[int] = ev_info.get("phase", None)

        # Preempt for EV if this intersection is on the route
        if ev_active and intersection_id in self._route_set and ev_phase is not None:
            return ev_phase

        # Fallback: MaxPressure
        return self._max_pressure(intersection_id, obs)

    def _max_pressure(self, intersection_id: str, obs: np.ndarray) -> int:
        """MaxPressure phase selection.

        Selects the phase whose incoming queue count minus outgoing queue
        count is largest, i.e. the phase that would relieve the most
        pressure.  When the network object is not available or does not
        expose the required methods we fall back to a simple heuristic
        that picks the phase with the highest raw observation value.
        """
        try:
            incoming = np.asarray(
                self.network.get_incoming_counts(intersection_id, obs)
            )
            outgoing = np.asarray(
                self.network.get_outgoing_counts(intersection_id, obs)
            )
            pressure = incoming - outgoing
            return int(np.argmax(pressure))
        except (AttributeError, TypeError):
            # Graceful degradation: treat obs as per-phase queue lengths
            num_phases = self._num_phases(intersection_id)
            if obs.size >= num_phases:
                per_phase = obs[:num_phases]
            else:
                per_phase = obs
            return int(np.argmax(per_phase))

    def _num_phases(self, intersection_id: str) -> int:
        """Return the number of phases at an intersection."""
        try:
            return self.network.num_phases(intersection_id)
        except (AttributeError, TypeError):
            return 4  # sensible default for a standard 4-way intersection

    @staticmethod
    def _local_ev_info(
        intersection_id: str, ev_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Build a per-intersection EV info dict from the global one."""
        phase_map: Dict[str, int] = ev_info.get("phase_map", {})
        return {
            "active": ev_info.get("active", False),
            "phase": phase_map.get(intersection_id, None),
            "intersection_id": intersection_id,
        }

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"GreedyPreemptPolicy(route_len={len(self.route)}, "
            f"route={self.route[:3]}{'...' if len(self.route) > 3 else ''})"
        )
