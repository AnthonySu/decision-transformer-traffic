"""MaxPressure adaptive traffic signal control baseline.

MaxPressure is a classic decentralised adaptive signal control method.
At each intersection it selects the phase that maximises *pressure* —
the difference between upstream and downstream vehicle densities for
the movements served by that phase.

Reference:
    Varaiya, P. (2013).  "Max pressure control of a network of
    signalized intersections."  *Transportation Research Part C*, 36,
    pp. 177-195.

This implementation works directly with the ``Network`` dict produced by
:mod:`src.envs.network_utils` and is compatible with :class:`EVCorridorEnv`.
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from src.envs.network_utils import Network


class MaxPressurePolicy:
    """MaxPressure adaptive signal control.

    At each intersection, selects the phase that maximises pressure:

        pressure(phase) = sum over movements in phase of
                          (upstream_density - downstream_density)

    Optionally gives priority to the emergency vehicle's current link
    (force green for the EV if ``ev_priority=True``).

    Parameters
    ----------
    ev_priority : bool
        If *True*, override MaxPressure at the intersection the EV is
        currently approaching and force the phase that serves the EV link.
    """

    def __init__(self, ev_priority: bool = False) -> None:
        self.ev_priority = ev_priority

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def act(
        self,
        obs: np.ndarray,
        env: Any,
    ) -> np.ndarray:
        """Select an action compatible with ``env.action_space``.

        Parameters
        ----------
        obs : np.ndarray
            Current observation from the environment (unused — we read
            densities directly from the network).
        env : EVCorridorEnv
            The environment instance.  We access ``env.network`` for link
            densities and ``env.ev_route`` for the EV's planned path.

        Returns
        -------
        np.ndarray
            Action array (one phase per route intersection, padded to
            ``env.action_space`` length).
        """
        network: Network = env.network
        route = env.ev_route
        route_intersections: List[str] = [node for node, _ in route]

        # Determine which link the EV is currently on and which
        # intersection it is approaching.
        ev_link_id: Optional[str] = None
        ev_approaching_node: Optional[str] = None
        if hasattr(env, "_ev_link_idx") and not env._ev_arrived:
            idx = env._ev_link_idx
            if idx < len(route) - 1:
                _, link_id = route[idx]
                if link_id is not None:
                    ev_link_id = link_id
                    ev_approaching_node = network["links"][link_id]["target"]

        # Build action array
        max_route_len = env.action_space.nvec.shape[0]
        action = np.zeros(max_route_len, dtype=np.int64)

        for i, node_id in enumerate(route_intersections):
            if i >= max_route_len:
                break

            # EV priority override
            if (
                self.ev_priority
                and ev_approaching_node is not None
                and node_id == ev_approaching_node
                and ev_link_id is not None
            ):
                ev_link = network["links"][ev_link_id]
                action[i] = ev_link["phase_index"]
                continue

            action[i] = self._max_pressure_phase(network, node_id)

        return action

    # ------------------------------------------------------------------
    # Pressure computation
    # ------------------------------------------------------------------

    def _max_pressure_phase(self, network: Network, node_id: str) -> int:
        """Compute pressure for each phase at *node_id* and return the argmax.

        Pressure for a phase is the sum over incoming links whose
        ``phase_index`` matches that phase of:

            upstream_density - mean(downstream_densities)

        where *downstream_densities* are the densities of all outgoing
        links from the same intersection.
        """
        node = network["nodes"][node_id]
        num_phases: int = node.get("num_phases", 4)
        links = network["links"]

        # Pre-compute mean downstream (outgoing) density
        outgoing_densities: List[float] = []
        for lid in node["outgoing_links"]:
            outgoing_densities.append(links[lid]["density"])
        if outgoing_densities:
            mean_downstream = float(np.mean(outgoing_densities))
        else:
            mean_downstream = 0.0

        # Accumulate pressure per phase
        pressure = np.zeros(num_phases, dtype=np.float64)
        for lid in node["incoming_links"]:
            lk = links[lid]
            phase_idx = lk["phase_index"]
            if phase_idx < num_phases:
                pressure[phase_idx] += lk["density"] - mean_downstream

        return int(np.argmax(pressure))

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"MaxPressurePolicy(ev_priority={self.ev_priority})"
