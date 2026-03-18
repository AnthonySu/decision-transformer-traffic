"""Emergency vehicle tracking overlay for traffic networks.

Tracks an EV moving through a road network link-by-link,
independent of the underlying traffic simulation backend.
"""

from __future__ import annotations

from typing import Any

from src.envs.network_utils import Network, Route, signal_is_green_for_link


class EVTracker:
    """Tracks emergency vehicle position and progress through a route.

    The EV moves through the network link-by-link.  Its speed depends
    on the congestion level of the current link and whether the
    downstream signal is green.

    Parameters
    ----------
    network : Network
        The traffic network dictionary (must contain ``"nodes"`` and
        ``"links"`` with CTM-compatible fields).
    route : Route
        Ordered list of ``(node_id, outgoing_link_id | None)`` tuples.
    speed_factor : float
        Multiplier on free-flow speed for the EV (e.g. 1.5 means the
        EV travels 50 % faster than normal traffic in free-flow).
    """

    def __init__(
        self,
        network: Network,
        route: Route,
        speed_factor: float = 1.5,
    ) -> None:
        self.network = network
        self.route = route
        self.speed_factor = speed_factor
        self.reset()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the EV to the start of its route."""
        self.link_idx: int = 0
        self.progress: float = 0.0
        self.arrived: bool = False
        self.intersections_passed: int = 0
        self.total_time: float = 0.0
        self.wait_time: float = 0.0

    # ------------------------------------------------------------------
    # Stepping
    # ------------------------------------------------------------------

    def step(self, dt: float = 5.0) -> dict[str, Any]:
        """Advance the EV by one timestep.

        Parameters
        ----------
        dt : float
            Simulation timestep in seconds.

        Returns
        -------
        dict
            ``passed_intersection`` — True if the EV crossed an
            intersection this step.
            ``blocked`` — True if the EV was held at a red signal.
            ``speed`` — effective EV speed this step (m/s).
            ``arrived`` — True if the EV reached its destination.
        """
        info: dict[str, Any] = {
            "passed_intersection": False,
            "blocked": False,
            "speed": 0.0,
            "arrived": self.arrived,
        }

        self.total_time += dt

        if self.arrived or self.link_idx >= len(self.route) - 1:
            return info

        _, link_id = self.route[self.link_idx]
        if link_id is None:
            # Already at destination node (last entry in route)
            self.arrived = True
            info["arrived"] = True
            return info

        link = self.network["links"][link_id]
        density = link["density"]
        k_jam = link["k_jam"]
        v_free = link["v_free"]
        length = link["length"]

        # EV speed: free-flow * speed_factor * congestion factor
        congestion_factor = max(1.0 - density / k_jam, 0.05)
        ev_speed = v_free * self.speed_factor * congestion_factor

        # Check if EV is blocked at downstream intersection
        downstream_node = link["target"]
        if self.progress > 0.9:
            if not signal_is_green_for_link(self.network, downstream_node, link_id):
                ev_speed = 0.0
                info["blocked"] = True
                self.wait_time += dt

        info["speed"] = ev_speed
        self.progress += ev_speed * dt / length

        # Check if EV has moved to the next link
        if self.progress >= 1.0:
            self.progress = 0.0
            self.link_idx += 1
            self.intersections_passed += 1
            info["passed_intersection"] = True

            if self.link_idx >= len(self.route) - 1:
                self.arrived = True
                info["arrived"] = True

        return info

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def distance_to_intersection(self, intersection_idx: int) -> float:
        """Normalised distance from the EV to the given intersection.

        Parameters
        ----------
        intersection_idx : int
            Index of the intersection within the route.

        Returns
        -------
        float
            Value in ``[-1, 1]``.  Negative means the EV has already
            passed this intersection; 0 means it is there now; positive
            means it is approaching.
        """
        ev_position_continuous = self.link_idx + self.progress
        diff = intersection_idx - ev_position_continuous
        route_len = max(len(self.route) - 1, 1)
        return float(diff / route_len)

    @property
    def current_link_id(self) -> str | None:
        """The link the EV is currently traversing, or ``None`` if arrived."""
        if self.arrived or self.link_idx >= len(self.route):
            return None
        _, link_id = self.route[self.link_idx]
        return link_id

    @property
    def current_node_id(self) -> str:
        """The upstream node of the link the EV is currently on."""
        idx = min(self.link_idx, len(self.route) - 1)
        node_id, _ = self.route[idx]
        return node_id

    @property
    def position_fraction(self) -> float:
        """EV's position as a fraction of the total route ``[0, 1]``."""
        route_len = max(len(self.route) - 1, 1)
        return float((self.link_idx + self.progress) / route_len)
