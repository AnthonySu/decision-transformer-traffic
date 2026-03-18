"""Single-agent Gymnasium environment for EV corridor optimization.

A centralized controller manages ALL traffic signals along an emergency
vehicle's pre-computed route.  The EV is tracked as an overlay on top of
a macroscopic Cell Transmission Model (CTM) traffic simulation.

Two backends are supported via the ``use_lightsim`` flag:
    * **True** — delegates traffic simulation to a ``lightsim`` env
      (must be installed separately).
    * **False** (default) — uses a built-in grid CTM from
      :mod:`src.envs.network_utils`.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.envs.network_utils import (
    Network,
    Route,
    build_grid_network,
    compute_shortest_path,
    ctm_step,
    get_route_intersections,
    get_total_queue_length,
    random_od_pair,
    reset_densities,
    signal_is_green_for_link,
)


class EVCorridorEnv(gym.Env):
    """Gymnasium env for centralized EV corridor signal control.

    Parameters
    ----------
    rows, cols : int
        Grid size when using the built-in mock network.
    use_lightsim : bool
        If True, wrap a ``lightsim`` environment for traffic dynamics.
        If False, use the internal CTM grid simulator.
    max_steps : int
        Episode time-out in simulation steps.
    dt : float
        Simulation time-step in seconds.
    ev_speed_factor : float
        Multiplier on free-flow speed for the EV (e.g. 1.5).
    origin, destination : str | None
        Fixed OD pair.  If None, a random boundary pair is chosen each reset.
    return_to_go_scale : float
        Normalisation constant for the return-to-go feature.
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(
        self,
        rows: int = 4,
        cols: int = 4,
        use_lightsim: bool = False,
        max_steps: int = 200,
        dt: float = 5.0,
        ev_speed_factor: float = 1.5,
        origin: str | None = None,
        destination: str | None = None,
        return_to_go_scale: float = 100.0,
        render_mode: str | None = None,
        seed: int | None = None,
        # Aliases accepted by training scripts
        network_type: str | None = None,
        grid_rows: int | None = None,
        grid_cols: int | None = None,
        max_episode_steps: int | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.rows = grid_rows or rows
        self.cols = grid_cols or cols
        if max_episode_steps is not None:
            max_steps = max_episode_steps
        self.use_lightsim = use_lightsim
        self.max_steps = max_steps
        self.dt = dt
        self.ev_speed_factor = ev_speed_factor
        self._fixed_origin = origin
        self._fixed_destination = destination
        self.return_to_go_scale = return_to_go_scale
        self.render_mode = render_mode

        # RNG
        self._rng = np.random.default_rng(seed)

        # Build the network once (topology is static; densities are reset each episode)
        self._network: Network = build_grid_network(rows=rows, cols=cols)

        # Placeholder route — will be set in reset()
        self._route: Route = []
        self._route_intersections: list[str] = []
        self._num_route_intersections: int = 0

        # --- Determine max possible route length for space sizing ---
        # Worst case: traverse every node in the grid
        self._max_route_len = rows * cols
        max_incoming = 4  # at most 4 incoming links per intersection

        # Per-intersection obs dimension:
        #   one-hot phase (4) + incoming densities (4) + ev_distance (1)
        #   + time_norm (1) + return_to_go (1) = 11
        self._per_intersection_obs = 4 + max_incoming + 1 + 1 + 1
        obs_size = self._max_route_len * self._per_intersection_obs

        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # Action: one phase choice per route intersection
        # Flatten to single Discrete: product of num_phases across intersections.
        # With 4 phases and up to max_route_len intersections this is huge,
        # so we use MultiDiscrete instead.
        self.action_space = spaces.MultiDiscrete(
            [4] * self._max_route_len  # padded; unused slots are ignored
        )

        # --- EV state ---
        self._ev_link_idx: int = 0  # index into self._route
        self._ev_progress: float = 0.0  # [0, 1) on current link
        self._ev_arrived: bool = False
        self._step_count: int = 0
        self._return_to_go: float = 0.0
        self._intersections_passed: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Reset traffic
        reset_densities(self._network, base_density=self._rng.uniform(0.01, 0.04))

        # Reset signal phases randomly
        for node in self._network["nodes"].values():
            node["current_phase"] = int(self._rng.integers(0, node["num_phases"]))

        # Pick OD pair
        if self._fixed_origin and self._fixed_destination:
            origin, destination = self._fixed_origin, self._fixed_destination
        else:
            origin, destination = random_od_pair(self._network, rng=self._rng)

        self._route = compute_shortest_path(self._network, origin, destination)
        self._route_intersections = get_route_intersections(self._network, self._route)
        self._num_route_intersections = len(self._route_intersections)

        # EV starts at the first link of the route
        self._ev_link_idx = 0
        self._ev_progress = 0.0
        self._ev_arrived = False
        self._step_count = 0
        self._intersections_passed = 0

        # Initial return-to-go estimate (will be overwritten by DT during inference)
        self._return_to_go = self.return_to_go_scale

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        self._step_count += 1

        # --- Apply actions: set signal phases along route ---
        # Accept scalar (single phase applied to all) or array actions
        if np.isscalar(action) or (hasattr(action, 'ndim') and action.ndim == 0):
            action = np.full(self._max_route_len, int(action), dtype=np.int64)
        for i, node_id in enumerate(self._route_intersections):
            if i < len(action):
                phase = int(action[i]) % self._network["nodes"][node_id]["num_phases"]
                self._network["nodes"][node_id]["current_phase"] = phase

        # --- Run one CTM traffic step ---
        ctm_step(self._network, dt=self.dt)

        # --- Advance EV ---
        reward = -1.0  # time penalty
        ev_passed_intersection = False

        if not self._ev_arrived and self._ev_link_idx < len(self._route) - 1:
            _, link_id = self._route[self._ev_link_idx]
            if link_id is not None:
                link = self._network["links"][link_id]
                density = link["density"]
                k_jam = link["k_jam"]
                v_free = link["v_free"]
                length = link["length"]

                # EV speed: free-flow * speed_factor * congestion factor
                congestion_factor = max(1.0 - density / k_jam, 0.05)
                ev_speed = v_free * self.ev_speed_factor * congestion_factor

                # Check if EV is blocked at downstream intersection
                downstream_node = link["target"]
                if self._ev_progress > 0.9:
                    # Near the end of the link — check signal
                    if not signal_is_green_for_link(
                        self._network, downstream_node, link_id
                    ):
                        ev_speed = 0.0  # blocked by red

                self._ev_progress += ev_speed * self.dt / length

                # Check if EV has passed to next link
                if self._ev_progress >= 1.0:
                    self._ev_progress = 0.0
                    self._ev_link_idx += 1
                    self._intersections_passed += 1
                    ev_passed_intersection = True

                    # Check arrival
                    if self._ev_link_idx >= len(self._route) - 1:
                        self._ev_arrived = True

        # --- Reward components ---
        queue_penalty = -0.1 * get_total_queue_length(self._network)
        intersection_bonus = 5.0 if ev_passed_intersection else 0.0
        terminal_bonus = 0.0

        terminated = False
        truncated = False

        if self._ev_arrived:
            terminal_bonus = 50.0
            terminated = True
        elif self._step_count >= self.max_steps:
            terminal_bonus = -50.0
            truncated = True

        reward += queue_penalty + intersection_bonus + terminal_bonus

        # Update return-to-go (for DT conditioning in obs)
        self._return_to_go -= reward

        obs = self._get_obs()
        info = self._get_info()
        return obs, float(reward), terminated, truncated, info

    def render(self) -> str | None:
        """Text-based rendering of the EV's progress."""
        lines: list[str] = []
        lines.append(f"Step {self._step_count}/{self.max_steps}")
        lines.append(
            f"EV: link {self._ev_link_idx}/{len(self._route)-1}  "
            f"progress {self._ev_progress:.2f}  arrived={self._ev_arrived}"
        )
        lines.append(f"Intersections passed: {self._intersections_passed}")
        lines.append(f"Return-to-go: {self._return_to_go:.1f}")

        # Mini grid showing signal phases
        grid = [["." for _ in range(self.cols)] for _ in range(self.rows)]
        for nid in self._route_intersections:
            node = self._network["nodes"][nid]
            r, c = node["row"], node["col"]
            grid[r][c] = str(node["current_phase"])

        # Mark EV position
        if not self._ev_arrived and self._ev_link_idx < len(self._route):
            ev_node_id = self._route_intersections[
                min(self._ev_link_idx, self._num_route_intersections - 1)
            ]
            ev_node = self._network["nodes"][ev_node_id]
            grid[ev_node["row"]][ev_node["col"]] = "*"

        lines.append("Grid (phase / * = EV):")
        for row in grid:
            lines.append("  " + " ".join(row))

        text = "\n".join(lines)
        if self.render_mode == "human":
            print(text)
        return text

    # ------------------------------------------------------------------
    # Observation construction
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        """Build the flat observation vector.

        Layout per intersection (self._per_intersection_obs floats):
            [0:4]   one-hot phase
            [4:8]   incoming link densities (normalised by k_jam), padded to 4
            [8]     EV distance to this intersection (normalised)
            [9]     normalised elapsed time
            [10]    normalised return-to-go
        """
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        time_norm = self._step_count / max(self.max_steps, 1)
        rtg_norm = np.clip(self._return_to_go / self.return_to_go_scale, -1.0, 1.0)

        for i, node_id in enumerate(self._route_intersections):
            if i >= self._max_route_len:
                break
            offset = i * self._per_intersection_obs
            node = self._network["nodes"][node_id]

            # One-hot phase
            phase = node["current_phase"]
            obs[offset + phase] = 1.0

            # Incoming link densities (normalised)
            for j, lid in enumerate(node["incoming_links"][:4]):
                lk = self._network["links"][lid]
                obs[offset + 4 + j] = np.clip(lk["density"] / lk["k_jam"], 0.0, 1.0)

            # EV distance to this intersection
            ev_dist = self._compute_ev_distance_to_intersection(i)
            obs[offset + 8] = np.clip(ev_dist, -1.0, 1.0)

            # Time
            obs[offset + 9] = time_norm

            # Return-to-go
            obs[offset + 10] = rtg_norm

        return obs

    def _compute_ev_distance_to_intersection(self, intersection_idx: int) -> float:
        """Normalised distance from EV to the *intersection_idx*-th node on route.

        Returns a value in [-1, 1]:
            - Negative means the EV has already passed this intersection.
            - 0 means the EV is at this intersection.
            - Positive (up to 1) means the EV is approaching.
        """
        # EV is between route[_ev_link_idx] and route[_ev_link_idx+1]
        ev_position_continuous = self._ev_link_idx + self._ev_progress
        diff = intersection_idx - ev_position_continuous
        # Normalise by total route length
        route_len = max(len(self._route) - 1, 1)
        return float(diff / route_len)

    # ------------------------------------------------------------------
    # Public properties (used by baselines and data collector)
    # ------------------------------------------------------------------

    @property
    def network(self) -> Network:
        """The underlying traffic network dict."""
        return self._network

    @property
    def ev_route(self) -> Route:
        """The current EV route (list of (node, link) tuples)."""
        return self._route

    # ------------------------------------------------------------------
    # Info
    # ------------------------------------------------------------------

    def _get_info(self) -> dict[str, Any]:
        return {
            "ev_link_idx": self._ev_link_idx,
            "ev_progress": self._ev_progress,
            "ev_arrived": self._ev_arrived,
            "step": self._step_count,
            "intersections_passed": self._intersections_passed,
            "return_to_go": self._return_to_go,
            "route_length": len(self._route),
            "total_queue": get_total_queue_length(self._network),
            "ev_travel_time": self._step_count if self._ev_arrived else -1,
        }
