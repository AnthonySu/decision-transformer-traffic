"""Multi-agent PettingZoo environment for EV corridor optimization.

Each intersection along the EV's pre-computed route is an independent agent
that selects its own signal phase.  Agents observe local traffic conditions
plus EV proximity information and receive a mix of local and global rewards.

Two backends are supported via the ``use_lightsim`` flag:
    * **True** — delegates traffic simulation to a ``lightsim`` env
      (must be installed separately).
    * **False** (default) — uses a built-in grid CTM from
      :mod:`src.envs.network_utils`.
"""

from __future__ import annotations

import functools
from typing import Any

import numpy as np
from gymnasium import spaces
from pettingzoo import ParallelEnv

from src.envs.network_utils import (
    Network,
    Route,
    build_grid_network,
    compute_shortest_path,
    ctm_step,
    get_route_intersections,
    random_od_pair,
    reset_densities,
    signal_is_green_for_link,
)


class EVCorridorMAEnv(ParallelEnv):
    """PettingZoo ParallelEnv — one agent per intersection on the EV route.

    Parameters
    ----------
    rows, cols : int
        Grid dimensions for the mock network.
    use_lightsim : bool
        If True, use lightsim as the traffic backend (not yet implemented).
    max_steps : int
        Episode timeout.
    dt : float
        Simulation timestep in seconds.
    ev_speed_factor : float
        EV free-flow speed multiplier.
    origin, destination : str | None
        Fixed OD pair; random if None.
    return_to_go_scale : float
        Normalisation for the return-to-go feature.
    shared_reward_frac : float
        Fraction of the global EV-progress reward mixed into each agent's
        reward (the rest is purely local).
    """

    metadata = {"render_modes": ["human", "ansi"], "name": "ev_corridor_ma_v0"}

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
        shared_reward_frac: float = 0.3,
        render_mode: str | None = None,
        seed: int | None = None,
        # Aliases accepted by training scripts
        network_type: str | None = None,
        grid_rows: int | None = None,
        grid_cols: int | None = None,
        max_episode_steps: int | None = None,
        **kwargs,
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
        self.shared_reward_frac = shared_reward_frac
        self.render_mode = render_mode

        self._rng = np.random.default_rng(seed)

        # Network (topology fixed, state reset per episode)
        if self.use_lightsim:
            from src.envs.lightsim_adapter import LightSimAdapter

            scenario = kwargs.get("lightsim_scenario", "grid-4x4-v0")
            ls_kwargs = kwargs.get("lightsim_kwargs", {})
            self._ls_adapter = LightSimAdapter(scenario=scenario, **ls_kwargs)
            self._network: Network = self._ls_adapter.network
        else:
            self._ls_adapter = None
            self._network: Network = build_grid_network(rows=rows, cols=cols)

        # These are set in reset()
        self._route: Route = []
        self._route_intersections: list[str] = []
        self._num_route_intersections: int = 0

        # EV state
        self._ev_link_idx: int = 0
        self._ev_progress: float = 0.0
        self._ev_arrived: bool = False
        self._step_count: int = 0
        self._intersections_passed: int = 0
        self._ev_wait_steps: dict[str, int] = {}  # per-intersection wait counter

        # Per-agent return-to-go (for MADT conditioning)
        self._return_to_go: dict[str, float] = {}

        # PettingZoo requires these to be set before reset for API compliance.
        # We initialise with a dummy set; reset() will update.
        self.possible_agents: list[str] = []
        self.agents: list[str] = []

        # --- Observation / action space dimensions ---
        # Per-agent obs:
        #   own phase one-hot (4)
        #   incoming densities (4, padded)
        #   incoming queue proxies (4, padded)
        #   ev_distance (1)
        #   ev_speed_norm (1)
        #   ev_route_progress (1)
        #   neighbor phases (4 neighbors * 4 one-hot = 16, padded)
        #   return-to-go (1)
        #   time_norm (1)
        # Total: 4+4+4+1+1+1+16+1+1 = 33
        self._obs_size = 33

    # ------------------------------------------------------------------
    # PettingZoo API — spaces
    # ------------------------------------------------------------------

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        return spaces.Box(low=-1.0, high=1.0, shape=(self._obs_size,), dtype=np.float32)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        return spaces.Discrete(4)

    # ------------------------------------------------------------------
    # PettingZoo API — reset / step
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Reset traffic state
        if self.use_lightsim:
            self._ls_adapter.reset(rng=self._rng)
            self._network = self._ls_adapter.network
        else:
            reset_densities(self._network, base_density=self._rng.uniform(0.01, 0.04))

        # Reset signal phases randomly (for CTM; LightSim handles its own)
        if not self.use_lightsim:
            for node in self._network["nodes"].values():
                node["current_phase"] = int(self._rng.integers(0, node["num_phases"]))

        # OD pair
        if self._fixed_origin and self._fixed_destination:
            origin, destination = self._fixed_origin, self._fixed_destination
        else:
            origin, destination = random_od_pair(self._network, rng=self._rng)

        self._route = compute_shortest_path(self._network, origin, destination)
        self._route_intersections = get_route_intersections(self._network, self._route)
        self._num_route_intersections = len(self._route_intersections)

        # Agents = intersections on route
        self.possible_agents = [f"intersection_{i}" for i in range(self._num_route_intersections)]
        self.agents = list(self.possible_agents)

        # Clear the LRU caches so spaces match new agent names
        self.observation_space.cache_clear()
        self.action_space.cache_clear()

        # EV state
        self._ev_link_idx = 0
        self._ev_progress = 0.0
        self._ev_arrived = False
        self._step_count = 0
        self._intersections_passed = 0
        self._ev_wait_steps = {a: 0 for a in self.agents}
        self._return_to_go = {a: self.return_to_go_scale for a in self.agents}
        self._prev_ev_progress_global = 0.0

        observations = {a: self._get_agent_obs(a) for a in self.agents}
        infos = {a: self._get_agent_info(a) for a in self.agents}
        return observations, infos

    def step(
        self, actions: dict[str, int]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict],
    ]:
        self._step_count += 1

        # --- Apply actions ---
        for agent_name, act in actions.items():
            idx = self._agent_to_index(agent_name)
            node_id = self._route_intersections[idx]
            node = self._network["nodes"][node_id]
            node["current_phase"] = int(act) % node["num_phases"]

        # --- Traffic simulation step ---
        if self.use_lightsim:
            phase_actions: dict[str, int] = {}
            for agent_name, act in actions.items():
                idx = self._agent_to_index(agent_name)
                node_id = self._route_intersections[idx]
                n_phases = self._network["nodes"][node_id]["num_phases"]
                phase_actions[node_id] = int(act) % n_phases
            self._ls_adapter.step(phase_actions)
            self._network = self._ls_adapter.network
        else:
            ctm_step(self._network, dt=self.dt)

        # --- Advance EV ---
        ev_passed_node: str | None = None  # node_id the EV just crossed
        ev_was_blocked = False

        if not self._ev_arrived and self._ev_link_idx < len(self._route) - 1:
            _, link_id = self._route[self._ev_link_idx]
            if link_id is not None:
                link = self._network["links"][link_id]
                density = link["density"]
                k_jam = link["k_jam"]
                v_free = link["v_free"]
                length = link["length"]

                congestion_factor = max(1.0 - density / k_jam, 0.05)
                ev_speed = v_free * self.ev_speed_factor * congestion_factor

                # Check signal at downstream intersection
                downstream_node = link["target"]
                if self._ev_progress > 0.9:
                    if not signal_is_green_for_link(self._network, downstream_node, link_id):
                        ev_speed = 0.0
                        ev_was_blocked = True

                self._ev_progress += ev_speed * self.dt / length

                if self._ev_progress >= 1.0:
                    self._ev_progress = 0.0
                    ev_passed_node = self._route_intersections[
                        min(self._ev_link_idx + 1, self._num_route_intersections - 1)
                    ]
                    self._ev_link_idx += 1
                    self._intersections_passed += 1
                    if self._ev_link_idx >= len(self._route) - 1:
                        self._ev_arrived = True

        # Track per-intersection EV wait
        if ev_was_blocked and self._ev_link_idx < len(self._route) - 1:
            _, lid = self._route[self._ev_link_idx]
            if lid is not None:
                downstream = self._network["links"][lid]["target"]
                # Find agent for downstream intersection
                if downstream in self._route_intersections:
                    agent_idx = self._route_intersections.index(downstream)
                    agent_name = f"intersection_{agent_idx}"
                    if agent_name in self._ev_wait_steps:
                        self._ev_wait_steps[agent_name] += 1

        # --- Compute global EV progress signal ---
        route_len = max(len(self._route) - 1, 1)
        ev_progress_global = (self._ev_link_idx + self._ev_progress) / route_len

        # Compute EV progress delta (how much the EV advanced this step)
        prev_progress = getattr(self, "_prev_ev_progress_global", 0.0)
        ev_progress_delta = ev_progress_global - prev_progress
        self._prev_ev_progress_global = ev_progress_global

        # --- Rewards ---
        rewards: dict[str, float] = {}
        terminated: dict[str, bool] = {}
        truncated: dict[str, bool] = {}

        is_terminated = self._ev_arrived
        is_truncated = self._step_count >= self.max_steps and not self._ev_arrived

        for agent_name in self.agents:
            idx = self._agent_to_index(agent_name)
            node_id = self._route_intersections[idx]
            node = self._network["nodes"][node_id]

            # Local traffic penalty: sum of incoming densities as queue proxy
            local_queue = 0.0
            for lid in node["incoming_links"]:
                lk = self._network["links"][lid]
                local_queue += lk["density"] * lk["length"] * lk["num_lanes"]
            local_reward = -local_queue

            # EV component
            ev_reward = 0.0
            if ev_passed_node == node_id:
                ev_reward += 10.0  # bonus for letting EV through
            # Penalty proportional to how long EV waited at this intersection
            ev_reward -= 0.5 * self._ev_wait_steps.get(agent_name, 0)

            # Shared global component: combines absolute progress and step-wise delta.
            # The delta term gives ALL agents a positive signal whenever the EV advances,
            # addressing credit assignment by rewarding coordinated corridor clearing.
            shared_reward = ev_progress_global * 2.0 + ev_progress_delta * 10.0

            # Terminal bonuses (shared equally)
            terminal = 0.0
            if is_terminated:
                terminal = 50.0 / max(self._num_route_intersections, 1)
            elif is_truncated:
                terminal = -50.0 / max(self._num_route_intersections, 1)

            total = (
                (1.0 - self.shared_reward_frac) * (local_reward + ev_reward)
                + self.shared_reward_frac * shared_reward
                + terminal
            )
            rewards[agent_name] = float(total)
            terminated[agent_name] = is_terminated
            truncated[agent_name] = is_truncated

            # Update return-to-go
            self._return_to_go[agent_name] -= total

        # Build observations and infos
        observations = {a: self._get_agent_obs(a) for a in self.agents}
        infos = {a: self._get_agent_info(a) for a in self.agents}

        # If episode is over, clear agents list (PettingZoo convention)
        if is_terminated or is_truncated:
            self.agents = []

        return observations, rewards, terminated, truncated, infos

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def _get_agent_obs(self, agent: str) -> np.ndarray:
        """Build observation for a single agent (intersection).

        Layout (33 floats):
            [0:4]    own phase one-hot
            [4:8]    incoming link densities (normalised), padded to 4
            [8:12]   incoming queue proxies (normalised), padded to 4
            [12]     EV distance to this intersection (normalised)
            [13]     EV speed (normalised)
            [14]     EV progress along full route (0-1)
            [15:31]  neighbor phases: up to 4 neighbors, 4-d one-hot each
            [31]     return-to-go (normalised)
            [32]     time (normalised)
        """
        obs = np.zeros(self._obs_size, dtype=np.float32)
        idx = self._agent_to_index(agent)
        node_id = self._route_intersections[idx]
        node = self._network["nodes"][node_id]
        nodes = self._network["nodes"]
        links = self._network["links"]

        # Own phase (one-hot)
        obs[node["current_phase"]] = 1.0

        # Incoming densities & queue proxies
        for j, lid in enumerate(node["incoming_links"][:4]):
            lk = links[lid]
            obs[4 + j] = np.clip(lk["density"] / lk["k_jam"], 0.0, 1.0)
            queue_proxy = lk["density"] * lk["length"] * lk["num_lanes"]
            obs[8 + j] = np.clip(queue_proxy / 20.0, 0.0, 1.0)  # normalise

        # EV distance
        ev_pos = self._ev_link_idx + self._ev_progress
        route_len = max(len(self._route) - 1, 1)
        obs[12] = np.clip((idx - ev_pos) / route_len, -1.0, 1.0)

        # EV speed (normalised by free-flow * speed_factor)
        ev_speed_norm = 0.0
        if not self._ev_arrived and self._ev_link_idx < len(self._route) - 1:
            _, lid = self._route[self._ev_link_idx]
            if lid is not None:
                lk = links[lid]
                cong = max(1.0 - lk["density"] / lk["k_jam"], 0.05)
                ev_speed_norm = cong  # 1 = free-flow, 0 = jammed
        obs[13] = ev_speed_norm

        # EV route progress
        obs[14] = ev_pos / route_len

        # Neighbor phases (up to 4 neighbors on the route or in the grid)
        graph = self._network["graph"]
        neighbor_ids = list(graph.predecessors(node_id)) + list(graph.successors(node_id))
        # Deduplicate while preserving order
        seen = set()
        unique_neighbors = []
        for n in neighbor_ids:
            if n not in seen:
                seen.add(n)
                unique_neighbors.append(n)

        for ni, nid in enumerate(unique_neighbors[:4]):
            nnode = nodes[nid]
            base = 15 + ni * 4
            obs[base + nnode["current_phase"]] = 1.0

        # Return-to-go
        rtg = self._return_to_go.get(agent, 0.0)
        obs[31] = np.clip(rtg / self.return_to_go_scale, -1.0, 1.0)

        # Time
        obs[32] = self._step_count / max(self.max_steps, 1)

        return obs

    # ------------------------------------------------------------------
    # Info & helpers
    # ------------------------------------------------------------------

    def _get_agent_info(self, agent: str) -> dict[str, Any]:
        idx = self._agent_to_index(agent)
        return {
            "node_id": self._route_intersections[idx],
            "ev_link_idx": self._ev_link_idx,
            "ev_progress": self._ev_progress,
            "ev_arrived": self._ev_arrived,
            "step": self._step_count,
            "ev_wait_at_this": self._ev_wait_steps.get(agent, 0),
        }

    def _agent_to_index(self, agent: str) -> int:
        """Convert agent name like 'intersection_3' to integer index."""
        return int(agent.split("_")[1])

    def render(self) -> str | None:
        """Text rendering of the multi-agent state."""
        lines: list[str] = []
        lines.append(f"Step {self._step_count}/{self.max_steps}  "
                      f"EV link {self._ev_link_idx}/{len(self._route)-1}  "
                      f"progress {self._ev_progress:.2f}")

        for i, node_id in enumerate(self._route_intersections):
            node = self._network["nodes"][node_id]
            agent = f"intersection_{i}"
            marker = " <-- EV" if i == self._ev_link_idx else ""
            lines.append(
                f"  {agent} ({node_id}): phase={node['current_phase']}  "
                f"wait={self._ev_wait_steps.get(agent, 0)}{marker}"
            )

        text = "\n".join(lines)
        if self.render_mode == "human":
            print(text)
        return text
