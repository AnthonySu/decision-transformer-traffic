#!/usr/bin/env python3
"""Generate multi-agent trajectory dataset for MADT training.

Runs episodes in EVCorridorMAEnv using expert (greedy preemption), random,
and noisy-expert policies, then stores trajectories in HDF5 format compatible
with MultiAgentTrajectoryDataset.

HDF5 layout per episode:
    /episode_i/states       -> [T, n_agents, state_dim]
    /episode_i/actions      -> [T, n_agents]
    /episode_i/rewards      -> [T, n_agents]
    /episode_i/dones        -> [T]
    attrs: policy_name, episode_return, episode_length, n_agents

Usage::

    python scripts/generate_ma_dataset.py --rows 3 --cols 3 --episodes 200
    python scripts/generate_ma_dataset.py --rows 4 --cols 4 --episodes 500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

import h5py
import numpy as np

from src.envs.ev_corridor_ma_env import EVCorridorMAEnv

# ------------------------------------------------------------------
# Multi-agent policy wrappers
# ------------------------------------------------------------------

class MAGreedyPreemptPolicy:
    """Expert policy for MA env: grants green to EV approach direction.

    For the agent nearest to the EV, selects the phase that matches the
    link the EV is currently on.  Other agents use a simple max-density
    heuristic on their incoming links (observation-based).
    """

    def select_actions(
        self,
        obs_dict: dict[str, np.ndarray],
        env: EVCorridorMAEnv,
    ) -> dict[str, int]:
        actions: dict[str, int] = {}
        agents = list(obs_dict.keys())

        for agent_name in agents:
            idx = int(agent_name.split("_")[1])
            node_id = env._route_intersections[idx]

            # If EV is approaching this intersection, grant green to EV link
            if not env._ev_arrived and env._ev_link_idx < len(env._route) - 1:
                _, lid = env._route[env._ev_link_idx]
                if lid is not None:
                    downstream = env._network["links"][lid]["target"]
                    if downstream == node_id:
                        # Grant green to the phase serving the EV link
                        phase_idx = env._network["links"][lid]["phase_index"]
                        actions[agent_name] = phase_idx
                        continue

            # For non-EV-adjacent intersections: pick phase with max incoming
            # density (simple max-pressure heuristic from observation)
            obs = obs_dict[agent_name]
            # obs[4:8] are incoming densities
            densities = obs[4:8]
            # Map density index to phase: idx 0,1 -> phase 0 (N/S), 2,3 -> phase 1 (E/W)
            # Simplified: pick the phase with the highest total density
            phase_pressures = np.zeros(4)
            node = env._network["nodes"][node_id]
            for j, lid in enumerate(node["incoming_links"][:4]):
                lk = env._network["links"][lid]
                phase_pressures[lk["phase_index"]] += lk["density"] * lk["length"]
            actions[agent_name] = int(np.argmax(phase_pressures))

        return actions


class MACentralizedExpertPolicy:
    """Centralized expert that selects optimal actions for ALL agents jointly.

    Addresses the credit assignment problem by ensuring consistent, globally
    coordinated actions across all agents in the trajectory data:

    - On-route intersections ahead of the EV: grant green to the EV's
      travel direction so the corridor is pre-cleared.
    - On-route intersections behind the EV: use MaxPressure to manage
      residual traffic efficiently.
    - The EV's immediate intersection: always grant green to the EV link.

    This produces trajectories where all agents contribute coherently to EV
    progress, giving the MADT clearer credit assignment signal.
    """

    def select_actions(
        self,
        obs_dict: dict[str, np.ndarray],
        env: EVCorridorMAEnv,
    ) -> dict[str, int]:
        actions: dict[str, int] = {}
        agents = list(obs_dict.keys())

        for agent_name in agents:
            idx = int(agent_name.split("_")[1])
            node_id = env._route_intersections[idx]

            # --- EV's immediate intersection: grant green to EV link ---
            if not env._ev_arrived and env._ev_link_idx < len(env._route) - 1:
                _, lid = env._route[env._ev_link_idx]
                if lid is not None:
                    downstream = env._network["links"][lid]["target"]
                    if downstream == node_id:
                        phase_idx = env._network["links"][lid]["phase_index"]
                        actions[agent_name] = phase_idx
                        continue

            # --- On-route intersections AHEAD of EV: pre-clear corridor ---
            if idx > env._ev_link_idx and not env._ev_arrived:
                # Find the route link entering this intersection and grant
                # green to its direction so the EV can pass without stopping.
                if idx < len(env._route):
                    # The link from idx-1 to idx on the route
                    _, route_lid = env._route[idx - 1] if idx - 1 < len(env._route) else (None, None)
                    if route_lid is not None:
                        phase_idx = env._network["links"][route_lid]["phase_index"]
                        actions[agent_name] = phase_idx
                        continue

            # --- Behind EV or fallback: MaxPressure heuristic ---
            node = env._network["nodes"][node_id]
            phase_pressures = np.zeros(4)
            for lid in node["incoming_links"][:4]:
                lk = env._network["links"][lid]
                phase_pressures[lk["phase_index"]] += lk["density"] * lk["length"]
            actions[agent_name] = int(np.argmax(phase_pressures))

        return actions


class MARandomPolicy:
    """Uniform random policy for MA env."""

    def select_actions(
        self,
        obs_dict: dict[str, np.ndarray],
        env: EVCorridorMAEnv,
    ) -> dict[str, int]:
        return {a: env.action_space(a).sample() for a in obs_dict}


class MANoisyExpertPolicy:
    """Expert policy with random action noise for data diversity."""

    def __init__(self, expert: MAGreedyPreemptPolicy, noise_prob: float = 0.3):
        self.expert = expert
        self.noise_prob = noise_prob

    def select_actions(
        self,
        obs_dict: dict[str, np.ndarray],
        env: EVCorridorMAEnv,
    ) -> dict[str, int]:
        expert_actions = self.expert.select_actions(obs_dict, env)
        actions = {}
        for agent_name in obs_dict:
            if np.random.random() < self.noise_prob:
                actions[agent_name] = env.action_space(agent_name).sample()
            else:
                actions[agent_name] = expert_actions[agent_name]
        return actions


# ------------------------------------------------------------------
# Episode collection
# ------------------------------------------------------------------

def collect_ma_episode(
    env: EVCorridorMAEnv,
    policy: MAGreedyPreemptPolicy | MARandomPolicy | MANoisyExpertPolicy,
    n_agents: int,
) -> dict[str, np.ndarray] | None:
    """Run one episode and return trajectory arrays.

    Returns None if the episode has inconsistent agent count.
    """
    obs_dict, _info = env.reset()

    if len(env.agents) != n_agents:
        return None

    agents_snapshot = list(env.agents)
    ep_states: list[np.ndarray] = []
    ep_actions: list[np.ndarray] = []
    ep_rewards: list[np.ndarray] = []
    ep_dones: list[bool] = []

    for _ in range(env.max_steps):
        if not env.agents:
            break

        # Build state array [n_agents, state_dim]
        s_arr = np.stack([obs_dict[a] for a in agents_snapshot])
        ep_states.append(s_arr)

        # Get actions
        action_dict = policy.select_actions(obs_dict, env)
        a_arr = np.array([action_dict.get(a, 0) for a in agents_snapshot], dtype=np.int64)
        ep_actions.append(a_arr)

        obs_dict, rew_dict, term_dict, trunc_dict, _info = env.step(action_dict)
        r_arr = np.array([rew_dict.get(a, 0.0) for a in agents_snapshot], dtype=np.float32)
        ep_rewards.append(r_arr)

        any_done = any(term_dict.values()) or any(trunc_dict.values())
        ep_dones.append(any_done)

        if not env.agents:
            break

    if len(ep_states) < 2:
        return None

    return {
        "states": np.stack(ep_states),     # [T, n_agents, state_dim]
        "actions": np.stack(ep_actions),   # [T, n_agents]
        "rewards": np.stack(ep_rewards),   # [T, n_agents]
        "dones": np.array(ep_dones, dtype=bool),  # [T]
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def generate_dataset(
    rows: int = 3,
    cols: int = 3,
    max_steps: int = 100,
    num_expert: int = 100,
    num_random: int = 30,
    num_noisy: int = 70,
    save_path: str = "data/ma_dataset.h5",
    seed: int = 42,
    use_centralized_expert: bool = False,
    shared_reward_frac: float = 0.3,
) -> str:
    """Generate and save a multi-agent dataset.

    Parameters
    ----------
    use_centralized_expert : bool
        If True, use the centralized expert policy that pre-clears the
        corridor ahead of the EV (better credit assignment).
    shared_reward_frac : float
        Fraction of global EV-progress reward mixed into each agent's reward.
        Higher values improve credit assignment in offline data.

    Returns the save path.
    """
    origin = "n0_0"
    destination = f"n{rows - 1}_{cols - 1}"

    env = EVCorridorMAEnv(
        rows=rows, cols=cols, max_steps=max_steps,
        origin=origin, destination=destination, seed=seed,
        shared_reward_frac=shared_reward_frac,
    )
    obs_dict, _ = env.reset()
    n_agents = len(env.agents)
    state_dim = next(iter(obs_dict.values())).shape[0]
    expert_name = "centralized" if use_centralized_expert else "greedy"
    print(f"MA env: {rows}x{cols} grid, {n_agents} agents, state_dim={state_dim}, "
          f"expert={expert_name}, shared_reward_frac={shared_reward_frac}")

    if use_centralized_expert:
        expert = MACentralizedExpertPolicy()
    else:
        expert = MAGreedyPreemptPolicy()
    random_pol = MARandomPolicy()
    noisy_pol = MANoisyExpertPolicy(expert, noise_prob=0.3)

    policies = [
        (expert, num_expert, "expert"),
        (random_pol, num_random, "random"),
        (noisy_pol, num_noisy, "noisy_expert"),
    ]

    episodes: list[dict] = []
    rng = np.random.default_rng(seed)

    for policy, count, name in policies:
        collected = 0
        attempts = 0
        while collected < count and attempts < count * 3:
            attempts += 1
            ep_seed = int(rng.integers(0, 2**31))
            env._rng = np.random.default_rng(ep_seed)
            ep = collect_ma_episode(env, policy, n_agents)
            if ep is not None:
                ep["policy_name"] = name
                episodes.append(ep)
                collected += 1
        print(f"  Collected {collected}/{count} {name} episodes ({attempts} attempts)")

    # Save to HDF5
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with h5py.File(save_path, "w") as f:
        meta = {
            "num_episodes": len(episodes),
            "n_agents": n_agents,
            "state_dim": state_dim,
            "rows": rows,
            "cols": cols,
            "origin": origin,
            "destination": destination,
            "policy_names": list({ep["policy_name"] for ep in episodes}),
        }
        f.attrs["metadata"] = json.dumps(meta)

        for idx, ep in enumerate(episodes):
            grp = f.create_group(f"episode_{idx}")
            grp.create_dataset("states", data=ep["states"], compression="gzip")
            grp.create_dataset("actions", data=ep["actions"], compression="gzip")
            grp.create_dataset("rewards", data=ep["rewards"], compression="gzip")
            grp.create_dataset("dones", data=ep["dones"], compression="gzip")
            grp.attrs["policy_name"] = ep["policy_name"]
            grp.attrs["episode_return"] = float(np.sum(ep["rewards"]))
            grp.attrs["episode_length"] = ep["states"].shape[0]
            grp.attrs["n_agents"] = n_agents

    file_kb = Path(save_path).stat().st_size / 1024
    print(f"Saved {len(episodes)} episodes to {save_path} ({file_kb:.1f} KB)")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate MA dataset for MADT")
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=200,
                        help="Total episodes (split: 50%% expert, 15%% random, 35%% noisy)")
    parser.add_argument("--save-path", type=str, default="data/ma_dataset_3x3.h5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--centralized-expert", action="store_true",
                        help="Use centralized expert (pre-clears corridor for better credit assignment)")
    parser.add_argument("--shared-reward-frac", type=float, default=0.3,
                        help="Fraction of global EV-progress reward in each agent's reward")
    args = parser.parse_args()

    n_expert = max(1, int(args.episodes * 0.50))
    n_random = max(1, int(args.episodes * 0.15))
    n_noisy = args.episodes - n_expert - n_random

    t0 = time.time()
    generate_dataset(
        rows=args.rows,
        cols=args.cols,
        max_steps=args.max_steps,
        num_expert=n_expert,
        num_random=n_random,
        num_noisy=n_noisy,
        save_path=args.save_path,
        seed=args.seed,
        use_centralized_expert=args.centralized_expert,
        shared_reward_frac=args.shared_reward_frac,
    )
    print(f"Done in {time.time() - t0:.1f}s")
