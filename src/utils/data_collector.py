"""Offline dataset generation for Decision Transformer training.

Collects trajectory data from one or more policies, computes returns-to-go,
and persists everything to HDF5 for efficient, random-access loading during
training.

Storage layout (HDF5)::

    /metadata          – JSON-encoded global metadata
    /episode_0/
        states         – [T, state_dim]   or [T, n_agents, state_dim]
        actions        – [T]              or [T, n_agents]
        rewards        – [T]
        dones          – [T]
        returns_to_go  – [T]
    /episode_1/
        ...

Each episode group also carries attributes:
    policy_name, episode_return, episode_length, ev_travel_time
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Protocol, runtime_checkable

import h5py
import numpy as np
from tqdm import tqdm

# ------------------------------------------------------------------
# Typing helpers
# ------------------------------------------------------------------

@runtime_checkable
class Policy(Protocol):
    """Minimal protocol a policy must satisfy for data collection."""

    def select_action(self, obs: np.ndarray, ev_info: Dict[str, Any]) -> int: ...
    def reset(self) -> None: ...


# ------------------------------------------------------------------
# DataCollector
# ------------------------------------------------------------------

class DataCollector:
    """Collects trajectory data from policies for offline DT training.

    Parameters
    ----------
    env : gym.Env
        Gymnasium-compatible environment.  Must expose ``reset()`` and
        ``step()`` with the standard 5-tuple return.
    save_path : str
        Path for the output HDF5 file.
    """

    def __init__(
        self,
        env: Any,
        save_path: str = "data/offline_dataset.h5",
    ) -> None:
        self.env = env
        self.save_path = save_path

        # Accumulated episode data (in-memory until save)
        self._episodes: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Collection API
    # ------------------------------------------------------------------

    def collect_episodes(
        self,
        policy: Any,
        num_episodes: int,
        policy_name: str = "expert",
    ) -> List[Dict[str, Any]]:
        """Roll out *policy* for *num_episodes* and store the trajectories.

        Parameters
        ----------
        policy : Policy-like
            Object with ``select_action(obs, ev_info) -> int`` and
            optionally ``reset()``.
        num_episodes : int
            Number of full episodes to collect.
        policy_name : str
            Label stored in the metadata for provenance tracking.

        Returns
        -------
        list[dict]
            The newly collected episode dicts (also appended internally).
        """
        new_episodes: List[Dict[str, Any]] = []

        for _ in tqdm(range(num_episodes), desc=f"Collecting [{policy_name}]"):
            episode = self._rollout(policy, policy_name)
            new_episodes.append(episode)

        self._episodes.extend(new_episodes)
        return new_episodes

    def collect_mixed_dataset(
        self,
        expert_policy: Any,
        num_expert: int,
        num_random: int,
        num_suboptimal: int = 0,
    ) -> None:
        """Collect a mixed-quality dataset (expert + random + suboptimal).

        The suboptimal policy is created by adding Gaussian noise to the
        expert's actions (clamped to valid action space).

        Parameters
        ----------
        expert_policy : Policy-like
            The near-optimal policy (e.g. :class:`GreedyPreemptPolicy`).
        num_expert : int
            Episodes from the expert policy.
        num_random : int
            Episodes from a uniform-random policy.
        num_suboptimal : int
            Episodes from a noisy version of the expert.
        """
        # Expert episodes
        if num_expert > 0:
            self.collect_episodes(expert_policy, num_expert, policy_name="expert")

        # Random episodes
        if num_random > 0:
            random_policy = _RandomPolicy(self.env)
            self.collect_episodes(random_policy, num_random, policy_name="random")

        # Suboptimal (noisy-expert) episodes
        if num_suboptimal > 0:
            noisy_policy = _NoisyPolicy(expert_policy, self.env, noise_prob=0.3)
            self.collect_episodes(
                noisy_policy, num_suboptimal, policy_name="suboptimal"
            )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_dataset(self) -> None:
        """Write all collected episodes to HDF5."""
        os.makedirs(os.path.dirname(self.save_path) or ".", exist_ok=True)

        with h5py.File(self.save_path, "w") as f:
            # Global metadata
            meta = {
                "num_episodes": len(self._episodes),
                "policy_names": list({ep["policy_name"] for ep in self._episodes}),
            }
            f.attrs["metadata"] = json.dumps(meta)

            for idx, ep in enumerate(tqdm(self._episodes, desc="Saving")):
                grp = f.create_group(f"episode_{idx}")

                grp.create_dataset("states", data=ep["states"], compression="gzip")
                grp.create_dataset("actions", data=ep["actions"], compression="gzip")
                grp.create_dataset("rewards", data=ep["rewards"], compression="gzip")
                grp.create_dataset("dones", data=ep["dones"], compression="gzip")
                grp.create_dataset(
                    "returns_to_go", data=ep["returns_to_go"], compression="gzip"
                )

                # Per-episode attributes
                grp.attrs["policy_name"] = ep["policy_name"]
                grp.attrs["episode_return"] = ep["episode_return"]
                grp.attrs["episode_length"] = ep["episode_length"]
                grp.attrs["ev_travel_time"] = ep.get("ev_travel_time", -1.0)

        print(f"Saved {len(self._episodes)} episodes to {self.save_path}")

    @staticmethod
    def load_dataset(path: str) -> Dict[str, Any]:
        """Load an HDF5 dataset into a dict of numpy arrays.

        Parameters
        ----------
        path : str
            Path to the HDF5 file written by :meth:`save_dataset`.

        Returns
        -------
        dict
            Keys: ``"states"``, ``"actions"``, ``"rewards"``, ``"dones"``,
            ``"returns_to_go"``, ``"episode_ends"``, ``"metadata"``.
            Arrays are concatenated across episodes; ``episode_ends`` gives
            the cumulative end indices so individual episodes can be
            recovered.
        """
        all_states: List[np.ndarray] = []
        all_actions: List[np.ndarray] = []
        all_rewards: List[np.ndarray] = []
        all_dones: List[np.ndarray] = []
        all_rtg: List[np.ndarray] = []
        episode_ends: List[int] = []
        episode_meta: List[Dict[str, Any]] = []

        with h5py.File(path, "r") as f:
            metadata = json.loads(f.attrs.get("metadata", "{}"))
            num_episodes = metadata.get("num_episodes", 0)

            cumulative = 0
            for idx in range(num_episodes):
                grp = f[f"episode_{idx}"]
                states = np.array(grp["states"])
                actions = np.array(grp["actions"])
                rewards = np.array(grp["rewards"])
                dones = np.array(grp["dones"])
                rtg = np.array(grp["returns_to_go"])

                all_states.append(states)
                all_actions.append(actions)
                all_rewards.append(rewards)
                all_dones.append(dones)
                all_rtg.append(rtg)

                cumulative += len(rewards)
                episode_ends.append(cumulative)

                episode_meta.append({
                    "policy_name": str(grp.attrs.get("policy_name", "")),
                    "episode_return": float(grp.attrs.get("episode_return", 0.0)),
                    "episode_length": int(grp.attrs.get("episode_length", 0)),
                    "ev_travel_time": float(grp.attrs.get("ev_travel_time", -1.0)),
                })

        return {
            "states": np.concatenate(all_states, axis=0) if all_states else np.array([]),
            "actions": np.concatenate(all_actions, axis=0) if all_actions else np.array([]),
            "rewards": np.concatenate(all_rewards, axis=0) if all_rewards else np.array([]),
            "dones": np.concatenate(all_dones, axis=0) if all_dones else np.array([]),
            "returns_to_go": np.concatenate(all_rtg, axis=0) if all_rtg else np.array([]),
            "episode_ends": np.array(episode_ends, dtype=np.int64),
            "metadata": metadata,
            "episode_metadata": episode_meta,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rollout(self, policy: Any, policy_name: str) -> Dict[str, Any]:
        """Execute one full episode and return the trajectory dict."""
        if hasattr(policy, "reset"):
            policy.reset()

        obs, info = self.env.reset()
        ev_info = info.get("ev_info", {})

        states, actions, rewards, dones = [], [], [], []
        done = False
        ev_travel_time: float = -1.0

        while not done:
            states.append(np.array(obs, dtype=np.float32))
            action = policy.select_action(obs, ev_info)
            obs, reward, terminated, truncated, info = self.env.step(action)
            ev_info = info.get("ev_info", {})

            # Normalize action to consistent shape matching action_space
            action_arr = np.atleast_1d(np.asarray(action, dtype=np.int64)).flatten()
            # If scalar action in MultiDiscrete env, broadcast to full action size
            if hasattr(self.env.action_space, 'nvec') and action_arr.shape[0] == 1:
                action_arr = np.full(len(self.env.action_space.nvec), action_arr[0], dtype=np.int64)
            actions.append(action_arr)
            rewards.append(float(reward))
            dones.append(terminated or truncated)
            done = terminated or truncated

        # Extract EV travel time from final info if available
        if "ev_travel_time" in info:
            ev_travel_time = float(info["ev_travel_time"])

        states_arr = np.array(states, dtype=np.float32)
        actions_arr = np.array(actions, dtype=np.int64)
        rewards_arr = np.array(rewards, dtype=np.float32)
        dones_arr = np.array(dones, dtype=bool)
        rtg_arr = _compute_returns_to_go(rewards_arr)

        return {
            "states": states_arr,
            "actions": actions_arr,
            "rewards": rewards_arr,
            "dones": dones_arr,
            "returns_to_go": rtg_arr,
            "policy_name": policy_name,
            "episode_return": float(np.sum(rewards_arr)),
            "episode_length": len(rewards_arr),
            "ev_travel_time": ev_travel_time,
        }


# ------------------------------------------------------------------
# Helper policies
# ------------------------------------------------------------------

class _RandomPolicy:
    """Uniform-random action selection."""

    def __init__(self, env: Any) -> None:
        self.env = env

    def select_action(self, obs: Any, ev_info: Dict[str, Any]):
        action = self.env.action_space.sample()
        return action

    def reset(self) -> None:
        pass


class _NoisyPolicy:
    """Wraps an expert policy and randomly replaces actions with noise.

    With probability ``noise_prob`` the action is replaced by a uniformly
    random one; otherwise the expert action is used.
    """

    def __init__(
        self,
        expert: Any,
        env: Any,
        noise_prob: float = 0.3,
    ) -> None:
        self.expert = expert
        self.env = env
        self.noise_prob = noise_prob

    def select_action(self, obs: Any, ev_info: Dict[str, Any]):
        if np.random.random() < self.noise_prob:
            return self.env.action_space.sample()
        return self.expert.select_action(obs, ev_info)

    def reset(self) -> None:
        if hasattr(self.expert, "reset"):
            self.expert.reset()


# ------------------------------------------------------------------
# Utility functions
# ------------------------------------------------------------------

def _compute_returns_to_go(rewards: np.ndarray) -> np.ndarray:
    """Compute discounted returns-to-go (gamma=1, i.e. undiscounted sum).

    ``rtg[t] = sum(rewards[t:])``
    """
    rtg = np.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running += rewards[t]
        rtg[t] = running
    return rtg
