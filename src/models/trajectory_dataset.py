"""Trajectory datasets for Decision Transformer training.

Provides PyTorch Dataset classes that load offline trajectory data from HDF5
files, compute returns-to-go, normalize states, and sample fixed-length
context windows for training Decision Transformers.

Expected HDF5 structure for single-agent:
    /episode_0/states   -> [T, state_dim]
    /episode_0/actions  -> [T]
    /episode_0/rewards  -> [T]
    /episode_0/dones    -> [T]
    /episode_1/...

Expected HDF5 structure for multi-agent:
    /episode_0/states   -> [T, n_agents, state_dim]
    /episode_0/actions  -> [T, n_agents]
    /episode_0/rewards  -> [T, n_agents]
    /episode_0/dones    -> [T]
    /episode_1/...
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """Dataset of trajectory segments for Decision Transformer training.

    Loads episodes from an HDF5 file, computes returns-to-go, and serves
    fixed-length context windows with proper padding for shorter sequences.

    Args:
        data_path: Path to the HDF5 file containing episode data.
        context_length: Number of timesteps per training sample.
        discount: Discount factor for computing returns-to-go.
        normalize_states: Whether to z-score normalize states.
        normalize_returns: Whether to scale returns-to-go to roughly [-1, 1].
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        context_length: int = 30,
        discount: float = 1.0,
        normalize_states: bool = True,
        normalize_returns: bool = True,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.discount = discount
        self.normalize_states_flag = normalize_states
        self.normalize_returns_flag = normalize_returns

        self.episodes: List[Dict[str, np.ndarray]] = []
        self._load_data(data_path)

        # Compute normalization statistics
        self.state_mean: np.ndarray
        self.state_std: np.ndarray
        self.return_scale: float
        self._compute_statistics()

    def _load_data(self, data_path: Union[str, Path]) -> None:
        """Load all episodes from the HDF5 file and compute returns-to-go."""
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        with h5py.File(data_path, "r") as f:
            episode_keys = sorted(
                [k for k in f.keys() if k.startswith("episode")],
                key=lambda k: int(k.split("_")[-1]),
            )

            for key in episode_keys:
                grp = f[key]
                states = np.array(grp["states"], dtype=np.float32)     # [T, state_dim]
                actions = np.array(grp["actions"], dtype=np.int64)     # [T]
                rewards = np.array(grp["rewards"], dtype=np.float32)   # [T]
                dones = np.array(grp["dones"], dtype=np.bool_)         # [T]

                returns_to_go = self._compute_returns_to_go(rewards)

                self.episodes.append({
                    "states": states,
                    "actions": actions,
                    "rewards": rewards,
                    "returns_to_go": returns_to_go,
                    "dones": dones,
                    "length": len(states),
                })

        if len(self.episodes) == 0:
            raise ValueError(f"No episodes found in {data_path}")

    def _compute_returns_to_go(self, rewards: np.ndarray) -> np.ndarray:
        """Compute discounted returns-to-go: R_t = sum_{t'=t}^{T} gamma^{t'-t} * r_{t'}.

        Args:
            rewards: Array of shape [T].

        Returns:
            returns_to_go: Array of shape [T].
        """
        T = len(rewards)
        returns_to_go = np.zeros(T, dtype=np.float32)
        running_return = 0.0
        for t in reversed(range(T)):
            running_return = rewards[t] + self.discount * running_return
            returns_to_go[t] = running_return
        return returns_to_go

    def _compute_statistics(self) -> None:
        """Compute state normalization statistics and return scale from the full dataset."""
        all_states = np.concatenate([ep["states"] for ep in self.episodes], axis=0)
        self.state_mean = all_states.mean(axis=0)
        self.state_std = all_states.std(axis=0) + 1e-8  # avoid division by zero

        all_returns = np.concatenate([ep["returns_to_go"] for ep in self.episodes], axis=0)
        self.return_scale = max(np.abs(all_returns).max(), 1e-8)

    def __len__(self) -> int:
        """Total number of valid starting positions across all episodes."""
        return sum(ep["length"] for ep in self.episodes)

    def _get_episode_and_offset(self, idx: int) -> Tuple[int, int]:
        """Map a flat index to (episode_index, timestep_offset).

        Args:
            idx: Flat index into the dataset.

        Returns:
            Tuple of (episode_index, timestep_within_episode).
        """
        cumulative = 0
        for ep_idx, ep in enumerate(self.episodes):
            if idx < cumulative + ep["length"]:
                return ep_idx, idx - cumulative
            cumulative += ep["length"]
        # Should not reach here if idx < len(self)
        raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a context-length window starting from a position in the dataset.

        The window is centered around the mapped timestep. If the window
        extends beyond episode boundaries, the sequence is left-padded.

        Args:
            idx: Flat index into the dataset.

        Returns:
            Dictionary with keys:
                - states: [context_length, state_dim]
                - actions: [context_length]
                - returns_to_go: [context_length, 1]
                - timesteps: [context_length]
                - attention_mask: [context_length] (1 for real tokens, 0 for padding)
        """
        ep_idx, t = self._get_episode_and_offset(idx)
        ep = self.episodes[ep_idx]
        K = self.context_length

        # Determine the window: [start, start+K)
        # Try to end at t+1 so the queried timestep is the last in the window
        end = min(t + 1, ep["length"])
        start = max(end - K, 0)
        actual_len = end - start

        # Extract raw data
        states = ep["states"][start:end].copy()
        actions = ep["actions"][start:end].copy()
        rtg = ep["returns_to_go"][start:end].copy()

        # Normalize
        if self.normalize_states_flag:
            states = (states - self.state_mean) / self.state_std
        if self.normalize_returns_flag:
            rtg = rtg / self.return_scale

        # Timestep indices (absolute position within the episode)
        timesteps = np.arange(start, end, dtype=np.int64)

        # Pad to context_length (left-padding)
        pad_len = K - actual_len
        state_dim = states.shape[-1]

        padded_states = np.zeros((K, state_dim), dtype=np.float32)
        padded_actions = np.zeros(K, dtype=np.int64)
        padded_rtg = np.zeros(K, dtype=np.float32)
        padded_timesteps = np.zeros(K, dtype=np.int64)
        attention_mask = np.zeros(K, dtype=np.float32)

        padded_states[pad_len:] = states
        padded_actions[pad_len:] = actions
        padded_rtg[pad_len:] = rtg
        padded_timesteps[pad_len:] = timesteps
        attention_mask[pad_len:] = 1.0

        return {
            "states": torch.from_numpy(padded_states),
            "actions": torch.from_numpy(padded_actions),
            "returns_to_go": torch.from_numpy(padded_rtg).unsqueeze(-1),  # [K, 1]
            "timesteps": torch.from_numpy(padded_timesteps),
            "attention_mask": torch.from_numpy(attention_mask),
        }

    def get_state_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return state normalization statistics.

        Returns:
            Tuple of (state_mean, state_std), each of shape [state_dim].
        """
        return self.state_mean.copy(), self.state_std.copy()

    def get_return_scale(self) -> float:
        """Return the return normalization scale factor."""
        return self.return_scale


class MultiAgentTrajectoryDataset(Dataset):
    """Dataset of multi-agent trajectory segments for MADT training.

    Similar to TrajectoryDataset but handles per-agent trajectories where
    states, actions, and rewards have an additional agent dimension.

    Args:
        data_path: Path to the HDF5 file.
        n_agents: Number of agents in the environment.
        context_length: Number of timesteps per training sample.
        discount: Discount factor for returns-to-go.
        normalize_states: Whether to z-score normalize states.
        normalize_returns: Whether to scale returns-to-go.
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        n_agents: int,
        context_length: int = 20,
        discount: float = 1.0,
        normalize_states: bool = True,
        normalize_returns: bool = True,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.context_length = context_length
        self.discount = discount
        self.normalize_states_flag = normalize_states
        self.normalize_returns_flag = normalize_returns

        self.episodes: List[Dict[str, np.ndarray]] = []
        self._load_data(data_path)

        self.state_mean: np.ndarray
        self.state_std: np.ndarray
        self.return_scale: float
        self._compute_statistics()

    def _load_data(self, data_path: Union[str, Path]) -> None:
        """Load multi-agent episodes from HDF5."""
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        with h5py.File(data_path, "r") as f:
            episode_keys = sorted(
                [k for k in f.keys() if k.startswith("episode")],
                key=lambda k: int(k.split("_")[-1]),
            )

            for key in episode_keys:
                grp = f[key]
                # [T, n_agents, state_dim]
                states = np.array(grp["states"], dtype=np.float32)
                # [T, n_agents]
                actions = np.array(grp["actions"], dtype=np.int64)
                # [T, n_agents]
                rewards = np.array(grp["rewards"], dtype=np.float32)
                # [T]
                dones = np.array(grp["dones"], dtype=np.bool_)

                T = states.shape[0]

                # Compute per-agent returns-to-go: [T, n_agents]
                returns_to_go = np.zeros_like(rewards)
                for agent_i in range(self.n_agents):
                    returns_to_go[:, agent_i] = self._compute_returns_to_go(
                        rewards[:, agent_i]
                    )

                self.episodes.append({
                    "states": states,        # [T, n_agents, state_dim]
                    "actions": actions,      # [T, n_agents]
                    "rewards": rewards,      # [T, n_agents]
                    "returns_to_go": returns_to_go,  # [T, n_agents]
                    "dones": dones,          # [T]
                    "length": T,
                })

        if len(self.episodes) == 0:
            raise ValueError(f"No episodes found in {data_path}")

    def _compute_returns_to_go(self, rewards: np.ndarray) -> np.ndarray:
        """Compute discounted returns-to-go for a single agent."""
        T = len(rewards)
        returns_to_go = np.zeros(T, dtype=np.float32)
        running_return = 0.0
        for t in reversed(range(T)):
            running_return = rewards[t] + self.discount * running_return
            returns_to_go[t] = running_return
        return returns_to_go

    def _compute_statistics(self) -> None:
        """Compute normalization statistics across all agents and episodes."""
        # Flatten: [total_T * n_agents, state_dim]
        all_states = np.concatenate(
            [ep["states"].reshape(-1, ep["states"].shape[-1]) for ep in self.episodes],
            axis=0,
        )
        self.state_mean = all_states.mean(axis=0)
        self.state_std = all_states.std(axis=0) + 1e-8

        all_returns = np.concatenate(
            [ep["returns_to_go"].reshape(-1) for ep in self.episodes], axis=0
        )
        self.return_scale = max(np.abs(all_returns).max(), 1e-8)

    def __len__(self) -> int:
        return sum(ep["length"] for ep in self.episodes)

    def _get_episode_and_offset(self, idx: int) -> Tuple[int, int]:
        cumulative = 0
        for ep_idx, ep in enumerate(self.episodes):
            if idx < cumulative + ep["length"]:
                return ep_idx, idx - cumulative
            cumulative += ep["length"]
        raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a multi-agent context-length window.

        Args:
            idx: Flat index into the dataset.

        Returns:
            Dictionary with keys:
                - states: [n_agents, context_length, state_dim]
                - actions: [n_agents, context_length]
                - returns_to_go: [n_agents, context_length, 1]
                - timesteps: [n_agents, context_length]
                - attention_mask: [context_length] (shared across agents)
        """
        ep_idx, t = self._get_episode_and_offset(idx)
        ep = self.episodes[ep_idx]
        K = self.context_length
        N = self.n_agents

        end = min(t + 1, ep["length"])
        start = max(end - K, 0)
        actual_len = end - start

        # [actual_len, n_agents, state_dim] → [n_agents, actual_len, state_dim]
        states = ep["states"][start:end].copy().transpose(1, 0, 2)
        # [actual_len, n_agents] → [n_agents, actual_len]
        actions = ep["actions"][start:end].copy().T
        rtg = ep["returns_to_go"][start:end].copy().T  # [n_agents, actual_len]

        # Normalize
        if self.normalize_states_flag:
            states = (states - self.state_mean) / self.state_std
        if self.normalize_returns_flag:
            rtg = rtg / self.return_scale

        timesteps = np.arange(start, end, dtype=np.int64)

        # Pad to context_length (left-padding)
        pad_len = K - actual_len
        state_dim = states.shape[-1]

        padded_states = np.zeros((N, K, state_dim), dtype=np.float32)
        padded_actions = np.zeros((N, K), dtype=np.int64)
        padded_rtg = np.zeros((N, K), dtype=np.float32)
        padded_timesteps = np.zeros((N, K), dtype=np.int64)
        attention_mask = np.zeros(K, dtype=np.float32)

        padded_states[:, pad_len:, :] = states
        padded_actions[:, pad_len:] = actions
        padded_rtg[:, pad_len:] = rtg
        # All agents share the same timestep indices
        padded_timesteps[:, pad_len:] = np.broadcast_to(timesteps, (N, actual_len))
        attention_mask[pad_len:] = 1.0

        return {
            "states": torch.from_numpy(padded_states),
            "actions": torch.from_numpy(padded_actions),
            "returns_to_go": torch.from_numpy(padded_rtg).unsqueeze(-1),  # [N, K, 1]
            "timesteps": torch.from_numpy(padded_timesteps),
            "attention_mask": torch.from_numpy(attention_mask),
        }

    def get_state_stats(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return state normalization statistics."""
        return self.state_mean.copy(), self.state_std.copy()

    def get_return_scale(self) -> float:
        """Return the return normalization scale factor."""
        return self.return_scale
