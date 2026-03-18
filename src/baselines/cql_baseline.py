"""Conservative Q-Learning baseline for offline EV corridor optimization.

CQL adds a conservative regularizer to the Q-function to prevent
overestimation of out-of-distribution actions. This is the primary
value-based offline RL baseline for comparison with DT.

Reference:
    Kumar et al., "Conservative Q-Learning for Offline Reinforcement
    Learning", NeurIPS 2020.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Tuple, Union

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ======================================================================
# Q-Network
# ======================================================================

class QNetwork(nn.Module):
    """Simple feedforward Q-network for discrete actions.

    Maps (state) -> Q-values for each action.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the observation.
    act_dim : int
        Number of discrete actions.
    hidden_dim : int
        Width of hidden layers.
    n_layers : int
        Number of hidden layers.
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = state_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return Q-values for all actions: [batch, act_dim]."""
        return self.net(state)


# ======================================================================
# Offline Dataset Loader
# ======================================================================

class OfflineRLDataset:
    """Load an HDF5 dataset (same format as DT) into flat (s, a, r, s', done) tuples.

    Parameters
    ----------
    data_path : str | Path
        Path to the HDF5 file produced by DataCollector.
    """

    def __init__(self, data_path: Union[str, Path]) -> None:
        self.states: np.ndarray
        self.actions: np.ndarray
        self.rewards: np.ndarray
        self.next_states: np.ndarray
        self.dones: np.ndarray
        self._load(data_path)

    def _load(self, data_path: Union[str, Path]) -> None:
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        all_s: List[np.ndarray] = []
        all_a: List[np.ndarray] = []
        all_r: List[np.ndarray] = []
        all_ns: List[np.ndarray] = []
        all_d: List[np.ndarray] = []

        with h5py.File(data_path, "r") as f:
            episode_keys = sorted(
                [k for k in f.keys() if k.startswith("episode")],
                key=lambda k: int(k.split("_")[-1]),
            )
            for key in episode_keys:
                grp = f[key]
                states = np.array(grp["states"], dtype=np.float32)
                raw_actions = np.array(grp["actions"], dtype=np.int64)
                # For centralized env, reduce multi-dim actions to first element
                if raw_actions.ndim > 1:
                    actions = raw_actions[:, 0]
                else:
                    actions = raw_actions
                rewards = np.array(grp["rewards"], dtype=np.float32)
                dones = np.array(grp["dones"], dtype=np.bool_)

                T = len(states)
                if T < 2:
                    continue

                # Build transition tuples (s, a, r, s', done)
                all_s.append(states[:-1])
                all_a.append(actions[:-1])
                all_r.append(rewards[:-1])
                all_ns.append(states[1:])
                all_d.append(dones[:-1])

        self.states = np.concatenate(all_s, axis=0)
        self.actions = np.concatenate(all_a, axis=0)
        self.rewards = np.concatenate(all_r, axis=0)
        self.next_states = np.concatenate(all_ns, axis=0)
        self.dones = np.concatenate(all_d, axis=0).astype(np.float32)

    def as_tensors(
        self, device: torch.device | str = "cpu"
    ) -> Tuple[torch.Tensor, ...]:
        """Return all arrays as PyTorch tensors on the given device."""
        return (
            torch.tensor(self.states, dtype=torch.float32, device=device),
            torch.tensor(self.actions, dtype=torch.long, device=device),
            torch.tensor(self.rewards, dtype=torch.float32, device=device),
            torch.tensor(self.next_states, dtype=torch.float32, device=device),
            torch.tensor(self.dones, dtype=torch.float32, device=device),
        )

    def __len__(self) -> int:
        return len(self.states)


# ======================================================================
# CQL Agent
# ======================================================================

class CQLAgent:
    """Conservative Q-Learning agent for discrete action spaces.

    Implements CQL(H) with log-sum-exp conservative penalty::

        L_CQL = alpha * (log_sum_exp(Q(s, a')) - Q(s, a_data))

    added on top of the standard Bellman error.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the observation vector.
    act_dim : int
        Number of discrete actions.
    hidden_dim : int
        Width of Q-network hidden layers.
    n_layers : int
        Number of hidden layers in Q-network.
    lr : float
        Learning rate for Adam optimizer.
    gamma : float
        Discount factor.
    alpha : float
        CQL conservative penalty weight. Higher values produce more
        conservative Q-estimates.
    target_update_freq : int
        How often (in gradient steps) to hard-update the target network.
    device : str
        PyTorch device string.
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
        lr: float = 3e-4,
        gamma: float = 0.99,
        alpha: float = 1.0,
        target_update_freq: int = 100,
        device: str = "cpu",
    ) -> None:
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.alpha = alpha
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)

        # Q-network and target network
        self.q_net = QNetwork(state_dim, act_dim, hidden_dim, n_layers).to(self.device)
        self.q_target = copy.deepcopy(self.q_net)
        self.q_target.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

        # Training counters
        self._update_count = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_offline(
        self,
        dataset: OfflineRLDataset,
        n_epochs: int = 50,
        batch_size: int = 64,
        log_interval: int = 5,
    ) -> Dict[str, List[float]]:
        """Train on an offline dataset.

        Parameters
        ----------
        dataset : OfflineRLDataset
            Pre-loaded offline transitions.
        n_epochs : int
            Number of passes over the dataset.
        batch_size : int
            Mini-batch size.
        log_interval : int
            Print loss every this many epochs.

        Returns
        -------
        dict
            Training history with keys ``"bellman_loss"``, ``"cql_loss"``,
            ``"total_loss"`` (lists of per-epoch averages).
        """
        states_t, actions_t, rewards_t, next_states_t, dones_t = dataset.as_tensors(
            self.device
        )

        td = TensorDataset(states_t, actions_t, rewards_t, next_states_t, dones_t)
        loader = DataLoader(td, batch_size=batch_size, shuffle=True, drop_last=False)

        history: Dict[str, List[float]] = {
            "bellman_loss": [],
            "cql_loss": [],
            "total_loss": [],
        }

        for epoch in range(1, n_epochs + 1):
            epoch_bellman = 0.0
            epoch_cql = 0.0
            epoch_total = 0.0
            n_batches = 0

            for s, a, r, ns, d in loader:
                bellman, cql, total = self._update(s, a, r, ns, d)
                epoch_bellman += bellman
                epoch_cql += cql
                epoch_total += total
                n_batches += 1

            avg_bellman = epoch_bellman / max(n_batches, 1)
            avg_cql = epoch_cql / max(n_batches, 1)
            avg_total = epoch_total / max(n_batches, 1)

            history["bellman_loss"].append(avg_bellman)
            history["cql_loss"].append(avg_cql)
            history["total_loss"].append(avg_total)

            if epoch % log_interval == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:3d}/{n_epochs} | "
                    f"Bellman {avg_bellman:.4f} | "
                    f"CQL {avg_cql:.4f} | "
                    f"Total {avg_total:.4f}"
                )

        return history

    def _update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> Tuple[float, float, float]:
        """Single gradient step.

        Returns
        -------
        tuple[float, float, float]
            (bellman_loss, cql_loss, total_loss) as Python floats.
        """
        # Current Q-values for taken actions
        q_all = self.q_net(states)  # [B, act_dim]
        q_taken = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]

        # Target Q-values (double DQN style: use online net to select action)
        with torch.no_grad():
            q_next_online = self.q_net(next_states)
            best_actions = q_next_online.argmax(dim=1, keepdim=True)
            q_next_target = self.q_target(next_states)
            q_next = q_next_target.gather(1, best_actions).squeeze(1)
            target = rewards + self.gamma * (1.0 - dones) * q_next

        # Bellman loss
        bellman_loss = F.mse_loss(q_taken, target)

        # CQL conservative penalty: log_sum_exp(Q(s, a')) - Q(s, a_data)
        # log_sum_exp over all actions for each state
        logsumexp_q = torch.logsumexp(q_all, dim=1).mean()
        data_q = q_taken.mean()
        cql_loss = logsumexp_q - data_q

        total_loss = bellman_loss + self.alpha * cql_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target network
        self._update_count += 1
        if self._update_count % self.target_update_freq == 0:
            self.q_target.load_state_dict(self.q_net.state_dict())

        return (
            bellman_loss.item(),
            cql_loss.item(),
            total_loss.item(),
        )

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def select_action(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """Select an action given a single observation.

        Parameters
        ----------
        obs : np.ndarray
            Observation vector of shape ``[state_dim]``.
        deterministic : bool
            If True, return argmax Q. If False, sample proportionally
            to softmax Q-values.

        Returns
        -------
        int
            Selected action index.
        """
        with torch.no_grad():
            state = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.q_net(state).squeeze(0)

            if deterministic:
                return int(q_values.argmax().item())
            else:
                probs = F.softmax(q_values, dim=0)
                return int(torch.multinomial(probs, 1).item())

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Union[str, Path]) -> None:
        """Save agent state to disk.

        Parameters
        ----------
        path : str | Path
            File path (will be created / overwritten).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "q_target": self.q_target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "state_dim": self.state_dim,
                "act_dim": self.act_dim,
                "gamma": self.gamma,
                "alpha": self.alpha,
                "update_count": self._update_count,
            },
            path,
        )

    def load(self, path: Union[str, Path]) -> None:
        """Load agent state from disk.

        Parameters
        ----------
        path : str | Path
            Path to a checkpoint saved by :meth:`save`.
        """
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.q_target.load_state_dict(ckpt["q_target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self._update_count = ckpt.get("update_count", 0)
