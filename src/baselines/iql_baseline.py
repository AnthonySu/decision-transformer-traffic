"""Implicit Q-Learning baseline for offline EV corridor optimization.

IQL avoids querying out-of-distribution actions by using expectile
regression on the value function and advantage-weighted regression
for policy extraction. This makes it particularly stable for offline RL.

Reference:
    Kostrikov et al., "Offline Reinforcement Learning with Implicit
    Q-Learning", ICLR 2022.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.baselines.cql_baseline import OfflineRLDataset

# ======================================================================
# Network Components
# ======================================================================


class _MLP(nn.Module):
    """Simple feedforward MLP used for V, Q, and policy networks."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU())
            d = hidden_dim
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ======================================================================
# IQL Agent
# ======================================================================


class IQLPolicy:
    """Implicit Q-Learning (IQL) for offline traffic signal control.

    IQL (Kostrikov et al., 2022) learns a value function using expectile
    regression and extracts a policy using advantage-weighted regression.
    This avoids querying OOD actions, making it stable for offline RL.

    The three key components:

    1. **Value network V(s)** -- trained with expectile regression on Q
       values, using asymmetric L2 loss with parameter tau.
    2. **Q-network Q(s,a)** -- trained to match r + gamma * V(s'), where
       V is the learned value function (no max over actions needed).
    3. **Policy network pi(a|s)** -- trained with advantage-weighted
       regression: weight = exp(beta * (Q(s,a) - V(s))).

    Parameters
    ----------
    state_dim : int
        Dimensionality of the observation vector.
    act_dim : int
        Number of discrete actions.
    hidden_dim : int
        Width of hidden layers in all networks.
    n_layers : int
        Number of hidden layers in each network.
    lr : float
        Learning rate for Adam optimizers.
    gamma : float
        Discount factor.
    tau : float
        Expectile for value function regression. Values > 0.5 push V(s)
        toward the upper quantiles of Q, effectively approximating the
        max without explicit maximization. Default 0.7.
    beta : float
        Inverse temperature for advantage-weighted regression. Higher
        values make the policy more greedy. Default 3.0.
    target_update_rate : float
        Polyak averaging coefficient for Q-target updates. Default 0.005.
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
        tau: float = 0.7,
        beta: float = 3.0,
        target_update_rate: float = 0.005,
        device: str = "cpu",
    ) -> None:
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.target_update_rate = target_update_rate
        self.device = torch.device(device)

        # Q-network: maps state -> Q-values for each action [batch, act_dim]
        self.q_net = _MLP(state_dim, act_dim, hidden_dim, n_layers).to(self.device)
        self.q_target = copy.deepcopy(self.q_net)
        self.q_target.requires_grad_(False)

        # Value network: maps state -> scalar V(s) [batch, 1]
        self.v_net = _MLP(state_dim, 1, hidden_dim, n_layers).to(self.device)

        # Policy network: maps state -> action logits [batch, act_dim]
        self.policy_net = _MLP(state_dim, act_dim, hidden_dim, n_layers).to(self.device)

        # Separate optimizers for each component
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.v_optimizer = torch.optim.Adam(self.v_net.parameters(), lr=lr)
        self.policy_optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

        self._update_count = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        trajectories: OfflineRLDataset,
        n_epochs: int = 50,
        batch_size: int = 64,
        log_interval: int = 5,
    ) -> Dict[str, List[float]]:
        """Train on an offline dataset.

        Parameters
        ----------
        trajectories : OfflineRLDataset
            Pre-loaded offline transitions (s, a, r, s', done).
        n_epochs : int
            Number of passes over the dataset.
        batch_size : int
            Mini-batch size.
        log_interval : int
            Print losses every this many epochs.

        Returns
        -------
        dict
            Training history with keys ``"v_loss"``, ``"q_loss"``,
            ``"policy_loss"``, ``"total_loss"`` (lists of per-epoch averages).
        """
        states_t, actions_t, rewards_t, next_states_t, dones_t = (
            trajectories.as_tensors(self.device)
        )

        td = TensorDataset(states_t, actions_t, rewards_t, next_states_t, dones_t)
        loader = DataLoader(td, batch_size=batch_size, shuffle=True, drop_last=False)

        history: Dict[str, List[float]] = {
            "v_loss": [],
            "q_loss": [],
            "policy_loss": [],
            "total_loss": [],
        }

        for epoch in range(1, n_epochs + 1):
            epoch_v = 0.0
            epoch_q = 0.0
            epoch_pi = 0.0
            epoch_total = 0.0
            n_batches = 0

            for s, a, r, ns, d in loader:
                v_loss, q_loss, pi_loss = self._update(s, a, r, ns, d)
                epoch_v += v_loss
                epoch_q += q_loss
                epoch_pi += pi_loss
                epoch_total += v_loss + q_loss + pi_loss
                n_batches += 1

            denom = max(n_batches, 1)
            avg_v = epoch_v / denom
            avg_q = epoch_q / denom
            avg_pi = epoch_pi / denom
            avg_total = epoch_total / denom

            history["v_loss"].append(avg_v)
            history["q_loss"].append(avg_q)
            history["policy_loss"].append(avg_pi)
            history["total_loss"].append(avg_total)

            if epoch % log_interval == 0 or epoch == 1:
                print(
                    f"  Epoch {epoch:3d}/{n_epochs} | "
                    f"V {avg_v:.4f} | Q {avg_q:.4f} | "
                    f"Pi {avg_pi:.4f} | Total {avg_total:.4f}"
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
        """Single gradient step for all three networks.

        Returns
        -------
        tuple[float, float, float]
            (v_loss, q_loss, policy_loss) as Python floats.
        """
        # ----------------------------------------------------------
        # 1. Value network: expectile regression on Q-values
        # ----------------------------------------------------------
        with torch.no_grad():
            q_all = self.q_target(states)  # [B, act_dim]
            q_taken = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]

        v_pred = self.v_net(states).squeeze(1)  # [B]
        diff = q_taken - v_pred
        # Asymmetric L2: weight = tau if diff > 0 else (1 - tau)
        weight = torch.where(diff > 0, self.tau, 1.0 - self.tau)
        v_loss = (weight * diff.pow(2)).mean()

        self.v_optimizer.zero_grad()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.v_net.parameters(), max_norm=1.0)
        self.v_optimizer.step()

        # ----------------------------------------------------------
        # 2. Q-network: fitted Q with V(s') as target (no max)
        # ----------------------------------------------------------
        q_all_online = self.q_net(states)  # [B, act_dim]
        q_pred = q_all_online.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B]

        with torch.no_grad():
            v_next = self.v_net(next_states).squeeze(1)  # [B]
            q_target_val = rewards + self.gamma * (1.0 - dones) * v_next

        q_loss = F.mse_loss(q_pred, q_target_val)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.q_optimizer.step()

        # ----------------------------------------------------------
        # 3. Policy: advantage-weighted regression
        # ----------------------------------------------------------
        with torch.no_grad():
            # Recompute advantage using updated networks
            q_for_adv = self.q_net(states)
            q_a = q_for_adv.gather(1, actions.unsqueeze(1)).squeeze(1)
            v_s = self.v_net(states).squeeze(1)
            advantage = q_a - v_s
            # Clamp for numerical stability before exp
            exp_advantage = torch.exp(
                torch.clamp(self.beta * advantage, max=20.0)
            )
            # Normalize weights within batch
            weights = exp_advantage / exp_advantage.sum()

        logits = self.policy_net(states)  # [B, act_dim]
        log_probs = F.log_softmax(logits, dim=1)
        log_prob_taken = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Weighted negative log-likelihood
        policy_loss = -(weights * log_prob_taken).sum()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.policy_optimizer.step()

        # ----------------------------------------------------------
        # 4. Soft-update Q-target
        # ----------------------------------------------------------
        self._update_count += 1
        with torch.no_grad():
            for p, tp in zip(self.q_net.parameters(), self.q_target.parameters()):
                tp.data.mul_(1.0 - self.target_update_rate)
                tp.data.add_(self.target_update_rate * p.data)

        return v_loss.item(), q_loss.item(), policy_loss.item()

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """Select an action given a single observation.

        Parameters
        ----------
        obs : np.ndarray
            Observation vector of shape ``[state_dim]``.
        deterministic : bool
            If True, return argmax of policy logits. If False, sample
            from the softmax distribution.

        Returns
        -------
        int
            Selected action index.
        """
        with torch.no_grad():
            state = torch.tensor(
                obs, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            logits = self.policy_net(state).squeeze(0)

            if deterministic:
                return int(logits.argmax().item())
            else:
                probs = F.softmax(logits, dim=0)
                return int(torch.multinomial(probs, 1).item())

    def select_action(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """Alias for :meth:`act` to match CQLAgent interface."""
        return self.act(obs, deterministic=deterministic)

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
                "v_net": self.v_net.state_dict(),
                "policy_net": self.policy_net.state_dict(),
                "q_optimizer": self.q_optimizer.state_dict(),
                "v_optimizer": self.v_optimizer.state_dict(),
                "policy_optimizer": self.policy_optimizer.state_dict(),
                "state_dim": self.state_dim,
                "act_dim": self.act_dim,
                "gamma": self.gamma,
                "tau": self.tau,
                "beta": self.beta,
                "target_update_rate": self.target_update_rate,
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
        self.v_net.load_state_dict(ckpt["v_net"])
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.q_optimizer.load_state_dict(ckpt["q_optimizer"])
        self.v_optimizer.load_state_dict(ckpt["v_optimizer"])
        self.policy_optimizer.load_state_dict(ckpt["policy_optimizer"])
        self._update_count = ckpt.get("update_count", 0)
