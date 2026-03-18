"""Constrained Decision Transformer for safe offline reinforcement learning.

Extends the Decision Transformer with cost-conditioned generation, inspired by:
    Liu, Z., Lu, Z., Xiong, H., Zhong, Z., Hu, Z., Zhang, S., ... & Liu, Z. (2023).
    Constrained Decision Transformer for Offline Safe Reinforcement Learning.
    ICML 2023.

The key extension is dual conditioning: the model receives both a *return-to-go*
(how much reward remains) and a *cost-to-go* (how much constraint budget remains).
At inference the operator specifies two knobs—target return and cost budget—so the
policy can maximise EV travel-time reduction while respecting a hard limit on
civilian delay.

Sequence layout (4 tokens per timestep):
    (R_0, C_0, s_0, a_0, R_1, C_1, s_1, a_1, ...)

where R_t = return-to-go, C_t = cost-to-go, s_t = state, a_t = action.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.decision_transformer import CausalSelfAttention, TransformerBlock


class ConstrainedDecisionTransformer(nn.Module):
    """Decision Transformer with safety constraints for civilian delay limits.

    Extends the standard DT by interleaving *cost-to-go* tokens into the
    sequence so the model can learn the joint (return, cost) trade-off
    surface from offline data.  At deployment, dispatchers specify both a
    target return (desired EV travel-time reduction) **and** a maximum
    civilian delay budget, yielding a two-knob control interface.

    Architecture overview
    ---------------------
    * 4 tokens per timestep: ``(R_t, C_t, s_t, a_t)``
    * Shared GPT-2-style causal transformer backbone.
    * Two prediction heads:
        - ``predict_action``: maps state-position hidden states to action logits.
        - ``predict_cost``: maps cost-position hidden states to next-step cost
          (used as an auxiliary training signal and for online cost tracking).

    Args:
        state_dim: Dimensionality of the state observation vector.
        act_dim: Number of discrete actions.
        hidden_dim: Transformer hidden / embedding dimension.
        n_layers: Number of transformer blocks.
        n_heads: Number of attention heads per block.
        max_length: Maximum context length (number of timesteps in a window).
        max_ep_len: Maximum episode length (for learned timestep embeddings).
        dropout: Dropout probability used throughout the model.
        activation: Activation function for feed-forward layers (``"gelu"`` or ``"relu"``).
    """

    TOKENS_PER_STEP: int = 4  # (return, cost, state, action)

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_dim: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        max_length: int = 30,
        max_ep_len: int = 300,
        dropout: float = 0.1,
        activation: str = "gelu",
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.max_length = max_length

        # ---- Embedding layers ----
        self.embed_state = nn.Linear(state_dim, hidden_dim)
        self.embed_action = nn.Embedding(act_dim, hidden_dim)
        self.embed_return = nn.Linear(1, hidden_dim)
        self.embed_cost = nn.Linear(1, hidden_dim)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_dim)

        # Layer norm applied after summing modality + timestep embeddings
        self.ln = nn.LayerNorm(hidden_dim)

        # ---- Transformer backbone ----
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, dropout, activation)
            for _ in range(n_layers)
        ])

        # ---- Prediction heads ----
        act_fn: nn.Module = nn.GELU() if activation == "gelu" else nn.ReLU()

        self.predict_action = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, act_dim),
        )

        self.predict_cost = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            act_fn,
            nn.Linear(hidden_dim, 1),
        )

        # ---- Causal mask ----
        # Maximum interleaved sequence length = 4 * max_length
        max_seq = self.TOKENS_PER_STEP * max_length
        causal_mask = torch.triu(
            torch.ones(max_seq, max_seq, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_mask", causal_mask)

        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Initialize weights with small normal distribution for linear layers."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        costs_to_go: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for training.

        Args:
            states: Float tensor of shape ``[B, T, state_dim]``.
            actions: Long tensor of shape ``[B, T]`` (discrete action indices).
            returns_to_go: Float tensor of shape ``[B, T, 1]``.
            costs_to_go: Float tensor of shape ``[B, T, 1]``.  Each entry is the
                remaining constraint budget at that timestep (e.g. maximum
                additional civilian delay allowed from this point onward).
            timesteps: Long tensor of shape ``[B, T]`` (absolute timestep indices).

        Returns:
            Dictionary with keys:

            * ``"action_logits"``: ``[B, T, act_dim]`` — logits from state
              token positions (used for the behavioural-cloning cross-entropy
              loss).
            * ``"cost_preds"``: ``[B, T, 1]`` — predicted next-step cost from
              cost token positions (used for the auxiliary cost-prediction MSE
              loss).
        """
        B, T, _ = states.shape

        # ---- Compute embeddings (all share additive timestep embedding) ----
        time_emb = self.embed_timestep(timesteps)  # [B, T, H]

        return_emb = self.embed_return(returns_to_go) + time_emb  # [B, T, H]
        cost_emb = self.embed_cost(costs_to_go) + time_emb        # [B, T, H]
        state_emb = self.embed_state(states) + time_emb            # [B, T, H]
        action_emb = self.embed_action(actions) + time_emb         # [B, T, H]

        # ---- Interleave as (R_0, C_0, s_0, a_0, R_1, C_1, s_1, a_1, ...) ----
        # [B, T, 4, H] → [B, 4*T, H]
        sequence = torch.stack(
            [return_emb, cost_emb, state_emb, action_emb], dim=2
        )
        sequence = sequence.reshape(B, self.TOKENS_PER_STEP * T, self.hidden_dim)

        # Apply layer norm
        sequence = self.ln(sequence)

        # ---- Transformer forward ----
        for block in self.blocks:
            sequence = block(sequence, causal_mask=self.causal_mask)

        # ---- Extract modality-specific positions ----
        step_indices = torch.arange(0, T, device=states.device)

        # Cost tokens are at positions 4*t + 1 for t = 0 .. T-1
        cost_positions = step_indices * self.TOKENS_PER_STEP + 1
        cost_hidden = sequence[:, cost_positions, :]  # [B, T, H]
        cost_preds = self.predict_cost(cost_hidden)    # [B, T, 1]

        # State tokens are at positions 4*t + 2 for t = 0 .. T-1
        state_positions = step_indices * self.TOKENS_PER_STEP + 2
        state_hidden = sequence[:, state_positions, :]    # [B, T, H]
        action_logits = self.predict_action(state_hidden)  # [B, T, act_dim]

        return {
            "action_logits": action_logits,
            "cost_preds": cost_preds,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        costs_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> int:
        """Get a single action for inference (autoregressive generation).

        Truncates the input sequences to the most recent ``max_length``
        timesteps, runs a forward pass, and returns the predicted action
        for the last timestep.

        This is the **two-knob interface**: the caller controls both
        ``returns_to_go`` (target performance) and ``costs_to_go``
        (remaining safety budget) to steer behaviour.

        Args:
            states: Float tensor of shape ``[1, seq_len, state_dim]``.
            actions: Long tensor of shape ``[1, seq_len]``.
            returns_to_go: Float tensor of shape ``[1, seq_len, 1]``.
            costs_to_go: Float tensor of shape ``[1, seq_len, 1]``.
            timesteps: Long tensor of shape ``[1, seq_len]``.
            sample: If ``True``, sample from the distribution; otherwise argmax.
            temperature: Softmax temperature for sampling.
            top_k: If set, restrict sampling to top-k logits.

        Returns:
            Predicted action as a Python int.
        """
        # Truncate to context window
        states = states[:, -self.max_length:]
        actions = actions[:, -self.max_length:]
        returns_to_go = returns_to_go[:, -self.max_length:]
        costs_to_go = costs_to_go[:, -self.max_length:]
        timesteps = timesteps[:, -self.max_length:]

        outputs = self.forward(states, actions, returns_to_go, costs_to_go, timesteps)
        logits = outputs["action_logits"][:, -1, :]  # [1, act_dim]

        if sample:
            logits = logits / temperature
            if top_k is not None:
                topk_vals, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < topk_vals[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, num_samples=1).item()
        else:
            action = logits.argmax(dim=-1).item()

        return action

    @torch.no_grad()
    def get_action_with_cost(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        costs_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> Tuple[int, float]:
        """Get an action together with the predicted next-step cost.

        Same as :meth:`get_action` but additionally returns the model's
        cost prediction, which the caller can use to update the running
        cost-to-go budget for subsequent steps.

        Returns:
            Tuple of ``(action, predicted_cost)`` where *action* is a
            Python int and *predicted_cost* is a Python float.
        """
        states = states[:, -self.max_length:]
        actions = actions[:, -self.max_length:]
        returns_to_go = returns_to_go[:, -self.max_length:]
        costs_to_go = costs_to_go[:, -self.max_length:]
        timesteps = timesteps[:, -self.max_length:]

        outputs = self.forward(states, actions, returns_to_go, costs_to_go, timesteps)
        logits = outputs["action_logits"][:, -1, :]  # [1, act_dim]
        cost_pred = outputs["cost_preds"][:, -1, 0].item()  # scalar

        if sample:
            logits = logits / temperature
            if top_k is not None:
                topk_vals, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < topk_vals[:, -1:]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, num_samples=1).item()
        else:
            action = logits.argmax(dim=-1).item()

        return action, cost_pred

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_num_params(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def compute_loss(
        outputs: Dict[str, torch.Tensor],
        target_actions: torch.Tensor,
        target_costs: torch.Tensor,
        cost_weight: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """Compute the combined training loss.

        The total loss is::

            L = L_action + cost_weight * L_cost

        where ``L_action`` is the cross-entropy loss over discrete actions and
        ``L_cost`` is the MSE loss for next-step cost prediction.

        Args:
            outputs: Dict returned by :meth:`forward`.
            target_actions: Long tensor ``[B, T]`` of ground-truth action indices.
            target_costs: Float tensor ``[B, T, 1]`` of ground-truth next-step
                costs (e.g. the actual civilian delay incurred at each step).
            cost_weight: Scalar weight balancing the cost auxiliary loss
                relative to the action loss.

        Returns:
            Dictionary with keys ``"loss"`` (total), ``"action_loss"``,
            and ``"cost_loss"``.
        """
        action_logits = outputs["action_logits"]  # [B, T, act_dim]
        cost_preds = outputs["cost_preds"]          # [B, T, 1]

        # Action cross-entropy: flatten batch and time dims
        B, T, A = action_logits.shape
        action_loss = F.cross_entropy(
            action_logits.reshape(B * T, A),
            target_actions.reshape(B * T),
        )

        # Cost MSE
        cost_loss = F.mse_loss(cost_preds, target_costs)

        total_loss = action_loss + cost_weight * cost_loss

        return {
            "loss": total_loss,
            "action_loss": action_loss,
            "cost_loss": cost_loss,
        }
