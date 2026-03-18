"""Decision Transformer for offline reinforcement learning.

Implements the Decision Transformer architecture from:
    Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., Abbeel, P.,
    Srinivas, A., & Mordatch, I. (2021). Decision Transformer: Reinforcement Learning
    via Sequence Modeling. NeurIPS 2021.

The model frames RL as conditional sequence modeling: given a desired return-to-go,
past states, and past actions, it autoregressively predicts the next action.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Multi-head causal (masked) self-attention.

    Implements scaled dot-product attention with a causal mask so that each
    position can only attend to itself and earlier positions in the sequence.
    """

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"

        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape [batch, seq_len, hidden_dim].
            causal_mask: Boolean mask of shape [seq_len, seq_len] where True
                indicates positions that should be masked (not attended to).

        Returns:
            Output tensor of shape [batch, seq_len, hidden_dim].
        """
        B, T, C = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention: [B, n_heads, T, T]
        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale

        if causal_mask is not None:
            attn = attn.masked_fill(causal_mask[:T, :T].unsqueeze(0).unsqueeze(0), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)  # [B, n_heads, T, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))
        return out


class TransformerBlock(nn.Module):
    """A single transformer block with pre-norm architecture (GPT-2 style).

    Applies LayerNorm before attention and before the feed-forward network,
    with residual connections around each sub-layer.
    """

    def __init__(self, hidden_dim: int, n_heads: int, dropout: float = 0.1,
                 activation: str = "gelu") -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, n_heads, dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)

        act_fn = nn.GELU() if activation == "gelu" else nn.ReLU()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            act_fn,
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), causal_mask=causal_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class DecisionTransformer(nn.Module):
    """Decision Transformer for offline reinforcement learning.

    Frames RL as sequence modeling over (return-to-go, state, action) triples.
    A GPT-2 style causal transformer processes the interleaved sequence and
    predicts discrete actions conditioned on desired returns and past trajectory.

    Args:
        state_dim: Dimensionality of the state observation vector.
        act_dim: Number of discrete actions.
        hidden_dim: Transformer hidden / embedding dimension.
        n_layers: Number of transformer blocks.
        n_heads: Number of attention heads per block.
        max_length: Maximum context length (number of timesteps in a window).
        max_ep_len: Maximum episode length (for timestep embeddings).
        dropout: Dropout probability used throughout the model.
        activation: Activation function for feed-forward layers ("gelu" or "relu").
    """

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
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_dim)

        # Layer norm applied after summing modality + timestep embeddings
        self.ln = nn.LayerNorm(hidden_dim)

        # ---- Transformer backbone ----
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, dropout, activation)
            for _ in range(n_layers)
        ])

        # ---- Prediction head ----
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU() if activation == "gelu" else nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
        )

        # Pre-compute causal mask (upper-triangular = True → masked)
        # Maximum interleaved sequence length = 3 * max_length
        max_seq = 3 * max_length
        causal_mask = torch.triu(torch.ones(max_seq, max_seq, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask)

        self.apply(self._init_weights)

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

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for training.

        Args:
            states: Float tensor of shape [batch, seq_len, state_dim].
            actions: Long tensor of shape [batch, seq_len] (discrete action indices).
            returns_to_go: Float tensor of shape [batch, seq_len, 1].
            timesteps: Long tensor of shape [batch, seq_len] (absolute timestep indices).

        Returns:
            action_logits: Float tensor of shape [batch, seq_len, act_dim].
                Logits predicted from state token positions.
        """
        B, T, _ = states.shape

        # ---- Compute embeddings ----
        time_emb = self.embed_timestep(timesteps)  # [B, T, H]

        state_emb = self.embed_state(states) + time_emb         # [B, T, H]
        action_emb = self.embed_action(actions) + time_emb       # [B, T, H]
        return_emb = self.embed_return(returns_to_go) + time_emb # [B, T, H]

        # ---- Interleave as (R_1, s_1, a_1, R_2, s_2, a_2, ...) ----
        # Stack along a new token-type dimension then reshape
        # [B, T, 3, H] → [B, 3*T, H]
        sequence = torch.stack([return_emb, state_emb, action_emb], dim=2)
        sequence = sequence.reshape(B, 3 * T, self.hidden_dim)

        # Apply layer norm
        sequence = self.ln(sequence)

        # ---- Transformer forward ----
        for block in self.blocks:
            sequence = block(sequence, causal_mask=self.causal_mask)

        # ---- Extract state positions and predict actions ----
        # State tokens are at positions 1, 4, 7, ... (indices 3*t + 1 for t=0..T-1)
        state_positions = torch.arange(0, T, device=states.device) * 3 + 1
        state_hidden = sequence[:, state_positions, :]  # [B, T, H]

        action_logits = self.predict_action(state_hidden)  # [B, T, act_dim]
        return action_logits

    @torch.no_grad()
    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> int:
        """Get a single action for inference (autoregressive generation).

        Truncates the input sequences to the most recent ``max_length`` timesteps,
        runs a forward pass, and returns the predicted action for the last timestep.

        Args:
            states: Float tensor of shape [1, seq_len, state_dim].
            actions: Long tensor of shape [1, seq_len].
            returns_to_go: Float tensor of shape [1, seq_len, 1].
            timesteps: Long tensor of shape [1, seq_len].
            sample: If True, sample from the distribution; otherwise take argmax.
            temperature: Softmax temperature for sampling.
            top_k: If set, restrict sampling to top-k logits.

        Returns:
            Predicted action as a Python int.
        """
        # Truncate to max_length
        states = states[:, -self.max_length:]
        actions = actions[:, -self.max_length:]
        returns_to_go = returns_to_go[:, -self.max_length:]
        timesteps = timesteps[:, -self.max_length:]

        action_logits = self.forward(states, actions, returns_to_go, timesteps)
        logits = action_logits[:, -1, :]  # last timestep: [1, act_dim]

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

    def get_num_params(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
