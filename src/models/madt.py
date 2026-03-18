"""Multi-Agent Decision Transformer with Graph Attention.

Extends the Decision Transformer with Graph Attention Network (GAT) layers for
spatial inter-agent communication over a traffic network topology. Each
intersection agent maintains its own (return-to-go, state, action) sequence,
and GAT layers enable agents to incorporate neighbor information at each timestep.

Reference architectures:
    - Decision Transformer (Chen et al., 2021)
    - Graph Attention Networks (Velickovic et al., 2018)
    - MADT: Multi-Agent Decision Transformer (various, 2022-2023)
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.decision_transformer import TransformerBlock


class GraphAttentionHead(nn.Module):
    """Single-head graph attention.

    Computes attention coefficients between connected nodes using a learned
    linear transformation and LeakyReLU-activated attention logits, then
    aggregates neighbor features accordingly.

    Args:
        embed_dim: Dimensionality of input and output node features.
        dropout: Dropout probability on attention coefficients.
    """

    def __init__(self, embed_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.W = nn.Linear(embed_dim, embed_dim, bias=False)
        # Attention mechanism: a^T [Wh_i || Wh_j]
        self.a_src = nn.Linear(embed_dim, 1, bias=False)
        self.a_dst = nn.Linear(embed_dim, 1, bias=False)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features of shape [batch, n_agents, embed_dim].
            adj: Adjacency matrix of shape [n_agents, n_agents]. Non-zero
                entries indicate edges (including self-loops).

        Returns:
            Updated node features of shape [batch, n_agents, embed_dim].
        """
        # x: [B, N, D]
        Wh = self.W(x)  # [B, N, D]

        # Compute attention logits for all pairs via additive mechanism
        e_src = self.a_src(Wh)  # [B, N, 1]
        e_dst = self.a_dst(Wh)  # [B, N, 1]
        # e_src[i] + e_dst[j] for all (i, j) → [B, N, N]
        attn_logits = e_src + e_dst.transpose(-2, -1)
        attn_logits = self.leaky_relu(attn_logits)

        # Mask non-adjacent pairs with -inf
        mask = (adj == 0).unsqueeze(0)  # [1, N, N]
        attn_logits = attn_logits.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(attn_logits, dim=-1)  # [B, N, N]
        attn_weights = self.dropout(attn_weights)

        # Handle potential NaN from softmax over all -inf rows (isolated nodes)
        attn_weights = attn_weights.nan_to_num(0.0)

        out = torch.matmul(attn_weights, Wh)  # [B, N, D]
        return out


class GraphAttention(nn.Module):
    """Multi-head graph attention for inter-agent communication.

    Concatenates outputs from multiple attention heads and projects back to
    the embedding dimension, followed by a residual connection and layer norm.

    Args:
        embed_dim: Dimensionality of node features.
        n_heads: Number of parallel attention heads.
        dropout: Dropout probability.
    """

    def __init__(self, embed_dim: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.heads = nn.ModuleList([
            GraphAttentionHead(embed_dim, dropout) for _ in range(n_heads)
        ])
        self.out_proj = nn.Linear(n_heads * embed_dim, embed_dim)
        self.ln = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features of shape [batch, n_agents, embed_dim].
            adj: Adjacency matrix of shape [n_agents, n_agents].

        Returns:
            Updated node features of shape [batch, n_agents, embed_dim].
        """
        head_outputs = [head(x, adj) for head in self.heads]
        multi_head = torch.cat(head_outputs, dim=-1)  # [B, N, n_heads * D]
        out = self.dropout(self.out_proj(multi_head))  # [B, N, D]
        # Residual connection + layer norm
        out = self.ln(out + x)
        return out


class MultiAgentDecisionTransformer(nn.Module):
    """Multi-Agent Decision Transformer with graph attention.

    Each agent processes its own (return-to-go, state, action) sequence through
    shared embedding layers and a shared causal transformer. Before the
    transformer, GAT layers allow agents to exchange information with their
    topological neighbors in the traffic network at each timestep.

    Args:
        state_dim: Dimensionality of per-agent state observation.
        act_dim: Number of discrete actions per agent.
        n_agents: Number of agents (intersections) in the network.
        adj_matrix: Adjacency matrix of shape [n_agents, n_agents] defining
            the traffic network topology. Should include self-loops.
        hidden_dim: Transformer / embedding hidden dimension.
        n_layers: Number of transformer blocks.
        n_heads: Number of attention heads in the transformer.
        gat_heads: Number of attention heads in GAT layers.
        gat_layers: Number of stacked GAT layers.
        max_length: Maximum context window (number of timesteps).
        max_ep_len: Maximum episode length (for timestep embeddings).
        dropout: Dropout probability.
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        n_agents: int,
        adj_matrix: torch.Tensor,
        hidden_dim: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        gat_heads: int = 4,
        gat_layers: int = 2,
        max_length: int = 20,
        max_ep_len: int = 300,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.state_dim = state_dim
        self.act_dim = act_dim
        self.n_agents = n_agents
        self.hidden_dim = hidden_dim
        self.max_length = max_length

        # Store adjacency as a buffer (not a parameter)
        self.register_buffer("adj_matrix", adj_matrix.float())

        # ---- Shared embedding layers (all agents share weights) ----
        self.embed_state = nn.Linear(state_dim, hidden_dim)
        self.embed_action = nn.Embedding(act_dim, hidden_dim)
        self.embed_return = nn.Linear(1, hidden_dim)
        self.embed_timestep = nn.Embedding(max_ep_len, hidden_dim)

        # Agent identity embedding so the shared transformer can distinguish agents
        self.embed_agent = nn.Embedding(n_agents, hidden_dim)

        self.ln = nn.LayerNorm(hidden_dim)

        # ---- Graph Attention layers ----
        self.gat_layers = nn.ModuleList([
            GraphAttention(hidden_dim, n_heads=gat_heads, dropout=dropout)
            for _ in range(gat_layers)
        ])

        # ---- Shared causal transformer ----
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # ---- Prediction head ----
        self.predict_action = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, act_dim),
        )

        # Causal mask
        max_seq = 3 * max_length
        causal_mask = torch.triu(torch.ones(max_seq, max_seq, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", causal_mask)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _embed_and_fuse(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Embed per-agent sequences and apply GAT for inter-agent communication.

        Args:
            states: [batch, n_agents, seq_len, state_dim].
            actions: [batch, n_agents, seq_len].
            returns_to_go: [batch, n_agents, seq_len, 1].
            timesteps: [batch, n_agents, seq_len].

        Returns:
            State embeddings enhanced with neighbor information:
                [batch, n_agents, seq_len, hidden_dim].
        """
        B, N, T, _ = states.shape

        # Compute timestep embeddings: [B, N, T, H]
        time_emb = self.embed_timestep(timesteps)

        # Agent identity embeddings: [N, H] → [1, N, 1, H]
        agent_ids = torch.arange(N, device=states.device)
        agent_emb = self.embed_agent(agent_ids).unsqueeze(0).unsqueeze(2)

        # Embed each modality and add positional info
        state_emb = self.embed_state(states) + time_emb + agent_emb     # [B, N, T, H]
        action_emb = self.embed_action(actions) + time_emb + agent_emb  # [B, N, T, H]
        return_emb = self.embed_return(returns_to_go) + time_emb + agent_emb  # [B, N, T, H]

        # ---- GAT: inter-agent communication at each timestep ----
        # Reshape state_emb for GAT: process each (batch, timestep) as a graph
        # [B, N, T, H] → [B*T, N, H]
        state_for_gat = state_emb.permute(0, 2, 1, 3).reshape(B * T, N, self.hidden_dim)

        for gat in self.gat_layers:
            state_for_gat = gat(state_for_gat, self.adj_matrix)

        # Reshape back: [B*T, N, H] → [B, N, T, H]
        state_emb = state_for_gat.reshape(B, T, N, self.hidden_dim).permute(0, 2, 1, 3)

        return state_emb, action_emb, return_emb

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        agent_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for training.

        Processes all agents in parallel using shared weights. Each agent's
        sequence is independently passed through the causal transformer after
        GAT-based spatial fusion.

        Args:
            states: Float tensor [batch, n_agents, seq_len, state_dim].
            actions: Long tensor [batch, n_agents, seq_len].
            returns_to_go: Float tensor [batch, n_agents, seq_len, 1].
            timesteps: Long tensor [batch, n_agents, seq_len].
            agent_mask: Optional bool tensor [batch, n_agents] where True
                indicates active agents. Inactive agents' outputs are zeroed.

        Returns:
            action_logits: Float tensor [batch, n_agents, seq_len, act_dim].
        """
        B, N, T, _ = states.shape

        state_emb, action_emb, return_emb = self._embed_and_fuse(
            states, actions, returns_to_go, timesteps
        )

        # ---- Process each agent through the shared transformer ----
        # Flatten batch and agent dims: [B*N, T, H]
        # Interleave (R, s, a) for each agent
        return_emb = return_emb.reshape(B * N, T, self.hidden_dim)
        state_emb = state_emb.reshape(B * N, T, self.hidden_dim)
        action_emb = action_emb.reshape(B * N, T, self.hidden_dim)

        # [B*N, T, 3, H] → [B*N, 3*T, H]
        sequence = torch.stack([return_emb, state_emb, action_emb], dim=2)
        sequence = sequence.reshape(B * N, 3 * T, self.hidden_dim)
        sequence = self.ln(sequence)

        for block in self.blocks:
            sequence = block(sequence, causal_mask=self.causal_mask)

        # Extract state positions: indices 1, 4, 7, ... = 3*t + 1
        state_positions = torch.arange(0, T, device=states.device) * 3 + 1
        state_hidden = sequence[:, state_positions, :]  # [B*N, T, H]

        action_logits = self.predict_action(state_hidden)  # [B*N, T, act_dim]
        action_logits = action_logits.reshape(B, N, T, self.act_dim)

        # Zero out inactive agents if mask is provided
        if agent_mask is not None:
            # agent_mask: [B, N] → [B, N, 1, 1]
            action_logits = action_logits * agent_mask.unsqueeze(-1).unsqueeze(-1).float()

        return action_logits

    @torch.no_grad()
    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        agent_idx: int,
        sample: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> int:
        """Get a single action for a specific agent during inference.

        Args:
            states: Float tensor [1, n_agents, seq_len, state_dim].
            actions: Long tensor [1, n_agents, seq_len].
            returns_to_go: Float tensor [1, n_agents, seq_len, 1].
            timesteps: Long tensor [1, n_agents, seq_len].
            agent_idx: Index of the agent to predict an action for.
            sample: If True, sample from the distribution; otherwise argmax.
            temperature: Softmax temperature for sampling.
            top_k: If set, restrict sampling to top-k logits.

        Returns:
            Predicted action as a Python int.
        """
        # Truncate to max_length
        states = states[:, :, -self.max_length:]
        actions = actions[:, :, -self.max_length:]
        returns_to_go = returns_to_go[:, :, -self.max_length:]
        timesteps = timesteps[:, :, -self.max_length:]

        action_logits = self.forward(states, actions, returns_to_go, timesteps)
        # Extract logits for the target agent at the last timestep
        logits = action_logits[0, agent_idx, -1, :]  # [act_dim]

        if sample:
            logits = logits / temperature
            if top_k is not None:
                topk_vals, _ = torch.topk(logits, top_k)
                logits[logits < topk_vals[-1]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, num_samples=1).item()
        else:
            action = logits.argmax(dim=-1).item()

        return action

    def get_num_params(self) -> int:
        """Return the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
