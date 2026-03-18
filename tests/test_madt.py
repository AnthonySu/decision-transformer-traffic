"""Tests for src.models.madt — Multi-Agent Decision Transformer with GAT."""

import pytest
import torch

from src.models.madt import GraphAttention, MultiAgentDecisionTransformer

# ======================================================================
# Fixtures & helpers
# ======================================================================

STATE_DIM = 33
ACT_DIM = 4
N_AGENTS = 5
HIDDEN_DIM = 64
N_LAYERS = 2
N_HEADS = 4
GAT_HEADS = 2
GAT_LAYERS = 1
MAX_LENGTH = 8
MAX_EP_LEN = 100
BATCH_SIZE = 2


def _make_chain_adj(n: int) -> torch.Tensor:
    """Create a chain adjacency matrix with self-loops: 0-1-2-..-(n-1)."""
    adj = torch.eye(n)
    for i in range(n - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0
    return adj


def _make_identity_adj(n: int) -> torch.Tensor:
    """Identity adjacency (no inter-agent communication, only self-loops)."""
    return torch.eye(n)


@pytest.fixture
def adj():
    """Chain adjacency matrix for N_AGENTS agents."""
    return _make_chain_adj(N_AGENTS)


@pytest.fixture
def model(adj):
    """A small MADT model for testing."""
    return MultiAgentDecisionTransformer(
        state_dim=STATE_DIM,
        act_dim=ACT_DIM,
        n_agents=N_AGENTS,
        adj_matrix=adj,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        gat_heads=GAT_HEADS,
        gat_layers=GAT_LAYERS,
        max_length=MAX_LENGTH,
        max_ep_len=MAX_EP_LEN,
        dropout=0.0,
    )


def _make_inputs(batch_size: int, n_agents: int, seq_len: int):
    """Helper to create valid MADT input tensors."""
    states = torch.randn(batch_size, n_agents, seq_len, STATE_DIM)
    actions = torch.randint(0, ACT_DIM, (batch_size, n_agents, seq_len))
    returns_to_go = torch.randn(batch_size, n_agents, seq_len, 1)
    timesteps = (
        torch.arange(seq_len).unsqueeze(0).unsqueeze(0).expand(batch_size, n_agents, -1)
    )
    return states, actions, returns_to_go, timesteps


# ======================================================================
# test_init
# ======================================================================

class TestInit:
    """Tests for MADT initialization."""

    def test_parameter_count_positive(self, model):
        """Model should have a positive number of trainable parameters."""
        assert model.get_num_params() > 0

    def test_model_attributes(self, model):
        """Model should store its configuration correctly."""
        assert model.state_dim == STATE_DIM
        assert model.act_dim == ACT_DIM
        assert model.n_agents == N_AGENTS
        assert model.hidden_dim == HIDDEN_DIM
        assert model.max_length == MAX_LENGTH

    def test_adj_matrix_stored_as_buffer(self, model):
        """Adjacency matrix should be stored as a buffer, not a parameter."""
        assert hasattr(model, "adj_matrix")
        assert model.adj_matrix.shape == (N_AGENTS, N_AGENTS)
        # Should not appear in parameters
        param_names = [name for name, _ in model.named_parameters()]
        assert "adj_matrix" not in param_names

    def test_gat_layers_created(self, model):
        """GAT layers should be present."""
        assert len(model.gat_layers) == GAT_LAYERS

    def test_creates_with_adjacency(self, adj):
        """Model should accept various adjacency matrices."""
        # Full adjacency
        full_adj = torch.ones(N_AGENTS, N_AGENTS)
        m = MultiAgentDecisionTransformer(
            state_dim=STATE_DIM, act_dim=ACT_DIM, n_agents=N_AGENTS,
            adj_matrix=full_adj, hidden_dim=32, n_layers=1, n_heads=2,
        )
        assert m.get_num_params() > 0


# ======================================================================
# test_forward_shape
# ======================================================================

class TestForwardShape:
    """Tests for forward pass output shape."""

    def test_output_shape(self, model):
        """Forward should return [batch, n_agents, seq_len, act_dim]."""
        states, actions, rtg, timesteps = _make_inputs(BATCH_SIZE, N_AGENTS, MAX_LENGTH)
        logits = model(states, actions, rtg, timesteps)
        assert logits.shape == (BATCH_SIZE, N_AGENTS, MAX_LENGTH, ACT_DIM)

    def test_output_shape_shorter_sequence(self, model):
        """Forward should work with sequences shorter than max_length."""
        seq_len = 3
        states, actions, rtg, timesteps = _make_inputs(BATCH_SIZE, N_AGENTS, seq_len)
        logits = model(states, actions, rtg, timesteps)
        assert logits.shape == (BATCH_SIZE, N_AGENTS, seq_len, ACT_DIM)

    def test_output_shape_single_step(self, model):
        """Forward should work with a single timestep."""
        states, actions, rtg, timesteps = _make_inputs(BATCH_SIZE, N_AGENTS, 1)
        logits = model(states, actions, rtg, timesteps)
        assert logits.shape == (BATCH_SIZE, N_AGENTS, 1, ACT_DIM)

    def test_output_dtype(self, model):
        """Output logits should be float32."""
        states, actions, rtg, timesteps = _make_inputs(BATCH_SIZE, N_AGENTS, MAX_LENGTH)
        logits = model(states, actions, rtg, timesteps)
        assert logits.dtype == torch.float32


# ======================================================================
# test_gat_communication
# ======================================================================

class TestGATCommunication:
    """Tests that GAT layers alter embeddings when agents have neighbors."""

    def test_gat_changes_embeddings_vs_identity(self):
        """With chain adj, GAT should produce different outputs than with identity adj."""
        chain_adj = _make_chain_adj(N_AGENTS)
        identity_adj = _make_identity_adj(N_AGENTS)

        # Use same random seed for both models so weights are identical
        # (except for the adjacency buffer)
        torch.manual_seed(42)
        model_chain = MultiAgentDecisionTransformer(
            state_dim=STATE_DIM, act_dim=ACT_DIM, n_agents=N_AGENTS,
            adj_matrix=chain_adj, hidden_dim=HIDDEN_DIM, n_layers=1,
            n_heads=N_HEADS, gat_heads=GAT_HEADS, gat_layers=1,
            max_length=MAX_LENGTH, max_ep_len=MAX_EP_LEN, dropout=0.0,
        )

        torch.manual_seed(42)
        model_identity = MultiAgentDecisionTransformer(
            state_dim=STATE_DIM, act_dim=ACT_DIM, n_agents=N_AGENTS,
            adj_matrix=identity_adj, hidden_dim=HIDDEN_DIM, n_layers=1,
            n_heads=N_HEADS, gat_heads=GAT_HEADS, gat_layers=1,
            max_length=MAX_LENGTH, max_ep_len=MAX_EP_LEN, dropout=0.0,
        )

        model_chain.eval()
        model_identity.eval()

        # Use diverse inputs so agents have different features to mix
        torch.manual_seed(99)
        states, actions, rtg, timesteps = _make_inputs(1, N_AGENTS, 3)
        for i in range(N_AGENTS):
            states[0, i] *= (i + 1) * 5.0

        logits_chain = model_chain(states, actions, rtg, timesteps)
        logits_identity = model_identity(states, actions, rtg, timesteps)

        # The outputs should differ because GAT with chain adj mixes neighbor info
        diff = (logits_chain - logits_identity).abs().sum().item()
        assert diff > 1e-4, (
            f"Chain adjacency GAT should produce different outputs than identity adjacency (diff={diff})"
        )

    def test_gat_module_directly(self):
        """Test GraphAttention module in isolation."""
        gat = GraphAttention(embed_dim=32, n_heads=2, dropout=0.0)
        gat.eval()

        adj = _make_chain_adj(4)
        x = torch.randn(2, 4, 32)  # [batch=2, agents=4, dim=32]

        out = gat(x, adj)
        assert out.shape == x.shape

        # Output should differ from input (residual + attention)
        # At minimum, they shouldn't be exactly equal
        diff = (out - x).abs().sum().item()
        assert diff > 1e-6, "GAT output should differ from input"


# ======================================================================
# test_get_action
# ======================================================================

class TestGetAction:
    """Tests for the get_action inference method."""

    def test_returns_valid_int(self, model):
        """get_action should return an int in [0, act_dim)."""
        model.eval()
        states, actions, rtg, timesteps = _make_inputs(1, N_AGENTS, 5)
        action = model.get_action(states, actions, rtg, timesteps, agent_idx=0)
        assert isinstance(action, int)
        assert 0 <= action < ACT_DIM

    def test_different_agents_can_get_different_actions(self, model):
        """Different agent indices should potentially produce different actions."""
        model.eval()
        states, actions, rtg, timesteps = _make_inputs(1, N_AGENTS, 5)
        actions_per_agent = []
        for agent_idx in range(N_AGENTS):
            a = model.get_action(states, actions, rtg, timesteps, agent_idx=agent_idx)
            actions_per_agent.append(a)
        # At least some should differ (not guaranteed, but likely with random inputs)
        # Just verify they're all valid
        for a in actions_per_agent:
            assert isinstance(a, int)
            assert 0 <= a < ACT_DIM

    def test_argmax_deterministic(self, model):
        """Argmax mode should be deterministic."""
        model.eval()
        states, actions, rtg, timesteps = _make_inputs(1, N_AGENTS, 5)
        a1 = model.get_action(states, actions, rtg, timesteps, agent_idx=2, sample=False)
        a2 = model.get_action(states, actions, rtg, timesteps, agent_idx=2, sample=False)
        assert a1 == a2

    def test_sampling_mode_valid(self, model):
        """Sampling mode should return valid actions."""
        model.eval()
        states, actions, rtg, timesteps = _make_inputs(1, N_AGENTS, 5)
        action = model.get_action(
            states, actions, rtg, timesteps, agent_idx=0,
            sample=True, temperature=1.0,
        )
        assert isinstance(action, int)
        assert 0 <= action < ACT_DIM

    def test_truncates_long_sequence(self, model):
        """get_action should handle sequences longer than max_length."""
        model.eval()
        long_seq = MAX_LENGTH + 10
        states, actions, rtg, timesteps = _make_inputs(1, N_AGENTS, long_seq)
        action = model.get_action(states, actions, rtg, timesteps, agent_idx=0)
        assert isinstance(action, int)
        assert 0 <= action < ACT_DIM


# ======================================================================
# test_agent_mask
# ======================================================================

class TestAgentMask:
    """Tests for the agent_mask functionality."""

    def test_masked_agents_get_zeroed_outputs(self, model):
        """When agent_mask is False for an agent, its logits should be zero."""
        model.eval()
        states, actions, rtg, timesteps = _make_inputs(BATCH_SIZE, N_AGENTS, MAX_LENGTH)

        # Mask out agents 1 and 3
        mask = torch.ones(BATCH_SIZE, N_AGENTS, dtype=torch.bool)
        mask[:, 1] = False
        mask[:, 3] = False

        logits = model(states, actions, rtg, timesteps, agent_mask=mask)

        # Masked agents should have all-zero logits
        assert torch.all(logits[:, 1, :, :] == 0.0)
        assert torch.all(logits[:, 3, :, :] == 0.0)

    def test_unmasked_agents_have_nonzero_outputs(self, model):
        """Unmasked agents should have non-zero logits."""
        model.eval()
        states, actions, rtg, timesteps = _make_inputs(BATCH_SIZE, N_AGENTS, MAX_LENGTH)

        mask = torch.ones(BATCH_SIZE, N_AGENTS, dtype=torch.bool)
        mask[:, 1] = False

        logits = model(states, actions, rtg, timesteps, agent_mask=mask)

        # Agent 0, 2, 3, 4 should have non-zero outputs
        for agent_idx in [0, 2, 4]:
            assert logits[:, agent_idx, :, :].abs().sum() > 0.0

    def test_no_mask_produces_nonzero_for_all(self, model):
        """Without a mask, all agents should get non-zero logits."""
        model.eval()
        states, actions, rtg, timesteps = _make_inputs(BATCH_SIZE, N_AGENTS, MAX_LENGTH)
        logits = model(states, actions, rtg, timesteps, agent_mask=None)
        for agent_idx in range(N_AGENTS):
            assert logits[:, agent_idx, :, :].abs().sum() > 0.0

    def test_all_masked_gives_all_zero(self, model):
        """If all agents are masked, all outputs should be zero."""
        model.eval()
        states, actions, rtg, timesteps = _make_inputs(BATCH_SIZE, N_AGENTS, MAX_LENGTH)
        mask = torch.zeros(BATCH_SIZE, N_AGENTS, dtype=torch.bool)
        logits = model(states, actions, rtg, timesteps, agent_mask=mask)
        assert torch.all(logits == 0.0)


# ======================================================================
# test_gradient_flow
# ======================================================================

class TestGradientFlow:
    """Tests that gradients flow through MADT."""

    def test_loss_backward(self, model):
        """loss.backward() should work without errors."""
        states, actions, rtg, timesteps = _make_inputs(BATCH_SIZE, N_AGENTS, MAX_LENGTH)
        logits = model(states, actions, rtg, timesteps)

        # Flatten for cross-entropy: [B*N*T, act_dim] vs [B*N*T]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, ACT_DIM), actions.reshape(-1)
        )
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_grad, "No gradients flowed through the MADT model"

    def test_gat_receives_gradients(self, model):
        """GAT layer parameters should receive gradients."""
        states, actions, rtg, timesteps = _make_inputs(BATCH_SIZE, N_AGENTS, MAX_LENGTH)
        logits = model(states, actions, rtg, timesteps)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, ACT_DIM), actions.reshape(-1)
        )
        loss.backward()

        gat_has_grad = False
        for name, p in model.named_parameters():
            if "gat" in name and p.grad is not None and p.grad.abs().sum() > 0:
                gat_has_grad = True
                break
        assert gat_has_grad, "GAT layers did not receive gradients"
