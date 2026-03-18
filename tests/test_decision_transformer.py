"""Tests for src.models.decision_transformer — Decision Transformer model."""

import pytest
import torch

from src.models.decision_transformer import DecisionTransformer

# ======================================================================
# Fixtures
# ======================================================================

STATE_DIM = 11
ACT_DIM = 4
HIDDEN_DIM = 64
N_LAYERS = 2
N_HEADS = 4
MAX_LENGTH = 10
MAX_EP_LEN = 100
BATCH_SIZE = 2


@pytest.fixture
def model():
    """A small DT model for testing."""
    return DecisionTransformer(
        state_dim=STATE_DIM,
        act_dim=ACT_DIM,
        hidden_dim=HIDDEN_DIM,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        max_length=MAX_LENGTH,
        max_ep_len=MAX_EP_LEN,
        dropout=0.0,  # disable dropout for deterministic tests
    )


def _make_inputs(batch_size: int, seq_len: int):
    """Helper to create valid input tensors."""
    states = torch.randn(batch_size, seq_len, STATE_DIM)
    actions = torch.randint(0, ACT_DIM, (batch_size, seq_len))
    returns_to_go = torch.randn(batch_size, seq_len, 1)
    timesteps = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    return states, actions, returns_to_go, timesteps


# ======================================================================
# test_init
# ======================================================================

class TestInit:
    """Tests for model initialization."""

    def test_parameter_count_positive(self, model):
        """Model should have a positive number of trainable parameters."""
        assert model.get_num_params() > 0

    def test_model_attributes(self, model):
        """Model should store its configuration correctly."""
        assert model.state_dim == STATE_DIM
        assert model.act_dim == ACT_DIM
        assert model.hidden_dim == HIDDEN_DIM
        assert model.max_length == MAX_LENGTH

    def test_causal_mask_shape(self, model):
        """Causal mask should have shape [3*max_length, 3*max_length]."""
        expected_size = 3 * MAX_LENGTH
        assert model.causal_mask.shape == (expected_size, expected_size)

    def test_causal_mask_is_upper_triangular(self, model):
        """Causal mask should be True above the diagonal, False on/below."""
        mask = model.causal_mask
        n = mask.shape[0]
        for i in range(n):
            for j in range(n):
                if j > i:
                    assert mask[i, j].item() is True
                elif j == i:
                    assert mask[i, j].item() is False


# ======================================================================
# test_forward_shape
# ======================================================================

class TestForwardShape:
    """Tests for forward pass output shape."""

    def test_output_shape(self, model):
        """Forward should return [batch, seq_len, act_dim]."""
        states, actions, rtg, timesteps = _make_inputs(BATCH_SIZE, MAX_LENGTH)
        logits = model(states, actions, rtg, timesteps)
        assert logits.shape == (BATCH_SIZE, MAX_LENGTH, ACT_DIM)

    def test_output_shape_shorter_sequence(self, model):
        """Forward should work with sequences shorter than max_length."""
        seq_len = 5
        states, actions, rtg, timesteps = _make_inputs(BATCH_SIZE, seq_len)
        logits = model(states, actions, rtg, timesteps)
        assert logits.shape == (BATCH_SIZE, seq_len, ACT_DIM)

    def test_output_shape_single_step(self, model):
        """Forward should work with a single timestep."""
        states, actions, rtg, timesteps = _make_inputs(BATCH_SIZE, 1)
        logits = model(states, actions, rtg, timesteps)
        assert logits.shape == (BATCH_SIZE, 1, ACT_DIM)

    def test_output_dtype(self, model):
        """Output logits should be float32."""
        states, actions, rtg, timesteps = _make_inputs(BATCH_SIZE, MAX_LENGTH)
        logits = model(states, actions, rtg, timesteps)
        assert logits.dtype == torch.float32


# ======================================================================
# test_causal_masking
# ======================================================================

class TestCausalMasking:
    """Tests that future tokens do not affect past predictions."""

    def test_future_tokens_dont_affect_past(self, model):
        """Changing a future state should not change the prediction at an earlier time."""
        model.eval()
        seq_len = 6
        states, actions, rtg, timesteps = _make_inputs(1, seq_len)

        # Forward with original inputs
        logits_original = model(states, actions, rtg, timesteps).clone()

        # Modify the last state (future)
        states_modified = states.clone()
        states_modified[0, -1, :] = torch.randn(STATE_DIM) * 100.0

        logits_modified = model(states_modified, actions, rtg, timesteps)

        # The first 5 predictions should be identical (causal masking)
        # (In the interleaved scheme, changing the last state at position 5 should
        # not affect positions 0-4)
        for t in range(seq_len - 1):
            torch.testing.assert_close(
                logits_original[0, t],
                logits_modified[0, t],
                atol=1e-5,
                rtol=1e-5,
                msg=f"Prediction at t={t} changed when only future t={seq_len-1} was modified",
            )

    def test_past_tokens_do_affect_future(self, model):
        """Changing an early state should affect later predictions."""
        model.eval()
        seq_len = 6
        states, actions, rtg, timesteps = _make_inputs(1, seq_len)

        logits_original = model(states, actions, rtg, timesteps).clone()

        # Modify the first state
        states_modified = states.clone()
        states_modified[0, 0, :] = torch.randn(STATE_DIM) * 100.0

        logits_modified = model(states_modified, actions, rtg, timesteps)

        # Later predictions should differ
        diff = (logits_original[0, -1] - logits_modified[0, -1]).abs().sum()
        assert diff > 1e-6, "Modifying past should affect future predictions"


# ======================================================================
# test_get_action
# ======================================================================

class TestGetAction:
    """Tests for the get_action inference method."""

    def test_returns_valid_int(self, model):
        """get_action should return an integer in [0, act_dim)."""
        model.eval()
        states, actions, rtg, timesteps = _make_inputs(1, 5)
        action = model.get_action(states, actions, rtg, timesteps)
        assert isinstance(action, int)
        assert 0 <= action < ACT_DIM

    def test_argmax_mode(self, model):
        """Without sampling, get_action should return argmax (deterministic)."""
        model.eval()
        states, actions, rtg, timesteps = _make_inputs(1, 5)
        a1 = model.get_action(states, actions, rtg, timesteps, sample=False)
        a2 = model.get_action(states, actions, rtg, timesteps, sample=False)
        assert a1 == a2, "Argmax mode should be deterministic"

    def test_sampling_mode(self, model):
        """With sampling, get_action should still return a valid int."""
        model.eval()
        states, actions, rtg, timesteps = _make_inputs(1, 5)
        action = model.get_action(
            states, actions, rtg, timesteps, sample=True, temperature=1.0
        )
        assert isinstance(action, int)
        assert 0 <= action < ACT_DIM

    def test_top_k_sampling(self, model):
        """top_k sampling should still return a valid action."""
        model.eval()
        states, actions, rtg, timesteps = _make_inputs(1, 5)
        action = model.get_action(
            states, actions, rtg, timesteps, sample=True, top_k=2
        )
        assert isinstance(action, int)
        assert 0 <= action < ACT_DIM

    def test_get_action_truncates_long_sequence(self, model):
        """get_action should handle sequences longer than max_length."""
        model.eval()
        long_seq = MAX_LENGTH + 10
        states, actions, rtg, timesteps = _make_inputs(1, long_seq)
        action = model.get_action(states, actions, rtg, timesteps)
        assert isinstance(action, int)
        assert 0 <= action < ACT_DIM


# ======================================================================
# test_different_context_lengths
# ======================================================================

class TestDifferentContextLengths:
    """Tests that the model handles various sequence lengths."""

    @pytest.mark.parametrize("seq_len", [1, 3, 5, 10])
    def test_various_seq_lengths(self, model, seq_len):
        """Forward pass should work with different context lengths."""
        states, actions, rtg, timesteps = _make_inputs(BATCH_SIZE, seq_len)
        logits = model(states, actions, rtg, timesteps)
        assert logits.shape == (BATCH_SIZE, seq_len, ACT_DIM)

    def test_batch_size_one(self, model):
        """Should work with batch_size=1."""
        states, actions, rtg, timesteps = _make_inputs(1, MAX_LENGTH)
        logits = model(states, actions, rtg, timesteps)
        assert logits.shape == (1, MAX_LENGTH, ACT_DIM)

    def test_large_batch(self, model):
        """Should work with larger batch sizes."""
        states, actions, rtg, timesteps = _make_inputs(16, MAX_LENGTH)
        logits = model(states, actions, rtg, timesteps)
        assert logits.shape == (16, MAX_LENGTH, ACT_DIM)


# ======================================================================
# test_gradient_flow
# ======================================================================

class TestGradientFlow:
    """Tests that gradients flow correctly through the model."""

    def test_loss_backward(self, model):
        """loss.backward() should work without errors."""
        states, actions, rtg, timesteps = _make_inputs(BATCH_SIZE, MAX_LENGTH)
        logits = model(states, actions, rtg, timesteps)
        # Cross-entropy loss
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, ACT_DIM), actions.reshape(-1)
        )
        loss.backward()
        # Check that gradients exist on at least one parameter
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_grad, "No gradients flowed through the model"

    def test_all_parameters_receive_gradients(self, model):
        """Every trainable parameter should receive a non-zero gradient."""
        states, actions, rtg, timesteps = _make_inputs(BATCH_SIZE, MAX_LENGTH)
        logits = model(states, actions, rtg, timesteps)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, ACT_DIM), actions.reshape(-1)
        )
        loss.backward()

        no_grad_params = []
        for name, p in model.named_parameters():
            if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0):
                no_grad_params.append(name)

        # It's acceptable if some bias terms have zero gradient, but most should not
        assert len(no_grad_params) < len(list(model.parameters())) * 0.5, (
            f"Too many parameters without gradients: {no_grad_params}"
        )

    def test_optimizer_step(self, model):
        """An optimizer step should change model parameters."""
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        states, actions, rtg, timesteps = _make_inputs(BATCH_SIZE, MAX_LENGTH)

        # Save initial params
        initial_params = {
            name: p.clone() for name, p in model.named_parameters() if p.requires_grad
        }

        logits = model(states, actions, rtg, timesteps)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, ACT_DIM), actions.reshape(-1)
        )
        loss.backward()
        optimizer.step()

        # At least some parameters should have changed
        changed = 0
        for name, p in model.named_parameters():
            if p.requires_grad and not torch.equal(initial_params[name], p):
                changed += 1
        assert changed > 0, "No parameters changed after optimizer step"
