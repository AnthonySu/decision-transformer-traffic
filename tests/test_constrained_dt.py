"""Tests for src.models.constrained_dt — Constrained Decision Transformer."""

import torch
import pytest

from src.models.constrained_dt import ConstrainedDecisionTransformer


# ======================================================================
# Constants
# ======================================================================

STATE_DIM = 11
ACT_DIM = 4
HIDDEN_DIM = 64
N_LAYERS = 2
N_HEADS = 4
MAX_LENGTH = 10
MAX_EP_LEN = 100
BATCH_SIZE = 2


# ======================================================================
# Fixtures
# ======================================================================

@pytest.fixture
def model():
    """A small Constrained DT model for testing."""
    return ConstrainedDecisionTransformer(
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
    """Helper to create valid input tensors for ConstrainedDT."""
    states = torch.randn(batch_size, seq_len, STATE_DIM)
    actions = torch.randint(0, ACT_DIM, (batch_size, seq_len))
    returns_to_go = torch.randn(batch_size, seq_len, 1)
    costs_to_go = torch.rand(batch_size, seq_len, 1)  # non-negative costs
    timesteps = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    return states, actions, returns_to_go, costs_to_go, timesteps


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

    def test_has_cost_embedding(self, model):
        """Model should have a cost embedding layer (embed_cost)."""
        assert hasattr(model, "embed_cost")
        assert isinstance(model.embed_cost, torch.nn.Linear)
        assert model.embed_cost.in_features == 1
        assert model.embed_cost.out_features == HIDDEN_DIM

    def test_has_cost_prediction_head(self, model):
        """Model should have a cost prediction head (predict_cost)."""
        assert hasattr(model, "predict_cost")

    def test_causal_mask_shape(self, model):
        """Causal mask should have shape [4*max_length, 4*max_length]."""
        expected_size = 4 * MAX_LENGTH  # 4 tokens per step for constrained DT
        assert model.causal_mask.shape == (expected_size, expected_size)

    def test_tokens_per_step(self, model):
        """ConstrainedDT uses 4 tokens per step (R, C, s, a)."""
        assert model.TOKENS_PER_STEP == 4


# ======================================================================
# test_forward_shape
# ======================================================================

class TestForwardShape:
    """Tests for forward pass output shapes."""

    def test_output_is_dict(self, model):
        """Forward should return a dictionary."""
        states, actions, rtg, ctg, timesteps = _make_inputs(BATCH_SIZE, MAX_LENGTH)
        outputs = model(states, actions, rtg, ctg, timesteps)
        assert isinstance(outputs, dict)

    def test_action_logits_shape(self, model):
        """Forward should return action_logits of shape [B, T, act_dim]."""
        states, actions, rtg, ctg, timesteps = _make_inputs(BATCH_SIZE, MAX_LENGTH)
        outputs = model(states, actions, rtg, ctg, timesteps)
        assert "action_logits" in outputs
        assert outputs["action_logits"].shape == (BATCH_SIZE, MAX_LENGTH, ACT_DIM)

    def test_cost_preds_shape(self, model):
        """Forward should return cost_preds of shape [B, T, 1]."""
        states, actions, rtg, ctg, timesteps = _make_inputs(BATCH_SIZE, MAX_LENGTH)
        outputs = model(states, actions, rtg, ctg, timesteps)
        assert "cost_preds" in outputs
        assert outputs["cost_preds"].shape == (BATCH_SIZE, MAX_LENGTH, 1)

    def test_output_shape_shorter_sequence(self, model):
        """Forward should work with sequences shorter than max_length."""
        seq_len = 5
        states, actions, rtg, ctg, timesteps = _make_inputs(BATCH_SIZE, seq_len)
        outputs = model(states, actions, rtg, ctg, timesteps)
        assert outputs["action_logits"].shape == (BATCH_SIZE, seq_len, ACT_DIM)
        assert outputs["cost_preds"].shape == (BATCH_SIZE, seq_len, 1)

    def test_output_shape_single_step(self, model):
        """Forward should work with a single timestep."""
        states, actions, rtg, ctg, timesteps = _make_inputs(BATCH_SIZE, 1)
        outputs = model(states, actions, rtg, ctg, timesteps)
        assert outputs["action_logits"].shape == (BATCH_SIZE, 1, ACT_DIM)
        assert outputs["cost_preds"].shape == (BATCH_SIZE, 1, 1)

    def test_output_dtype(self, model):
        """Output tensors should be float32."""
        states, actions, rtg, ctg, timesteps = _make_inputs(BATCH_SIZE, MAX_LENGTH)
        outputs = model(states, actions, rtg, ctg, timesteps)
        assert outputs["action_logits"].dtype == torch.float32
        assert outputs["cost_preds"].dtype == torch.float32


# ======================================================================
# test_cost_conditioning
# ======================================================================

class TestCostConditioning:
    """Tests that different cost-to-go values produce different actions."""

    def test_different_costs_produce_different_outputs(self, model):
        """Different cost-to-go values should produce different action logits."""
        model.eval()
        states, actions, rtg, ctg, timesteps = _make_inputs(1, 5)

        # Run with low cost budget
        ctg_low = torch.zeros_like(ctg)
        outputs_low = model(states, actions, rtg, ctg_low, timesteps)

        # Run with high cost budget
        ctg_high = torch.ones_like(ctg) * 100.0
        outputs_high = model(states, actions, rtg, ctg_high, timesteps)

        # Action logits should differ when cost conditioning changes
        diff = (outputs_low["action_logits"] - outputs_high["action_logits"]).abs().sum()
        assert diff > 1e-6, (
            "Different cost-to-go values should produce different action logits"
        )

    def test_different_costs_produce_different_cost_preds(self, model):
        """Different cost-to-go values should produce different cost predictions."""
        model.eval()
        states, actions, rtg, ctg, timesteps = _make_inputs(1, 5)

        ctg_low = torch.zeros_like(ctg)
        outputs_low = model(states, actions, rtg, ctg_low, timesteps)

        ctg_high = torch.ones_like(ctg) * 100.0
        outputs_high = model(states, actions, rtg, ctg_high, timesteps)

        diff = (outputs_low["cost_preds"] - outputs_high["cost_preds"]).abs().sum()
        assert diff > 1e-6, (
            "Different cost-to-go values should produce different cost predictions"
        )


# ======================================================================
# test_get_action
# ======================================================================

class TestGetAction:
    """Tests for the get_action inference method."""

    def test_returns_valid_int(self, model):
        """get_action should return an integer in [0, act_dim)."""
        model.eval()
        states, actions, rtg, ctg, timesteps = _make_inputs(1, 5)
        action = model.get_action(states, actions, rtg, ctg, timesteps)
        assert isinstance(action, int)
        assert 0 <= action < ACT_DIM

    def test_argmax_mode(self, model):
        """Without sampling, get_action should return argmax (deterministic)."""
        model.eval()
        states, actions, rtg, ctg, timesteps = _make_inputs(1, 5)
        a1 = model.get_action(states, actions, rtg, ctg, timesteps, sample=False)
        a2 = model.get_action(states, actions, rtg, ctg, timesteps, sample=False)
        assert a1 == a2, "Argmax mode should be deterministic"

    def test_sampling_mode(self, model):
        """With sampling, get_action should still return a valid int."""
        model.eval()
        states, actions, rtg, ctg, timesteps = _make_inputs(1, 5)
        action = model.get_action(
            states, actions, rtg, ctg, timesteps, sample=True, temperature=1.0
        )
        assert isinstance(action, int)
        assert 0 <= action < ACT_DIM

    def test_top_k_sampling(self, model):
        """top_k sampling should still return a valid action."""
        model.eval()
        states, actions, rtg, ctg, timesteps = _make_inputs(1, 5)
        action = model.get_action(
            states, actions, rtg, ctg, timesteps, sample=True, top_k=2
        )
        assert isinstance(action, int)
        assert 0 <= action < ACT_DIM

    def test_truncates_long_sequence(self, model):
        """get_action should handle sequences longer than max_length."""
        model.eval()
        long_seq = MAX_LENGTH + 10
        states, actions, rtg, ctg, timesteps = _make_inputs(1, long_seq)
        action = model.get_action(states, actions, rtg, ctg, timesteps)
        assert isinstance(action, int)
        assert 0 <= action < ACT_DIM

    def test_get_action_with_cost(self, model):
        """get_action_with_cost should return (int, float)."""
        model.eval()
        states, actions, rtg, ctg, timesteps = _make_inputs(1, 5)
        action, cost = model.get_action_with_cost(states, actions, rtg, ctg, timesteps)
        assert isinstance(action, int)
        assert 0 <= action < ACT_DIM
        assert isinstance(cost, float)


# ======================================================================
# test_compute_loss
# ======================================================================

class TestComputeLoss:
    """Tests for the compute_loss static method."""

    def test_loss_computation(self, model):
        """compute_loss should return dict with loss, action_loss, cost_loss."""
        states, actions, rtg, ctg, timesteps = _make_inputs(BATCH_SIZE, MAX_LENGTH)
        outputs = model(states, actions, rtg, ctg, timesteps)

        target_actions = torch.randint(0, ACT_DIM, (BATCH_SIZE, MAX_LENGTH))
        target_costs = torch.rand(BATCH_SIZE, MAX_LENGTH, 1)

        losses = ConstrainedDecisionTransformer.compute_loss(
            outputs, target_actions, target_costs
        )

        assert "loss" in losses
        assert "action_loss" in losses
        assert "cost_loss" in losses

    def test_loss_is_scalar(self, model):
        """Each loss should be a scalar tensor."""
        states, actions, rtg, ctg, timesteps = _make_inputs(BATCH_SIZE, MAX_LENGTH)
        outputs = model(states, actions, rtg, ctg, timesteps)

        target_actions = torch.randint(0, ACT_DIM, (BATCH_SIZE, MAX_LENGTH))
        target_costs = torch.rand(BATCH_SIZE, MAX_LENGTH, 1)

        losses = ConstrainedDecisionTransformer.compute_loss(
            outputs, target_actions, target_costs
        )

        for key in ["loss", "action_loss", "cost_loss"]:
            assert losses[key].dim() == 0, f"{key} should be a scalar"
            assert losses[key].item() >= 0, f"{key} should be non-negative"

    def test_cost_weight_affects_total(self, model):
        """Different cost_weight should produce different total loss."""
        states, actions, rtg, ctg, timesteps = _make_inputs(BATCH_SIZE, MAX_LENGTH)
        outputs = model(states, actions, rtg, ctg, timesteps)

        target_actions = torch.randint(0, ACT_DIM, (BATCH_SIZE, MAX_LENGTH))
        target_costs = torch.rand(BATCH_SIZE, MAX_LENGTH, 1)

        losses_low = ConstrainedDecisionTransformer.compute_loss(
            outputs, target_actions, target_costs, cost_weight=0.01
        )
        losses_high = ConstrainedDecisionTransformer.compute_loss(
            outputs, target_actions, target_costs, cost_weight=10.0
        )

        # action_loss is the same but total differs due to cost_weight
        torch.testing.assert_close(
            losses_low["action_loss"], losses_high["action_loss"]
        )
        assert losses_low["loss"].item() != losses_high["loss"].item()


# ======================================================================
# test_gradient_flow
# ======================================================================

class TestGradientFlow:
    """Tests that gradients flow correctly through the model."""

    def test_loss_backward(self, model):
        """loss.backward() should work without errors."""
        states, actions, rtg, ctg, timesteps = _make_inputs(BATCH_SIZE, MAX_LENGTH)
        outputs = model(states, actions, rtg, ctg, timesteps)

        target_actions = torch.randint(0, ACT_DIM, (BATCH_SIZE, MAX_LENGTH))
        target_costs = torch.rand(BATCH_SIZE, MAX_LENGTH, 1)

        losses = ConstrainedDecisionTransformer.compute_loss(
            outputs, target_actions, target_costs
        )
        losses["loss"].backward()

        # Check that gradients exist on at least one parameter
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
            if p.requires_grad
        )
        assert has_grad, "No gradients flowed through the model"

    def test_cost_embedding_receives_gradients(self, model):
        """The cost embedding layer should receive gradients."""
        states, actions, rtg, ctg, timesteps = _make_inputs(BATCH_SIZE, MAX_LENGTH)
        outputs = model(states, actions, rtg, ctg, timesteps)

        target_actions = torch.randint(0, ACT_DIM, (BATCH_SIZE, MAX_LENGTH))
        target_costs = torch.rand(BATCH_SIZE, MAX_LENGTH, 1)

        losses = ConstrainedDecisionTransformer.compute_loss(
            outputs, target_actions, target_costs
        )
        losses["loss"].backward()

        # embed_cost weight should have non-zero gradient
        assert model.embed_cost.weight.grad is not None
        assert model.embed_cost.weight.grad.abs().sum() > 0, (
            "embed_cost should receive gradients"
        )

    def test_optimizer_step(self, model):
        """An optimizer step should change model parameters."""
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        states, actions, rtg, ctg, timesteps = _make_inputs(BATCH_SIZE, MAX_LENGTH)

        # Save initial params
        initial_params = {
            name: p.clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }

        outputs = model(states, actions, rtg, ctg, timesteps)
        target_actions = torch.randint(0, ACT_DIM, (BATCH_SIZE, MAX_LENGTH))
        target_costs = torch.rand(BATCH_SIZE, MAX_LENGTH, 1)

        losses = ConstrainedDecisionTransformer.compute_loss(
            outputs, target_actions, target_costs
        )
        losses["loss"].backward()
        optimizer.step()

        # At least some parameters should have changed
        changed = 0
        for name, p in model.named_parameters():
            if p.requires_grad and not torch.equal(initial_params[name], p):
                changed += 1
        assert changed > 0, "No parameters changed after optimizer step"
