"""Shared pytest fixtures for the EV-DT test suite."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

# Ensure project root is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@pytest.fixture
def small_network():
    """Build a minimal 3x3 grid network for testing."""
    from src.envs.network_utils import build_grid_network
    return build_grid_network(rows=3, cols=3)


@pytest.fixture
def small_env():
    """Create a small 3x3 EVCorridorEnv for testing."""
    from src.envs.ev_corridor_env import EVCorridorEnv
    env = EVCorridorEnv(rows=3, cols=3, max_steps=50, seed=42)
    return env


@pytest.fixture
def small_ma_env():
    """Create a small 3x3 EVCorridorMAEnv for testing."""
    from src.envs.ev_corridor_ma_env import EVCorridorMAEnv
    env = EVCorridorMAEnv(rows=3, cols=3, max_steps=50, seed=42,
                          origin="n0_0", destination="n2_2")
    return env


@pytest.fixture
def small_dt():
    """Create a tiny Decision Transformer for testing."""
    from src.models.decision_transformer import DecisionTransformer
    return DecisionTransformer(
        state_dim=99, act_dim=4, hidden_dim=32,
        n_layers=1, n_heads=2, max_length=5, max_ep_len=50,
    )


@pytest.fixture
def rng():
    """Provide a seeded numpy RNG."""
    return np.random.default_rng(42)


@pytest.fixture(autouse=True)
def torch_seed():
    """Set torch seed for reproducible tests."""
    torch.manual_seed(42)
