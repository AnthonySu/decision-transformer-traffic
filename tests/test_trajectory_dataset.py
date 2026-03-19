"""Tests for src.models.trajectory_dataset — TrajectoryDataset and MultiAgentTrajectoryDataset."""

import h5py
import numpy as np
import pytest
import torch

from src.models.trajectory_dataset import (
    MultiAgentTrajectoryDataset,
    TrajectoryDataset,
)

# ======================================================================
# Helpers — synthetic HDF5 data generation
# ======================================================================


def _make_single_agent_h5(
    path: str,
    n_episodes: int = 3,
    episode_lengths: list | None = None,
    state_dim: int = 8,
    seed: int = 0,
) -> str:
    """Create an HDF5 file with synthetic single-agent trajectory data.

    Returns the path to the created file.
    """
    rng = np.random.default_rng(seed)
    if episode_lengths is None:
        episode_lengths = [20, 15, 25]

    with h5py.File(path, "w") as f:
        for i in range(n_episodes):
            T = episode_lengths[i] if i < len(episode_lengths) else 20
            grp = f.create_group(f"episode_{i}")
            grp.create_dataset(
                "states", data=rng.standard_normal((T, state_dim)).astype(np.float32)
            )
            grp.create_dataset(
                "actions", data=rng.integers(0, 4, size=T).astype(np.int64)
            )
            grp.create_dataset(
                "rewards", data=(rng.standard_normal(T) * 2 - 1).astype(np.float32)
            )
            dones = np.zeros(T, dtype=bool)
            dones[-1] = True
            grp.create_dataset("dones", data=dones)
    return path


def _make_multi_agent_h5(
    path: str,
    n_episodes: int = 2,
    episode_lengths: list | None = None,
    n_agents: int = 3,
    state_dim: int = 6,
    seed: int = 0,
) -> str:
    """Create an HDF5 file with synthetic multi-agent trajectory data."""
    rng = np.random.default_rng(seed)
    if episode_lengths is None:
        episode_lengths = [15, 20]

    with h5py.File(path, "w") as f:
        for i in range(n_episodes):
            T = episode_lengths[i] if i < len(episode_lengths) else 15
            grp = f.create_group(f"episode_{i}")
            grp.create_dataset(
                "states",
                data=rng.standard_normal((T, n_agents, state_dim)).astype(np.float32),
            )
            grp.create_dataset(
                "actions",
                data=rng.integers(0, 4, size=(T, n_agents)).astype(np.int64),
            )
            grp.create_dataset(
                "rewards",
                data=(rng.standard_normal((T, n_agents)) * 2 - 1).astype(np.float32),
            )
            dones = np.zeros(T, dtype=bool)
            dones[-1] = True
            grp.create_dataset("dones", data=dones)
    return path


@pytest.fixture
def sa_h5_path(tmp_path):
    """Single-agent HDF5 dataset with 3 episodes of lengths [20, 15, 25]."""
    return _make_single_agent_h5(str(tmp_path / "sa_data.h5"))


@pytest.fixture
def sa_h5_path_short(tmp_path):
    """Single-agent HDF5 dataset with short episodes (shorter than typical context)."""
    return _make_single_agent_h5(
        str(tmp_path / "sa_short.h5"),
        n_episodes=2,
        episode_lengths=[3, 5],
        seed=7,
    )


@pytest.fixture
def sa_h5_path_single(tmp_path):
    """Single-agent HDF5 dataset with exactly 1 episode."""
    return _make_single_agent_h5(
        str(tmp_path / "sa_single.h5"),
        n_episodes=1,
        episode_lengths=[10],
        seed=99,
    )


@pytest.fixture
def ma_h5_path(tmp_path):
    """Multi-agent HDF5 dataset with 2 episodes."""
    return _make_multi_agent_h5(str(tmp_path / "ma_data.h5"))


# ======================================================================
# TrajectoryDataset tests
# ======================================================================


class TestTrajectoryDatasetConstruction:
    """Test construction and basic properties."""

    def test_construction_len(self, sa_h5_path):
        """Dataset length equals total timesteps across all episodes."""
        ds = TrajectoryDataset(sa_h5_path, context_length=10)
        assert len(ds) == 20 + 15 + 25  # sum of episode lengths

    def test_num_episodes_loaded(self, sa_h5_path):
        """All episodes are loaded from HDF5."""
        ds = TrajectoryDataset(sa_h5_path, context_length=10)
        assert len(ds.episodes) == 3

    def test_state_dim_property(self, sa_h5_path):
        """state_dim property matches the data."""
        ds = TrajectoryDataset(sa_h5_path, context_length=10)
        assert ds.state_dim == 8

    def test_act_dim_property(self, sa_h5_path):
        """act_dim should be at least 4 (minimum phases)."""
        ds = TrajectoryDataset(sa_h5_path, context_length=10)
        assert ds.act_dim >= 4

    def test_file_not_found(self, tmp_path):
        """Raises FileNotFoundError for missing data file."""
        with pytest.raises(FileNotFoundError):
            TrajectoryDataset(str(tmp_path / "nonexistent.h5"))

    def test_empty_h5(self, tmp_path):
        """Raises ValueError when HDF5 has no episodes."""
        empty_path = str(tmp_path / "empty.h5")
        with h5py.File(empty_path, "w") as f:
            f.attrs["metadata"] = "empty"
        with pytest.raises(ValueError, match="No episodes found"):
            TrajectoryDataset(empty_path)


class TestTrajectoryDatasetGetItem:
    """Test __getitem__ returns correct structure and shapes."""

    def test_getitem_keys(self, sa_h5_path):
        """Returned dict has all expected keys."""
        ds = TrajectoryDataset(sa_h5_path, context_length=10)
        sample = ds[0]
        expected_keys = {"states", "actions", "returns_to_go", "timesteps",
                         "attention_mask", "masks"}
        assert set(sample.keys()) == expected_keys

    def test_getitem_shapes(self, sa_h5_path):
        """All tensors have correct shapes matching context_length."""
        K = 10
        ds = TrajectoryDataset(sa_h5_path, context_length=K)
        sample = ds[0]

        assert sample["states"].shape == (K, 8)
        assert sample["actions"].shape == (K,)
        assert sample["returns_to_go"].shape == (K, 1)
        assert sample["timesteps"].shape == (K,)
        assert sample["attention_mask"].shape == (K,)
        assert sample["masks"].shape == (K,)

    def test_getitem_dtypes(self, sa_h5_path):
        """Verify tensor dtypes are correct."""
        ds = TrajectoryDataset(sa_h5_path, context_length=10)
        sample = ds[0]

        assert sample["states"].dtype == torch.float32
        assert sample["actions"].dtype == torch.int64
        assert sample["returns_to_go"].dtype == torch.float32
        assert sample["timesteps"].dtype == torch.int64
        assert sample["attention_mask"].dtype == torch.float32

    def test_masks_alias(self, sa_h5_path):
        """masks should be identical to attention_mask (alias)."""
        ds = TrajectoryDataset(sa_h5_path, context_length=10)
        sample = ds[5]
        assert torch.equal(sample["masks"], sample["attention_mask"])

    def test_getitem_last_index(self, sa_h5_path):
        """Accessing the last valid index works without error."""
        ds = TrajectoryDataset(sa_h5_path, context_length=10)
        sample = ds[len(ds) - 1]
        assert sample["states"].shape[0] == 10

    def test_getitem_out_of_range(self, sa_h5_path):
        """Accessing an out-of-range index raises IndexError."""
        ds = TrajectoryDataset(sa_h5_path, context_length=10)
        with pytest.raises(IndexError):
            ds[len(ds)]

    def test_getitem_various_indices(self, sa_h5_path):
        """Multiple indices all return correctly shaped data."""
        ds = TrajectoryDataset(sa_h5_path, context_length=10)
        for idx in [0, 5, 19, 20, 34, len(ds) - 1]:
            sample = ds[idx]
            assert sample["states"].shape == (10, 8)


class TestReturnsToGoComputation:
    """Verify returns-to-go computation correctness."""

    def test_rtg_undiscounted(self, tmp_path):
        """With discount=1.0, RTG[t] = sum(rewards[t:])."""
        # Create a simple episode with known rewards
        path = str(tmp_path / "rtg_test.h5")
        rewards = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        expected_rtg = np.array([15.0, 14.0, 12.0, 9.0, 5.0], dtype=np.float32)

        with h5py.File(path, "w") as f:
            grp = f.create_group("episode_0")
            grp.create_dataset("states", data=np.zeros((5, 4), dtype=np.float32))
            grp.create_dataset("actions", data=np.zeros(5, dtype=np.int64))
            grp.create_dataset("rewards", data=rewards)
            grp.create_dataset("dones", data=np.array([0, 0, 0, 0, 1], dtype=bool))

        ds = TrajectoryDataset(
            path, context_length=5, discount=1.0,
            normalize_states=False, normalize_returns=False,
        )
        np.testing.assert_allclose(
            ds.episodes[0]["returns_to_go"], expected_rtg, atol=1e-5
        )

    def test_rtg_discounted(self, tmp_path):
        """With discount=0.9, RTG[t] = r[t] + 0.9*r[t+1] + 0.81*r[t+2] + ..."""
        path = str(tmp_path / "rtg_disc.h5")
        rewards = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        gamma = 0.9
        # RTG[2] = 1.0
        # RTG[1] = 1.0 + 0.9*1.0 = 1.9
        # RTG[0] = 1.0 + 0.9*1.9 = 2.71
        expected_rtg = np.array([2.71, 1.9, 1.0], dtype=np.float32)

        with h5py.File(path, "w") as f:
            grp = f.create_group("episode_0")
            grp.create_dataset("states", data=np.zeros((3, 4), dtype=np.float32))
            grp.create_dataset("actions", data=np.zeros(3, dtype=np.int64))
            grp.create_dataset("rewards", data=rewards)
            grp.create_dataset("dones", data=np.array([0, 0, 1], dtype=bool))

        ds = TrajectoryDataset(
            path, context_length=3, discount=gamma,
            normalize_states=False, normalize_returns=False,
        )
        np.testing.assert_allclose(
            ds.episodes[0]["returns_to_go"], expected_rtg, atol=1e-5
        )

    def test_rtg_single_step(self, tmp_path):
        """Single-step episode: RTG equals the reward."""
        path = str(tmp_path / "rtg_one.h5")
        with h5py.File(path, "w") as f:
            grp = f.create_group("episode_0")
            grp.create_dataset("states", data=np.ones((1, 2), dtype=np.float32))
            grp.create_dataset("actions", data=np.array([1], dtype=np.int64))
            grp.create_dataset("rewards", data=np.array([7.0], dtype=np.float32))
            grp.create_dataset("dones", data=np.array([True]))

        ds = TrajectoryDataset(
            path, context_length=5, discount=1.0,
            normalize_states=False, normalize_returns=False,
        )
        assert ds.episodes[0]["returns_to_go"][0] == pytest.approx(7.0)

    def test_rtg_negative_rewards(self, tmp_path):
        """RTG works correctly with negative rewards."""
        path = str(tmp_path / "rtg_neg.h5")
        rewards = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
        expected_rtg = np.array([-6.0, -5.0, -3.0], dtype=np.float32)

        with h5py.File(path, "w") as f:
            grp = f.create_group("episode_0")
            grp.create_dataset("states", data=np.zeros((3, 2), dtype=np.float32))
            grp.create_dataset("actions", data=np.zeros(3, dtype=np.int64))
            grp.create_dataset("rewards", data=rewards)
            grp.create_dataset("dones", data=np.array([0, 0, 1], dtype=bool))

        ds = TrajectoryDataset(
            path, context_length=3, discount=1.0,
            normalize_states=False, normalize_returns=False,
        )
        np.testing.assert_allclose(
            ds.episodes[0]["returns_to_go"], expected_rtg, atol=1e-5
        )


class TestContextWindow:
    """Verify context windowing behavior."""

    def test_full_window(self, tmp_path):
        """When episode is long enough, the window is fully populated."""
        path = str(tmp_path / "ctx.h5")
        T = 20
        K = 5
        with h5py.File(path, "w") as f:
            grp = f.create_group("episode_0")
            grp.create_dataset("states", data=np.arange(T * 4).reshape(T, 4).astype(np.float32))
            grp.create_dataset("actions", data=np.arange(T, dtype=np.int64) % 4)
            grp.create_dataset("rewards", data=np.ones(T, dtype=np.float32))
            grp.create_dataset("dones", data=np.zeros(T, dtype=bool))

        ds = TrajectoryDataset(
            path, context_length=K, normalize_states=False, normalize_returns=False,
        )
        # Index K-1 (=4) means timestep 4; window should be [0..4], full K=5 tokens
        sample = ds[K - 1]
        assert sample["attention_mask"].sum().item() == K
        # Timesteps should be [0, 1, 2, 3, 4]
        assert torch.equal(sample["timesteps"], torch.arange(K, dtype=torch.int64))

    def test_window_at_start(self, tmp_path):
        """At the very first timestep the window has only 1 real token."""
        path = str(tmp_path / "ctx_start.h5")
        T = 10
        K = 5
        with h5py.File(path, "w") as f:
            grp = f.create_group("episode_0")
            grp.create_dataset("states", data=np.ones((T, 2), dtype=np.float32))
            grp.create_dataset("actions", data=np.zeros(T, dtype=np.int64))
            grp.create_dataset("rewards", data=np.ones(T, dtype=np.float32))
            grp.create_dataset("dones", data=np.zeros(T, dtype=bool))

        ds = TrajectoryDataset(
            path, context_length=K, normalize_states=False, normalize_returns=False,
        )
        sample = ds[0]
        # Only 1 real token (timestep 0), rest is padding
        assert sample["attention_mask"].sum().item() == 1
        # The real token is at the last position (left-padded)
        assert sample["attention_mask"][-1].item() == 1.0
        assert sample["attention_mask"][0].item() == 0.0

    def test_window_slides(self, tmp_path):
        """Consecutive indices produce sliding windows."""
        path = str(tmp_path / "ctx_slide.h5")
        T = 10
        K = 3
        with h5py.File(path, "w") as f:
            grp = f.create_group("episode_0")
            states = np.arange(T * 2).reshape(T, 2).astype(np.float32)
            grp.create_dataset("states", data=states)
            grp.create_dataset("actions", data=np.arange(T, dtype=np.int64) % 4)
            grp.create_dataset("rewards", data=np.ones(T, dtype=np.float32))
            grp.create_dataset("dones", data=np.zeros(T, dtype=bool))

        ds = TrajectoryDataset(
            path, context_length=K, normalize_states=False, normalize_returns=False,
        )
        # At idx=4 (timestep 4), window covers timesteps [2,3,4]
        s4 = ds[4]
        assert torch.equal(s4["timesteps"], torch.tensor([2, 3, 4], dtype=torch.int64))
        # At idx=5 (timestep 5), window covers timesteps [3,4,5]
        s5 = ds[5]
        assert torch.equal(s5["timesteps"], torch.tensor([3, 4, 5], dtype=torch.int64))


class TestShortEpisodes:
    """Test episodes shorter than context_length are handled with padding."""

    def test_short_episode_padding(self, sa_h5_path_short):
        """Short episodes produce left-padded samples."""
        K = 10  # context_length > episode lengths (3 and 5)
        ds = TrajectoryDataset(sa_h5_path_short, context_length=K)

        # First episode has length 3; last timestep at idx=2
        sample = ds[2]
        # At most 3 real tokens
        assert sample["attention_mask"].sum().item() <= 3
        # Padding zeros at start
        assert sample["attention_mask"][0].item() == 0.0

    def test_short_episode_shapes(self, sa_h5_path_short):
        """Even with short episodes, shapes match context_length."""
        K = 10
        ds = TrajectoryDataset(sa_h5_path_short, context_length=K)
        sample = ds[0]
        assert sample["states"].shape == (K, 8)
        assert sample["actions"].shape == (K,)
        assert sample["returns_to_go"].shape == (K, 1)

    def test_padding_values_are_zero(self, sa_h5_path_short):
        """Padded positions should have zero states, actions, RTG, and timesteps."""
        K = 10
        ds = TrajectoryDataset(
            sa_h5_path_short, context_length=K,
            normalize_states=False, normalize_returns=False,
        )
        sample = ds[0]
        # Index 0 has 1 real token at position K-1; positions 0..K-2 are padding
        pad_len = K - 1
        assert torch.all(sample["states"][:pad_len] == 0)
        assert torch.all(sample["actions"][:pad_len] == 0)
        assert torch.all(sample["returns_to_go"][:pad_len] == 0)
        assert torch.all(sample["timesteps"][:pad_len] == 0)


class TestSingleEpisode:
    """Test dataset with a single episode."""

    def test_single_episode_len(self, sa_h5_path_single):
        """Length equals the single episode's length."""
        ds = TrajectoryDataset(sa_h5_path_single, context_length=5)
        assert len(ds) == 10

    def test_single_episode_all_indices(self, sa_h5_path_single):
        """Every index in a single-episode dataset is valid."""
        ds = TrajectoryDataset(sa_h5_path_single, context_length=5)
        for i in range(len(ds)):
            sample = ds[i]
            assert sample["states"].shape[0] == 5

    def test_single_episode_episodes_list(self, sa_h5_path_single):
        """Only one episode is stored."""
        ds = TrajectoryDataset(sa_h5_path_single, context_length=5)
        assert len(ds.episodes) == 1


class TestNormalization:
    """Test state and return normalization."""

    def test_state_stats_shape(self, sa_h5_path):
        """State mean and std have correct shape [state_dim]."""
        ds = TrajectoryDataset(sa_h5_path, context_length=10)
        mean, std = ds.get_state_stats()
        assert mean.shape == (8,)
        assert std.shape == (8,)

    def test_state_std_positive(self, sa_h5_path):
        """State std is strictly positive (includes epsilon guard)."""
        ds = TrajectoryDataset(sa_h5_path, context_length=10)
        _, std = ds.get_state_stats()
        assert np.all(std > 0)

    def test_normalized_states_reasonable(self, sa_h5_path):
        """Normalized states should have roughly zero mean across the dataset."""
        ds = TrajectoryDataset(sa_h5_path, context_length=10, normalize_states=True)
        # Collect all states from full-window samples (skip padding)
        all_states = []
        for ep in ds.episodes:
            normalized = (ep["states"] - ds.state_mean) / ds.state_std
            all_states.append(normalized)
        all_states = np.concatenate(all_states, axis=0)
        mean_abs = np.abs(all_states.mean(axis=0))
        assert np.all(mean_abs < 0.5)  # roughly centered

    def test_return_scale_positive(self, sa_h5_path):
        """Return scale is strictly positive."""
        ds = TrajectoryDataset(sa_h5_path, context_length=10)
        assert ds.get_return_scale() > 0

    def test_normalization_disabled(self, tmp_path):
        """When normalization is disabled, raw values are returned."""
        path = str(tmp_path / "no_norm.h5")
        states = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)
        with h5py.File(path, "w") as f:
            grp = f.create_group("episode_0")
            grp.create_dataset("states", data=states)
            grp.create_dataset("actions", data=np.array([0, 1, 2], dtype=np.int64))
            grp.create_dataset("rewards", data=np.array([1.0, 2.0, 3.0], dtype=np.float32))
            grp.create_dataset("dones", data=np.array([0, 0, 1], dtype=bool))

        ds = TrajectoryDataset(
            path, context_length=3,
            normalize_states=False, normalize_returns=False,
        )
        sample = ds[2]  # last timestep, full window [0,1,2]
        # States should match raw values exactly
        np.testing.assert_allclose(
            sample["states"].numpy(), states, atol=1e-6
        )

    def test_normalization_enabled_differs(self, tmp_path):
        """Normalized states differ from raw states."""
        path = str(tmp_path / "norm_on.h5")
        states = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32)
        with h5py.File(path, "w") as f:
            grp = f.create_group("episode_0")
            grp.create_dataset("states", data=states)
            grp.create_dataset("actions", data=np.array([0, 1, 2], dtype=np.int64))
            grp.create_dataset("rewards", data=np.array([1.0, 2.0, 3.0], dtype=np.float32))
            grp.create_dataset("dones", data=np.array([0, 0, 1], dtype=bool))

        ds = TrajectoryDataset(
            path, context_length=3,
            normalize_states=True, normalize_returns=False,
        )
        sample = ds[2]
        # Normalized states should NOT equal raw states
        assert not np.allclose(sample["states"].numpy(), states)


class TestMultiDimActions:
    """Test handling of multi-dimensional actions (reduced to first element)."""

    def test_multidim_actions_reduced(self, tmp_path):
        """Multi-dim actions are reduced to first column."""
        path = str(tmp_path / "multidim.h5")
        T = 5
        with h5py.File(path, "w") as f:
            grp = f.create_group("episode_0")
            grp.create_dataset("states", data=np.zeros((T, 3), dtype=np.float32))
            # 2D actions: [T, 2]
            actions = np.array([[0, 1], [2, 3], [1, 0], [3, 2], [0, 1]], dtype=np.int64)
            grp.create_dataset("actions", data=actions)
            grp.create_dataset("rewards", data=np.ones(T, dtype=np.float32))
            grp.create_dataset("dones", data=np.zeros(T, dtype=bool))

        ds = TrajectoryDataset(path, context_length=5, normalize_states=False)
        # Should have used only the first column
        np.testing.assert_array_equal(
            ds.episodes[0]["actions"], actions[:, 0]
        )


# ======================================================================
# MultiAgentTrajectoryDataset tests
# ======================================================================


class TestMAConstruction:
    """Test MultiAgentTrajectoryDataset construction."""

    def test_ma_construction_len(self, ma_h5_path):
        """Dataset length equals total timesteps across all episodes."""
        ds = MultiAgentTrajectoryDataset(ma_h5_path, n_agents=3, context_length=5)
        assert len(ds) == 15 + 20

    def test_ma_num_episodes(self, ma_h5_path):
        """All episodes are loaded."""
        ds = MultiAgentTrajectoryDataset(ma_h5_path, n_agents=3, context_length=5)
        assert len(ds.episodes) == 2

    def test_ma_episode_shapes(self, ma_h5_path):
        """Episode data arrays have correct multi-agent shapes."""
        ds = MultiAgentTrajectoryDataset(ma_h5_path, n_agents=3, context_length=5)
        ep = ds.episodes[0]
        assert ep["states"].shape == (15, 3, 6)
        assert ep["actions"].shape == (15, 3)
        assert ep["rewards"].shape == (15, 3)
        assert ep["returns_to_go"].shape == (15, 3)

    def test_ma_file_not_found(self, tmp_path):
        """Raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            MultiAgentTrajectoryDataset(
                str(tmp_path / "missing.h5"), n_agents=3
            )

    def test_ma_empty_h5(self, tmp_path):
        """Raises ValueError for empty HDF5."""
        path = str(tmp_path / "empty_ma.h5")
        with h5py.File(path, "w") as f:
            pass
        with pytest.raises(ValueError, match="No episodes found"):
            MultiAgentTrajectoryDataset(path, n_agents=3)


class TestMAGetItem:
    """Test MultiAgentTrajectoryDataset __getitem__."""

    def test_ma_getitem_keys(self, ma_h5_path):
        """Returned dict has expected keys."""
        ds = MultiAgentTrajectoryDataset(ma_h5_path, n_agents=3, context_length=5)
        sample = ds[0]
        expected_keys = {"states", "actions", "returns_to_go", "timesteps", "attention_mask"}
        assert set(sample.keys()) == expected_keys

    def test_ma_getitem_shapes(self, ma_h5_path):
        """All tensors have correct [n_agents, K, ...] shapes."""
        K = 5
        N = 3
        ds = MultiAgentTrajectoryDataset(ma_h5_path, n_agents=N, context_length=K)
        sample = ds[10]

        assert sample["states"].shape == (N, K, 6)
        assert sample["actions"].shape == (N, K)
        assert sample["returns_to_go"].shape == (N, K, 1)
        assert sample["timesteps"].shape == (N, K)
        assert sample["attention_mask"].shape == (K,)

    def test_ma_getitem_dtypes(self, ma_h5_path):
        """Verify multi-agent tensor dtypes."""
        ds = MultiAgentTrajectoryDataset(ma_h5_path, n_agents=3, context_length=5)
        sample = ds[0]
        assert sample["states"].dtype == torch.float32
        assert sample["actions"].dtype == torch.int64
        assert sample["returns_to_go"].dtype == torch.float32
        assert sample["timesteps"].dtype == torch.int64
        assert sample["attention_mask"].dtype == torch.float32

    def test_ma_getitem_all_indices(self, ma_h5_path):
        """All valid indices produce correctly shaped samples."""
        K = 5
        N = 3
        ds = MultiAgentTrajectoryDataset(ma_h5_path, n_agents=N, context_length=K)
        for i in range(len(ds)):
            sample = ds[i]
            assert sample["states"].shape == (N, K, 6)

    def test_ma_getitem_out_of_range(self, ma_h5_path):
        """Out-of-range index raises IndexError."""
        ds = MultiAgentTrajectoryDataset(ma_h5_path, n_agents=3, context_length=5)
        with pytest.raises(IndexError):
            ds[len(ds)]

    def test_ma_attention_mask_shared(self, ma_h5_path):
        """Attention mask is shared across agents (1D, not per-agent)."""
        ds = MultiAgentTrajectoryDataset(ma_h5_path, n_agents=3, context_length=5)
        sample = ds[0]
        assert sample["attention_mask"].dim() == 1

    def test_ma_padding_at_start(self, ma_h5_path):
        """First timestep in an episode has left-padding."""
        K = 10
        ds = MultiAgentTrajectoryDataset(ma_h5_path, n_agents=3, context_length=K)
        sample = ds[0]
        # Only 1 real token at idx 0
        assert sample["attention_mask"].sum().item() == 1
        assert sample["attention_mask"][-1].item() == 1.0


class TestMAReturnsToGo:
    """Test per-agent RTG computation in MultiAgentTrajectoryDataset."""

    def test_ma_rtg_per_agent(self, tmp_path):
        """Each agent's RTG is computed independently."""
        path = str(tmp_path / "ma_rtg.h5")
        N = 2
        T = 4
        rewards = np.array([
            [1.0, 2.0],  # t=0
            [3.0, 4.0],  # t=1
            [5.0, 6.0],  # t=2
            [7.0, 8.0],  # t=3
        ], dtype=np.float32)

        with h5py.File(path, "w") as f:
            grp = f.create_group("episode_0")
            grp.create_dataset("states", data=np.zeros((T, N, 3), dtype=np.float32))
            grp.create_dataset("actions", data=np.zeros((T, N), dtype=np.int64))
            grp.create_dataset("rewards", data=rewards)
            grp.create_dataset("dones", data=np.array([0, 0, 0, 1], dtype=bool))

        ds = MultiAgentTrajectoryDataset(
            path, n_agents=N, context_length=4, discount=1.0,
            normalize_states=False, normalize_returns=False,
        )
        rtg = ds.episodes[0]["returns_to_go"]

        # Agent 0: [1+3+5+7, 3+5+7, 5+7, 7] = [16, 15, 12, 7]
        np.testing.assert_allclose(rtg[:, 0], [16.0, 15.0, 12.0, 7.0], atol=1e-5)
        # Agent 1: [2+4+6+8, 4+6+8, 6+8, 8] = [20, 18, 14, 8]
        np.testing.assert_allclose(rtg[:, 1], [20.0, 18.0, 14.0, 8.0], atol=1e-5)

    def test_ma_rtg_discounted(self, tmp_path):
        """Discounted per-agent RTG with gamma < 1."""
        path = str(tmp_path / "ma_rtg_disc.h5")
        N = 2
        T = 3
        gamma = 0.5
        rewards = np.array([
            [1.0, 2.0],
            [1.0, 2.0],
            [1.0, 2.0],
        ], dtype=np.float32)

        with h5py.File(path, "w") as f:
            grp = f.create_group("episode_0")
            grp.create_dataset("states", data=np.zeros((T, N, 2), dtype=np.float32))
            grp.create_dataset("actions", data=np.zeros((T, N), dtype=np.int64))
            grp.create_dataset("rewards", data=rewards)
            grp.create_dataset("dones", data=np.array([0, 0, 1], dtype=bool))

        ds = MultiAgentTrajectoryDataset(
            path, n_agents=N, context_length=3, discount=gamma,
            normalize_states=False, normalize_returns=False,
        )
        rtg = ds.episodes[0]["returns_to_go"]

        # Agent 0: RTG[2]=1, RTG[1]=1+0.5*1=1.5, RTG[0]=1+0.5*1.5=1.75
        np.testing.assert_allclose(rtg[:, 0], [1.75, 1.5, 1.0], atol=1e-5)
        # Agent 1: RTG[2]=2, RTG[1]=2+0.5*2=3.0, RTG[0]=2+0.5*3.0=3.5
        np.testing.assert_allclose(rtg[:, 1], [3.5, 3.0, 2.0], atol=1e-5)


class TestMANormalization:
    """Test normalization in MultiAgentTrajectoryDataset."""

    def test_ma_state_stats_shape(self, ma_h5_path):
        """State stats are per-feature (flattened across agents)."""
        ds = MultiAgentTrajectoryDataset(ma_h5_path, n_agents=3, context_length=5)
        mean, std = ds.get_state_stats()
        assert mean.shape == (6,)  # state_dim
        assert std.shape == (6,)

    def test_ma_state_std_positive(self, ma_h5_path):
        """State std is strictly positive."""
        ds = MultiAgentTrajectoryDataset(ma_h5_path, n_agents=3, context_length=5)
        _, std = ds.get_state_stats()
        assert np.all(std > 0)

    def test_ma_return_scale_positive(self, ma_h5_path):
        """Return scale is strictly positive."""
        ds = MultiAgentTrajectoryDataset(ma_h5_path, n_agents=3, context_length=5)
        assert ds.get_return_scale() > 0
