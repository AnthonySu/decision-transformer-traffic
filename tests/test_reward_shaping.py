"""Tests for src.envs.reward_shaping — configurable reward functions."""


from src.envs.reward_shaping import REWARD_PRESETS, RewardFunction


class TestRewardFunction:
    def test_default_init(self):
        rf = RewardFunction()
        assert "time_penalty" in rf.weights

    def test_compute_returns_tuple(self):
        rf = RewardFunction()
        state = {
            "ev_passed_intersection": False,
            "ev_arrived": False,
            "timeout": False,
            "ev_blocked": False,
            "total_queue": 100.0,
        }
        total, components = rf.compute(state)
        assert isinstance(total, float)
        assert isinstance(components, dict)

    def test_arrival_bonus(self):
        rf = RewardFunction()
        state = {
            "ev_passed_intersection": False,
            "ev_arrived": True,
            "timeout": False,
            "ev_blocked": False,
            "total_queue": 50.0,
        }
        total, components = rf.compute(state)
        assert total > 0  # arrival bonus should dominate

    def test_timeout_penalty(self):
        rf = RewardFunction()
        state = {
            "ev_passed_intersection": False,
            "ev_arrived": False,
            "timeout": True,
            "ev_blocked": False,
            "total_queue": 50.0,
        }
        total, _ = rf.compute(state)
        assert total < 0  # timeout should give negative reward

    def test_custom_weights(self):
        rf = RewardFunction(weights={"time_penalty": -2.0})
        assert rf.weights["time_penalty"] == -2.0


class TestRewardPresets:
    def test_all_presets_exist(self):
        assert "default" in REWARD_PRESETS
        assert "ev_priority" in REWARD_PRESETS
        assert "balanced" in REWARD_PRESETS
        assert "minimal_disruption" in REWARD_PRESETS

    def test_from_preset(self):
        rf = RewardFunction.from_preset("ev_priority")
        assert isinstance(rf, RewardFunction)

    def test_ev_priority_has_higher_ev_weight(self):
        default = RewardFunction.from_preset("default")
        priority = RewardFunction.from_preset("ev_priority")
        # ev_priority should weight intersection bonus more
        assert abs(priority.weights.get("intersection_bonus", 0)) >= abs(
            default.weights.get("intersection_bonus", 0)
        )
