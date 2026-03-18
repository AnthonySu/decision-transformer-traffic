"""Tests for src.utils.config_utils — configuration management."""



from src.utils.config_utils import (
    config_to_flat,
    load_config,
    merge_configs,
)


class TestLoadConfig:
    def test_loads_yaml(self):
        cfg = load_config("configs/default.yaml")
        assert "env" in cfg
        assert "dt" in cfg

    def test_loads_smoke_config(self):
        cfg = load_config("configs/smoke_test.yaml")
        assert cfg["env"]["rows"] == 3


class TestMergeConfigs:
    def test_simple_override(self):
        base = {"a": 1, "b": 2}
        override = {"b": 3}
        merged = merge_configs(base, override)
        assert merged["a"] == 1
        assert merged["b"] == 3

    def test_nested_merge(self):
        base = {"env": {"rows": 4, "cols": 4}, "dt": {"lr": 0.001}}
        override = {"env": {"rows": 6}}
        merged = merge_configs(base, override)
        assert merged["env"]["rows"] == 6
        assert merged["env"]["cols"] == 4
        assert merged["dt"]["lr"] == 0.001

    def test_base_unchanged(self):
        base = {"a": {"b": 1}}
        override = {"a": {"b": 2}}
        merge_configs(base, override)
        assert base["a"]["b"] == 1  # original not mutated


class TestConfigToFlat:
    def test_simple_flatten(self):
        cfg = {"a": 1, "b": {"c": 2, "d": 3}}
        flat = config_to_flat(cfg)
        assert flat["a"] == 1
        assert flat["b.c"] == 2
        assert flat["b.d"] == 3

    def test_deep_nesting(self):
        cfg = {"a": {"b": {"c": {"d": 4}}}}
        flat = config_to_flat(cfg)
        assert flat["a.b.c.d"] == 4

    def test_prefix(self):
        cfg = {"x": 1}
        flat = config_to_flat(cfg, prefix="cfg")
        assert flat["cfg.x"] == 1
