"""
Unit tests for SFL configuration module.
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

from sfl.utils.config import load_config, reset_config, _get_env, _merge_dict
from sfl.types import SFLConfig


class TestConfigHelpers:
    """Tests for configuration helper functions."""
    
    def test_get_env_default(self):
        """Test _get_env returns default when not set."""
        result = _get_env("NONEXISTENT_KEY_12345", "default_value")
        assert result == "default_value"
    
    def test_get_env_with_value(self, monkeypatch):
        """Test _get_env returns env value when set."""
        monkeypatch.setenv("SFL_TEST_KEY", "test_value")
        result = _get_env("TEST_KEY", "default")
        assert result == "test_value"
    
    def test_get_env_bool_true(self, monkeypatch):
        """Test _get_env handles boolean true values."""
        for true_val in ["true", "1", "yes", "on", "TRUE"]:
            monkeypatch.setenv("SFL_BOOL_TEST", true_val)
            result = _get_env("BOOL_TEST", False, bool)
            assert result is True
    
    def test_get_env_bool_false(self, monkeypatch):
        """Test _get_env handles boolean false values."""
        monkeypatch.setenv("SFL_BOOL_TEST", "false")
        result = _get_env("BOOL_TEST", True, bool)
        assert result is False
    
    def test_get_env_int(self, monkeypatch):
        """Test _get_env handles int casting."""
        monkeypatch.setenv("SFL_INT_TEST", "42")
        result = _get_env("INT_TEST", 0, int)
        assert result == 42
    
    def test_merge_dict_simple(self):
        """Test _merge_dict with simple values."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _merge_dict(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}
    
    def test_merge_dict_nested(self):
        """Test _merge_dict with nested dicts."""
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 10, "z": 20}}
        result = _merge_dict(base, override)
        assert result == {"a": {"x": 1, "y": 10, "z": 20}, "b": 3}


class TestLoadConfig:
    """Tests for load_config function."""
    
    def setup_method(self):
        """Reset config before each test."""
        reset_config()
    
    def test_load_defaults(self):
        """Test loading with all defaults."""
        config = load_config()
        
        assert isinstance(config, SFLConfig)
        assert config.federation.num_clients == 2
        assert config.federation.num_rounds == 1
        assert config.client.base_secret == 7.0
    
    def test_load_from_yaml(self):
        """Test loading from YAML file."""
        yaml_content = {
            "federation": {
                "num_clients": 5,
                "num_rounds": 10,
            },
            "client": {
                "base_secret": 100.0,
            },
        }
        
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(yaml_content, f)
            config_path = f.name
        
        try:
            config = load_config(config_path=config_path)
            assert config.federation.num_clients == 5
            assert config.federation.num_rounds == 10
            assert config.client.base_secret == 100.0
        finally:
            Path(config_path).unlink()
    
    def test_cli_overrides(self):
        """Test CLI overrides take precedence."""
        cli_overrides = {
            "federation": {"num_clients": 8},
        }
        
        config = load_config(cli_overrides=cli_overrides)
        assert config.federation.num_clients == 8
    
    def test_env_overrides(self, monkeypatch):
        """Test environment variables override defaults."""
        monkeypatch.setenv("SFL_NUM_CLIENTS", "10")
        monkeypatch.setenv("SFL_NUM_ROUNDS", "5")
        
        reset_config()
        config = load_config()
        
        assert config.federation.num_clients == 10
        assert config.federation.num_rounds == 5


class TestResetConfig:
    """Tests for reset_config function."""
    
    def test_reset_clears_cache(self):
        """Test reset_config clears the config cache."""
        # Load config
        config1 = load_config(cli_overrides={"federation": {"num_clients": 99}})
        assert config1.federation.num_clients == 99
        
        # Reset and reload
        reset_config()
        config2 = load_config()
        assert config2.federation.num_clients == 2  # Back to default
