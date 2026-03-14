"""Unit tests for SFL client module — FederationConfig validation."""

import pytest

from sfl.types import FederationConfig


class TestFederationConfig:
    """Tests for FederationConfig validation."""
    
    def test_valid_config(self):
        """Test valid configuration passes."""
        config = FederationConfig(
            num_clients=4,
            num_rounds=10,
            min_available_clients=2,
        )
        assert config.num_clients == 4
        assert config.num_rounds == 10
    
    def test_invalid_num_clients(self):
        """Test invalid num_clients raises ValueError."""
        with pytest.raises(ValueError, match="num_clients"):
            FederationConfig(num_clients=0)
    
    def test_invalid_num_rounds(self):
        """Test invalid num_rounds raises ValueError."""
        with pytest.raises(ValueError, match="num_rounds"):
            FederationConfig(num_rounds=0)
    
    def test_min_available_exceeds_total(self):
        """Test min_available_clients > num_clients raises ValueError."""
        with pytest.raises(ValueError, match="min_available_clients"):
            FederationConfig(num_clients=2, min_available_clients=5)



