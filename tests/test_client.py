"""
Unit tests for SFL client module.

These tests verify the functionality of the federated client
implementations without requiring the full NVFlare/Flower stack.
"""

import numpy as np
import pytest

from sfl.client.sum_client import SumClient
from sfl.types import FederationConfig


class TestSumClient:
    """Tests for the SumClient class."""
    
    def test_compute_update(self):
        """Test compute_update returns correct values."""
        client = SumClient(client_id=1, secret=7.5)
        
        params, num_examples, metrics = client.compute_update([], {})
        
        # Check parameters
        assert len(params) == 1
        assert params[0].dtype == np.float32
        assert float(params[0].item()) == 7.5
        
        # Check num_examples
        assert num_examples == 1
        
        # Check metrics
        assert metrics["client_id"] == 1
        assert metrics["client_secret"] == 7.5
    
    def test_fit(self):
        """Test fit method delegates to compute_update."""
        client = SumClient(client_id=2, secret=10.0)
        
        params, num_examples, metrics = client.fit([], {})
        
        assert float(params[0].item()) == 10.0
        assert metrics["client_secret"] == 10.0
    
    def test_evaluate(self):
        """Test evaluate returns defaults."""
        client = SumClient(client_id=0, secret=5.0)
        
        loss, num_examples, metrics = client.evaluate([], {})
        
        assert loss == 0.0
        assert num_examples == 1
        assert metrics == {}
    
    def test_get_initial_parameters(self):
        """Test initial parameters are correct."""
        client = SumClient()
        
        params = client.get_initial_parameters()
        
        assert len(params) == 1
        assert params[0].dtype == np.float32
        assert float(params[0].item()) == 0.0


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



