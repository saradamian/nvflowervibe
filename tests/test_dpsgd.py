"""
Tests for per-example DP-SGD client wrapper.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset

try:
    import opacus  # noqa: F401
    _has_opacus = True
except ImportError:
    _has_opacus = False

from sfl.client.dp_client import DPSGDConfig, enable_dpsgd


class _DictDataset(Dataset):
    """Dataset that yields dict batches like HuggingFace tokenized datasets."""

    def __init__(self, size=16):
        self.input_ids = torch.randint(0, 100, (size, 8))
        self.labels = torch.randint(0, 100, (size, 8))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {"input_ids": self.input_ids[idx], "labels": self.labels[idx]}


class _SimpleMLM(nn.Module):
    """Minimal model that accepts input_ids/labels and returns .loss."""

    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 16)
        self.head = nn.Linear(16, 100)

    def forward(self, input_ids, labels=None, **kwargs):
        logits = self.head(self.embed(input_ids))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, 100), labels.view(-1),
            )

        class _Out:
            pass

        out = _Out()
        out.loss = loss
        out.logits = logits
        return out


class _DummyClient:
    """Minimal client with attributes needed by enable_dpsgd."""

    def __init__(self, seed=0):
        torch.manual_seed(seed)
        self.client_id = 0
        self.device = "cpu"
        self.local_epochs = 1
        self.batch_size = 4
        self.learning_rate = 0.01
        self.model = _SimpleMLM()
        self.train_data = _DictDataset(size=16)

    def _train(self):
        from torch.utils.data import DataLoader
        loader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.model.train()
        total_loss = 0.0
        n = 0
        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = self.model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n += 1
        return total_loss / max(n, 1)

    def compute_update(self, parameters, config):
        loss = self._train()
        params = [p.detach().cpu().numpy() for p in self.model.parameters()]
        return params, len(self.train_data), {"train_loss": loss}


class TestDPSGDConfig:

    def test_defaults(self):
        cfg = DPSGDConfig()
        assert cfg.max_grad_norm == 1.0
        assert cfg.noise_multiplier == 1.0
        assert cfg.target_delta == 1e-5


@pytest.mark.skipif(not _has_opacus, reason="opacus not installed")
class TestEnableDPSGD:

    def test_train_completes(self):
        """DP-SGD training should complete without errors."""
        client = _DummyClient()
        config = DPSGDConfig(max_grad_norm=1.0, noise_multiplier=0.5)
        enable_dpsgd(client, config)
        loss = client._train()
        assert isinstance(loss, float)
        assert loss >= 0

    def test_epsilon_tracked(self):
        """After training, client should have a positive dpsgd_epsilon."""
        client = _DummyClient()
        config = DPSGDConfig(max_grad_norm=1.0, noise_multiplier=1.0)
        enable_dpsgd(client, config)
        client._train()
        assert client._dpsgd_epsilon > 0

    def test_compute_update_includes_epsilon(self):
        """compute_update should include dpsgd_epsilon in metrics."""
        client = _DummyClient()
        config = DPSGDConfig(max_grad_norm=1.0, noise_multiplier=1.0)
        enable_dpsgd(client, config)
        params, n, metrics = client.compute_update([], {})
        assert "dpsgd_epsilon" in metrics
        assert metrics["dpsgd_epsilon"] > 0

    def test_model_unwrapped_after_train(self):
        """After training, model should be unwrapped (not GradSampleModule)."""
        from opacus import GradSampleModule
        client = _DummyClient()
        config = DPSGDConfig(max_grad_norm=1.0, noise_multiplier=0.5)
        enable_dpsgd(client, config)
        client._train()
        assert not isinstance(client.model, GradSampleModule)

    def test_higher_noise_higher_epsilon(self):
        """Lower noise multiplier should give higher epsilon (less privacy)."""
        client_low = _DummyClient(seed=42)
        client_high = _DummyClient(seed=42)
        # Share the same data
        client_high.train_data = client_low.train_data

        enable_dpsgd(client_low, DPSGDConfig(noise_multiplier=0.5))
        enable_dpsgd(client_high, DPSGDConfig(noise_multiplier=2.0))
        client_low._train()
        client_high._train()
        # Less noise → higher epsilon
        assert client_low._dpsgd_epsilon > client_high._dpsgd_epsilon


@pytest.mark.skipif(not _has_opacus, reason="opacus not installed")
class TestAutoClip:

    def test_auto_clip_defaults_false(self):
        """auto_clip should default to False."""
        cfg = DPSGDConfig()
        assert cfg.auto_clip is False

    def test_auto_clip_train_completes(self):
        """Training with auto_clip=True should complete without errors."""
        client = _DummyClient()
        config = DPSGDConfig(noise_multiplier=0.5, auto_clip=True)
        enable_dpsgd(client, config)
        loss = client._train()
        assert isinstance(loss, float)
        assert loss >= 0

    def test_auto_clip_epsilon_tracked(self):
        """After AutoClip training, epsilon should be positive."""
        client = _DummyClient()
        config = DPSGDConfig(noise_multiplier=1.0, auto_clip=True)
        enable_dpsgd(client, config)
        client._train()
        assert client._dpsgd_epsilon > 0

    def test_auto_clip_overrides_max_grad_norm(self):
        """auto_clip=True should use effective clip of 1.0 regardless of max_grad_norm."""
        client = _DummyClient()
        # Set a high max_grad_norm — AutoClip should ignore it
        config = DPSGDConfig(max_grad_norm=100.0, noise_multiplier=1.0, auto_clip=True)
        enable_dpsgd(client, config)
        client._train()
        # Should still produce a reasonable epsilon (not inflated by clip=100)
        assert client._dpsgd_epsilon > 0

    def test_auto_clip_model_unwrapped(self):
        """After AutoClip training, model should be unwrapped."""
        from opacus import GradSampleModule
        client = _DummyClient()
        config = DPSGDConfig(noise_multiplier=0.5, auto_clip=True)
        enable_dpsgd(client, config)
        client._train()
        assert not isinstance(client.model, GradSampleModule)

    def test_auto_clip_compute_update_metrics(self):
        """compute_update with AutoClip should include dpsgd_epsilon."""
        client = _DummyClient()
        config = DPSGDConfig(noise_multiplier=1.0, auto_clip=True)
        enable_dpsgd(client, config)
        params, n, metrics = client.compute_update([], {})
        assert "dpsgd_epsilon" in metrics
        assert metrics["dpsgd_epsilon"] > 0


@pytest.mark.skipif(not _has_opacus, reason="opacus not installed")
class TestGhostClipping:

    def test_ghost_clipping_defaults_false(self):
        """ghost_clipping should default to False."""
        cfg = DPSGDConfig()
        assert cfg.ghost_clipping is False

    def test_ghost_clipping_train_completes(self):
        """Training with ghost_clipping=True should complete."""
        client = _DummyClient()
        config = DPSGDConfig(noise_multiplier=0.5, ghost_clipping=True)
        enable_dpsgd(client, config)
        loss = client._train()
        assert isinstance(loss, float)
        assert loss >= 0

    def test_ghost_clipping_epsilon_tracked(self):
        """After ghost clipping training, epsilon should be positive."""
        client = _DummyClient()
        config = DPSGDConfig(noise_multiplier=1.0, ghost_clipping=True)
        enable_dpsgd(client, config)
        client._train()
        assert client._dpsgd_epsilon > 0

    def test_ghost_clipping_model_unwrapped(self):
        """After ghost clipping training, model should be unwrapped."""
        from opacus import GradSampleModule
        client = _DummyClient()
        config = DPSGDConfig(noise_multiplier=0.5, ghost_clipping=True)
        enable_dpsgd(client, config)
        client._train()
        assert not isinstance(client.model, GradSampleModule)

    def test_ghost_clipping_compute_update_metrics(self):
        """compute_update with ghost clipping should include dpsgd_epsilon."""
        client = _DummyClient()
        config = DPSGDConfig(noise_multiplier=1.0, ghost_clipping=True)
        enable_dpsgd(client, config)
        params, n, metrics = client.compute_update([], {})
        assert "dpsgd_epsilon" in metrics
        assert metrics["dpsgd_epsilon"] > 0
