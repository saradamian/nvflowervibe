"""
Per-example DP-SGD via Opacus for federated clients.

Wraps a client's local training with Opacus PrivacyEngine to provide
per-sample (ε,δ)-DP guarantees independent of server-side aggregate DP.

Requires: pip install opacus>=1.5  (or: pip install sfl[dpsgd])
"""

import types
from dataclasses import dataclass

from sfl.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DPSGDConfig:
    """Configuration for per-example DP-SGD training.

    Args:
        max_grad_norm: Per-example gradient clip norm.
        noise_multiplier: Gaussian noise multiplier for DP-SGD.
        target_delta: Delta parameter for (ε,δ)-DP accounting.
    """
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.0
    target_delta: float = 1e-5


def enable_dpsgd(client, config: DPSGDConfig):
    """Enable Opacus DP-SGD on a client with a _train() method.

    Replaces the client's ``_train`` method with one that wraps the
    model, optimizer, and dataloader with Opacus PrivacyEngine for
    per-example gradient clipping and noise. Also patches
    ``compute_update`` to include the per-round ε in fit metrics.

    Args:
        client: An ESM2Client (or any client with _train/compute_update).
        config: DP-SGD configuration.

    Returns:
        The same client, mutated in place.
    """
    import torch
    from opacus import PrivacyEngine
    from torch.utils.data import DataLoader

    client._dpsgd_config = config
    client._dpsgd_epsilon = 0.0

    _original_compute_update = client.compute_update

    def _dpsgd_train(self):
        self.model.train()
        loader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True,
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate,
        )

        privacy_engine = PrivacyEngine()
        dp_model, dp_optimizer, dp_loader = privacy_engine.make_private(
            module=self.model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=config.noise_multiplier,
            max_grad_norm=config.max_grad_norm,
        )

        total_loss = 0.0
        total_batches = 0

        for epoch in range(self.local_epochs):
            for batch in dp_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                dp_optimizer.zero_grad()
                outputs = dp_model(**batch)
                loss = outputs.loss
                loss.backward()
                dp_optimizer.step()
                total_loss += loss.item()
                total_batches += 1

        self._dpsgd_epsilon = privacy_engine.get_epsilon(config.target_delta)
        logger.info(
            "Client %d: DP-SGD ε=%.4f (δ=%g, σ=%.2f, C=%.2f)",
            self.client_id, self._dpsgd_epsilon,
            config.target_delta, config.noise_multiplier, config.max_grad_norm,
        )

        # Unwrap GradSampleModule to restore the original model
        self.model = dp_model._module
        return total_loss / max(total_batches, 1)

    client._train = types.MethodType(_dpsgd_train, client)

    def _dpsgd_compute_update(parameters, config):
        params, n, metrics = _original_compute_update(parameters, config)
        metrics["dpsgd_epsilon"] = client._dpsgd_epsilon
        return params, n, metrics

    client.compute_update = _dpsgd_compute_update
    return client
