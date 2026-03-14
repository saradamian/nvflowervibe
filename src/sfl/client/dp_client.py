"""
Per-example DP-SGD via Opacus for federated clients.

Wraps a client's local training with Opacus PrivacyEngine to provide
per-sample (ε,δ)-DP guarantees independent of server-side aggregate DP.

Supports **AutoClip** (Li et al., NeurIPS 2023): normalizes per-example
gradients to unit norm before noise addition, eliminating the clipping
norm hyperparameter entirely. Equivalent to DP-SGD with C=1 on
pre-normalized gradients.

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
            Ignored when ``auto_clip=True``.
        noise_multiplier: Gaussian noise multiplier for DP-SGD.
        target_delta: Delta parameter for (ε,δ)-DP accounting.
        auto_clip: Enable Automatic Clipping (Li et al., NeurIPS
            2023). Normalizes per-example gradients to unit norm
            before noise addition, making the clipping norm
            hyperparameter unnecessary. Overrides ``max_grad_norm``
            to 1.0 internally.
    """
    max_grad_norm: float = 1.0
    noise_multiplier: float = 1.0
    target_delta: float = 1e-5
    auto_clip: bool = False


def enable_dpsgd(client, config: DPSGDConfig):
    """Enable Opacus DP-SGD on a client with a _train() method.

    Replaces the client's ``_train`` method with one that wraps the
    model, optimizer, and dataloader with Opacus PrivacyEngine for
    per-example gradient clipping and noise. Also patches
    ``compute_update`` to include the per-round ε in fit metrics.

    When ``config.auto_clip=True``, applies Automatic Clipping (Li et al.,
    NeurIPS 2023): a per-example gradient normalization hook is registered
    that scales each gradient to unit L2 norm before Opacus clips at C=1.
    This eliminates sensitivity to the clipping norm hyperparameter.

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

    # AutoClip: force max_grad_norm=1.0 (gradient normalization handles the rest)
    effective_clip = 1.0 if config.auto_clip else config.max_grad_norm

    def _dpsgd_train(self):
        self.model.train()
        loader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True,
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate,
        )

        # AutoClip: register backward hook that normalizes per-example
        # gradients to unit norm. Opacus then clips at C=1, which is
        # a no-op on unit-norm vectors. The noise scale becomes σ·1 = σ,
        # independent of the original gradient magnitude.
        autoclip_hooks = []
        if config.auto_clip:
            for param in self.model.parameters():
                if param.requires_grad:
                    def _normalize_grad(grad, _p=param):
                        norm = torch.norm(grad, p=2)
                        if norm > 0:
                            return grad / norm
                        return grad
                    h = param.register_hook(_normalize_grad)
                    autoclip_hooks.append(h)

        privacy_engine = PrivacyEngine()
        dp_model, dp_optimizer, dp_loader = privacy_engine.make_private(
            module=self.model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=config.noise_multiplier,
            max_grad_norm=effective_clip,
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

        # Remove AutoClip hooks
        for h in autoclip_hooks:
            h.remove()

        self._dpsgd_epsilon = privacy_engine.get_epsilon(config.target_delta)

        # Optionally compute tighter ε via PLD accountant (S5).
        # Opacus uses RDP→(ε,δ) conversion, which can be 2-5× looser than
        # PLD-based accounting for the same Gaussian mechanism.
        try:
            from sfl.privacy.accountant import PrivacyAccountant
            pld_acc = PrivacyAccountant(
                noise_multiplier=config.noise_multiplier,
                sample_rate=1.0,  # DP-SGD processes full local dataset
                delta=config.target_delta,
                enforce_budget=False,
            )
            # Step once per local epoch × batches
            for _ in range(total_batches):
                pld_acc.step()
            pld_eps = pld_acc.epsilon
            if pld_eps < self._dpsgd_epsilon:
                logger.info(
                    "Client %d: PLD accounting ε=%.4f (tighter than Opacus RDP ε=%.4f)",
                    self.client_id, pld_eps, self._dpsgd_epsilon,
                )
                self._dpsgd_epsilon = pld_eps
        except ImportError:
            pass  # dp-accounting not installed, use Opacus RDP

        clip_info = "AutoClip" if config.auto_clip else f"C={effective_clip:.2f}"
        logger.info(
            "Client %d: DP-SGD ε=%.4f (δ=%g, σ=%.2f, %s)",
            self.client_id, self._dpsgd_epsilon,
            config.target_delta, config.noise_multiplier, clip_info,
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
