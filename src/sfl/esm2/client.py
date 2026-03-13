"""
ESM2 federated learning client.

Implements a Flower client that fine-tunes an ESM2 protein language
model on a local partition of protein sequences using masked language modeling.

Extends BaseFederatedClient from the core SFL framework, implementing
compute_update() for training and overriding evaluate() for evaluation.
"""

from typing import Any, Dict, Tuple

import torch
from torch.utils.data import DataLoader
from flwr.client import Client
from flwr.common import Context

from sfl.client.base import BaseFederatedClient
from sfl.types import Parameters, Metrics, Config, ClientUpdate
from sfl.esm2.model import (
    DEFAULT_MODEL_NAME,
    get_parameters,
    load_model,
    load_tokenizer,
    set_parameters,
)
from sfl.esm2.dataset import load_demo_dataset, partition_dataset
from sfl.utils.logging import get_logger

logger = get_logger(__name__)


class ESM2Client(BaseFederatedClient):
    """Flower client for federated ESM2 fine-tuning.

    Extends BaseFederatedClient, implementing compute_update() for
    local training and overriding evaluate() for model evaluation.

    Each client holds a local partition of protein sequences and trains
    an ESM2 model via masked language modeling. Model parameters are
    exchanged with the server as NumPy arrays each round.

    Args:
        client_id: Unique identifier for this client.
        partition_id: Dataset partition index.
        num_partitions: Total number of FL partitions.
        model_name: HuggingFace ESM2 model identifier.
        learning_rate: Optimizer learning rate.
        local_epochs: Number of local training epochs per FL round.
        batch_size: Training batch size.
        max_length: Maximum tokenized sequence length.
        device: Torch device string (auto-detected if None).
    """

    def __init__(
        self,
        client_id: int = 0,
        partition_id: int = 0,
        num_partitions: int = 2,
        model_name: str = DEFAULT_MODEL_NAME,
        learning_rate: float = 5e-5,
        local_epochs: int = 1,
        batch_size: int = 4,
        max_length: int = 128,
        device: str | None = None,
    ) -> None:
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(client_id=client_id, device=resolved_device)
        self.partition_id = partition_id
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.batch_size = batch_size

        # Load model and tokenizer
        self.model = load_model(model_name).to(self.device)  # type: ignore[assignment]
        self.tokenizer = load_tokenizer(model_name)

        # Load and partition dataset
        full_dataset = load_demo_dataset(
            tokenizer=self.tokenizer,
            max_length=max_length,
        )
        self.train_data = partition_dataset(
            full_dataset,
            num_partitions=num_partitions,
            partition_id=partition_id,
        )

        logger.info(
            f"ESM2Client {client_id} ready: partition={partition_id}, "
            f"samples={len(self.train_data)}, device={self.device}"
        )

    def get_parameters(self, config: Config) -> Parameters:
        """Return current model parameters."""
        return get_parameters(self.model)

    def compute_update(
        self,
        parameters: Parameters,
        config: Config,
    ) -> ClientUpdate:
        """Train the model on local data for one FL round.

        Implements the BaseFederatedClient contract: receive global
        parameters, train locally, return updated parameters.

        Args:
            parameters: Global model parameters from the server.
            config: Server-sent configuration for this round.

        Returns:
            Tuple of (updated parameters, num_examples, metrics).
        """
        # Load global parameters into model
        set_parameters(self.model, parameters)

        # Train
        train_loss = self._train()
        num_examples = len(self.train_data)

        metrics: Metrics = {
            "client_id": self.client_id,
            "train_loss": train_loss,
            "num_examples": num_examples,
        }

        logger.info(
            f"Client {self.client_id}: fit complete — "
            f"loss={train_loss:.4f}, examples={num_examples}"
        )

        return get_parameters(self.model), num_examples, metrics

    def evaluate(
        self,
        parameters: Parameters,
        config: Config,
    ) -> Tuple[float, int, Metrics]:
        """Evaluate the model on local data.

        Overrides BaseFederatedClient's default no-op evaluation
        with actual model evaluation on the local partition.

        Args:
            parameters: Global model parameters from the server.
            config: Server-sent configuration.

        Returns:
            Tuple of (loss, num_examples, metrics).
        """
        set_parameters(self.model, parameters)

        loss = self._evaluate()
        num_examples = len(self.train_data)

        logger.info(
            f"Client {self.client_id}: evaluate — loss={loss:.4f}"
        )

        return loss, num_examples, {"eval_loss": loss}

    def _train(self) -> float:
        """Run local training loop.

        Returns:
            Average training loss over all batches.
        """
        self.model.train()
        loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
        )
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate
        )

        total_loss = 0.0
        total_batches = 0

        for epoch in range(self.local_epochs):
            for batch in loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                total_batches += 1

        return total_loss / max(total_batches, 1)

    @torch.no_grad()
    def _evaluate(self) -> float:
        """Run evaluation loop.

        Returns:
            Average loss over all batches.
        """
        self.model.eval()
        loader = DataLoader(
            self.train_data,
            batch_size=self.batch_size,
        )

        total_loss = 0.0
        total_batches = 0

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(**batch)
            total_loss += outputs.loss.item()
            total_batches += 1

        return total_loss / max(total_batches, 1)


def client_fn(context: Context) -> Client:
    """Factory function for creating ESM2 clients.

    Called by Flower's ClientApp to instantiate a client per partition.

    Args:
        context: Flower context with node_id, node_config, run_config.

    Returns:
        Configured ESM2Client instance wrapped for Flower.
    """
    from sfl.esm2.config import get_run_config

    partition_id = context.node_config.get("partition-id", 0)
    if isinstance(partition_id, str):
        partition_id = int(partition_id)

    num_partitions = context.node_config.get("num-partitions", None)
    if num_partitions is not None:
        num_partitions = int(num_partitions)

    node_id = int(context.node_id)
    partition_id = int(partition_id)

    # Read config: prefer shared run config (set by runner),
    # fall back to context.run_config (set by pyproject.toml / NVFlare)
    cfg = get_run_config()
    run_config = context.run_config or {}

    # num_partitions from node_config is authoritative (set by Flower simulation)
    if num_partitions is None:
        num_partitions = int(run_config.get("num-clients", cfg.num_clients))
    model_name = str(run_config.get("esm2-model", cfg.model_name))
    learning_rate = float(run_config.get("learning-rate", cfg.learning_rate))
    local_epochs = int(run_config.get("local-epochs", cfg.local_epochs))
    batch_size = int(run_config.get("batch-size", cfg.batch_size))
    max_length = int(run_config.get("max-length", cfg.max_length))

    logger.info(
        f"Creating ESM2Client: node_id={node_id}, partition_id={partition_id}, "
        f"model={model_name}"
    )

    client = ESM2Client(
        client_id=node_id,
        partition_id=partition_id,
        num_partitions=num_partitions,
        model_name=model_name,
        learning_rate=learning_rate,
        local_epochs=local_epochs,
        batch_size=batch_size,
        max_length=max_length,
    )

    return client.to_client()
