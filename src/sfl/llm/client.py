"""
LLM federated learning client.

Implements a Flower client that fine-tunes a causal language model
(e.g., GPT-2) on a local partition of text data.

Extends BaseFederatedClient from the core SFL framework, implementing
compute_update() for training and overriding evaluate() for perplexity
evaluation.

Supports both full fine-tuning and LoRA-based parameter-efficient
fine-tuning. When LoRA is enabled, only adapter parameters are
exchanged with the server, reducing communication and enabling
adapter-aware privacy.
"""

import math
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from flwr.client import Client
from flwr.common import Context

from sfl.client.base import BaseFederatedClient
from sfl.types import Parameters, Metrics, Config, ClientUpdate
from sfl.llm.model import (
    DEFAULT_MODEL_NAME,
    apply_lora,
    get_lora_parameters,
    get_parameters,
    load_model,
    load_tokenizer,
    set_lora_parameters,
    set_parameters,
)
from sfl.llm.dataset import (
    load_demo_dataset,
    load_dataset_from_hub,
    partition_dataset,
    split_train_eval,
)
from sfl.utils.logging import get_logger

logger = get_logger(__name__)


class LLMClient(BaseFederatedClient):
    """Flower client for federated causal LM fine-tuning.

    Extends BaseFederatedClient, implementing compute_update() for
    local training and overriding evaluate() for perplexity evaluation.

    Each client holds a local partition of text data and trains a
    causal LM. Model parameters (or LoRA adapter weights) are exchanged
    with the server as NumPy arrays each round.

    Args:
        client_id: Unique identifier for this client.
        partition_id: Dataset partition index.
        num_partitions: Total number of FL partitions.
        model_name: HuggingFace causal LM identifier.
        learning_rate: Optimizer learning rate.
        local_epochs: Number of local training epochs per FL round.
        batch_size: Training batch size.
        max_length: Maximum tokenized sequence length.
        device: Torch device string (auto-detected if None).
        dataset_name: HuggingFace dataset name (None = built-in demo).
        text_column: Column name for text in HuggingFace datasets.
        max_samples: Max samples from dataset (None = all).
        use_lora: Whether to use LoRA adapters.
        lora_r: LoRA rank.
        lora_alpha: LoRA scaling factor.
    """

    def __init__(
        self,
        client_id: int = 0,
        partition_id: int = 0,
        num_partitions: int = 2,
        model_name: str = DEFAULT_MODEL_NAME,
        learning_rate: float = 5e-5,
        local_epochs: int = 1,
        batch_size: int = 2,
        max_length: int = 128,
        device: str | None = None,
        dataset_name: str | None = None,
        text_column: str = "text",
        max_samples: int | None = None,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: int = 16,
    ) -> None:
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        super().__init__(client_id=client_id, device=resolved_device)
        self.partition_id = partition_id
        self.learning_rate = learning_rate
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.use_lora = use_lora

        # Load model and tokenizer
        self.model = load_model(model_name)
        self.tokenizer = load_tokenizer(model_name)

        # Apply LoRA if requested (before moving to device)
        if use_lora:
            self.model = apply_lora(self.model, r=lora_r, alpha=lora_alpha)

        self.model = self.model.to(self.device)

        # Load dataset -- HuggingFace Hub or built-in demo
        if dataset_name:
            full_dataset = load_dataset_from_hub(
                dataset_name=dataset_name,
                tokenizer=self.tokenizer,
                text_column=text_column,
                max_samples=max_samples,
                max_length=max_length,
            )
        else:
            full_dataset = load_demo_dataset(
                tokenizer=self.tokenizer,
                max_length=max_length,
            )

        # Split train/eval, then partition across clients
        train_split, eval_split = split_train_eval(full_dataset)
        self.train_data = partition_dataset(
            train_split,
            num_partitions=num_partitions,
            partition_id=partition_id,
        )
        self.eval_data = partition_dataset(
            eval_split,
            num_partitions=num_partitions,
            partition_id=partition_id,
        )

        logger.info(
            f"LLMClient {client_id} ready: partition={partition_id}, "
            f"train={len(self.train_data)}, eval={len(self.eval_data)}, "
            f"device={self.device}, lora={use_lora}"
        )

    def get_parameters(self, config: Config) -> Parameters:
        """Return current model parameters (or LoRA params if enabled)."""
        if self.use_lora:
            return get_lora_parameters(self.model)
        return get_parameters(self.model)

    def compute_update(
        self,
        parameters: Parameters,
        config: Config,
    ) -> ClientUpdate:
        """Train the model on local data for one FL round.

        Implements the BaseFederatedClient contract: receive global
        parameters, train locally, return updated parameters.

        When LoRA is enabled, only adapter weights are received and returned.

        Args:
            parameters: Global model parameters from the server.
            config: Server-sent configuration for this round.

        Returns:
            Tuple of (updated parameters, num_examples, metrics).
        """
        # Load global parameters into model
        if self.use_lora:
            set_lora_parameters(self.model, parameters)
        else:
            set_parameters(self.model, parameters)

        # Train
        train_loss = self._train()
        num_examples = len(self.train_data)

        metrics: Metrics = {
            "client_id": self.client_id,
            "train_loss": train_loss,
            "num_examples": num_examples,
        }

        # Compute perplexity from training loss
        if train_loss < 100:  # Guard against overflow
            metrics["train_perplexity"] = math.exp(train_loss)

        logger.info(
            f"Client {self.client_id}: fit complete -- "
            f"loss={train_loss:.4f}, examples={num_examples}"
        )

        if self.use_lora:
            return get_lora_parameters(self.model), num_examples, metrics
        return get_parameters(self.model), num_examples, metrics

    def evaluate(
        self,
        parameters: Parameters,
        config: Config,
    ) -> Tuple[float, int, Metrics]:
        """Evaluate the model on held-out eval data.

        Computes cross-entropy loss and perplexity on the local eval
        partition.

        Args:
            parameters: Global model parameters from the server.
            config: Server-sent configuration.

        Returns:
            Tuple of (loss, num_examples, metrics).
        """
        if self.use_lora:
            set_lora_parameters(self.model, parameters)
        else:
            set_parameters(self.model, parameters)

        # Use eval split if available, fall back to train data
        eval_dataset = self.eval_data if len(self.eval_data) > 0 else self.train_data
        loss = self._evaluate(eval_dataset)
        num_examples = len(eval_dataset)

        metrics: Metrics = {"eval_loss": loss}
        if loss < 100:  # Guard against overflow
            metrics["perplexity"] = math.exp(loss)

        logger.info(
            f"Client {self.client_id}: evaluate -- loss={loss:.4f}, "
            f"perplexity={metrics.get('perplexity', 'N/A')}"
        )

        return loss, num_examples, metrics

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
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.learning_rate,
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
    def _evaluate(self, dataset=None) -> float:
        """Run evaluation loop.

        Args:
            dataset: Dataset to evaluate on (defaults to train_data).

        Returns:
            Average loss over all batches.
        """
        if dataset is None:
            dataset = self.train_data
        self.model.eval()
        loader = DataLoader(
            dataset,
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
    """Factory function for creating LLM clients.

    Called by Flower's ClientApp to instantiate a client per partition.

    Args:
        context: Flower context with node_id, node_config, run_config.

    Returns:
        Configured LLMClient instance wrapped for Flower.
    """
    from sfl.llm.config import get_run_config

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
    model_name = str(run_config.get("llm-model", cfg.model_name))
    learning_rate = float(run_config.get("learning-rate", cfg.learning_rate))
    local_epochs = int(run_config.get("local-epochs", cfg.local_epochs))
    batch_size = int(run_config.get("batch-size", cfg.batch_size))
    max_length = int(run_config.get("max-length", cfg.max_length))
    use_lora = str(run_config.get("use-lora", cfg.use_lora)).lower() in ("true", "1")
    lora_r = int(run_config.get("lora-r", cfg.lora_r))
    lora_alpha = int(run_config.get("lora-alpha", cfg.lora_alpha))

    logger.info(
        f"Creating LLMClient: node_id={node_id}, partition_id={partition_id}, "
        f"model={model_name}, lora={use_lora}"
    )

    # Dataset config
    dataset_name = run_config.get("dataset-name", cfg.dataset_name)
    if dataset_name is not None:
        dataset_name = str(dataset_name)
    text_column = str(run_config.get("text-column", cfg.text_column))
    max_samples_raw = run_config.get("max-samples", cfg.max_samples)
    max_samples = int(max_samples_raw) if max_samples_raw is not None else None

    client = LLMClient(
        client_id=node_id,
        partition_id=partition_id,
        num_partitions=num_partitions,
        model_name=model_name,
        learning_rate=learning_rate,
        local_epochs=local_epochs,
        batch_size=batch_size,
        max_length=max_length,
        dataset_name=dataset_name,
        text_column=text_column,
        max_samples=max_samples,
        use_lora=use_lora,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
    )

    # Wrap with Opacus DP-SGD if configured
    import os
    if os.environ.get("SFL_DPSGD_ENABLED", "").lower() == "true":
        from sfl.client.dp_client import DPSGDConfig, enable_dpsgd
        dpsgd_cfg = DPSGDConfig(
            max_grad_norm=float(os.environ.get("SFL_DPSGD_CLIP", "1.0")),
            noise_multiplier=float(os.environ.get("SFL_DPSGD_NOISE", "1.0")),
            target_delta=float(os.environ.get("SFL_DPSGD_DELTA", "1e-5")),
            auto_clip=os.environ.get("SFL_DPSGD_AUTOCLIP", "").lower() == "true",
            ghost_clipping=os.environ.get("SFL_DPSGD_GHOST", "").lower() == "true",
        )
        enable_dpsgd(client, dpsgd_cfg)

    return client.to_client()
