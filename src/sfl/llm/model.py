"""
LLM model utilities for federated learning.

Handles model loading, parameter extraction/injection for Flower,
tokenizer setup, and optional LoRA adapter management.

Supports both full fine-tuning and LoRA-based parameter-efficient
fine-tuning. When LoRA is enabled, only adapter weights are exchanged
between clients and server, reducing communication cost and enabling
adapter-aware privacy mechanisms.
"""

from collections import OrderedDict
from typing import List, Optional

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from sfl.utils.logging import get_logger

logger = get_logger(__name__)

# Smallest GPT-2 model -- practical for FL demos on CPU
DEFAULT_MODEL_NAME = "gpt2"


def load_model(model_name: str = DEFAULT_MODEL_NAME) -> PreTrainedModel:
    """Load a causal language model for fine-tuning.

    Args:
        model_name: HuggingFace model identifier (e.g. "gpt2", "gpt2-medium").

    Returns:
        Causal LM ready for training.
    """
    logger.info(f"Loading causal LM: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    logger.info(
        f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters"
    )
    return model


def load_tokenizer(model_name: str = DEFAULT_MODEL_NAME) -> PreTrainedTokenizerBase:
    """Load the tokenizer for a causal LM.

    Sets pad_token to eos_token if not already defined, which is
    required for GPT-2 and similar decoder-only models.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Tokenizer instance with pad_token configured.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # GPT-2 and similar models lack a pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_parameters(model: torch.nn.Module) -> List[NDArray[np.float32]]:
    """Extract all model parameters as a list of NumPy arrays.

    This is the format Flower uses to transmit parameters between
    server and clients.

    Args:
        model: PyTorch model.

    Returns:
        List of NumPy arrays, one per model parameter tensor.
    """
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model: torch.nn.Module, parameters: List[NDArray[np.float32]]) -> None:
    """Load parameters from NumPy arrays into a PyTorch model.

    Args:
        model: PyTorch model to update.
        parameters: List of NumPy arrays matching the model's state_dict.

    Raises:
        ValueError: If the number of parameter tensors doesn't match.
    """
    keys = list(model.state_dict().keys())
    if len(keys) != len(parameters):
        raise ValueError(
            f"Parameter count mismatch: model has {len(keys)} tensors, "
            f"received {len(parameters)}"
        )
    state_dict = OrderedDict(
        {k: torch.from_numpy(v.copy()) for k, v in zip(keys, parameters)}
    )
    model.load_state_dict(state_dict, strict=True)


def apply_lora(
    model: PreTrainedModel,
    r: int = 8,
    alpha: int = 16,
    target_modules: Optional[List[str]] = None,
) -> PreTrainedModel:
    """Apply LoRA adapters to a pretrained model.

    Wraps the model with low-rank adapters using the peft library.
    Only adapter parameters will be trainable; base model weights are frozen.

    Args:
        model: Base pretrained model.
        r: LoRA rank (number of low-rank dimensions).
        alpha: LoRA scaling factor (effective scale = alpha / r).
        target_modules: List of module name patterns to apply LoRA to.
            Defaults to attention projection layers for GPT-2-style models.

    Returns:
        Model wrapped with LoRA adapters.

    Raises:
        ImportError: If the peft library is not installed.
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        raise ImportError(
            "LoRA requires the 'peft' library. Install with: pip install peft"
        )

    if target_modules is None:
        # Default targets for GPT-2 style models (attention projections)
        target_modules = ["c_attn", "c_proj"]

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        f"LoRA applied: {trainable:,} trainable / {total:,} total "
        f"({100 * trainable / total:.2f}%)"
    )
    return model


def get_lora_parameters(model: torch.nn.Module) -> List[NDArray[np.float32]]:
    """Extract only LoRA adapter parameters as NumPy arrays.

    Only returns parameters whose names contain 'lora_' (the adapter
    weights). This enables adapter-aware privacy: only adapter deltas
    are clipped, noised, and transmitted.

    Args:
        model: PyTorch model with LoRA adapters applied.

    Returns:
        List of NumPy arrays for LoRA adapter weights only.
    """
    return [
        val.cpu().numpy()
        for name, val in model.state_dict().items()
        if "lora_" in name
    ]


def set_lora_parameters(model: torch.nn.Module, parameters: List[NDArray[np.float32]]) -> None:
    """Load LoRA adapter parameters from NumPy arrays.

    Only updates parameters whose names contain 'lora_'. Base model
    weights remain unchanged.

    Args:
        model: PyTorch model with LoRA adapters.
        parameters: List of NumPy arrays for LoRA weights only.

    Raises:
        ValueError: If the number of LoRA parameter tensors doesn't match.
    """
    lora_keys = [name for name in model.state_dict().keys() if "lora_" in name]
    if len(lora_keys) != len(parameters):
        raise ValueError(
            f"LoRA parameter count mismatch: model has {len(lora_keys)} "
            f"LoRA tensors, received {len(parameters)}"
        )
    # Build partial state dict with only LoRA weights
    current_state = model.state_dict()
    for key, param in zip(lora_keys, parameters):
        current_state[key] = torch.from_numpy(param.copy())
    model.load_state_dict(current_state, strict=True)
