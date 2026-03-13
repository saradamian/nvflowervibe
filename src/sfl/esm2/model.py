"""
ESM2 model utilities for federated learning.

Handles model loading, parameter extraction/injection for Flower,
and model configuration.
"""

from collections import OrderedDict
from typing import List

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import AutoModelForMaskedLM, AutoTokenizer, EsmForMaskedLM, PreTrainedTokenizerBase

from sfl.utils.logging import get_logger

logger = get_logger(__name__)

# Smallest ESM2 model — practical for FL demos
DEFAULT_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"


def load_model(model_name: str = DEFAULT_MODEL_NAME) -> EsmForMaskedLM:
    """Load an ESM2 model for masked language modeling.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        ESM2 model ready for training.
    """
    logger.info(f"Loading ESM2 model: {model_name}")
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    logger.info(
        f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters"
    )
    return model


def load_tokenizer(model_name: str = DEFAULT_MODEL_NAME) -> PreTrainedTokenizerBase:
    """Load the tokenizer for an ESM2 model.

    Args:
        model_name: HuggingFace model identifier.

    Returns:
        Tokenizer instance.
    """
    return AutoTokenizer.from_pretrained(model_name)


def get_parameters(model: torch.nn.Module) -> List[NDArray[np.float32]]:
    """Extract model parameters as a list of NumPy arrays.

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
