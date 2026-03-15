"""
LLM Federated Fine-Tuning Module.

Provides federated fine-tuning of causal language models (e.g., GPT-2)
using the Flower framework, orchestrated by NVFlare.

Supports both full fine-tuning and LoRA-based parameter-efficient
fine-tuning with adapter-aware privacy.

Example usage (standalone):
    python jobs/llm_runner.py
    python jobs/llm_runner.py --num-clients 4 --num-rounds 5
    python jobs/llm_runner.py --use-lora --model gpt2
"""

from sfl.llm.client import LLMClient, client_fn
from sfl.llm.server import server_fn

# Flower apps for NVFlare integration
from flwr.client import ClientApp
from flwr.server import ServerApp

client_app = ClientApp(client_fn=client_fn)
server_app = ServerApp(server_fn=server_fn)

__all__ = [
    "LLMClient",
    "client_fn",
    "server_fn",
    "client_app",
    "server_app",
]
