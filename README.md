# SFL - Simple Federated Learning Demo

A clean, maintainable, and extensible federated learning demonstration using **NVIDIA FLARE (NVFlare)** and **Flower** frameworks.

## 🎯 Overview

This project demonstrates a "Federated Sum" example where multiple clients contribute secret values, and the server aggregates them securely. It showcases:

- **NVFlare + Flower integration** for production-ready federated learning
- **Clean architecture** with separation of concerns
- **Type hints** throughout for maintainability
- **Configurable** via YAML, environment variables, or CLI
- **Extensible** design for adding custom strategies and clients

## 📁 Project Structure

```
sfl/
├── README.md                 # This file
├── pyproject.toml           # Project metadata & Flower configuration
├── requirements.txt         # Pinned dependencies
├── .env.example            # Environment variables template
├── config/
│   └── default.yaml        # Default configuration
├── src/
│   └── sfl/
│       ├── __init__.py
│       ├── client/
│       │   ├── __init__.py
│       │   ├── base.py     # Abstract base client
│       │   └── sum_client.py
│       ├── server/
│       │   ├── __init__.py
│       │   ├── strategy.py # Custom FedAvg strategy
│       │   └── app.py      # Server application
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── config.py   # Configuration management
│       │   └── logging.py  # Logging utilities
│       └── types.py        # Shared type definitions
├── jobs/
│   └── runner.py           # NVFlare job runner
├── scripts/
│   ├── setup.sh           # Environment setup
│   └── run_simulation.sh  # Quick run script
└── tests/
    └── test_client.py     # Unit tests
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -U pip wheel setuptools
pip install -r requirements.txt

# Or use the setup script
./scripts/setup.sh
```

### 2. Run the Simulation

#### Option A: NVFlare + Flower (Full Stack)

```bash
# Quick run with defaults (2 clients, 1 round)
python jobs/runner.py

# Custom configuration
python jobs/runner.py --num-clients 4 --num-rounds 3

# Using config file
python jobs/runner.py --config config/default.yaml
```

#### Option B: Pure Flower Simulation

```bash
# Quick run with defaults (2 clients, 1 round)
python jobs/flower_runner.py

# Custom configuration
python jobs/flower_runner.py --num-clients 4 --num-rounds 3

# Using config file
python jobs/flower_runner.py --config config/default.yaml
```

*See [INSTALL_NOTE.md](INSTALL_NOTE.md) for details about NVFlare compatibility issues.*

### 3. Expected Output

```
[server] round=1 client_vals=[7.0, 8.0] federated_sum=15.0
```

## ⚙️ Configuration

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

### YAML Configuration

Edit `config/default.yaml` for persistent settings:

```yaml
federation:
  num_clients: 2
  num_rounds: 1
  min_available_clients: 2

client:
  base_secret: 7.0

logging:
  level: INFO
```

### CLI Arguments

```
--num-clients     Number of federated clients (default: 2)
--num-rounds      Number of training rounds (default: 1)
--config          Path to YAML config file
--verbose, -v     Enable verbose (DEBUG) logging
```

## 🔧 Extending the Project

### Adding a New Client Type

1. Create a new client in `src/sfl/client/`:

```python
from sfl.client.base import BaseFederatedClient

class MyCustomClient(BaseFederatedClient):
    def compute_update(self, parameters, config):
        # Your custom logic here
        pass
```

2. Register it in `src/sfl/client/__init__.py`

### Adding a New Aggregation Strategy

1. Extend `SumFedAvg` in `src/sfl/server/strategy.py`
2. Override `aggregate_fit()` with custom logic

## 🐛 Troubleshooting

See [INSTALL_NOTE.md](INSTALL_NOTE.md) for detailed troubleshooting.

### Version Requirements

The NVFlare + Flower integration requires specific versions:

```
nvflare==2.7.1
flwr[simulation]==1.17.0
click>=8.1.0,<8.2.0  # Important: Click 8.2+ breaks Typer
```

### Common Issues

1. **"--serverappio-api-address" not recognized**: Upgrade Flower to 1.17.0
2. **CLI crashes with Typer trace**: Downgrade Click to < 8.2
3. **"--format" not valid for flwr run**: Upgrade Flower to 1.17.0

## 📚 Resources

- [NVFlare Documentation](https://nvflare.readthedocs.io/)
- [Flower Documentation](https://flower.ai/docs/)
- [NVFlare + Flower Integration Guide](https://nvflare.readthedocs.io/en/main/hello-world/hello-flower/)

## 📝 License

MIT License - See LICENSE file for details.
