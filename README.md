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
├── config/
│   └── default.yaml        # Default configuration
├── docs/
│   ├── DEPLOYMENT.md       # Detailed deployment guide
│   └── INSTALL_NOTE.md     # Installation & troubleshooting
├── src/
│   └── sfl/
│       ├── __init__.py
│       ├── types.py        # Shared type definitions
│       ├── client/
│       │   ├── __init__.py
│       │   ├── base.py     # Abstract base client
│       │   └── sum_client.py
│       ├── server/
│       │   ├── __init__.py
│       │   ├── strategy.py # Custom FedAvg strategy
│       │   └── app.py      # Server application
│       └── utils/
│           ├── __init__.py
│           ├── config.py   # Configuration management
│           └── logging.py  # Logging utilities
├── jobs/
│   ├── runner.py           # NVFlare SimEnv runner
│   ├── flower_runner.py    # Pure Flower runner (fallback)
│   └── poc_runner.py       # NVFlare POC mode runner
├── scripts/
│   ├── setup.sh              # Environment setup
│   ├── run_simulation.sh     # Quick run script
│   └── generate_nvflare_job.py  # POC/Production job generator
└── tests/
    ├── __init__.py
    ├── test_client.py
    └── test_config.py
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

Three deployment modes available, from simplest to production-ready:

#### Mode 1: SimEnv (Recommended for Development)

Single-process simulation - fastest for development and testing:

```bash
# Quick run with defaults (2 clients, 1 round)
python jobs/runner.py

# Custom configuration
python jobs/runner.py --num-clients 4 --num-rounds 3

# Using config file
python jobs/runner.py --config config/default.yaml
```

#### Mode 2: Pure Flower (Fallback Option)

Pure Flower simulation without NVFlare (useful if NVFlare has compatibility issues):

```bash
# Quick run with defaults
python jobs/flower_runner.py

# Custom configuration
python jobs/flower_runner.py --num-clients 4 --num-rounds 3

# Using config file
python jobs/flower_runner.py --config config/default.yaml
```

*See [docs/INSTALL_NOTE.md](docs/INSTALL_NOTE.md) for details about NVFlare compatibility issues.*

#### Mode 3: POC Mode (Multi-Process Testing)

Multi-process local deployment simulating real network behavior:

```bash
# Step 1: Prepare POC environment (creates server + client processes)
python jobs/poc_runner.py prepare --num-clients 4 --clean

# Step 2: Start all services
python jobs/poc_runner.py start

# Step 3: Submit job via admin console
python jobs/poc_runner.py submit

# Step 4: Monitor and stop
python jobs/poc_runner.py status
python jobs/poc_runner.py stop

# Clean up POC workspace
python jobs/poc_runner.py clean
```

See the **Deployment Modes** section below for production deployments.

### 3. Expected Output

```
[server] round=1 client_vals=[7.0, 8.0] federated_sum=15.0
```

---

## 📦 Deployment Modes

This project supports multiple deployment modes, from development to production:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     NVFlare Deployment Modes                        │
├─────────────┬──────────────┬──────────────┬────────────────────────┤
│   SimEnv    │     POC      │  Provision   │      Production        │
├─────────────┼──────────────┼──────────────┼────────────────────────┤
│ Single      │ Multi-process│ Multi-machine│ Multi-machine          │
│ process     │ local        │ generated    │ with TLS + Auth        │
├─────────────┼──────────────┼──────────────┼────────────────────────┤
│ Development │ Integration  │ Staging      │ Production             │
│ & Testing   │ Testing      │ & Testing    │ Deployment             │
└─────────────┴──────────────┴──────────────┴────────────────────────┘
```

### Mode 1: SimEnv (Current Default)

**Use for**: Quick development, algorithm testing, CI/CD

```bash
python jobs/runner.py --num-clients 4 --num-rounds 3
```

**Pros**: Fast, simple, no setup needed  
**Cons**: No network simulation, no security

---

### Mode 2: POC Mode (Proof of Concept)

**Use for**: Local multi-process testing, simulating real network behavior

#### Full POC Workflow

```bash
# 1. Prepare POC environment with 4 clients
python jobs/poc_runner.py prepare --num-clients 4 --clean

# This creates /tmp/nvflare/poc/ with:
# - server/
# - site-1/, site-2/, site-3/, site-4/
# - admin/
# - jobs/sfl-job/

# 2. Start all services (server + clients)
python jobs/poc_runner.py start

# Or start components individually:
nvflare poc start -p server
nvflare poc start -p site-1
nvflare poc start -p admin

# 3. Submit job via admin console
python jobs/poc_runner.py submit

# Or submit manually:
nvflare poc start -p admin
# In admin console:
> submit_job /tmp/nvflare/poc/jobs/sfl-job/sfl-federated-sum
> list_jobs
> check_status server

# 4. Monitor status
python jobs/poc_runner.py status

# 5. Stop services
python jobs/poc_runner.py stop

# 6. Clean up (optional)
python jobs/poc_runner.py clean
```

#### Manual POC Setup (Alternative)

```bash
# Prepare POC manually
nvflare poc prepare -n 4

# Generate job from Flower app
python scripts/generate_nvflare_job.py \
    --output /tmp/nvflare/poc/jobs/sfl-job \
    --num-clients 4

# Start and submit
nvflare poc start
nvflare poc start -p admin
# > submit_job /tmp/nvflare/poc/jobs/sfl-job/sfl-federated-sum
```

---

### Mode 3: Production Mode (Provisioned)

**Use for**: Real deployments with TLS certificates, authentication, authorization

#### Step 1: Generate Project Template

```bash
# Generate sample project.yml
nvflare provision -g
```

#### Step 2: Configure project.yml

Create or edit `project.yml`:

```yaml
# project.yml
api_version: 3
name: sfl-production
description: SFL Federated Learning Production Setup

participants:
  # The overseer manages HA (High Availability)
  - name: overseer
    type: overseer
    org: sfl-org
    protocol: https
    api_root: /api/v1
    port: 8443

  # FL Server
  - name: sfl-server
    type: server
    org: sfl-org
    fed_learn_port: 8002
    admin_port: 8003
    enable_byoc: true

  # FL Clients (sites)
  - name: hospital-a
    type: client
    org: hospital-a-org
    
  - name: hospital-b
    type: client
    org: hospital-b-org
    
  - name: hospital-c
    type: client
    org: hospital-c-org

  # Admin users
  - name: admin@sfl.org
    type: admin
    org: sfl-org
    role: project_admin

builders:
  - path: nvflare.lighter.impl.workspace.WorkspaceBuilder
    args:
      template_file: master_template.yml
      
  - path: nvflare.lighter.impl.template.TemplateBuilder
  
  - path: nvflare.lighter.impl.static_file.StaticFileBuilder
    args:
      config_folder: config
      
  - path: nvflare.lighter.impl.cert.CertBuilder
  
  - path: nvflare.lighter.impl.signature.SignatureBuilder
```

#### Step 3: Run Provisioning

```bash
# Generate startup kits for all participants
nvflare provision -p project.yml -w ./workspace

# This creates:
# workspace/
# ├── sfl-server/
# │   └── startup/
# │       ├── start.sh
# │       ├── fed_server.json
# │       └── rootCA.pem (TLS cert)
# ├── hospital-a/
# │   └── startup/
# │       ├── start.sh
# │       └── ... (client certs)
# └── admin@sfl.org/
#     └── startup/
#         └── fl_admin.sh
```

#### Step 4: Distribute Startup Kits

Send each participant their startup kit:
- `sfl-server/` → Deploy to server machine
- `hospital-a/` → Send to Hospital A
- `hospital-b/` → Send to Hospital B
- `admin@sfl.org/` → Keep for admin access

#### Step 5: Start Services

**On Server Machine:**
```bash
cd sfl-server/startup
./start.sh
```

**On Each Client Machine:**
```bash
cd hospital-a/startup
./start.sh
```

**Admin Console:**
```bash
cd admin@sfl.org/startup
./fl_admin.sh

# Connect and submit job
> submit_job /path/to/sfl/job
```

---

### Mode 4: Docker Deployment

For containerized deployments, see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md#4-docker-deployment) for:
- Dockerfile templates for server and clients
- Docker Compose configuration
- Container orchestration setup

---

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

All runners support:

```
--num-clients     Number of federated clients (default: 2)
--num-rounds      Number of training rounds (default: 1)
--config          Path to YAML config file
--verbose, -v     Enable verbose (DEBUG) logging
```

POC runner additional commands:

```
prepare           Prepare POC environment
  --num-clients   Number of clients to create
  --clean         Clean existing POC before creating new one
  
start             Start POC services
  --component     Start specific component (server, site-1, admin, etc.)
  
submit            Submit job via admin console
stop              Stop all POC services
status            Check status of POC services
clean             Remove POC workspace
```

---

## 🔧 Extending the Project

### Adding a New Client Type

1. Create a new client in `src/sfl/client/`:

```python
from sfl.client.base import BaseFederatedClient
from flwr.common import NDArrays, Scalar

class MyCustomClient(BaseFederatedClient):
    def get_parameters(self, config: dict[str, Scalar]) -> NDArrays:
        # Return client parameters
        pass
        
    def fit(self, parameters: NDArrays, config: dict[str, Scalar]) -> tuple[NDArrays, int, dict[str, Scalar]]:
        # Your custom training logic here
        pass
```

2. Register it in `src/sfl/client/__init__.py`

### Adding a New Aggregation Strategy

1. Extend `SumFedAvg` in `src/sfl/server/strategy.py`:

```python
from flwr.server.strategy import FedAvg

class MyCustomStrategy(FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        # Your custom aggregation logic
        pass
```

2. Update server app in `src/sfl/server/app.py` to use your strategy

---

## 🐛 Troubleshooting

See [docs/INSTALL_NOTE.md](docs/INSTALL_NOTE.md) for detailed troubleshooting.

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
4. **POC services fail to start**: Check if ports 8002, 8003 are available
5. **Job submission fails**: Ensure POC is prepared with `python jobs/poc_runner.py prepare`

### NVFlare Commands Quick Reference

| Command | Description |
|---------|-------------|
| `nvflare poc prepare -n 4` | Prepare POC with 4 clients |
| `nvflare poc start` | Start all POC services |
| `nvflare poc start -p admin` | Start admin console |
| `nvflare poc stop` | Stop all POC services |
| `nvflare poc clean` | Remove POC workspace |
| `nvflare provision -g` | Generate project.yml template |
| `nvflare provision -p project.yml` | Provision from project file |
| `nvflare preflight_check` | Check system readiness |
| `nvflare dashboard` | Start web dashboard |

---

## 📚 Resources

- [NVFlare Documentation](https://nvflare.readthedocs.io/)
- [Flower Documentation](https://flower.ai/docs/)
- [NVFlare + Flower Integration Guide](https://nvflare.readthedocs.io/en/main/hello-world/hello-flower/)
- [NVFlare POC Mode Guide](https://nvflare.readthedocs.io/en/main/getting_started.html#poc-mode)
- [NVFlare Production Deployment](https://nvflare.readthedocs.io/en/main/programming_guide/provisioning_tool.html)

---

## 🔐 Security Features (Production Mode)

### TLS/SSL Encryption
- All communication encrypted with certificates
- Mutual TLS (mTLS) authentication

### Authorization
- Role-based access control (RBAC)
- Job-level permissions

### Audit Logging
- All actions logged
- Compliance-ready audit trail

### Secure Aggregation (Optional)
- Homomorphic encryption support
- Differential privacy

---

## 📈 Recommended Progression

1. **Start**: Use `SimEnv` mode (`python jobs/runner.py`) for development
2. **Test**: Use `POC mode` (`python jobs/poc_runner.py`) for integration testing
3. **Stage**: Use `Provision` mode for staging environment
4. **Deploy**: Full production with TLS + monitoring

---

## 📝 License

MIT License - See LICENSE file for details.
