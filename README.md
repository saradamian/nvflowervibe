# SFL - Simple Federated Learning Framework

An extensible federated learning framework using **NVIDIA FLARE (NVFlare)** and **Flower**, with a production-ready ESM2 protein language model application.

## Overview

This project provides:

- **Extensible base framework** — abstract `BaseFederatedClient` with type-safe contracts, configurable `FederationConfig`, shared type aliases
- **ESM2 protein model application** — federated fine-tuning of ESM2 (8M–35M params) via masked language modeling, built on the base framework
- **NVFlare + Flower integration** — run with pure Flower simulation or NVFlare orchestration
- **Layered privacy** — differential privacy (server + client DP-SGD with AutoClip and Ghost Clipping), PLD-based accounting with shuffle-model amplification, adaptive and per-layer clipping, privacy filters with error feedback, SecAgg+ with partial freezing, and HE
- **Byzantine robustness** — Multi-Krum, Trimmed Mean, and FoundationFL (NDSS 2025) aggregation strategies
- **Configurable** via YAML, environment variables, or CLI
- **357 tests** (243 fast + 114 slow) with GitHub Actions CI on every PR
- **Privacy auditing** — `PrivacyAuditor` for empirical DP validation through real mod chains
- **HPC production** — round-level checkpointing, metric export (CSV/JSON/TensorBoard), mTLS/token auth, resource-aware scheduling, SLURM deployment examples

### Applications

| Application | Description | Runner |
|-------------|-------------|--------|
| **Federated Sum** (demo) | Clients contribute secret values, server aggregates | `python jobs/runner.py` |
| **ESM2 FL** (protein) | Federated fine-tuning of ESM2 protein language model | `python jobs/esm2_runner.py` |
| **LLM FL** (language) | Federated fine-tuning of causal LMs (GPT-2, with LoRA) | `python jobs/llm_runner.py` |

## Project Structure

```
sfl/
├── pyproject.toml              # Project metadata & Flower app config
├── requirements.txt            # Pinned dependencies
├── config/
│   ├── default.yaml            # Default federation config
│   └── esm2.yaml               # ESM2-specific config
├── docs/
│   ├── DEPLOYMENT.md           # Deployment guide (SimEnv → Production)
│   └── INSTALL_NOTE.md         # Installation & troubleshooting
├── src/sfl/
│   ├── __init__.py             # Exports FederationConfig, ClientConfig, etc.
│   ├── types.py                # Shared types: Parameters, Metrics, Config, ClientUpdate
│   ├── client/
│   │   ├── base.py             # BaseFederatedClient (abstract, device-aware)
│   │   ├── inference.py        # BaseInferenceClient (federated inference/eval)
│   │   ├── dp_client.py        # Per-example DP-SGD via Opacus
│   │   └── sum_client.py       # SumClient — demo implementation
│   ├── server/
│   │   ├── strategy.py         # SumFedAvg — custom aggregation strategy
│   │   ├── robust.py           # Multi-Krum, Trimmed Mean, FoundationFL (Byzantine-robust)
│   │   └── app.py              # Server application (sum demo)
│   ├── esm2/
│   │   ├── __init__.py         # Exports client_app, server_app (Flower apps)
│   │   ├── config.py           # ESM2RunConfig (composes FederationConfig)
│   │   ├── model.py            # ESM2 load/save, parameter serialization
│   │   ├── dataset.py          # Protein sequences, MLM masking, IID partitioning
│   │   ├── client.py           # ESM2Client (extends BaseFederatedClient)
│   │   └── server.py           # FedAvg seeded with pretrained ESM2 weights
│   ├── llm/
│   │   ├── __init__.py         # Exports client_app, server_app
│   │   ├── config.py           # LLMRunConfig (composes FederationConfig)
│   │   ├── model.py            # Causal LM load/save, LoRA support
│   │   ├── dataset.py          # Text dataset, causal LM tokenization
│   │   ├── client.py           # LLMClient (extends BaseFederatedClient)
│   │   └── server.py           # FedAvg seeded with pretrained weights
│   ├── privacy/
│   │   ├── __init__.py         # Privacy module exports
│   │   ├── accountant.py       # PLD/PRV privacy accounting + budget enforcement + auxiliary composition
│   │   ├── adaptive_clip.py    # Adaptive clipping norm (Andrew et al. 2021)
│   │   ├── audit.py            # PrivacyAuditor — empirical DP validation through real mod chains
│   │   ├── dp.py               # DP wrappers, noise calibration, accounting wrapper
│   │   ├── filters.py          # Privacy filters (Percentile, SVT, Compression, Freeze, Adapter)
│   │   ├── runner_utils.py     # Shared CLI args + privacy mod builder for runners
│   │   ├── he.py               # Homomorphic encryption (TenSEAL CKKS, demo only)
│   │   └── secagg.py           # SecAgg+ configuration
│   └── utils/
│       ├── config.py           # YAML + env + CLI config management
│       ├── params.py           # Mixed-precision parameter downcast/upcast
│       └── logging.py          # Rich/simple/JSON logging
├── jobs/
│   ├── runner.py               # Sum demo — NVFlare SimEnv runner
│   ├── esm2_runner.py          # ESM2 — Flower or NVFlare backend
│   ├── flower_runner.py        # Sum demo — pure Flower fallback
│   └── poc_runner.py           # NVFlare POC mode runner
├── tests/
│   ├── test_client.py          # SumClient + FederationConfig tests
│   ├── test_config.py          # Config utility tests
│   ├── test_integration.py     # End-to-end aggregation pipeline tests
│   ├── test_privacy.py         # DP wrapping, adaptive clip, SecAgg, server DP
│   ├── test_accountant.py      # Privacy accountant, budget, composition
│   ├── test_filters.py         # Percentile, SVT, compression, HE filters
│   ├── test_dpsgd.py           # Per-example DP-SGD (Opacus)
│   ├── test_robust.py          # Multi-Krum, Trimmed Mean, FoundationFL aggregation
│   ├── test_esm2_config.py     # ESM2RunConfig + FederationConfig composition
│   ├── test_esm2_model.py      # Model loading, parameter roundtrip
│   ├── test_esm2_dataset.py    # MLM masking, partitioning
│   ├── test_esm2_client.py     # ESM2Client inheritance, training, evaluation
│   └── test_esm2_server.py     # Server FedAvg setup, config overrides
├── CONTRIBUTING.md             # Contribution guide
├── SECURITY_AUDIT.md           # Security audit findings
├── UPGRADE_PLAN.md             # Upgrade plan for security improvements
├── docs/
│   ├── DEPLOYMENT.md           # Deployment guide (SimEnv → Production)
│   ├── INSTALL_NOTE.md         # Installation & troubleshooting
│   └── PRIVACY.md              # Privacy & security guide
├── scripts/
│   ├── setup.sh                # Environment setup
│   ├── run_simulation.sh       # Quick run script
│   └── generate_nvflare_job.py # POC/Production job generator
└── .github/workflows/
    └── ci.yml                  # GitHub Actions: pytest on PR to main
```

## Quick Start

### 1. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
pip install -r requirements.txt
```

### 2. Run the Sum Demo

```bash
# Default: 2 clients, 1 round
python jobs/runner.py

# Custom
python jobs/runner.py --num-clients 4 --num-rounds 3
```

Expected output:
```
[server] round=1 client_vals=[7.0, 8.0] federated_sum=15.0
```

### 3. Run ESM2 Federated Training

```bash
# Quick demo — 2 clients, 3 rounds, 8M-param ESM2 model
python jobs/esm2_runner.py

# Custom settings
python jobs/esm2_runner.py --num-clients 4 --num-rounds 5 --learning-rate 1e-4

# Larger model
python jobs/esm2_runner.py --model facebook/esm2_t12_35M_UR50D

# NVFlare backend
python jobs/esm2_runner.py --backend nvflare --num-clients 2 --num-rounds 2
```

GPU auto-detection: if CUDA GPUs are available, they are automatically allocated across simulation clients.

### 4. Privacy & Security

SFL supports layered privacy: DP with automatic budget enforcement,
adaptive clipping, per-example DP-SGD, privacy filters, gradient
compression, Byzantine-robust aggregation, SecAgg+, and HE.

```bash
# Server-side DP with automatic privacy accounting
python jobs/esm2_runner.py --dp --dp-noise 0.1

# Client-side DP
python jobs/esm2_runner.py --dp --dp-mode client --dp-noise 0.5

# Adaptive clipping (Andrew et al. 2021)
python jobs/esm2_runner.py --dp --dp-adaptive-clip --dp-target-quantile 0.5

# Per-example DP-SGD via Opacus (client-local)
python jobs/esm2_runner.py --dpsgd --dpsgd-noise 1.0 --dpsgd-clip 1.0

# Percentile privacy filter (only share top 10% of weight diffs)
python jobs/esm2_runner.py --percentile-privacy 10

# Calibrated percentile noise (formal ε-DP guarantee)
python jobs/esm2_runner.py --percentile-privacy 10 --percentile-epsilon 1.0

# SVT differential privacy (formal ε-DP)
python jobs/esm2_runner.py --svt-privacy --svt-epsilon 0.1

# Gradient compression (TopK sparsification + noise)
python jobs/esm2_runner.py --compress 0.1 --compress-topk

# Exclude embedding layers from aggregation
python jobs/esm2_runner.py --exclude-layers 0,1

# AutoClip DP-SGD (no clipping norm tuning needed)
python jobs/esm2_runner.py --dpsgd --dpsgd-autoclip --dpsgd-noise 1.0

# Ghost Clipping (memory-efficient DP-SGD)
python jobs/esm2_runner.py --dpsgd --dpsgd-ghost --dpsgd-noise 1.0

# Shuffle-model DP amplification
python jobs/esm2_runner.py --dp --dp-shuffle

# Per-layer clipping
python jobs/esm2_runner.py --per-layer-clip 1.0

# Compression with error feedback
python jobs/esm2_runner.py --compress 0.1 --compress-error-feedback

# Partial freezing (Lambda-SecAgg)
python jobs/esm2_runner.py --secagg --freeze-layers 4,5,6

# Byzantine-robust aggregation (Multi-Krum)
python jobs/esm2_runner.py --aggregation krum --krum-byzantine 1

# Byzantine-robust aggregation (Trimmed Mean)
python jobs/esm2_runner.py --aggregation trimmed-mean --trim-ratio 0.1

# FoundationFL trust scoring (NDSS 2025)
python jobs/esm2_runner.py --aggregation foundation-fl --ffl-threshold 0.1

# Combine multiple layers
python jobs/esm2_runner.py --dp --dp-adaptive-clip \
    --percentile-privacy 10 --exclude-layers 0,1
```

See [docs/PRIVACY.md](docs/PRIVACY.md) for the full privacy guide,
including HE limitations, confidential computing, and trade-off analysis.

### 5. Run Tests

```bash
# Fast tests only (CI default, ~2.5 min)
python -m pytest tests/ -v -m "not slow"

# Full suite including slow tests (torch/GPU/PRV, ~8 min)
python -m pytest tests/ -v

# Specific module
python -m pytest tests/test_filters.py -v
```

Tests are split into **fast** (176 tests, no heavy dependencies) and **slow** (62 tests, requires torch/opacus/dp-accounting). CI runs fast tests only via `@pytest.mark.slow`.

---

## Deployment Modes

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for full details.

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

## Configuration

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

## Extending the Framework

### Adding a New FL Application

The ESM2 module demonstrates the pattern. To add your own:

1. **Create a module** in `src/sfl/your_app/`

2. **Extend `BaseFederatedClient`** — implement `compute_update()` and optionally override `evaluate()`:

```python
from sfl.client.base import BaseFederatedClient
from sfl.types import Parameters, Config, ClientUpdate

class MyClient(BaseFederatedClient):
    def __init__(self, client_id=0, device=None, **kwargs):
        resolved_device = device or "cpu"
        super().__init__(client_id=client_id, device=resolved_device)
        # Load your model, data, etc.

    def compute_update(self, parameters: Parameters, config: Config) -> ClientUpdate:
        # Load params → train → return updated params
        return updated_params, num_examples, metrics

    def evaluate(self, parameters: Parameters, config: Config):
        # Optional: override for real evaluation
        return loss, num_examples, {"eval_loss": loss}
```

3. **Compose config** with `FederationConfig`:

```python
from dataclasses import dataclass, field
from sfl.types import FederationConfig

@dataclass
class MyRunConfig:
    federation: FederationConfig = field(default_factory=FederationConfig)
    model_name: str = "my-model"
    # your hyperparams...
```

4. **Create a runner** in `jobs/` and register Flower apps in `__init__.py`

5. **Add tests** in `tests/test_your_app_*.py`

---

## Troubleshooting

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

## Resources

- [NVFlare Documentation](https://nvflare.readthedocs.io/)
- [Flower Documentation](https://flower.ai/docs/)
- [NVFlare + Flower Integration Guide](https://nvflare.readthedocs.io/en/main/hello-world/hello-flower/)
- [NVFlare POC Mode Guide](https://nvflare.readthedocs.io/en/main/getting_started.html#poc-mode)
- [NVFlare Production Deployment](https://nvflare.readthedocs.io/en/main/programming_guide/provisioning_tool.html)

---

## Security Features (Production Mode)

### Differential Privacy
- **Server-side DP** — Gaussian noise added to aggregate (Flower strategy wrapper)
- **Client-side DP-SGD** — per-example gradient clipping via Opacus
- **AutoClip** — automatic gradient normalization, eliminating clipping norm tuning (Li et al., NeurIPS 2023)
- **Ghost Clipping** — memory-efficient two-pass DP-SGD, O(B+P) instead of O(B*P) (Li et al., 2022)
- **PLD/PRV accounting** — tight (ε,δ)-DP tracking via `dp-accounting` (PLD) or `prv_accountant` (PRV with error bounds)
- **Configurable PRV precision** — `eps_error` default 0.01 for ±1% epsilon accuracy (PR #36)
- **Shuffle-model amplification** — tighter central ε via anonymous channel (Feldman et al., 2021)
- **Automatic budget enforcement** — training auto-stops when ε exceeds budget
- **Adaptive clipping** — geometric norm update (Andrew et al. 2021) with optional noisy quantile
- **Per-layer clipping** — independent L2 clips per parameter tensor (Yu et al., ICLR 2022)
- **Joint composition** — server + client DP-SGD epsilon composed via sequential theorem
- **Auxiliary composition** — `compose_auxiliary()` for adaptive clip/SVT DP costs via ComposedDpEvent (PR #36)
- **Noise calibration** — `calibrate_gaussian_sigma()` computes minimal noise for target (ε,δ)

### Privacy Filters
- **PercentilePrivacy** — top-K% sparsification with optional calibrated noise
  - **Adaptive K accounting** — when K is data-dependent, epsilon is split ε/3 (selection) + 2ε/3 (noise) for formal DP guarantee. Use `fixed_k` for data-independent sensitivity and full epsilon allocation (PR #37)
- **SVT** — Sparse Vector Technique with optimal budget allocation (Lyu et al. 2017)
  - **Enhanced observability** — `svt_acceptance_rate` metric, low-acceptance warnings (PR #36)
- **Gradient Compression** — TopK / random masking with (ε,δ)-calibrated noise and error feedback
- **ExcludeVars** — zero out sensitive layers (embeddings, classifier heads)
- **Partial Freezing** — strip frozen layers from updates to reduce SecAgg overhead (Lambda-SecAgg)
  - **Shape reconstruction** — frozen layer shapes stored in `_frozen_shapes` metric for correct server-side restoration (PR #36)

### Privacy Auditing

- **PrivacyAuditor** — empirical DP validation via canary gradient analysis (PR #37)
  - `run_audit()` — basic canary-vs-random cosine similarity measurement
  - `run_pipeline_audit()` — sends canary gradients through actual Flower client mods (SVT, percentile, compression) and measures information leakage end-to-end
  - Exported from `sfl.privacy` for programmatic use

### Byzantine Robustness
- **Multi-Krum** — tolerates up to f Byzantine clients (Blanchard et al., NeurIPS 2017)
  - **JL projection** — CSRNG-seeded random projection for high-dimensional updates, preventing adversary pre-computation (PR #36)
  - **Update norm verification** — `verify_update_norms()` rejects oversized updates server-side (PR #36)
- **Trimmed Mean** — coordinate-wise outlier trimming (Yin et al., ICML 2018)
- **FoundationFL** — trust scoring via cosine similarity to root dataset (NDSS 2025)
  - **Mandatory root update** — requires `root_update` or explicit `allow_untrusted_reference=True` to prevent Byzantine majority from manipulating client mean (PR #37)

### Secure Aggregation & Encryption
- **SecAgg+** — Flower's secret-sharing protocol; server sees only the aggregate
  - **Threshold enforcement** — `reconstruction_threshold < ceil(2*num_shares/3)` now raises `ValueError` instead of warning (PR #36)
- **Partial Freezing** — reduces SecAgg cost by sending only trainable layers
- **Homomorphic Encryption** — TenSEAL CKKS for encrypted aggregation (demo-scale)

### TLS/SSL Encryption
- All communication encrypted with certificates
- Mutual TLS (mTLS) authentication

### Authorization
- Role-based access control (RBAC)
- Job-level permissions

### Audit Logging
- All actions logged
- Compliance-ready audit trail

---

## Recommended Progression

1. **Start**: Use `SimEnv` mode (`python jobs/runner.py` or `python jobs/esm2_runner.py`) for development
2. **Test**: Use `POC mode` (`python jobs/poc_runner.py`) for integration testing
3. **Stage**: Use `Provision` mode for staging environment
4. **Deploy**: Full production with TLS + monitoring

---

## License

MIT License - See LICENSE file for details.
