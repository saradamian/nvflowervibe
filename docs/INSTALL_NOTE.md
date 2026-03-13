# Installation & Troubleshooting

## Quick Install

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (order matters)
pip install -U pip wheel setuptools
pip install -r requirements.txt
```

Or use the setup script:

```bash
./scripts/setup.sh
```

---

## Version Requirements

The NVFlare + Flower integration has strict version dependencies:

| Package | Required Version | Why |
| ------- | ---------------- | --- |
| `nvflare` | 2.7.1 | Flower integration via `FlowerRecipe` |
| `flwr[simulation]` | 1.17.0 | Compatible CLI and `ServerApp`/`ClientApp` API |
| `click` | >= 8.1.0, < 8.2.0 | Click 8.2+ breaks Typer (used internally by Flower) |
| `typer` | >= 0.12.0 | CLI framework used by Flower |
| `numpy` | >= 1.21.0, < 2.0.0 | NumPy 2.0 has breaking API changes |
| `ray` | >= 2.31.0 | Simulation backend for Flower |

### Why These Exact Versions?

- **NVFlare 2.7.1** introduced `FlowerRecipe` for Flower integration. Earlier versions don't support it.
- **Flower 1.17.0** is the version NVFlare 2.7.1 was tested against. Older versions lack `--serverappio-api-address` support.
- **Click < 8.2** is critical because Click 8.2 changed internal APIs that Typer depends on, causing CLI crashes.

---

## Common Issues

### 1. `"--serverappio-api-address" is not a recognized option`

**Cause:** Flower version is too old.

**Fix:**

```bash
pip install flwr[simulation]==1.17.0
```

### 2. CLI crashes with Typer traceback

**Cause:** Click 8.2+ breaks Typer's internal usage of Click APIs.

**Symptoms:**

```text
TypeError: __init__() got an unexpected keyword argument 'is_eager'
```

**Fix:**

```bash
pip install "click>=8.1.0,<8.2.0"
```

### 3. `"--format" is not a valid option for flwr run`

**Cause:** Flower version too old or too new.

**Fix:**

```bash
pip install flwr[simulation]==1.17.0
```

### 4. `ModuleNotFoundError: No module named 'nvflare.app_opt.flower'`

**Cause:** NVFlare installed without Flower optional dependency, or wrong NVFlare version.

**Fix:**

```bash
pip install nvflare==2.7.1
```

### 5. POC services fail to start

**Cause:** Ports 8002/8003 already in use, or POC workspace not prepared.

**Fix:**

```bash
# Check for port conflicts
lsof -i :8002
lsof -i :8003

# Kill conflicting processes if needed, then:
python jobs/poc_runner.py clean
python jobs/poc_runner.py prepare --num-clients 2 --clean
python jobs/poc_runner.py start
```

### 6. `ImportError: cannot import name 'run_simulation' from 'flwr.simulation'`

**Cause:** Flower installed without simulation extras.

**Fix:**

```bash
pip install "flwr[simulation]==1.17.0"
```

### 7. Ray fails to initialize

**Cause:** Insufficient system resources or conflicting Ray instances.

**Fix:**

```bash
# Stop existing Ray instances
ray stop

# Reinstall Ray
pip install "ray>=2.31.0"
```

### 8. `rich` not found (degraded logging output)

**Cause:** `rich` package not installed. The logging system falls back to simple format automatically, but output won't be colorized.

**Fix:**

```bash
pip install "rich>=13.0.0"
```

---

## Platform Notes

### macOS (Apple Silicon)

Some dependencies may need Rosetta or specific builds:

```bash
# If Ray fails on M1/M2
pip install ray --no-cache-dir

# If grpcio fails
pip install grpcio --no-binary :all:
```

### Linux

Ensure you have the required system libraries:

```bash
# Ubuntu/Debian
sudo apt-get install python3-dev build-essential

# RHEL/CentOS
sudo yum install python3-devel gcc
```

### Windows

NVFlare POC mode is best supported on Linux/macOS. On Windows:

- Use WSL2 for POC and production modes
- SimEnv mode works natively on Windows

---

## Verifying Installation

Run the test suite to verify everything is correctly installed:

```bash
# Run unit tests
python -m pytest tests/ -v

# Quick smoke test — run SimEnv simulation
python jobs/runner.py --num-clients 2 --num-rounds 1

# Verify NVFlare is available
python -c "from nvflare.app_opt.flower.recipe import FlowerRecipe; print('NVFlare OK')"

# Verify Flower is available
python -c "from flwr.simulation import run_simulation; print('Flower OK')"
```

Expected output from a successful simulation run:

```text
[server] round=1 client_vals=[7.0, 8.0] federated_sum=15.0
```

---

## Upgrading

When upgrading dependencies, always pin exact versions to avoid compatibility breaks:

```bash
# Check current versions
pip list | grep -E "nvflare|flwr|click|typer"

# Upgrade carefully
pip install nvflare==2.7.1 flwr[simulation]==1.17.0 "click>=8.1.0,<8.2.0"
```

Do NOT run `pip install --upgrade` without version pins — this can pull in incompatible versions of Click or Flower.
