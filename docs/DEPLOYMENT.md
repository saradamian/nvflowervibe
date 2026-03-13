# SFL Deployment Guide

This guide covers all deployment modes for the SFL (Simple Federated Learning) demo, from local development to production.

## Prerequisites

- Python 3.10+
- NVFlare 2.7.1
- Flower 1.17.0
- Click >= 8.1.0, < 8.2.0 (see [INSTALL_NOTE.md](INSTALL_NOTE.md) for details)

For ESM2 federated training, additionally:
- PyTorch >= 2.0.0 (CUDA recommended for GPU acceleration)
- Transformers >= 4.36.0
- Datasets >= 2.14.0

```bash
pip install -r requirements.txt
```

---

## 1. SimEnv (Single-Process Simulation)

**Best for:** Development, debugging, CI/CD

SimEnv runs the entire federation (server + all clients) in a single process. This is the fastest way to iterate.

### Quick Start

```bash
# Default: 2 clients, 1 round
python jobs/runner.py

# Custom settings
python jobs/runner.py --num-clients 4 --num-rounds 3

# With config file
python jobs/runner.py --config config/default.yaml

# Verbose logging
python jobs/runner.py --verbose
```

### How It Works

1. The runner loads configuration (YAML → ENV → CLI priority)
2. Creates an NVFlare `FlowerRecipe` pointing to the Flower app in `pyproject.toml`
3. Creates a `SimEnv` with the specified number of clients/threads
4. Executes the recipe — NVFlare orchestrates the Flower server and clients in-process

### Configuration

All settings can be controlled via:
- `config/default.yaml` — persistent defaults
- Environment variables with `SFL_` prefix (e.g., `SFL_NUM_CLIENTS=4`)
- CLI arguments (highest priority)

See `python jobs/runner.py --help` for all options.

---

## 2. Pure Flower Simulation (Fallback)

**Best for:** Testing without NVFlare, isolating Flower-specific issues

If NVFlare has compatibility issues, you can run the Flower simulation directly:

```bash
python jobs/flower_runner.py --num-clients 2 --num-rounds 1
```

This bypasses NVFlare entirely and uses Flower's `run_simulation()` API directly with Ray as a backend.

---

## 3. POC Mode (Multi-Process Local)

**Best for:** Integration testing, simulating real network behavior locally

POC mode runs the server and each client as separate OS processes on your local machine, communicating over localhost.

### Full Workflow

```bash
# Step 1: Prepare — creates workspace at /tmp/nvflare/poc/
python jobs/poc_runner.py prepare --num-clients 4 --clean

# Step 2: Start all services (server + clients)
python jobs/poc_runner.py start

# Step 3: Submit job via admin console
python jobs/poc_runner.py submit

# Step 4: Monitor
python jobs/poc_runner.py status

# Step 5: Stop
python jobs/poc_runner.py stop

# Step 6: Clean up
python jobs/poc_runner.py clean
```

### What `prepare` Does

1. Calls `nvflare poc prepare -n <num_clients>` to scaffold the workspace
2. Runs `scripts/generate_nvflare_job.py` to convert the Flower app into an NVFlare job
3. Places the job in `/tmp/nvflare/poc/jobs/sfl-job/`

### Workspace Structure

After preparation, the POC workspace looks like:

```
/tmp/nvflare/poc/
├── server/           # FL server process
├── site-1/           # Client 1
├── site-2/           # Client 2
├── ...
├── admin/            # Admin console
├── jobs/
│   └── sfl-job/      # Generated NVFlare job
└── transfer/         # File transfer staging
```

### Starting Components Individually

```bash
# Start only the server
python jobs/poc_runner.py start --component server

# Start a specific client
python jobs/poc_runner.py start --component site-1

# Start admin console
python jobs/poc_runner.py start --component admin
```

### Manual Job Submission

From the admin console:

```
> submit_job /tmp/nvflare/poc/jobs/sfl-job/sfl-federated-sum
> list_jobs
> check_status server
> check_status client
> abort_job <job_id>
> bye
```

---

## 4. Production Mode (Provisioned)

**Best for:** Real multi-machine deployments with TLS and authentication

### Step 1: Generate Project Template

```bash
nvflare provision -g
```

### Step 2: Configure `project.yml`

```yaml
api_version: 3
name: sfl-production
description: SFL Federated Learning Production Setup

participants:
  - name: overseer
    type: overseer
    org: sfl-org
    protocol: https
    api_root: /api/v1
    port: 8443

  - name: sfl-server
    type: server
    org: sfl-org
    fed_learn_port: 8002
    admin_port: 8003
    enable_byoc: true

  - name: site-a
    type: client
    org: org-a

  - name: site-b
    type: client
    org: org-b

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

### Step 3: Provision

```bash
nvflare provision -p project.yml -w ./workspace
```

This generates startup kits with TLS certificates for each participant.

### Step 4: Distribute and Start

**Server:**
```bash
cd workspace/sfl-server/startup && ./start.sh
```

**Each Client:**
```bash
cd workspace/site-a/startup && ./start.sh
```

**Admin:**
```bash
cd workspace/admin@sfl.org/startup && ./fl_admin.sh
# > submit_job /path/to/sfl/job
```

---

## 5. Docker Deployment

### Dockerfile (Server/Client)

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY config/ config/
COPY pyproject.toml .

# For server
# CMD ["python", "-m", "nvflare.private.fed.app.server.server_train"]

# For client
# CMD ["python", "-m", "nvflare.private.fed.app.client.client_train"]
```

### Docker Compose

```yaml
version: "3.8"
services:
  server:
    build: .
    ports:
      - "8002:8002"
      - "8003:8003"
    volumes:
      - ./workspace/sfl-server:/workspace
    command: /workspace/startup/start.sh

  site-1:
    build: .
    volumes:
      - ./workspace/site-1:/workspace
    command: /workspace/startup/start.sh
    depends_on:
      - server

  site-2:
    build: .
    volumes:
      - ./workspace/site-2:/workspace
    command: /workspace/startup/start.sh
    depends_on:
      - server
```

---

## Port Reference

| Service | Port | Protocol |
|---------|------|----------|
| FL Server (federation) | 8002 | gRPC |
| FL Server (admin) | 8003 | gRPC |
| Overseer | 8443 | HTTPS |

---

## Security Notes

- **SimEnv / POC:** No encryption — suitable for development only.
- **Provisioned:** TLS certificates auto-generated. Distribute startup kits securely.
- **Production:** Use `CertBuilder` for proper PKI. Rotate certificates regularly.
- Client secrets are transmitted as Flower parameters — for real workloads, consider differential privacy or secure aggregation extensions.

---

## ESM2 Federated Training

The ESM2 module supports both Flower and NVFlare backends.

### Flower Backend (Recommended for Development)

```bash
# Quick demo — 2 clients, 3 rounds
python jobs/esm2_runner.py

# Custom settings
python jobs/esm2_runner.py --num-clients 4 --num-rounds 5 --learning-rate 1e-4

# Larger model
python jobs/esm2_runner.py --model facebook/esm2_t12_35M_UR50D
```

GPU auto-detection: CUDA GPUs are automatically detected and allocated evenly across simulation clients.

### NVFlare Backend

```bash
python jobs/esm2_runner.py --backend nvflare --num-clients 2 --num-rounds 2
```

This stages the source code into a temp directory, generates an ESM2-specific `pyproject.toml` pointing to `sfl.esm2:server_app` / `sfl.esm2:client_app`, and runs via NVFlare's `FlowerRecipe` + `SimEnv`.

### Performance Notes

| Backend | 2 clients, 2 rounds | Notes |
|---------|---------------------|-------|
| Flower | ~10s | Lightweight Ray actors, minimal overhead |
| NVFlare | ~38s | gRPC infrastructure, job staging, process management |
