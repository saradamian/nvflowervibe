# HPC Deployment Guide

Deploy SFL across multiple HPC centers with formal security guarantees — from hardware isolation to application-level differential privacy.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Security Layers                          │
│                                                             │
│  L1  Hardware    TEE / CVM (confidential computing)         │
│  L2  Network     mTLS on gRPC channel                       │
│  L3  Protocol    SecAgg+ (server never sees individual      │
│                  updates — information-theoretic guarantee)  │
│  L4  Aggregation Byzantine-robust (Multi-Krum / Trimmed     │
│                  Mean / FoundationFL)                        │
│  L5  Privacy     (ε,δ)-DP with PLD/PRV accounting          │
│  L6  Client      Per-example DP-SGD (Opacus)               │
│  L7  Comm        Gradient compression + error feedback       │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start (Single Node)

```bash
# Run 4-client simulation on one node
python jobs/esm2_runner.py \
    --num-clients 4 --num-rounds 10 \
    --dp --dp-noise 0.8 --dp-clip 5.0 \
    --secagg \
    --aggregation krum \
    --checkpoint-dir /scratch/$USER/checkpoints \
    --metrics-dir /scratch/$USER/metrics --metrics-format all
```

## Multi-Node SLURM Deployment

### File Layout

```
examples/hpc/
├── README.md                # This file
├── project.yml              # NVFlare provisioning config (customize hostnames)
├── nvflare_distributed.py   # NVFlare cross-site deployment helper
├── submit_server.sbatch     # SLURM job script for FL server
├── submit_client.sbatch     # SLURM job script for FL clients
├── launch_federation.sh     # One-command launcher for the full federation
├── generate_certs.sh        # Generate mTLS certificates for secure channel
└── monitor_training.sh      # Watch metrics and training progress
```

### Step 1: Generate TLS Certificates

```bash
# On the login node (shared filesystem)
./examples/hpc/generate_certs.sh /scratch/$USER/sfl-certs
```

This creates a self-signed CA + server/client certificates. For production, use your organization's PKI.

### Step 2: Launch the Federation

```bash
# Submit server + client jobs to SLURM
./examples/hpc/launch_federation.sh \
    --num-clients 8 \
    --num-rounds 50 \
    --partition gpu \
    --model facebook/esm2_t12_35M_UR50D \
    --dp --dp-noise 0.5 \
    --checkpoint-dir /scratch/$USER/checkpoints \
    --certs-dir /scratch/$USER/sfl-certs
```

Privacy/aggregation flags (`--dp`, `--secagg`, `--aggregation`, etc.) are translated to `SFL_*` env vars and propagated to **both** server and client jobs.

This submits:
1. A **server job** running `flower-superlink` (Fleet API + mTLS) + `flwr-serverapp` (ESM2 server logic)
2. N **client jobs** each running `flower-supernode` (connects to SuperLink) + `flwr-clientapp` (ESM2 client logic)

TLS is required by default. If certificates are missing, the scripts exit with an error. Set `--insecure` to opt out (development only).

### Step 3: Monitor

```bash
# Watch training progress
./examples/hpc/monitor_training.sh /scratch/$USER/metrics

# Check SLURM job status
squeue -u $USER -n sfl-server,sfl-client
```

## Environment Variables for HPC

| Variable | Purpose | Example |
|----------|---------|---------|
| `SFL_TLS_CA_CERT` | CA certificate path | `/scratch/certs/ca.pem` |
| `SFL_TLS_SERVER_CERT` | Server certificate | `/scratch/certs/server.pem` |
| `SFL_TLS_SERVER_KEY` | Server private key | `/scratch/certs/server.key` |
| `SFL_TLS_CLIENT_CERT` | Client certificate | `/scratch/certs/client.pem` |
| `SFL_TLS_CLIENT_KEY` | Client private key | `/scratch/certs/client.key` |
| `SFL_AUTH_TOKEN` | Bearer token (alternative to mTLS) | `my-secret-token` |
| `SFL_CHECKPOINT_DIR` | Round-level checkpoint directory | `/scratch/checkpoints` |
| `SFL_METRICS_DIR` | Metrics output directory | `/scratch/metrics` |
| `NCCL_SOCKET_IFNAME` | Network interface for NCCL | `ib0` (InfiniBand) |

## Wall-Time Safety

HPC jobs have wall-time limits. SFL handles this via:

1. **Round-level checkpointing** — after each round, model parameters + metrics are saved atomically
2. **Resume from checkpoint** — use `--resume` to continue from the last saved round. The server restores weights from the latest checkpoint **and** adjusts the remaining round count (e.g., if 15 of 50 rounds completed, it runs 35 more).
3. **Cleanup** — old checkpoints are pruned automatically (keeps last 3)

```bash
# Initial run (may be killed at wall-time)
./examples/hpc/launch_federation.sh --num-clients 4 --num-rounds 50

# Resume from where it stopped
./examples/hpc/launch_federation.sh --num-clients 4 --num-rounds 50 --resume
```

## Heterogeneous Clusters

Different HPC centers may have different hardware. SFL supports per-client resource allocation:

```bash
# Auto-detect GPUs on each node
python jobs/esm2_runner.py --num-clients 4

# Explicit: 2 CPUs + 0.5 GPU per client
python jobs/esm2_runner.py --num-clients 4 --client-cpus 2 --client-gpus 0.5

# Disable GPU auto-detection (CPU-only training)
python jobs/esm2_runner.py --num-clients 4 --no-auto-detect-gpu
```

## Security Checklist for Production

- [ ] **mTLS enabled** — `SFL_TLS_CA_CERT` set, certificates provisioned
- [ ] **SecAgg+ enabled** — `--secagg` flag, threshold ≥ 2/3 majority
- [ ] **DP accounting active** — `--dp` with budget cap (`--dp-max-epsilon`)
- [ ] **Byzantine robustness** — `--aggregation krum` or `trimmed-mean`
- [ ] **Server-side norm verification** — enabled by default in strategies
- [ ] **Checkpoint directory on shared filesystem** — `/scratch/` or `/gpfs/`
- [ ] **Metrics exported** — `--metrics-dir` for post-hoc analysis
- [ ] **CSRNG seeding** — all noise uses `secure_rng()` (enforced in code)
- [ ] **No `ExcludeVars` as privacy** — it's a communication optimization only
- [ ] **FoundationFL with `root_update`** — prevents Byzantine majority attacks

## Adapting for Your HPC Center

This example assumes SLURM. For other schedulers:

| Scheduler | Equivalent to `sbatch` | Node list variable |
|-----------|----------------------|-------------------|
| SLURM | `sbatch` | `$SLURM_NODELIST` |
| PBS/Torque | `qsub` | `$PBS_NODEFILE` |
| LSF | `bsub` | `$LSB_HOSTS` |
| SGE | `qsub` | `$PE_HOSTFILE` |

Replace the SLURM-specific lines in the scripts with your scheduler's equivalents. The SFL framework itself is scheduler-agnostic — it only needs a network address to connect clients to the server.

## NVFlare Cross-Site Deployment

For production cross-site federations, NVFlare provides the orchestration layer — handling provisioning (TLS certs, identity), job lifecycle, site policies, and fault tolerance. Flower runs inside NVFlare.

### Step 1: Customize `project.yml`

Edit `examples/hpc/project.yml` with your actual hostnames:

```yaml
participants:
  - name: server
    type: server
    org: sfl
    host: your-server-hostname.example.com  # CUSTOMIZE

  - name: site-local
    type: client
    org: sfl
    host: my-workstation.local              # CUSTOMIZE

  - name: site-hpc
    type: client
    org: sfl
    host: login01.hpc.example.edu           # CUSTOMIZE
```

### Step 2: Provision Startup Kits

```bash
python examples/hpc/nvflare_distributed.py provision
```

This generates a startup kit per participant containing TLS certificates, authorization policies, and launch scripts.

### Step 3: Distribute Kits to Sites

```bash
python examples/hpc/nvflare_distributed.py distribute \
    --site-local localhost \
    --site-hpc login01.hpc.example.edu
```

### Step 4: Launch NVFlare on Each Site

```bash
# On the server machine:
python examples/hpc/nvflare_distributed.py launch --role server

# On each client site:
python examples/hpc/nvflare_distributed.py launch --role client --site site-hpc

# Start admin console (for job submission):
python examples/hpc/nvflare_distributed.py launch --role admin
```

### Step 5: Submit an SFL Job

```bash
# Package and submit ESM2 federated training
python examples/hpc/nvflare_distributed.py submit-job \
    --runner esm2 \
    --num-rounds 50 \
    --num-clients 2 \
    --dp --dp-noise 0.5
```

### NVFlare + SLURM

On HPC clusters, launch NVFlare participants as SLURM jobs:

```bash
# Server
sbatch --wrap="python examples/hpc/nvflare_distributed.py launch --role server" \
    --job-name=nvflare-server --time=8:00:00

# Client (on each HPC site)
sbatch --wrap="python examples/hpc/nvflare_distributed.py launch --role client --site site-hpc" \
    --job-name=nvflare-client --gres=gpu:1 --time=8:00:00
```

### Why NVFlare + Flower?

| Layer | Responsibility |
|-------|---------------|
| **NVFlare** | Provisioning, identity/TLS, job lifecycle, site policies, fault tolerance |
| **Flower** | FL protocol (FedAvg, strategies), client/server communication, mods pipeline |
| **SFL** | Privacy mods (DP, SecAgg, filters), robust aggregation, domain models (ESM2, LLM) |

<!-- TODO: Add your center-specific details here:
  - Module loads (e.g., module load cuda/12.0)
  - Conda/venv activation
  - InfiniBand settings
  - Shared filesystem paths
  - Partition names and GPU types
  - Wall-time limits
  - Account/project charge codes
-->
