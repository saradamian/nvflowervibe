#!/bin/bash
#SBATCH --job-name=sfl-server
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/server_%j.out
#SBATCH --error=logs/server_%j.err
#
# SFL Federated Learning Server — SLURM Job Script
#
# Starts the Flower gRPC server and waits for clients.
# Uses mTLS when certificates exist in SFL_CERTS_DIR, otherwise --insecure.
#
# Usage:
#   sbatch submit_server.sbatch
#   sbatch --export=ALL,SFL_NUM_ROUNDS=50,SFL_NUM_CLIENTS=8 submit_server.sbatch
#
# Customize the variables below or pass via --export.

# ── Configuration (override via SLURM --export or environment) ──────
SFL_NUM_ROUNDS="${SFL_NUM_ROUNDS:-20}"
SFL_NUM_CLIENTS="${SFL_NUM_CLIENTS:-4}"
SFL_MODEL="${SFL_MODEL:-facebook/esm2_t6_8M_UR50D}"
SFL_CERTS_DIR="${SFL_CERTS_DIR:-/scratch/$USER/sfl-certs}"
SFL_CHECKPOINT_DIR="${SFL_CHECKPOINT_DIR:-/scratch/$USER/sfl-checkpoints/$SLURM_JOB_ID}"
SFL_METRICS_DIR="${SFL_METRICS_DIR:-/scratch/$USER/sfl-metrics/$SLURM_JOB_ID}"
SFL_RESUME="${SFL_RESUME:-false}"
SFL_SERVER_ADDRESS="${SFL_SERVER_ADDRESS:-0.0.0.0:8443}"

# ── Environment setup (CUSTOMIZE FOR YOUR CENTER) ──────────────────
# module load cuda/12.0 python/3.10
# source /path/to/your/venv/bin/activate

# Ensure log directory exists
mkdir -p logs "$SFL_CHECKPOINT_DIR" "$SFL_METRICS_DIR"

# ── Export server address for clients ──────────────────────────────
SERVER_HOSTNAME=$(hostname -f)
echo "${SERVER_HOSTNAME}:8443" > "${SFL_CERTS_DIR}/server_address.txt"
echo "Server address: ${SERVER_HOSTNAME}:8443"

# ── mTLS certificates ─────────────────────────────────────────────
export SFL_TLS_CA_CERT="${SFL_CERTS_DIR}/ca.pem"
export SFL_TLS_SERVER_CERT="${SFL_CERTS_DIR}/server.pem"
export SFL_TLS_SERVER_KEY="${SFL_CERTS_DIR}/server.key"

# ── Privacy/aggregation config (shared with clients via env) ──────
export SFL_DP_ENABLED=true
export SFL_DP_NOISE=0.8
export SFL_DP_CLIP=5.0
export SFL_DP_MODE=server
export SFL_DP_ADAPTIVE_CLIP=true
export SFL_SECAGG_ENABLED=true
export SFL_AGGREGATION=krum
export SFL_KRUM_BYZANTINE=1
export SFL_CHECKPOINT_DIR
export SFL_METRICS_DIR
export SFL_METRICS_FORMAT=all

# ── Launch Flower gRPC server ─────────────────────────────────────
#
# In multi-node HPC mode, the server runs as a Flower gRPC server
# via flower-server-app. Clients connect via flower-client-app
# (see submit_client.sbatch).
#
# The server_app reads SFL_* env vars for aggregation strategy,
# DP wrapping, checkpoint, and metrics configuration.
#
echo "Starting SFL Flower server: ${SFL_NUM_CLIENTS} clients, ${SFL_NUM_ROUNDS} rounds"
echo "Model: ${SFL_MODEL}"
echo "Checkpoint dir: ${SFL_CHECKPOINT_DIR}"
echo "Metrics dir: ${SFL_METRICS_DIR}"

# Build TLS flags — use mTLS when certs exist, --insecure otherwise
TLS_FLAGS=""
if [ -f "$SFL_TLS_CA_CERT" ] && [ -f "$SFL_TLS_SERVER_CERT" ] && [ -f "$SFL_TLS_SERVER_KEY" ]; then
    TLS_FLAGS="--ssl-ca-certfile $SFL_TLS_CA_CERT --ssl-certfile $SFL_TLS_SERVER_CERT --ssl-keyfile $SFL_TLS_SERVER_KEY"
    echo "TLS: mTLS enabled"
else
    TLS_FLAGS="--insecure"
    echo "WARNING: TLS certificates not found — running insecure (dev only)"
fi
echo "---"

flower-server-app sfl.esm2:server_app \
    $TLS_FLAGS \
    --server-address "$SFL_SERVER_ADDRESS" \
    --run-config "num-server-rounds=${SFL_NUM_ROUNDS} num-clients=${SFL_NUM_CLIENTS} esm2-model=${SFL_MODEL}" \
    2>&1

EXIT_CODE=$?
echo "Server exited with code ${EXIT_CODE}"
exit $EXIT_CODE
