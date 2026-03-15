#!/bin/bash
#SBATCH --job-name=sfl-client
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/client_%j_%a.out
#SBATCH --error=logs/client_%j_%a.err
#
# SFL Federated Learning Client — SLURM Job Script
#
# Connects to the FL server and participates in training.
# Submit as an array job: sbatch --array=0-7 submit_client.sbatch
#
# Usage:
#   sbatch --array=0-3 submit_client.sbatch
#   sbatch --array=0-7 --export=ALL,SFL_SERVER_ADDRESS=server01:8443 submit_client.sbatch

# ── Configuration ──────────────────────────────────────────────────
SFL_CERTS_DIR="${SFL_CERTS_DIR:-/scratch/$USER/sfl-certs}"
SFL_CLIENT_ID="${SLURM_ARRAY_TASK_ID:-0}"

# ── Environment setup (CUSTOMIZE FOR YOUR CENTER) ──────────────────
# module load cuda/12.0 python/3.10
# source /path/to/your/venv/bin/activate

mkdir -p logs

# ── Read server address ───────────────────────────────────────────
SERVER_ADDR_FILE="${SFL_CERTS_DIR}/server_address.txt"
MAX_WAIT=300  # Wait up to 5 minutes for server to start
WAITED=0

while [ ! -f "$SERVER_ADDR_FILE" ] && [ $WAITED -lt $MAX_WAIT ]; do
    echo "Waiting for server to start (${WAITED}s / ${MAX_WAIT}s)..."
    sleep 5
    WAITED=$((WAITED + 5))
done

if [ ! -f "$SERVER_ADDR_FILE" ]; then
    echo "ERROR: Server address file not found after ${MAX_WAIT}s"
    exit 1
fi

SFL_SERVER_ADDRESS="${SFL_SERVER_ADDRESS:-$(cat "$SERVER_ADDR_FILE")}"
echo "Connecting to server at ${SFL_SERVER_ADDRESS} as client ${SFL_CLIENT_ID}"

# ── mTLS certificates ─────────────────────────────────────────────
export SFL_TLS_CA_CERT="${SFL_CERTS_DIR}/ca.pem"
export SFL_TLS_CLIENT_CERT="${SFL_CERTS_DIR}/client.pem"
export SFL_TLS_CLIENT_KEY="${SFL_CERTS_DIR}/client.key"

# ── GPU diagnostics ───────────────────────────────────────────────
echo "Node: $(hostname)"
echo "GPUs:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  No NVIDIA GPUs"

# ── Launch client ─────────────────────────────────────────────────
# In multi-node mode, clients connect to the server via Flower's
# gRPC channel. The ClientApp is configured identically to simulation
# mode — the same privacy mods, same training code.
#
# NOTE: For multi-node Flower (non-simulation), you would use:
#   flower-client-app sfl.esm2:client_app \
#       --server-address "$SFL_SERVER_ADDRESS" \
#       --root-certificates "$SFL_TLS_CA_CERT"
#
# For NVFlare orchestration, this is handled by the NVFlare job
# submission system instead.

echo "Client ${SFL_CLIENT_ID} ready on $(hostname)"
echo "Waiting for server to assign work..."

flower-client-app sfl.esm2:client_app \
    --server-address "$SFL_SERVER_ADDRESS" \
    --root-certificates "$SFL_TLS_CA_CERT" \
    2>&1

EXIT_CODE=$?
echo "Client ${SFL_CLIENT_ID} exited with code ${EXIT_CODE}"
exit $EXIT_CODE
