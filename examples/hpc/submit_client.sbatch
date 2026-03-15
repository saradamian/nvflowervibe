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
# Launches Flower SuperNode + ClientApp on one node.
# Connects to the SuperLink started by submit_server.sbatch.
#
# Flower 1.17.0 multi-node architecture:
#   SuperNode  — per-client node (connects to SuperLink's Fleet API)
#   ClientApp  — runs client_fn logic (connects to SuperNode locally)
#
# Submit as an array job: sbatch --array=0-7 submit_client.sbatch
#
# Usage:
#   sbatch --array=0-3 submit_client.sbatch
#   sbatch --array=0-7 --export=ALL,SFL_SERVER_ADDRESS=server01:9092 submit_client.sbatch

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
echo "Connecting to SuperLink at ${SFL_SERVER_ADDRESS} as client ${SFL_CLIENT_ID}"

# ── GPU diagnostics ───────────────────────────────────────────────
echo "Node: $(hostname)"
echo "GPUs:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "  No NVIDIA GPUs"

# ── Build TLS flags ──────────────────────────────────────────────
# TLS is required by default. Use SFL_INSECURE=true to opt out (dev only).
SFL_TLS_CA_CERT="${SFL_CERTS_DIR}/ca.pem"

SUPERNODE_TLS_FLAGS=""
if [ "${SFL_INSECURE:-false}" = "true" ]; then
    SUPERNODE_TLS_FLAGS="--insecure"
    echo "WARNING: Running insecure (SFL_INSECURE=true) — dev only"
elif [ -f "$SFL_TLS_CA_CERT" ]; then
    SUPERNODE_TLS_FLAGS="--root-certificates $SFL_TLS_CA_CERT"
    echo "TLS: enabled (root cert: $SFL_TLS_CA_CERT)"
else
    echo "ERROR: TLS CA certificate not found at ${SFL_TLS_CA_CERT} and SFL_INSECURE is not set."
    echo "  Run: ./examples/hpc/generate_certs.sh ${SFL_CERTS_DIR}"
    echo "  Or set SFL_INSECURE=true for development (NOT recommended)."
    exit 1
fi

# ── Launch SuperNode (background) ────────────────────────────────
# SuperNode connects to the SuperLink's Fleet API and manages the
# local ClientApp lifecycle.
echo "Client ${SFL_CLIENT_ID} starting on $(hostname)"

flower-supernode \
    $SUPERNODE_TLS_FLAGS \
    --superlink "$SFL_SERVER_ADDRESS" \
    --node-config "partition-id=${SFL_CLIENT_ID}" \
    --isolation process \
    2>&1 | sed 's/^/[supernode] /' &
SUPERNODE_PID=$!

# Wait for SuperNode to start
sleep 3
if ! kill -0 $SUPERNODE_PID 2>/dev/null; then
    echo "ERROR: SuperNode failed to start"
    exit 1
fi
echo "SuperNode started (PID ${SUPERNODE_PID})"

# ── Launch ClientApp (foreground) ────────────────────────────────
# ClientApp connects to SuperNode's ClientAppIo API (localhost:9094)
# and runs the ESM2 client_fn. SFL_* env vars are read by
# auto_build_client_mods() to reconstruct privacy mods.
flwr-clientapp sfl.esm2:client_app \
    --insecure \
    2>&1 | sed 's/^/[clientapp] /'

EXIT_CODE=$?

# Clean up SuperNode
kill $SUPERNODE_PID 2>/dev/null
wait $SUPERNODE_PID 2>/dev/null

echo "Client ${SFL_CLIENT_ID} exited with code ${EXIT_CODE}"
exit $EXIT_CODE
