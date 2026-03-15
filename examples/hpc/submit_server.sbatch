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
# Launches Flower SuperLink + ServerApp on one node.
# Clients connect via flower-supernode (see submit_client.sbatch).
#
# Flower 1.17.0 multi-node architecture:
#   SuperLink  — central coordinator (fleet API, TLS, scheduling)
#   ServerApp  — runs server_fn logic (connects to SuperLink locally)
#
# Usage:
#   sbatch submit_server.sbatch
#   sbatch --export=ALL,SFL_NUM_ROUNDS=50,SFL_NUM_CLIENTS=8 submit_server.sbatch
#
# Customize the variables below or pass via --export.

# ── Configuration (override via SLURM --export or environment) ──────
export SFL_NUM_ROUNDS="${SFL_NUM_ROUNDS:-20}"
export SFL_NUM_CLIENTS="${SFL_NUM_CLIENTS:-4}"
export SFL_MODEL="${SFL_MODEL:-facebook/esm2_t6_8M_UR50D}"
SFL_CERTS_DIR="${SFL_CERTS_DIR:-/scratch/$USER/sfl-certs}"
export SFL_CHECKPOINT_DIR="${SFL_CHECKPOINT_DIR:-/scratch/$USER/sfl-checkpoints/$SLURM_JOB_ID}"
export SFL_METRICS_DIR="${SFL_METRICS_DIR:-/scratch/$USER/sfl-metrics/$SLURM_JOB_ID}"
export SFL_RESUME="${SFL_RESUME:-false}"
FLEET_API_ADDRESS="${SFL_SERVER_ADDRESS:-0.0.0.0:9092}"

# ── Environment setup (CUSTOMIZE FOR YOUR CENTER) ──────────────────
# module load cuda/12.0 python/3.10
# source /path/to/your/venv/bin/activate

# Ensure log directory exists
mkdir -p logs "$SFL_CHECKPOINT_DIR" "$SFL_METRICS_DIR"

# ── Export server address for clients ──────────────────────────────
SERVER_HOSTNAME=$(hostname -f)
FLEET_PORT="${FLEET_API_ADDRESS##*:}"
echo "${SERVER_HOSTNAME}:${FLEET_PORT}" > "${SFL_CERTS_DIR}/server_address.txt"
echo "Server address: ${SERVER_HOSTNAME}:${FLEET_PORT}"

# ── Privacy/aggregation config (exported so server_fn reads them) ──
# These are examples — customize or pass via --export from launch_federation.sh.
# Server-side vars are read by server_fn via SFL_* env var fallbacks.
# Client-side vars must be propagated to client jobs separately
# (launch_federation.sh handles this via SLURM --export).
export SFL_DP_ENABLED="${SFL_DP_ENABLED:-false}"
export SFL_DP_NOISE="${SFL_DP_NOISE:-0.8}"
export SFL_DP_CLIP="${SFL_DP_CLIP:-5.0}"
export SFL_DP_MODE="${SFL_DP_MODE:-server}"
export SFL_DP_ADAPTIVE_CLIP="${SFL_DP_ADAPTIVE_CLIP:-false}"
export SFL_SECAGG_ENABLED="${SFL_SECAGG_ENABLED:-false}"
export SFL_AGGREGATION="${SFL_AGGREGATION:-fedavg}"
export SFL_KRUM_BYZANTINE="${SFL_KRUM_BYZANTINE:-1}"
export SFL_METRICS_FORMAT="${SFL_METRICS_FORMAT:-csv}"

echo "Starting SFL Flower server: ${SFL_NUM_CLIENTS} clients, ${SFL_NUM_ROUNDS} rounds"
echo "Model: ${SFL_MODEL}"
echo "Checkpoint dir: ${SFL_CHECKPOINT_DIR}"
echo "Metrics dir: ${SFL_METRICS_DIR}"

# ── Build TLS flags ──────────────────────────────────────────────
# TLS is required by default. Use SFL_INSECURE=true to opt out (dev only).
SFL_TLS_CA_CERT="${SFL_CERTS_DIR}/ca.pem"
SFL_TLS_SERVER_CERT="${SFL_CERTS_DIR}/server.pem"
SFL_TLS_SERVER_KEY="${SFL_CERTS_DIR}/server.key"

SUPERLINK_TLS_FLAGS=""
if [ "${SFL_INSECURE:-false}" = "true" ]; then
    SUPERLINK_TLS_FLAGS="--insecure"
    echo "WARNING: Running insecure (SFL_INSECURE=true) — dev only"
elif [ -f "$SFL_TLS_CA_CERT" ] && [ -f "$SFL_TLS_SERVER_CERT" ] && [ -f "$SFL_TLS_SERVER_KEY" ]; then
    SUPERLINK_TLS_FLAGS="--ssl-ca-certfile $SFL_TLS_CA_CERT --ssl-certfile $SFL_TLS_SERVER_CERT --ssl-keyfile $SFL_TLS_SERVER_KEY"
    echo "TLS: mTLS enabled"
else
    echo "ERROR: TLS certificates not found in ${SFL_CERTS_DIR} and SFL_INSECURE is not set."
    echo "  Run: ./examples/hpc/generate_certs.sh ${SFL_CERTS_DIR}"
    echo "  Or set SFL_INSECURE=true for development (NOT recommended)."
    exit 1
fi
echo "---"

# ── Launch SuperLink (background) ────────────────────────────────
# SuperLink is the central Flower coordinator. It handles the Fleet API
# (client connections), ServerAppIo API, and scheduling.
flower-superlink \
    $SUPERLINK_TLS_FLAGS \
    --fleet-api-address "$FLEET_API_ADDRESS" \
    --isolation process \
    2>&1 | sed 's/^/[superlink] /' &
SUPERLINK_PID=$!

# Wait for SuperLink to start
sleep 3
if ! kill -0 $SUPERLINK_PID 2>/dev/null; then
    echo "ERROR: SuperLink failed to start"
    exit 1
fi
echo "SuperLink started (PID ${SUPERLINK_PID})"

# ── Launch ServerApp (foreground) ────────────────────────────────
# ServerApp connects to SuperLink's ServerAppIo API (localhost:9091)
# and runs the ESM2 server_fn which reads SFL_* env vars.
flwr-serverapp sfl.esm2:server_app \
    --insecure \
    --run-once \
    2>&1 | sed 's/^/[serverapp] /'

EXIT_CODE=$?

# Clean up SuperLink
kill $SUPERLINK_PID 2>/dev/null
wait $SUPERLINK_PID 2>/dev/null

echo "Server exited with code ${EXIT_CODE}"
exit $EXIT_CODE
