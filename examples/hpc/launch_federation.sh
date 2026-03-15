#!/bin/bash
set -euo pipefail

# Launch a complete SFL federation on SLURM.
#
# Usage:
#   ./launch_federation.sh --num-clients 8 --num-rounds 50
#   ./launch_federation.sh --num-clients 4 --partition a100 --certs-dir /scratch/certs
#
# This script:
#   1. Generates TLS certificates (if not already present)
#   2. Submits the server job
#   3. Submits client array jobs (one per client)
#   4. Prints monitoring commands

# ── Defaults ──────────────────────────────────────────────────────
NUM_CLIENTS=4
NUM_ROUNDS=20
PARTITION="gpu"
MODEL="facebook/esm2_t6_8M_UR50D"
CERTS_DIR="/scratch/$USER/sfl-certs"
CHECKPOINT_DIR="/scratch/$USER/sfl-checkpoints"
METRICS_DIR="/scratch/$USER/sfl-metrics"
RESUME=false
EXTRA_ARGS=""

# ── Parse arguments ───────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-clients)  NUM_CLIENTS="$2";   shift 2 ;;
        --num-rounds)   NUM_ROUNDS="$2";    shift 2 ;;
        --partition)    PARTITION="$2";      shift 2 ;;
        --model)        MODEL="$2";         shift 2 ;;
        --certs-dir)    CERTS_DIR="$2";     shift 2 ;;
        --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
        --metrics-dir)  METRICS_DIR="$2";   shift 2 ;;
        --resume)       RESUME=true;        shift ;;
        *)              EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "============================================"
echo "  SFL Federation Launcher"
echo "============================================"
echo "  Clients:     $NUM_CLIENTS"
echo "  Rounds:      $NUM_ROUNDS"
echo "  Partition:   $PARTITION"
echo "  Model:       $MODEL"
echo "  Certs:       $CERTS_DIR"
echo "  Checkpoints: $CHECKPOINT_DIR"
echo "  Metrics:     $METRICS_DIR"
echo "  Resume:      $RESUME"
echo "============================================"

# ── Step 1: Certificates ─────────────────────────────────────────
if [ ! -f "$CERTS_DIR/ca.pem" ]; then
    echo "[1/3] Generating TLS certificates..."
    "$SCRIPT_DIR/generate_certs.sh" "$CERTS_DIR"
else
    echo "[1/3] TLS certificates already exist in $CERTS_DIR"
fi

# ── Step 2: Submit server ─────────────────────────────────────────
echo "[2/3] Submitting server job..."
SERVER_JOB=$(sbatch \
    --partition="$PARTITION" \
    --export=ALL,SFL_NUM_CLIENTS=$NUM_CLIENTS,SFL_NUM_ROUNDS=$NUM_ROUNDS,SFL_MODEL=$MODEL,SFL_CERTS_DIR=$CERTS_DIR,SFL_CHECKPOINT_DIR=$CHECKPOINT_DIR,SFL_METRICS_DIR=$METRICS_DIR,SFL_RESUME=$RESUME \
    --chdir="$PROJECT_DIR" \
    "$SCRIPT_DIR/submit_server.sbatch" \
    | awk '{print $NF}')
echo "  Server job: $SERVER_JOB"

# ── Step 3: Submit clients ────────────────────────────────────────
CLIENT_MAX=$((NUM_CLIENTS - 1))
echo "[3/3] Submitting $NUM_CLIENTS client jobs (array 0-$CLIENT_MAX)..."
CLIENT_JOB=$(sbatch \
    --partition="$PARTITION" \
    --array="0-$CLIENT_MAX" \
    --dependency=after:$SERVER_JOB \
    --export=ALL,SFL_CERTS_DIR=$CERTS_DIR \
    --chdir="$PROJECT_DIR" \
    "$SCRIPT_DIR/submit_client.sbatch" \
    | awk '{print $NF}')
echo "  Client job array: $CLIENT_JOB"

echo ""
echo "============================================"
echo "  Federation submitted!"
echo "============================================"
echo ""
echo "Monitor with:"
echo "  squeue -u $USER -n sfl-server,sfl-client"
echo "  tail -f logs/server_${SERVER_JOB}.out"
echo "  ./examples/hpc/monitor_training.sh $METRICS_DIR"
echo ""
echo "Cancel with:"
echo "  scancel $SERVER_JOB $CLIENT_JOB"
