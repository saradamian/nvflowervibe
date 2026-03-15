#!/bin/bash
set -euo pipefail

# Launch a complete SFL federation on SLURM.
#
# Usage:
#   ./launch_federation.sh --num-clients 8 --num-rounds 50
#   ./launch_federation.sh --num-clients 4 --partition a100 --certs-dir /scratch/certs
#   ./launch_federation.sh --num-clients 4 --dp --dp-noise 0.5 --secagg --aggregation krum
#
# This script:
#   1. Generates TLS certificates (if not already present)
#   2. Submits the server job (SuperLink + ServerApp)
#   3. Submits client array jobs (SuperNode + ClientApp per client)
#   4. Prints monitoring commands
#
# Privacy/aggregation flags (--dp, --secagg, --aggregation, etc.) are translated
# to SFL_* env vars and propagated to BOTH server and client SLURM jobs.

# ── Defaults ──────────────────────────────────────────────────────
NUM_CLIENTS=4
NUM_ROUNDS=20
PARTITION="gpu"
MODEL="facebook/esm2_t6_8M_UR50D"
CERTS_DIR="/scratch/$USER/sfl-certs"
CHECKPOINT_DIR="/scratch/$USER/sfl-checkpoints"
METRICS_DIR="/scratch/$USER/sfl-metrics"
RESUME=false
INSECURE=false

# Privacy/aggregation defaults (empty = not set)
DP_ENABLED=""
DP_NOISE=""
DP_CLIP=""
DP_MODE=""
DP_ADAPTIVE_CLIP=""
SECAGG_ENABLED=""
AGGREGATION=""
KRUM_BYZANTINE=""
TRIM_RATIO=""
METRICS_FORMAT="csv"

# ── Parse arguments ───────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --num-clients)       NUM_CLIENTS="$2";       shift 2 ;;
        --num-rounds)        NUM_ROUNDS="$2";        shift 2 ;;
        --partition)         PARTITION="$2";          shift 2 ;;
        --model)             MODEL="$2";             shift 2 ;;
        --certs-dir)         CERTS_DIR="$2";         shift 2 ;;
        --checkpoint-dir)    CHECKPOINT_DIR="$2";    shift 2 ;;
        --metrics-dir)       METRICS_DIR="$2";       shift 2 ;;
        --metrics-format)    METRICS_FORMAT="$2";    shift 2 ;;
        --resume)            RESUME=true;            shift ;;
        --insecure)          INSECURE=true;          shift ;;
        # Privacy/aggregation flags
        --dp)                DP_ENABLED=true;         shift ;;
        --dp-noise)          DP_NOISE="$2";          shift 2 ;;
        --dp-clip)           DP_CLIP="$2";           shift 2 ;;
        --dp-mode)           DP_MODE="$2";           shift 2 ;;
        --dp-adaptive-clip)  DP_ADAPTIVE_CLIP=true;  shift ;;
        --secagg)            SECAGG_ENABLED=true;    shift ;;
        --aggregation)       AGGREGATION="$2";       shift 2 ;;
        --krum-byzantine)    KRUM_BYZANTINE="$2";    shift 2 ;;
        --trim-ratio)        TRIM_RATIO="$2";        shift 2 ;;
        *)
            echo "ERROR: Unknown argument: $1"
            echo "Run with --help for usage information."
            exit 1
            ;;
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
echo "  Insecure:    $INSECURE"
[ -n "$DP_ENABLED" ]      && echo "  DP:          enabled (noise=${DP_NOISE:-default}, clip=${DP_CLIP:-default})"
[ -n "$SECAGG_ENABLED" ]  && echo "  SecAgg:      enabled"
[ -n "$AGGREGATION" ]     && echo "  Aggregation: $AGGREGATION"
echo "============================================"

# ── Build SFL_* env var export string ─────────────────────────────
# These vars are passed to BOTH server and client jobs so that
# server_fn and auto_build_client_mods() see identical config.
SFL_EXPORT="SFL_NUM_CLIENTS=$NUM_CLIENTS"
SFL_EXPORT="$SFL_EXPORT,SFL_NUM_ROUNDS=$NUM_ROUNDS"
SFL_EXPORT="$SFL_EXPORT,SFL_MODEL=$MODEL"
SFL_EXPORT="$SFL_EXPORT,SFL_CERTS_DIR=$CERTS_DIR"
SFL_EXPORT="$SFL_EXPORT,SFL_CHECKPOINT_DIR=$CHECKPOINT_DIR"
SFL_EXPORT="$SFL_EXPORT,SFL_METRICS_DIR=$METRICS_DIR"
SFL_EXPORT="$SFL_EXPORT,SFL_METRICS_FORMAT=$METRICS_FORMAT"
SFL_EXPORT="$SFL_EXPORT,SFL_RESUME=$RESUME"
SFL_EXPORT="$SFL_EXPORT,SFL_INSECURE=$INSECURE"

[ -n "$DP_ENABLED" ]      && SFL_EXPORT="$SFL_EXPORT,SFL_DP_ENABLED=true"
[ -n "$DP_NOISE" ]        && SFL_EXPORT="$SFL_EXPORT,SFL_DP_NOISE=$DP_NOISE"
[ -n "$DP_CLIP" ]         && SFL_EXPORT="$SFL_EXPORT,SFL_DP_CLIP=$DP_CLIP"
[ -n "$DP_MODE" ]         && SFL_EXPORT="$SFL_EXPORT,SFL_DP_MODE=$DP_MODE"
[ -n "$DP_ADAPTIVE_CLIP" ] && SFL_EXPORT="$SFL_EXPORT,SFL_DP_ADAPTIVE_CLIP=true"
[ -n "$SECAGG_ENABLED" ]  && SFL_EXPORT="$SFL_EXPORT,SFL_SECAGG_ENABLED=true"
[ -n "$AGGREGATION" ]     && SFL_EXPORT="$SFL_EXPORT,SFL_AGGREGATION=$AGGREGATION"
[ -n "$KRUM_BYZANTINE" ]  && SFL_EXPORT="$SFL_EXPORT,SFL_KRUM_BYZANTINE=$KRUM_BYZANTINE"
[ -n "$TRIM_RATIO" ]      && SFL_EXPORT="$SFL_EXPORT,SFL_TRIM_RATIO=$TRIM_RATIO"

# ── Step 1: Certificates ─────────────────────────────────────────
if [ "$INSECURE" = "true" ]; then
    echo "[1/3] Skipping TLS certificates (insecure mode)"
elif [ ! -f "$CERTS_DIR/ca.pem" ]; then
    echo "[1/3] Generating TLS certificates..."
    "$SCRIPT_DIR/generate_certs.sh" "$CERTS_DIR"
else
    echo "[1/3] TLS certificates already exist in $CERTS_DIR"
fi

# ── Step 2: Submit server ─────────────────────────────────────────
echo "[2/3] Submitting server job..."
SERVER_JOB=$(sbatch \
    --partition="$PARTITION" \
    --export=ALL,$SFL_EXPORT \
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
    --export=ALL,$SFL_EXPORT \
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
