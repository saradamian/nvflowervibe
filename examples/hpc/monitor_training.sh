#!/bin/bash
set -euo pipefail

# Monitor SFL federation training progress.
#
# Usage:
#   ./monitor_training.sh /scratch/$USER/sfl-metrics
#   ./monitor_training.sh /scratch/$USER/sfl-metrics --watch

METRICS_DIR="${1:?Usage: $0 <metrics-directory> [--watch]}"
WATCH="${2:-}"

show_status() {
    echo "============================================"
    echo "  SFL Training Monitor"
    echo "  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "============================================"

    # Show SLURM job status
    echo ""
    echo "── SLURM Jobs ────────────────────────────"
    squeue -u "$USER" -n sfl-server,sfl-client \
        -o "%.8i %.9P %.20j %.8T %.10M %.6D %R" 2>/dev/null || echo "  (squeue not available)"

    # Show latest metrics
    echo ""
    echo "── Latest Metrics ────────────────────────"
    if [ -f "$METRICS_DIR/metrics.json" ]; then
        python3 -c "
import json, sys
with open('$METRICS_DIR/metrics.json') as f:
    data = json.load(f)
rounds = data.get('rounds', [])
if not rounds:
    print('  No rounds completed yet.')
    sys.exit(0)
latest = rounds[-1]
print(f'  Completed rounds: {len(rounds)}')
for k, v in latest.items():
    if k != 'round_num':
        print(f'  {k}: {v}')
" 2>/dev/null || echo "  (waiting for first round)"
    elif [ -f "$METRICS_DIR/metrics.csv" ]; then
        echo "  $(head -1 "$METRICS_DIR/metrics.csv")"
        echo "  $(tail -1 "$METRICS_DIR/metrics.csv")"
        echo "  Total rounds: $(( $(wc -l < "$METRICS_DIR/metrics.csv") - 1 ))"
    else
        echo "  No metrics files found in $METRICS_DIR"
    fi

    # Show checkpoint status
    echo ""
    echo "── Checkpoints ───────────────────────────"
    CKPT_PARENT="$(dirname "$METRICS_DIR")/sfl-checkpoints"
    if [ -d "$CKPT_PARENT" ]; then
        LATEST_CKPT=$(ls -d "$CKPT_PARENT"/*/round_* 2>/dev/null | sort -V | tail -1)
        if [ -n "$LATEST_CKPT" ]; then
            echo "  Latest: $LATEST_CKPT"
            echo "  Size: $(du -sh "$LATEST_CKPT" 2>/dev/null | cut -f1)"
        else
            echo "  No checkpoints yet"
        fi
    else
        echo "  Checkpoint directory not found"
    fi

    # Show GPU utilization on the server node
    echo ""
    echo "── GPU Utilization (this node) ──────────"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader 2>/dev/null || echo "  (no GPUs on this node)"

    echo ""
    echo "============================================"
}

if [ "$WATCH" = "--watch" ]; then
    while true; do
        clear
        show_status
        sleep 10
    done
else
    show_status
fi
