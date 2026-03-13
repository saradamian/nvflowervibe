#!/bin/bash
#SBATCH --job-name=esm2-fl
#SBATCH --output=logs/esm2-fl_%j.out
#SBATCH --error=logs/esm2-fl_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu

# ==============================================================
# ESM2 Federated Learning — SLURM Job Script
#
# This runs Flower simulation on a single node with multiple
# simulated FL clients sharing the available GPUs.
#
# Usage:
#   sbatch scripts/slurm_esm2.sh                     # defaults
#   NUM_CLIENTS=8 NUM_ROUNDS=10 sbatch scripts/slurm_esm2.sh
#   DATASET=ur50 sbatch scripts/slurm_esm2.sh        # HF dataset
#
# Adjust #SBATCH directives above to match your HPC cluster.
# ==============================================================

set -euo pipefail

# ── Configurable parameters (override via environment) ──
NUM_CLIENTS="${NUM_CLIENTS:-8}"
NUM_ROUNDS="${NUM_ROUNDS:-5}"
MODEL="${MODEL:-facebook/esm2_t6_8M_UR50D}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_LENGTH="${MAX_LENGTH:-128}"
DATASET="${DATASET:-}"
SEQUENCE_COLUMN="${SEQUENCE_COLUMN:-sequence}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
BACKEND="${BACKEND:-flower}"

# ── Paths ──
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
SAVE_DIR="${SAVE_DIR:-${PROJECT_DIR}/outputs/esm2_${SLURM_JOB_ID:-local}}"
VENV_DIR="${VENV_DIR:-${PROJECT_DIR}/.venv}"
LOG_DIR="${PROJECT_DIR}/logs"

mkdir -p "${LOG_DIR}" "${SAVE_DIR}"

# ── Environment ──
echo "============================================================"
echo "ESM2 Federated Learning — SLURM Job"
echo "============================================================"
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          $(hostname)"
echo "GPUs:          ${CUDA_VISIBLE_DEVICES:-none}"
echo "Clients:       ${NUM_CLIENTS}"
echo "Rounds:        ${NUM_ROUNDS}"
echo "Model:         ${MODEL}"
echo "Dataset:       ${DATASET:-built-in demo}"
echo "Save dir:      ${SAVE_DIR}"
echo "============================================================"

# Activate virtual environment
if [[ -d "${VENV_DIR}" ]]; then
    source "${VENV_DIR}/bin/activate"
else
    echo "WARNING: No venv at ${VENV_DIR}, using system Python"
fi

# Load modules if available (common on HPC)
if command -v module &>/dev/null; then
    module load cuda 2>/dev/null || true
    module load python 2>/dev/null || true
fi

# Print environment info
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
"

# ── Build command ──
CMD="python ${PROJECT_DIR}/jobs/esm2_runner.py \
    --num-clients ${NUM_CLIENTS} \
    --num-rounds ${NUM_ROUNDS} \
    --model ${MODEL} \
    --local-epochs ${LOCAL_EPOCHS} \
    --learning-rate ${LEARNING_RATE} \
    --batch-size ${BATCH_SIZE} \
    --max-length ${MAX_LENGTH} \
    --save-dir ${SAVE_DIR} \
    --backend ${BACKEND}"

if [[ -n "${DATASET}" ]]; then
    CMD="${CMD} --dataset ${DATASET} --sequence-column ${SEQUENCE_COLUMN}"
fi

if [[ -n "${MAX_SAMPLES}" ]]; then
    CMD="${CMD} --max-samples ${MAX_SAMPLES}"
fi

echo ""
echo "Running: ${CMD}"
echo ""

# ── Run ──
eval "${CMD}"

echo ""
echo "============================================================"
echo "Job complete. Model saved to: ${SAVE_DIR}"
echo "============================================================"
