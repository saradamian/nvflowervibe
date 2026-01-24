#!/bin/bash
# Quick run script for SFL simulation
# Usage: ./scripts/run_simulation.sh [options]

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [[ -f "${PROJECT_ROOT}/.venv/bin/activate" ]]; then
    source "${PROJECT_ROOT}/.venv/bin/activate"
fi

# Default values
NUM_CLIENTS=${SFL_NUM_CLIENTS:-2}
NUM_ROUNDS=${SFL_NUM_ROUNDS:-1}

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}SFL - Running Federated Simulation${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "Clients: ${YELLOW}${NUM_CLIENTS}${NC}"
echo -e "Rounds:  ${YELLOW}${NUM_ROUNDS}${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Run the simulation with any additional arguments passed
python jobs/runner.py \
    --num-clients "$NUM_CLIENTS" \
    --num-rounds "$NUM_ROUNDS" \
    "$@"
