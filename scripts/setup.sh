#!/bin/bash
# SFL Environment Setup Script
# This script creates a virtual environment and installs dependencies

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}SFL - Simple Federated Learning Setup${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

# Check Python version
echo -e "${YELLOW}Checking Python version...${NC}"
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo "$PYTHON_VERSION" | cut -d. -f1)
PYTHON_MINOR=$(echo "$PYTHON_VERSION" | cut -d. -f2)

if [[ "$PYTHON_MAJOR" -lt 3 ]] || [[ "$PYTHON_MAJOR" -eq 3 && "$PYTHON_MINOR" -lt 9 ]]; then
    echo -e "${RED}Error: Python 3.9+ is required (found $PYTHON_VERSION)${NC}"
    exit 1
fi
echo -e "${GREEN}Python $PYTHON_VERSION detected${NC}"

# Create virtual environment
VENV_DIR="${PROJECT_ROOT}/.venv"

if [[ -d "$VENV_DIR" ]]; then
    echo -e "${YELLOW}Virtual environment already exists at $VENV_DIR${NC}"
    read -p "Recreate it? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$VENV_DIR"
        python3 -m venv "$VENV_DIR"
        echo -e "${GREEN}Virtual environment recreated${NC}"
    fi
else
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}Virtual environment created at $VENV_DIR${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source "$VENV_DIR/bin/activate"

# Upgrade pip
echo -e "${YELLOW}Upgrading pip, wheel, setuptools...${NC}"
pip install -U pip wheel setuptools --quiet

# Install dependencies
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install -r requirements.txt

# Install project in development mode
echo -e "${YELLOW}Installing SFL in development mode...${NC}"
pip install -e . --quiet

# Copy .env.example if .env doesn't exist
if [[ ! -f "${PROJECT_ROOT}/.env" ]] && [[ -f "${PROJECT_ROOT}/.env.example" ]]; then
    echo -e "${YELLOW}Creating .env from .env.example...${NC}"
    cp "${PROJECT_ROOT}/.env.example" "${PROJECT_ROOT}/.env"
    echo -e "${GREEN}.env file created${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Setup complete!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "To activate the environment:"
echo -e "  ${YELLOW}source .venv/bin/activate${NC}"
echo ""
echo -e "To run the simulation:"
echo -e "  ${YELLOW}python jobs/runner.py${NC}"
echo ""
echo -e "For more options:"
echo -e "  ${YELLOW}python jobs/runner.py --help${NC}"
