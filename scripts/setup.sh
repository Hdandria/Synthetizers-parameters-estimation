#!/bin/bash
# Setup script for Synthesizer Parameters Estimation
# Installs dependencies and configures the environment

set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Synthesizer Parameters Estimation - Setup${NC}"
echo ""

echo -e "${BLUE}[1/6] Checking prerequisites${NC}"

# Python
if ! command -v python3 &> /dev/null; then
  echo -e "${RED}Error: Python 3 not found${NC}"
  echo "Install from https://www.python.org/downloads/"
  exit 1
fi
PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
echo -e "  ${GREEN}Python: $PYTHON_VERSION${NC}"

# pip
if ! command -v pip3 &> /dev/null && ! command -v pip &> /dev/null; then
  echo -e "${RED}Error: pip not found${NC}"
  exit 1
fi
echo -e "  ${GREEN}pip: OK${NC}"

# Docker
if ! command -v docker &> /dev/null; then
  echo -e "  ${YELLOW}Docker not found (needed for containerized training)${NC}"
  echo "  Install: https://docs.docker.com/engine/install/"
else
  echo -e "  ${GREEN}Docker: $(docker --version | cut -d' ' -f3 | tr -d ',')${NC}"
  
  if ! docker ps &> /dev/null; then
    echo -e "  ${YELLOW}Docker needs permissions - adding you to docker group${NC}"
    sudo usermod -aG docker $USER
    echo -e "    Log out and back in for this to take effect (or run: newgrp docker)${NC}"
  fi
fi

# jq
if ! command -v jq &> /dev/null; then
  echo -e "  ${YELLOW}Installing jq...${NC}"
  if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get update -qq && sudo apt-get install -y jq
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install jq
  else
    echo -e "  ${YELLOW}Install jq manually: https://stedolan.github.io/jq/${NC}"
  fi
fi
echo -e "  ${GREEN}jq: OK${NC}"

# Install uv package manager
echo ""
echo -e "${BLUE}[2/6] Installing uv package manager${NC}"

if command -v uv &> /dev/null; then
  echo -e "  ${GREEN}Already installed ($(uv --version))${NC}"
else
  echo "  Installing uv..."
  pip3 install uv
  echo -e "  ${GREEN}uv installed${NC}"
fi

# Install ovhai CLI
echo ""
echo -e "${BLUE}[3/6] Installing OVH AI CLI${NC}"

if command -v ovhai &> /dev/null; then
  echo -e "  ${GREEN}Already installed ($(ovhai version))${NC}"
else
  echo "  Downloading ovhai..."
  curl -fsSL https://cli.bhs.ai.cloud.ovh.net/install.sh | bash

  if ! command -v ovhai &> /dev/null; then
    echo -e "  ${YELLOW}Adding ovhai to PATH${NC}"
    echo 'export PATH="$HOME/.ovhai/bin:$PATH"' >> ~/.bashrc
    export PATH="$HOME/.ovhai/bin:$PATH"
  fi

  echo -e "  ${GREEN}ovhai installed${NC}"
fi

# Install Python dependencies
echo ""
echo -e "${BLUE}[4/6] Installing Python dependencies${NC}"

if [[ -f pyproject.toml ]]; then
  echo "  Installing packages..."
  uv pip install --system -r pyproject.toml
  echo -e "  ${GREEN}Dependencies installed${NC}"
else
  echo -e "  ${RED}pyproject.toml not found${NC}"
  exit 1
fi

# Create .env file
echo ""
echo -e "${BLUE}[5/6] Setting up environment${NC}"

if [[ -f .env ]]; then
  echo -e "  ${GREEN}.env already exists${NC}"
else
  if [[ -f .env.example ]]; then
    cp .env.example .env
    echo -e "  ${GREEN}Created .env from template${NC}"
    echo -e "  ${YELLOW}Edit .env with your credentials${NC}"
  else
    echo -e "  ${RED}.env.example not found${NC}"
    exit 1
  fi
fi

# Create directories
echo ""
echo -e "${BLUE}[6/6] Creating directories${NC}"

mkdir -p datasets outputs logs
echo -e "  ${GREEN}Created: datasets/, outputs/, logs/${NC}"

# OVH login check
echo ""
echo -e "${BLUE}Checking OVH authentication${NC}"

if ovhai me &> /dev/null; then
  echo -e "  ${GREEN}Logged in to OVH AI Training${NC}"
else
  echo -e "  ${YELLOW}Not logged in - run: ovhai login${NC}"
fi

# Configure S3 datastore
if ovhai me &> /dev/null; then
  echo ""
  echo -e "${BLUE}Configuring S3 datastore${NC}"
  
  if [[ -f .env ]]; then
    source .env
    
    if ovhai datastore list 2>/dev/null | grep -q "^s3-GRA"; then
      echo -e "  ${GREEN}s3-GRA datastore configured${NC}"
    else
      if [[ -n "${AWS_ACCESS_KEY_ID:-}" && -n "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
        echo "  Setting up s3-GRA..."
        ovhai datastore add s3 s3-GRA \
          "${AWS_ENDPOINT_URL:-https://s3.gra.io.cloud.ovh.net}" \
          "gra" \
          "${AWS_ACCESS_KEY_ID}" \
          "${AWS_SECRET_ACCESS_KEY}" \
          --store-credentials-locally 2>/dev/null || true
        echo -e "  ${GREEN}Datastore configured${NC}"
      else
        echo -e "  ${YELLOW}S3 credentials not in .env${NC}"
      fi
    fi
  fi
fi

# Summary
echo ""
echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo ""
echo "  1. Configure credentials:"
echo -e "     ${BLUE}nano .env${NC}"
echo ""
echo "  2. Log in to OVH:"
echo -e "     ${BLUE}ovhai login${NC}"
echo ""
echo "  3. Launch training:"
echo -e "     ${BLUE}./launch.sh flow_multi/dataset_20k_40k${NC}"
echo ""
echo "  4. Monitor progress:"
echo -e "     ${BLUE}./scripts/ovh/logs.sh <job-id>${NC}"
echo ""
