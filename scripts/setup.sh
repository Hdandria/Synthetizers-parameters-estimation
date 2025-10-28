#!/bin/bash
# One-time setup script for ML training infrastructure
# Installs required tools and initializes the project

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}Setting up ML training infrastructure...${NC}"
echo ""

# Check prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

# Docker
if ! command -v docker &> /dev/null; then
  echo -e "${RED}Error: Docker not found${NC}"
  echo "Install Docker: https://docs.docker.com/engine/install/"
  exit 1
fi
echo -e "  ${GREEN}Docker: OK${NC}"

# Check Docker permissions
if ! docker ps &> /dev/null; then
  echo -e "  ${YELLOW}Docker needs permissions. Adding you to docker group...${NC}"
  sudo usermod -aG docker $USER
  echo ""
  echo -e "  ${YELLOW}IMPORTANT: Log out and log back in for Docker permissions to take effect${NC}"
  echo "  Or run: newgrp docker"
  echo ""
fi

# jq
if ! command -v jq &> /dev/null; then
  echo -e "  ${YELLOW}jq not found, installing...${NC}"
  if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get update && sudo apt-get install -y jq
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install jq
  else
    echo -e "${RED}Error: Please install jq manually: https://stedolan.github.io/jq/${NC}"
    exit 1
  fi
fi
echo -e "  ${GREEN}jq: OK${NC}"

# Install ovhai CLI
echo ""
echo -e "${BLUE}Checking ovhai CLI...${NC}"

if command -v ovhai &> /dev/null; then
  echo -e "  ${GREEN}Already installed ($(ovhai version))${NC}"
else
  echo "  Downloading ovhai CLI..."
  curl -fsSL https://cli.bhs.ai.cloud.ovh.net/install.sh | bash
  
  # Add to PATH if needed
  if ! command -v ovhai &> /dev/null; then
    echo -e "  ${YELLOW}Adding ovhai to PATH...${NC}"
    echo 'export PATH="$HOME/.ovhai/bin:$PATH"' >> ~/.bashrc
    export PATH="$HOME/.ovhai/bin:$PATH"
  fi
  
  echo -e "  ${GREEN}ovhai CLI installed${NC}"
fi

# Install Terraform
echo ""
echo -e "${BLUE}Checking Terraform...${NC}"

if command -v terraform &> /dev/null; then
  echo -e "  ${GREEN}Already installed ($(terraform version | head -n1))${NC}"
else
  echo -e "  ${YELLOW}Terraform not found, installing...${NC}"
  
  if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
    echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
    sudo apt update && sudo apt install terraform
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew tap hashicorp/tap
    brew install hashicorp/tap/terraform
  else
    echo -e "  ${YELLOW}Please install Terraform manually: https://www.terraform.io/downloads${NC}"
  fi
  
  echo -e "  ${GREEN}Terraform installed${NC}"
fi

# Create .env file
echo ""
echo -e "${BLUE}Setting up environment file...${NC}"

if [[ -f .env ]]; then
  echo -e "  ${GREEN}.env file already exists${NC}"
else
  if [[ -f .env.example ]]; then
    cp .env.example .env
    echo -e "  ${YELLOW}Created .env from template${NC}"
    echo -e "  ${YELLOW}Please edit .env with your credentials (nano .env)${NC}"
  else
    echo -e "${RED}Error: .env.example not found${NC}"
    exit 1
  fi
fi

# Initialize Terraform
echo ""
echo -e "${BLUE}Initializing Terraform...${NC}"

cd terraform
terraform init
echo -e "  ${GREEN}Terraform initialized${NC}"
cd ..

# Create directories
echo ""
echo -e "${BLUE}Creating directories...${NC}"

mkdir -p datasets outputs logs
echo -e "  ${GREEN}Directories created${NC}"

# Summary
echo ""
echo "---"
echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo ""
echo "  1. Configure credentials:"
echo -e "     ${BLUE}nano .env${NC}"
echo ""
echo "  2. Deploy infrastructure:"
echo -e "     ${BLUE}cd terraform && terraform apply${NC}"
echo ""
echo "  3. Launch your first experiment:"
echo -e "     ${BLUE}./launch.sh flow_multi/dataset_20k_40k${NC}"
echo ""
echo "  4. Monitor training:"
echo -e "     ${BLUE}./scripts/status.sh <job-id>${NC}"
echo ""
