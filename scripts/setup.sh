#!/bin/bash
################################################################################
# One-Time Setup Script
################################################################################
# This script installs required tools and initializes the infrastructure
################################################################################

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${GREEN}🚀 Setting up ML Training Infrastructure${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

################################################################################
# Check prerequisites
################################################################################

echo -e "${BLUE}📋 Checking prerequisites...${NC}"

# Docker
if ! command -v docker &> /dev/null; then
  echo -e "${RED}❌ Docker not found${NC}"
  echo "Install Docker: https://docs.docker.com/engine/install/"
  exit 1
fi
echo -e "${GREEN}✓${NC} Docker installed

# Check Docker permissions
if ! docker ps &> /dev/null; then
  echo -e "${YELLOW}⚠️  Docker requires root permissions${NC}"
  echo "Adding your user to the docker group..."
  sudo usermod -aG docker $USER
  echo ""
  echo -e "${YELLOW}⚠️  IMPORTANT: Log out and log back in for Docker permissions to take effect${NC}"
  echo "Or run: newgrp docker"
  echo ""
fi"

# jq
if ! command -v jq &> /dev/null; then
  echo -e "${YELLOW}⚠️  jq not found, installing...${NC}"
  if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get update && sudo apt-get install -y jq
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install jq
  else
    echo -e "${RED}Please install jq manually: https://stedolan.github.io/jq/${NC}"
    exit 1
  fi
fi
echo -e "${GREEN}✓${NC} jq installed"

################################################################################
# Install ovhai CLI
################################################################################

echo ""
echo -e "${BLUE}📦 Installing ovhai CLI...${NC}"

if command -v ovhai &> /dev/null; then
  echo -e "${GREEN}✓${NC} ovhai CLI already installed ($(ovhai version))"
else
  echo "Downloading ovhai CLI..."
  curl -fsSL https://cli.bhs.ai.cloud.ovh.net/install.sh | bash
  
  # Add to PATH if needed
  if ! command -v ovhai &> /dev/null; then
    echo -e "${YELLOW}Adding ovhai to PATH...${NC}"
    echo 'export PATH="$HOME/.ovhai/bin:$PATH"' >> ~/.bashrc
    export PATH="$HOME/.ovhai/bin:$PATH"
  fi
  
  echo -e "${GREEN}✓${NC} ovhai CLI installed"
fi

################################################################################
# Install Terraform (if not present)
################################################################################

echo ""
echo -e "${BLUE}📦 Checking Terraform...${NC}"

if command -v terraform &> /dev/null; then
  echo -e "${GREEN}✓${NC} Terraform already installed ($(terraform version | head -n1))"
else
  echo -e "${YELLOW}Terraform not found. Installing...${NC}"
  
  if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    wget -O- https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp-archive-keyring.gpg
    echo "deb [signed-by=/usr/share/keyrings/hashicorp-archive-keyring.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/hashicorp.list
    sudo apt update && sudo apt install terraform
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew tap hashicorp/tap
    brew install hashicorp/tap/terraform
  else
    echo -e "${YELLOW}Please install Terraform manually: https://www.terraform.io/downloads${NC}"
  fi
  
  echo -e "${GREEN}✓${NC} Terraform installed"
fi

################################################################################
# Create .env file if not exists
################################################################################

echo ""
echo -e "${BLUE}📄 Setting up environment file...${NC}"

if [[ -f .env ]]; then
  echo -e "${GREEN}✓${NC} .env file already exists"
else
  if [[ -f .env.example ]]; then
    cp .env.example .env
    echo -e "${YELLOW}⚠️  Created .env from template${NC}"
    echo -e "${YELLOW}   Please edit .env with your credentials:${NC}"
    echo -e "   ${BLUE}nano .env${NC}"
  else
    echo -e "${RED}❌ .env.example not found${NC}"
    exit 1
  fi
fi

################################################################################
# Initialize Terraform
################################################################################

echo ""
echo -e "${BLUE}🏗️  Initializing Terraform...${NC}"

cd terraform

if [[ ! -f terraform.tfvars ]]; then
  if [[ -f terraform.tfvars.example ]]; then
    cp terraform.tfvars.example terraform.tfvars
    echo -e "${YELLOW}⚠️  Created terraform.tfvars from template${NC}"
    echo -e "${YELLOW}   Please edit terraform.tfvars with your OVH credentials:${NC}"
    echo -e "   ${BLUE}cd terraform && nano terraform.tfvars${NC}"
    echo ""
    echo -e "   Get OVH API credentials here:"
    echo -e "   ${BLUE}https://www.ovh.com/auth/api/createToken${NC}"
    echo ""
    echo -e "   Required scopes: GET/POST/PUT/DELETE on /cloud/*"
    echo ""
    echo -e "${YELLOW}   After editing, run:${NC}"
    echo -e "   ${GREEN}cd terraform && terraform init && terraform apply${NC}"
    cd ..
    exit 0
  else
    echo -e "${RED}❌ terraform.tfvars.example not found${NC}"
    cd ..
    exit 1
  fi
fi

terraform init

echo -e "${GREEN}✓${NC} Terraform initialized"

cd ..

################################################################################
# Create necessary directories
################################################################################

echo ""
echo -e "${BLUE}📁 Creating directories...${NC}"

mkdir -p datasets outputs logs

echo -e "${GREEN}✓${NC} Directories created"

################################################################################
# Summary
################################################################################

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e "${GREEN}✅ Setup complete!${NC}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo ""
echo "1. Configure OVH credentials:"
echo -e "   ${BLUE}cd terraform && nano terraform.tfvars${NC}"
echo ""
echo "2. Deploy infrastructure:"
echo -e "   ${BLUE}cd terraform && terraform apply${NC}"
echo ""
echo "3. Launch your first experiment:"
echo -e "   ${BLUE}./launch.sh surge/base${NC}"
echo ""
echo "4. Monitor training:"
echo -e "   ${BLUE}./scripts/status.sh <job-id>${NC}"
echo ""
echo -e "${GREEN}Happy training! 🎉${NC}"
