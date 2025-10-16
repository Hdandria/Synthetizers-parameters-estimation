#!/bin/bash

# Simple setup script for remote GPU server

echo "🚀 Setting up Synthesizer Parameters Estimation on remote server..."

# Check if .env exists
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp env.template .env
    echo "⚠️  Please edit .env file with your actual credentials:"
    echo "   - WANDB_API_KEY: Your Weights & Biases API key"
    echo "   - S3_BUCKET: Your S3 bucket name"
    echo "   - S3_PLUGIN_PATH: Path to plugins in S3 (default: /plugins)"
    echo ""
    echo "   Then run: nano .env"
    exit 1
fi

# Build Docker image
echo "🐳 Building Docker image..."
docker build -t synth-param-estimation:latest .

# Create output directories
echo "📁 Creating output directories..."
mkdir -p outputs logs

echo "✅ Setup complete!"
echo ""
echo "🚀 To launch all experiments:"
echo "   ./scripts/launch_flow_multi.sh"
echo ""
echo "📊 Monitor experiments at:"
echo "   https://wandb.ai/your-entity/synth-prediction"
