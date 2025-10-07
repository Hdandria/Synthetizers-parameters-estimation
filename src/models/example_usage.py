"""
Example usage of the simplified Surge parameter estimation model

This script demonstrates how to use the model programmatically
"""
import torch
import yaml
import numpy as np

from model import create_model
from dataloader import create_dataloaders


def example_training():
    """Example: How to train a model"""
    print("=" * 60)
    print("Example 1: Training a Model")
    print("=" * 60)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create dataloaders
    train_loader, val_loader, _ = create_dataloaders(config)
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model(config).to(device)
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Simple training loop (1 epoch)
    model.train()
    for i, batch in enumerate(train_loader):
        if i >= 10:  # Just show first 10 batches
            break
        
        mel_spec = batch['mel_spec'].to(device)
        params = batch['params'].to(device)
        noise = batch['noise'].to(device)
        
        # Sample timestep
        t = torch.rand(params.shape[0], 1, device=device)
        
        # Interpolate
        x_t = noise * (1 - t) + params * t
        
        # Get prediction
        if config['model']['type'] == 'flow_matching':
            conditioning = model.encode(mel_spec)
            pred = model.vector_field(x_t, t, conditioning)
            target = params - noise
        else:
            pred = model(mel_spec)
            target = params
        
        # Compute loss
        loss = torch.nn.functional.mse_loss(pred, target)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Batch {i+1}/10: loss = {loss.item():.4f}")
    
    print("\nTraining example complete!")


def example_inference():
    """Example: How to run inference"""
    print("\n" + "=" * 60)
    print("Example 2: Running Inference")
    print("=" * 60)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model(config).to(device)
    
    # Create fake mel spectrogram (for demo)
    fake_mel = torch.randn(1, 2, 128, 401).to(device)
    
    model.eval()
    with torch.no_grad():
        if config['model']['type'] == 'flow_matching':
            # Flow matching inference
            conditioning = model.encode(fake_mel)
            
            # Start from noise
            x = torch.randn(1, config['model']['vector_field']['n_params']).to(device)
            
            # Simple Euler ODE solver
            num_steps = 50
            dt = 1.0 / num_steps
            t = torch.zeros(1, 1, device=device)
            
            for _ in range(num_steps):
                v = model.vector_field(x, t, conditioning)
                x = x + dt * v
                t = t + dt
            
            predicted_params = x
        else:
            # Feedforward inference
            predicted_params = model(fake_mel)
    
    # Convert from [-1, 1] to [0, 1]
    predicted_params = (predicted_params + 1) / 2
    
    print(f"Predicted {predicted_params.shape[1]} parameters")
    print(f"Range: [{predicted_params.min():.3f}, {predicted_params.max():.3f}]")
    print(f"First 10 params: {predicted_params[0, :10].cpu().numpy()}")
    
    print("\nInference example complete!")


def example_custom_model():
    """Example: How to modify the model architecture"""
    print("\n" + "=" * 60)
    print("Example 3: Custom Model Architecture")
    print("=" * 60)
    
    import torch.nn as nn
    from model import SinusoidalEncoding, PatchEmbed
    
    # Create a custom lightweight encoder
    class LightweightEncoder(nn.Module):
        def __init__(self, d_model=256, n_layers=4):
            super().__init__()
            self.patch_embed = PatchEmbed(d_model=d_model)
            
            # Simpler transformer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=4,
                dim_feedforward=d_model * 2,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.out = nn.Linear(d_model, d_model)
        
        def forward(self, x):
            x = self.patch_embed(x)
            x = self.transformer(x)
            x = x.transpose(1, 2)
            x = self.pool(x).squeeze(-1)
            return self.out(x)
    
    # Create custom model
    encoder = LightweightEncoder()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoder = encoder.to(device)
    
    # Test it
    fake_mel = torch.randn(4, 2, 128, 401).to(device)
    output = encoder(fake_mel)
    
    print(f"Custom encoder output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    
    print("\nCustom model example complete!")


def example_data_loading():
    """Example: How to work with the data"""
    print("\n" + "=" * 60)
    print("Example 4: Data Loading and Processing")
    print("=" * 60)
    
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Note: This requires actual data files to exist
    # Showing the structure for reference
    
    print("Dataset structure:")
    print("- HDF5 file contains:")
    print("  - 'mel_spec': (N, 2, 128, 401) float32")
    print("  - 'param_array': (N, 92) float32")
    print("  - 'audio': (N, 2, 176400) float16 (optional)")
    
    print("\nDataloader returns batches with:")
    print("  - 'mel_spec': normalized mel spectrograms")
    print("  - 'params': parameters scaled to [-1, 1]")
    print("  - 'noise': random noise for flow matching")
    
    print("\nTo create your own dataset, see src/data/generate_vst_dataset.py")
    
    print("\nData loading example complete!")


def main():
    """Run all examples"""
    print("\n" + "=" * 60)
    print("Surge Parameter Estimation - Usage Examples")
    print("=" * 60)
    
    try:
        example_training()
    except Exception as e:
        print(f"Training example failed (need dataset): {e}")
    
    try:
        example_inference()
    except Exception as e:
        print(f"Inference example failed: {e}")
    
    try:
        example_custom_model()
    except Exception as e:
        print(f"Custom model example failed: {e}")
    
    example_data_loading()
    
    print("\n" + "=" * 60)
    print("All examples complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()

