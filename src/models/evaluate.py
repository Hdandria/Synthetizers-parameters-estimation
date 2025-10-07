"""
Simple evaluation script for Surge parameter estimation
"""
import torch
import yaml
import numpy as np
from pathlib import Path
from tqdm import tqdm

from model import create_model
from dataloader import create_dataloaders


def sample_flow(model, mel_spec, num_steps=100, cfg_strength=4.0, device='cuda'):
    """Sample from the flow model using ODE solver"""
    batch_size = mel_spec.shape[0]
    n_params = model.vector_field.out_proj.out_features
    
    # Start from noise
    x = torch.randn(batch_size, n_params, device=device)
    
    # Encode conditioning
    with torch.no_grad():
        conditioning = model.encode(mel_spec)
    
    # Integrate from t=0 to t=1 using RK4
    dt = 1.0 / num_steps
    t = torch.zeros(batch_size, 1, device=device)
    
    for _ in range(num_steps):
        with torch.no_grad():
            # Compute velocity with CFG
            v_cond = model.vector_field(x, t, conditioning)
            v_uncond = model.vector_field(x, t, None)
            v = v_uncond + cfg_strength * (v_cond - v_uncond)
            
            # RK4 step
            k1 = v
            k2 = model.vector_field(x + dt * k1 / 2, t + dt / 2, conditioning)
            k2_uncond = model.vector_field(x + dt * k1 / 2, t + dt / 2, None)
            k2 = k2_uncond + cfg_strength * (k2 - k2_uncond)
            
            k3 = model.vector_field(x + dt * k2 / 2, t + dt / 2, conditioning)
            k3_uncond = model.vector_field(x + dt * k2 / 2, t + dt / 2, None)
            k3 = k3_uncond + cfg_strength * (k3 - k3_uncond)
            
            k4 = model.vector_field(x + dt * k3, t + dt, conditioning)
            k4_uncond = model.vector_field(x + dt * k3, t + dt, None)
            k4 = k4_uncond + cfg_strength * (k4 - k4_uncond)
            
            x = x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
        
        t = t + dt
    
    return x


def evaluate_flow_matching(model, test_loader, config, device):
    """Evaluate flow matching model"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    total_mse = 0
    num_batches = 0
    
    print('Evaluating...')
    with torch.no_grad():
        for batch in tqdm(test_loader):
            mel_spec = batch['mel_spec'].to(device)
            params = batch['params'].to(device)
            
            # Sample from flow
            pred_params = sample_flow(
                model, mel_spec,
                num_steps=config['eval']['num_sample_steps'],
                cfg_strength=config['eval']['cfg_strength'],
                device=device
            )
            
            # Compute MSE
            mse = torch.nn.functional.mse_loss(pred_params, params)
            total_mse += mse.item()
            num_batches += 1
            
            # Store for analysis
            all_predictions.append(pred_params.cpu().numpy())
            all_targets.append(params.cpu().numpy())
    
    # Aggregate results
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    avg_mse = total_mse / num_batches
    
    # Compute per-parameter MSE
    per_param_mse = np.mean((all_predictions - all_targets) ** 2, axis=0)
    
    print(f'\nResults:')
    print(f'Average MSE: {avg_mse:.4f}')
    print(f'Per-param MSE - Min: {per_param_mse.min():.4f}, Max: {per_param_mse.max():.4f}, Mean: {per_param_mse.mean():.4f}')
    
    # Save results
    output_dir = Path(config['eval']['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    
    np.savez(
        output_dir / 'predictions.npz',
        predictions=all_predictions,
        targets=all_targets,
        per_param_mse=per_param_mse
    )
    
    print(f'\nSaved results to {output_dir / "predictions.npz"}')
    
    return {
        'avg_mse': avg_mse,
        'per_param_mse': per_param_mse,
        'predictions': all_predictions,
        'targets': all_targets
    }


def evaluate_feedforward(model, test_loader, config, device):
    """Evaluate feedforward model"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    total_mse = 0
    num_batches = 0
    
    print('Evaluating...')
    with torch.no_grad():
        for batch in tqdm(test_loader):
            mel_spec = batch['mel_spec'].to(device)
            params = batch['params'].to(device)
            
            # Forward pass
            pred_params = model(mel_spec)
            
            # Compute MSE
            mse = torch.nn.functional.mse_loss(pred_params, params)
            total_mse += mse.item()
            num_batches += 1
            
            # Store for analysis
            all_predictions.append(pred_params.cpu().numpy())
            all_targets.append(params.cpu().numpy())
    
    # Aggregate results
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    avg_mse = total_mse / num_batches
    
    # Compute per-parameter MSE
    per_param_mse = np.mean((all_predictions - all_targets) ** 2, axis=0)
    
    print(f'\nResults:')
    print(f'Average MSE: {avg_mse:.4f}')
    print(f'Per-param MSE - Min: {per_param_mse.min():.4f}, Max: {per_param_mse.max():.4f}, Mean: {per_param_mse.mean():.4f}')
    
    # Save results
    output_dir = Path(config['eval']['output_dir'])
    output_dir.mkdir(exist_ok=True, parents=True)
    
    np.savez(
        output_dir / 'predictions.npz',
        predictions=all_predictions,
        targets=all_targets,
        per_param_mse=per_param_mse
    )
    
    print(f'\nSaved results to {output_dir / "predictions.npz"}')
    
    return {
        'avg_mse': avg_mse,
        'per_param_mse': per_param_mse,
        'predictions': all_predictions,
        'targets': all_targets
    }


def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = config['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, using CPU')
        device = 'cpu'
    
    print(f'Using device: {device}')
    
    # Create dataloaders
    print('Loading data...')
    _, _, test_loader = create_dataloaders(config)
    print(f'Test batches: {len(test_loader)}')
    
    # Create model
    print('Creating model...')
    model = create_model(config).to(device)
    
    # Load checkpoint
    checkpoint_path = config['eval']['checkpoint']
    print(f'Loading checkpoint from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate
    if config['model']['type'] == 'flow_matching':
        results = evaluate_flow_matching(model, test_loader, config, device)
    else:
        results = evaluate_feedforward(model, test_loader, config, device)
    
    print('\nEvaluation complete!')


if __name__ == '__main__':
    main()

