"""
Simple training script for Surge parameter estimation
"""
import torch
import torch.nn as nn
import yaml
from pathlib import Path
from tqdm import tqdm
import random
import numpy as np

from model import create_model
from dataloader import create_dataloaders


def set_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rk4_step(model, x, t, dt, conditioning, cfg_strength):
    """4th order Runge-Kutta ODE solver step"""
    # Compute with and without conditioning for CFG
    def f(x_in, t_in):
        v_cond = model.vector_field(x_in, t_in, conditioning)
        v_uncond = model.vector_field(x_in, t_in, None)
        return v_uncond + cfg_strength * (v_cond - v_uncond)
    
    k1 = f(x, t)
    k2 = f(x + dt * k1 / 2, t + dt / 2)
    k3 = f(x + dt * k2 / 2, t + dt / 2)
    k4 = f(x + dt * k3, t + dt)
    
    return x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)


def sample_flow(model, mel_spec, num_steps=50, cfg_strength=4.0, device='cuda'):
    """Sample from the flow model using ODE solver"""
    batch_size = mel_spec.shape[0]
    n_params = model.vector_field.out_proj.out_features
    
    # Start from noise
    x = torch.randn(batch_size, n_params, device=device)
    
    # Encode conditioning
    with torch.no_grad():
        conditioning = model.encode(mel_spec)
    
    # Integrate from t=0 to t=1
    dt = 1.0 / num_steps
    t = torch.zeros(batch_size, 1, device=device)
    
    for _ in range(num_steps):
        with torch.no_grad():
            x = rk4_step(model, x, t, dt, conditioning, cfg_strength)
        t = t + dt
    
    return x


def train_flow_matching(model, train_loader, val_loader, config, device):
    """Train the flow matching model"""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate'])
    )
    
    # Warmup + cosine schedule
    warmup_steps = config['training']['warmup_steps']
    total_steps = len(train_loader) * config['training']['num_epochs']
    
    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        else:
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    
    cfg_dropout = config['training']['cfg_dropout_rate']
    num_epochs = config['training']['num_epochs']
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in pbar:
            mel_spec = batch['mel_spec'].to(device)
            params = batch['params'].to(device)
            noise = batch['noise'].to(device)
            
            batch_size = params.shape[0]
            
            # Sample timestep
            t = torch.rand(batch_size, 1, device=device)
            
            # Interpolate between noise and params
            x_t = noise * (1 - t) + params * t
            
            # Target velocity (for rectified flow)
            target = params - noise
            
            # Encode conditioning
            conditioning = model.encode(mel_spec)
            
            # Apply CFG dropout
            if cfg_dropout > 0:
                mask = torch.rand(batch_size, device=device) > cfg_dropout
                conditioning = conditioning * mask[:, None]
            
            # Predict velocity
            pred = model.vector_field(x_t, t, conditioning)
            
            # Compute loss
            loss = nn.functional.mse_loss(pred, target)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            global_step += 1
            
            pbar.set_postfix({'loss': loss.item(), 'lr': scheduler.get_last_lr()[0]})
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                mel_spec = batch['mel_spec'].to(device)
                params = batch['params'].to(device)
                
                # Sample from flow
                pred_params = sample_flow(
                    model, mel_spec,
                    num_steps=config['training']['num_sample_steps'],
                    cfg_strength=config['training']['cfg_strength'],
                    device=device
                )
                
                # Compute MSE
                loss = nn.functional.mse_loss(pred_params, params)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_every'] == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f'Saved checkpoint to {checkpoint_path}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = checkpoint_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, best_path)
            print(f'Saved best model (val_loss={val_loss:.4f})')


def train_feedforward(model, train_loader, val_loader, config, device):
    """Train the simple feedforward model"""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config['training']['learning_rate'])
    )
    
    num_epochs = config['training']['num_epochs']
    checkpoint_dir = Path(config['training']['checkpoint_dir'])
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch in pbar:
            mel_spec = batch['mel_spec'].to(device)
            params = batch['params'].to(device)
            
            # Forward pass
            pred = model(mel_spec)
            loss = nn.functional.mse_loss(pred, params)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                mel_spec = batch['mel_spec'].to(device)
                params = batch['params'].to(device)
                pred = model(mel_spec)
                loss = nn.functional.mse_loss(pred, params)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        print(f'Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % config['training']['save_every'] == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = checkpoint_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, best_path)
            print(f'Saved best model (val_loss={val_loss:.4f})')


def main():
    # Load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(config['seed'])
    
    # Setup device
    device = config['device']
    if device == 'cuda' and not torch.cuda.is_available():
        print('CUDA not available, using CPU')
        device = 'cpu'
    
    print(f'Using device: {device}')
    
    # Create dataloaders
    print('Loading data...')
    train_loader, val_loader, test_loader = create_dataloaders(config)
    print(f'Train batches: {len(train_loader)}')
    print(f'Val batches: {len(val_loader)}')
    
    # Create model
    print('Creating model...')
    model = create_model(config).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Model parameters: {num_params:,}')
    
    # Train
    print('Starting training...')
    if config['model']['type'] == 'flow_matching':
        train_flow_matching(model, train_loader, val_loader, config, device)
    else:
        train_feedforward(model, train_loader, val_loader, config, device)
    
    print('Training complete!')


if __name__ == '__main__':
    main()

