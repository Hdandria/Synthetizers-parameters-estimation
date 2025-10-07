"""
Simplified model architecture for Surge parameter estimation
"""
import math
import torch
import torch.nn as nn


class SinusoidalEncoding(nn.Module):
    """Encode timesteps using sinusoidal positional encoding"""
    def __init__(self, d_model):
        super().__init__()
        half = d_model // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(0, half) / half)
        self.register_buffer('freqs', freqs)
    
    def forward(self, t):
        # t: (batch, 1)
        args = t * self.freqs[None, :]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)


class PatchEmbed(nn.Module):
    """Convert mel spectrogram patches to embeddings"""
    def __init__(self, patch_size=16, stride=10, in_channels=2, d_model=768):
        super().__init__()
        self.projection = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size,
            stride=stride
        )
    
    def forward(self, x):
        # x: (batch, channels, mel_bins, time)
        x = self.projection(x)  # (batch, d_model, H', W')
        x = x.flatten(2).transpose(1, 2)  # (batch, num_patches, d_model)
        return x


class SimpleTransformerEncoder(nn.Module):
    """Simple transformer encoder for mel spectrograms"""
    def __init__(self, d_model=768, n_heads=8, n_layers=12):
        super().__init__()
        
        self.patch_embed = PatchEmbed(d_model=d_model)
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.randn(1, 1500, d_model) * 0.02)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
    
    def forward(self, mel_spec):
        # Embed patches
        x = self.patch_embed(mel_spec)
        
        # Add positional encoding
        x = x + self.pos_embed[:, :x.shape[1], :]
        
        # Apply transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        return self.out_proj(x)


class VectorField(nn.Module):
    """MLP-based vector field for flow matching"""
    def __init__(self, n_params=92, d_model=1024, hidden_dim=2048, num_layers=6):
        super().__init__()
        
        # Time encoding
        self.time_encoding = SinusoidalEncoding(256)
        
        # Conditioning FFN
        self.cond_ffn = nn.Sequential(
            nn.Linear(d_model + 256, d_model),  # conditioning + time
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        # Input projection
        self.in_proj = nn.Linear(n_params, d_model)
        
        # MLP layers
        layers = []
        for i in range(num_layers):
            layers.append(nn.LayerNorm(d_model))
            layers.append(nn.Linear(d_model, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, d_model))
        self.mlp = nn.Sequential(*layers)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, n_params)
        
        # CFG dropout token
        self.cfg_token = nn.Parameter(torch.randn(1, d_model))
    
    def forward(self, x, t, conditioning=None):
        """
        x: (batch, n_params) - current state
        t: (batch, 1) - timestep
        conditioning: (batch, d_model) - conditioning vector (or None for CFG)
        """
        # Handle CFG dropout
        if conditioning is None:
            conditioning = self.cfg_token.expand(x.shape[0], -1)
        
        # Encode time
        t_enc = self.time_encoding(t)
        
        # Combine conditioning and time
        z = torch.cat([conditioning, t_enc], dim=-1)
        z = self.cond_ffn(z)
        
        # Process x through MLP with conditioning
        x = self.in_proj(x)
        
        # Simple additive conditioning
        for i, layer in enumerate(self.mlp):
            if i % 4 == 0:  # Before each block
                x = x + z
            x = layer(x)
        
        return self.out_proj(x)


class FlowMatchingModel(nn.Module):
    """Complete flow matching model"""
    def __init__(self, config):
        super().__init__()
        
        encoder_cfg = config['model']['encoder']
        vf_cfg = config['model']['vector_field']
        
        self.encoder = SimpleTransformerEncoder(
            d_model=encoder_cfg['d_model'],
            n_heads=encoder_cfg['n_heads'],
            n_layers=encoder_cfg['n_layers']
        )
        
        self.vector_field = VectorField(
            n_params=vf_cfg['n_params'],
            d_model=encoder_cfg['d_model'],  # Match encoder output
            hidden_dim=vf_cfg['hidden_dim'],
            num_layers=vf_cfg['num_layers']
        )
    
    def forward(self, mel_spec, x, t, conditioning=None):
        """
        Forward pass for training or inference
        If conditioning is None, uses unconditional path
        """
        if conditioning is None:
            conditioning = self.encoder(mel_spec)
        
        return self.vector_field(x, t, conditioning)
    
    def encode(self, mel_spec):
        """Just encode mel spectrogram to conditioning vector"""
        return self.encoder(mel_spec)


class SimpleFeedForward(nn.Module):
    """Simple feed-forward baseline model"""
    def __init__(self, config):
        super().__init__()
        
        encoder_cfg = config['model']['encoder']
        n_params = config['model']['vector_field']['n_params']
        
        self.encoder = SimpleTransformerEncoder(
            d_model=encoder_cfg['d_model'],
            n_heads=encoder_cfg['n_heads'],
            n_layers=encoder_cfg['n_layers']
        )
        
        # Direct prediction head
        self.predictor = nn.Sequential(
            nn.Linear(encoder_cfg['d_model'], 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, n_params)
        )
    
    def forward(self, mel_spec):
        z = self.encoder(mel_spec)
        return self.predictor(z)


def create_model(config):
    """Factory function to create model based on config"""
    model_type = config['model']['type']
    
    if model_type == 'flow_matching':
        return FlowMatchingModel(config)
    elif model_type == 'feedforward':
        return SimpleFeedForward(config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

