"""
Simple prediction script for Surge parameter estimation on audio files
"""
import torch
import yaml
import numpy as np
import librosa
from pathlib import Path
import argparse

from model import create_model


def load_audio_and_compute_mel(audio_path, sample_rate=44100, duration=4.0):
    """Load audio file and compute mel spectrogram"""
    # Load audio
    audio, sr = librosa.load(audio_path, sr=sample_rate, duration=duration, mono=False)
    
    # Ensure stereo
    if len(audio.shape) == 1:
        audio = np.stack([audio, audio])
    elif audio.shape[0] == 1:
        audio = np.concatenate([audio, audio], axis=0)
    
    # Pad or trim to exact duration
    target_length = int(sample_rate * duration)
    if audio.shape[1] < target_length:
        audio = np.pad(audio, ((0, 0), (0, target_length - audio.shape[1])))
    else:
        audio = audio[:, :target_length]
    
    # Compute mel spectrogram
    n_fft = int(0.025 * sample_rate)
    hop_length = int(sample_rate / 100.0)
    
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=128,
        n_fft=n_fft,
        hop_length=hop_length,
        window='hamming'
    )
    
    mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec


def sample_flow(model, mel_spec, num_steps=100, cfg_strength=4.0, device='cuda'):
    """Sample from the flow model"""
    batch_size = mel_spec.shape[0]
    n_params = model.vector_field.out_proj.out_features
    
    # Start from noise
    x = torch.randn(batch_size, n_params, device=device)
    
    # Encode conditioning
    with torch.no_grad():
        conditioning = model.encode(mel_spec)
    
    # Integrate ODE
    dt = 1.0 / num_steps
    t = torch.zeros(batch_size, 1, device=device)
    
    for _ in range(num_steps):
        with torch.no_grad():
            # CFG
            v_cond = model.vector_field(x, t, conditioning)
            v_uncond = model.vector_field(x, t, None)
            v = v_uncond + cfg_strength * (v_cond - v_uncond)
            
            # Euler step (simpler than RK4 for inference)
            x = x + dt * v
        
        t = t + dt
    
    return x


def predict(audio_path, config_path='config.yaml', checkpoint_path=None):
    """Predict parameters for an audio file"""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup device
    device = config['device']
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    # Create model
    model = create_model(config).to(device)
    
    # Load checkpoint
    if checkpoint_path is None:
        checkpoint_path = config['eval']['checkpoint']
    
    print(f'Loading model from {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load and process audio
    print(f'Loading audio from {audio_path}')
    mel_spec = load_audio_and_compute_mel(audio_path)
    
    # Normalize if stats available
    data_path = Path(config['data']['dataset_path'])
    stats_file = data_path / 'stats.npz'
    if stats_file.exists():
        stats = np.load(stats_file)
        mel_spec = (mel_spec - stats['mean']) / stats['std']
    
    # Convert to tensor and add batch dimension
    mel_spec = torch.from_numpy(mel_spec).float().unsqueeze(0).to(device)
    
    # Predict
    print('Predicting parameters...')
    with torch.no_grad():
        if config['model']['type'] == 'flow_matching':
            params = sample_flow(
                model, mel_spec,
                num_steps=config['eval']['num_sample_steps'],
                cfg_strength=config['eval']['cfg_strength'],
                device=device
            )
        else:
            params = model(mel_spec)
    
    params = params.cpu().numpy()[0]
    
    # Convert from [-1, 1] back to [0, 1]
    params = (params + 1) / 2
    
    print(f'\nPredicted {len(params)} parameters')
    print(f'Parameter range: [{params.min():.3f}, {params.max():.3f}]')
    
    return params


def main():
    parser = argparse.ArgumentParser(description='Predict Surge parameters from audio')
    parser.add_argument('audio_file', help='Path to audio file')
    parser.add_argument('--config', default='config.yaml', help='Config file')
    parser.add_argument('--checkpoint', help='Model checkpoint (default: from config)')
    parser.add_argument('--output', '-o', help='Output file for parameters (.npy)')
    
    args = parser.parse_args()
    
    # Predict
    params = predict(args.audio_file, args.config, args.checkpoint)
    
    # Save if requested
    if args.output:
        np.save(args.output, params)
        print(f'Saved parameters to {args.output}')
    else:
        print('\nPredicted parameters:')
        print(params)


if __name__ == '__main__':
    main()

