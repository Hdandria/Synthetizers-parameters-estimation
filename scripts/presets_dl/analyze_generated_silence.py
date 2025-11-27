import os
import json
import numpy as np
import soundfile as sf
from pathlib import Path

SAMPLES_DIR = "generated_samples_30"
SILENCE_THRESHOLD = 1e-4

def analyze_silence():
    wav_files = list(Path(SAMPLES_DIR).glob("*.wav"))
    print(f"Analyzing {len(wav_files)} samples in {SAMPLES_DIR}...\n")
    
    silent_count = 0
    osc_silent_count = 0
    
    for wav_path in wav_files:
        # Check Audio Silence
        audio, sr = sf.read(wav_path)
        rms = np.sqrt(np.mean(audio**2))
        
        if rms < SILENCE_THRESHOLD:
            silent_count += 1
            print(f"SILENT: {wav_path.name} (RMS: {rms:.6f})")
            
            # Inspect Vital File
            vital_path = wav_path.with_suffix(".vital")
            if vital_path.exists():
                try:
                    with open(vital_path, 'r', encoding='utf-8', errors='ignore') as f:
                        data = json.load(f)
                    settings = data.get('settings', {})
                    
                    osc1 = settings.get('oscillator_1_level', 0.0)
                    osc2 = settings.get('oscillator_2_level', 0.0)
                    osc3 = settings.get('oscillator_3_level', 0.0)
                    sample_level = settings.get('sample_level', 0.0)
                    sample_name = settings.get('sample', {}).get('name', 'None')
                    vol = settings.get('volume', 0.0)
                    
                    print(f"  -> Osc Levels: 1={osc1:.2f}, 2={osc2:.2f}, 3={osc3:.2f}")
                    print(f"  -> Sample: '{sample_name}' (Level={sample_level:.2f})")
                    print(f"  -> Master Vol: {vol:.2f}")
                    
                    if osc1 == 0 and osc2 == 0 and osc3 == 0:
                        print("  => CAUSE: All Oscillators are Silent (Level 0)")
                        osc_silent_count += 1
                    elif sample_level > 0 and (osc1 == 0 and osc2 == 0 and osc3 == 0):
                         print("  => CAUSE: Relies only on Sample (which is skipped)")
                    
                except Exception as e:
                    print(f"  -> Error reading vital file: {e}")
            else:
                print("  -> Vital file not found")
            print("-" * 40)
            
    print(f"\nSummary:")
    print(f"Total Samples: {len(wav_files)}")
    print(f"Silent Samples: {silent_count}")
    print(f"Silent due to Osc Levels=0: {osc_silent_count}")

if __name__ == "__main__":
    analyze_silence()
