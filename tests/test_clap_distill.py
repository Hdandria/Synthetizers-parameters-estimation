import torch
import time
import sys
import numpy as np
from transformers import ClapModel, ClapProcessor
import torchaudio
import matplotlib.pyplot as plt

# Fix path
sys.path.append(".")

from src.models.components.transformer import AudioSpectrogramTransformer
from src.models.surge_flow_matching_module import SurgeFlowMatchingModule
from src.models.components.transformer import ApproxEquivTransformer, LearntProjection


def test_fixes_and_performance():
    print("Testing Fixes and Performance...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. Verify d_model fix
    print("\n1. Verifying AudioSpectrogramTransformer d_model fix...")
    encoder = AudioSpectrogramTransformer(d_model=768)
    if hasattr(encoder, "d_model") and encoder.d_model == 768:
        print("SUCCESS: Encoder has correct d_model attribute.")
    else:
        print("FAILURE: Encoder missing d_model or incorrect value.")

    # 2. Verify SurgeFlowMatchingModule initialization
    print("\n2. Verifying SurgeFlowMatchingModule initialization...")
    try:
        # Mock components
        # projection = LearntProjection(
        #     d_model=512, d_token=512, num_params=90, num_tokens=128
        # )
        # vector_field = ApproxEquivTransformer(
        #      projection=projection, d_model=512
        # )
        vector_field = torch.nn.Linear(512, 512)  # Dummy
        # Initialize module with distill=True
        model = SurgeFlowMatchingModule(
            encoder=encoder,
            vector_field=vector_field,
            optimizer=torch.optim.Adam,
            scheduler=None,
            distill=True,
            distill_weight=1.0,
        ).to(device)
        print("SUCCESS: SurgeFlowMatchingModule initialized with distill=True.")
    except Exception as e:
        print(f"FAILURE: SurgeFlowMatchingModule initialization failed: {e}")
        return

    # 3. Compare Torchaudio vs ClapProcessor
    print("\n3. Comparing Torchaudio vs ClapProcessor implementation...")

    # Load original processor for ground truth
    try:
        processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")
    except Exception as e:
        print(f"Skipping comparison, cannot load ClapProcessor: {e}")
        return

    # Generate dummy audio
    batch_size = 4
    audio = torch.randn(batch_size, 2, 44100).to(device)  # Stereo

    # Run Module's internal logic (Torchaudio)
    # We need to access the internal transforms created in __init__
    # Replicating logic from _compute_clap_loss to inspect intermediate steps if needed,
    # or just calling _compute_clap_loss if we can mock the rest.
    # Let's just run the preprocessing part manually using the model's attributes to verify match.

    # A. Torchaudio Path (New)
    start_time = time.time()
    with torch.no_grad():
        audio_mono = audio.mean(dim=1)
        audio_48k = model.resampler(audio_mono)

        # Pad/Crop
        target_length = 480000
        current_length = audio_48k.shape[-1]
        if current_length < target_length:
            audio_48k = torch.nn.functional.pad(audio_48k, (0, target_length - current_length))
        elif current_length > target_length:
            audio_48k = audio_48k[..., :target_length]

        # Experiment with norm=None
        mel_transform_test = torchaudio.transforms.MelSpectrogram(
            sample_rate=48000,
            n_fft=1024,
            win_length=1024,
            hop_length=480,
            f_min=0,
            f_max=None,
            n_mels=64,
            center=True,
            pad_mode="reflect",
            power=2.0,
            norm=None,  # Try None instead of "slaney"
            mel_scale="htk",
        ).to(device)

        mels = mel_transform_test(audio_48k)

        # Try manual log10 with ref=1.0 (implied if not scaled)
        # librosa.power_to_db(S, ref=1.0) -> 10 * log10(S / 1.0)
        mels_db = 10 * torch.log10(mels + 1e-10)

        # Ensure top_db clamping if needed
        # mels_db = torch.max(mels_db, mels_db.max() - 80)

        input_features = mels_db.transpose(1, 2).unsqueeze(1)

        # Run CLAP model
        outputs_new = model.clap.audio_model(
            input_features=input_features,
            is_longer=torch.tensor([False] * batch_size, device=device),
        )
        embedding_new = model.clap.audio_projection(outputs_new.pooler_output)

    gpu_time = time.time() - start_time
    print(f"GPU (Torchaudio) Path Time: {gpu_time:.4f}s")
    print(f"GPU Embedding Shape: {embedding_new.shape}")

    # B. Processor Path (Old)
    start_time = time.time()
    audio_np = (
        model.resampler(audio.mean(dim=1)).detach().cpu().numpy()
    )  # Resample on GPU first to be fair on sampling rate?
    # Actually the old method resampled on GPU then moved to CPU.
    # But wait, the old method used `model.resampler` which was initialized in the module.
    # We are simulating the "Old" full flow:

    # Re-instantiate resampler for fairness (or reuse model's)
    # The old code did:
    # audio_48k = self.resampler(audio_mono)
    # audio_np = audio_48k.detach().cpu().numpy()
    # inputs = self.clap_processor(audio=list(audio_np), ...)

    audio_np_raw = audio_48k.cpu().numpy()  # We already matched the resampling step above
    # Note: Processor also handles padding/truncation, so we should pass the resampled audio
    # but maybe before padding?
    # Actually processor takes raw audio array.

    inputs = processor(
        audio=list(audio_np_raw), sampling_rate=48000, return_tensors="pt", padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs_old = model.clap.audio_model(**inputs)
        embedding_old = model.clap.audio_projection(outputs_old.pooler_output)

    cpu_time = time.time() - start_time
    print(f"CPU (Processor) Path Time: {cpu_time:.4f}s")
    print(f"CPU Embedding Shape: {embedding_old.shape}")

    # Compare embeddings
    # Cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(embedding_new, embedding_old)
    print(f"Cosine Similarity (Mean): {cos_sim.mean().item():.4f}")
    print(f"Cosine Similarity (Min): {cos_sim.min().item():.4f}")

    if cos_sim.mean().item() > 0.9:
        print("SUCCESS: Embeddings are sufficiently similar.")
    else:
        print("WARNING: Embeddings differ significantly. Check preprocessing parameters.")
        # Debug Mels
        # For Processor, we need to extract what it did.
        # input_features in 'inputs' has shape (B, 1, T, F)
        feat_old = inputs["input_features"]
        feat_new = input_features

        print(f"Feature Shape Old: {feat_old.shape}")
        print(f"Feature Shape New: {feat_new.shape}")

        # Check if Old has 4 channels because it replicates the mono input?
        # processor inputs: audio list of numpy arrays.
        # If we passed list of 4 arrays, maybe it treated them as 4 channels? No, batch of 4.

        # Check content of channels in Old
        print(f"Old Channel 0 vs 1 diff: {(feat_old[:, 0] - feat_old[:, 1]).abs().mean()}")
        print(f"Old Channel 0 vs 2 diff: {(feat_old[:, 0] - feat_old[:, 2]).abs().mean()}")
        print(f"Old Channel 0 vs 3 diff: {(feat_old[:, 0] - feat_old[:, 3]).abs().mean()}")

        # Scale difference?
        print(f"Old Mean: {feat_old.mean()}, Std: {feat_old.std()}")
        print(f"New Mean: {feat_new.mean()}, Std: {feat_new.std()}")

        diff = (feat_old[:, 0:1] - feat_new).abs().mean()
        print(f"Feature Mean Diff (vs Chan 0): {diff.item()}")


if __name__ == "__main__":
    test_fixes_and_performance()
