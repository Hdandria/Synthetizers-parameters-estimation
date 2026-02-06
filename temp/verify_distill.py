import rootutils

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

import torch
from transformers import ClapModel, ClapProcessor
import torchaudio
from src.models.surge_flow_matching_module import SurgeFlowMatchingModule
from src.models.components.transformer import (
    AudioSpectrogramTransformer,
    ApproxEquivTransformer,
    LearntProjection,
)


def test_distillation():
    print("Initializing components...")
    # Mock components
    encoder = AudioSpectrogramTransformer(
        d_model=512,
        n_heads=8,
        n_layers=2,
        n_conditioning_outputs=8,
        patch_size=16,
        patch_stride=10,
        input_channels=2,
        spec_shape=[128, 401],
    )

    # Mock Vector Field parts
    projection = LearntProjection(d_model=512, d_token=512, num_params=10, num_tokens=10)
    vector_field = ApproxEquivTransformer(
        projection=projection,
        num_layers=2,
        d_model=512,
        conditioning_dim=512,
        num_heads=8,
        d_ff=512,
        num_tokens=10,
        learn_projection=True,
    )

    print("Initializing SurgeFlowMatchingModule with distill=True...")
    # Initialize module
    module = SurgeFlowMatchingModule(
        encoder=encoder,
        vector_field=vector_field,
        optimizer=torch.optim.Adam,
        scheduler=None,
        distill=True,
        distill_weight=1.0,
        num_params=10,
    )

    # Mock data
    batch_size = 2
    audio = torch.randn(batch_size, 2, 44100 * 4)  # 4 seconds of stereo audio
    batch = {"audio": audio}

    # Mock conditioning embedding (from encoder)
    conditioning_embedding = torch.randn(batch_size, 8, 512)  # (B, n_tokens, d_model)

    print("Computing CLAP loss...")
    # Compute loss
    loss = module._compute_clap_loss(batch, conditioning_embedding)

    print(f"Loss computed: {loss.item()}")
    assert loss.item() > 0
    print("Test passed!")


if __name__ == "__main__":
    test_distillation()
