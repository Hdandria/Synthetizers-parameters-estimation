from collections.abc import Callable
from functools import partial
from typing import Any, Dict, Literal, Optional, Tuple

import torch
from lightning import LightningModule
from lightning.pytorch.utilities import grad_norm
from transformers import ClapModel, ClapProcessor
import torchaudio


def call_with_cfg(
    f: Callable,
    x: torch.Tensor,
    t: torch.Tensor,
    conditioning: torch.Tensor,
    cfg_strength: float,
):
    y_c = f(x, t, conditioning)
    y_u = f(x, t, None)

    return (1 - cfg_strength) * y_u + cfg_strength * y_c


def rk4_with_cfg(
    f: Callable,
    x: torch.Tensor,
    t: torch.Tensor,
    dt: float,
    conditioning: torch.Tensor,
    cfg_strength: float,
):
    f = partial(call_with_cfg, f, conditioning=conditioning, cfg_strength=cfg_strength)
    k1 = f(x, t)
    k2 = f(x + dt * k1 / 2, t + dt / 2)
    k3 = f(x + dt * k2 / 2, t + dt / 2)
    k4 = f(x + dt * k3, t + dt)

    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


class SurgeFlowMatchingModule(LightningModule):
    def __init__(
        self,
        encoder: torch.nn.Module,
        vector_field: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        conditioning: Literal["mel", "m2l"] = "mel",
        warmup_steps: int = 5000,
        cfg_dropout_rate: float = 0.1,
        rectified_sigma_min: float = 0.0,
        validation_sample_steps: int = 50,
        validation_cfg_strength: float = 4.0,
        test_sample_steps: int = 100,
        test_cfg_strength: float = 4.0,
        compile: bool = False,
        num_params: int = 90,
        distill: bool = False,
        distill_weight: float = 10.0,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.encoder = encoder
        self.vector_field = vector_field

        if distill:
            self.clap = ClapModel.from_pretrained("laion/clap-htsat-fused")
            self.clap.eval()
            self.clap.requires_grad_(False)
            # Remove text model to save memory
            if hasattr(self.clap, "text_model"):
                del self.clap.text_model
            if hasattr(self.clap, "text_projection"):
                del self.clap.text_projection

            # Use torchaudio for GPU acceleration instead of ClapProcessor
            # CLAP expects 48kHz audio.
            self.resampler = torchaudio.transforms.Resample(44100, 48000)

            # CLAP Mel Spectrogram parameters (from laion/clap-htsat-fused config)
            # n_fft=1024, hop_length=480, n_mels=64
            self.mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=48000,
                n_fft=1024,
                win_length=1024,
                hop_length=480,
                f_min=0,
                f_max=None,  # Default for torchaudio, check if CLAP sets this
                n_mels=64,
                center=True,
                pad_mode="reflect",
                power=2.0,
                norm=None,  # Verified to match CLAP better than "slaney"
                mel_scale="htk",  # Check if this is correct for CLAP
            )
            # self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB() # Uses top_db shifting, mismatching CLAP

            # Project from encoder dim to CLAP dim (512)
            # Assuming encoder.d_model is available. If not, we might need to pass it explicitly.
            # Using 512 as CLAP embedding size.
            encoder_dim = getattr(encoder, "d_model", 512)
            self.projection = torch.nn.Linear(encoder_dim, 512)

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        pass

    def _sample_time(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.rand(n, 1, device=device)

    def _weight_time(self, t: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(t)

    def _basic_sample(self, params: torch.Tensor, oversample: float = 1.0):
        if oversample == 1.0:
            x0 = torch.randn_like(params)
        elif oversample < 1.0:
            raise ValueError(f"oversample must be >= 1.0, got {oversample}")
        else:
            n = int(oversample * params.shape[0])
            x0 = torch.randn(n, *params.shape[1:], device=params.device)
        x1 = params

        return x0, x1

    def _rectified_probability_path(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
        x_t = x0 * (1 - t) * (1 - self.hparams.rectified_sigma_min) + x1 * t

        return x_t

    def _sample_probability_path(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor):
        x_t = self._rectified_probability_path(x0, x1, t)
        return x_t

    def _rectified_vector_field(self, x0: torch.Tensor, x1: torch.Tensor):
        return x1 - x0

    def _evaluate_target_field(
        self, x0: torch.Tensor, x1: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ):
        target = self._rectified_vector_field(x0, x1)
        return target

    def _get_conditioning_from_batch(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        if self.hparams.conditioning == "mel":
            return batch["mel_spec"]
        elif self.hparams.conditioning == "m2l":
            return batch["m2l"]
        else:
            raise ValueError(f"Unknown conditioning {self.hparams.conditioning}")

    def _train_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], dropout_rate: float | None = None
    ):
        if dropout_rate is None:
            dropout_rate = self.hparams.cfg_dropout_rate

        conditioning = self._get_conditioning_from_batch(batch)
        params = batch["params"]
        noise = batch["noise"]

        # Get conditioning vector
        conditioning = self.encoder(conditioning)
        z = self.vector_field.apply_dropout(conditioning, dropout_rate)

        with torch.no_grad():
            # Sample time-steps
            t = self._sample_time(params.shape[0], params.device)
            w = self._weight_time(t)

            x0 = noise
            x1 = params

            # we sample a point along the trajectory
            x_t = self._sample_probability_path(x0, x1, t)
            target = self._evaluate_target_field(x0, x1, x_t, t)

        prediction = self.vector_field(x_t, t, z)

        # compute and weight loss
        loss = (prediction - target).square().mean(dim=-1)
        loss = loss * w
        loss = loss.mean()

        if self.hparams.distill:
            clap_loss = self._compute_clap_loss(batch, conditioning)
            loss = loss + self.hparams.distill_weight * clap_loss
            self.log("train/clap_loss", clap_loss, on_step=True, on_epoch=True, prog_bar=True)

        penalty = None
        if hasattr(self.vector_field, "penalty"):
            penalty = self.vector_field.penalty()

        return loss, penalty

    def _compute_clap_loss(
        self, batch: dict[str, torch.Tensor], conditioning_embedding: torch.Tensor
    ) -> torch.Tensor:
        audio = batch["audio"]  # (B, 2, L)

        # Force fp32 for the entire mel preprocessing + CLAP forward pass.
        # Under fp16 autocast, 1e-10 underflows to 0 → log10(0) = -inf → NaN.
        with torch.amp.autocast(device_type=self.device.type, enabled=False):
            audio = audio.float()

            # 1. Resample and Preprocess Audio for CLAP
            audio_mono = audio.mean(dim=1)  # (B, 2, L) -> (B, L)
            audio_48k = self.resampler(audio_mono)  # Resample to 48k

            # Pad or truncate to 480,000 samples (10 seconds at 48kHz)
            target_length = 480000
            current_length = audio_48k.shape[-1]
            if current_length < target_length:
                audio_48k = torch.nn.functional.pad(audio_48k, (0, target_length - current_length))
            elif current_length > target_length:
                audio_48k = audio_48k[..., :target_length]

            # Mel spectrogram + log scale
            mels = self.mel_transform(audio_48k)  # (B, n_mels, T)
            mels_db = 10 * torch.log10(mels + 1e-10)

            # ClapFeatureExtractor stacks 4 copies for non-fused input: (B, 4, T, n_mels)
            mels_2d = mels_db.transpose(1, 2)  # (B, T, n_mels)
            input_features = torch.stack([mels_2d, mels_2d, mels_2d, mels_2d], dim=1)

            inputs = {
                "input_features": input_features,
                "is_longer": torch.tensor([False] * audio.shape[0], device=self.device),
            }

            # 2. Get Teacher Embedding (CLAP) — no grad, fp32
            with torch.no_grad():
                outputs = self.clap.audio_model(**inputs)
                teacher_embedding = self.clap.audio_projection(outputs.pooler_output)

            # 3. Get Student Embedding (Projected Encoder Output)
            student_embedding = conditioning_embedding.float()
            if student_embedding.dim() == 3:
                student_embedding = student_embedding.mean(dim=1)  # Global Average Pooling

            student_projected = self.projection(student_embedding)  # (B, 512)

            # 4. Compute Loss (fp32 for numerical stability)
            loss = torch.nn.functional.mse_loss(student_projected, teacher_embedding)

        return loss

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, penalty = self._train_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        if penalty is not None:
            self.log("train/penalty", penalty, on_step=True, on_epoch=True, prog_bar=True)

        return loss + (penalty if penalty is not None else 0.0)

    def on_train_epoch_end(self) -> None:
        pass

    def _warp_time(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def _sample(
        self,
        conditioning: torch.Tensor | None,
        noise: torch.Tensor,
        steps: int,
        cfg_strength: float,
    ):
        if conditioning is not None:
            conditioning = self.encoder(conditioning)

        t = torch.zeros(noise.shape[0], 1, device=noise.device)
        dt = 1.0 / steps

        sample = noise

        for _ in range(steps):
            warped_t = self._warp_time(t)
            warped_t_plus_dt = self._warp_time(t + dt)
            warped_dt = warped_t_plus_dt - warped_t

            sample = rk4_with_cfg(
                self.vector_field,
                sample,
                warped_t,
                warped_dt,
                conditioning,
                cfg_strength,
            )
            t = t + dt

        return sample

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, penalty = self._train_step(batch, dropout_rate=0.0)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        if penalty is not None:
            self.log("val/penalty", penalty, on_step=False, on_epoch=True, prog_bar=True)

        return loss + (penalty if penalty is not None else 0.0)

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        conditioning = self._get_conditioning_from_batch(batch)
        pred_params = self._sample(
            conditioning,
            torch.randn_like(batch["params"]),
            self.hparams.test_sample_steps,
            self.hparams.test_cfg_strength,
        )

        param_mse = (pred_params - batch["params"]).square().mean()
        self.log("test/param_mse", param_mse, on_step=False, on_epoch=True, prog_bar=True)

        return param_mse

    def on_test_epoch_end(self) -> None:
        pass

    def predict_step(self, batch: dict[str, Any], batch_idx: int):
        conditioning = self._get_conditioning_from_batch(batch)
        return (
            self._sample(
                conditioning,
                torch.randn(
                    conditioning.shape[0],
                    self.hparams.num_params,
                    device=conditioning.device,
                ),
                self.hparams.test_sample_steps,
                self.hparams.test_cfg_strength,
            ),
            batch,
        )

    def setup(self, stage: str) -> None:
        if not self.hparams.compile:
            return

        self.vector_field = torch.compile(self.vector_field)
        self.encoder = torch.compile(self.encoder)

    def on_before_optimizer_step(self, optimizer) -> None:
        vf_norms = grad_norm(self.vector_field, 2.0)
        encoder_norms = grad_norm(self.encoder, 2.0)

        vf_norms = {f"vector_field/{k}": v for k, v in vf_norms.items()}
        encoder_norms = {f"encoder/{k}": v for k, v in encoder_norms.items()}

        self.log_dict(vf_norms, on_step=True, on_epoch=False)
        self.log_dict(encoder_norms, on_step=True, on_epoch=False)

    def configure_optimizers(self) -> dict[str, Any]:
        trainable_params = [p for p in self.trainer.model.parameters() if p.requires_grad]
        optimizer = self.hparams.optimizer(params=trainable_params)

        if self.hparams.warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, 1e-10, 1.0, self.hparams.warmup_steps
            )
        else:
            warmup_scheduler = None

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
        else:
            scheduler = None

        if warmup_scheduler is not None and scheduler is None:
            scheduler = warmup_scheduler
        elif warmup_scheduler is not None and scheduler is not None:
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, scheduler],
                milestones=[self.hparams.warmup_steps],
            )

        if scheduler is not None:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    # "monitor": "val/chamfer",
                    "interval": "step",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}

    # Lightning hook called right before the checkpoint state_dict is loaded.
    # Handles mismatches between compiled and non-compiled checkpoints.
    # When compile=True, setup() wraps modules BEFORE checkpoint loading during predict/test,
    # so we may need to add _orig_mod prefixes to checkpoint keys.
    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:  # type: ignore[override]
        state_dict = checkpoint.get("state_dict", {})
        if not state_dict:
            return

        # Check if checkpoint has compiled prefixes
        checkpoint_has_compiled = any(
            k.startswith("encoder._orig_mod.") or k.startswith("vector_field._orig_mod.")
            for k in state_dict.keys()
        )

        # Check if checkpoint has non-compiled encoder/vector_field keys
        checkpoint_has_noncompiled = any(
            (k.startswith("encoder.") and not k.startswith("encoder._orig_mod."))
            or (k.startswith("vector_field.") and not k.startswith("vector_field._orig_mod."))
            for k in state_dict.keys()
        )

        # If checkpoint is already in the right format or has no relevant keys, return
        if not checkpoint_has_compiled and not checkpoint_has_noncompiled:
            return

        new_state_dict: dict[str, Any] = {}

        # When loading into a model that will be/is compiled, add _orig_mod prefix
        if self.hparams.compile and checkpoint_has_noncompiled:
            for k, v in state_dict.items():
                if k.startswith("encoder.") and not k.startswith("encoder._orig_mod."):
                    new_key = "encoder._orig_mod." + k[len("encoder.") :]
                elif k.startswith("vector_field.") and not k.startswith("vector_field._orig_mod."):
                    new_key = "vector_field._orig_mod." + k[len("vector_field.") :]
                else:
                    new_key = k
                new_state_dict[new_key] = v
            checkpoint["state_dict"] = new_state_dict
        # When loading into a non-compiled model, strip _orig_mod prefix
        elif not self.hparams.compile and checkpoint_has_compiled:
            for k, v in state_dict.items():
                if k.startswith("encoder._orig_mod."):
                    new_key = "encoder." + k[len("encoder._orig_mod.") :]
                elif k.startswith("vector_field._orig_mod."):
                    new_key = "vector_field." + k[len("vector_field._orig_mod.") :]
                else:
                    new_key = k
                new_state_dict[new_key] = v
            checkpoint["state_dict"] = new_state_dict


if __name__ == "__main__":
    _ = SurgeFlowMatchingModule(None, None, None, None)
