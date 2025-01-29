import math
from functools import partial
from typing import Any, Callable, Dict, Literal, Tuple

import ot as pot
import torch
from lightning import LightningModule
from scipy.optimize import linear_sum_assignment

from src.metrics import (ChamferDistance, LinearAssignmentDistance,
                         LogSpectralDistance)
from src.utils.math import divmod


def late_curve(x, a):
    if a == 0.0:
        return x
    return (1 - torch.exp(-a * x)) / (1 - math.exp(-a))


def cosine_curve(x):
    return 0.5 + 0.5 * torch.cos(torch.pi * (1 + x))


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
        cfg_dropout_rate: float = 0.1,
        rectified_sigma_min: float = 0.0,
        validation_sample_steps: int = 50,
        validation_cfg_strength: float = 4.0,
        test_sample_steps: int = 100,
        test_cfg_strength: float = 4.0,
        sinkhorn_reg: float = 0.05,
        sinkhorn_thresh: float = 1e-6,
        ot_replace: bool = True,
        compile: bool = False,
        params_per_token: int = 2,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.encoder = encoder
        self.vector_field = vector_field

    def on_train_start(self):
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks

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

    def _rectified_probability_path(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ):
        x_t = x0 * (1 - t) * (1 - self.hparams.rectified_sigma_min) + x1 * t

        return x_t

    def _sample_probability_path(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor
    ):
        x_t = self._rectified_probability_path(x0, x1, t)
        return x_t

    def _rectified_vector_field(self, x0: torch.Tensor, x1: torch.Tensor):
        return x1 - x0

    def _evaluate_target_field(
        self, x0: torch.Tensor, x1: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ):
        target = self._rectified_vector_field(x0, x1)
        return target

    def _train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        mel_spec = batch["mel_spec"]
        params = batch["params"]
        noise = batch["noise"]

        # Get conditioning vector
        conditioning = self.encoder(mel_spec)
        z = self.vector_field.apply_dropout(conditioning, self.hparams.cfg_dropout_rate)

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

        penalty = None
        if hasattr(self.vector_field, "penalty"):
            penalty = self.vector_field.penalty()

        return loss, penalty

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, penalty = self._train_step(batch)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        if penalty is not None:
            self.log(
                "train/penalty", penalty, on_step=True, on_epoch=True, prog_bar=True
            )

        return loss + penalty

    def on_train_epoch_end(self) -> None:
        pass

    def _warp_time(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def _sample(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        steps: int,
        cfg_strength: float,
    ):
        x, y, sample, _ = batch

        # sample = torch.randn_like(y)
        conditioning = self.encoder(x)
        t = torch.zeros(sample.shape[0], 1, device=sample.device)
        dt = 1.0 / steps

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

        return sample, y, x

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        preds, targets, inputs = self._sample(
            batch,
            self.hparams.validation_sample_steps,
            self.hparams.validation_cfg_strength,
        )

    def on_validation_epoch_end(self):
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        preds, targets, inputs = self._sample(
            batch, self.hparams.test_sample_steps, self.hparams.test_cfg_strength
        )

        *_, synth_fn = batch
        self.test_lsd(preds, inputs, synth_fn)
        self.test_chamfer(preds, targets)
        self.test_lad(preds, targets)
        self.log("test/lsd", self.test_lsd, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "test/chamfer",
            self.test_chamfer,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log("test/lad", self.test_lad, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        # TODO: implement metrics
        # self.log("test/lsd", self.test_lsd, on_step=False, on_epoch=True, prog_bar=True)
        # etc...
        pass

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.vector_field = torch.compile(self.vector_field)
            self.encoder = torch.compile(self.encoder)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())

        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    # "monitor": "val/chamfer",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }

        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = SurgeFlowMatchingModule(None, None, None, None)
