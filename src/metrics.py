from typing import Callable

import torch
from scipy.optimize import linear_sum_assignment
from torchmetrics import Metric

from src.models.components.loss import chamfer_loss, params_to_tokens


def complex_to_dbfs(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Convert complex frequency domain signal to dBFS (decibels relative to full scale).
    
    This function computes the power spectral density in dBFS from complex frequency
    domain data, which is commonly used for audio quality assessment.
    
    Args:
        z: Complex tensor representing frequency domain signal
        eps: Small epsilon value to prevent log(0) in dB calculation
        
    Returns:
        Tensor containing power spectral density in dBFS
        
    Example:
        ```python
        # Convert FFT output to dBFS
        fft_output = torch.fft.rfft(audio_signal)
        dbfs = complex_to_dbfs(fft_output)
        ```
    """
    squared_modulus = z.real.square() + z.imag.square()
    clamped = torch.clamp(squared_modulus, min=eps)
    return 10 * torch.log10(clamped)


class LogSpectralDistance(Metric):
    """Log Spectral Distance (LSD) metric for audio quality assessment.
    
    LSD measures the perceptual difference between two audio signals in the
    frequency domain. It's computed as the root mean square of the difference
    between log power spectral densities, making it suitable for evaluating
    audio synthesis quality.
    
    Args:
        eps: Small epsilon value to prevent numerical issues in log calculation
        **kwargs: Additional arguments passed to the base Metric class
        
    Example:
        ```python
        lsd_metric = LogSpectralDistance()
        
        # Update with predicted and target signals
        lsd_metric.update(predicted_params, target_audio, synth_function)
        
        # Compute final metric
        lsd_value = lsd_metric.compute()
        ```
    """
    
    def __init__(self, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.add_state("lsd", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.eps = eps

    def update(
        self,
        predicted_params: torch.Tensor,
        target_signal: torch.Tensor,
        synth_fn: Callable,
    ):
        """Update the LSD metric with a new batch of predictions.
        
        Args:
            predicted_params: Predicted synthesizer parameters
            target_signal: Ground truth audio signal
            synth_fn: Function that synthesizes audio from parameters
        """
        pred_signal = synth_fn(predicted_params)

        pred_fft = torch.fft.rfft(pred_signal, norm="forward")
        target_fft = torch.fft.rfft(target_signal, norm="forward")

        pred_power = complex_to_dbfs(pred_fft, self.eps)
        target_power = complex_to_dbfs(target_fft, self.eps)

        self.lsd += (pred_power - target_power).square().mean(dim=-1).sqrt().mean()
        self.count += 1

    def compute(self) -> torch.Tensor:
        """Compute the final LSD metric value.
        
        Returns:
            Average Log Spectral Distance across all updates
        """
        lsd = self.lsd / self.count
        return lsd


class SpectralDistance(Metric):
    """Spectral Distance (SD) metric for audio quality assessment.
    
    SD measures the L1 distance between magnitude spectra of predicted and target
    audio signals. This provides a simpler alternative to LSD that doesn't require
    log transformations.
    
    Args:
        eps: Small epsilon value (currently unused but kept for consistency)
        **kwargs: Additional arguments passed to the base Metric class
        
    Example:
        ```python
        sd_metric = SpectralDistance()
        
        # Update with predicted and target signals
        sd_metric.update(predicted_params, target_audio, synth_function)
        
        # Compute final metric
        sd_value = sd_metric.compute()
        ```
    """
    
    def __init__(self, eps: float = 1e-8, **kwargs):
        super().__init__(**kwargs)
        self.add_state("sd", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.eps = eps

    def update(
        self,
        predicted_params: torch.Tensor,
        target_signal: torch.Tensor,
        synth_fn: Callable,
    ):
        """Update the SD metric with a new batch of predictions.
        
        Args:
            predicted_params: Predicted synthesizer parameters
            target_signal: Ground truth audio signal
            synth_fn: Function that synthesizes audio from parameters
        """
        pred_signal = synth_fn(predicted_params)

        pred_fft = torch.fft.rfft(pred_signal, norm="forward")
        target_fft = torch.fft.rfft(target_signal, norm="forward")

        pred_mag = pred_fft.abs()
        target_mag = target_fft.abs()

        self.sd += torch.nn.functional.l1_loss(pred_mag, target_mag)
        self.count += 1

    def compute(self) -> torch.Tensor:
        """Compute the final SD metric value.
        
        Returns:
            Average Spectral Distance across all updates
        """
        return self.sd / self.count


class ChamferDistance(Metric):
    """Chamfer Distance metric for permutation-invariant parameter comparison.
    
    Chamfer distance is a permutation-invariant metric that measures the distance
    between two sets of parameters by finding the optimal assignment between them.
    This is particularly useful for evaluating models that predict parameters with
    permutation symmetries (e.g., multiple oscillators in a synthesizer).
    
    Args:
        params_per_token: Number of parameters per token/component
        **kwargs: Additional arguments passed to the base Metric class
        
    Example:
        ```python
        chamfer_metric = ChamferDistance(params_per_token=2)
        
        # Update with predicted and target parameters
        chamfer_metric.update(predicted_params, target_params)
        
        # Compute final metric
        chamfer_value = chamfer_metric.compute()
        ```
    """
    
    def __init__(self, params_per_token: int, **kwargs):
        super().__init__(**kwargs)
        self.add_state(
            "chamfer_distance", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.params_per_token = params_per_token

    def update(self, predicted: torch.Tensor, target: torch.Tensor):
        """Update the Chamfer distance metric with a new batch.
        
        Args:
            predicted: Predicted parameters tensor
            target: Ground truth parameters tensor
        """
        self.chamfer_distance += chamfer_loss(predicted, target, self.params_per_token)
        self.count += 1

    def compute(self) -> torch.Tensor:
        """Compute the final Chamfer distance metric value.
        
        Returns:
            Average Chamfer distance across all updates
        """
        return self.chamfer_distance / self.count


class LinearAssignmentDistance(Metric):
    """Linear Assignment Distance metric for permutation-invariant evaluation.
    
    This metric uses the Hungarian algorithm to find the optimal assignment between
    predicted and target parameter sets, then computes the average distance under
    this optimal assignment. This provides a more accurate measure than Chamfer
    distance for cases where the number of components is small.
    
    Args:
        params_per_token: Number of parameters per token/component
        **kwargs: Additional arguments passed to the base Metric class
        
    Example:
        ```python
        lad_metric = LinearAssignmentDistance(params_per_token=2)
        
        # Update with predicted and target parameters
        lad_metric.update(predicted_params, target_params)
        
        # Compute final metric
        lad_value = lad_metric.compute()
        ```
    """
    
    def __init__(self, params_per_token: int, **kwargs):
        super().__init__(**kwargs)
        self.add_state(
            "linear_assignment_distance",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")
        self.params_per_token = params_per_token

    def update(self, predicted: torch.Tensor, target: torch.Tensor):
        """Update the Linear Assignment Distance metric with a new batch.
        
        Args:
            predicted: Predicted parameters tensor
            target: Ground truth parameters tensor
        """
        predicted_tokens = params_to_tokens(predicted, self.params_per_token)
        target_tokens = params_to_tokens(target, self.params_per_token)

        dist = torch.cdist(predicted_tokens, target_tokens)
        dist_c = dist.detach().cpu()

        cost = 0.0
        for b in range(dist_c.shape[0]):
            row_ind, col_ind = linear_sum_assignment(dist_c[b])
            cost = cost + dist[b, row_ind, col_ind].mean()

        self.count += dist.shape[0]
        self.linear_assignment_distance += cost

    def compute(self) -> torch.Tensor:
        """Compute the final Linear Assignment Distance metric value.
        
        Returns:
            Average Linear Assignment Distance across all updates
        """
        return self.linear_assignment_distance / self.count
