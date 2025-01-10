import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from einops import rearrange
from lightning.pytorch.callbacks import Callback

from src.models.components.transformer import (ApproxEquivTransformer,
                                               LearntProjection,
                                               LearntProjectionII,
                                               PositionalEncoding)
from src.models.ksin_flow_matching_module import KSinFlowMatchingModule


class PlotLossPerTimestep(Callback):
    """Takes a batch from the validation dataloader, and runs it through the model at
    a number of different values for t. Plots the loss as a function of t.
    """

    def __init__(self, num_timesteps: int = 100):
        super().__init__()
        self.num_timesteps = num_timesteps

    def _get_val_batch(self, trainer):
        val_dl = trainer.val_dataloaders
        return next(iter(val_dl))

    def _compute_losses(self, trainer, pl_module):
        batch = self._get_val_batch(trainer)
        signal, params, _ = batch

        # Get conditioning vector
        conditioning = pl_module.encoder(signal)
        z = pl_module.vector_field.apply_dropout(
            conditioning, pl_module.hparams.cfg_dropout_rate
        )

        x0, x1, z = pl_module._sample_x0_and_x1(params, z)

        losses = []
        for n in range(self.num_timesteps):
            t = torch.full(
                (signal.shape[0], 1), n / (self.num_timesteps - 1), device=signal.device
            )
            x_t = pl_module._sample_probability_path(x0, x1, t)
            target = pl_module._evaluate_target_field(x0, x1, x_t, t)

            prediction = pl_module.vector_field(x_t, t, z)
            loss = (prediction - target).square().mean(dim=-1)
            losses.append(loss)

        return torch.stack(losses, dim=-1)

    def _aggregate_losses(self, losses):
        mean = losses.mean(dim=0)
        std = losses.std(dim=0)
        lower_ci = mean - 2 * std
        upper_ci = mean + 2 * std
        return mean, lower_ci, upper_ci

    def _plot_losses(self, losses):
        t = np.linspace(0, 1, self.num_timesteps)
        mean, lower_ci, upper_ci = self._aggregate_losses(losses)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(t, mean.cpu().numpy())
        ax.fill_between(t, lower_ci.cpu().numpy(), upper_ci.cpu().numpy(), alpha=0.2)
        ax.set_xlabel("t")
        ax.set_ylabel("Loss")
        ax.set_title("Loss per noise level / timestep")
        return fig

    def _log_plot(self, fig, trainer):
        plot = wandb.Image(fig)
        wandb.log({"plot": plot}, step=trainer.global_step)
        plt.close(fig)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        losses = self._compute_losses(trainer, pl_module)
        fig = self._plot_losses(losses)
        self._log_plot(fig, trainer)


def _self_similarity(x):
    y = x.permute(1, 0, 2)
    sim = torch.nn.functional.cosine_similarity(x, y, dim=-1)
    return sim


class PlotPositionalEncodingSimilarity(Callback):
    def _compute_similarity(self, pl_module):
        if pl_module.vector_field.pe_type == "initial":
            return _self_similarity(pl_module.vector_field.pe.pe)
        elif pl_module.vector_field.pe_type == "layerwise":
            return [_self_similarity(pe.pe) for pe in pl_module.vector_field.pe]

    def _plot_single_similarity(self, sim):
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))

        ax.imshow(sim.cpu().numpy(), vmin=-1, vmax=1, aspect="equal")
        fig.tight_layout()
        fig.suptitle("Positional Encoding Similarity")

        return fig

    def _plot_multiple_similarities(self, sims):
        n_pe = len(sims)
        n_rows = int(np.sqrt(n_pe))
        n_cols = int(np.ceil(n_pe / n_rows))

        fig, ax = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 2 * n_rows))

        for i, sim in enumerate(sims):
            ax[i // n_cols, i % n_cols].imshow(
                sim.cpu().numpy(), vmin=-1, vmax=1, aspect="equal"
            )
            ax[i // n_cols, i % n_cols].set_title(
                f"PE {i // n_cols}-{i % n_cols}", fontsize=8
            )

        for i in range(n_pe, n_rows * n_cols):
            ax[i // n_cols, i % n_cols].axis("off")

        fig.tight_layout()
        fig.suptitle("Positional Encoding Similarities")

        return fig

    def _plot_similarity(self, sim):
        if isinstance(sim, torch.Tensor):
            return self._plot_single_similarity(sim)
        else:
            return self._plot_multiple_similarities(sim)

    def _log_plot(self, fig, trainer):
        plot = wandb.Image(fig)
        wandb.log({"pos_enc_similarity": plot}, step=trainer.global_step)
        plt.close(fig)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not isinstance(pl_module, KSinFlowMatchingModule):
            return

        if not isinstance(pl_module.vector_field, ApproxEquivTransformer):
            return

        if pl_module.vector_field.pe_type == "none":
            return

        pe_sims = self._compute_similarity(pl_module)
        fig = self._plot_similarity(pe_sims)
        self._log_plot(fig, trainer)


class PlotLearntProjection(Callback):
    def _compute_grids(self, pl_module):
        in_proj = pl_module.vector_field.projection.proj_in
        out_proj = pl_module.vector_field.projection.proj_out

        in_grid = torch.norm(in_proj, p=2.0, dim=1)
        if out_proj is not None:
            out_grid = torch.norm(out_proj, p=2.0, dim=0)
        else:
            out_grid = None

        return in_grid, out_grid

    def _compute_similarity(self, pl_module):
        in_proj = pl_module.vector_field.projection.proj_in  # K x D x N
        num_tokens = in_proj.size(0)
        num_params = in_proj.size(2)

        in_proj = rearrange(in_proj, "k d n -> 1 (k n) d")
        in_sim = _self_similarity(in_proj)

        out_proj = pl_module.vector_field.projection.proj_out  # D x K x N
        if out_proj is not None:
            out_proj = rearrange(out_proj, "d k n -> 1 (k n) d")
            out_sim = _self_similarity(out_proj)
        else:
            out_sim = None

        return in_sim, out_sim, num_tokens, num_params

    def _plot_grids(self, in_grid, out_grid):
        fig, ax = plt.subplots(2, 1, figsize=(5, 5))

        ax[0].imshow(in_grid.cpu().numpy(), aspect="equal")
        ax[0].set_title("In Projection")

        if out_grid is not None:
            ax[1].imshow(out_grid.cpu().numpy(), aspect="equal")
            ax[1].set_title("Out Projection")

        ax[0].set_xlabel("params")
        ax[1].set_xlabel("params")

        ax[0].set_ylabel("tokens")
        ax[1].set_ylabel("tokens")

        fig.tight_layout()
        fig.suptitle("Learnt Projection")

        return fig

    def _plot_similarities(self, in_sim, out_sim, num_tokens, num_params):
        num_plots = 2 if out_sim is not None else 1
        fig, ax = plt.subplots(num_plots, 1, figsize=(5, 5 * num_plots), squeeze=False)

        ax[0, 0].imshow(in_sim.cpu().numpy(), aspect="equal")
        ax[0, 0].set_title("In Projection")

        p_ticks = range(num_params * num_tokens)
        p_labels = [f"P{i+1}" for i in range(num_params)] * num_tokens

        t_ticks = [i * num_params for i in range(num_tokens)]
        t_labels = [f"\nT{i+1}" for i in range(num_tokens)]

        ax[0, 0].set_xticks(p_ticks, labels=p_labels)
        ax[0, 0].secondary_xaxis(location=0).set_xticks(t_ticks, labels=t_labels)

        ax[0, 0].set_yticks(p_ticks, labels=p_labels)
        ax[0, 0].secondary_yaxis(location=0).set_yticks(t_ticks, labels=t_labels)

        if out_sim is not None:
            ax[1, 0].imshow(out_sim.cpu().numpy(), aspect="equal")
            ax[1, 0].set_title("Out Projection")

            ax[1, 0].set_xticks(p_ticks, labels=p_labels)
            ax[1, 0].secondary_xaxis(location=0).set_xticks(t_ticks, labels=t_labels)

            ax[1, 0].set_yticks(p_ticks, labels=p_labels)
            ax[1, 0].secondary_yaxis(location=0).set_yticks(t_ticks, labels=t_labels)

        fig.tight_layout()
        fig.suptitle("Learnt Projection Similarity")

        return fig

    def _log_plots(self, fig_grids, fig_sims, trainer):
        plot_grids = wandb.Image(fig_grids)
        plot_sims = wandb.Image(fig_sims)
        wandb.log(
            {"projection": plot_grids, "proj_similarity": plot_sims},
            step=trainer.global_step,
        )
        plt.close(fig_grids)
        plt.close(fig_sims)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not isinstance(pl_module, KSinFlowMatchingModule):
            return

        if not isinstance(pl_module.vector_field, ApproxEquivTransformer):
            return

        if not isinstance(pl_module.vector_field.projection, LearntProjection):
            return

        grids = self._compute_grids(pl_module)
        sims = self._compute_similarity(pl_module)
        fig_grids = self._plot_grids(*grids)
        fig_sims = self._plot_similarities(*sims)
        self._log_plots(fig_grids, fig_sims, trainer)


class PlotLearntProjectionII(Callback):
    def _get_assignment(self, pl_module):
        return pl_module.vector_field.projection.assignment

    def _plot_assignments(self, pl_module):
        assignment = self._get_assignment(pl_module)
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        ax.imshow(assignment.cpu().numpy(), aspect="equal")
        ax.set_title("Assignment")

        ax.set_xlabel("params")
        ax.set_ylabel("tokens")
        fig.tight_layout()
        fig.suptitle("Learnt Assignment")

        return fig

    def _get_value_similarity(self, pl_module):
        proj = (
            pl_module.vector_field.projection.value_encoding
        )  # num_params x d_embed x d_model

        sim_proj = torch.nn.functional.cosine_similarity(
            proj[None], proj[:, None], dim=-1
        )

        return sim_proj

    def _get_output_similarity(self, pl_module):
        proj = (
            pl_module.vector_field.projection.out_projection.T
        )  # num_params x d_embed x d_model

        sim_proj = torch.nn.functional.cosine_similarity(
            proj[None], proj[:, None], dim=-1
        )

        return sim_proj

    def _plot_projections(self, pl_module):
        fig, ax = plt.subplots(2, 1, figsize=(5, 10))

        val_sim = self._get_value_similarity(pl_module)
        out_sim = self._get_output_similarity(pl_module)

        ax[0].imshow(val_sim.cpu().numpy(), aspect="equal")
        ax[0].set_title("Value Projection")
        ax[0].set_xlabel("params")
        ax[0].set_ylabel("params")

        ax[1].imshow(out_sim.cpu().numpy(), aspect="equal")
        ax[1].set_title("Out Projection")
        ax[1].set_xlabel("params")
        ax[1].set_ylabel("params")

        fig.tight_layout()

        return fig

    def _log_plots(self, fig_ass, fig_value, trainer):
        plot_ass = wandb.Image(fig_ass)
        plot_value = wandb.Image(fig_value)
        wandb.log(
            {"assignment": plot_ass, "value": plot_value}, step=trainer.global_step
        )

        plt.close(fig_ass)
        plt.close(fig_value)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if not isinstance(pl_module, KSinFlowMatchingModule):
            return

        if not isinstance(pl_module.vector_field, ApproxEquivTransformer):
            return

        if not isinstance(pl_module.vector_field.projection, LearntProjectionII):
            return

        fig_ass = self._plot_assignments(pl_module)
        fig_value = self._plot_projections(pl_module)
        self._log_plots(fig_ass, fig_value, trainer)
