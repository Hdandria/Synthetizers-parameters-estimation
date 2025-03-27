import math
import os
from pathlib import Path
from typing import Literal, Tuple

import click
import hydra
import matplotlib.pyplot as plt
import numpy as np
import rootutils
import torch
from IPython import embed
from loguru import logger
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.data.vst import param_specs
from src.models.components.transformer import LearntProjection
from src.utils import register_resolvers


def wandb_dir_to_ckpt_and_hparams(
    wandb_dir: Path, ckpt_type: Literal["best", "last"]
) -> Tuple[Path, Path]:
    log_dir = wandb_dir.parent.parent
    ckpt_dir = log_dir / "checkpoints"

    csv_dir = log_dir / "csv"
    hparam_files = csv_dir.glob("*/hparams.yaml")
    hparam_file = max(hparam_files, key=lambda x: x.stat().st_mtime)

    if ckpt_type == "last":
        logger.info(f"Using last checkpoint for {log_dir}")
        ckpt_file = ckpt_dir / "last.ckpt"
    elif ckpt_type == "best":
        logger.info(f"Using best checkpoint for {log_dir}")
        ckpt_files = ckpt_dir.glob("epoch*.ckpt")

        # most recent file
        ckpt_files = sorted(ckpt_files, key=lambda x: x.stat().st_mtime, reverse=True)
        ckpt_file = ckpt_files[0]

    return ckpt_file, hparam_file


def get_state_dict(ckpt_file: Path, map_location: str = "cuda") -> dict:
    logger.info(f"Loading checkpoint from {ckpt_file}")
    ckpt = torch.load(ckpt_file, map_location=map_location, weights_only=False)
    state_dict = ckpt["state_dict"]
    return state_dict


def instantiate_model(
    model_cfg: DictConfig, ckpt_file: Path, map_location: str = "cuda"
) -> torch.nn.Module:

    logger.info(f"Instantiating model from {ckpt_file} with config:")
    logger.info(OmegaConf.to_yaml(model_cfg))
    model = hydra.utils.instantiate(model_cfg)

    logger.info("Model instantiated")
    model.to(device=map_location)

    state_dict = get_state_dict(ckpt_file, map_location=map_location)

    logger.info("Mapping state dict to params")
    model.setup(None)
    model.load_state_dict(state_dict)

    return model


def instantiate_datamodule(data_cfg: DictConfig):
    logger.info("Instantiating datamodule with config:")
    logger.info(OmegaConf.to_yaml(data_cfg))
    dm = hydra.utils.instantiate(data_cfg)
    dm.setup("fit")

    return dm


def sort_assignment(assignment: np.ndarray):
    assignment = np.abs(assignment)
    idxs = np.argsort(assignment, axis=-1)
    idxs = np.lexsort(idxs.T)
    assignment = assignment[idxs]
    return assignment


def longest_matching_initial_substring(a: str, b: str) -> str:
    longest = ""
    for i in range(min(len(a), len(b))):
        if a[i] == b[i]:
            longest = longest + a[i]
        else:
            break

    return longest


def strip_scene_id(param_name: str) -> str:
    if param_name.startswith("a_"):
        return param_name[2:]

    return param_name


PREFIXES = (
    "amp_eg",
    "filter_eg",
    "filter_1",
    "filter_2",
    "waveshaper",
    "osc_1",
    "osc_2",
    "osc_3",
    "lfo_1",
    "lfo_2",
    "lfo_3",
    "lfo_4",
    "lfo_5",
    "lfo_6",
    "noise",
    "ring_modulation_1x2",
    "ring_modulation_2x3",
    "fx_a1",
    "fx_a2",
    "fx_a3",
    "fm",
)

RENAMES = {
    "amp_eg": "Amp. EG",
    "filter_eg": "Filt. EG",
    "feedback": "Feedback",
    "filter_balance": "Filt. Balance",
    "filter_configuration": "Filt. Routing",
    "highpass": "HPF",
    "filter_1": "Filter 1",
    "filter_2": "Filter 2",
    "waveshaper": "Waveshaper",
    "osc_1": "Osc. 1",
    "osc_2": "Osc. 2",
    "osc_3": "Osc. 3",
    "osc_drift": "Osc. Drift",
    "fm": "Freq. Mod.",
    "lfo_1": "LFO 1",
    "lfo_2": "LFO 2",
    "lfo_3": "LFO 3",
    "lfo_4": "LFO 4",
    "lfo_5": "LFO 5",
    "lfo_6": "Pitch EG",
    "noise": "Noise",
    "pan": "Pan",
    "ring_modulation_1x2": "Ring Mod. 1x2",
    "ring_modulation_2x3": "Ring Mod. 2x3",
    "vca_gain": "VCA Gain",
    "width": "Width",
    "fx_a1": "FX: Chorus",
    "fx_a2": "FX: Delay",
    "fx_a3": "FX: Reverb",
    "pitch": "Note Pitch",
    "note_start_and_end": "Note On/Off",
}


def get_labels(spec: str):
    param_spec = param_specs[spec]

    synth_intervals = [(p.name, len(p)) for p in param_spec.synth_params]
    note_intervals = [(p.name, len(p)) for p in param_spec.note_params]
    intervals = synth_intervals + note_intervals

    intervals = [(strip_scene_id(n), l) for n, l in intervals]
    true_intervals = []

    current_prefix = None
    current_prefix_length = 0

    for cur_name, cur_len in intervals:
        should_continue = False
        for prefix in PREFIXES:
            if cur_name.startswith(prefix):
                if prefix != current_prefix and current_prefix is not None:
                    true_intervals.append((current_prefix, current_prefix_length))

                    current_prefix = prefix
                    current_prefix_length = cur_len

                    should_continue = True
                    break
                if prefix == current_prefix:
                    current_prefix_length += cur_len

                    should_continue = True
                    break
                if current_prefix is None:
                    current_prefix = prefix
                    current_prefix_length = cur_len

                    should_continue = True
                    break

        if should_continue:
            continue

        if current_prefix is not None:
            true_intervals.append((current_prefix, current_prefix_length))

        current_prefix = None
        current_prefix_length = 0

        true_intervals.append((cur_name, cur_len))

    true_intervals = [
        (RENAMES.get(name, name), length) for name, length in true_intervals
    ]

    return true_intervals


def add_labels(fig: plt.Figure, ax: plt.Axes, spec: str):
    intervals = get_labels(spec)
    labels = [label for label, _ in intervals]

    centers = []
    start = 0
    for label, length in intervals:
        print(f"{label}: {length}")
        center = start + (length - 1) / 2
        centers.append(center)
        start += length

        ax.axvline(start - 0.5, color="k", alpha=0.5)

    ax.set_xticks(centers)
    ax.set_xticklabels(labels)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    text_objs = ax.get_xticklabels()
    bboxes = [txt.get_window_extent(renderer=renderer) for txt in text_objs]
    y_shift_per_collision = 1  # points to shift for each collision
    current_shift = 0
    last_xend = -1e9  # track right edge of the last label

    denom = math.cos(math.pi / 4)
    min_perp_dist = 100

    for txt, bbox in zip(text_objs, bboxes):
        # if this bbox starts before the last one ends, we have an overlap
        perp_dist = (bbox.x1 - last_xend) / denom

        if perp_dist < min_perp_dist:
            shift = (min_perp_dist - perp_dist) * denom
            current_shift -= shift
        else:
            # reset shift if no overlap
            current_shift = 0
        # move the text by modifying its 'y' position in data coordinates
        # You can also do this in axes or figure fraction coordinates if you prefer.
        x0, y0 = txt.get_position()
        txt.set_position((x0, y0 + current_shift / 72.0))  # 72 points per inch
        last_xend = bbox.x1


def plot_assignment(proj: LearntProjection, spec: str):
    assignment = proj.assignment.detach().cpu().numpy()
    assignment = sort_assignment(assignment)

    ratio = assignment.shape[1] / assignment.shape[0]

    plt.rcParams.update({"font.size": 14})
    fig, ax = plt.subplots(1, 1, figsize=(12 * ratio, 12))

    maxval = np.abs(assignment).max().item()
    img = ax.imshow(
        assignment,
        aspect="equal",
        vmin=-maxval,
        vmax=maxval,
        cmap="RdBu",
    )
    # fig.colorbar(img, ax=ax)

    # ax.set_title("Assignment")

    add_labels(fig, ax, spec)

    ax.set_xlabel("params")
    ax.set_ylabel("tokens")
    fig.tight_layout()
    fig.suptitle("Learnt Assignment")
    fig.tight_layout()

    return fig


def plot_param2tok(proj: LearntProjection, out_dir: str, spec: str):
    logger.info("Plotting assignment")
    assignment_fig = plot_assignment(proj, spec)
    logger.info("Plotting done")
    logger.info(f"Saving to {out_dir}")
    os.makedirs(out_dir, exist_ok=True)
    assignment_fig.savefig(f"{out_dir}/assignment.pdf")
    logger.info("Saved")


@click.command()
@click.argument("wandb_id", type=str)
@click.argument("out_dir", type=str)
@click.option("--spec", "-s", type=str, default="surge_xt")
@click.option("--log-dir", "-l", type=str, default="logs")
@click.option("--ckpt_type", "-c", type=str, default="last")
@click.option("--device", "-d", type=str, default="cuda")
def main(
    wandb_id: str,
    out_dir: str,
    spec: str,
    log_dir: str = "logs",
    ckpt_type: Literal["best", "last"] = "last",
    device: str = "cuda",
):

    register_resolvers()
    log_dir = Path(log_dir)
    possible_wandb_dirs = list(log_dir.glob(f"**/*{wandb_id}/"))
    logger.info(f"Found {len(possible_wandb_dirs)} log dirs matching wandb id")

    ckpts_and_hparams = list(
        map(lambda x: wandb_dir_to_ckpt_and_hparams(x, ckpt_type), possible_wandb_dirs)
    )

    if len(ckpts_and_hparams) > 1:
        # take the one with the most recently updated hparam file
        ckpt_file, hparam_file = max(
            ckpts_and_hparams, key=lambda x: x[1].stat().st_mtime
        )
    elif len(ckpts_and_hparams) == 1:
        ckpt_file, hparam_file = ckpts_and_hparams[0]
    else:
        raise RuntimeError("Could not find wandb id in any of the log directories.")

    cfg = OmegaConf.load(hparam_file)

    model = instantiate_model(cfg.model, ckpt_file, device)
    torch.set_grad_enabled(False)

    plot_param2tok(model.vector_field.projection, out_dir, spec)


if __name__ == "__main__":
    main()
