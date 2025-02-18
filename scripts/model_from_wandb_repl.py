from pathlib import Path
from typing import Literal, Tuple

import click
import hydra
import torch
from IPython import embed
from loguru import logger
from omegaconf import DictConfig, OmegaConf


def wandb_dir_to_ckpt_and_hparams(
    wandb_dir: Path, ckpt_type: Literal["best", "last"]
) -> Tuple[Path, Path]:
    log_dir = wandb_dir.parent
    hparam_file = log_dir / "csv" / "hparams.yaml"
    ckpt_dir = log_dir / "checkpoints"

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
    ckpt = torch.load(ckpt_file, map_location=map_location, weight_only=False)
    state_dict = ckpt["state_dict"]
    return state_dict


def instantiate_model(
    model_cfg: DictConfig, ckpt_file: Path, map_location: str = "cuda"
) -> torch.nn.Module:
    logger.info(f"Instantiating model from {ckpt_file} with config:")
    logger.info(OmegaConf.to_yaml(model_cfg))

    state_dict = get_state_dict(ckpt_file, map_location=map_location)

    model = hydra.utils.instantiate(model_cfg)
    model.to(device=map_location)

    model.load_state_dict(state_dict)

    return model


def instantiate_datamodule(data_cfg: DictConfig):
    logger.info("Instantiating datamodule with config:")
    logger.info(OmegaConf.to_yaml(data_cfg))
    return hydra.utils.instantiate(data_cfg)


@click.command()
@click.argument("wandb_id", type=str)
@click.option("--log-dir", "-l", type=str, default="logs")
@click.option("--ckpt_type", "-c", type=str, default="last")
@click.option("--device", "-d", type=str, default="cuda")
def main(
    wandb_id: str,
    log_dir: str = "logs",
    ckpt_type: Literal["best", "last"] = "last",
    device: str = "cuda",
):
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
        ckpt_file, hparam_file = next(ckpts_and_hparams)
    else:
        raise RuntimeError("Could not find wandb id in any of the log directories.")

    cfg = OmegaConf.load(hparam_file)

    model = instantiate_model(cfg.model, ckpt_file, device)
    datamodule = instantiate_datamodule(cfg.data)

    logger.info("Starting REPL...")
    embed(user_ns={"model": model, "datamodule": datamodule, "cfg": cfg})


if __name__ == "__main__":
    main()
