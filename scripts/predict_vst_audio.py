import os
from pathlib import Path
from typing import Optional

import click
import h5py
import rootutils
import torch
from pedalboard.io import AudioFile
from tqdm import tqdm

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.vst import load_plugin, load_preset, render_params
from src.data.vst.surge_xt_param_spec import SURGE_XT_PARAM_SPEC


@click.command()
@click.argument("pred_dir", type=str)
@click.argument("output_dir", type=str)
@click.option("--hdf5_path", "-h", type=str, default=None)
@click.option("--plugin_path", "-p", type=str, default="plugins/Surge XT.vst3")
@click.option("--preset_path", "-r", type=str, default="presets/surge-base.vstpreset")
@click.option("--sample_rate", "-s", type=float, default=44100.0)
@click.option("--channels", "-c", type=int, default=2)
@click.option("--velocity", "-v", type=int, default=100)
@click.option("--note_duration_seconds", "-n", type=float, default=1.5)
@click.option("--signal_duration_seconds", "-d", type=float, default=4.0)
def main(
    pred_dir: str,
    output_dir: str,
    hdf5_path: Optional[str] = None,
    plugin_path: str = "plugins/Surge XT.vst3",
    preset_path: str = "presets/surge-base.vstpreset",
    sample_rate: float = 44100.0,
    channels: int = 2,
    velocity: int = 100,
    note_duration_seconds: float = 1.5,
    signal_duration_seconds: float = 4.0,
):
    # we take in:
    # - a directory of .pt files
    # - an output dir
    # - optionally an hdf5 path
    # with defaults:
    # VST path
    # preset path
    # sample rate
    # channels
    #
    # the pt file contains parameters, aligned with the entries of the hdf5 file
    # so, steps:
    # 1. load and prepare the VST
    # 2. list the .pt files
    # 3. foreach .pt file
    # 4. iterate over its internal rows and render the audio
    # 5. save {i}_pred.wav and {i}_target.wav, with target coming from the hdf5 file
    os.makedirs(output_dir, exist_ok=True)

    # 1. load and prepare the VST
    plugin = load_plugin(plugin_path)
    load_preset(plugin, preset_path)

    # 2. list the .pt files with accompanying indices (each file has name
    # pred-{index}.pt, and we want to sort by index)
    pred_dir = Path(pred_dir)
    pt_files = [f for f in pred_dir.glob("*.pt") if f.is_file()]
    pt_files = sorted(pt_files, key=lambda f: int(f.stem.split("-")[1]))

    # 3. Open the hdf5 file if we have one
    if hdf5_path is not None:
        f = h5py.File(hdf5_path, "r")
        target_audio = f["audio"]
    else:
        target_audio = None

    # 4. foreach .pt file
    current_offset = 0
    for i, pt_file in tqdm(enumerate(pt_files)):
        params = torch.load(pt_file, map_location="cpu")

        if target_audio is not None:
            batch_target = target_audio[
                current_offset : current_offset + params.shape[0]
            ]
        else:
            batch_target = None

        # 5. iterate over its internal rows and render the audio
        for j, row in tqdm(enumerate(params)):
            file_idx = current_offset + j
            pred_audio = render_params(
                plugin,
                row,
                velocity,
                note_duration_seconds,
                signal_duration_seconds,
                sample_rate,
                channels,
            )
            with AudioFile(f"{file_idx}_pred.wav", "w", sample_rate, channels) as f:
                f.write(pred_audio.T)

            # 6. save {file_idx}_pred.wav and {file_idx}_target.wav, with target
            # coming from the hdf5 file
            if batch_target is not None:
                row_target = target_audio[j]
                with AudioFile(
                    f"{file_idx}_target.wav", "w", sample_rate, channels
                ) as f:
                    f.write(row_target.T)

        current_offset += params.shape[0]


if __name__ == "__main__":
    main()
