import os
from pathlib import Path
from typing import List, Optional

import click
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rootutils
import torch
from pedalboard.io import AudioFile
from tqdm import tqdm, trange

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.vst import load_plugin, load_preset, render_params
from src.data.vst.param_spec import ParamSpec
from src.data.vst.surge_xt_param_spec import (
    SURGE_MINI_PARAM_SPEC,
    SURGE_SIMPLE_PARAM_SPEC,
    SURGE_XT_PARAM_SPEC,
)


def make_spectrogram(audio: np.ndarray, sample_rate: float) -> np.ndarray:
    channels = audio.shape[0]

    specs = []
    for channel in range(channels):
        spec = librosa.feature.melspectrogram(
            y=audio[channel],
            sr=sample_rate,
            n_mels=128,
            n_fft=2048,
            hop_length=512,
            window="hamming",
        )
        spec_db = librosa.power_to_db(spec, ref=np.max)
        specs.append(spec_db)

    return specs


def write_spectrograms(
    pred_audio: np.ndarray,
    target_audio: np.ndarray,
    sample_rate: float,
    save_path: str,
) -> np.ndarray:
    pred_specs = make_spectrogram(pred_audio, sample_rate)
    target_specs = make_spectrogram(target_audio, sample_rate)

    channels = len(pred_specs) + len(target_specs)

    fig, axs = plt.subplots(channels, 1, figsize=(8, 3 * channels))

    for i, spec in enumerate(pred_specs):
        spec = librosa.amplitude_to_db(spec, ref=np.max)
        librosa.display.specshow(
            spec,
            sr=sample_rate,
            hop_length=512,
            x_axis="time",
            y_axis="mel",
            ax=axs[i],
            cmap="magma",
        )
        axs[i].set_title(f"Pred (Chan {i+1})")

    for i, spec in enumerate(target_specs):
        spec = librosa.amplitude_to_db(spec, ref=np.max)
        librosa.display.specshow(
            spec,
            sr=sample_rate,
            hop_length=512,
            x_axis="time",
            y_axis="mel",
            ax=axs[i + len(pred_specs)],
            cmap="magma",
        )
        axs[i + len(pred_specs)].set_title(f"Target (Chan {i+1})")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def params_to_csv(
    target_params: np.ndarray,
    pred_params: np.ndarray,
    save_path: str,
    param_spec: ParamSpec,
) -> None:
    """Write the target and predicted parameters to a CSV file."""
    row_names = param_spec.names

    df = pd.DataFrame(
        data={"Target": target_params, "Pred": pred_params}, index=row_names
    )
    df.to_csv(save_path)


@click.command()
@click.argument("pred_dir", type=str)
@click.argument("output_dir", type=str)
@click.option("--plugin_path", "-p", type=str, default="plugins/Surge XT.vst3")
@click.option("--preset_path", "-r", type=str, default="presets/surge-base.vstpreset")
@click.option("--sample_rate", "-s", type=float, default=44100.0)
@click.option("--channels", "-c", type=int, default=2)
@click.option("--velocity", "-v", type=int, default=100)
@click.option("--note_duration_seconds", "-n", type=float, default=1.5)
@click.option("--signal_duration_seconds", "-d", type=float, default=4.0)
@click.option("--param_spec", type=str, default="surge_xt")
@click.option("--rerender_target", "-t", is_flag=True, default=False)
@click.option("--exclude", "-x", multiple=True, default=[])
def main(
    pred_dir: str,
    output_dir: str,
    plugin_path: str = "plugins/Surge XT.vst3",
    preset_path: str = "presets/surge-base.vstpreset",
    sample_rate: float = 44100.0,
    channels: int = 2,
    velocity: int = 100,
    note_duration_seconds: float = 1.5,
    signal_duration_seconds: float = 4.0,
    param_spec: str = "surge_xt",
    rerender_target: bool = False,
    exclude: List[str] = [],
):
    if param_spec in ("surge", "surge_xt"):
        param_spec = SURGE_XT_PARAM_SPEC
    elif param_spec in ("mini", "surge_mini", "surge_xt_mini"):
        param_spec = SURGE_MINI_PARAM_SPEC
    elif param_spec in ("simple", "surge_simple", "surge_xt_simple"):
        param_spec = SURGE_SIMPLE_PARAM_SPEC
    else:
        raise ValueError(f"Invalid param_spec: {param_spec}")

    os.makedirs(output_dir, exist_ok=True)

    # 1. load and prepare the VST
    plugin = load_plugin(plugin_path)
    load_preset(plugin, preset_path)

    # 2. list the .pt files with accompanying indices (each file has name
    # pred-{index}.pt, and we want to sort by index)
    pred_dir = Path(pred_dir)
    pred_files = [f for f in pred_dir.glob("pred-*.pt") if f.is_file()]
    indices = [int(f.stem.split("-")[1]) for f in pred_files]
    target_param_files = [pred_dir / f"target-params-{i}.pt" for i in indices]
    target_audio_files = [pred_dir / f"target-audio-{i}.pt" for i in indices]

    # 4. foreach .pt file
    current_offset = 0
    for i, (pred_file, target_param_file, target_audio_file) in tqdm(
        enumerate(zip(pred_files, target_param_files, target_audio_files))
    ):
        pred_params = torch.load(pred_file, map_location="cpu")
        target_params = torch.load(target_param_file, map_location="cpu")
        target_audio = torch.load(target_audio_file, map_location="cpu").numpy()

        # 5. iterate over its internal rows and render the audio
        for j in trange(pred_params.shape[0]):
            file_idx = current_offset + j
            sample_dir = os.path.join(output_dir, f"sample_{file_idx}")
            os.makedirs(sample_dir, exist_ok=True)

            row_params = pred_params[j].numpy()
            row_params_scaled = (row_params + 1) / 2
            row_params_scaled = np.clip(row_params_scaled, 0, 1)
            row_params_dict, note = param_spec.from_numpy(row_params_scaled, exclude=exclude)

            load_preset(plugin, preset_path)
            pred_audio = render_params(
                plugin,
                row_params_dict,
                int(note),
                velocity,
                note_duration_seconds,
                signal_duration_seconds,
                sample_rate,
                channels,
            )

            out_target = os.path.join(sample_dir, "target.wav")
            if rerender_target:
                target_params_ = target_params[j].numpy()
                target_params_ = (target_params_ + 1) / 2
                target_params_ = np.clip(target_params_, 0, 1)
                target_param_dict, target_note = param_spec.from_numpy(target_params_)

                load_preset(plugin, preset_path)
                new_target = render_params(
                    plugin,
                    target_param_dict,
                    int(target_note),
                    velocity,
                    note_duration_seconds,
                    signal_duration_seconds,
                    sample_rate,
                    channels,
                )
                with AudioFile(out_target, "w", sample_rate, channels) as f:
                    f.write(new_target.T)

            else:
                with AudioFile(out_target, "w", sample_rate, channels) as f:
                    f.write(target_audio[j].T)

            out_pred = os.path.join(sample_dir, "pred.wav")
            with AudioFile(out_pred, "w", sample_rate, channels) as f:
                f.write(pred_audio.T)

            write_spectrograms(
                pred_audio,
                target_audio[j],
                sample_rate,
                os.path.join(sample_dir, "spec.png"),
            )

            params_to_csv(
                target_params[j].numpy(),
                row_params,
                os.path.join(sample_dir, "params.csv"),
                param_spec,
            )

        current_offset += pred_params.shape[0]


if __name__ == "__main__":
    main()
