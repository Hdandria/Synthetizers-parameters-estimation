import hashlib
import random
from dataclasses import dataclass

import click
import h5py
import librosa
import numpy as np
import rootutils
from loguru import logger
from pedalboard import VST3Plugin
from pyloudnorm import Meter
from tqdm import trange

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.vst import load_plugin, load_preset, render_params  # noqa
from src.data.vst.param_spec import ParamSpec  # noqa
from src.data.vst.surge_xt_param_spec import SURGE_MINI_PARAM_SPEC  # noqa
from src.data.vst.surge_xt_param_spec import SURGE_XT_PARAM_SPEC  # noqa
from src.data.vst.surge_xt_param_spec import SURGE_SIMPLE_PARAM_SPEC


def sample_midi_note(min_pitch: int = 32, max_pitch: int = 96):
    # Validate pitch range
    if min_pitch > max_pitch:
        raise ValueError("min_pitch must be less than or equal to max_pitch.")

    pitch = random.randint(min_pitch, max_pitch)
    return pitch


def _hash_params(params: dict[str, float]) -> str:
    param_str = []
    for k, v in params.items():
        param_str.append(f"{k}={v},")
    param_str = "".join(param_str)
    md5 = hashlib.md5()
    md5.update(param_str.encode("utf-8"))
    return md5.hexdigest()


@dataclass
class VSTDataSample:
    parameters: dict[str, float]
    midi_note: int

    min_pitch: int
    max_pitch: int

    sample_rate: float
    channels: int

    param_spec: ParamSpec

    audio: np.ndarray
    mel_spec: np.ndarray
    param_array: np.ndarray = None

    identifier: str = None

    def __post_init__(self):
        self.identifier = _hash_params(self.parameters)
        self.param_array = self.param_spec.to_numpy(
            self.parameters, self.midi_note, self.min_pitch, self.max_pitch
        )


def make_spectrogram(audio: np.ndarray, sample_rate: float) -> np.ndarray:
    """Values hardcoded to be roughly like those used by the audio spectrogram
    transformer. i.e. 100 frames per second, 128 mels, ~25ms window, hamming
    window."""

    n_fft = int(0.025 * sample_rate)
    hop_length = int(sample_rate / 100.0)
    window = "hamming"

    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_mels=128,
        n_fft=n_fft,
        hop_length=hop_length,
        window=window,
    )
    spec_db = librosa.power_to_db(spec, ref=np.max)
    return spec_db


def generate_sample(
    min_pitch: int = 36,
    max_pitch: int = 84,
    velocity: int = 100,
    note_duration_seconds: float = 1.5,
    signal_duration_seconds: float = 4.0,
    sample_rate: float = 44100.0,
    channels: int = 2,
    min_loudness: float = -55.0,
    param_spec: ParamSpec = SURGE_XT_PARAM_SPEC,
    plugin_path: str = "plugins/Surge XT.vst3",
    preset_path: str = "presets/surge-mini.vstpreset",
) -> VSTDataSample:
    while True:
        logger.debug("sampling params")
        params = param_spec.sample()

        logger.debug("sampling note")
        note = sample_midi_note(
            min_pitch=min_pitch,
            max_pitch=max_pitch,
        )

        plugin = load_plugin(plugin_path)
        output = render_params(
            plugin,
            params,
            note,
            velocity,
            note_duration_seconds,
            signal_duration_seconds,
            sample_rate,
            channels,
            preset_path=preset_path,
        )

        meter = Meter(sample_rate)
        loudness = meter.integrated_loudness(output.T)
        logger.debug(f"loudness: {loudness}")
        if loudness < min_loudness:
            logger.debug("loudness too low, skipping")
            continue

        break

    logger.debug("making spectrogram")
    spectrogram = make_spectrogram(output, sample_rate)

    return VSTDataSample(
        parameters=params,
        midi_note=note,
        audio=output.T,
        mel_spec=spectrogram,
        min_pitch=min_pitch,
        max_pitch=max_pitch,
        sample_rate=sample_rate,
        channels=channels,
        param_spec=param_spec,
    )


def save_sample(
    sample: VSTDataSample,
    audio_dataset: h5py.Dataset,
    mel_dataset: h5py.Dataset,
    param_dataset: h5py.Dataset,
    idx: int,
) -> None:
    audio_dataset[idx, :, :] = sample.audio.T
    mel_dataset[idx, :, :] = sample.mel_spec
    param_dataset[idx, :] = sample.param_array


def make_dataset(
    hdf5_file: h5py.File,
    num_samples: int,
    plugin_path: str = "plugins/Surge XT.vst3",
    preset_path: str = "presets/surge-base.vstpreset",
    sample_rate: float = 44100.0,
    channels: int = 2,
    min_pitch: int = 36,
    max_pitch: int = 84,
    velocity: int = 100,
    note_duration_seconds: float = 1.5,
    signal_duration_seconds: float = 4.0,
    min_loudness: float = -55.0,
    param_spec: ParamSpec = SURGE_XT_PARAM_SPEC,
) -> None:

    audio_dataset = hdf5_file.create_dataset(
        "audio",
        (num_samples, channels, sample_rate * signal_duration_seconds),
        dtype=np.float16,
        compression="gzip",
    )
    mel_dataset = hdf5_file.create_dataset(
        "mel_spec",
        (num_samples, 2, 128, 401),
        dtype=np.float32,
        compression="gzip",
    )
    param_dataset = hdf5_file.create_dataset(
        "param_array",
        (num_samples, len(param_spec) + 1),  # +1 for MIDI note
        dtype=np.float32,
        compression="gzip",
    )

    audio_dataset.attrs["min_pitch"] = min_pitch
    audio_dataset.attrs["max_pitch"] = max_pitch
    audio_dataset.attrs["velocity"] = velocity
    audio_dataset.attrs["note_duration_seconds"] = note_duration_seconds
    audio_dataset.attrs["signal_duration_seconds"] = signal_duration_seconds
    audio_dataset.attrs["sample_rate"] = sample_rate
    audio_dataset.attrs["channels"] = channels
    audio_dataset.attrs["min_loudness"] = min_loudness

    for i in trange(num_samples):
        sample = generate_sample(
            min_pitch=min_pitch,
            max_pitch=max_pitch,
            velocity=velocity,
            note_duration_seconds=note_duration_seconds,
            signal_duration_seconds=signal_duration_seconds,
            sample_rate=sample_rate,
            channels=channels,
            min_loudness=min_loudness,
            param_spec=param_spec,
            plugin_path=plugin_path,
            preset_path=preset_path,
        )
        save_sample(sample, audio_dataset, mel_dataset, param_dataset, i)


@click.command()
@click.argument("data_file", type=str, required=True)
@click.argument("num_samples", type=int, required=True)
@click.option("--plugin_path", "-p", type=str, default="plugins/Surge XT.vst3")
@click.option("--preset_path", "-r", type=str, default="presets/surge-base.vstpreset")
@click.option("--sample_rate", "-s", type=float, default=44100.0)
@click.option("--channels", "-c", type=int, default=2)
@click.option("--min_pitch", "-m", type=int, default=36)
@click.option("--max_pitch", "-M", type=int, default=84)
@click.option("--velocity", "-v", type=int, default=100)
@click.option("--note_duration_seconds", "-n", type=float, default=1.5)
@click.option("--signal_duration_seconds", "-d", type=float, default=4.0)
@click.option("--min_loudness", "-l", type=float, default=-55.0)
@click.option("--param_spec", "-t", type=str, default="surge_xt")
def main(
    data_file: str,
    num_samples: int,
    plugin_path: str = "plugins/Surge XT.vst3",
    preset_path: str = "presets/surge-base.vstpreset",
    sample_rate: float = 44100.0,
    channels: int = 2,
    min_pitch: int = 36,
    max_pitch: int = 84,
    velocity: int = 100,
    note_duration_seconds: float = 1.5,
    signal_duration_seconds: float = 4.0,
    min_loudness: float = -55.0,
    param_spec: str = "surge_xt",
):
    if param_spec in ("surge", "surge_xt"):
        param_spec = SURGE_XT_PARAM_SPEC
    elif param_spec in ("mini", "surge_mini", "surge_xt_mini"):
        param_spec = SURGE_MINI_PARAM_SPEC
    elif param_spec in ("simple", "surge_simple", "surge_xt_simple"):
        param_spec = SURGE_SIMPLE_PARAM_SPEC
    else:
        raise ValueError(f"Invalid param_spec: {param_spec}")

    with h5py.File(data_file, "w") as f:
        make_dataset(
            f,
            num_samples,
            plugin_path,
            preset_path,
            sample_rate,
            channels,
            min_pitch,
            max_pitch,
            velocity,
            note_duration_seconds,
            signal_duration_seconds,
            min_loudness,
            param_spec,
        )


if __name__ == "__main__":
    main()
