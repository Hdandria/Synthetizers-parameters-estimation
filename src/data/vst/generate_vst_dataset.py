import _thread
import hashlib
import random
import threading
import time
from concurrent.futures import ProcessPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import click
import h5py
import librosa
import mido
import numpy as np
import rootutils
from loguru import logger
from mpi4py import MPI
from pedalboard import VST3Plugin
from pedalboard.io import AudioFile
from pyloudnorm import Meter
from tqdm import trange

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.vst.param_spec import ParamSpec
from src.data.vst.surge_xt_param_spec import SURGE_XT_PARAM_SPEC


def call_with_interrupt(fn: Callable, sleep_time: float = 2.0):
    """
    Calls the function fn on the main thread, while another thread
    sends a KeyboardInterrupt (SIGINT) to the main thread.
    """

    def send_interrupt():
        # Brief sleep so that fn starts before we send the interrupt
        time.sleep(sleep_time)
        _thread.interrupt_main()

    # Create and start the thread that sends the interrupt
    t = threading.Thread(target=send_interrupt)
    t.start()

    try:
        fn()
    except KeyboardInterrupt:
        print("Interrupted main thread.")
    finally:
        t.join()


def prepare_plugin(plugin: VST3Plugin) -> None:
    call_with_interrupt(plugin.show_editor)


def load_plugin(plugin_path: str) -> VST3Plugin:
    logger.info(f"Loading plugin {plugin_path}")
    p = VST3Plugin(plugin_path)
    logger.info(f"Plugin {plugin_path} loaded")
    logger.info("Preparing plugin for preset load...")
    prepare_plugin(p)
    logger.info("Plugin ready")
    return p


def load_preset(plugin: VST3Plugin, preset_path: str) -> None:
    logger.info(f"Loading preset {preset_path}")
    plugin.load_preset(preset_path)
    logger.info(f"Preset {preset_path} loaded")


def set_params(plugin: VST3Plugin, params: dict[str, float]) -> None:
    for k, v in params.items():
        plugin.parameters[k].raw_value = v


def sample_midi_note(
    min_pitch: int = 32,
    max_pitch: int = 96,
    velocity: int = 100,
    duration_seconds: float = 1.0,
):
    """
    Creates a MIDI sequence with a single note on followed by a note off after one second.

    Parameters:
        min_pitch (int): The minimum pitch value (inclusive).
        max_pitch (int): The maximum pitch value (inclusive).
        velocity (int): The velocity of the note (default is 100).

    Returns:
        list of tuples: Each tuple contains MIDI bytes and timestamp in seconds.
    """
    # Validate pitch range
    if min_pitch > max_pitch:
        raise ValueError("min_pitch must be less than or equal to max_pitch.")

    pitch = random.randint(min_pitch, max_pitch)

    events = []
    note_on = mido.Message("note_on", note=pitch, velocity=velocity, time=0)
    events.append((note_on.bytes(), 0.0))
    note_off = mido.Message("note_off", note=pitch, velocity=velocity, time=0)
    events.append((note_off.bytes(), duration_seconds))

    return tuple(events), pitch


def _hash_params(params: dict[str, float]) -> str:
    param_str = []
    for k, v in params.items():
        param_str.append(f"{k}={v},")
    param_str = "".join(param_str)
    md5 = hashlib.md5()
    md5.update(param_str.encode("utf-8"))
    return md5.hexdigest()


def _hash_float(v: float) -> str:
    md5 = hashlib.md5()
    md5.update(str(v).encode("utf-8"))
    return md5.hexdigest()


@dataclass
class VSTDataSample:
    parameters: dict[str, float]
    midi_note: int

    min_pitch: int
    max_pitch: int

    sample_rate: float
    channels: int

    audio: np.ndarray
    mel_spec: np.ndarray
    param_array: np.ndarray = None

    identifier: str = None

    def __post_init__(self):
        self.identifier = _hash_params(self.parameters)
        self.param_array = SURGE_XT_PARAM_SPEC.to_numpy(
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
    plugin: VST3Plugin,
    min_pitch: int = 36,
    max_pitch: int = 84,
    velocity: int = 100,
    note_duration_seconds: float = 1.5,
    signal_duration_seconds: float = 4.0,
    sample_rate: float = 44100.0,
    channels: int = 2,
    min_loudness: float = -55.0,
) -> VSTDataSample:
    while True:
        logger.debug("flushing plugin")
        plugin.process([], 1.0, sample_rate, channels, 8192, True)  # flush

        logger.debug("sampling params")
        params = SURGE_XT_PARAM_SPEC.sample()

        logger.debug("setting params")
        set_params(plugin, params)

        logger.debug("sampling note")
        events, note = sample_midi_note(
            min_pitch=min_pitch,
            max_pitch=max_pitch,
            velocity=velocity,
            duration_seconds=note_duration_seconds,
        )

        logger.debug("rendering audio")
        output = plugin.process(
            events, signal_duration_seconds, sample_rate, channels, 8192, True
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
    )


def write_wav(audio: np.ndarray, path: str, sample_rate: float, channels: int) -> None:
    with AudioFile(str(path), "w", sample_rate, channels) as f:
        f.write(audio.T)


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
    offset: int,
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
) -> None:
    plugin = load_plugin(plugin_path)
    load_preset(plugin, preset_path)

    audio_dataset = hdf5_file.create_dataset(
        "audio",
        (num_samples, channels, sample_rate * signal_duration_seconds),
        dtype=np.float32,
        compression="lzf",
    )
    mel_dataset = hdf5_file.create_dataset(
        "mel_spec",
        (num_samples, 400, 128),
        dtype=np.float32,
        compression="lzf",
    )
    param_dataset = hdf5_file.create_dataset(
        "param_array",
        (num_samples, len(SURGE_XT_PARAM_SPEC) + 1),  # +1 for MIDI note
        dtype=np.float32,
        compression="lzf",
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
            plugin,
            min_pitch=min_pitch,
            max_pitch=max_pitch,
            velocity=velocity,
            note_duration_seconds=note_duration_seconds,
            signal_duration_seconds=signal_duration_seconds,
            sample_rate=sample_rate,
            channels=channels,
            min_loudness=min_loudness,
        )
        save_sample(sample, audio_dataset, mel_dataset, param_dataset, offset + i)


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
):
    comm = MPI.COMM_WORLD
    num_workers = comm.Get_size()
    rank = comm.Get_rank()

    samples_per_worker = num_samples // num_workers

    with h5py.File(data_file, "w", driver="mpio", comm=comm) as f:
        offset = rank * samples_per_worker
        make_dataset(
            f,
            samples_per_worker,
            offset,
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
        )


if __name__ == "__main__":
    main()
