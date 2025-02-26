"""Runs evaluations in the paper.
Expects audio in the following folder structure:

audio/
    sample_0/
        target.wav
        pred.wav
        ...
    sample_1/
        ...
    ...

We compute the following metrics:

1. MSS: log-Mel multi-scale spectrogram (10ms, 25ms, 100ms) windows and
    (5ms, 10ms, 50ms) hop lengths, (32, 64, 128) mels, hann window, L1 distance.
2. JTFS: joint time-frequency scattering transform, L1 distance.
3. wMFCC: dynamic time-warping cost between MFCCs (50ms window, 10ms hop), 128 mels, L1 distance
4. f0 features: intermediate features from some sort of pitch NN (check speech
    literature for an option here?). cosine sim.
5. amp env: compute RMS amp envelopes (50ms window, 25ms hop). take cosine similarity
    (i.e. normalized dot prod).
"""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

import click
import librosa
import numpy as np
import torch
import pesto
import pandas as pd
from dtw import dtw
from kymatio.numpy import Scattering1D
from loguru import logger
from pedalboard.io import AudioFile


def subdir_matches_pattern(dir: Path) -> bool:
    """Returns true if subdir contains pred.wav and target.wav"""
    return (dir / "target.wav").exists() and (dir / "pred.wav").exists()


def find_possible_subdirs(audio_dir: Path) -> List[Path]:
    all_subdirectories = [d for d in audio_dir.glob("*") if d.is_dir()]
    matching_dirs = [d for d in all_subdirectories if subdir_matches_pattern(d)]
    return matching_dirs


MEL_PARAMS = [
    (10, 5, 32),
    (25, 10, 64),
    (100, 50, 128),
]


def compute_mel_specs(y: np.ndarray, sample_rate: float = 44100.0):
    mel_specs = []
    for window_size, hop_size, n_mels in MEL_PARAMS:
        window_size = int(window_size * sample_rate / 1000.0)
        hop_size = int(hop_size * sample_rate / 1000.0)

        spec = librosa.feature.melspectrogram(
            y=y,
            sr=sample_rate,
            n_mels=n_mels,
            n_fft=window_size,
            hop_length=hop_size,
            window="hann",
        )
        spec_db = librosa.power_to_db(spec, ref=np.max)
        mel_specs.append(spec_db)

    return mel_specs


def compute_mss(target: np.ndarray, pred: np.ndarray) -> float:
    logger.info("Computing MSS...")
    target_specs = compute_mel_specs(target)
    pred_specs = compute_mel_specs(pred)

    dist = 0.0
    for target_spec, pred_spec in zip(target_specs, pred_specs):
        dist += np.mean(np.abs(target_spec - pred_spec))

    dist = dist / len(target_specs)
    return dist


scatter = None


def compute_jtfs(y: np.ndarray, J: int = 10, Q: int = 12):
    global scatter
    if scatter is None:
        scatter = Scattering1D(J=J, Q=Q, shape=y.shape[-1])

    return scatter(y)


def compute_jtfs_distance(
    target: np.ndarray, pred: np.ndarray, J: int = 10, Q: int = 12
) -> float:
    logger.info("Computing JTFS...")

    target_jtfs = compute_jtfs(target, J, Q)
    pred_jtfs = compute_jtfs(pred, J, Q)

    dist = np.mean(np.abs(target_jtfs - pred_jtfs))
    return dist


def compute_mfcc(target: np.ndarray, sample_rate: float = 44100.0) -> np.ndarray:
    window_length = int(0.05 * sample_rate)
    hop_length = int(0.01 * sample_rate)

    mfcc = librosa.feature.mfcc(
        y=target,
        sr=sample_rate,
        n_mfcc=20,
        n_fft=window_length,
        hop_length=hop_length,
        n_mels=128,
    )

    return mfcc


def compute_wmfcc(target: np.ndarray, pred: np.ndarray) -> float:
    logger.info("Computing wMFCC...")

    target_mfcc = compute_mfcc(target)
    pred_mfcc = compute_mfcc(pred)

    target_mfcc = target_mfcc.reshape(-1, target_mfcc.shape[-1])
    pred_mfcc = pred_mfcc.reshape(-1, pred_mfcc.shape[-1])

    def l1(a, b):
        return np.mean(np.abs(a - b))

    dist = dtw(target_mfcc.T, pred_mfcc.T, dist_method=l1, distance_only=True)
    return dist.normalizedDistance


pesto_model = None
def get_pesto_activations(audio: np.ndarray, sample_rate: float = 44100.0) -> np.ndarray:
    global pesto_model
    if pesto_model is None:
        pesto_model = pesto.load_model("mir-1k_g7", step_size=20.0)

    x = torch.from_numpy(audio)
    x = x.mean(0)
    _, _, _, activations = pesto_model(x, sample_rate)

    return activations.numpy()


def compute_f0(target: np.ndarray, pred: np.ndarray) -> float:
    logger.info("Computing f0...")
    target_activations = get_pesto_activations(target)
    pred_activations = get_pesto_activations(pred)

    target_norm = np.linalg.vector_norm(target_activations, axis=-1, ord=2)
    pred_norm = np.linalg.vector_norm(pred_activations, axis=-1, ord=2)
    cosine_sim = np.einsum("ij,ij->i", target_activations, pred_activations) / (
        target_norm * pred_norm
    )
    return cosine_sim.mean()



def compute_amp_env(target: np.ndarray, pred: np.ndarray) -> float:
    logger.info("Computing amp env...")
    return 0.0


def compute_metrics_on_dir(audio_dir: Path) -> dict[str, float]:
    target_file = AudioFile(str(audio_dir / "target.wav"))
    pred_file = AudioFile(str(audio_dir / "pred.wav"))

    target = target_file.read(target_file.frames)
    pred = pred_file.read(pred_file.frames)

    target_file.close()
    pred_file.close()

    mss = compute_mss(target, pred)
    # jtfs = compute_jtfs_distance(target, pred)
    jtfs = 0.0
    wmfcc = compute_wmfcc(target, pred)
    f0 = compute_f0(target, pred)
    amp_env = compute_amp_env(target, pred)

    return dict(mss=mss, jtfs=jtfs, wmfcc=wmfcc, f0=f0, amp_env=amp_env)


def compute_metrics(audio_dirs: List[Path], output_dir: Path):
    idxs = []
    rows = []
    for dir in audio_dirs:
        metrics = compute_metrics_on_dir(dir)
        rows.append(metrics)
        idxs.append(dir.name.rsplit("_", 1)[1])

    pid = multiprocessing.current_process().pid

    df = pd.DataFrame(rows, index=idxs)
    metric_file = output_dir / f"metrics-{pid}.csv"
    df.to_csv(metric_file)

    return metric_file


@click.command()
@click.argument("audio_dir", type=str)
@click.argument("output_dir", type=str, default="metrics")
@click.option("--num_workers", "-w", type=int, default=8)
def main(audio_dir: str, output_dir: str, num_workers: int):
    # 1. make a list of all subdirectories that match the expected structure
    # 2. divide list up into sublists per worker
    # 3. send each list to a worker and begin processing. each worker dumps metrics to
    # its own file.
    # 4. when a worker returns, take its csv file and append it to the master list
    # 5. when all workers are done, compute the mean of each metric across the master
    # list
    audio_dir = Path(audio_dir)
    audio_dirs = find_possible_subdirs(audio_dir)

    os.makedirs(output_dir, exist_ok=True)
    output_dir = Path(output_dir)

    sublist_length = len(audio_dirs) // num_workers
    sublists = [
        audio_dirs[i * sublist_length : (i + 1) * sublist_length]
        for i in range(num_workers)
    ]

    metric_dfs = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(compute_metrics, sublist, output_dir)
            for sublist in sublists
        ]

        for future in as_completed(futures):
            metric_file = future.result()
            metric_df = pd.read_csv(metric_file)
            # set index to first column
            metric_df.set_index(metric_df.columns[0], inplace=True)
            metric_dfs.append(metric_df)

    df = pd.concat(metric_dfs)
    df.to_csv(output_dir / "metrics.csv")

    columnwise_means = df.mean(axis=0)
    columnwise_stds = df.std(axis=0)
    print("Means...")
    print(columnwise_means)

    print("Stds...")
    print(columnwise_stds)

    # make new DF of aggregated metrics, with mean and std columns
    df = pd.DataFrame(
        {
            "mean": columnwise_means,
            "std": columnwise_stds,
        }
    )
    df.to_csv(output_dir / "aggregated_metrics.csv")


if __name__ == "__main__":
    main()
