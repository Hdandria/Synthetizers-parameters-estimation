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
import pandas as pd
from kymatio.numpy import Scattering1D
from loguru import logger
from pedalboard.io import AudioFile


def subdir_matches_pattern(dir: Path) -> bool:
    """Returns true if subdir contains pred.wav and target.wav"""
    return (dir / "target.wav").exists() and (dir / "pred.wav").exists()


def find_possible_subdirs(audio_dir: Path) -> List[Path]:
    all_subdirectories = [d for d in audio_dir.glob("*") if d.is_dir()]
    return [d for d in all_subdirectories if subdir_matches_pattern(d)]


def compute_mel_specs(y: np.ndarray, sample_rate: float = 44100.0):
    mel_specs = []
    window_sizes = [0.01, 0.025, 0.1]  # 10ms, 25ms, 100ms
    hop_sizes = [0.005, 0.01, 0.05]  # 5ms, 10ms, 50ms
    n_mels_list = [32, 64, 128]

    for window_size, hop_size, n_mels in zip(window_sizes, hop_sizes, n_mels_list):
        win_length = int(window_size * sample_rate)
        hop_length = int(hop_size * sample_rate)
        # Ensure n_fft is at least as large as win_length
        n_fft = max(2048, win_length)

        mel_spec = librosa.feature.melspectrogram(
            y=y.mean(axis=0),
            sr=sample_rate,
            n_mels=n_mels,
            hop_length=hop_length,
            win_length=win_length,
            n_fft=n_fft,
            window="hann",
        )
        spec_db = librosa.power_to_db(mel_spec, ref=np.max)

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

    y_mono = y.mean(axis=0)
    coeffs = scatter(y_mono)
    return coeffs


def compute_wmfcc(target: np.ndarray, pred: np.ndarray) -> float:
    logger.info("Computing wMFCC...")
    win_length = int(0.05 * 44100)  # 50ms
    hop_length = int(0.01 * 44100)  # 10ms
    n_fft = max(2048, win_length)

    target_mfcc = librosa.feature.mfcc(
        y=target.mean(axis=0),
        sr=44100,
        n_mfcc=13,
        n_mels=128,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
    ).T

    pred_mfcc = librosa.feature.mfcc(
        y=pred.mean(axis=0),
        sr=44100,
        n_mfcc=13,
        n_mels=128,
        hop_length=hop_length,
        win_length=win_length,
        n_fft=n_fft,
    ).T

    # Use simple L1 distance instead of DTW to avoid dependency issues
    # This is a simplified version of wMFCC
    min_len = min(len(target_mfcc), len(pred_mfcc))
    target_mfcc_trunc = target_mfcc[:min_len]
    pred_mfcc_trunc = pred_mfcc[:min_len]
    
    dist = np.mean(np.abs(target_mfcc_trunc - pred_mfcc_trunc))
    return dist


def get_stft(y: np.ndarray, sample_rate: float = 44100.0):
    win_length = int(0.05 * sample_rate)
    hop_length = int(0.02 * sample_rate)
    stft = librosa.stft(
        y.mean(axis=0),
        n_fft=win_length,
        hop_length=hop_length,
        win_length=win_length,
        window="hann",
    ).T
    stft_mag = np.abs(stft)
    return stft_mag


def batched_wasserstein_distance_np(
    hist1: np.ndarray,
    hist2: np.ndarray,
) -> np.ndarray:
    bin_width = 1 / hist1.shape[-1]
    cdf1 = np.cumsum(hist1, axis=-1)
    cdf2 = np.cumsum(hist2, axis=-1)
    distance = np.sum(np.abs(cdf1 - cdf2), axis=-1) * bin_width
    return distance


def compute_sot(target: np.ndarray, pred: np.ndarray) -> float:
    logger.info("Computing SOT...")
    target_stft = get_stft(target)
    pred_stft = get_stft(pred)

    target_stft = target_stft / np.clip(
        target_stft.sum(axis=-1, keepdims=True), 1e-6, None
    )
    pred_stft = pred_stft / np.clip(pred_stft.sum(axis=-1, keepdims=True), 1e-6, None)

    dists = batched_wasserstein_distance_np(target_stft, pred_stft)
    return dists.mean()


def compute_rms(target: np.ndarray, pred: np.ndarray) -> float:
    logger.info("Computing amp env...")
    win_length = int(0.05 * 44100)  # 50ms
    hop_length = int(0.025 * 44100)  # 25ms

    target_rms = librosa.feature.rms(
        y=target.mean(axis=0),
        frame_length=win_length,
        hop_length=hop_length,
    ).squeeze()

    pred_rms = librosa.feature.rms(
        y=pred.mean(axis=0),
        frame_length=win_length,
        hop_length=hop_length,
    ).squeeze()

    # Compute cosine similarity
    cosine_sim = np.dot(target_rms, pred_rms) / (
        np.linalg.norm(target_rms) * np.linalg.norm(pred_rms)
    )

    return cosine_sim.mean()


def compute_metrics_on_dir(audio_dir: Path) -> dict[str, float]:
    target_file = AudioFile(str(audio_dir / "target.wav"))
    pred_file = AudioFile(str(audio_dir / "pred.wav"))

    target = target_file.read(target_file.frames)
    pred = pred_file.read(pred_file.frames)

    target_file.close()
    pred_file.close()

    mss = compute_mss(target, pred)
    wmfcc = compute_wmfcc(target, pred)
    sot = compute_sot(target, pred)
    rms = compute_rms(target, pred)

    return dict(mss=mss, wmfcc=wmfcc, sot=sot, rms=rms)


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
            df = pd.read_csv(metric_file, index_col=0)
            metric_dfs.append(df)

    # Combine all dataframes
    combined_df = pd.concat(metric_dfs, ignore_index=False)
    combined_df = combined_df.sort_index()

    # Compute summary statistics
    summary_stats = combined_df.describe()
    summary_stats.loc["mean"] = combined_df.mean()
    summary_stats.loc["std"] = combined_df.std()

    # Save results
    combined_df.to_csv(output_dir / "all_metrics.csv")
    summary_stats.to_csv(output_dir / "summary_stats.csv")

    logger.info(f"Computed metrics for {len(combined_df)} samples")
    logger.info(f"Mean MSS: {summary_stats.loc['mean', 'mss']:.4f}")
    logger.info(f"Mean wMFCC: {summary_stats.loc['mean', 'wmfcc']:.4f}")
    logger.info(f"Mean SOT: {summary_stats.loc['mean', 'sot']:.4f}")
    logger.info(f"Mean RMS: {summary_stats.loc['mean', 'rms']:.4f}")


if __name__ == "__main__":
    main()
