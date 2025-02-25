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
import numpy as np
import pandas as pd
from pedalboard.io import AudioFile


def subdir_matches_pattern(dir: Path) -> bool:
    """Returns true if subdir contains pred.wav and target.wav"""
    return (dir / "target.wav").exists() and (dir / "pred.wav").exists()


def find_possible_subdirs(audio_dir: Path) -> List[Path]:
    all_subdirectories = [d for d in audio_dir.glob("*") if d.is_dir()]
    matching_dirs = [d for d in all_subdirectories if subdir_matches_pattern(d)]
    return matching_dirs


def compute_mss(target: np.ndarray, pred: np.ndarray) -> float:
    return 0.0


def compute_jtfs(target: np.ndarray, pred: np.ndarray) -> float:
    return 0.0


def compute_wmfcc(target: np.ndarray, pred: np.ndarray) -> float:
    return 0.0


def compute_f0(target: np.ndarray, pred: np.ndarray) -> float:
    return 0.0


def compute_amp_env(target: np.ndarray, pred: np.ndarray) -> float:
    return 0.0


def compute_metrics_on_dir(audio_dir: Path) -> dict[str, float]:
    target = AudioFile(audio_dir / "target.wav")
    pred = AudioFile(audio_dir / "pred.wav")

    target = target.read(target.frames)
    pred = pred.read(pred.frames)

    mss = compute_mss(target, pred)
    jtfs = compute_jtfs(target, pred)
    wmfcc = compute_wmfcc(target, pred)
    f0 = compute_f0(target, pred)
    amp_env = compute_amp_env(target, pred)

    target.close()
    pred.close()

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
        futures = [executor.submit(compute_metrics, sublist, output_dir) for sublist in sublists]

        for future in as_completed(futures):
            metric_file = future.result()
            metric_dfs.append(pd.read_csv(metric_file))

    df = pd.concat(metric_dfs)
    df.to_csv(output_dir / "metrics.csv")


if __name__ == "__main__":
    main()
