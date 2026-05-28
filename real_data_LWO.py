"""
Compare Split CP, Jackknife, Jackknife+, and LWO on shared real-data chunks.

For each chosen column, we split the raw series into chunks of length ``n+1``
with a configurable raw-data gap between consecutive chunks. Each chunk is
treated as one dataset: the first ``n`` raw observations define the training
and calibration history, and the ``(n+1)``-th raw observation is the one-step-
ahead test target.

The script prints an aggregate summary. It intentionally does not save files
or create plots.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from utils import (
    lwo_mean_and_se as mean_and_se,
    lwo_precompute_loo_predictions_and_scores as precompute_loo_predictions_and_scores,
    lwo_run_jackknife as run_jackknife,
    lwo_run_jackknife_plus as run_jackknife_plus,
    lwo_run_lwo_jackknife as run_lwo_jackknife,
    lwo_run_split_cp as run_split_cp,
)

DATA_DIR = Path(__file__).resolve().parent / "real_data"


@dataclass
class RealDataConfig:
    dataset_name: str = "traffic.txt"
    column_indices: tuple = (6,)
    lag: int = 24
    n_history: int = 100
    alpha: float = 0.1
    tau: int = 24
    chunk_gap: int = 100
    split_ratio: float = 0.5
    gap: int = 0
    predictor: str = "ridge"
    ridge_lambda: float = 1.0
    knn_k: int = 5
    kernel_bandwidth: float = 0.1
    evaluation_start_fraction: float = 0.0
    max_chunks_per_column: Optional[int] = None
    seed: int = 12345


def load_dataset(file_name, data_dir=DATA_DIR):
    file_path = Path(data_dir) / file_name
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path, header=None)
    if not df.empty and df.iloc[0].map(lambda x: isinstance(x, str)).all():
        df = pd.read_csv(file_path, header=0)
    return df


def get_series_from_dataset(df, column_index=0):
    series = df.iloc[:, column_index].to_numpy(dtype=float)
    return series[np.isfinite(series)]


def build_lagged_train_test_from_chunk(chunk, lag, n_history):
    chunk = np.asarray(chunk, dtype=float)
    if chunk.ndim != 1:
        raise ValueError("chunk must be a 1D array")
    if len(chunk) != n_history + 1:
        raise ValueError(
            f"Expected a chunk of length n+1={n_history + 1}, got {len(chunk)}"
        )
    if n_history <= lag:
        raise ValueError(f"Need n > lag, got n={n_history}, lag={lag}")

    X_train = []
    y_train = []
    for t in range(lag, n_history):
        X_train.append(chunk[t - lag : t])
        y_train.append(chunk[t])

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_test = np.asarray(chunk[n_history - lag : n_history], dtype=float)
    y_test = float(chunk[n_history])
    return X_train, y_train, x_test, y_test


def choose_chunk_starts(series_length, config):
    start_index = max(0, int(config.evaluation_start_fraction * series_length))
    chunk_length = config.n_history + 1
    step = chunk_length + max(0, config.chunk_gap)

    starts = list(range(start_index, series_length - chunk_length + 1, step))
    if config.max_chunks_per_column is not None:
        starts = starts[: config.max_chunks_per_column]
    return starts


def format_duration(seconds):
    seconds = int(max(0, round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def run_experiment(config):
    df = load_dataset(config.dataset_name)
    methods = ["Split CP", "Jackknife", "Jackknife+", f"LWO(tau={config.tau})"]
    per_round = {
        method: {"coverage": [], "width": []}
        for method in methods
    }
    column_jobs = []

    for column_index in config.column_indices:
        series = get_series_from_dataset(df, column_index=column_index)
        starts = choose_chunk_starts(len(series), config)
        if not starts:
            raise ValueError(
                f"Column {column_index} is too short for chunking with "
                f"n_history={config.n_history}, chunk_gap={config.chunk_gap}, "
                f"evaluation_start_fraction={config.evaluation_start_fraction}."
            )
        column_jobs.append((column_index, series, starts))

    total_chunks = sum(len(starts) for _, _, starts in column_jobs)
    completed_chunks = 0
    experiment_start_time = time.perf_counter()
    print(f"Total chunks to process: {total_chunks}", flush=True)

    for column_index, series, starts in column_jobs:
        print(f"Column {column_index}: {len(starts)} chunks", flush=True)
        for round_idx, start in enumerate(starts, start=1):
            chunk_start_time = time.perf_counter()
            print(
                f"Starting chunk {completed_chunks + 1}/{total_chunks} "
                f"(column={column_index}, round={round_idx}/{len(starts)}, start={start})",
                flush=True,
            )
            chunk = series[start : start + config.n_history + 1]
            X_train, y_train, x_test, y_test = build_lagged_train_test_from_chunk(
                chunk=chunk,
                lag=config.lag,
                n_history=config.n_history,
            )

            split_cov, split_wid = run_split_cp(X_train, y_train, x_test, y_test, config)
            preds_loo_test, scores_loo = precompute_loo_predictions_and_scores(
                X_train, y_train, x_test, config
            )
            jk_cov, jk_wid = run_jackknife(
                X_train, y_train, x_test, y_test, config, scores_loo=scores_loo
            )
            jkp_cov, jkp_wid = run_jackknife_plus(
                X_train,
                y_train,
                x_test,
                y_test,
                config,
                preds_loo_test=preds_loo_test,
                scores_loo=scores_loo,
            )
            lwo_cov, lwo_wid = run_lwo_jackknife(X_train, y_train, x_test, y_test, config)

            round_results = {
                "Split CP": (split_cov, split_wid),
                "Jackknife": (jk_cov, jk_wid),
                "Jackknife+": (jkp_cov, jkp_wid),
                f"LWO(tau={config.tau})": (lwo_cov, lwo_wid),
            }

            for method, (coverage, width) in round_results.items():
                per_round[method]["coverage"].append(coverage)
                per_round[method]["width"].append(width)

            completed_chunks += 1
            elapsed = time.perf_counter() - experiment_start_time
            last_chunk_time = time.perf_counter() - chunk_start_time
            avg_chunk_time = elapsed / completed_chunks
            eta = avg_chunk_time * (total_chunks - completed_chunks)
            print(
                f"Finished chunk {completed_chunks}/{total_chunks} "
                f"({100.0 * completed_chunks / total_chunks:.1f}%) | "
                f"last={format_duration(last_chunk_time)}, "
                f"avg/chunk={format_duration(avg_chunk_time)}, "
                f"elapsed={format_duration(elapsed)}, "
                f"ETA={format_duration(eta)}",
                flush=True,
            )

        print(
            f"Completed column {column_index} "
            f"({len(starts)} chunks from {config.dataset_name})"
        )

    summary = {}
    for method, stats in per_round.items():
        cov_mean, cov_se = mean_and_se(stats["coverage"])
        wid_mean, wid_se = mean_and_se(stats["width"])
        summary[method] = {
            "coverage_mean": cov_mean,
            "coverage_se": cov_se,
            "width_mean": wid_mean,
            "width_se": wid_se,
        }

    return summary


def main():
    # Edit the experiment settings here.
    config = RealDataConfig(
        dataset_name="traffic.txt",
        column_indices=(6,),
        lag=24,
        n_history=100,
        alpha=0.1,
        tau=20,
        chunk_gap=48,
        split_ratio=0.5,
        gap=0,
        predictor="knn",  # "ridge", "knn", "kernel", "mlp", or "dt"
        ridge_lambda=1.0,
        knn_k=5,
        kernel_bandwidth=0.1,
        evaluation_start_fraction=0.0,
        max_chunks_per_column=None,
        seed=12345,
    )

    print("=" * 72)
    print("Chunked real-data comparison")
    print("=" * 72)
    print(
        f"dataset={config.dataset_name}, cols={list(config.column_indices)}, "
        f"lag={config.lag}, n={config.n_history}, alpha={config.alpha}, "
        f"tau={config.tau}, chunk_gap={config.chunk_gap}, "
        f"predictor={config.predictor}, "
        f"eval_start={config.evaluation_start_fraction:.2f}"
    )

    summary = run_experiment(config)

    print("\nSummary")
    print("-" * 72)
    print(f"{'Method':<18} {'Coverage Mean':<16} {'Coverage SE':<14} {'Width Mean':<14} {'Width SE':<12}")
    print("-" * 72)
    for method, stats in summary.items():
        print(
            f"{method:<18} "
            f"{stats['coverage_mean']:<16.4f} "
            f"{stats['coverage_se']:<14.4f} "
            f"{stats['width_mean']:<14.4f} "
            f"{stats['width_se']:<12.4f}"
        )


if __name__ == "__main__":
    main()
