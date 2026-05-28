"""
Compare Split CP, Jackknife, and LWO on a shared multivariate MA(1) simulation.

The response is vector-valued, so each method outputs a Euclidean ball and we
report width as the ball diameter.
"""

from dataclasses import dataclass, replace

import numpy as np
from utils import (
    lwo_conservative_quantile as conservative_quantile,
    lwo_fit_model as fit_model,
    lwo_mean_and_se as mean_and_se,
    lwo_predict_model as predict_model,
    lwo_precompute_loo_scores as precompute_loo_scores,
    lwo_residual_norm as residual_norm,
    lwo_run_lwo_jackknife as run_lwo_jackknife,
    lwo_run_split_cp as run_split_cp,
)


@dataclass
class ExperimentConfig:
    n: int = 80
    dimension: int = 20
    lag: int = 10
    alpha: float = 0.1
    tau: int = 10
    rounds: int = 200
    split_ratio: float = 0.5
    gap: int = 0
    predictor: str = "knn"
    ridge_lambda: float = 1.0
    knn_k: int = 5
    kernel_bandwidth: float = 1.0
    noise_std: float = 1.0
    seed: int = 12345


def generate_multivariate_ma1(length, dimension, noise_std, rng):
    """
    Generate X_t = omega_t + omega_{t-1}, omega_t ~ N(0, noise_std^2 I_d).
    """
    innovations = rng.normal(loc=0.0, scale=noise_std, size=(length + 1, dimension))
    return innovations[1:] + innovations[:-1]


def build_lagged_train_test(series, lag, n_train_time):
    """
    Use the first n_train_time observations as the training/calibration history and
    the (n_train_time + 1)-th observation as the test target.

    For each target X_t, the feature is vec(X_{t-lag}, ..., X_{t-1}).
    """
    series = np.asarray(series, dtype=float)
    if series.ndim != 2:
        raise ValueError("series must have shape (time, dimension)")
    if n_train_time <= lag:
        raise ValueError(f"Need n > lag, got n={n_train_time}, lag={lag}")
    if len(series) != n_train_time + 1:
        raise ValueError(
            f"Expected a series of length n+1={n_train_time + 1}, got {len(series)}"
        )

    X_train = []
    y_train = []
    for t in range(lag, n_train_time):
        X_train.append(series[t - lag : t].reshape(-1))
        y_train.append(series[t])

    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    x_test = series[n_train_time - lag : n_train_time].reshape(-1)
    y_test = np.asarray(series[n_train_time], dtype=float)
    return X_train, y_train, x_test, y_test


def run_experiment(config):
    rng = np.random.default_rng(config.seed)
    methods = ["Split CP", "Jackknife", f"LWO(tau={config.tau})"]
    per_round = {
        method: {"coverage": [], "width": []}
        for method in methods
    }

    for round_idx in range(config.rounds):
        series = generate_multivariate_ma1(
            length=config.n + 1,
            dimension=config.dimension,
            noise_std=config.noise_std,
            rng=rng,
        )
        X_train, y_train, x_test, y_test = build_lagged_train_test(
            series=series,
            lag=config.lag,
            n_train_time=config.n,
        )

        split_cov, split_wid = run_split_cp(X_train, y_train, x_test, y_test, config)
        per_round["Split CP"]["coverage"].append(split_cov)
        per_round["Split CP"]["width"].append(split_wid)

        residuals = precompute_loo_scores(X_train, y_train, config)

        full_model = fit_model(X_train, y_train, config)
        jk_radius = conservative_quantile(residuals, config.alpha)
        jk_center = predict_model(full_model, x_test)
        jk_cov = float(residual_norm(y_test, jk_center) <= jk_radius)
        jk_wid = float(2.0 * jk_radius)
        per_round["Jackknife"]["coverage"].append(jk_cov)
        per_round["Jackknife"]["width"].append(jk_wid)

        lwo_cov, lwo_wid = run_lwo_jackknife(X_train, y_train, x_test, y_test, config)
        per_round[f"LWO(tau={config.tau})"]["coverage"].append(lwo_cov)
        per_round[f"LWO(tau={config.tau})"]["width"].append(lwo_wid)

        progress_step = max(1, config.rounds // 10)
        if (round_idx + 1) % progress_step == 0 or round_idx == config.rounds - 1:
            print(f"Completed {round_idx + 1}/{config.rounds} rounds")

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


def display_method_name(method):
    if method.startswith("LWO"):
        return "LWO jackknife"
    return method


def main():
    # Edit the experiment settings here.
    # To mimic the older multidimensional MA(1) experiment more closely, try:
    # n=50, lag=1, tau=2, predictor="knn".
    dimensions = [50]
    base_config = ExperimentConfig(
        n=200,
        dimension=dimensions,
        lag=1,
        alpha=0.1,
        tau=5,
        rounds=500,
        split_ratio=0.5,
        gap=0,
        predictor="knn",  # "ridge", "knn", "kernel", "mlp", or "dt"
        ridge_lambda=1.0,
        knn_k=10,
        kernel_bandwidth=0.5,
        noise_std=1.0,
        seed=12345,
    )

    print("=" * 72)
    print("Shared-data multidimensional MA(1) dimension sweep")
    print("=" * 72)
    print(
        f"dimensions={dimensions}, L={base_config.lag}, n={base_config.n}, "
        f"rounds={base_config.rounds}, alpha={base_config.alpha}, "
        f"tau={base_config.tau}, predictor={base_config.predictor}"
    )

    for dimension in dimensions:
        config = replace(base_config, dimension=dimension)
        print("\n" + "-" * 72)
        print(f"Running dimension d={dimension}")
        print("-" * 72)

        summary = run_experiment(config)

        print("\nSummary")
        print(f"{'Method':<26} {'Coverage Mean':<16} {'Coverage SE':<14} {'Width Mean':<14} {'Width SE':<12}")
        for method, stats in summary.items():
            print(
                f"{display_method_name(method):<26} "
                f"{stats['coverage_mean']:<16.4f} "
                f"{stats['coverage_se']:<14.4f} "
                f"{stats['width_mean']:<14.4f} "
                f"{stats['width_se']:<12.4f}"
            )


if __name__ == "__main__":
    main()
