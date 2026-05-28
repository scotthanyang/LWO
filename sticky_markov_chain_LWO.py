"""
Compare Split CP, Jackknife, Jackknife+, and LWO on a sticky Markov-chain
counterexample.

The raw covariate is transformed to the repeat-count feature used in the
counterexample experiments. The script prints aggregate coverage and width
summaries only; it does not save files or create plots.
"""

import math
import warnings
from dataclasses import dataclass

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore", category=ConvergenceWarning)


@dataclass
class ExperimentConfig:
    ntrial: int = 500
    n: int = 500
    geom_rho_values: tuple = (0.1,)
    alpha: float = 0.1
    effect_size: float = 1.0
    burn_in: int = 200
    predictor: str = "kernel"
    knn_k: int = 10
    max_depth: int = 5
    kernel_bandwidth: float = 0.001
    tau: int = 50
    gap: int = 0
    n_blocks: int = 10
    seed: int = 12345


def generate_data(n_obs, geom_rho, config, rng):
    """
    Generate the sticky Markov-chain counterexample data.

    Returns arrays of length n_obs after discarding burn-in observations.
    """
    n_total = n_obs + config.burn_in
    switch_before = -rng.geometric(geom_rho)
    switch_after = rng.geometric(geom_rho) + n_total + 1
    switch_during = np.argwhere(rng.uniform(size=n_total) <= geom_rho).T[0]
    switch_diff = np.r_[switch_during, switch_after] - np.r_[switch_before, switch_during]
    switch_mat = np.subtract.outer(np.arange(n_total), np.r_[switch_during, switch_after]) < 0
    switch_inds = np.min(
        switch_mat * np.outer(np.ones(n_total), np.arange(len(switch_diff)))
        + (1 - switch_mat) * (len(switch_diff) + 1),
        axis=1,
    ).astype(int)

    X = rng.normal(size=len(switch_diff))[switch_inds]
    Y = rng.normal(size=n_total) + config.effect_size * switch_diff[switch_inds]
    return X[config.burn_in :], Y[config.burn_in :]


def count_Xreps(X):
    """
    Build Xtil from raw training covariates by counting exact repeats.
    """
    X = np.asarray(X, dtype=float)
    return np.sum(np.subtract.outer(X, X) == 0, axis=1)


def count_reps_for_queries(X_train_raw, X_query_raw):
    """
    Count exact matches for each query inside a raw training subset.
    """
    X_train_raw = np.asarray(X_train_raw, dtype=float)
    X_query_raw = np.asarray(X_query_raw, dtype=float)
    if X_query_raw.ndim == 0:
        return float(np.sum(X_train_raw == X_query_raw))
    return np.sum(X_query_raw.reshape(-1, 1) == X_train_raw.reshape(1, -1), axis=1)


def conservative_quantile(values, alpha, mode="upper"):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.nan

    if mode == "upper":
        k = min(max(math.ceil((1.0 - alpha) * (values.size + 1)), 1), values.size)
    elif mode == "lower":
        k = min(max(math.floor(alpha * (values.size + 1)), 1), values.size)
    else:
        raise ValueError("mode must be 'upper' or 'lower'")

    return float(np.partition(values, k - 1)[k - 1])


def predict_with_model(X_train, y_train, x_test, config):
    X_train_2d = np.asarray(X_train, dtype=float).reshape(-1, 1)
    y_train = np.asarray(y_train, dtype=float)
    x = np.asarray(x_test, dtype=float).reshape(1, -1)
    predictor = config.predictor.lower()

    if predictor == "kernel":
        diff = X_train_2d - x
        dist2 = np.einsum("ij,ij->i", diff, diff)
        scaled = dist2 / (max(float(config.kernel_bandwidth), 1e-8) ** 2)
        weights = np.exp(-0.5 * scaled)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-12:
            return float(y_train[np.argmin(dist2)])
        return float(np.sum(weights * y_train) / weight_sum)

    if predictor == "knn":
        model = KNeighborsRegressor(n_neighbors=max(1, min(config.knn_k, len(X_train_2d))))
    elif predictor == "ridge":
        model = Ridge(alpha=1.0)
    elif predictor == "mlp":
        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(20,),
                activation="relu",
                solver="adam",
                alpha=1e-3,
                max_iter=100,
                random_state=config.seed,
            ),
        )
    elif predictor == "dt":
        model = DecisionTreeRegressor(
            max_depth=config.max_depth,
            min_samples_leaf=2,
            random_state=config.seed,
        )
    else:
        raise ValueError(
            f"Unsupported predictor '{config.predictor}'. "
            "Use one of: 'ridge', 'knn', 'kernel', 'mlp', or 'dt'."
        )

    return float(model.fit(X_train_2d, y_train).predict(x)[0])


def run_split_cp(raw_X_train, y_train, raw_x_test, y_test, config):
    n_train = len(y_train)
    split_idx = int(0.5 * n_train)
    train_end = split_idx - config.gap
    cal_start = split_idx

    if train_end <= 0 or cal_start >= n_train:
        return np.nan, np.nan

    raw_X_A, y_A = raw_X_train[:train_end], y_train[:train_end]
    raw_X_B, y_B = raw_X_train[cal_start:], y_train[cal_start:]
    X_A = count_Xreps(raw_X_A)
    X_B = count_reps_for_queries(raw_X_A, raw_X_B)
    x_test = count_reps_for_queries(raw_X_A, raw_x_test)

    scores = [
        abs(y_B[i] - predict_with_model(X_A, y_A, X_B[i], config))
        for i in range(len(y_B))
    ]
    radius = conservative_quantile(scores, config.alpha, mode="upper")
    y_hat_test = predict_with_model(X_A, y_A, x_test, config)
    covered = (y_hat_test - radius) <= y_test <= (y_hat_test + radius)
    return float(covered), float(2.0 * radius)


def precompute_loo_predictions_and_scores(raw_X_train, y_train, raw_x_test, config):
    n_train = len(y_train)
    preds_loo_test = np.empty(n_train, dtype=float)
    scores_loo = np.empty(n_train, dtype=float)

    for i in range(n_train):
        mask = np.ones(n_train, dtype=bool)
        mask[i] = False
        raw_subset = raw_X_train[mask]
        X_subset = count_Xreps(raw_subset)
        x_leftout = count_reps_for_queries(raw_subset, raw_X_train[i])
        x_test = count_reps_for_queries(raw_subset, raw_x_test)

        y_hat_i = predict_with_model(X_subset, y_train[mask], x_leftout, config)
        scores_loo[i] = abs(y_train[i] - y_hat_i)
        preds_loo_test[i] = predict_with_model(X_subset, y_train[mask], x_test, config)

    return preds_loo_test, scores_loo


def run_jackknife(raw_X_train, y_train, raw_x_test, y_test, config, scores_loo=None):
    if scores_loo is None:
        _, scores_loo = precompute_loo_predictions_and_scores(raw_X_train, y_train, raw_x_test, config)

    X_train = count_Xreps(raw_X_train)
    x_test = count_reps_for_queries(raw_X_train, raw_x_test)
    radius = conservative_quantile(scores_loo, config.alpha, mode="upper")
    y_hat_test = predict_with_model(X_train, y_train, x_test, config)
    covered = (y_hat_test - radius) <= y_test <= (y_hat_test + radius)
    return float(covered), float(2.0 * radius)


def run_jackknife_plus(
    raw_X_train,
    y_train,
    raw_x_test,
    y_test,
    config,
    preds_loo_test=None,
    scores_loo=None,
):
    if preds_loo_test is None or scores_loo is None:
        preds_loo_test, scores_loo = precompute_loo_predictions_and_scores(
            raw_X_train,
            y_train,
            raw_x_test,
            config,
        )

    lower_vals = preds_loo_test - scores_loo
    upper_vals = preds_loo_test + scores_loo
    lower = conservative_quantile(lower_vals, config.alpha, mode="lower")
    upper = conservative_quantile(upper_vals, config.alpha, mode="upper")
    covered = lower <= y_test <= upper
    return float(covered), float(upper - lower)


def run_lwo_jackknife(raw_X_train, y_train, raw_x_test, y_test, config):
    if config.tau < 0:
        raise ValueError(f"tau must be nonnegative, got {config.tau}")

    n_train = len(y_train)
    scores = []
    for k in range(n_train):
        right = min(n_train, k + config.tau + 1)
        mask = np.ones(n_train, dtype=bool)
        mask[k:right] = False
        if not np.any(mask):
            continue

        raw_subset = raw_X_train[mask]
        X_subset = count_Xreps(raw_subset)
        x_k = count_reps_for_queries(raw_subset, raw_X_train[k])
        y_hat_k = predict_with_model(X_subset, y_train[mask], x_k, config)
        scores.append(abs(y_train[k] - y_hat_k))

    if not scores:
        return np.nan, np.nan

    X_train = count_Xreps(raw_X_train)
    x_test = count_reps_for_queries(raw_X_train, raw_x_test)
    radius = conservative_quantile(scores, config.alpha, mode="upper")
    y_hat_test = predict_with_model(X_train, y_train, x_test, config)
    covered = (y_hat_test - radius) <= y_test <= (y_hat_test + radius)
    return float(covered), float(2.0 * radius)


def block_mean_std(values, n_blocks):
    values = np.asarray(values, dtype=float)
    blocks = np.array_split(values, n_blocks)
    block_means = np.array([np.nanmean(block) for block in blocks], dtype=float)
    return float(np.nanmean(block_means)), float(np.nanstd(block_means, ddof=0))


def summarize_method(values, n_blocks):
    coverage_mean, coverage_std = block_mean_std(values["coverage"], n_blocks)
    width_mean, width_std = block_mean_std(values["width"], n_blocks)
    return {
        "coverage_mean": coverage_mean,
        "coverage_std": coverage_std,
        "width_mean": width_mean,
        "width_std": width_std,
    }


def run_experiment(config):
    if config.ntrial % config.n_blocks != 0:
        raise ValueError(
            f"ntrial ({config.ntrial}) must be divisible by n_blocks ({config.n_blocks})."
        )

    rng = np.random.default_rng(config.seed)
    methods = ["Split CP", "Jackknife", "Jackknife+", f"LWO(tau={config.tau})"]
    summaries = {}

    for geom_rho in config.geom_rho_values:
        per_round = {
            method: {"coverage": [], "width": []}
            for method in methods
        }

        for round_idx in range(config.ntrial):
            X, Y = generate_data(config.n + 1, geom_rho, config, rng)
            raw_X_train, y_train = X[: config.n], Y[: config.n]
            raw_x_test, y_test = X[config.n], Y[config.n]

            split_cov, split_wid = run_split_cp(raw_X_train, y_train, raw_x_test, y_test, config)
            preds_loo_test, scores_loo = precompute_loo_predictions_and_scores(
                raw_X_train,
                y_train,
                raw_x_test,
                config,
            )
            jk_cov, jk_wid = run_jackknife(
                raw_X_train,
                y_train,
                raw_x_test,
                y_test,
                config,
                scores_loo=scores_loo,
            )
            jkp_cov, jkp_wid = run_jackknife_plus(
                raw_X_train,
                y_train,
                raw_x_test,
                y_test,
                config,
                preds_loo_test=preds_loo_test,
                scores_loo=scores_loo,
            )
            lwo_cov, lwo_wid = run_lwo_jackknife(raw_X_train, y_train, raw_x_test, y_test, config)

            round_results = {
                "Split CP": (split_cov, split_wid),
                "Jackknife": (jk_cov, jk_wid),
                "Jackknife+": (jkp_cov, jkp_wid),
                f"LWO(tau={config.tau})": (lwo_cov, lwo_wid),
            }

            for method, (coverage, width) in round_results.items():
                per_round[method]["coverage"].append(coverage)
                per_round[method]["width"].append(width)

            progress_step = max(1, config.ntrial // 10)
            if (round_idx + 1) % progress_step == 0 or round_idx == config.ntrial - 1:
                print(f"Completed {round_idx + 1}/{config.ntrial} trials for rho={geom_rho:.2f}")

        summaries[geom_rho] = {
            method: summarize_method(values, config.n_blocks)
            for method, values in per_round.items()
        }

    return summaries


def display_method_name(method):
    if method.startswith("LWO"):
        return "LWO jackknife"
    return method


def main():
    config = ExperimentConfig(
        ntrial=500,
        n=1000,
        geom_rho_values=(0.1,),
        alpha=0.1,
        effect_size=1.0,
        burn_in=200,
        predictor="kernel",
        knn_k=10,
        max_depth=5,
        kernel_bandwidth=0.001,
        tau=50,
        gap=0,
        n_blocks=10,
        seed=12345,
    )

    print("=" * 72)
    print("Sticky Markov-chain comparison")
    print("=" * 72)
    print(
        f"rho_values={list(config.geom_rho_values)}, ntrial={config.ntrial}, "
        f"n={config.n}, alpha={config.alpha}, burn_in={config.burn_in}, "
        f"gap={config.gap}, tau={config.tau}, predictor={config.predictor}"
    )

    summaries = run_experiment(config)

    for geom_rho, summary in summaries.items():
        print("\n" + "-" * 72)
        print(f"Summary for rho={geom_rho:.2f}")
        print("-" * 72)
        print(
            f"{'Method':<18} {'Coverage Mean':<16} {'Coverage Std':<14} "
            f"{'Width Mean':<14} {'Width Std':<12}"
        )
        for method, stats in summary.items():
            print(
                f"{display_method_name(method):<18} "
                f"{stats['coverage_mean']:<16.4f} "
                f"{stats['coverage_std']:<14.4f} "
                f"{stats['width_mean']:<14.4f} "
                f"{stats['width_std']:<12.4f}"
            )


if __name__ == "__main__":
    main()
