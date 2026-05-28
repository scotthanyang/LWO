import warnings
from math import erf, sqrt
import math
import os, json, csv
import matplotlib.pyplot as plt
import time
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from scipy.spatial.distance import cdist
from collections import defaultdict

try:
    from data_generation import *
except ImportError:
    pass

try:
    from predictors import *
except ImportError:
    pass

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class NadarayaWatsonRegressor(BaseEstimator, RegressorMixin):
    """
    Nadaraya–Watson kernel regression.

    Parameters
    ----------
    k : int or None, default=5
        If set, use an *adaptive* bandwidth h(x) = distance to the k-th nearest
        neighbor of x. With kernel='uniform', this reproduces uniform k-NN.
        If None, use a global bandwidth 'bw'.
    bw : float or 'auto', default='auto'
        Global bandwidth used only when k is None. If 'auto', uses median
        pairwise distance on training X.
    kernel : {'uniform','epanechnikov','triangular','rbf','laplace','tricube','cosine'}, default='uniform'
        Kernel shape. For k-NN behavior, use 'uniform' (exact) or 'epanechnikov'/'triangular' (smooth k-NN).
    eps : float, default=1e-12
        Small constant to avoid division by zero.
    """
    def __init__(self, k=5, bw='auto', kernel='uniform', eps=1e-12):
        self.k = k
        self.bw = bw
        self.kernel = kernel
        self.eps = eps

    # --------- fit ----------
    def fit(self, X, y):

        self.X_ = X
        self.y_ = y
        self.n_features_in_ = X.shape[1]

        # Only needed when using global bandwidth (k is None)
        if self.k is None:
            if self.bw == 'auto':
                n = min(2000, X.shape[0])
                if X.shape[0] > n:
                    rng = np.random.default_rng(0)
                    Z = X[rng.choice(X.shape[0], size=n, replace=False)]
                else:
                    Z = X
                D = cdist(Z, Z, metric='euclidean')
                med = np.median(D[D > 0]) if np.any(D > 0) else 1.0
                self._bw_ = max(float(med), self.eps)
            else:
                self._bw_ = float(self.bw)
                if self._bw_ <= 0:
                    raise ValueError("bw must be > 0")
        return self

    def _pairwise_dist(self, Xq, Xr):
        # cross-part: last l of query vs first l of reference
        Q = Xq[:, ]
        R = Xr[:, ]
        return cdist(Q, R, metric='euclidean')

    def _auto_bandwidth(self, X):
        """Median distance (excluding zeros) under the chosen distance definition."""
        n = X.shape[0]
        if n <= 1:
            return 1.0
        if n > self.auto_max_rows:
            rng = np.random.default_rng(self.rng_seed)
            idx = rng.choice(n, size=self.auto_max_rows, replace=False)
            Z = X[idx]
        else:
            Z = X
        D = self._pairwise_dist(Z, Z)
        vals = D[D > 0]
        if vals.size == 0:
            return 1.0
        return float(np.median(vals))
        
    # --------- kernels ----------
    @staticmethod
    def _kernel_weights(u, kind):
        # u = distance / bandwidth (broadcasted), nonnegative
        k = kind.lower()
        if k == 'uniform' or k == 'tophat':
            return (u <= 1.0).astype(float)
        elif k == 'epanechnikov':
            return np.maximum(0.0, 1.0 - u**2)
        elif k == 'triangular':
            return np.maximum(0.0, 1.0 - u)
        elif k == 'tricube':
            v = np.minimum(1.0, u)
            return (1.0 - v**3)**3
        elif k == 'cosine':
            out = np.zeros_like(u)
            m = (u < 1.0)
            out[m] = np.cos(np.pi * u[m] / 2.0)
            return out
        elif k == 'rbf':
            return np.exp(-0.5 * u**2)
        elif k == 'laplace':
            return np.exp(-u)
        else:
            raise ValueError(f"Unknown kernel: {kind}")

    # --------- predict ----------
    def predict(self, X, mode='pred'):
        X = np.asarray(X, float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        #D = cdist(X, self.X_, metric='euclidean')  # (n_test, n_train)
        D = self._pairwise_dist(X, self.X_) # (n_test, n_train)
        if self.k is not None:
            # Adaptive bandwidth: h_i = distance to k-th nearest neighbor of x_i
            k_eff = max(1, min(int(self.k), self.X_.shape[0]-1))
            # kth order statistic per row
            hk = np.partition(D, kth=k_eff, axis=1)[:, k_eff]  # shape (n_test,)
            # guard against 0 bandwidth (duplicate points)
            hk = np.maximum(hk, self.eps)
            U = D / hk[:, None]  # normalize by per-row bandwidth
            W = self._kernel_weights(U, self.kernel)
        else:
            # Global bandwidth
            U = D / float(self._bw_)
            W = self._kernel_weights(U, self.kernel)

            if mode == 'cal':
                #print(np.linalg.norm(np.min(U,axis=1)))
                print(W)
        
        # Check if y_ is multidimensional
        y_arr = np.asarray(self.y_, dtype=float)
        is_multidimensional = y_arr.ndim > 1 and y_arr.shape[1] > 1 if y_arr.ndim > 1 else False
        
        row_sums = W.sum(axis=1)
        # Fallback to 1-NN when all weights are (numerically) zero
        zero_rows = (row_sums <= self.eps)
        
        # Initialize output with correct shape
        if is_multidimensional:
            out = np.empty((X.shape[0], y_arr.shape[1]), dtype=float)
        else:
            out = np.empty(X.shape[0], dtype=float)

        if np.any(zero_rows):
            nn_idx = np.argmin(D[zero_rows], axis=1)
            if is_multidimensional:
                out[zero_rows] = y_arr[nn_idx]
            else:
                out[zero_rows] = y_arr[nn_idx] if y_arr.ndim == 1 else y_arr[nn_idx, 0]

        if np.any(~zero_rows):
            if is_multidimensional:
                # For multidimensional: W @ y_ gives (n_test, d)
                out[~zero_rows] = (W[~zero_rows] @ y_arr) / (row_sums[~zero_rows, None] + self.eps)
            else:
                # For scalar: ensure y_arr is 1D
                y_1d = y_arr.ravel() if y_arr.ndim > 1 else y_arr
                out[~zero_rows] = (W[~zero_rows] @ y_1d) / (row_sums[~zero_rows] + self.eps)

        return out


def conservative_quantile(values, alpha, mode='upper'):
    """
    Lower = floor((α)(n+1))-th smallest
    Upper = ceil((1-α)(n+1))-th smallest
    """
    a = np.asarray(values, dtype=float); n = a.size
    k_lo = min(max(math.floor((alpha) * (n+1)), 1), n)
    k_hi = min(max(math.ceil((1 - alpha) * (n+1)), 1), n)
    # use partition to avoid full sort
    if mode == 'upper':
        hi = np.partition(a, k_hi-1)[k_hi-1]
        return float(hi)
    else:
        lo = np.partition(a, k_lo-1)[k_lo-1]
        return float(lo)

## Predictors
def fit_model(model_name, X, y, md=5, lag=None, series=None, bw='auto', kernel='rbf', seed=0, neib=5):
    # Convert y to numpy array and check if multidimensional
    y_arr = np.asarray(y, dtype=float)
    is_multidimensional = y_arr.ndim > 1 and y_arr.shape[1] > 1 if y_arr.ndim > 1 else False
    
    # For models that expect 1D y when d=1, reshape appropriately
    if model_name == "RF":
        # RandomForestRegressor supports multi-output, so use y as-is
        return RandomForestRegressor(n_estimators=10, max_depth=md, n_jobs=-1, random_state=42).fit(X, y_arr)
    if model_name == "DT":
        # DecisionTreeRegressor supports multi-output, so use y as-is
        return DecisionTreeRegressor(max_depth=md, min_samples_leaf=2,random_state=42).fit(X, y_arr)
    if model_name == "MLP":
        # MLPRegressor supports multi-output, but for d=1, reshape to 1D to avoid warnings
        if not is_multidimensional and y_arr.ndim > 1:
            y_mlp = y_arr.ravel()
        else:
            y_mlp = y_arr
        return make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(40,),
                         activation="relu",
                         solver="lbfgs",      # or "adam" to mimic torch+Adam
                         alpha=1e-3,          # L2; bump if overfitting
                         max_iter=100,
                         random_state=seed)
        ).fit(X, y_mlp)
    if model_name == "Ridge":
        # Ridge supports multi-output, so use y as-is
        return Ridge(alpha=1.0).fit(X, y_arr)
    if model_name == "Lasso":
        # Lasso supports multi-output, so use y as-is
        return Lasso(alpha=0.1).fit(X, y_arr)
    if model_name == "KNN":
        # KNeighborsRegressor supports multi-output, so use y as-is
        return KNeighborsRegressor(n_neighbors=neib, weights='distance').fit(X, y_arr)
    if model_name in ("KR", "KernelReg"):
        # Standardize then kernel regress (distances computed in standardized space)
        # NadarayaWatsonRegressor now handles multidimensional y
        return NadarayaWatsonRegressor(k=None, kernel='rbf',bw=0.001).fit(X, y_arr)
        # AR (1.85, -0.88) 'epanechnikov' bw=11
    if model_name == "SVR":  # RBF SVR
        # SVR only supports single output, so use MultiOutputRegressor for multidimensional
        from sklearn.multioutput import MultiOutputRegressor
        if is_multidimensional:
            # Use MultiOutputRegressor wrapper for multidimensional outputs
            return MultiOutputRegressor(SVR(kernel='rbf', C=1, gamma='scale')).fit(X, y_arr)
        else:
            # For scalar or d=1, reshape to 1D if needed
            y_svr = y_arr.ravel() if y_arr.ndim > 1 else y_arr
            return SVR(kernel='rbf', C=15.0, gamma='scale').fit(X, y_svr)
    if model_name == "ODT":
        return ObliqueDecisionTreeRegressor(
            max_depth=md,                # you already pass md
            min_samples_split=50,
            min_samples_leaf=20,
            n_directions=24,             # try 32–128
            normalize=True,
            random_state=seed
        ).fit(X, y)
    raise ValueError(f"Unsupported model: {model_name}")

def predict_model(model_name, fitted, x_row, lag=None, mode='pred'):
    x = np.asarray(x_row, dtype=float)

    # All other models: sklearn estimators or pipelines (incl. sklearn MLPRegressor)
    X = x.reshape(1, -1)
    if model_name == 'KR':
        pred = fitted.predict(X, mode=mode)[0]
    else:
        pred = fitted.predict(X)[0]
    
    # Return as array to handle both scalar and multidimensional outputs
    pred = np.asarray(pred, dtype=float)
    # If scalar, return as float for backward compatibility; otherwise return array
    if pred.ndim == 0:
        return float(pred)
    return pred


def fit_full_training_model_from_features(X_train, y_train, series_train, lag, model_name, md=5, seed=42, neib=10):
    """Fit one model on the entire training segment (used for test predictions)."""
    return fit_model(model_name, X_train, y_train, md=md, seed=seed, neib=neib)



def plot_coverage_vs_phi(summary, ar_coefs_list, models, tau_list, figsize=(8,5)):
    """
    For each model, plot coverage vs AR coefficient with one line per tau.
    Uses mean ± std from `summary`.
    """

    # ensure numeric x-axis
    phis = [float(phi if np.isscalar(phi) else phi[0]) for phi in ar_coefs_list]

    for model in models:
        plt.figure(figsize=figsize)
        for tau in tau_list:
            means = []
            wids  = []
            for phi in ar_coefs_list:
                rec = summary[phi][model][tau]
                means.append(rec["coverage_mean"])
            means = np.array(means, float)
            label = tau if tau in ('split', 'jk+','jk') else f"τ={tau}"
            # plot with error bars (ignore all-NaN rows)
            if np.all(np.isnan(means)):
                continue
            plt.errorbar(phis, means, marker='o', capsize=3, linestyle='-',
                         label=label)

        plt.axhline(1-0.1, color='gray', linestyle='--', linewidth=1, label='target (0.9)')  # adjust if alpha!=0.1
        plt.xlabel("AR coefficient φ")
        plt.ylabel("Empirical coverage")
        plt.title(f"Coverage vs φ — Model: {model}")
        plt.legend()
        plt.tight_layout()
        plt.show()

def plot_width_vs_phi(summary, ar_coefs_list, models, tau_list, figsize=(8,5)):
    """
    For each model, plot coverage vs AR coefficient with one line per tau.
    Uses mean ± std from `summary`.
    """

    # ensure numeric x-axis
    phis = [float(phi if np.isscalar(phi) else phi[0]) for phi in ar_coefs_list]

    for model in models:
        plt.figure(figsize=figsize)
        for tau in tau_list:
            means = []
            for phi in ar_coefs_list:
                rec = summary[phi][model][tau]
                means.append(rec["width_mean"])
            means = np.array(means, float)
            label = tau if tau in ('split', 'jk+','jk') else f"τ={tau}"
            # plot with error bars (ignore all-NaN rows)
            if np.all(np.isnan(means)):
                continue
            plt.errorbar(phis, means, marker='o', capsize=3, linestyle='-',
                         label=label)

        plt.axhline(1-0.1, color='gray', linestyle='--', linewidth=1, label='target (0.9)')  # adjust if alpha!=0.1
        plt.xlabel("AR coefficient φ")
        plt.ylabel("Width")
        plt.title(f"Width vs φ — Model: {model}")
        plt.legend()
        plt.tight_layout()
        plt.show()


def summary_to_table(summary):
    """
    Flatten your nested summary {phi->{model->{tau->{coverage_mean,...}}}}
    to a list of row dicts for easy CSV/JSON saving.
    """
    rows = []
    for phi, by_model in summary.items():
        # phi might be scalar or a tuple/list; store a readable repr
        phi_val = float(phi) if np.isscalar(phi) else tuple(phi)
        for model, by_tau in by_model.items():
            for tau, stats in by_tau.items():
                rows.append({
                    "phi": phi_val,
                    "model": model,
                    "tau": tau,  # 'split' or int
                    "coverage_mean": float(stats.get("coverage_mean", np.nan)),
                    "coverage_std":  float(stats.get("coverage_std",  np.nan)),
                    "width_mean":    float(stats.get("width_mean",    np.nan)),
                    "width_std":     float(stats.get("width_std",     np.nan)),
                })
    return rows

def plot_coverage_vs_n_from_summaries(summaries, ar_coefs_list, models, tau_list, alpha_target=0.1, figsize=(7,4)):
    """
    summaries: list of (n, summary) pairs where summary == the nested dict returned by run_comprehensive_experiments
               i.e., summary[ac][model][tau] has keys: 'coverage_mean','coverage_std','width_mean','width_std'
    We plot coverage mean ± std vs n, one figure per model, one line per tau.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # assume a single AR setting per sweep (use the first one)
    # if you actually sweep ar_coefs_list too, you can extend this to loop over ac
    ac0 = ar_coefs_list[0]

    # collect by model,tau
    by_model_tau = {m: {tau: defaultdict(list) for tau in tau_list} for m in models}
    for n, summ in summaries:
        rec_ac = summ[ac0]  # dict by model
        for m in models:
            for tau in tau_list:
                stats = rec_ac[m][tau]
                by_model_tau[m][tau]['n'].append(n)
                by_model_tau[m][tau]['cov_mean'].append(stats['coverage_mean'])
                by_model_tau[m][tau]['cov_std'].append(stats['coverage_std'])

    # plot one figure per model
    for m in models:
        plt.figure(figsize=figsize)
        for tau in tau_list:
            dat = by_model_tau[m][tau]
            if not dat['n']:
                continue
            n_arr = np.array(dat['n'], float)
            mu    = np.array(dat['cov_mean'], float)
            sd    = np.array(dat['cov_std'], float)

            order = np.argsort(n_arr)
            n_arr, mu, sd = n_arr[order], mu[order], sd[order]

            label = tau if (isinstance(tau, str) and tau in ('split','jk+','jackknife+')) else f"τ={tau}"
            plt.errorbar(n_arr, mu, marker='o', capsize=3, linestyle='-', label=label)

        plt.axhline(1.0 - alpha_target, linestyle='--', color='gray', linewidth=1, label=f"target {1.0-alpha_target:.2f}")
        plt.xlabel("Training length n")
        plt.ylabel("Empirical coverage")
        plt.title(f"Coverage vs n — Model: {m}")
        plt.legend()
        plt.tight_layout()
        plt.show()

def plot_coverage_vs_dimension_from_summaries(summaries, models, tau_list, alpha_target=0.1, figsize=(7,4)):
    """
    summaries: list of (dimension, summary) pairs where summary == the nested dict returned by run_comprehensive_experiments
               i.e., summary[ac][model][tau] has keys: 'coverage_mean','coverage_std','width_mean','width_std'
    We plot coverage mean ± std vs dimension, one figure per model, one line per tau.
    """
    # collect by model,tau
    by_model_tau = {m: {tau: defaultdict(list) for tau in tau_list} for m in models}
    for dim, summ in summaries:
        # For MA(1) multidimensional, the dimension is passed as ar_coefs_list = [dim]
        # So summary[dim] contains the results
        rec_ac = summ[dim]  # dict by model
        for m in models:
            for tau in tau_list:
                stats = rec_ac[m][tau]
                by_model_tau[m][tau]['dim'].append(dim)
                by_model_tau[m][tau]['cov_mean'].append(stats['coverage_mean'])
                by_model_tau[m][tau]['cov_std'].append(stats['coverage_std'])

    # plot one figure per model
    for m in models:
        plt.figure(figsize=figsize)
        for tau in tau_list:
            dat = by_model_tau[m][tau]
            if not dat['dim']:
                continue
            dim_arr = np.array(dat['dim'], float)
            mu      = np.array(dat['cov_mean'], float)
            sd      = np.array(dat['cov_std'], float)

            order = np.argsort(dim_arr)
            dim_arr, mu, sd = dim_arr[order], mu[order], sd[order]

            label = tau if (isinstance(tau, str) and tau in ('split','jk+','jackknife+','jk','jackknife')) else f"τ={tau}"
            plt.plot(dim_arr, mu, marker='o', linestyle='-', label=label)

        plt.axhline(1.0 - alpha_target, linestyle='--', color='gray', linewidth=1, label=f"target {1.0-alpha_target:.2f}")
        plt.xlabel("Dimension (d)")
        plt.ylabel("Empirical coverage")
        plt.title(f"Coverage vs Dimension — Model: {m}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

def plot_width_vs_dimension_from_summaries(summaries, models, tau_list, figsize=(7,4)):
    """
    summaries: list of (dimension, summary) pairs where summary == the nested dict returned by run_comprehensive_experiments
               i.e., summary[ac][model][tau] has keys: 'coverage_mean','coverage_std','width_mean','width_std'
    We plot width mean ± std vs dimension, one figure per model, one line per tau.
    """
    # collect by model,tau
    by_model_tau = {m: {tau: defaultdict(list) for tau in tau_list} for m in models}
    for dim, summ in summaries:
        # For MA(1) multidimensional, the dimension is passed as ar_coefs_list = [dim]
        # So summary[dim] contains the results
        rec_ac = summ[dim]  # dict by model
        for m in models:
            for tau in tau_list:
                stats = rec_ac[m][tau]
                by_model_tau[m][tau]['dim'].append(dim)
                by_model_tau[m][tau]['width_mean'].append(stats['width_mean'])
                by_model_tau[m][tau]['width_std'].append(stats['width_std'])

    # plot one figure per model
    for m in models:
        plt.figure(figsize=figsize)
        for tau in tau_list:
            dat = by_model_tau[m][tau]
            if not dat['dim']:
                continue
            dim_arr = np.array(dat['dim'], float)
            mu      = np.array(dat['width_mean'], float)
            sd      = np.array(dat['width_std'], float)

            order = np.argsort(dim_arr)
            dim_arr, mu, sd = dim_arr[order], mu[order], sd[order]

            label = tau if (isinstance(tau, str) and tau in ('split','jk+','jackknife+','jk','jackknife')) else f"τ={tau}"
            plt.plot(dim_arr, mu, marker='o', linestyle='-', label=label)

        plt.xlabel("Dimension (d)")
        plt.ylabel("Width")
        plt.title(f"Width vs Dimension — Model: {m}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def lwo_conservative_quantile(values, alpha, mode="upper"):
    """
    Conservative finite-sample quantile used by the cleaned LWO scripts.
    """
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


def lwo_residual_norm(y_true, y_pred):
    """
    Absolute residual for scalar responses and Euclidean norm for vector responses.
    """
    residual = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    if residual.ndim == 0:
        return abs(float(residual))
    return float(np.linalg.norm(residual.reshape(-1), ord=2))


def lwo_fit_model(X, y, config):
    """
    Fit the lightweight predictors used by ma1_LWO.py and real_data_LWO.py.

    The MLP uses a scaled-target variant for scalar responses and the older
    direct multi-output fit for vector responses.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    if y.ndim not in (1, 2):
        raise ValueError("y must be 1D for scalar responses or 2D for vector responses")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of rows")
    if len(X) == 0:
        raise ValueError("Cannot fit on an empty dataset")

    if config.predictor == "ridge":
        x_mean = X.mean(axis=0)
        y_mean = y.mean(axis=0)
        X_centered = X - x_mean
        y_centered = y - y_mean
        gram = X_centered.T @ X_centered
        gram.flat[:: gram.shape[0] + 1] += config.ridge_lambda
        coef = np.linalg.solve(gram, X_centered.T @ y_centered)
        return {
            "type": "ridge",
            "x_mean": x_mean,
            "y_mean": y_mean,
            "coef": coef,
        }

    if config.predictor == "knn":
        return {
            "type": "knn",
            "X": X,
            "y": y,
            "k": max(1, min(config.knn_k, len(X))),
        }

    if config.predictor == "kernel":
        return {
            "type": "kernel",
            "X": X,
            "y": y,
            "bandwidth": max(float(config.kernel_bandwidth), 1e-8),
        }

    if config.predictor == "mlp":
        is_scalar_response = y.ndim == 1
        if is_scalar_response:
            base_model = make_pipeline(
                StandardScaler(),
                MLPRegressor(
                    hidden_layer_sizes=(20,),
                    activation="relu",
                    solver="adam",
                    alpha=1e-4,
                    learning_rate_init=1e-3,
                    tol=1e-5,
                    max_iter=500,
                    early_stopping=False,
                    random_state=config.seed,
                ),
            )
            model = TransformedTargetRegressor(
                regressor=base_model,
                transformer=StandardScaler(),
            ).fit(X, y)
            return {"type": "sklearn", "model": model}

        model = make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(8,),
                activation="relu",
                solver="adam",
                alpha=1e-3,
                learning_rate_init=5e-3,
                tol=1e-3,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=5,
                max_iter=30,
                random_state=config.seed,
            ),
        ).fit(X, y)
        return {"type": "sklearn", "model": model}

    if config.predictor == "dt":
        model = DecisionTreeRegressor(
            max_depth=5,
            min_samples_leaf=2,
            random_state=config.seed,
        ).fit(X, y)
        return {"type": "sklearn", "model": model}

    raise ValueError(
        f"Unsupported predictor '{config.predictor}'. "
        "Use 'ridge', 'knn', 'kernel', 'mlp', or 'dt'."
    )


def lwo_predict_model(model, x_row):
    x = np.asarray(x_row, dtype=float).reshape(1, -1)

    if model["type"] == "ridge":
        pred = (x - model["x_mean"]) @ model["coef"] + model["y_mean"]
    elif model["type"] == "knn":
        diff = model["X"] - x
        dist2 = np.einsum("ij,ij->i", diff, diff)
        k_eff = model["k"]
        nn_idx = np.argpartition(dist2, kth=k_eff - 1)[:k_eff]
        pred = model["y"][nn_idx].mean(axis=0)
    elif model["type"] == "kernel":
        diff = model["X"] - x
        dist2 = np.einsum("ij,ij->i", diff, diff)
        scaled = dist2 / (model["bandwidth"] ** 2)
        weights = np.exp(-0.5 * scaled)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 1e-12:
            pred = model["y"][np.argmin(dist2)]
        elif np.asarray(model["y"]).ndim == 1:
            pred = np.dot(weights, model["y"]) / weight_sum
        else:
            pred = (weights[:, None] * model["y"]).sum(axis=0) / weight_sum
    elif model["type"] == "sklearn":
        pred = model["model"].predict(x)[0]
    else:
        raise ValueError(f"Unsupported fitted model type '{model['type']}'")

    pred = np.asarray(pred, dtype=float)
    if pred.size == 1:
        return float(pred.reshape(-1)[0])
    return pred.reshape(-1)


def lwo_run_split_cp(X_train, y_train, x_test, y_test, config):
    n_rows = len(y_train)
    split_idx = int(config.split_ratio * n_rows)
    train_end = split_idx - config.gap
    cal_start = split_idx

    if train_end <= 0 or cal_start >= n_rows:
        return np.nan, np.nan

    model = lwo_fit_model(X_train[:train_end], y_train[:train_end], config)
    scores = np.array(
        [
            lwo_residual_norm(y_train[i], lwo_predict_model(model, X_train[i]))
            for i in range(cal_start, n_rows)
        ],
        dtype=float,
    )
    radius = lwo_conservative_quantile(scores, config.alpha, mode="upper")
    center = lwo_predict_model(model, x_test)
    covered = lwo_residual_norm(y_test, center) <= radius
    return float(covered), float(2.0 * radius)


def lwo_precompute_loo_predictions_and_scores(X_train, y_train, x_test, config):
    n_rows = len(y_train)
    preds_loo_test = []
    scores_loo = np.empty(n_rows, dtype=float)

    for i in range(n_rows):
        mask = np.ones(n_rows, dtype=bool)
        mask[i] = False
        model = lwo_fit_model(X_train[mask], y_train[mask], config)
        preds_loo_test.append(lwo_predict_model(model, x_test))
        scores_loo[i] = lwo_residual_norm(y_train[i], lwo_predict_model(model, X_train[i]))

    return np.asarray(preds_loo_test, dtype=float), scores_loo


def lwo_precompute_loo_scores(X_train, y_train, config):
    n_rows = len(y_train)
    residuals = np.empty(n_rows, dtype=float)

    for i in range(n_rows):
        mask = np.ones(n_rows, dtype=bool)
        mask[i] = False
        model = lwo_fit_model(X_train[mask], y_train[mask], config)
        residuals[i] = lwo_residual_norm(y_train[i], lwo_predict_model(model, X_train[i]))

    return residuals


def lwo_run_jackknife(X_train, y_train, x_test, y_test, config, scores_loo=None):
    if scores_loo is None:
        _, scores_loo = lwo_precompute_loo_predictions_and_scores(X_train, y_train, x_test, config)

    full_model = lwo_fit_model(X_train, y_train, config)
    center = lwo_predict_model(full_model, x_test)
    radius = lwo_conservative_quantile(scores_loo, config.alpha, mode="upper")
    covered = lwo_residual_norm(y_test, center) <= radius
    return float(covered), float(2.0 * radius)


def lwo_run_jackknife_plus(
    X_train,
    y_train,
    x_test,
    y_test,
    config,
    preds_loo_test=None,
    scores_loo=None,
):
    if preds_loo_test is None or scores_loo is None:
        preds_loo_test, scores_loo = lwo_precompute_loo_predictions_and_scores(
            X_train, y_train, x_test, config
        )

    preds_loo_test = np.asarray(preds_loo_test, dtype=float).reshape(-1)
    scores_loo = np.asarray(scores_loo, dtype=float)
    lower_vals = preds_loo_test - scores_loo
    upper_vals = preds_loo_test + scores_loo
    lower = lwo_conservative_quantile(lower_vals, config.alpha, mode="lower")
    upper = lwo_conservative_quantile(upper_vals, config.alpha, mode="upper")
    covered = lower <= float(np.asarray(y_test, dtype=float)) <= upper
    return float(covered), float(upper - lower)


def lwo_run_lwo_jackknife(X_train, y_train, x_test, y_test, config):
    n_rows = len(y_train)
    if config.tau < 0:
        raise ValueError(f"tau must be nonnegative, got {config.tau}")

    scores = []
    for k in range(n_rows):
        right = min(n_rows, k + config.tau + config.lag + 1)
        mask = np.ones(n_rows, dtype=bool)
        mask[k:right] = False
        if not np.any(mask):
            continue
        model = lwo_fit_model(X_train[mask], y_train[mask], config)
        scores.append(lwo_residual_norm(y_train[k], lwo_predict_model(model, X_train[k])))

    radius = lwo_conservative_quantile(np.asarray(scores, dtype=float), config.alpha, mode="upper")
    full_model = lwo_fit_model(X_train, y_train, config)
    center = lwo_predict_model(full_model, x_test)
    covered = lwo_residual_norm(y_test, center) <= radius
    return float(covered), float(2.0 * radius)


def lwo_mean_and_se(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.nan, np.nan
    mean = float(np.mean(finite))
    if finite.size == 1:
        return mean, 0.0
    se = float(np.std(finite, ddof=1) / math.sqrt(finite.size))
    return mean, se