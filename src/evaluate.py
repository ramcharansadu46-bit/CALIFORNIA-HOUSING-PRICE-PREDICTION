"""Metrics and statistical helpers (all implemented from their textbook formulas)."""
import numpy as np
from scipy import stats
from scipy import sparse
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors


def regression_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    rel = np.abs(y_true - y_pred) / y_true
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "MedAPE_%": float(np.median(rel) * 100),
        "Within20%": float(np.mean(rel <= 0.20) * 100),
    }


def nadeau_bengio_ttest(diff, n_train, n_test):
    """Corrected resampled t-test (Nadeau & Bengio, 2003) for paired CV scores.

    diff: per-fold score differences (model A - model B).  Variance is inflated by
    (1/k + n_test/n_train) because CV training sets overlap -> plain paired t-test is too liberal.
    """
    diff = np.asarray(diff, float)
    k = len(diff)
    var = diff.var(ddof=1)
    t = diff.mean() / np.sqrt((1.0 / k + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return float(t), float(p)


def paired_bootstrap_rmse_diff(y, pred_a, pred_b, n_boot=2000, seed=42):
    """Bootstrap CI of RMSE(A) - RMSE(B) on the same test rows (negative => A better)."""
    rng = np.random.default_rng(seed)
    y, pa, pb = map(np.asarray, (y, pred_a, pred_b))
    ea, eb = (y - pa) ** 2, (y - pb) ** 2
    n = len(y)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = np.sqrt(ea[idx].mean()) - np.sqrt(eb[idx].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.sqrt(ea.mean()) - np.sqrt(eb.mean())), float(lo), float(hi)


def morans_i(values, lat, lon, k=8, n_perm=499, seed=42):
    """Global Moran's I with row-standardised k-nearest-neighbour weights (haversine metric)
    and a one-sided permutation p-value.  Returns (I, expected_I, z_perm, p, z, lag)."""
    values = np.asarray(values, float)
    coords = np.radians(np.column_stack([lat, lon]))
    nn = NearestNeighbors(n_neighbors=k + 1, metric="haversine").fit(coords)
    _, idx = nn.kneighbors(coords)
    idx = idx[:, 1:]                       # drop self
    n = len(values)
    rows = np.repeat(np.arange(n), k)
    W = sparse.csr_matrix((np.full(n * k, 1.0 / k), (rows, idx.ravel())), shape=(n, n))
    z = values - values.mean()
    denom = (z ** 2).sum()
    I = float(n / n * (z @ (W @ z)) / denom)      # S0 = n for row-standardised W
    rng = np.random.default_rng(seed)
    perm = np.empty(n_perm)
    for b in range(n_perm):
        zp = rng.permutation(z)
        perm[b] = zp @ (W @ zp) / denom
    p = float((1 + np.sum(perm >= I)) / (n_perm + 1))
    z_score = float((I - perm.mean()) / perm.std())
    return I, float(-1.0 / (n - 1)), z_score, p, z, W @ z
