"""Cluster bootstrap on per-cluster histograms, paired comparisons and Holm correction."""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .metrics import curve_from_hist, pauc


def cluster_weights(n_clusters: int, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n_clusters) multinomial resampling counts. Deterministic given seed."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters), size=n_boot).astype(np.float32)


def aligned(clusters_a: Sequence[str], ha: np.ndarray, clusters_b: Sequence[str], hb: np.ndarray,
            order: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Reorder two systems' cluster histograms to a common cluster order (paired design)."""
    ia = {c: i for i, c in enumerate(clusters_a)}
    ib = {c: i for i, c in enumerate(clusters_b)}
    missing = [c for c in order if c not in ia or c not in ib]
    if missing:
        raise ValueError(f"systems scored on different clusters, e.g. {missing[:3]}")
    return ha[[ia[c] for c in order]], hb[[ib[c] for c in order]]


def pauc_stat(max_over: float = 0.05) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    def f(hp: np.ndarray, hn: np.ndarray) -> np.ndarray:
        leak, over = curve_from_hist(hp, hn)
        return pauc(leak, over, max_over)
    return f


def paired_bootstrap(pii_a: np.ndarray, non_a: np.ndarray, pii_b: np.ndarray, non_b: np.ndarray,
                     stat: Callable, n_boot: int = 10_000, seed: int = 0, batch: int = 500) -> dict:
    """Paired cluster bootstrap of stat(A) - stat(B). Inputs are (C, NBINS) histograms with rows
    aligned to the same clusters. Returns point estimates, 95% CI and a two-sided p-value."""
    C = pii_a.shape[0]
    point_a = float(stat(pii_a.sum(0), non_a.sum(0)))
    point_b = float(stat(pii_b.sum(0), non_b.sum(0)))
    diffs = []
    W = cluster_weights(C, n_boot, seed)
    for i in range(0, n_boot, batch):
        w = W[i:i + batch]
        da = stat(w @ pii_a, w @ non_a)
        db = stat(w @ pii_b, w @ non_b)
        diffs.append(np.asarray(da) - np.asarray(db))
    d = np.concatenate(diffs)
    p = 2 * min((np.sum(d <= 0) + 1) / (n_boot + 1), (np.sum(d >= 0) + 1) / (n_boot + 1))
    return {"a": point_a, "b": point_b, "diff": point_a - point_b,
            "ci_low": float(np.quantile(d, 0.025)), "ci_high": float(np.quantile(d, 0.975)),
            "p_value": float(min(1.0, p)), "n_clusters": int(C), "n_boot": int(n_boot)}


def single_bootstrap_ci(pii: np.ndarray, non: np.ndarray, stat: Callable, n_boot: int = 2000,
                        seed: int = 0, batch: int = 500) -> tuple[float, float]:
    W = cluster_weights(pii.shape[0], n_boot, seed)
    vals = np.concatenate([np.asarray(stat(W[i:i + batch] @ pii, W[i:i + batch] @ non))
                           for i in range(0, n_boot, batch)])
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm step-down. Returns adjusted p-values and reject decisions for the whole family."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    stop = False
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        reject = (not stop) and p <= alpha / (m - i)
        if not reject:
            stop = True
        out[k] = {"p": p, "p_holm": adj, "reject": reject}
    return out
