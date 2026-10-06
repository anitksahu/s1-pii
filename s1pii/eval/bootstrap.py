"""Cluster bootstrap on per-cluster histograms, paired and multi-seed comparisons, Holm.

Resampling weights are generated per batch from a ``SeedSequence`` so memory is bounded
by the batch size, and the same seed always yields the same weights. In a paired
comparison every system (and every training seed of a system) is evaluated under the
same weight matrix, so the bootstrap distribution of the difference is paired at the
cluster level. Training seeds are fixed and averaged; they are not resampled, so these
intervals propagate document-cluster variation only, not training-seed variation.
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .metrics import curve_from_hist, pauc

Hist = tuple[np.ndarray, np.ndarray]   # (pii (C, NBINS), non (C, NBINS)), rows aligned to clusters


def weight_batches(n_clusters: int, n_boot: int, seed: int, batch: int = 500):
    ss = np.random.SeedSequence(seed)
    for i, child in enumerate(ss.spawn((n_boot + batch - 1) // batch)):
        size = min(batch, n_boot - i * batch)
        rng = np.random.default_rng(child)
        yield rng.multinomial(n_clusters, np.full(n_clusters, 1.0 / n_clusters), size=size).astype(np.float64)


def cluster_weights(n_clusters: int, n_boot: int, seed: int) -> np.ndarray:
    return np.concatenate(list(weight_batches(n_clusters, n_boot, seed)))


def align(clusters: Sequence[str], h: Hist, order: Sequence[str]) -> Hist:
    ix = {c: i for i, c in enumerate(clusters)}
    missing = [c for c in order if c not in ix]
    if missing:
        raise ValueError(f"system scored on different clusters, e.g. {missing[:3]}")
    sel = [ix[c] for c in order]
    return h[0][sel], h[1][sel]


def pauc_stat(max_over: float = 0.05) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    def f(hp, hn):
        leak, over = curve_from_hist(hp, hn)
        return pauc(leak, over, max_over)
    return f


def _mean_stat(stat, hists: Sequence[Hist], w=None):
    if w is None:
        return float(np.mean([stat(h[0].sum(0), h[1].sum(0)) for h in hists]))
    return np.mean([np.asarray(stat(w @ h[0], w @ h[1])) for h in hists], axis=0)


def paired_bootstrap_multi(a_seeds: Sequence[Hist], b_seeds: Sequence[Hist], stat: Callable,
                           n_boot: int = 10_000, seed: int = 0, batch: int = 500) -> dict:
    """Bootstrap of mean_seeds stat(A) - mean_seeds stat(B), all under shared cluster weights.
    Two-sided p-value = 2 * min(P(d <= 0), P(d >= 0)) with the +1 correction."""
    C = a_seeds[0][0].shape[0]
    for h in (*a_seeds, *b_seeds):
        if h[0].shape[0] != C:
            raise ValueError("all systems must be aligned to the same clusters")
    pa, pb = _mean_stat(stat, a_seeds), _mean_stat(stat, b_seeds)
    d = np.concatenate([_mean_stat(stat, a_seeds, w) - _mean_stat(stat, b_seeds, w)
                        for w in weight_batches(C, n_boot, seed, batch)])
    p = 2 * min((np.sum(d <= 0) + 1) / (n_boot + 1), (np.sum(d >= 0) + 1) / (n_boot + 1))
    per_seed = [float(stat(h[0].sum(0), h[1].sum(0))) for h in a_seeds]
    return {"a": pa, "b": pb, "diff": pa - pb, "a_per_seed": per_seed,
            "ci_low": float(np.quantile(d, 0.025)), "ci_high": float(np.quantile(d, 0.975)),
            "p_value": float(min(1.0, p)), "n_clusters": int(C), "n_boot": int(n_boot)}


def paired_bootstrap(pii_a, non_a, pii_b, non_b, stat: Callable, n_boot: int = 10_000,
                     seed: int = 0, batch: int = 500) -> dict:
    return paired_bootstrap_multi([(pii_a, non_a)], [(pii_b, non_b)], stat, n_boot, seed, batch)


def single_bootstrap_ci(pii: np.ndarray, non: np.ndarray, stat: Callable, n_boot: int = 2000,
                        seed: int = 0, batch: int = 500) -> tuple[float, float]:
    vals = np.concatenate([np.asarray(stat(w @ pii, w @ non))
                           for w in weight_batches(pii.shape[0], n_boot, seed, batch)])
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm step-down: adjusted p-values and reject decisions for the whole family."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running, stop = {}, 0.0, False
    for i, (k, p) in enumerate(items):
        running = min(1.0, max(running, (m - i) * p))
        reject = (not stop) and p <= alpha / (m - i)
        stop = stop or not reject
        out[k] = {"p": p, "p_holm": running, "reject": reject}
    return out


def c0_decision(results: dict[str, dict], alpha: float = 0.05, needed: int = 4) -> dict:
    """Apply the preregistered C0 rule. ``results`` maps 'dataset|baseline' to the output of
    ``paired_bootstrap_multi`` with A = S1 and B = baseline (lower pAUC is better). A win on a
    comparison requires diff < 0 and Holm-adjusted p < alpha over the whole family. A dataset
    counts as won only if S1 wins against every baseline evaluated on it."""
    h = holm({k: v["p_value"] for k, v in results.items()}, alpha)
    per_ds: dict[str, list[bool]] = {}
    for k, v in results.items():
        ds = k.split("|")[0]
        per_ds.setdefault(ds, []).append(v["diff"] < 0 and h[k]["reject"])
    won = sorted(ds for ds, wins in per_ds.items() if all(wins))
    return {"family_size": len(results), "holm": h, "datasets_won": won,
            "n_datasets": len(per_ds), "needed": needed, "c0_holds": len(won) >= needed}
