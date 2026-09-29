"""Span-level calibration over a candidate set, with identical recalibration for every system.

Candidate set: every admissible predicted span with score >= floor (0.05). Matching to
merged gold PII mentions is one-to-one: pairs with character IoU >= 0.5 (typed or
type-agnostic) are matched greedily by descending score, then descending IoU, so the
highest-scoring candidate on a gold mention is the positive and duplicates are negatives.
Candidates overlapping only IGNORE gold are dropped. Gold mentions with no matched
candidate are reported as ``uncovered``; they appear in the leak metric instead.
Recalibration never touches the headline pAUC, which uses raw scores (prereg).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .metrics import DocView

FLOOR = 0.05


def _iou(a0, a1, b0, b1) -> float:
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union else 0.0


@dataclass
class Candidates:
    scores: np.ndarray
    targets: np.ndarray
    types: list[str]
    clusters: list[str] = field(default_factory=list)
    uncovered_gold: int = 0
    total_gold: int = 0


def candidates(views: Sequence[DocView], typed: bool = False, floor: float = FLOOR,
               iou: float = 0.5) -> Candidates:
    sc, tg, ty, cl = [], [], [], []
    unc = tot = 0
    for v in views:
        gold = v.gold
        ign = [(s.start, s.end) for s in v.doc.ignore_spans()]
        preds = [p for p in v.preds if p.score >= floor]
        pairs = []
        for pi, p in enumerate(preds):
            for gi, (a, b, lab) in enumerate(gold):
                if typed and lab != p.label_canonical:
                    continue
                j = _iou(p.start, p.end, a, b)
                if j >= iou:
                    pairs.append((-p.score, -j, pi, gi))
        pairs.sort()
        up, ug = set(), set()
        for _, _, pi, gi in pairs:
            if pi not in up and gi not in ug:
                up.add(pi); ug.add(gi)
        for pi, p in enumerate(preds):
            if pi not in up:
                touches_gold = any(min(b, p.end) > max(a, p.start) for a, b, _ in gold)
                if not touches_gold and any(min(b, p.end) > max(a, p.start) for a, b in ign):
                    continue
            sc.append(p.score); tg.append(int(pi in up)); ty.append(p.label_canonical); cl.append(v.cluster_id)
        tot += len(gold)
        unc += len(gold) - len(ug)
    return Candidates(np.array(sc, float), np.array(tg, int), ty, cl, unc, tot)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


def adaptive_ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    if len(p) == 0:
        return float("nan")
    order = np.argsort(p, kind="stable")
    bins = np.array_split(order, min(n_bins, len(p)))
    return float(sum(len(b) / len(p) * abs(p[b].mean() - y[b].mean()) for b in bins if len(b)))


def cluster_ci(c: Candidates, fn, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Bootstrap CI of fn(p, y) resampling clusters, not candidates."""
    ids = sorted(set(c.clusters))
    rows = {k: [] for k in ids}
    for i, k in enumerate(c.clusters):
        rows[k].append(i)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(ids), len(ids))
        idx = np.concatenate([rows[ids[j]] for j in pick]) if len(ids) else np.array([], int)
        idx = idx.astype(int)
        vals.append(fn(c.scores[idx], c.targets[idx]))
    return float(np.nanquantile(vals, 0.025)), float(np.nanquantile(vals, 0.975))


class TypeIsotonic:
    """Per-type isotonic recalibration, fitted on the calibration split only. Types with
    fewer than ``min_n`` candidates fall back to a pooled fit."""

    def __init__(self, min_n: int = 50):
        self.min_n = min_n
        self.models: dict[str, object] = {}
        self.pooled = None

    def fit(self, c: Candidates) -> "TypeIsotonic":
        from sklearn.isotonic import IsotonicRegression
        mk = lambda: IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        self.pooled = mk().fit(c.scores, c.targets)
        types = np.array(c.types)
        for t in set(c.types):
            m = types == t
            if m.sum() >= self.min_n:
                self.models[t] = mk().fit(c.scores[m], c.targets[m])
        return self

    def transform(self, scores: np.ndarray, types: Sequence[str]) -> np.ndarray:
        out = np.empty(len(scores))
        types = np.array(types)
        for t in set(types.tolist()):
            m = types == t
            out[m] = self.models.get(t, self.pooled).predict(scores[m])
        return np.clip(out, 0, 1)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Binary temperature on logit(score) minimizing NLL (golden-section search on log T)."""
    def nll(logT):
        p = np.clip(1 / (1 + np.exp(-logits / np.exp(logT))), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    a, b = -3.0, 3.0
    g = (5 ** 0.5 - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    for _ in range(80):
        if nll(c) < nll(d):
            b = d
        else:
            a = c
        c, d = b - g * (b - a), a + g * (b - a)
    return float(np.exp((a + b) / 2))


def risk_coverage(p: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-p, kind="stable")
    yk = y[order]
    k = np.arange(1, len(p) + 1)
    return k / len(p), np.cumsum(1 - yk) / k
