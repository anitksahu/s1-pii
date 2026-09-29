"""Span-level calibration over a candidate set, with identical recalibration for every system.

Candidate set: every predicted span with score >= floor (0.05). Its target is 1 if it
matches a gold PII span (character IoU >= 0.5; typed or type-agnostic), else 0. Candidates
overlapping only IGNORE gold are dropped. Gold spans with no candidate are counted
separately as ``uncovered`` because calibration of scores cannot describe them; they show up
in the leak metric instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..schema import Doc, Span

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
    uncovered_gold: int
    total_gold: int


def candidates(docs: Sequence[Doc], preds_by_doc: dict, typed: bool = False, floor: float = FLOOR,
               iou: float = 0.5) -> Candidates:
    sc, tg, ty = [], [], []
    unc = tot = 0
    for d in docs:
        gold = d.pii_spans()
        ign = d.ignore_spans()
        preds = [p for p in preds_by_doc.get(d.doc_id, []) if p.score >= floor]
        covered = set()
        for p in preds:
            hit = False
            for gi, g in enumerate(gold):
                if typed and g.label_canonical != p.label_canonical:
                    continue
                if _iou(p.start, p.end, g.start, g.end) >= iou:
                    hit = True; covered.add(gi)
            if not hit and any(min(g.end, p.end) > max(g.start, p.start) for g in ign) and \
                    not any(min(g.end, p.end) > max(g.start, p.start) for g in gold):
                continue
            sc.append(p.score); tg.append(int(hit)); ty.append(p.label_canonical)
        tot += len(gold)
        unc += len(gold) - len(covered)
    return Candidates(np.array(sc, float), np.array(tg, int), ty, unc, tot)


def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


def adaptive_ece(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Equal-mass binning ECE."""
    if len(p) == 0:
        return float("nan")
    order = np.argsort(p, kind="stable")
    bins = np.array_split(order, min(n_bins, len(p)))
    return float(sum(len(b) / len(p) * abs(p[b].mean() - y[b].mean()) for b in bins if len(b)))


def ece_ci(p: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    vals = [adaptive_ece(p[i], y[i]) for i in (rng.integers(0, len(p), len(p)) for _ in range(n_boot))]
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


class TypeIsotonic:
    """Per-type isotonic recalibration, fitted on the calibration split only. Types with
    fewer than ``min_n`` candidates fall back to a pooled fit."""

    def __init__(self, min_n: int = 50):
        self.min_n = min_n
        self.models: dict[str, object] = {}
        self.pooled = None

    def fit(self, c: Candidates) -> "TypeIsotonic":
        from sklearn.isotonic import IsotonicRegression
        self.pooled = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(c.scores, c.targets)
        types = np.array(c.types)
        for t in set(c.types):
            m = types == t
            if m.sum() >= self.min_n:
                self.models[t] = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(
                    c.scores[m], c.targets[m])
        return self

    def transform_span(self, s: Span) -> float:
        model = self.models.get(s.label_canonical, self.pooled)
        return float(np.clip(model.predict([s.score])[0], 0, 1))


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Binary temperature on logit(score) by minimizing NLL (golden-section on log T)."""
    def nll(logT):
        z = logits / np.exp(logT)
        p = 1 / (1 + np.exp(-z))
        p = np.clip(p, 1e-7, 1 - 1e-7)
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
    """Selective risk (fraction wrong among kept) vs coverage when keeping candidates in
    descending score order and predicting 'PII' for the kept ones."""
    order = np.argsort(-p, kind="stable")
    yk = y[order]
    k = np.arange(1, len(p) + 1)
    return k / len(p), np.cumsum(1 - yk) / k
