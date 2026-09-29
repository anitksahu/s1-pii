"""Character-level redaction metrics.

Definitions (preregistered in prereg/v0.md)
------------------------------------------
* Only alphanumeric characters (``str.isalnum``) count; whitespace and punctuation never do.
* Gold character classes, by precedence when gold spans overlap: PII > IGNORE > NOT_PII.
  Characters under no gold span are non-PII. IGNORE characters are excluded from both the
  numerator and denominator of every metric.
* Predictions are expanded to whole whitespace-delimited words (``align.expand_to_words``);
  each character gets the maximum score of any prediction covering it (-1 if none).
* A character is masked at threshold t iff its score >= t.
* leak(t) = unmasked gold-PII chars / gold-PII chars;
  over(t) = masked non-PII chars / non-PII chars. Both are micro-averaged over documents.

Scores are quantized to a 1/1000 grid so that every metric is computed from per-cluster
histograms; this makes the cluster bootstrap exact and cheap. Thresholds are k/1000.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ..schema import Doc, Span, IGNORE, CANONICAL_TYPES
from .align import expand_to_words, word_bounds

NBINS = 1002          # index 0: no prediction; 1 + floor(1000 * score) for score in [0, 1]
GRID = np.arange(1001) / 1000.0

LAB_NON, LAB_IGN, LAB_PII = 0, 1, 2


def char_labels(doc: Doc) -> tuple[np.ndarray, np.ndarray]:
    """Per-character gold class (0 non, 1 ignore, 2 pii) and PII type index (-1 if none)."""
    n = len(doc.text)
    lab = np.zeros(n, dtype=np.int8)
    typ = np.full(n, -1, dtype=np.int8)
    for s in doc.spans:
        if s.label_canonical == IGNORE:
            seg = lab[s.start:s.end]
            seg[seg < LAB_IGN] = LAB_IGN
    for s in doc.spans:
        if s.is_pii:
            lab[s.start:s.end] = LAB_PII
            t = typ[s.start:s.end]
            t[t < 0] = CANONICAL_TYPES.index(s.label_canonical)
    return lab, typ


def alnum_mask(text: str) -> np.ndarray:
    return np.fromiter((c.isalnum() for c in text), dtype=bool, count=len(text))


def char_scores(doc: Doc, preds: Sequence[Span], word_granularity: bool = True) -> np.ndarray:
    sc = np.full(len(doc.text), -1.0)
    bounds = word_bounds(doc.text) if word_granularity else None
    for p in preds:
        s, e = (expand_to_words(doc.text, p.start, p.end, bounds) if word_granularity else (p.start, p.end))
        np.maximum(sc[s:e], p.score, out=sc[s:e])
    return sc


def to_bins(scores: np.ndarray) -> np.ndarray:
    b = np.zeros(scores.shape, dtype=np.int64)
    hit = scores >= 0
    b[hit] = 1 + np.floor(np.clip(scores[hit], 0, 1) * 1000 + 1e-9).astype(np.int64)
    return b


@dataclass
class DocStats:
    doc_id: str
    cluster_id: str
    hist_pii: np.ndarray      # (NBINS,)
    hist_non: np.ndarray      # (NBINS,)
    n_chars: int


def doc_stats(doc: Doc, preds: Sequence[Span], word_granularity: bool = True) -> DocStats:
    lab, _ = char_labels(doc)
    an = alnum_mask(doc.text)
    bins = to_bins(char_scores(doc, preds, word_granularity))
    return DocStats(
        doc.doc_id, doc.cluster_id or doc.doc_id,
        np.bincount(bins[an & (lab == LAB_PII)], minlength=NBINS),
        np.bincount(bins[an & (lab == LAB_NON)], minlength=NBINS),
        len(doc.text),
    )


def cluster_histograms(stats: Iterable[DocStats]) -> tuple[list[str], np.ndarray, np.ndarray]:
    idx: dict[str, int] = {}
    pii, non = [], []
    for s in stats:
        if s.cluster_id not in idx:
            idx[s.cluster_id] = len(idx)
            pii.append(np.zeros(NBINS, dtype=np.int64))
            non.append(np.zeros(NBINS, dtype=np.int64))
        k = idx[s.cluster_id]
        pii[k] += s.hist_pii
        non[k] += s.hist_non
    return list(idx), np.array(pii), np.array(non)


def curve_from_hist(hist_pii: np.ndarray, hist_non: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """leak and over-redaction for thresholds GRID (k/1000, k=0..1000), plus the point
    'mask nothing' appended at the end (over=0, leak=1). Works on 1-D or batched (B, NBINS)."""
    hp = np.atleast_2d(hist_pii).astype(np.float64)
    hn = np.atleast_2d(hist_non).astype(np.float64)
    npii = hp.sum(1, keepdims=True)
    nnon = hn.sum(1, keepdims=True)
    # masked iff bin >= 1 + k  -> count of bins >= j via reverse cumsum
    rp = np.cumsum(hp[:, ::-1], axis=1)[:, ::-1]   # rp[:, j] = # with bin >= j
    rn = np.cumsum(hn[:, ::-1], axis=1)[:, ::-1]
    masked_pii = rp[:, 1:]                         # j = 1..1001 -> k = 0..1000
    masked_non = rn[:, 1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        leak = np.where(npii > 0, 1 - masked_pii / npii, np.nan)
        over = np.where(nnon > 0, masked_non / nnon, np.nan)
    leak = np.concatenate([leak, np.ones((leak.shape[0], 1))], axis=1)
    over = np.concatenate([over, np.zeros((over.shape[0], 1))], axis=1)
    if np.ndim(hist_pii) == 1:
        return leak[0], over[0]
    return leak, over


def best_leak_at(leak: np.ndarray, over: np.ndarray, budget: float) -> np.ndarray:
    """Lowest leak among operating points with over-redaction <= budget (no interpolation)."""
    L = np.atleast_2d(leak); O = np.atleast_2d(over)
    masked = np.where(O <= budget + 1e-12, L, np.inf)
    out = masked.min(1)
    return out if np.ndim(leak) > 1 else out[0]


def pauc(leak: np.ndarray, over: np.ndarray, max_over: float = 0.05, steps: int = 101) -> np.ndarray:
    """Normalized partial area under the best-achievable leak curve over over-redaction
    budgets in [0, max_over]: the mean over budgets of the lowest leak reachable within
    budget. Lower is better; 1.0 means nothing is ever masked. Budgets beyond a system's
    reachable over-redaction use its lowest reachable leak (what it can actually do)."""
    xs = np.linspace(0, max_over, steps)
    vals = np.stack([best_leak_at(leak, over, x) for x in xs], axis=-1)
    trap = getattr(np, "trapezoid", None) or np.trapz
    return trap(vals, xs, axis=-1) / max_over


def reachable_max_over(over: np.ndarray) -> float:
    return float(np.nanmax(over))


def at_threshold(hist_pii: np.ndarray, hist_non: np.ndarray, t: float) -> dict:
    k = int(round(t * 1000))
    leak, over = curve_from_hist(hist_pii, hist_non)
    tp = hist_pii[1 + k:].sum(); fn = hist_pii[:1 + k].sum(); fp = hist_non[1 + k:].sum()
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * r / (p + r) if p == p and r == r and p + r else float("nan")
    return {"threshold": t, "leak": float(leak[k]), "over": float(over[k]),
            "char_precision": float(p), "char_recall": float(r), "char_f1": float(f1)}


# ------------------------------------------------------------------ span-level metrics

def _masked_at(scores: np.ndarray, t: float) -> np.ndarray:
    return scores >= t - 1e-12


def span_exposure(doc: Doc, preds: Sequence[Span], t: float, word_granularity: bool = True) -> tuple[int, int]:
    """(# gold PII spans with at least one unmasked alphanumeric char, # gold PII spans with
    at least one alphanumeric char). Characters inside IGNORE-only regions are irrelevant here
    because we only look inside PII spans."""
    sc = char_scores(doc, preds, word_granularity)
    m = _masked_at(sc, t)
    an = alnum_mask(doc.text)
    exposed = total = 0
    for s in doc.pii_spans():
        seg = an[s.start:s.end]
        if not seg.any():
            continue
        total += 1
        exposed += int((~m[s.start:s.end] & seg).any())
    return exposed, total


def consistency(doc: Doc, preds: Sequence[Span], t: float, word_granularity: bool = True) -> tuple[int, int]:
    """Multi-mention consistency: among gold PII entities mentioned 2+ times (same casefolded
    surface within a document), how many have every mention fully masked."""
    sc = char_scores(doc, preds, word_granularity)
    m = _masked_at(sc, t)
    an = alnum_mask(doc.text)
    groups: dict[str, list[Span]] = {}
    for s in doc.pii_spans():
        groups.setdefault(doc.text[s.start:s.end].strip().casefold(), []).append(s)
    ok = tot = 0
    for g in groups.values():
        if len(g) < 2:
            continue
        tot += 1
        ok += int(all(not (~m[s.start:s.end] & an[s.start:s.end]).any() for s in g))
    return ok, tot


def span_prf(docs: Sequence[Doc], preds_by_doc: dict, t: float, *, typed: bool, mode: str) -> dict:
    """Span P/R/F1 at threshold t. mode='strict' (exact offsets) or 'partial' (any overlap).
    Greedy one-to-one matching by overlap size. IGNORE gold spans absorb predictions (a
    prediction overlapping only IGNORE gold is neither TP nor FP)."""
    tp = fp = fn = 0
    for d in docs:
        gold = d.pii_spans()
        ign = d.ignore_spans()
        pr = [p for p in preds_by_doc.get(d.doc_id, []) if p.score >= t - 1e-12]
        used = set()
        for p in pr:
            best, bo = None, 0
            for gi, g in enumerate(gold):
                if gi in used or (typed and g.label_canonical != p.label_canonical):
                    continue
                if mode == "strict":
                    ov = (g.end - g.start) if (g.start, g.end) == (p.start, p.end) else 0
                else:
                    ov = max(0, min(g.end, p.end) - max(g.start, p.start))
                if ov > bo:
                    best, bo = gi, ov
            if best is not None:
                used.add(best); tp += 1
            elif any(min(g.end, p.end) > max(g.start, p.start) for g in ign):
                continue
            else:
                fp += 1
        fn += len(gold) - len(used)
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if (p == p and r == r and p + r) else float("nan")
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def per_type_recall(docs: Sequence[Doc], preds_by_doc: dict, t: float) -> dict:
    hit = np.zeros(len(CANONICAL_TYPES)); tot = np.zeros(len(CANONICAL_TYPES))
    for d in docs:
        lab, typ = char_labels(d)
        an = alnum_mask(d.text)
        m = _masked_at(char_scores(d, preds_by_doc.get(d.doc_id, [])), t)
        sel = an & (lab == LAB_PII)
        np.add.at(tot, typ[sel], 1)
        np.add.at(hit, typ[sel & m], 1)
    return {CANONICAL_TYPES[i]: {"recall": float(hit[i] / tot[i]) if tot[i] else float("nan"),
                                 "support_chars": int(tot[i])} for i in range(len(CANONICAL_TYPES))}


def length_bucket(n: int) -> str:
    return "<1k" if n < 1000 else ("1k-10k" if n < 10000 else ">10k")
