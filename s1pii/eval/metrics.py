"""Character-level redaction metrics.

Definitions (preregistered in prereg/v0.md)
------------------------------------------
* Only alphanumeric characters (``str.isalnum``) count; whitespace and punctuation never do.
* Gold character classes, by precedence when gold spans overlap: PII > IGNORE > NOT_PII.
  Characters under no gold span are non-PII. IGNORE characters are excluded from both the
  numerator and denominator of every metric.
* Predictions whose type is not admissible for the benchmark (``taxonomy.ADMISSIBLE``) are
  dropped for every system. Remaining predictions are expanded to whole alphanumeric runs
  (``align.expand_to_words``); each character gets the maximum *raw* score of any prediction
  covering it (-1 if none). Scores are quantized to a 1/1000 grid (bins); a character is
  masked at threshold k/1000 iff its bin >= 1 + k. Every metric uses the same bins.
* leak(t) = unmasked gold-PII chars / gold-PII chars;
  over(t) = masked non-PII chars / non-PII chars; both micro-averaged over documents.
* pAUC: exact integral over over-redaction budgets x in [0, max_over] of the lowest leak
  among operating points with over <= x, divided by max_over. Budgets beyond a system's
  reachable over-redaction use its lowest reachable leak. Lower is better.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ..schema import Doc, Span, IGNORE, CANONICAL_TYPES
from .align import expand_to_words, word_bounds

NBINS = 1002          # bin 0: no prediction; 1 + floor(1000 * score) for score in [0, 1]
GRID = np.arange(1001) / 1000.0
MASK_NOTHING = 1.001  # threshold sentinel: nothing is masked

LAB_NON, LAB_IGN, LAB_PII = 0, 1, 2


def k_of(t: float) -> int:
    """Threshold -> grid index k; masked iff bin >= 1 + k. k = 1001 masks nothing."""
    return min(1001, max(0, int(np.ceil(round(t * 1000, 6)))))


def char_labels(doc: Doc) -> tuple[np.ndarray, np.ndarray]:
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


def filter_admissible(preds: Sequence[Span], allowed: frozenset[str] | None) -> list[Span]:
    return list(preds) if allowed is None else [p for p in preds if p.label_canonical in allowed]


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


def merged_gold(doc: Doc) -> list[tuple[int, int, str]]:
    """Union of overlapping gold PII spans (type of the earliest). Used by span-level metrics
    so overlapping annotations from several annotators count as one mention."""
    out: list[list] = []
    for s in sorted(doc.pii_spans(), key=lambda s: (s.start, -s.end)):
        if out and s.start < out[-1][1]:
            out[-1][1] = max(out[-1][1], s.end)
        else:
            out.append([s.start, s.end, s.label_canonical])
    return [tuple(x) for x in out]


@dataclass
class DocView:
    """Everything a metric needs for one (doc, system) pair, computed once."""
    doc: Doc
    preds: list[Span]
    lab: np.ndarray
    typ: np.ndarray
    alnum: np.ndarray
    bins: np.ndarray
    gold: list[tuple[int, int, str]]

    @property
    def cluster_id(self) -> str:
        return self.doc.cluster_id or self.doc.doc_id

    def masked(self, t: float) -> np.ndarray:
        return self.bins >= 1 + k_of(t)

    def hist(self) -> tuple[np.ndarray, np.ndarray]:
        return (np.bincount(self.bins[self.alnum & (self.lab == LAB_PII)], minlength=NBINS),
                np.bincount(self.bins[self.alnum & (self.lab == LAB_NON)], minlength=NBINS))


def view(doc: Doc, preds: Sequence[Span], allowed: frozenset[str] | None = None,
         word_granularity: bool = True) -> DocView:
    kept = filter_admissible(preds, allowed)
    lab, typ = char_labels(doc)
    return DocView(doc, kept, lab, typ, alnum_mask(doc.text),
                   to_bins(char_scores(doc, kept, word_granularity)), merged_gold(doc))


def cluster_histograms(views: Iterable[DocView]) -> tuple[list[str], np.ndarray, np.ndarray]:
    idx: dict[str, int] = {}
    pii, non = [], []
    for v in views:
        hp, hn = v.hist()
        if v.cluster_id not in idx:
            idx[v.cluster_id] = len(idx)
            pii.append(np.zeros(NBINS)); non.append(np.zeros(NBINS))
        pii[idx[v.cluster_id]] += hp
        non[idx[v.cluster_id]] += hn
    return list(idx), np.array(pii, dtype=np.float64), np.array(non, dtype=np.float64)


# ------------------------------------------------------------------ curves

def curve_from_hist(hist_pii: np.ndarray, hist_non: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """leak and over for thresholds GRID (k = 0..1000) plus the mask-nothing point (k = 1001).
    Works on 1-D histograms or batched (B, NBINS)."""
    hp = np.atleast_2d(hist_pii).astype(np.float64)
    hn = np.atleast_2d(hist_non).astype(np.float64)
    npii = hp.sum(1, keepdims=True)
    nnon = hn.sum(1, keepdims=True)
    rp = np.cumsum(hp[:, ::-1], axis=1)[:, ::-1]      # rp[:, j] = # chars with bin >= j
    rn = np.cumsum(hn[:, ::-1], axis=1)[:, ::-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        leak = np.where(npii > 0, 1 - rp[:, 1:] / npii, np.nan)
        over = np.where(nnon > 0, rn[:, 1:] / nnon, 0.0)
    leak = np.concatenate([leak, np.ones((leak.shape[0], 1))], axis=1)
    over = np.concatenate([over, np.zeros((over.shape[0], 1))], axis=1)
    return (leak[0], over[0]) if np.ndim(hist_pii) == 1 else (leak, over)


def best_leak_at(leak: np.ndarray, over: np.ndarray, budget: float):
    L = np.atleast_2d(leak); O = np.atleast_2d(over)
    out = np.where(O <= budget + 1e-12, L, np.inf).min(1)
    return out if np.ndim(leak) > 1 else float(out[0])


def pauc(leak: np.ndarray, over: np.ndarray, max_over: float = 0.05):
    """Exact normalized integral of the best-achievable-leak step function on [0, max_over]."""
    L = np.atleast_2d(leak).astype(np.float64); O = np.atleast_2d(over).astype(np.float64)
    order = np.argsort(O, axis=1, kind="stable")
    Os = np.take_along_axis(O, order, 1)
    Ls = np.minimum.accumulate(np.take_along_axis(L, order, 1), axis=1)
    x0 = np.minimum(Os, max_over)
    x1 = np.minimum(np.concatenate([Os[:, 1:], np.full((Os.shape[0], 1), np.inf)], 1), max_over)
    area = (Ls * (x1 - x0)).sum(1) / max_over
    return area if np.ndim(leak) > 1 else float(area[0])


def reachable_max_over(over: np.ndarray) -> float:
    return float(np.nanmax(over))


def at_threshold(hist_pii: np.ndarray, hist_non: np.ndarray, t: float) -> dict:
    k = k_of(t)
    leak, over = curve_from_hist(hist_pii, hist_non)
    tp = hist_pii[1 + k:].sum(); fn = hist_pii[:1 + k].sum(); fp = hist_non[1 + k:].sum()
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * r / (p + r) if (p == p and r == r and p + r) else float("nan")
    return {"threshold": t, "leak": float(leak[k]), "over": float(over[k]),
            "char_precision": float(p), "char_recall": float(r), "char_f1": float(f1)}


# ------------------------------------------------------------------ span-level metrics

def span_exposure(v: DocView, t: float) -> tuple[int, int]:
    """(# merged gold PII mentions with >= 1 unmasked alphanumeric char, # mentions with any
    alphanumeric char)."""
    m = v.masked(t)
    exposed = total = 0
    for a, b, _ in v.gold:
        seg = v.alnum[a:b]
        if not seg.any():
            continue
        total += 1
        exposed += int((~m[a:b] & seg).any())
    return exposed, total


def consistency(v: DocView, t: float) -> dict:
    """Multi-mention consistency over gold entities mentioned 2+ times in a document (same
    casefolded surface). ``all_masked``: every mention fully masked. ``conditional``: among
    entities with at least one mention fully masked, how many have all mentions masked."""
    m = v.masked(t)
    groups: dict[str, list[tuple[int, int]]] = {}
    for a, b, _ in v.gold:
        groups.setdefault(v.doc.text[a:b].strip().casefold(), []).append((a, b))
    full = lambda a, b: not (~m[a:b] & v.alnum[a:b]).any()
    res = {"all_masked": 0, "groups": 0, "cond_num": 0, "cond_den": 0}
    for g in groups.values():
        if len(g) < 2:
            continue
        flags = [full(a, b) for a, b in g]
        res["groups"] += 1
        res["all_masked"] += int(all(flags))
        if any(flags):
            res["cond_den"] += 1
            res["cond_num"] += int(all(flags))
    return res


def _max_matching(pairs: list[tuple[int, int, int]], n_pred: int, n_gold: int) -> tuple[set, set]:
    """Maximum-cardinality one-to-one matching, ties broken by total overlap. Order independent."""
    if not pairs:
        return set(), set()
    from scipy.optimize import linear_sum_assignment
    big = 1 + sum(ov for ov, _, _ in pairs)
    w = np.zeros((n_pred, n_gold))
    for ov, pi, gi in pairs:
        w[pi, gi] = max(w[pi, gi], big + ov)
    rows, cols = linear_sum_assignment(w, maximize=True)
    keep = [(r, c) for r, c in zip(rows, cols) if w[r, c] > 0]
    return {r for r, _ in keep}, {c for _, c in keep}


def span_prf(views: Sequence[DocView], t: float, *, typed: bool, mode: str) -> dict:
    """Span P/R/F1 at threshold t against merged gold. mode 'strict' (exact offsets) or
    'partial' (any character overlap). One-to-one matching is maximum-cardinality with ties
    broken by total overlap (order-independent). A prediction overlapping only IGNORE gold is neither TP nor FP."""
    k = k_of(t)
    tp = fp = fn = 0
    for v in views:
        gold = v.gold
        ign = [(s.start, s.end) for s in v.doc.ignore_spans()]
        pr = [p for p in v.preds if 1 + int(np.floor(p.score * 1000 + 1e-9)) >= 1 + k]
        pairs = []
        for pi, p in enumerate(pr):
            for gi, (a, b, lab) in enumerate(gold):
                if b <= p.start or a >= p.end or (typed and lab != p.label_canonical):
                    continue
                if mode == "strict":
                    if (a, b) == (p.start, p.end):
                        pairs.append((b - a, pi, gi))
                else:
                    pairs.append((min(b, p.end) - max(a, p.start), pi, gi))
        up, ug = _max_matching(pairs, len(pr), len(gold))
        tp += len(up)
        for pi, p in enumerate(pr):
            if pi in up:
                continue
            if any(min(b, p.end) > max(a, p.start) for a, b in ign) and \
                    not any(min(b, p.end) > max(a, p.start) for a, b, _ in gold):
                continue
            fp += 1
        fn += len(gold) - len(ug)
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f = 2 * p * r / (p + r) if (p == p and r == r and p + r) else float("nan")
    return {"precision": p, "recall": r, "f1": f, "tp": tp, "fp": fp, "fn": fn}


def per_type_recall(views: Sequence[DocView], t: float) -> dict:
    hit = np.zeros(len(CANONICAL_TYPES)); tot = np.zeros(len(CANONICAL_TYPES))
    for v in views:
        sel = v.alnum & (v.lab == LAB_PII)
        m = v.masked(t)
        np.add.at(tot, v.typ[sel], 1)
        np.add.at(hit, v.typ[sel & m], 1)
    return {CANONICAL_TYPES[i]: {"recall": float(hit[i] / tot[i]) if tot[i] else float("nan"),
                                 "support_chars": int(tot[i])} for i in range(len(CANONICAL_TYPES))}


def length_bucket(n: int) -> str:
    return "<1k" if n < 1000 else ("1k-10k" if n < 10000 else ">10k")
