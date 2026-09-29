"""One-call evaluation of a system's predictions on a dataset, and the headline comparison."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..schema import Doc, Span, validate_predictions
from . import metrics as M
from . import bootstrap as B
from . import calibration as C

BUDGETS = (0.005, 0.01, 0.02, 0.05)


def histograms(docs: Sequence[Doc], preds_by_doc: dict[str, list[Span]], word_granularity: bool = True):
    stats = []
    for d in docs:
        preds = preds_by_doc.get(d.doc_id, [])
        validate_predictions(d, preds)
        stats.append(M.doc_stats(d, preds, word_granularity))
    return M.cluster_histograms(stats)


def evaluate(docs: Sequence[Doc], preds_by_doc: dict[str, list[Span]], *, default_threshold: float,
             dev_threshold: float | None = None, max_over: float = 0.05, n_boot_ci: int = 2000,
             seed: int = 0) -> dict:
    clusters, hp, hn = histograms(docs, preds_by_doc)
    tp, tn = hp.sum(0), hn.sum(0)
    leak, over = M.curve_from_hist(tp, tn)
    out = {
        "n_docs": len(docs), "n_clusters": len(clusters),
        "pii_chars": int(tp.sum()), "non_pii_chars": int(tn.sum()),
        "pauc": float(M.pauc(leak, over, max_over)),
        "pauc_ci": B.single_bootstrap_ci(hp, hn, B.pauc_stat(max_over), n_boot_ci, seed),
        "reachable_max_over": M.reachable_max_over(over),
        "best_leak_at_budget": {str(b): float(M.best_leak_at(leak, over, b)) for b in BUDGETS},
        "operating_points": {},
    }
    points = {"default": default_threshold}
    if dev_threshold is not None:
        points["dev_tuned"] = dev_threshold
    for name, t in points.items():
        op = M.at_threshold(tp, tn, t)
        exp = [M.span_exposure(d, preds_by_doc.get(d.doc_id, []), t) for d in docs]
        con = [M.consistency(d, preds_by_doc.get(d.doc_id, []), t) for d in docs]
        e_num, e_den = sum(a for a, _ in exp), sum(b for _, b in exp)
        c_num, c_den = sum(a for a, _ in con), sum(b for _, b in con)
        op.update({
            "span_exposure": e_num / e_den if e_den else float("nan"),
            "consistency": c_num / c_den if c_den else float("nan"),
            "span_strict_typed": M.span_prf(docs, preds_by_doc, t, typed=True, mode="strict"),
            "span_partial_untyped": M.span_prf(docs, preds_by_doc, t, typed=False, mode="partial"),
            "per_type_recall": M.per_type_recall(docs, preds_by_doc, t),
        })
        buckets: dict[str, list[Doc]] = {}
        for d in docs:
            buckets.setdefault(M.length_bucket(len(d.text)), []).append(d)
        op["recall_by_length"] = {}
        for k, ds in buckets.items():
            _, bp, bn = histograms(ds, preds_by_doc)
            op["recall_by_length"][k] = M.at_threshold(bp.sum(0), bn.sum(0), t)["char_recall"]
        out["operating_points"][name] = op
    cand = C.candidates(docs, preds_by_doc)
    out["calibration_raw"] = {"brier": C.brier(cand.scores, cand.targets),
                              "adaptive_ece": C.adaptive_ece(cand.scores, cand.targets),
                              "n_candidates": int(len(cand.scores)),
                              "uncovered_gold": cand.uncovered_gold, "total_gold": cand.total_gold}
    return out



def tune_threshold(docs: Sequence[Doc], preds_by_doc: dict, over_budget: float) -> float:
    """Shared-dev operating point: the lowest threshold whose over-redaction <= budget on the
    development split. Identical rule for every system; never run on a test split."""
    _, hp, hn = histograms(docs, preds_by_doc)
    leak, over = M.curve_from_hist(hp.sum(0), hn.sum(0))
    ok = np.where(over[:-1] <= over_budget + 1e-12)[0]
    return float(M.GRID[ok.min()]) if len(ok) else 1.0


def compare(docs: Sequence[Doc], preds_a: dict, preds_b: dict, *, max_over: float = 0.05,
            n_boot: int = 10_000, seed: int = 0) -> dict:
    ca, hpa, hna = histograms(docs, preds_a)
    cb, hpb, hnb = histograms(docs, preds_b)
    hpa, hpb = B.aligned(ca, hpa, cb, hpb, ca)
    hna, hnb = B.aligned(ca, hna, cb, hnb, ca)
    return B.paired_bootstrap(hpa, hna, hpb, hnb, B.pauc_stat(max_over), n_boot, seed)
