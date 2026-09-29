"""One-call evaluation of a system on a dataset, threshold tuning, and paired comparison."""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ..schema import Doc, Span, validate_predictions
from .. import taxonomy as tx
from . import metrics as M
from . import bootstrap as B
from . import calibration as C

BUDGETS = (0.005, 0.01, 0.02, 0.05)


class IncompletePredictions(ValueError):
    pass


def views(docs: Sequence[Doc], preds_by_doc: dict[str, list[Span]], dataset: str | None) -> list[M.DocView]:
    want = {d.doc_id for d in docs}
    have = set(preds_by_doc)
    if want - have:
        raise IncompletePredictions(f"{len(want - have)} docs have no prediction entry (a doc with no "
                                    f"predictions needs an explicit empty list), e.g. {sorted(want - have)[:3]}")
    allowed = tx.admissible(dataset) if dataset else None
    out = []
    for d in docs:
        validate_predictions(d, preds_by_doc[d.doc_id])
        out.append(M.view(d, preds_by_doc[d.doc_id], allowed))
    return out


def evaluate(docs: Sequence[Doc], preds_by_doc: dict[str, list[Span]], *, dataset: str,
             default_threshold: float, dev_threshold: float | None = None, max_over: float = 0.05,
             n_boot_ci: int = 2000, seed: int = 0) -> dict:
    vs = views(docs, preds_by_doc, dataset)
    clusters, hp, hn = M.cluster_histograms(vs)
    tp, tn = hp.sum(0), hn.sum(0)
    leak, over = M.curve_from_hist(tp, tn)
    rmo = M.reachable_max_over(over)
    out = {
        "dataset": dataset, "n_docs": len(docs), "n_clusters": len(clusters),
        "pii_chars": int(tp.sum()), "non_pii_chars": int(tn.sum()),
        "pauc": M.pauc(leak, over, max_over),
        "pauc_ci": B.single_bootstrap_ci(hp, hn, B.pauc_stat(max_over), n_boot_ci, seed),
        "reachable_max_over": rmo, "reachable_below_budget": rmo < max_over,
        "best_leak_at_budget": {str(b): M.best_leak_at(leak, over, b) for b in BUDGETS},
        "operating_points": {},
    }
    points = {"default": default_threshold}
    if dev_threshold is not None:
        points["dev_tuned"] = dev_threshold
    buckets: dict[str, list[M.DocView]] = {}
    for v in vs:
        buckets.setdefault(M.length_bucket(len(v.doc.text)), []).append(v)
    for name, t in points.items():
        op = M.at_threshold(tp, tn, t)
        exp = [M.span_exposure(v, t) for v in vs]
        con = [M.consistency(v, t) for v in vs]
        e_den = sum(b for _, b in exp)
        g = sum(c["groups"] for c in con); cd = sum(c["cond_den"] for c in con)
        op.update({
            "span_exposure": sum(a for a, _ in exp) / e_den if e_den else float("nan"),
            "consistency_all": sum(c["all_masked"] for c in con) / g if g else float("nan"),
            "consistency_conditional": sum(c["cond_num"] for c in con) / cd if cd else float("nan"),
            "span_strict_typed": M.span_prf(vs, t, typed=True, mode="strict"),
            "span_partial_untyped": M.span_prf(vs, t, typed=False, mode="partial"),
            "per_type_recall": M.per_type_recall(vs, t),
            "recall_by_length": {k: M.at_threshold(*[x.sum(0) for x in M.cluster_histograms(b)[1:]], t)["char_recall"]
                                 for k, b in buckets.items()},
        })
        out["operating_points"][name] = op
    cand = C.candidates(vs)
    out["calibration_raw"] = {"brier": C.brier(cand.scores, cand.targets),
                              "adaptive_ece": C.adaptive_ece(cand.scores, cand.targets),
                              "n_candidates": int(len(cand.scores)),
                              "uncovered_gold": cand.uncovered_gold, "total_gold": cand.total_gold}
    return out


def tune_threshold(docs: Sequence[Doc], preds_by_doc: dict, dataset: str, over_budget: float = 0.01) -> float:
    """Shared-dev operating point: the lowest grid threshold with over-redaction <= budget on
    the development/calibration split. Same rule for every system; never run on test.
    Returns ``metrics.MASK_NOTHING`` if no threshold meets the budget."""
    _, hp, hn = M.cluster_histograms(views(docs, preds_by_doc, dataset))
    leak, over = M.curve_from_hist(hp.sum(0), hn.sum(0))
    ok = np.where(over[:-1] <= over_budget + 1e-12)[0]
    return float(M.GRID[ok.min()]) if len(ok) else M.MASK_NOTHING


def system_hist(docs: Sequence[Doc], preds_by_doc: dict, dataset: str, order: Sequence[str] | None = None):
    clusters, hp, hn = M.cluster_histograms(views(docs, preds_by_doc, dataset))
    order = order or clusters
    return list(order), B.align(clusters, (hp, hn), order)


def compare(docs: Sequence[Doc], preds_a_seeds: Sequence[dict], preds_b_seeds: Sequence[dict], *,
            dataset: str, max_over: float = 0.05, n_boot: int = 10_000, seed: int = 0) -> dict:
    """Paired cluster bootstrap of mean-over-seeds pAUC(A) - pAUC(B) under shared weights."""
    order, _ = system_hist(docs, preds_a_seeds[0], dataset)
    ha = [system_hist(docs, p, dataset, order)[1] for p in preds_a_seeds]
    hb = [system_hist(docs, p, dataset, order)[1] for p in preds_b_seeds]
    return B.paired_bootstrap_multi(ha, hb, B.pauc_stat(max_over), n_boot, seed)
