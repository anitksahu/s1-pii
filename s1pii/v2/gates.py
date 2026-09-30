"""Preregistered gates that decide the conditional GPU stages (prereg v1, PLAN_v2).

Computed only on calibration data (Nemotron dev slice, never test) and on held-back
training items, from the cheap path's outputs, before any conditional stage runs:

* Gate A (fine-tune the top encoder layers for typing) fires if either
  - the minimum over the 6 heads of held-back typing accuracy (name + description texts)
    is below 0.90, or
  - for either variant, the mean over its seeds of held-out typing accuracy on Nemotron
    calibration gold held-out spans (exact-match candidates, C3 label set) is below 0.50.
* Gate B (type-agnostic level 1 with quasi-identifier supervision) fires if, for either
  variant, the mean over its seeds of level-1 recall (a candidate with exactly the gold boundaries and
  P_b >= 0.5) on Nemotron calibration quasi-identifier and held-out gold spans is below 0.60.
* Stage C (flat label-conditioned CRF ablation) is an ablation, not accuracy-driven: it runs
  if the GPU-hour cap leaves room after A and B.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import taxonomy as tx
from ..schema import IGNORE, read_jsonl
from . import labels as LB
from .features import load_store

THRESH = {"A_head_val_acc": 0.90, "A_heldout_typing_acc": 0.50, "B_level1_recall": 0.60}


def level1_recall(store: Path, docs_path: Path, raw_labels: set[str], min_pb: float = 0.5) -> dict:
    z = load_store(store, ("doc_ids", "offsets", "start", "end", "pb"))
    docs = {d.doc_id: d for d in read_jsonl(docs_path)}
    hit = tot = 0
    for di, did in enumerate(z["doc_ids"].tolist()):
        a, b = int(z["offsets"][di]), int(z["offsets"][di + 1])
        cand = {(int(s), int(e)) for s, e, p in zip(z["start"][a:b], z["end"][a:b], z["pb"][a:b]) if p >= min_pb}
        for s in docs[did].spans:
            if s.label_raw.lower() in raw_labels:
                tot += 1; hit += (s.start, s.end) in cand
    return {"recall": hit / tot if tot else None, "n": tot}


def heldout_typing(store: Path, head_dir: Path, docs_path: Path, heldout_labels: set[str]) -> dict:
    from .infer import decide
    L = LB.c3_label_set()
    D = decide(store, head_dir, L)
    docs = {d.doc_id: d for d in read_jsonl(docs_path)}
    ok = n = 0
    names = [l.name for l in L.labels]
    for di, did in enumerate(D.doc_ids.tolist()):
        a, b = int(D.offsets[di]), int(D.offsets[di + 1])
        pos = {(int(D.start[r]), int(D.end[r])): r for r in range(a, b)}
        for s in docs[did].spans:
            raw = s.label_raw.lower()
            if raw in heldout_labels and (s.start, s.end) in pos:
                n += 1; ok += names[int(D.top[pos[(s.start, s.end)]])] == raw
    return {"acc": ok / n if n else None, "n_matched": n}


def evaluate_gates(head_dirs: list[Path], nem_calib_stores: dict[str, tuple[Path, Path]], nem_calib_docs: Path) -> dict:
    """``nem_calib_stores``: "<variant>-s<seed>" -> (Nemotron calibration store, head dir), for
    both variants. Held-out typing is required on each variant (A fires if either is low);
    level-1 recall uses the all-sources seeds (the variant scored on quasi-identifiers in C3)."""
    held = LB.load_heldout()
    held_labels = {x for x in held["synonyms_all_sources"]}
    quasi = {r.lower() for r, c in tx.NEMOTRON.items() if c == IGNORE}
    vals = []
    for h in head_dirs:
        man = json.loads((Path(h) / "head_manifest.json").read_text())
        vals.append(man.get("val", {}).get("desc", {}).get("typing_acc"))
    ht = {k: heldout_typing(s, h, nem_calib_docs, held_labels) for k, (s, h) in nem_calib_stores.items()}
    rec = {k: level1_recall(s, nem_calib_docs, quasi | held_labels) for k, (s, _) in nem_calib_stores.items()}
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return float(np.mean(xs)) if xs else 0.0
    ht_mean = {v: mean([x["acc"] for k, x in ht.items() if k.startswith(v)]) for v in ("all-sources", "no-nemotron")}
    rec_mean = {v: mean([x["recall"] for k, x in rec.items() if k.startswith(v)]) for v in ("all-sources", "no-nemotron")}
    known = [v for v in vals if v is not None]
    head_min = min(known) if known else None
    return {"thresholds": THRESH, "head_val_acc": vals, "head_val_acc_min": head_min,
            "heldout_typing": ht, "heldout_typing_mean": ht_mean, "level1_recall": rec, "level1_recall_mean": rec_mean,
            "A_fires": (head_min is None or head_min < THRESH["A_head_val_acc"])
                       or min(ht_mean.values()) < THRESH["A_heldout_typing_acc"],
            "B_fires": min(rec_mean.values()) < THRESH["B_level1_recall"]}
