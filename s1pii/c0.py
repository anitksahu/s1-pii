"""Leakage audit and the preregistered C0 decision.

    python -m s1pii.c0 audit                  # training sources vs every headline test split
    python -m s1pii.c0 decide --out results/c0.json

Audit (prereg): each headline test split is audited against the training sources of the S1
variant scored on it (``no-nemotron`` for Nemotron, ``all-sources`` elsewhere). Whole
clusters containing a flagged document are removed from headline scoring for every system.

C0: for every (headline set, baseline) pair, paired cluster bootstrap of mean-over-seeds
pAUC(S1) - pAUC(baseline) under shared weights; Holm over the family; a set is won iff S1
wins against every baseline evaluated on it; C0 holds iff >= 4 of 6 sets are won (3 of 5
if Fresh-Real is absent, declared).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from .schema import read_jsonl, Doc
from .audit.dedup import audit, drop_flagged
from .data import loaders as L
from .data.synth import generate
from .eval.evaluate import compare
from .eval.bootstrap import c0_decision
from .ledger import read_predictions, append, dataset_hash
from . import ledger
from . import bench

HEADLINE = ["tab_direct", "spy_medical", "spy_legal", "pii_trace", "nemotron", "fresh_real"]
BASELINES = ["gliner2_pii", "nvidia_gliner_pii", "gliner25_base_zeroshot"]
EXCLUDED = {("nemotron", "nvidia_gliner_pii")}                     # trained on Nemotron-PII
S1_VARIANT = {"nemotron": "no-nemotron"}                           # default: all-sources


def variant_for(dataset: str) -> str:
    return S1_VARIANT.get(dataset, "all-sources")


def training_sources(variant: str, synth_n: int = 20000, synth_seed: int = 17) -> list[Doc]:
    docs = generate(synth_n, seed=synth_seed)
    docs += bench.dev_slice(L.load("gretel", "train", purpose="train"))[1]
    if variant == "all-sources":
        docs += bench.dev_slice(L.load("nemotron", "train", purpose="train"))[1]
    return docs


def audit_path(dataset: str) -> Path:
    return ledger.RESULTS / "audit" / f"{dataset}.json"


def run_audit(datasets: list[str] | None = None, **kw) -> dict:
    out, cache = {}, {}
    for ds in datasets or HEADLINE:
        test_path = bench.split_paths(ds)["test"]
        if not test_path.exists():
            out[ds] = "missing test split"; continue
        v = variant_for(ds)
        if v not in cache:
            cache[v] = training_sources(v, **kw)
        test = read_jsonl(test_path)
        rep = audit(cache[v], test)
        rep.update({"dataset": ds, "variant": v, "test_hash": dataset_hash(test)})
        p = audit_path(ds); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, indent=2))
        out[ds] = {"n_test": rep["n_test"], "n_flagged": rep["n_flagged"]}
    return out


def audited_test_docs(dataset: str) -> list[Doc]:
    test = read_jsonl(bench.split_paths(dataset)["test"])
    p = audit_path(dataset)
    if not p.exists():
        raise FileNotFoundError(f"no leakage audit for {dataset}; run `python -m s1pii.c0 audit` first")
    rep = json.loads(p.read_text())
    if rep["test_hash"] != dataset_hash(test):
        raise ValueError(f"audit for {dataset} was run on different test docs")
    return drop_flagged(test, rep)


def prediction_index(pred_dir: Path | None = None) -> dict[tuple[str, str], Path]:
    """(system, docs_path) -> merged prediction file, from each file's meta line."""
    idx = {}
    for p in sorted((pred_dir or ledger.RESULTS / "predictions").glob("*.jsonl")):
        with open(p, encoding="utf-8") as f:
            meta = json.loads(f.readline()).get("_meta", {})
        if meta.get("system") and meta.get("docs_path"):
            idx[(meta["system"], str(Path(meta["docs_path"]).resolve()))] = p
    return idx


def decide(n_boot: int = 10_000, seed: int = 0, s1_prefix: str = "s1_", pred_dir: Path | None = None) -> dict:
    idx = prediction_index(pred_dir)
    results, missing = {}, []
    available = [d for d in HEADLINE if bench.split_paths(d)["test"].exists()]
    for ds in available:
        test_path = str(bench.split_paths(ds)["test"].resolve())
        docs = audited_test_docs(ds)
        keep = {d.doc_id for d in docs}
        s1_systems = sorted(s for (s, p) in idx if p == test_path and s.startswith(f"{s1_prefix}{variant_for(ds)}-s"))
        if len(s1_systems) < 3:
            missing.append(f"{ds}: {len(s1_systems)} S1 seeds found (need 3)")
            continue
        load = lambda s: {k: v for k, v in read_predictions(idx[(s, test_path)])[1].items() if k in keep}
        s1 = [load(s) for s in s1_systems]
        for b in BASELINES:
            if (ds, b) in EXCLUDED:
                continue
            if (b, test_path) not in idx:
                missing.append(f"{ds}: no predictions for {b}"); continue
            results[f"{ds}|{b}"] = compare(docs, s1, [load(b)], dataset=ds, n_boot=n_boot, seed=seed)
    needed = 4 if "fresh_real" in available else 3
    decision = c0_decision(results, needed=needed) if results else {"c0_holds": None}
    out = {"available_sets": available, "missing": missing, "comparisons": results, "decision": decision}
    append({"kind": "c0", **out}, headline=not missing)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("audit")
    d = sub.add_parser("decide"); d.add_argument("--out", type=Path, default=None); d.add_argument("--n-boot", type=int, default=10_000)
    a = ap.parse_args(argv)
    if a.cmd == "audit":
        print(json.dumps(run_audit(), indent=2))
    else:
        r = decide(n_boot=a.n_boot)
        (a.out or ledger.RESULTS / "c0.json").write_text(json.dumps(r, indent=2, default=str))
        print(json.dumps(r["decision"], indent=2, default=str), r["missing"])


if __name__ == "__main__":
    main(sys.argv[1:])
