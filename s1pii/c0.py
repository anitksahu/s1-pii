"""Leakage audit, the C2 propagation decision, and the preregistered C0 decision.

    python -m s1pii.c0 audit --models $DRIVE/models          # per S1 variant, from training manifests
    python -m s1pii.c0 c2 --pred <PII-TRACE calib predictions of s1_all-sources-s1>
    python -m s1pii.c0 decide [--fresh-real-absent]

Audit: every headline test split is audited against the training sources of the S1 variant
scored on it (``no-nemotron`` for Nemotron, ``all-sources`` elsewhere), rebuilt from the
trained model's manifest and checked against its recorded source counts. One index per
variant. Whole clusters containing a flagged document are removed for every system.

C2 (on the PII-TRACE *calibration* split, before C0): precision of propagated spans and the
multi-mention consistency with and without them (propagation is post-processing, so the
no-propagation system is the same prediction set minus propagated spans). The decision is
written to ``c2.json`` and ``decide`` enforces it on every S1 prediction file.

C0: the family must be complete (17 comparisons, or 16 with ``--fresh-real-absent``), every
S1 cell must have exactly seeds 1..3 of the right variant with the headline config, every
prediction file must match the current test split, and no (system, split) may have more
than one prediction file. Otherwise ``c0_holds`` is None and the reasons are listed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .schema import read_jsonl, Doc
from .audit.dedup import AuditIndex, drop_flagged
from .data import loaders as L
from .data.synth import generate
from .eval.evaluate import compare, views
from .eval.bootstrap import c0_decision
from .eval import metrics as M
from .ledger import read_predictions, append, dataset_hash
from . import ledger, bench

HEADLINE = ["tab_direct", "spy_medical", "spy_legal", "pii_trace", "nemotron", "fresh_real"]
BASELINES = ["gliner2_pii", "nvidia_gliner_pii", "gliner25_base_zeroshot"]
EXCLUDED = {("nemotron", "nvidia_gliner_pii")}
S1_VARIANT = {"nemotron": "no-nemotron"}
SEEDS = (1, 2, 3)
FLOOR = 0.01


class DuplicatePredictions(RuntimeError):
    pass


def variant_for(dataset: str) -> str:
    return S1_VARIANT.get(dataset, "all-sources")


def s1_system(variant: str, seed: int) -> str:
    return f"s1_{variant}-s{seed}"


# ------------------------------------------------------------------ audit

def training_sources_from_manifest(manifest: dict) -> list[Doc]:
    cfg = manifest["config"]
    docs = generate(cfg["synth_n"], seed=cfg["synth_seed"])
    counts = {"synthetic_conv": len(docs)}
    g = bench.dev_slice(L.load("gretel", "train", purpose="train"))[1]
    docs += g; counts["gretel"] = len(g)
    if cfg["variant"] == "all-sources":
        n = bench.dev_slice(L.load("nemotron", "train", purpose="train"))[1]
        docs += n; counts["nemotron"] = len(n)
    want = manifest.get("data", {}).get("source_counts")
    if cfg.get("max_train_docs"):
        raise ValueError("headline models must not use max_train_docs")
    if not want:
        raise ValueError("model manifest has no data.source_counts; cannot verify the audit sources")
    if want != counts:
        raise ValueError(f"rebuilt training sources {counts} differ from the trained model's {want}")
    return docs


def audit_path(dataset: str) -> Path:
    return ledger.RESULTS / "audit" / f"{dataset}.json"


def run_audit(models_dir: Path, datasets: list[str] | None = None) -> dict:
    """``models_dir`` holds ``<variant>-s<seed>/final/s1_manifest.json``; seed 1 of each
    variant defines the training sources (all seeds share data by construction)."""
    out, index, seed_hashes = {}, {}, {}
    for ds in datasets or HEADLINE:
        test_path = bench.split_paths(ds)["test"]
        if not test_path.exists():
            out[ds] = "missing test split"; continue
        v = variant_for(ds)
        if v not in index:
            man = json.loads((models_dir / f"{v}-s1" / "final" / "s1_manifest.json").read_text())
            hashes = {}
            for sd in SEEDS:
                mp = models_dir / f"{v}-s{sd}" / "final" / "s1_manifest.json"
                if not mp.exists():
                    continue
                d = json.loads(mp.read_text()).get("data", {})
                if d.get("source_counts") != man.get("data", {}).get("source_counts"):
                    raise ValueError(f"{v}-s{sd} was trained on different sources than {v}-s1")
                hashes[f"s{sd}"] = d.get("examples_hash")
            # synth-v0.1 drew dates of birth relative to the run date, so runs started on a
            # different UTC day have different DOB strings (same sources and counts): reported
            seed_hashes[v] = hashes
            index[v] = AuditIndex(training_sources_from_manifest(man))
        test = read_jsonl(test_path)
        rep = index[v].query(test)
        rep.update({"dataset": ds, "variant": v, "test_hash": dataset_hash(test),
                    "training_examples_hash_per_seed": seed_hashes.get(v)})
        p = audit_path(ds); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rep, indent=2))
        out[ds] = {k: rep[k] for k in ("n_test", "n_flagged", "dropped_doc_frac", "dropped_cluster_frac")}
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


# ------------------------------------------------------------------ prediction files

def prediction_index(pred_dir: Path | list[Path] | None = None) -> dict[tuple[str, str], Path]:
    """(system, resolved docs_path) -> the single merged prediction file. Raises if a key has
    more than one file (stale predictions from an older environment): delete or archive the
    stale ones explicitly. ``pred_dir`` may be a list of directories (v2 + v1 baselines)."""
    groups: dict[tuple[str, str], list[Path]] = {}
    dirs = pred_dir if isinstance(pred_dir, list) else [pred_dir or ledger.RESULTS / "predictions"]
    for p in sorted(q for d in dirs for q in d.glob("*.jsonl")):
        with open(p, encoding="utf-8") as f:
            meta = json.loads(f.readline()).get("_meta", {})
        if meta.get("system") and meta.get("docs_path"):
            groups.setdefault((meta["system"], str(Path(meta["docs_path"]).resolve())), []).append(p)
    dups = {k: v for k, v in groups.items() if len(v) > 1}
    if dups:
        raise DuplicatePredictions("multiple prediction files for " +
                                   "; ".join(f"{k[0]} on {Path(k[1]).name}: {[x.name for x in v]}" for k, v in dups.items()))
    return {k: v[0] for k, v in groups.items()}


def _load_checked(path: Path, test_docs_all: list[Doc], keep: set[str]) -> tuple[dict, dict]:
    meta, preds = read_predictions(path)
    if meta.get("dataset_hash") != dataset_hash(test_docs_all):
        raise ValueError(f"{path.name} was produced on different test docs")
    return meta, {k: v for k, v in preds.items() if k in keep}


# ------------------------------------------------------------------ C2

def c2_path() -> Path:
    return ledger.RESULTS / "c2.json"


def decide_propagation(pred_path: Path, min_precision: float = 0.90) -> dict:
    """C2 on the PII-TRACE calibration split, from one S1 prediction file made with
    propagation on. The file must be the calibration split, never test."""
    calib_path = bench.split_paths("pii_trace")["calib"]
    docs = read_jsonl(calib_path)
    meta, preds = read_predictions(pred_path)
    if meta.get("dataset_hash") != dataset_hash(docs):
        raise ValueError("C2 must be decided on the PII-TRACE calibration split")
    if not meta.get("config", {}).get("propagation"):
        raise ValueError("C2 needs predictions made with propagation on")
    ver = meta.get("versions", {})
    if meta.get("system") != s1_system("all-sources", 1) or ver.get("variant") != "all-sources" or ver.get("seed") != 1:
        raise ValueError(f"C2 must use {s1_system('all-sources', 1)} predictions, got {meta.get('system')}")
    prop = tp = exact = 0
    for d in docs:
        gold = d.pii_spans()
        for s in preds[d.doc_id]:
            if s.label_raw == "propagated":
                prop += 1
                tp += any(min(g.end, s.end) > max(g.start, s.start) for g in gold)
                exact += any(g.start == s.start and g.end == s.end for g in gold)
    without = {k: [s for s in v if s.label_raw != "propagated"] for k, v in preds.items()}
    def cons(p):
        vs = views(docs, p, "pii_trace")
        c = [M.consistency(v, 0.5) for v in vs]
        g = sum(x["groups"] for x in c)
        gc = sum(x["cond_den"] for x in c)
        return {"all": sum(x["all_masked"] for x in c) / g if g else float("nan"),
                "conditional": sum(x["cond_num"] for x in c) / gc if gc else float("nan")}
    precision = tp / prop if prop else float("nan")
    cw, co = cons(preds), cons(without)
    better = cw["all"] > co["all"] and not (cw["conditional"] < co["conditional"])
    res = {"propagated_spans": prop, "precision": precision, "exact_precision": exact / prop if prop else float("nan"),
           "consistency_with": cw, "consistency_without": co,
           "propagation": bool(prop and precision >= min_precision and better), "source": str(pred_path)}
    c2_path().parent.mkdir(parents=True, exist_ok=True)
    c2_path().write_text(json.dumps(res, indent=2))
    append({"kind": "c2", **res})
    return res


# ------------------------------------------------------------------ C0

def _check_s1_meta(meta: dict, variant: str, seed: int, propagation: bool) -> list[str]:
    v, c = meta.get("versions", {}), meta.get("config", {})
    errs = []
    if v.get("variant") != variant or v.get("seed") != seed:
        errs.append(f"variant/seed {v.get('variant')}/{v.get('seed')} != {variant}/{seed}")
    if c.get("validators") is not True:
        errs.append("validators off")
    if c.get("propagation") is not propagation:
        errs.append(f"propagation {c.get('propagation')} != C2 decision {propagation}")
    if c.get("floor") != FLOOR or c.get("o_exit_bias") != 0.0:
        errs.append(f"floor/o_exit_bias {c.get('floor')}/{c.get('o_exit_bias')}")
    return errs


def decide(n_boot: int = 10_000, seed: int = 0, fresh_real_absent: bool = False, pred_dir: Path | list[Path] | None = None,
           system_fn=None, meta_check=None, kind: str = "c0") -> dict:
    """``system_fn(variant, seed)`` names the S1 system (default v1); ``meta_check(meta,
    variant, seed, dataset)`` returns problems for an S1 prediction file (default: the v1
    headline config and the C2 decision)."""
    system_fn = system_fn or s1_system
    problems: list[str] = []
    if meta_check is not None:
        propagation = None
    elif not c2_path().exists():
        problems.append("C2 decision missing (run `python -m s1pii.c0 c2` on PII-TRACE calibration)")
        propagation = None
    else:
        propagation = json.loads(c2_path().read_text())["propagation"]
    if fresh_real_absent and "fresh_real" in HEADLINE and bench.split_paths("fresh_real")["test"].exists():
        problems.append("--fresh-real-absent given but the Fresh-Real test split exists; the fallback is only allowed when it is not ready")
    sets = [d for d in HEADLINE if not (fresh_real_absent and d == "fresh_real")]
    expected = sum(1 for d in sets for b in BASELINES if (d, b) not in EXCLUDED)
    idx = prediction_index(pred_dir)
    results = {}
    for ds in sets:
        test_path = bench.split_paths(ds)["test"]
        if not test_path.exists():
            problems.append(f"{ds}: test split missing"); continue
        all_docs = read_jsonl(test_path)
        docs = audited_test_docs(ds)
        keep = {d.doc_id for d in docs}
        tp = str(test_path.resolve())
        v = variant_for(ds)
        s1 = []
        for sd in SEEDS:
            key = (system_fn(v, sd), tp)
            if key not in idx:
                problems.append(f"{ds}: missing {key[0]}"); continue
            meta, preds = _load_checked(idx[key], all_docs, keep)
            if meta_check is not None:
                problems += [f"{ds}/{key[0]}: {e}" for e in meta_check(meta, v, sd, ds)]
            elif propagation is not None:
                problems += [f"{ds}/{key[0]}: {e}" for e in _check_s1_meta(meta, v, sd, propagation)]
            s1.append(preds)
        for b in BASELINES:
            if (ds, b) in EXCLUDED:
                continue
            if (b, tp) not in idx:
                problems.append(f"{ds}: missing {b}"); continue
            bmeta, bp = _load_checked(idx[(b, tp)], all_docs, keep)
            from .adapters.base import load_config
            pinned = load_config()["systems"][b].get("revision")
            if pinned is None:
                problems.append(f"{b}: revision not pinned in baselines.yaml")
            elif bmeta.get("revision") != pinned:
                problems.append(f"{ds}/{b}: predictions from revision {bmeta.get('revision')} != pinned {pinned}")
            if len(s1) == len(SEEDS):
                results[f"{ds}|{b}"] = compare(docs, s1, [bp], dataset=ds, n_boot=n_boot, seed=seed)
    if len(results) != expected:
        problems.append(f"family has {len(results)} comparisons, expected {expected}")
    needed = 3 if fresh_real_absent else 4
    decision = c0_decision(results, needed=needed) if results else {}
    if problems:
        decision["c0_holds"] = None
    out = {"sets": sets, "fresh_real_absent": fresh_real_absent, "expected_family": expected,
           "problems": problems, "comparisons": results, "decision": decision}
    append({"kind": kind, **out}, headline=not problems)
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a_ = sub.add_parser("audit"); a_.add_argument("--models", type=Path, required=True)
    c_ = sub.add_parser("c2"); c_.add_argument("--pred", type=Path, required=True)
    d_ = sub.add_parser("decide"); d_.add_argument("--n-boot", type=int, default=10_000)
    d_.add_argument("--fresh-real-absent", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "audit":
        print(json.dumps(run_audit(a.models), indent=2))
    elif a.cmd == "c2":
        print(json.dumps(decide_propagation(a.pred), indent=2))
    else:
        r = decide(n_boot=a.n_boot, fresh_real_absent=a.fresh_real_absent)
        (ledger.RESULTS / "c0.json").write_text(json.dumps(r, indent=2, default=str))
        print(json.dumps(r["decision"], indent=2, default=str)[:4000], "\nproblems:", r["problems"])


if __name__ == "__main__":
    main(sys.argv[1:])
