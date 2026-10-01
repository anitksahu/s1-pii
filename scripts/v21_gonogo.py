"""v2.1 go/no-go (exploratory; decides whether the v2.1 flat plan is worth ~8.5 A100-h).

Question: with a frozen v1 encoder, do sentence-encoder label embeddings (bge-base) make the
flat label-conditioned CRF type held-out labels better than v1-encoder label embeddings, all
else equal? A GO says nothing about the fine-tuned setting of the full plan (which keeps a
v1-label arm to measure that).

    python scripts/v21_gonogo.py prep    # CPU runtime: training subset, stratified eval sample, bge-base probe
    python scripts/v21_gonogo.py train   # GPU: M1 (v1 labels) and M2 (bge labels) x seeds 1, 2; 1000 steps
    python scripts/v21_gonogo.py eval    # GPU: M0 (v2 flat, 3000 steps), M1 and M2 per seed
    python scripts/v21_gonogo.py decide  # CPU: the rule in docs/PLAN_v2.1_draft.md

Arms differ only in the label encoder: both use L2-normalized label vectors from at most 128
tokens, the same text sampling (name / paraphrase / description, name dropout 0.4), label-set
sizes 4-24, steps, seeds and training subset (no-nemotron seed-1 docs, the 30% subset of the v2
flat run). Evaluation reads Nemotron *calibration* docs only (also used for thresholds in the
full plan; test is untouched).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

D = Path(os.environ.get("S1PII_DRIVE", "/content/drive/MyDrive/s1pii"))
G = D / "v21" / "gonogo"
VARIANT, FRAC, STEPS = "no-nemotron", 0.3, 1000
SEEDS = (1, 2)
MIN_PER_LABEL, MAX_DOCS, MIN_N_MACRO, MIN_LABELS = 30, 300, 20, 5
M0_SEEN_FLOOR, MAX_SKIPPED = 0.30, 0.05     # INVALID (not NO-GO) below these sanity bounds


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def ensure_heldout():
    """The held-out file is tracked from prereg-v1; fall back to the Drive copy if missing."""
    from s1pii.v2 import labels as LB
    p = LB.heldout_path()
    if not p.exists():
        shutil.copy(D / "v2" / "heldout_labels.yaml", p)
    return {x.lower() for x in LB.load_heldout()["labels"]}


def v1_dir(seed: int = 1) -> Path:
    return D / "models" / f"{VARIANT}-s{seed}" / "final"


def prep():
    from s1pii.schema import write_jsonl, read_jsonl
    from s1pii import bench
    from s1pii.model.train import TrainConfig, training_docs
    held = ensure_heldout()
    G.mkdir(parents=True, exist_ok=True)
    if not (G / "train_docs.jsonl").exists():
        man = json.loads((v1_dir() / "s1_manifest.json").read_text())
        c = man["config"]
        docs, counts = training_docs(TrainConfig(variant=VARIANT, seed=1, synth_n=c["synth_n"], synth_seed=c["synth_seed"]))
        want = man.get("data", {}).get("source_counts")
        if want and want != counts:
            raise ValueError(f"rebuilt sources {counts} != trained {want}")
        keep = [d for d in docs if int(hashlib.sha256(f"flat:{d.doc_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < FRAC]
        write_jsonl(keep, G / "train_docs.jsonl")
        log("train docs", len(keep), counts)
    if not (G / "eval_docs.jsonl").exists():
        cal = sorted(read_jsonl(bench.split_paths("nemotron")["calib"]), key=lambda d: d.doc_id)
        has = {d.doc_id: Counter((s.label_raw or "").lower() for s in d.spans if (s.label_raw or "").lower() in held) for d in cal}
        avail = sum(has.values(), Counter())
        rng = np.random.default_rng(0)
        chosen, chosen_ids, got = [], set(), Counter()
        for lab in sorted(avail, key=lambda l: (avail[l], l)):        # rarest first
            pool = [d for d in cal if has[d.doc_id][lab] and d.doc_id not in chosen_ids]
            for i in rng.permutation(len(pool)):
                if got[lab] >= MIN_PER_LABEL or len(chosen) >= MAX_DOCS:
                    break
                chosen.append(pool[i]); chosen_ids.add(pool[i].doc_id); got.update(has[pool[i].doc_id])
        write_jsonl(sorted(chosen, key=lambda d: d.doc_id), G / "eval_docs.jsonl")
        n_macro = sum(v >= MIN_N_MACRO for v in got.values())
        info = {"eval_docs": len(chosen), "eval_docs_sha": _sha(G / "eval_docs.jsonl"), "per_label_n": dict(sorted(got.items())),
                "available_in_calib": dict(sorted(avail.items())), "labels_n_ge_20": n_macro, "heldout": sorted(held),
                "note": "Nemotron calibration docs read by the go/no-go (test untouched)"}
        (G / "prep.json").write_text(json.dumps(info, indent=2))
        log(info)
        if n_macro < MIN_LABELS:
            raise SystemExit(f"only {n_macro} held-out labels reach n >= {MIN_N_MACRO}: go/no-go not informative")
    if not (G / "probe.json").exists():
        probe()


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def probe():
    """Paraphrase/description -> label name top-1 over the 55 Nemotron labels, bge-base; pins the revision."""
    from s1pii.v2 import labels as LB
    from s1pii.v2.head import label_text_variants
    from s1pii.v2.flat import LabelEncoder
    held = ensure_heldout()
    names = sorted({l.name for l in LB.c3_label_set().labels})
    var = {n: label_text_variants(n) for n in names}
    nt = [var[n]["name"][0] for n in names]
    q = [(i, t, n in held) for i, n in enumerate(names) for t in var[n]["para"] + var[n]["desc"] if t != nt[i]]
    le = LabelEncoder("bge", device="cpu")
    nv = le(nt).numpy(); qv = le([t for _, t, _ in q]).numpy()

    def top1(sel):
        rows = [(i, v) for (i, _, h), v in zip(q, qv) if sel(h)]
        return sum(int(np.argmax(nv @ v) == i) for i, v in rows) / len(rows)
    r = {"model": le.model, "revision": le.revision, "all_top1": top1(lambda h: True), "heldout_top1": top1(lambda h: h)}
    (G / "probe.json").write_text(json.dumps(r, indent=2))
    log(r)


def arms():
    rev = json.loads((G / "probe.json").read_text())["revision"]
    return {"M1_v1labels": dict(label_encoder="v1", label_normalize=True, label_max_length=128),
            "M2_bgelabels": dict(label_encoder="bge", label_revision=rev)}


def train():
    from s1pii.schema import read_jsonl
    from s1pii.v2 import labels as LB
    from s1pii.v2.flat import train_flat
    ensure_heldout()
    docs = read_jsonl(G / "train_docs.jsonl")
    held = LB.heldout_nodes()
    m0 = json.loads((D / "v2" / "flat" / f"{VARIANT}-s1" / "flat_manifest.json").read_text())
    if sorted(m0["heldout_nodes"]) != sorted(held):
        raise RuntimeError("held-out nodes differ from the v2 flat model (M0)")
    for seed in SEEDS:
        for name, kw in arms().items():
            out = G / f"{name}-s{seed}"
            if (out / "flat_manifest.json").exists():
                log(out.name, "exists"); continue
            t0 = time.time()
            train_flat(v1_dir(), docs, out, held, seed=seed, steps=STEPS, text_mode="sample", log=log, **kw)
            log(out.name, f"trained in {(time.time() - t0) / 60:.1f} min")


def _score(docs, pr, held, vocab):
    """Per gold span: exact-boundary match and correct argmax label; grouped held-out / seen (training vocab)."""
    rows = []
    for d in docs:
        for s in d.spans:
            lab = (s.label_raw or "").lower()
            grp = "heldout" if lab in held else ("seen" if lab in vocab else None)
            if grp is None:
                continue
            ex = [p for p in pr.get(d.doc_id, []) if p.start == s.start and p.end == s.end]
            best = max(ex, key=lambda p: p.score) if ex else None
            rows.append({"doc": d.doc_id, "label": lab, "grp": grp, "match": best is not None,
                         "correct": best is not None and best.label_raw.lower() == lab})
    return rows


def evaluate():
    from s1pii.schema import read_jsonl
    from s1pii.v2 import labels as LB
    from s1pii.v2.flat import predict_flat
    held = ensure_heldout()
    docs = read_jsonl(G / "eval_docs.jsonl")
    L = LB.c3_label_set(True)
    v1sha = json.loads((v1_dir() / "s1_manifest.json").read_text())["weights_sha256"]
    models = {"M0_v2flat_3000": D / "v2" / "flat" / f"{VARIANT}-s1"}
    models.update({f"{n}-s{s}": G / f"{n}-s{s}" for s in SEEDS for n in arms()})
    (G / "eval").mkdir(exist_ok=True)
    for name, fd in models.items():
        out = G / "eval" / f"{name}.json"
        if out.exists():
            log(name, "evaluated"); continue
        fman = json.loads((fd / "flat_manifest.json").read_text())
        if fman["v1"] != v1sha:
            raise RuntimeError(f"{name}: trained on another v1 encoder")
        if sorted(fman["heldout_nodes"]) != sorted(LB.heldout_nodes()):
            raise RuntimeError(f"{name}: different held-out set")
        t0 = time.time()
        pr = predict_flat(v1_dir(), fd, docs, L, system=name, score="typed", floor=1e-4, label_floor=1e-4, max_span=128)
        rows = _score(docs, pr, held, set(fman["vocab"]))
        out.write_text(json.dumps({"rows": rows, "minutes": round((time.time() - t0) / 60, 1),
                                   "eval_docs_sha": _sha(G / "eval_docs.jsonl"), "steps": fman.get("steps"),
                                   "skipped_steps": fman.get("skipped_steps")}))
        log(name, summarize(rows))


def summarize(rows):
    r = {}
    for grp in ("heldout", "seen"):
        g = [x for x in rows if x["grp"] == grp]
        per = {}
        for x in g:
            c = per.setdefault(x["label"], Counter()); c["n"] += 1; c["m"] += x["match"]; c["a"] += x["correct"]
        big = {k: v for k, v in per.items() if v["n"] >= MIN_N_MACRO}
        r[grp] = {"n": len(g), "match": float(np.mean([x["match"] for x in g])) if g else None,
                  "acc": float(np.mean([x["correct"] for x in g])) if g else None,
                  "macro_acc": float(np.mean([v["a"] / v["n"] for v in big.values()])) if big else None,
                  "per_label": {k: {"n": v["n"], "match": round(v["m"] / v["n"], 3), "acc": round(v["a"] / v["n"], 3)}
                                for k, v in sorted(per.items())}}
    return r


def _macro(rows, labels):
    per = {}
    for x in rows:
        if x["grp"] == "heldout" and x["label"] in labels:
            c = per.setdefault(x["label"], [0, 0]); c[0] += 1; c[1] += x["correct"]
    return float(np.mean([a / n for n, a in per.values()])) if per else 0.0


def _counts(rows, labels):
    c = {}
    for x in rows:
        if x["grp"] == "heldout" and x["label"] in labels:
            v = c.setdefault(x["label"], [0, 0]); v[0] += 1; v[1] += x["correct"]
    return c


def decide():
    """INVALID (sanity failure) if M0's seen-label accuracy < 0.30 or any arm skipped > 5% of
    steps or the evals were scored on different samples. Otherwise GO iff, for EACH seed: macro
    held-out typed accuracy (labels with n >= 20, at least 5 such labels; exact boundary and
    correct label over all gold spans) of M2 >= M1 + 0.15 and >= 0.40; doc-level bootstrap 95% CI
    lower bound of the difference > 0; M2 > M1 on a majority of those labels (raw counts); M2
    held-out match >= M1 - 0.05; M2 seen-label accuracy >= M1 - 0.03."""
    need = ["M0_v2flat_3000"] + [f"{n}-s{s}" for s in SEEDS for n in arms()]
    missing = [k for k in need if not (G / "eval" / f"{k}.json").exists()]
    if missing:
        raise SystemExit(f"missing evals: {missing}")
    ev = {k: json.loads((G / "eval" / f"{k}.json").read_text()) for k in need}
    sums = {k: summarize(v["rows"]) for k, v in ev.items()}
    out = {"seeds": {}, "summaries": {k: {g: {kk: vv for kk, vv in s[g].items() if kk != "per_label"}
                                           for g in ("heldout", "seen")} for k, s in sums.items()}}
    invalid = []
    if len({v["eval_docs_sha"] for v in ev.values()}) != 1:
        invalid.append("evals scored on different eval samples")
    if (sums["M0_v2flat_3000"]["seen"]["acc"] or 0) < M0_SEEN_FLOOR:
        invalid.append(f"M0 seen-label accuracy < {M0_SEEN_FLOOR}")
    for k, v in ev.items():
        if k != "M0_v2flat_3000" and (v.get("skipped_steps") or 0) > MAX_SKIPPED * (v.get("steps") or STEPS):
            invalid.append(f"{k}: skipped steps {v['skipped_steps']}")
    go = True
    for seed in SEEDS:
        r1, r2 = ev[f"M1_v1labels-s{seed}"]["rows"], ev[f"M2_bgelabels-s{seed}"]["rows"]
        k1 = sorted((x["doc"], x["label"]) for x in r1 if x["grp"] == "heldout")
        if k1 != sorted((x["doc"], x["label"]) for x in r2 if x["grp"] == "heldout"):
            invalid.append(f"seed {seed}: M1 and M2 scored on different held-out spans"); go = False; continue
        s1, s2 = summarize(r1), summarize(r2)
        labs = sorted(k for k, v in s1["heldout"]["per_label"].items() if v["n"] >= MIN_N_MACRO)
        if len(labs) < MIN_LABELS:
            invalid.append(f"seed {seed}: only {len(labs)} labels with n >= {MIN_N_MACRO}"); go = False; continue
        m1, m2 = _macro(r1, labs), _macro(r2, labs)
        c1, c2 = _counts(r1, labs), _counts(r2, labs)
        wins = sum(c2[l][1] / c2[l][0] > c1[l][1] / c1[l][0] for l in labs)
        docs = sorted({x["doc"] for x in r1})
        rng = np.random.default_rng(0)
        by1, by2 = {}, {}
        for x in r1: by1.setdefault(x["doc"], []).append(x)
        for x in r2: by2.setdefault(x["doc"], []).append(x)
        diffs = []
        for _ in range(2000):
            bs = rng.choice(len(docs), len(docs))
            b1 = [x for i in bs for x in by1[docs[i]]]; b2 = [x for i in bs for x in by2.get(docs[i], [])]
            diffs.append(_macro(b2, labs) - _macro(b1, labs))
        lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
        cond = {"macro_gain>=0.15": m2 >= m1 + 0.15, "macro_M2>=0.40": m2 >= 0.40, "ci_low>0": lo > 0,
                "majority_of_labels": wins > len(labs) / 2,
                "match_not_worse": s2["heldout"]["match"] >= s1["heldout"]["match"] - 0.05,
                "seen_not_worse": (s2["seen"]["acc"] or 0.0) >= (s1["seen"]["acc"] or 0.0) - 0.03}
        go &= all(cond.values())
        out["seeds"][seed] = {"labels": labs, "macro_M1": m1, "macro_M2": m2, "wins": wins, "ci95_diff": [lo, hi], **cond}
    out["invalid"] = invalid
    out["verdict"] = "INVALID" if invalid else ("GO" if go else "NO-GO")
    (G / "decision.json").write_text(json.dumps(out, indent=2))
    log(json.dumps(out, indent=1)[:3000])


if __name__ == "__main__":
    {"prep": prep, "probe": probe, "train": train, "eval": evaluate, "decide": decide}[sys.argv[1]]()
