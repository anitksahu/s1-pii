"""v2 claims: C0' (redaction under the benchmark label set), C3 (held-out labels), C4
(character-level calibration and abstention), robustness sweeps.

    python -m s1pii.v2.evaluate c0prime
    python -m s1pii.v2.evaluate c4
    python -m s1pii.v2.evaluate c3 --gliner <GLiNER2.5 C3 prediction file>

C4 (prereg v1). Unit: alphanumeric character, as for leak and over-redaction; IGNORE
characters are excluded. Per system and benchmark, on the calibration split:
* t_hi = the dev threshold of v1 (lowest threshold with over-redaction <= 1%);
* the deferral band [t_lo, t_hi) grows downward from t_hi, one score bin at a time, while
  the deferred share of scored characters stays <= 1 - coverage (coverage 0.95 primary,
  0.90 descriptive).
On test: selective leak = PII chars below t_lo / PII chars not deferred; selective
over-redaction = non-PII chars >= t_hi / non-PII chars not deferred. Win rule: paired
cluster bootstrap (3 seeds, shared weights) of selective leak at 95% coverage, Holm over
(baselines x benchmarks); a benchmark is won when S1 is significantly lower against every
baseline; C4 holds on >= 3 of 5 benchmarks. Character-level ECE after isotonic calibration
fitted on the calibration split is descriptive.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .. import bench, c0, ledger
from ..schema import read_jsonl
from ..ledger import read_predictions, dataset_hash
from ..eval import metrics as M
from ..eval.bootstrap import paired_bootstrap_multi, holm
from ..eval.evaluate import views, tune_threshold
from . import labels as LB
from .labels import c3_label_set

V2_PRED = ledger.RESULTS / "predictions_v2"
C4_SETS = ["tab_direct", "spy_medical", "spy_legal", "pii_trace", "nemotron"]
COVERAGES = (0.95, 0.90)


class NoHeadline(RuntimeError):
    pass


def headline_state() -> dict:
    """The preregistered headline stage, written by run_v2 from the gates (calibration data
    only). Required: ``$S1PII_V2_STATE`` must point at state.json, every gates file it lists
    must exist next to it with the recorded hash. There is no override."""
    import hashlib, os
    p = Path(os.environ.get("S1PII_V2_STATE", ""))
    if not p.is_file():
        raise NoHeadline("S1PII_V2_STATE must point at the chain's state.json (headline fixed by the gates)")
    st = json.loads(p.read_text())
    gates = st.get("gates") or {}
    if "cheap" not in gates:
        raise NoHeadline("state.json has no gate record: run `run_v2 conditional` (gates) before any claim")
    for tag, h in gates.items():
        g = p.parent / f"gates-{tag}.json"
        if not g.is_file() or hashlib.sha256(g.read_bytes()).hexdigest() != h:
            raise NoHeadline(f"gates file for {tag} missing or changed since the headline was fixed")
    return {"headline": st["headline"], "state_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "gates": gates}


def headline_tag() -> str:
    return headline_state()["headline"]


def v2_system(variant: str, seed: int, tag: str | None = None) -> str:
    return f"s1v2-{tag or headline_tag()}_{variant}-s{seed}"


def bench_system(variant: str, seed: int, tag: str | None = None) -> str:
    return f"s1v2bench-{tag or headline_tag()}_{variant}-s{seed}"


def _meta_checker(label_set):
    def check(meta: dict, variant: str, seed: int, dataset: str) -> list[str]:
        errs = []
        v, c = meta.get("versions", {}), meta.get("config", {})
        if v.get("variant") != variant or v.get("seed") != seed:
            errs.append(f"variant/seed {v.get('variant')}/{v.get('seed')} != {variant}/{seed}")
        want = label_set(dataset).hash()
        if c.get("labels_hash") != want:
            errs.append(f"label set {c.get('labels_hash')} != frozen set {want}")
        if c.get("validators") is not True or c.get("floor") != c0.FLOOR:
            errs.append("validators/floor differ from the headline config")
        rep = meta.get("report", {})
        if rep.get("dropped_topk", 0) > 0:
            errs.append(f"{rep['dropped_topk']} candidates dropped by the top-k cap (headline runs must drop none)")
        return errs
    return check


def c0prime(n_boot: int = 10_000, secondary: bool = False) -> dict:
    """C0 protocol and win rule unchanged; baselines = the v1 prediction files. Primary: S1
    queried with exactly the canonical labels.yaml strings the baselines received. Secondary
    (``secondary=True``): S1 queried with each benchmark's own labels (more information than
    the baselines had; reported, not a claim)."""
    hs = headline_state()
    tag = hs["headline"]
    if secondary:
        fn, ls, kind = (lambda v, sd: bench_system(v, sd, tag)), LB.load_benchmark_label_set, "c0prime_benchlabels"
    else:
        fn, ls, kind = (lambda v, sd: v2_system(v, sd, tag)), (lambda ds: LB.canonical_label_set()), "c0prime"
    r = c0.decide(n_boot=n_boot, fresh_real_absent=True, pred_dir=[V2_PRED, ledger.RESULTS / "predictions"],
                  system_fn=fn, meta_check=_meta_checker(ls), kind=kind)
    r["headline_state"] = hs
    return r


# ------------------------------------------------------------------ C4

def band_from_calib(vs, coverage: float, over_budget: float = 0.01, docs=None, preds=None, dataset=None) -> tuple[int, int]:
    """(lo_bin, hi_bin): masked iff bin >= hi_bin, deferred iff lo_bin <= bin < hi_bin.
    If no threshold meets the over-redaction budget on calibration (``MASK_NOTHING``), the
    operating point is infeasible: hi = 1002 masks nothing and the selective leak is 1.0 by
    construction. ``c4`` reports such cells as infeasible instead of scoring them."""
    t_hi = tune_threshold(docs, preds, dataset, over_budget=over_budget)
    hi = 1 + M.k_of(t_hi)
    hp = sum(v.hist()[0] for v in vs); hn = sum(v.hist()[1] for v in vs)
    tot = hp.sum() + hn.sum()
    cnt = hp + hn
    lo, deferred = hi, 0
    while lo > 1 and deferred + cnt[lo - 1] <= (1 - coverage) * tot:
        lo -= 1; deferred += cnt[lo]
    return lo, hi


def calib_masked(vs, hi: int) -> int:
    """Calibration characters masked at hi. Zero means the operating point masks nothing: the
    1% budget is met only by masking nothing (in the v2 run every system's SPY threshold was
    1.0 or MASK_NOTHING), so the selective leak is 1.0 by construction."""
    hp = sum(v.hist()[0] for v in vs); hn = sum(v.hist()[1] for v in vs)
    return int((hp + hn)[hi:].sum())


def selective_hist(vs, lo: int, hi: int, order: list[str]):
    """Per-cluster (pii_kept, pii_leaked) and (non_kept, non_masked) counts, aligned to order."""
    ix = {c: i for i, c in enumerate(order)}
    hp = np.zeros((len(order), 3)); hn = np.zeros((len(order), 3))
    for v in vs:
        i = ix[v.cluster_id]
        b = v.bins; a = v.alnum
        pii = a & (v.lab == M.LAB_PII); non = a & (v.lab == M.LAB_NON)
        kept = (b < lo) | (b >= hi)
        hp[i] += [np.sum(pii & kept), np.sum(pii & (b < lo)), np.sum(pii)]
        hn[i] += [np.sum(non & kept), np.sum(non & (b >= hi)), np.sum(non)]
    return hp, hn


def sel_leak(hp, hn):
    return hp[..., 1] / np.maximum(hp[..., 0], 1)


def sel_over(hp, hn):
    return hn[..., 1] / np.maximum(hn[..., 0], 1)


def coverage(hp, hn):
    return (hp[..., 0] + hn[..., 0]) / np.maximum(hp[..., 2] + hn[..., 2], 1)


ECE_BINS = 15
ECE_BOUND = 0.05     # prereg v1: upper 95% CI of test ECE (mean over seeds) must stay below this


def ece_hist(vs_cal, vs_test, order: list[str]):
    """Isotonic map fitted on calibration characters (score -> P(PII)); per test cluster the
    equal-width-bin sums of calibrated p and of labels (hp, 2 x bins) and bin counts (hn).
    Restricted to characters where a decision happens: covered by a candidate (bin > 0).
    Unscored characters would pile into one bin near 0 and make ECE vacuous."""
    from sklearn.isotonic import IsotonicRegression
    def xy(v):
        m = v.alnum & (v.lab != M.LAB_IGN) & (v.bins > 0)
        return (v.bins[m] - 1) / 1000.0, (v.lab[m] == M.LAB_PII).astype(float)
    sc = np.concatenate([xy(v)[0] for v in vs_cal]); yc = np.concatenate([xy(v)[1] for v in vs_cal])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(sc, yc)
    ix = {c: i for i, c in enumerate(order)}
    hp = np.zeros((len(order), 2 * ECE_BINS)); hn = np.zeros((len(order), ECE_BINS))
    for v in vs_test:
        s_, y = xy(v)
        p = iso.predict(s_) if len(s_) else s_
        b = np.minimum((p * ECE_BINS).astype(int), ECE_BINS - 1)
        i = ix[v.cluster_id]
        hp[i, :ECE_BINS] += np.bincount(b, weights=p, minlength=ECE_BINS)
        hp[i, ECE_BINS:] += np.bincount(b, weights=y, minlength=ECE_BINS)
        hn[i] += np.bincount(b, minlength=ECE_BINS)
    return hp, hn


def ece_stat(hp, hn):
    return np.abs(hp[..., :ECE_BINS] - hp[..., ECE_BINS:]).sum(-1) / np.maximum(hn.sum(-1), 1)


def brier_skill(vs_cal, vs_test) -> dict:
    """Brier skill of the isotonic-calibrated score vs the calibration base rate, on scored characters."""
    from sklearn.isotonic import IsotonicRegression
    def xy(vs):
        s_, y = [], []
        for v in vs:
            m = v.alnum & (v.lab != M.LAB_IGN) & (v.bins > 0)
            s_.append((v.bins[m] - 1) / 1000.0); y.append(v.lab[m] == M.LAB_PII)
        return np.concatenate(s_), np.concatenate(y).astype(float)
    sc, yc = xy(vs_cal); st, yt = xy(vs_test)
    if len(sc) == 0 or len(st) == 0:
        return {"brier_skill": None, "n_scored_chars": int(len(st))}
    p = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(sc, yc).predict(st)
    b, b0 = np.mean((p - yt) ** 2), np.mean((yc.mean() - yt) ** 2)
    return {"brier_skill": float(1 - b / b0) if b0 > 0 else None, "n_scored_chars": int(len(yt))}


def char_ece(vs_cal, vs_test, n_bins: int = 15) -> dict:
    from sklearn.isotonic import IsotonicRegression
    def xy(vs):
        s, y = [], []
        for v in vs:
            m = v.alnum & (v.lab != M.LAB_IGN)
            s.append(np.where(v.bins[m] > 0, (v.bins[m] - 1) / 1000.0, 0.0)); y.append(v.lab[m] == M.LAB_PII)
        return np.concatenate(s), np.concatenate(y).astype(float)
    sc, yc = xy(vs_cal); st, yt = xy(vs_test)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(sc, yc)
    p = iso.predict(st)
    q = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, n_bins - 1)
    ece = sum(abs(p[idx == b].mean() - yt[idx == b].mean()) * (idx == b).mean() for b in range(n_bins) if (idx == b).any())
    return {"ece": float(ece), "brier": float(np.mean((p - yt) ** 2)), "n_chars": int(len(yt))}


def c4(n_boot: int = 10_000, seed: int = 0) -> dict:
    idx = c0.prediction_index([V2_PRED, ledger.RESULTS / "predictions"])
    results, problems, desc = {}, [], {}
    ece_ci, ece_cmp, coverage_flags = {}, {}, []
    infeasible: list[str] = []   # "<dataset>|<system>": no calib threshold meets the 1% budget
    hs_ = headline_state()
    for ds in C4_SETS:
        paths = bench.split_paths(ds)
        cal_docs, test_all = read_jsonl(paths["calib"]), read_jsonl(paths["test"])
        test_docs = c0.audited_test_docs(ds); keep = {d.doc_id for d in test_docs}
        v = c0.variant_for(ds)
        systems = {"s1v2": [v2_system(v, sd) for sd in c0.SEEDS]}
        for b in c0.BASELINES:
            if (ds, b) not in c0.EXCLUDED:
                systems[b] = [b]
        per_sys, per_ece = {}, {}
        order = None
        infeasible_sys = set()
        for name, members in systems.items():
            hs = {cov: [] for cov in COVERAGES}
            eces, eh = [], []
            for sysname in members:
                kc, kt = (sysname, str(paths["calib"].resolve())), (sysname, str(paths["test"].resolve()))
                if kc not in idx or kt not in idx:
                    problems.append(f"{ds}: missing calib or test predictions for {sysname}"); break
                _, pc = read_predictions(idx[kc])
                tmeta, pt = read_predictions(idx[kt])
                if tmeta.get("dataset_hash") != dataset_hash(test_all):
                    problems.append(f"{ds}/{sysname}: test predictions on different docs"); break
                pt = {k: x for k, x in pt.items() if k in keep}
                vc, vt = views(cal_docs, pc, ds), views(test_docs, pt, ds)
                if order is None:
                    order = sorted({x.cluster_id for x in vt})
                for cov in COVERAGES:
                    lo, hi = band_from_calib(vc, cov, docs=cal_docs, preds=pc, dataset=ds)
                    if hi > 1001 or calib_masked(vc, hi) == 0:
                        infeasible_sys.add(name)
                    hs[cov].append(selective_hist(vt, lo, hi, order))
                eces.append(brier_skill(vc, vt))
                eh.append(ece_hist(vc, vt, order))
            else:
                per_sys[name] = hs
                per_ece[name] = eh
                desc.setdefault(ds, {})[name] = {
                    "brier_skill_mean": float(np.mean([e["brier_skill"] for e in eces if e["brier_skill"] is not None]))
                                        if any(e["brier_skill"] is not None for e in eces) else None,
                    "n_scored_chars": [e["n_scored_chars"] for e in eces],
                    "ece_binned_mean": float(np.mean([ece_stat(h[0].sum(0), h[1].sum(0)) for h in eh])),
                    **{f"test_coverage{int(c * 100)}": float(np.mean([coverage(h[0].sum(0), h[1].sum(0)) for h in hs[c]]))
                       for c in COVERAGES},
                    **{f"cov{int(c * 100)}": {"sel_leak": float(np.mean([sel_leak(h[0].sum(0), h[1].sum(0)) for h in hs[c]])),
                                              "sel_over": float(np.mean([sel_over(h[0].sum(0), h[1].sum(0)) for h in hs[c]]))}
                       for c in COVERAGES}}
        for name, d in desc.get(ds, {}).items():
            for c in COVERAGES:
                if d[f"test_coverage{int(c * 100)}"] < c - 0.01:
                    coverage_flags.append(f"{ds}/{name}: test coverage {d[f'test_coverage{int(c * 100)}']:.3f} < nominal {c} - 0.01")
        if "s1v2" not in per_sys:
            continue
        zero = [(np.zeros_like(per_ece["s1v2"][0][0]), np.zeros_like(per_ece["s1v2"][0][1]))]
        ece_ci[ds] = paired_bootstrap_multi(per_ece["s1v2"], zero, ece_stat, n_boot, seed)
        for b in sorted(infeasible_sys):
            infeasible.append(f"{ds}|{b}")
        for b in systems:
            if b == "s1v2" or b not in per_sys:
                continue
            if "s1v2" in infeasible_sys or b in infeasible_sys:
                continue                     # no operating point within budget: not a comparison
            results[f"{ds}|{b}"] = paired_bootstrap_multi(per_sys["s1v2"][0.95], per_sys[b][0.95],
                                                          lambda hp, hn: sel_leak(hp, hn), n_boot, seed)
            ece_cmp[f"{ds}|{b}"] = paired_bootstrap_multi(per_ece["s1v2"], per_ece[b], ece_stat, n_boot, seed)
    h = holm({k: r["p_value"] for k, r in results.items()}) if results else {}
    he = holm({k: r["p_value"] for k, r in ece_cmp.items()}) if ece_cmp else {}
    ece_worse = sorted(k for k, r in ece_cmp.items() if r["diff"] > 0 and he[k]["reject"])
    ece_over = sorted(ds for ds, r in ece_ci.items() if r["ci_high"] >= ECE_BOUND)
    won = {}
    for k, r in results.items():
        ds = k.split("|")[0]
        won.setdefault(ds, []).append(r["diff"] < 0 and h[k]["reject"])
    won_sets = sorted(d for d, w in won.items() if all(w))
    c4a = len(won_sets) >= 3
    c4b = not ece_worse and not ece_over and len(ece_ci) == len(C4_SETS)
    out = {"c4a_selective_leak": {"comparisons": results, "holm": h, "datasets_won": won_sets, "holds": c4a,
                                  "infeasible": infeasible,
                                  "infeasible_note": "v2.1 (post hoc): cells whose calibration operating point masks no "
                                                     "calibration character (MASK_NOTHING, or a threshold above every "
                                                     "score) are excluded from the Holm family and the win rule; a dataset with "
                                                     "an infeasible S1 cell cannot be won"},
           "c4b_calibration": {"bound": ECE_BOUND, "s1_ece_ci": ece_ci, "vs_baselines": ece_cmp, "holm": he,
                               "significantly_worse": ece_worse, "above_bound": ece_over, "holds": c4b},
           "c4_holds": (c4a and c4b) if not problems else None,
           "property_claimed": ("calibrated abstention" if c4a and c4b else "selective abstention" if c4a else None)
                               if not problems else None,
           "coverage_flags": coverage_flags, "descriptive": desc, "problems": problems, "headline_state": hs_}
    ledger.append({"kind": "c4", **out}, headline=not problems)
    return out


# ------------------------------------------------------------------ C3

def greedy_flat(spans):
    """Non-overlapping decoding: highest score first."""
    out = []
    for s in sorted(spans, key=lambda s: -s.score):
        if all(s.end <= o.start or s.start >= o.end for o in out):
            out.append(s)
    return out


def typed_prf(docs, preds, labels: set[str], t: float) -> dict:
    """Strict (exact boundaries and label) micro and macro F1 restricted to ``labels``."""
    tp = {l: 0 for l in labels}; fp = dict(tp); fn = dict(tp)
    by_doc = {}
    for d in docs:
        gold = {(s.start, s.end, s.label_raw.lower()) for s in d.spans if s.label_raw.lower() in labels}
        pr = greedy_flat([p for p in preds.get(d.doc_id, []) if p.score >= t])
        pred = {(p.start, p.end, p.label_raw.lower()) for p in pr if p.label_raw.lower() in labels}
        for x in pred & gold:
            tp[x[2]] += 1
        for x in pred - gold:
            fp[x[2]] += 1
        for x in gold - pred:
            fn[x[2]] += 1
        by_doc[d.doc_id] = (pred, gold)
    def f1(a, b, c):
        return 2 * a / max(1, 2 * a + b + c)
    per = {l: {"f1": f1(tp[l], fp[l], fn[l]), "support": tp[l] + fn[l]} for l in labels}
    return {"micro_f1": f1(sum(tp.values()), sum(fp.values()), sum(fn.values())),
            "macro_f1": float(np.mean([v["f1"] for v in per.values() if v["support"]])) if per else 0.0,
            "per_label": per}


def gold_typing_accuracy(docs, preds, labels: set[str]) -> dict:
    """Threshold-free: for each gold span with a label in ``labels``, the label of the
    highest-scoring prediction with exactly its boundaries (none -> wrong)."""
    ok = n = matched = 0
    for d in docs:
        by = {}
        for p in preds.get(d.doc_id, []):
            k = (p.start, p.end)
            if k not in by or p.score > by[k].score:
                by[k] = p
        for s in d.spans:
            if s.label_raw.lower() in labels:
                n += 1
                p = by.get((s.start, s.end))
                matched += p is not None
                ok += p is not None and p.label_raw.lower() == s.label_raw.lower()
    return {"acc": ok / n if n else None, "matched_frac": matched / n if n else None, "n": n}


def tune_typed_threshold(docs, preds, labels: set[str]) -> float:
    """Per-system threshold maximizing micro F1 on calibration, seen labels only."""
    grid = np.round(np.arange(0.05, 0.96, 0.05), 2)
    return float(max(grid, key=lambda t: typed_prf(docs, preds, labels, t)["micro_f1"]))


def c3(gliner_path: Path | None, gliner2pii_path: Path | None = None, t: float = 0.5, n_boot: int = 10_000,
       seed: int = 0, variant: str = "no-nemotron", gliner_calib: Path | None = None) -> dict:
    """Primary: the no-nemotron variant (never saw Nemotron text or labels), 3 seeds, typed
    predictions under ``c3_label_set``; secondary: ``variant='all-sources'`` (in-domain: trained
    on Nemotron train, including held-out spans under coarse v1 types). Win rule on strict
    micro and macro F1 at t = 0.5 vs GLiNER2.5 zero-shot; threshold-free gold typing accuracy
    and calibration-tuned thresholds (seen labels only) are secondary."""
    from .. import taxonomy as tx
    hs = headline_state(); tag = hs["headline"]
    h = LB.load_heldout()
    held = set(h["labels"])
    seen = {r.lower() for r in tx.NEMOTRON} - held
    test = read_jsonl(bench.split_paths("nemotron")["test"])
    calib = read_jsonl(bench.split_paths("nemotron")["calib"])
    idx = c0.prediction_index([V2_PRED])
    tp = str(bench.split_paths("nemotron")["test"].resolve())
    cp = str(bench.split_paths("nemotron")["calib"].resolve())
    s1, s1c = [], []
    for sd in c0.SEEDS:
        sysn = f"s1v2c3-{tag}_{variant}-s{sd}"
        if (sysn, tp) not in idx:
            return {"problems": [f"missing {sysn}"]}
        s1.append(read_predictions(idx[(sysn, tp)])[1])
        s1c.append(read_predictions(idx[(sysn, cp)])[1] if (sysn, cp) in idx else None)
    res = {"variant": variant, "primary": variant == "no-nemotron", "heldout": sorted(held), "threshold": t,
           "s1v2": [typed_prf(test, p, held, t) for p in s1],
           "s1v2_gold_typing": [gold_typing_accuracy(test, p, held) for p in s1],
           "s1v2_tuned": [typed_prf(test, p, held, tune_typed_threshold(calib, pc, seen)) if pc else None
                          for p, pc in zip(s1, s1c)],
           "caveat": ("all-sources was trained on Nemotron train: same generator and templates, and held-out spans "
                      "supervised under coarse v1 types; only the label names are unseen") if variant == "all-sources" else
                     "no-nemotron never saw Nemotron text or labels; its v1 encoder saw other sources' spans only"}
    comps = {}
    if gliner_calib is None:
        gliner_calib = idx.get(("gliner25_base_zeroshot_c3", cp))
    for name, path in (("gliner25_base_zeroshot", gliner_path), ("gliner2_pii", gliner2pii_path)):
        if path is None:
            continue
        _, gp = read_predictions(Path(path))
        res[name] = typed_prf(test, gp, held, t)
        res[name + "_gold_typing"] = gold_typing_accuracy(test, gp, held)
        if name == "gliner25_base_zeroshot":            # the fair zero-shot comparator
            if gliner_calib is not None:
                res[name + "_tuned"] = typed_prf(test, gp, held, tune_typed_threshold(calib, read_predictions(Path(gliner_calib))[1], seen))
            for avg in ("micro", "macro"):
                comps[f"{name}|{avg}"] = _f1_bootstrap(test, s1, gp, sorted(held), t, avg, n_boot, seed)
    hh = holm({k: v["p_value"] for k, v in comps.items()}) if comps else {}
    res.update({"comparisons": comps, "holm": hh,
                "c3_holds": (all(v["diff"] > 0 and hh[k]["reject"] for k, v in comps.items()) if comps else None)
                            if variant == "no-nemotron" else None,
                "headline_state": hs,
                "gliner2_pii_note": "GLiNER2-PII may have been trained on these labels; reported, not a fair zero-shot comparator"})
    ledger.append({"kind": "c3" if variant == "no-nemotron" else "c3_secondary", **res},
                  headline=bool(comps) and variant == "no-nemotron")
    return res


def _counts_by_cluster(docs, preds, labels: list[str], t):
    """cluster -> (|labels|, 3) counts of (tp, fp, fn)."""
    li = {l: i for i, l in enumerate(labels)}
    by = {}
    for d in docs:
        gold = {(s.start, s.end, s.label_raw.lower()) for s in d.spans if s.label_raw.lower() in li}
        pr = greedy_flat([p for p in preds.get(d.doc_id, []) if p.score >= t])
        pred = {(p.start, p.end, p.label_raw.lower()) for p in pr if p.label_raw.lower() in li}
        c = by.setdefault(d.cluster_id or d.doc_id, np.zeros((len(labels), 3)))
        for x in pred & gold:
            c[li[x[2]], 0] += 1
        for x in pred - gold:
            c[li[x[2]], 1] += 1
        for x in gold - pred:
            c[li[x[2]], 2] += 1
    return by


def _f1_bootstrap(docs, s1_seeds, base, labels: list[str], t, avg: str, n_boot, seed):
    """Paired cluster bootstrap of F1 (mean over S1 seeds) minus baseline F1; micro or macro
    (macro over labels with gold support in the resample)."""
    cl = sorted({d.cluster_id or d.doc_id for d in docs})
    nl = len(labels)
    def mat(p):
        by = _counts_by_cluster(docs, p, labels, t)
        return np.stack([by[c].reshape(-1) for c in cl])          # (C, nl*3)
    dummy = np.zeros((len(cl), 1))
    a = [(mat(p), dummy) for p in s1_seeds]
    b = [(mat(base), dummy)]
    def f1(hp, hn):
        x = hp.reshape(hp.shape[:-1] + (nl, 3))
        tp, fp, fn = x[..., 0], x[..., 1], x[..., 2]
        if avg == "micro":
            T, P, N = tp.sum(-1), fp.sum(-1), fn.sum(-1)
            return 2 * T / np.maximum(2 * T + P + N, 1)
        f = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        sup = (tp + fn) > 0
        return (f * sup).sum(-1) / np.maximum(sup.sum(-1), 1)
    return paired_bootstrap_multi(a, b, f1, n_boot, seed)


# ------------------------------------------------------------------ sweeps

def sweep_label_set_size(store: Path, head_dir: Path, docs_path: Path, dataset: str, sizes=(3, 5, 10, 20, 40, 60),
                         seed: int = 0) -> list[dict]:
    """Drift of P(sensitive) and pAUC when the benchmark L is padded with non-sensitive
    distractor labels (sampled from the native pool, excluding compatible ones) up to |L|."""
    from .infer import decide, to_spans
    from .features import load_store
    base = LB.load_benchmark_label_set(dataset)
    rng = np.random.default_rng(seed)
    docs = read_jsonl(docs_path)
    z = load_store(store)
    pool = [n for n in sorted(LB.NATIVE) if not any(LB.compatible(n, l.name) for l in base.labels)]
    ref = decide(store, head_dir, base, z=z)
    out = []
    for n in sizes:
        if n <= len(base.labels):
            L = base
        else:
            extra = rng.choice(pool, min(len(pool), n - len(base.labels)), replace=False)
            L = LB.LabelSet(f"{dataset}+{n}", base.labels + [LB.native_label(x, sensitive=False) for x in extra])
        D = decide(store, head_dir, L, z=z)
        preds = to_spans(D, L, "sweep")
        vs = views(docs, preds, dataset)
        hp = sum(v.hist()[0] for v in vs); hn = sum(v.hist()[1] for v in vs)
        leak, over = M.curve_from_hist(hp, hn)
        out.append({"size": len(L.labels), "pauc": float(M.pauc(leak, over)),
                    "mean_abs_drift": float(np.mean(np.abs(D.p_sens - ref.p_sens))),
                    "none_share": float((D.probs.argmax(1) == len(L.labels)).mean())})
    return out


def sweep_rephrasings(store: Path, head_dir: Path, docs_path: Path, dataset: str) -> list[dict]:
    """Same labels, three phrasings: names only, name + description, first paraphrase."""
    from .infer import decide, to_spans
    from .features import load_store
    base = LB.load_benchmark_label_set(dataset)
    docs = read_jsonl(docs_path)
    z = load_store(store)
    variants = {
        "name+description": base,
        "names_only": LB.LabelSet(base.name, base.labels, descriptions=False),
        "paraphrase": LB.LabelSet(base.name, [LB.Label(LB.paraphrases(l.name)[0].replace(" ", "_") if l.name in LB.NATIVE else l.name,
                                                       "", l.sensitive, l.resolved_canonical(), l.resolved_node())
                                              for l in base.labels], descriptions=False),
    }
    ref = decide(store, head_dir, base, z=z)
    out = []
    for name, L in variants.items():
        D = decide(store, head_dir, L, z=z)
        preds = to_spans(D, L, "sweep")
        vs = views(docs, preds, dataset)
        hp = sum(v.hist()[0] for v in vs); hn = sum(v.hist()[1] for v in vs)
        leak, over = M.curve_from_hist(hp, hn)
        out.append({"variant": name, "pauc": float(M.pauc(leak, over)),
                    "mean_abs_drift": float(np.mean(np.abs(D.p_sens - ref.p_sens)))})
    return out


def sweep_texts(dataset: str) -> list[str]:
    """Every label text the sweeps can ask for (embed these into the store first)."""
    base = LB.load_benchmark_label_set(dataset)
    t = set(base.texts()) | set(LB.LabelSet("x", base.labels, False).texts())
    t |= {LB.paraphrases(l.name)[0] for l in base.labels if l.name in LB.NATIVE}
    t |= {LB.native_label(n).text(True) for n in LB.NATIVE}
    return sorted(t)


def _flat_verified(meta: dict) -> bool:
    """A flat-CRF prediction counts only if its model file exists with the recorded hash (the
    v2 run had a stale no-nemotron flat file with no trained model behind it)."""
    d, sha = meta.get("model_dir"), meta.get("model_sha")
    if not d or not sha:
        return False
    m = Path(d) / "flat_manifest.json"
    return m.exists() and json.loads(m.read_text()).get("weights_sha256") == sha


def ablation_table() -> dict:
    """pAUC on each benchmark test split (audited), seed 1. Comparable groups (v2.1):
    * canonical labels (what every baseline received): headline, cheap, names_only and no_none
      (both ablations of the cheap path, so compare them with cheap), flat_canonical;
    * benchmark labels: headline_bench vs flat_bench;
    * v1 (9 fixed types, no labels at inference) for reference.
    Flat rows require a verified model hash; unverified files are reported, not scored."""
    idx = c0.prediction_index([V2_PRED, ledger.RESULTS / "predictions"])
    tag = headline_tag()
    out, unverified = {}, []
    for ds in C4_SETS:
        v = c0.variant_for(ds)
        test = c0.audited_test_docs(ds); keep = {d.doc_id for d in test}
        tp = str(bench.split_paths(ds)["test"].resolve())
        row = {}
        for name, sysn in (("headline", v2_system(v, 1, tag)), ("cheap", v2_system(v, 1, "cheap")),
                           ("names_only", f"s1v2ablnames_{v}-s1"), ("no_none", f"s1v2ablnonone_{v}-s1"),
                           ("flat_canonical", f"s1v2flatcanon_{v}-s1"),
                           ("headline_bench", bench_system(v, 1, tag)), ("flat_bench", f"s1v2flat_{v}-s1"),
                           ("v1", c0.s1_system(v, 1))):
            if (sysn, tp) not in idx:
                row[name] = None; continue
            meta, pr = read_predictions(idx[(sysn, tp)])
            if name.startswith("flat") and not _flat_verified(meta):
                unverified.append(f"{ds}/{sysn}: {idx[(sysn, tp)].name}")
                row[name] = None; continue
            vs = views(test, {k: x for k, x in pr.items() if k in keep}, ds)
            hp = sum(x.hist()[0] for x in vs); hn = sum(x.hist()[1] for x in vs)
            leak, over = M.curve_from_hist(hp, hn)
            row[name] = float(M.pauc(leak, over))
        out[ds] = row
    res = {"table": out, "unverified_flat_files": unverified}
    ledger.append({"kind": "v2_ablations", "tag": tag, **res})
    return res


def sweeps(work: Path, drive: Path) -> dict:
    """Robustness sweeps (prereg) on seed 1 stores: |L| from 3 to 60 and rephrasings."""
    tag = headline_tag()
    out = {}
    for ds in C4_SETS:
        v = c0.variant_for(ds)
        hits = [h for h in sorted((work / "stores").glob(f"{ds}-test-{tag}-{v}-s1-*")) if (h / "meta.json").exists()]
        if len(hits) != 1:
            out[ds] = {"problem": f"store not found ({len(hits)})"}; continue
        head = drive / "heads" / f"{tag}-{v}-s1"
        docs_path = bench.split_paths(ds)["test"]
        out[ds] = {"size": sweep_label_set_size(hits[0], head, docs_path, ds),
                   "rephrase": sweep_rephrasings(hits[0], head, docs_path, ds)}
    ledger.append({"kind": "v2_sweeps", "tag": tag, "sweeps": out})
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("c0prime")
    sub.add_parser("c4")
    c = sub.add_parser("c3"); c.add_argument("--gliner", type=Path); c.add_argument("--gliner2pii", type=Path)
    sub.add_parser("ablations")
    w = sub.add_parser("sweeps"); w.add_argument("--work", type=Path, required=True); w.add_argument("--drive", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.cmd == "c0prime":
        r = {"primary": c0prime(), "secondary_benchmark_labels": c0prime(secondary=True)}
    elif a.cmd == "c4":
        r = c4()
    elif a.cmd == "ablations":
        r = ablation_table()
    elif a.cmd == "sweeps":
        r = sweeps(a.work, a.drive)
    else:
        gl = a.gliner
        if gl is None:
            tp = str(bench.split_paths("nemotron")["test"].resolve())
            gl = c0.prediction_index([V2_PRED]).get(("gliner25_base_zeroshot_c3", tp))
        r = {"primary": c3(gl, a.gliner2pii), "secondary_all_sources": c3(gl, a.gliner2pii, variant="all-sources")}
    (ledger.RESULTS / f"{a.cmd}.json").write_text(json.dumps(r, indent=2, default=str))
    print(json.dumps({k: v for k, v in r.items() if k not in ("comparisons", "per_label", "descriptive")}, indent=2, default=str)[:4000])


if __name__ == "__main__":
    main(sys.argv[1:])
