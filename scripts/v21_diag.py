"""v2.1 failure analysis (exploratory, CPU, calibration data only; fixed in advance as unable to
reopen C3): why does the flat CRF propose held-out spans but almost never type them?

    python scripts/v21_diag.py run      # writes $D/v21/gonogo/diag.json

Inputs (all from the go/no-go): the eval sample eval_docs.jsonl (190 Nemotron calibration docs),
the frozen v1 encoder (no-nemotron-s1), models M0 (v2 flat), M1 (v1-encoder labels) and M2 (bge
labels) x seeds 1, 2. Gold spans: the 10 held-out labels and the seen labels (model vocabulary).

A. Label ranking at forced gold boundaries. Transitions in FlatCRF are shared across labels at the
   level of tag kinds, so for a fixed segment with O outside it, the label-dependent part of the
   path score is the emission sum (S_k, or B_k + I_k... + E_k). We rank the 55 C3 labels by it:
   gold top-1 / top-5 / median rank, top confusion targets.
B. Restricted label sets: held-out spans scored among the 10 held-out labels only, and among
   {gold + 4 random seen labels} (fixed seed).
C. Per-label prior correction: a bias per label fitted by cross-entropy on one half of the docs
   (split by doc-id hash), evaluated on the other half, both directions.
D. Projection collapse: cosine of each held-out label to its nearest seen label, in raw label-
   encoder space vs after the model's label map; paraphrase/description -> name top-1 after the map.
E. Separability upper bound: a supervised logistic probe on the frozen v1 span features
   [h_i; h_j; mean] over the held-out types, 5-fold cross-validation grouped by doc.
F. Hybrid upper bound: v2 headline spans (s1v2c3-A_no-nemotron-s1, calibration) typed with the
   label of the most-overlapping GLiNER2.5 zero-shot span, vs GLiNER2.5 alone, on the same docs.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

D = Path(os.environ.get("S1PII_DRIVE", "/content/drive/MyDrive/s1pii"))
G = D / "v21" / "gonogo"
VARIANT, SEEDS, WIN = "no-nemotron", (1, 2), 512


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def models():
    out = {"M0_v2flat_3000": D / "v2" / "flat" / f"{VARIANT}-s1"}
    for s in SEEDS:
        out[f"M1_v1labels-s{s}"] = G / f"M1_v1labels-s{s}"
        out[f"M2_bgelabels-s{s}"] = G / f"M2_bgelabels-s{s}"
    return out


def half(doc_id: str) -> int:
    return int(hashlib.sha256(doc_id.encode()).hexdigest()[:8], 16) % 2


def span_scores(em: np.ndarray, i: int, j: int, nt: int) -> np.ndarray:
    """Label-dependent path score of segment [i, j] (token indices, inclusive) for every label."""
    from s1pii.model.crf import tag
    out = np.zeros(nt)
    for k in range(nt):
        if i == j:
            out[k] = em[i, tag("S", k)]
        else:
            out[k] = em[i, tag("B", k)] + em[i + 1:j, tag("I", k)].sum() + em[j, tag("E", k)]
    return out


def run():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from s1pii.schema import read_jsonl
    from s1pii.v2 import labels as LB
    from s1pii.v2.flat import FlatModel, LabelEncoder
    from s1pii.v2.head import label_text_variants
    from s1pii.model.train import load_exported
    from s1pii.model.encode import tokenize_doc, special_ids
    torch.set_grad_enabled(False)
    for v in ("S1PII_RESULTS", "S1PII_DATA"):
        if not os.environ.get(v):
            raise RuntimeError(f"{v} not set (run the setup cell first)")
    held = {x.lower() for x in LB.load_heldout()["labels"]}
    docs = read_jsonl(G / "eval_docs.jsonl")
    L = LB.c3_label_set(True)
    names = [l.name for l in L.labels]
    nt = len(names)
    v1, tok, man = load_exported(D / "models" / f"{VARIANT}-s1" / "final", "cpu")
    enc = v1.encoder.eval()
    pre, suf = special_ids(tok)
    size = WIN - len(pre) - len(suf)
    M = {}
    for name, fd in models().items():
        fman = json.loads((fd / "flat_manifest.json").read_text())
        if fman["v1"] != man["weights_sha256"]:
            raise RuntimeError(f"{name}: different v1 encoder")
        lenc = LabelEncoder.from_spec(fman.get("label_encoder"), enc, tok, "cpu")
        fm = FlatModel(enc.config.hidden_size, label_dim=lenc.dim, label_linear=fman.get("label_linear", False))
        fm.load_state_dict(torch.load(fd / "flat.pt", map_location="cpu")); fm.eval()
        M[name] = {"fm": fm, "lenc": lenc, "lv": lenc(L.texts()), "vocab": set(fman["vocab"])}
    hidx = [names.index(x) for x in sorted(held)]           # fails fast if a held-out label is not in L
    # ---- per gold span: one encoder pass per window, scores from every model, probe features
    rows, feats = [], []
    t0 = time.time()
    for di, d in enumerate(docs):
        td = tokenize_doc(d, tok)
        st, en = td.offsets[:, 0], td.offsets[:, 1]
        cache = {}                                   # window start -> hidden states
        for s in d.spans:
            lab = (s.label_raw or "").lower()
            if lab not in names:
                continue
            hit = np.nonzero((st < s.end) & (en > s.start) & (en > st))[0]
            if len(hit) == 0 or hit[-1] - hit[0] + 1 >= size:
                continue
            i, j = int(hit[0]), int(hit[-1])
            a = max(0, min((i + j) // 2 - size // 2, len(td.ids) - size)); b = min(len(td.ids), a + size)
            if a not in cache:
                ids = torch.as_tensor([pre + td.ids[a:b] + suf])
                cache[a] = enc(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state.float()[0, len(pre):len(pre) + b - a]
            h = cache[a]
            ii, jj = i - a, j - a
            feats.append(torch.cat([h[ii], h[jj], h[ii:jj + 1].mean(0)]).numpy())
            r = {"doc": d.doc_id, "label": lab, "gold": names.index(lab), "held": lab in held,
                 "aligned": int(st[i]) == s.start and int(en[j]) == s.end, "scores": {}}
            for name, m in M.items():
                em = m["fm"].emissions(h[None], m["lv"])[0].numpy()
                r["scores"][name] = span_scores(em, ii, jj, nt)
            rows.append(r)
        if di % 20 == 0:
            log(f"doc {di}/{len(docs)} spans {len(rows)} {time.time() - t0:.0f}s")
    res = {"exploratory": True, "c3_reopenable": False, "data": "Nemotron calibration (go/no-go eval sample)",
           "n_spans": len(rows), "n_heldout": sum(r["held"] for r in rows), "models": {}}
    for name, m in M.items():
        seen_ok = lambda r: (not r["held"]) and r["label"] in m["vocab"]
        unseen = np.array([n not in m["vocab"] for n in names])
        out = {}
        for grp, sel in (("heldout", lambda r: r["held"]), ("seen", seen_ok)):
            g = [r for r in rows if sel(r)]
            ranks = np.array([int((r["scores"][name] > r["scores"][name][r["gold"]]).sum()) + 1 for r in g])
            conf = Counter(names[int(np.argmax(r["scores"][name]))] for r in g if int(np.argmax(r["scores"][name])) != r["gold"])
            o = {"n": len(g), "top1": float((ranks == 1).mean()), "top5": float((ranks <= 5).mean()),
                 "median_rank": float(np.median(ranks)), "chance_top1": 1 / nt, "top_confusions": conf.most_common(8)}
            if grp == "heldout":
                rng = np.random.default_rng(0)                  # same distractors for every model
                o["acc_among_heldout_only"] = float(np.mean([hidx[int(np.argmax(r["scores"][name][hidx]))] == r["gold"] for r in g]))
                o["chance_heldout_only"] = 1 / len(hidx)
                seen_pool = sorted(names.index(x) for x in m["vocab"] if x in names)
                acc5 = []
                for r in g:
                    cand = [r["gold"]] + list(rng.choice(seen_pool, 4, replace=False))
                    acc5.append(cand[int(np.argmax(r["scores"][name][cand]))] == r["gold"])
                o["acc_gold_plus_4_seen"] = float(np.mean(acc5)); o["chance_gold_plus_4"] = 0.2
                per = {}
                for r, rk in zip(g, ranks):
                    per.setdefault(r["label"], []).append(rk)
                o["per_label_median_rank"] = {k: float(np.median(v)) for k, v in sorted(per.items())}
            out[grp] = o
        # C. prior correction, fitted on one doc half and tested on the other (both directions):
        #    none; one offset on labels outside the model's training vocabulary (the prior test);
        #    per-label biases with L2 (ORACLE: uses held-out gold)
        acc = {v: {"heldout": [], "seen": []} for v in ("none", "unseen_offset", "per_label_oracle")}
        for fit in (0, 1):
            tr = [r for r in rows if half(r["doc"]) == fit and (r["held"] or seen_ok(r))]
            te = [r for r in rows if half(r["doc"]) != fit and (r["held"] or seen_ok(r))]
            if not tr or not te:
                continue
            X = torch.as_tensor(np.stack([r["scores"][name] for r in tr]), dtype=torch.float32)
            y = torch.as_tensor([r["gold"] for r in tr])
            U = torch.as_tensor(unseen, dtype=torch.float32)
            beta = torch.zeros((), requires_grad=True); bias = torch.zeros(nt, requires_grad=True)
            with torch.enable_grad():
                for params, f in (([beta], lambda: X + beta * U), ([bias], lambda: X + bias)):
                    opt = torch.optim.Adam(params, lr=0.1)
                    for _ in range(300):
                        opt.zero_grad()
                        loss = torch.nn.functional.cross_entropy(f(), y) + (1e-2 * (bias ** 2).sum() if params[0] is bias else 0)
                        loss.backward(); opt.step()
            adds = {"none": np.zeros(nt), "unseen_offset": float(beta) * unseen, "per_label_oracle": bias.detach().numpy()}
            for v, add in adds.items():
                for r in te:
                    acc[v]["heldout" if r["held"] else "seen"].append(int(np.argmax(r["scores"][name] + add)) == r["gold"])
        out["prior_correction_acc"] = {v: {k: float(np.mean(x)) if x else None for k, x in d_.items()} for v, d_ in acc.items()}
        # D. projection collapse: held-out -> nearest seen minus seen -> nearest other seen, per space
        seen_i = [names.index(x) for x in m["vocab"] if x in names]
        def gap(V):
            Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
            S = Vn @ Vn.T
            h_ = float(np.mean([S[h, seen_i].max() for h in hidx]))
            s_ = float(np.mean([np.max([S[a_, b_] for b_ in seen_i if b_ != a_]) for a_ in seen_i])) if len(seen_i) > 1 else None
            return {"heldout_nearest_seen": h_, "seen_nearest_other_seen": s_, "gap": None if s_ is None else h_ - s_}
        var = {n: label_text_variants(n) for n in names}
        qs = [(i, t) for i, n in enumerate(names) for t in var[n]["para"] + var[n]["desc"] if t != var[n]["name"][0]]
        qv = m["lenc"]([t for _, t in qs])
        nv = m["lenc"]([var[n]["name"][0] for n in names])
        def top1(Nv, Qv, sel):
            Nn = Nv / np.linalg.norm(Nv, axis=1, keepdims=True); Qn = Qv / np.linalg.norm(Qv, axis=1, keepdims=True)
            z = [int(np.argmax(Nn @ q)) == i for (i, _), q in zip(qs, Qn) if sel(i)]
            return float(np.mean(z)) if z else None
        isheld = lambda i: names[i] in held
        pn, pq = m["fm"].label(nv).numpy(), m["fm"].label(qv).numpy()
        out["collapse"] = {"raw": gap(m["lv"].numpy()), "projected": gap(m["fm"].label(m["lv"]).numpy()),
                           "paraphrase_top1": {"raw_heldout": top1(nv.numpy(), qv.numpy(), isheld),
                                               "raw_seen": top1(nv.numpy(), qv.numpy(), lambda i: not isheld(i)),
                                               "projected_heldout": top1(pn, pq, isheld),
                                               "projected_seen": top1(pn, pq, lambda i: not isheld(i))}}
        res["models"][name] = out
        log(name, json.dumps({g: {k: v for k, v in out[g].items() if k in ("top1", "top5", "median_rank",
                                                                           "acc_among_heldout_only", "acc_gold_plus_4_seen")}
                              for g in ("heldout", "seen")}), out["prior_correction_acc"], out["collapse"])
    (G / "diag.json").write_text(json.dumps(res, indent=2, default=str))
    # E. separability probe on frozen v1 span features, doc-grouped CV, balanced accuracy + doc bootstrap
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import balanced_accuracy_score, recall_score
    Xf = np.stack(feats) if feats else np.zeros((0, 1))
    for grp, sel in (("heldout", lambda r: r["held"]), ("seen", lambda r: not r["held"])):
        idx = [k for k, r in enumerate(rows) if sel(r)]
        y = np.array([rows[k]["label"] for k in idx]); groups = np.array([rows[k]["doc"] for k in idx])
        k = min(5, len(set(groups)))
        if k < 2 or len(set(y)) < 2:
            res[f"probe_{grp}"] = {"balanced_acc": None, "problem": "too few docs or classes"}; continue
        pred = np.array([None] * len(idx), dtype=object)
        for trn, tst in GroupKFold(k).split(Xf[idx], y, groups):
            if len(set(y[trn])) < 2:
                continue
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=0.1))
            clf.fit(Xf[idx][trn], y[trn]); pred[tst] = clf.predict(Xf[idx][tst])
        ok = np.array([p is not None for p in pred])
        yy, pp = y[ok], pred[ok].astype(str)
        rng = np.random.default_rng(0); udocs = sorted(set(groups[ok])); bys = {}
        for t_, gdoc in enumerate(groups[ok]):
            bys.setdefault(gdoc, []).append(t_)
        boots = []
        for _ in range(1000):
            sel_ = [t_ for di_ in rng.choice(len(udocs), len(udocs)) for t_ in bys[udocs[di_]]]
            if len(set(yy[sel_])) > 1:
                boots.append(balanced_accuracy_score(yy[sel_], pp[sel_]))
        labs_ = sorted(set(yy))
        res[f"probe_{grp}"] = {"balanced_acc": float(balanced_accuracy_score(yy, pp)),
                               "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))] if boots else None,
                               "per_class_recall": dict(zip(labs_, map(float, recall_score(yy, pp, labels=labs_, average=None)))),
                               "n_classes": len(labs_), "chance": 1 / len(labs_), "n": int(ok.sum())}
        log("probe", grp, {k_: v for k_, v in res[f"probe_{grp}"].items() if k_ != "per_class_recall"})
    (G / "diag.json").write_text(json.dumps(res, indent=2, default=str))
    try:
        res["hybrid"] = hybrid(docs, held)
    except Exception as e:                           # never lose the rest of the analysis
        res["hybrid"] = {"problem": repr(e)}
    (G / "diag.json").write_text(json.dumps(res, indent=2, default=str))
    log("wrote", G / "diag.json", res["hybrid"])


def hybrid(docs, held):
    """S1 v2 headline spans typed by the most-overlapping GLiNER2.5 span (no training)."""
    from s1pii import bench, c0
    from s1pii.ledger import read_predictions
    R = Path(os.environ.get("S1PII_RESULTS", D / "results"))
    idx = c0.prediction_index([R / "predictions_v2", R / "predictions"])
    cp = str(bench.split_paths("nemotron")["calib"].resolve())
    s1k, gk = ("s1v2c3-A_no-nemotron-s1", cp), ("gliner25_base_zeroshot_c3", cp)
    if s1k not in idx or gk not in idx:
        return {"problem": f"missing calibration predictions: {[k[0] for k in (s1k, gk) if k not in idx]}"}
    _, s1 = read_predictions(idx[s1k]); _, gl = read_predictions(idx[gk])
    low = lambda p: (p.label_raw or "").lower()
    c = Counter()
    for d in docs:
        for s in d.spans:
            lab = (s.label_raw or "").lower()
            if lab not in held:
                continue
            c["n"] += 1
            gx = [p for p in gl.get(d.doc_id, []) if p.start == s.start and p.end == s.end]
            if gx:
                c["gliner_match"] += 1
                c["gliner_correct"] += low(max(gx, key=lambda p: p.score)) == lab
            sx = [p for p in s1.get(d.doc_id, []) if p.start == s.start and p.end == s.end]
            if sx:
                c["s1_match"] += 1
                ov = [(min(p.end, s.end) - max(p.start, s.start), p) for p in gl.get(d.doc_id, [])
                      if p.start < s.end and p.end > s.start]
                if ov:
                    c["hybrid_typed"] += 1
                    c["hybrid_correct"] += low(max(ov, key=lambda x: (x[0], x[1].score))[1]) == lab
    n = max(c["n"], 1)
    return {"n": c["n"], "gliner_match": c["gliner_match"] / n, "gliner_acc": c["gliner_correct"] / n,
            "s1_match": c["s1_match"] / n, "hybrid_acc": c["hybrid_correct"] / n,
            "hybrid_typed_frac": c["hybrid_typed"] / n,
            "note": "S1 = v2 headline (stage A) typed C3 spans after the 0.01 floor; flat-model spans were not saved"}


if __name__ == "__main__":
    {"run": run}[sys.argv[1]]()
