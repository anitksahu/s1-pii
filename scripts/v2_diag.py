"""Phase 0 CPU diagnostics for the v2 run (read-only; writes results/v2_diag.json on Drive).

A. provenance of every flat-CRF prediction file (the phantom nemotron flat row)
B. C4: score saturation at 1.0 and threshold feasibility on the calibration splits
C. C3: held-out gold spans on Nemotron test: token length, candidate presence in the
   prediction files, typing accuracy given a match, confusions; S1 headline/cheap vs GLiNER2.5
D. gate records (level-1 recall, head val accuracy, held-out typing)
E. label space probe: paraphrase/description -> name top-1 retrieval for the v1 encoder,
   the stage A typing encoder, the head's label projection, and a sentence encoder
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

D = Path(os.environ.get("S1PII_DRIVE", "/content/drive/MyDrive/s1pii"))
V2 = D / "v2"
RES = Path(os.environ.get("S1PII_RESULTS", D / "results"))
OUT = {}


def section(name):
    print(f"\n===== {name}", flush=True)


def A_provenance():
    section("A flat provenance")
    rows = []
    for f in sorted((RES / "predictions_v2").glob("flat-*.jsonl")):
        with open(f) as fh:
            meta = json.loads(fh.readline()).get("_meta", {})
        rows.append({"file": f.name, "mtime_utc": time.strftime("%m-%d %H:%M", time.gmtime(f.stat().st_mtime)),
                     "system": meta.get("system"), "docs": Path(meta.get("docs_path", "")).name,
                     "keys": sorted(meta)[:14]})
    done = sorted(p.name for p in (V2 / "done").glob("flat*"))
    models = sorted(str(p.relative_to(V2)) for p in (V2 / "flat").rglob("*") if p.is_file()) if (V2 / "flat").exists() else []
    units = []
    gh = V2 / "gpu_hours.jsonl"
    if gh.exists():
        units = [json.loads(l) for l in gh.read_text().splitlines() if '"flat' in l]
    OUT["A"] = {"files": rows, "done_markers": done, "flat_model_files": models, "gpu_units": units}
    for r in rows:
        print(r["file"], r["mtime_utc"], r["system"], r["docs"])
    print("done:", done); print("models:", models); print("units:", [(u["unit"], u.get("ok")) for u in units])


def B_c4_saturation():
    section("B C4 saturation")
    from s1pii import bench, c0, ledger
    from s1pii.schema import read_jsonl
    from s1pii.ledger import read_predictions
    from s1pii.eval import metrics as M
    from s1pii.eval.evaluate import views, tune_threshold
    idx = c0.prediction_index([RES / "predictions_v2", RES / "predictions"])
    out = {}
    for ds in ["tab_direct", "spy_medical", "spy_legal", "pii_trace", "nemotron"]:
        v = c0.variant_for(ds)
        cp = bench.split_paths(ds)["calib"]
        docs = read_jsonl(cp)
        key = str(cp.resolve())
        systems = [f"s1v2-A_{v}-s1", f"s1v2-cheap_{v}-s1"] + sorted({s for (s, p) in idx if p == key and not s.startswith("s1v2")})
        out[ds] = {}
        for sysn in systems:
            if (sysn, key) not in idx:
                continue
            _, pr = read_predictions(idx[(sysn, key)])
            vs = views(docs, pr, ds)
            hp = sum(x.hist()[0] for x in vs); hn = sum(x.hist()[1] for x in vs)
            leak, over = M.curve_from_hist(hp, hn)
            t = tune_threshold(docs, pr, ds)
            r = {"nonpii_at_1.0": float(hn[1001] / max(hn.sum(), 1)), "pii_at_1.0": float(hp[1001] / max(hp.sum(), 1)),
                 "nonpii_scored": float(hn[1:].sum() / max(hn.sum(), 1)), "min_over_on_grid": float(over[:-1].min()),
                 "t": t, "infeasible": t == M.MASK_NOTHING}
            out[ds][sysn] = r
            print(ds, sysn, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items()})
    OUT["B"] = out


def C_heldout():
    section("C held-out spans (Nemotron test)")
    from s1pii import bench, c0
    from s1pii.schema import read_jsonl
    from s1pii.ledger import read_predictions
    from s1pii.v2 import labels as LB
    from transformers import AutoTokenizer
    held = set(LB.load_heldout()["labels"])
    tp = bench.split_paths("nemotron")["test"]
    docs = {d.doc_id: d for d in read_jsonl(tp)}
    idx = c0.prediction_index([RES / "predictions_v2", RES / "predictions"])
    key = str(tp.resolve())
    tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-large")
    gold = [(did, s) for did, d in docs.items() for s in d.spans if s.label_raw.lower() in held]
    lens = Counter()
    per_label_len = {}
    for did, s in gold:
        n = len(tok(docs[did].text[s.start:s.end], add_special_tokens=False)["input_ids"])
        b = "<=8" if n <= 8 else ("9-64" if n <= 64 else ">64")
        lens[b] += 1
        per_label_len.setdefault(s.label_raw.lower(), Counter())[b] += 1
    res = {"n_gold": len(gold), "token_len": dict(lens), "per_label_n": {k: sum(v.values()) for k, v in per_label_len.items()},
           "per_label_over64": {k: v[">64"] for k, v in per_label_len.items()}, "systems": {}}
    print("held-out:", sorted(held)); print("n gold", len(gold), dict(lens))
    for sysn in ["s1v2c3-A_no-nemotron-s1", "s1v2c3-cheap_no-nemotron-s1", "s1v2c3-B_no-nemotron-s1",
                 "s1v2c3-A_all-sources-s1", "gliner25_base_zeroshot_c3"]:
        if (sysn, key) not in idx:
            print("missing", sysn); continue
        _, pr = read_predictions(idx[(sysn, key)])
        per = {}
        conf = Counter()
        for did, s in gold:
            lab = s.label_raw.lower()
            c = per.setdefault(lab, Counter())
            c["n"] += 1
            exact = [p for p in pr.get(did, []) if p.start == s.start and p.end == s.end]
            if not exact:
                ov = [p for p in pr.get(did, []) if p.start < s.end and p.end > s.start]
                c["overlap_only" if ov else "absent"] += 1
                continue
            c["matched"] += 1
            best = max(exact, key=lambda p: p.score)
            c["correct"] += best.label_raw.lower() == lab
            c["correct_t05"] += best.label_raw.lower() == lab and best.score >= 0.5
            if best.label_raw.lower() != lab:
                conf[(lab, best.label_raw.lower())] += 1
        tot = sum(per.values(), Counter())
        r = {"matched_frac": tot["matched"] / tot["n"], "overlap_only_frac": tot["overlap_only"] / tot["n"],
             "acc_given_match": tot["correct"] / max(tot["matched"], 1),
             "acc_t05_given_match": tot["correct_t05"] / max(tot["matched"], 1),
             "per_label": {k: {"n": v["n"], "match": round(v["matched"] / v["n"], 3),
                               "acc|match": round(v["correct"] / max(v["matched"], 1), 3)} for k, v in sorted(per.items())},
             "top_confusions": [f"{a}->{b}:{n}" for (a, b), n in conf.most_common(12)]}
        res["systems"][sysn] = r
        print(sysn, {k: (round(x, 3) if isinstance(x, float) else x) for k, x in r.items() if k != "per_label"})
        print("  per_label", r["per_label"])
    OUT["C"] = res


def D_gates():
    section("D gates")
    out = {}
    for tag in ("cheap", "B", "A"):
        p = V2 / f"gates-{tag}.json"
        if not p.exists():
            continue
        g = json.loads(p.read_text())
        out[tag] = {k: g.get(k) for k in ("head_val_acc", "head_val_acc_min", "heldout_typing_mean", "level1_recall_mean",
                                          "A_fires", "B_fires")}
        out[tag]["heldout_matched"] = {k: v.get("n_matched") for k, v in (g.get("heldout_typing") or {}).items()}
        print(tag, out[tag])
    OUT["D"] = out


def _embed_hf(model, tok, texts, device="cpu"):
    import torch
    out = []
    with torch.no_grad():
        for s in range(0, len(texts), 64):
            enc = tok(texts[s:s + 64], padding=True, return_tensors="pt", return_special_tokens_mask=True,
                      truncation=True, max_length=64)
            sp = enc.pop("special_tokens_mask")
            h = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state.float()
            m = (enc["attention_mask"].bool() & ~sp.bool()).unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).numpy())
    return np.concatenate(out)


def _retrieval(names_vec, queries):
    """queries: list of (true index, vec). Top-1 and top-5 by cosine."""
    n = names_vec / np.linalg.norm(names_vec, axis=1, keepdims=True)
    t1 = t5 = 0
    for i, q in queries:
        s = n @ (q / np.linalg.norm(q))
        order = np.argsort(-s)
        t1 += order[0] == i; t5 += i in order[:5]
    return {"top1": t1 / len(queries), "top5": t5 / len(queries), "n": len(queries)}


def E_label_probe():
    section("E label space probe")
    import torch
    from s1pii.v2 import labels as LB
    from s1pii.v2.head import label_text_variants, load_head
    from s1pii.model.train import load_exported
    held = set(LB.load_heldout()["labels"])
    names = sorted({l.name for l in LB.c3_label_set().labels})
    var = {n: label_text_variants(n) for n in names}
    name_txt = [var[n]["name"][0] for n in names]
    q = [(i, t, n in held) for i, n in enumerate(names) for t in var[n]["para"] + var[n]["desc"] if t != name_txt[i]]
    qt = [t for _, t, _ in q]
    encs = {}
    model, tok, _ = load_exported(D / "models" / "no-nemotron-s1" / "final", "cpu")
    encs["v1_encoder"] = (model.encoder.eval(), tok)
    try:
        from s1pii.v2.finetune import load_typing_encoder
        te, _ = load_typing_encoder(V2 / "typing" / "no-nemotron-s1", "cpu")
        encs["A_typing_encoder"] = (te, tok)
    except Exception as e:
        print("no typing encoder:", e)
    res = {}
    for name, (enc, tk) in encs.items():
        nv = _embed_hf(enc, tk, name_txt); qv = _embed_hf(enc, tk, qt)
        res[name] = {"all": _retrieval(nv, [(i, v) for (i, _, _), v in zip(q, qv)]),
                     "heldout": _retrieval(nv, [(i, v) for (i, _, h), v in zip(q, qv) if h])}
        hd = V2 / "heads" / "cheap-no-nemotron-s1" if name == "v1_encoder" else V2 / "typing" / "no-nemotron-s1" / "head"
        if (hd / "head_manifest.json").exists():
            head, _ = load_head(hd, "cpu")
            with torch.no_grad():
                pn = head.label_vec(torch.as_tensor(nv)).numpy(); pq = head.label_vec(torch.as_tensor(qv)).numpy()
            res[f"{name}+head"] = {"all": _retrieval(pn, [(i, v) for (i, _, _), v in zip(q, pq)]),
                                   "heldout": _retrieval(pn, [(i, v) for (i, _, h), v in zip(q, pq) if h])}
        else:
            print("no head at", hd)
    try:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
        nv = st.encode(name_txt); qv = st.encode(qt)
        res["bge_small"] = {"all": _retrieval(nv, [(i, v) for (i, _, _), v in zip(q, qv)]),
                            "heldout": _retrieval(nv, [(i, v) for (i, _, h), v in zip(q, qv) if h])}
    except Exception as e:
        print("sentence encoder failed:", e)
    for k, v in res.items():
        print(k, v)
    OUT["E"] = {"n_labels": len(names), "n_queries": len(q), "results": res}


if __name__ == "__main__":
    import sys
    steps = sys.argv[1:] or ["A", "B", "C", "D", "E"]
    fns = {"A": A_provenance, "B": B_c4_saturation, "C": C_heldout, "D": D_gates, "E": E_label_probe}
    for s in steps:
        t0 = time.time()
        try:
            fns[s]()
        except Exception as e:
            import traceback
            traceback.print_exc()
            OUT[s] = {"error": repr(e)}
        print(f"[{s} {time.time() - t0:.0f}s]", flush=True)
    p = RES / "v2_diag.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    old.update(OUT)
    p.write_text(json.dumps(old, indent=2, default=str))
    print("wrote", p)
