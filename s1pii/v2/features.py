"""v2 feature extraction from a trained v1 model (frozen), the only GPU-heavy v2 step.

For every document: v1 token windows (50% overlap) -> encoder hidden states H and CRF
lattices -> level-1 candidates. P_b(span) is the exact probability that the segment
[i, j] is an entity of *any* type: the sum over the 9 v1 types of the exact segment
marginals (the events are disjoint, so the sum is exact). Per-type v1 marginals are kept as
features. Candidates: P_b >= floor, at most ``top_k`` per window (drops are counted).
Validator hits are added as candidates with P_b = 1 (as in v1).

Span representation: [h_i ; h_j ; mean(h_i..h_j)] in float16. Extra spans (gold spans and
boundary-shifted gold spans for training) get the same representation; their P_b is the
candidate value if the span is a candidate, else 0.

Output: a feature store directory with one ``.npz`` per shard plus ``meta.json``; label
texts are embedded with the same frozen encoder (mean over real tokens) into
``labels.npz``. Store identity = (v1 weights sha, docs hash, FEATURE_VERSION, config).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch

from ..schema import Doc, read_jsonl, CANONICAL_TYPES
from ..model.encode import tokenize_doc, token_windows, special_ids, Example
from ..model.s1 import collate
from ..model.crf import span_logprobs
from ..model.postprocess import validator_spans
from ..ledger import dataset_hash

FEATURE_VERSION = "v2-features-0.3"
# validator hits are near-certain but not 1.0: a hard 1.0 ties at the top score bin and made
# the C4 operating point infeasible in the v2 run (v2.1)
VALIDATOR_PB = 0.999


class DiskLow(RuntimeError):
    pass
KIND_CAND, KIND_VALIDATOR, KIND_EXTRA = 0, 1, 2


@dataclass
class FeatureConfig:
    max_len: int = 1024
    floor: float = 0.01            # level-1 candidate floor on P_b
    type_floor: float = 0.001      # per-type floor before summing (<= floor / 9: no loss at the P_b floor)
    top_k: int = 512               # candidates per window (a drop fails a headline run)
    max_span_tokens: int = 64
    validators: bool = True
    batch_size: int = 32           # windows per encoder / lattice batch (across documents)
    store_min_pb: float = 0.0      # training stores keep only candidates with P_b >= this (0.05)


def _cum(h: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.zeros(1, h.shape[1], dtype=h.dtype, device=h.device), torch.cumsum(h, 0)], 0)


def span_reps(h: torch.Tensor, ij: np.ndarray) -> np.ndarray:
    """h (n, H) float32 on device; ij (m, 2) inclusive token indices -> (m, 3H) float16."""
    if len(ij) == 0:
        return np.zeros((0, 3 * h.shape[1]), dtype=np.float16)
    c = _cum(h)
    i = torch.as_tensor(ij[:, 0], device=h.device); j = torch.as_tensor(ij[:, 1], device=h.device)
    mean = (c[j + 1] - c[i]) / (j - i + 1).unsqueeze(1).to(h.dtype)
    return torch.cat([h[i], h[j], mean], 1).half().cpu().numpy()


class Extractor:
    def __init__(self, model_dir: Path, cfg: FeatureConfig | None = None, device: str | None = None,
                 typing_encoder: Path | None = None):
        """``model_dir``: level-1 model (v1 export, or a v2 level-1 export with nt = 1).
        ``typing_encoder``: optional fine-tuned typing encoder (``finetune.py`` export); span
        and label representations then come from it, P_b still from ``model_dir``."""
        from ..model.train import load_exported
        self.cfg = cfg or FeatureConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tok, self.manifest = load_exported(model_dir)
        self.model.to(self.device).eval()
        self.typer, self.typer_manifest = None, None
        if typing_encoder is not None:
            from .finetune import load_typing_encoder
            self.typer, self.typer_manifest = load_typing_encoder(typing_encoder, self.device)
        import copy
        self.crf = copy.deepcopy(self.model.crf).cpu().double().eval()
        self.pre, self.suf = special_ids(self.tok)
        self.size = self.cfg.max_len - len(self.pre) - len(self.suf)
        self.stride = max(1, self.size // 2)
        self.hidden_size = self.model.encoder.config.hidden_size
        self.report = {"windows": 0, "candidates": 0, "dropped_topk": 0, "validator": 0, "extra": 0,
                       "extra_unplaced": 0}

    def identity(self) -> dict:
        return {"feature_version": FEATURE_VERSION, "v1_weights": self.manifest["weights_sha256"],
                "v1_variant": self.manifest["config"]["variant"], "v1_seed": self.manifest["config"]["seed"],
                "level1_types": self.crf.nt,
                "typing_encoder": (self.typer_manifest or {}).get("weights_sha256"),
                "config": asdict(self.cfg)}

    # -------------------------------------------------------------- encoder
    @torch.no_grad()
    def _hidden_and_emissions(self, exs: list[Example]):
        out = []
        pad = self.tok.pad_token_id or 0
        for s in range(0, len(exs), self.cfg.batch_size):
            chunk = exs[s:s + self.cfg.batch_size]
            b = collate(chunk, pad, with_targets=False)
            ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if self.device == "cuda"
                   else torch.autocast("cpu", enabled=False))
            ids, att = b["input_ids"].to(self.device), b["attention_mask"].to(self.device)
            with ctx:
                h = self.model.hidden(ids, att)
                ht = self.typer(input_ids=ids, attention_mask=att).last_hidden_state if self.typer is not None else None
            h = h.float()
            em = self.model.emissions_from_hidden(h)
            hr = h if ht is None else ht.float()
            for k, e in enumerate(chunk):
                sl = slice(e.n_prefix, e.n_prefix + e.n_tok)
                out.append((hr[k, sl], em[k, sl].double().cpu().numpy()))
        return out

    @torch.no_grad()
    def _lattices(self, ems: list[np.ndarray]):
        """Batched forward-backward on the model device, float64 where supported (as v1)."""
        dev = self.device
        dt = torch.float64 if dev == "cpu" or torch.cuda.get_device_capability()[0] >= 8 else torch.float32
        crf = self.crf.to(dev).to(dt)
        out = []
        for i in range(0, len(ems), self.cfg.batch_size):
            chunk = ems[i:i + self.cfg.batch_size]
            L = max(len(e) for e in chunk)
            em = torch.zeros(len(chunk), L, chunk[0].shape[1], dtype=dt, device=dev)
            mask = torch.zeros(len(chunk), L, dtype=torch.bool, device=dev)
            for k, e in enumerate(chunk):
                em[k, :len(e)] = torch.as_tensor(e, dtype=dt, device=dev); mask[k, :len(e)] = True
            alpha, logz = crf._forward(em, mask)
            beta = crf._backward(em, mask)
            for k, e in enumerate(chunk):
                out.append((alpha[k, :len(e)].double().cpu().numpy(), beta[k, :len(e)].double().cpu().numpy(),
                            float(logz[k])))
        self.crf.cpu().double()
        return out

    # -------------------------------------------------------------- one document
    def doc_features(self, d: Doc, extra: list[tuple[int, int, str]] | None = None) -> dict:
        return self.docs_features([d], {d.doc_id: extra} if extra else None)[0]

    def _place(self, td, wins, spans):
        """Assign char spans to the window where they are most central -> (placed, missed)."""
        cfg = self.cfg
        starts, ends = td.offsets[:, 0], td.offsets[:, 1]
        placed, missed = [], 0
        for cs, ce, lab in spans:
            hit = np.nonzero((starts < ce) & (ends > cs) & (ends > starts))[0]
            if len(hit) == 0:
                missed += 1; continue
            ti, tj = int(hit[0]), int(hit[-1])
            if tj - ti + 1 > cfg.max_span_tokens:
                missed += 1; continue
            ok = [(abs((ti + tj) / 2 - (a + b) / 2), wi) for wi, (a, b) in enumerate(wins)
                  if a <= ti and tj < b and not (ti == a and a > 0) and not (tj == b - 1 and b < len(td.ids))]
            if not ok:
                ok = [(0, wi) for wi, (a, b) in enumerate(wins) if a <= ti and tj < b]
            if not ok:
                missed += 1; continue
            wi = min(ok)[1]
            placed.append((wi, ti - wins[wi][0], tj - wins[wi][0], cs, ce, lab))
        return placed, missed

    def docs_features(self, docs: list[Doc], extras: dict | None = None) -> list[dict]:
        """Features for many docs with windows batched across documents (length-sorted batches
        of ``batch_size`` windows through the encoder and the forward-backward lattices)."""
        cfg, nt = self.cfg, self.crf.nt
        prep = []
        jobs = []                                   # (doc index, window index, Example)
        for di, d in enumerate(docs):
            td = tokenize_doc(d, self.tok)
            wins = token_windows(len(td.ids), self.size, self.stride) if td.ids else []
            val = [(s.start, s.end, "") for s in validator_spans(d)] if cfg.validators else []
            pv, _ = self._place(td, wins, val) if val and wins else ([], 0)
            px, mx = self._place(td, wins, (extras or {}).get(d.doc_id) or []) if wins else ([], 0)
            prep.append({"td": td, "wins": wins, "val": pv, "extra": px, "missed": mx, "cands": [], "vrep": {}, "xrep": {}})
            for wi, (a, b) in enumerate(wins):
                jobs.append((di, wi, Example(d.doc_id, np.asarray(self.pre + td.ids[a:b] + self.suf, dtype=np.int32),
                                             len(self.pre), b - a, np.zeros(b - a, dtype=np.int64), a)))
            self.report["windows"] += len(wins)
        jobs.sort(key=lambda j: len(j[2].input_ids))
        bs = cfg.batch_size
        for s0 in range(0, len(jobs), bs):
            chunk = jobs[s0:s0 + bs]
            hs = self._hidden_and_emissions([j[2] for j in chunk])
            lats = self._lattices([em for _, em in hs])
            for (di, wi, ex), (h, em), (alpha, beta, logz) in zip(chunk, hs, lats):
                P = prep[di]; td = P["td"]; a, b = P["wins"][wi]; n = b - a
                raw = span_logprobs(self.crf, em, alpha, beta, logz, n, floor=cfg.type_floor, max_len=cfg.max_span_tokens)
                agg: dict[tuple[int, int], np.ndarray] = {}
                for i, j, t, p in raw:
                    if (i == 0 and a > 0) or (j == n - 1 and b < len(td.ids)):
                        continue
                    agg.setdefault((i, j), np.zeros(nt))[t] += p
                items = [(ij, v) for ij, v in agg.items() if v.sum() >= max(cfg.floor, cfg.store_min_pb)]
                items.sort(key=lambda x: -x[1].sum())
                if len(items) > cfg.top_k:
                    self.report["dropped_topk"] += len(items) - cfg.top_k
                    items = items[:cfg.top_k]
                want = [ij for ij, _ in items]
                vx = [(x[1], x[2]) for x in P["val"] if x[0] == wi]
                xx = [(x[1], x[2]) for x in P["extra"] if x[0] == wi]
                allij = want + vx + xx
                if not allij:
                    continue
                reps = span_reps(h, np.array(allij, dtype=np.int64))
                for (ij, v), r in zip(items, reps[:len(want)]):
                    P["cands"].append((ij, v, r, a))
                for k, x in enumerate([x for x in P["val"] if x[0] == wi]):
                    P["vrep"][(x[3], x[4])] = reps[len(want) + k]
                for k, x in enumerate([x for x in P["extra"] if x[0] == wi]):
                    P["xrep"][(x[3], x[4], x[5])] = reps[len(want) + len(vx) + k]
        out = []
        for d, P in zip(docs, prep):
            td = P["td"]
            best: dict[tuple[int, int], dict] = {}
            for (i, j), v, r, a in P["cands"]:
                cs, ce = int(td.offsets[a + i][0]), int(td.offsets[a + j][1])
                if ce <= cs:
                    continue
                pb = float(min(1.0, v.sum()))
                if (cs, ce) not in best or pb > best[(cs, ce)]["pb"]:
                    best[(cs, ce)] = {"pb": pb, "v1": v.astype(np.float32), "rep": r, "kind": KIND_CAND, "label": ""}
            for key, r in P["vrep"].items():
                self.report["validator"] += 1
                if key in best:
                    best[key]["pb"] = VALIDATOR_PB; best[key]["kind"] = KIND_VALIDATOR
                else:
                    best[key] = {"pb": VALIDATOR_PB, "v1": np.zeros(nt, np.float32), "rep": r, "kind": KIND_VALIDATOR, "label": ""}
            rows = list(best.items())
            self.report["candidates"] += len(rows)
            for (cs, ce, lab), r in P["xrep"].items():
                c = best.get((cs, ce))
                rows.append(((cs, ce), {"pb": c["pb"] if c else 0.0, "v1": c["v1"] if c else np.zeros(nt, np.float32),
                                        "rep": r, "kind": KIND_EXTRA, "label": lab}))
            self.report["extra"] += len(P["xrep"]); self.report["extra_unplaced"] += P["missed"]
            H3 = 3 * self.hidden_size
            out.append({"doc_id": d.doc_id,
                        "start": np.array([k[0] for k, _ in rows], dtype=np.int32),
                        "end": np.array([k[1] for k, _ in rows], dtype=np.int32),
                        "pb": np.array([r["pb"] for _, r in rows], dtype=np.float32),
                        "v1": np.stack([r["v1"] for _, r in rows]).astype(np.float32) if rows else np.zeros((0, nt), np.float32),
                        "rep": np.stack([r["rep"] for _, r in rows]) if rows else np.zeros((0, H3), np.float16),
                        "kind": np.array([r["kind"] for _, r in rows], dtype=np.int8),
                        "label": [r["label"] for _, r in rows]})
        return out

    # -------------------------------------------------------------- label texts
    @torch.no_grad()
    def embed_texts(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        out = []
        for s in range(0, len(texts), batch_size):
            enc = self.tok(texts[s:s + batch_size], padding=True, return_tensors="pt",
                           return_special_tokens_mask=True, truncation=True, max_length=64)
            sp = enc.pop("special_tokens_mask")
            ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if self.device == "cuda"
                   else torch.autocast("cpu", enabled=False))
            enc_model = self.typer if self.typer is not None else self.model.encoder
            with ctx:
                h = enc_model(input_ids=enc["input_ids"].to(self.device),
                              attention_mask=enc["attention_mask"].to(self.device)).last_hidden_state.float()
            m = (enc["attention_mask"].bool() & ~sp.bool()).to(self.device).unsqueeze(-1).float()
            out.append(((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy())
        return np.concatenate(out).astype(np.float32) if out else np.zeros((0, self.hidden_size), np.float32)


# ------------------------------------------------------------------ store

def _concat(docs_feats: list[dict]) -> dict:
    lens = [len(f["start"]) for f in docs_feats]
    return {"doc_ids": np.array([f["doc_id"] for f in docs_feats]),
            "offsets": np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
            "start": np.concatenate([f["start"] for f in docs_feats]) if docs_feats else np.zeros(0, np.int32),
            "end": np.concatenate([f["end"] for f in docs_feats]) if docs_feats else np.zeros(0, np.int32),
            "pb": np.concatenate([f["pb"] for f in docs_feats]) if docs_feats else np.zeros(0, np.float32),
            "v1": np.concatenate([f["v1"] for f in docs_feats]) if docs_feats else np.zeros((0, 1), np.float32),
            "rep": np.concatenate([f["rep"] for f in docs_feats]) if docs_feats else np.zeros((0, 0), np.float16),
            "kind": np.concatenate([f["kind"] for f in docs_feats]) if docs_feats else np.zeros(0, np.int8),
            "label": np.array([l for f in docs_feats for l in f["label"]], dtype=object)}


def store_key(identity: dict, docs_hash: str, extra_hash: str = "") -> str:
    return hashlib.sha256(json.dumps([identity, docs_hash, extra_hash], sort_keys=True).encode()).hexdigest()[:16]


def extract(model_dir: Path, docs: list[Doc], out_root: Path, *, name: str, extras: dict | None = None,
            shard_size: int = 500, cfg: FeatureConfig | None = None, label_texts: list[str] | None = None,
            extractor: Extractor | None = None, typing_encoder: Path | None = None, progress=None,
            min_free_gb: float = 0) -> Path:
    """Write (or resume) a feature store for ``docs``. ``extras`` maps doc_id -> list of
    (start, end, label) spans to represent (training). Returns the store directory."""
    ex = extractor or Extractor(model_dir, cfg, typing_encoder=typing_encoder)
    docs = sorted(docs, key=lambda d: d.doc_id)
    dh = dataset_hash(docs)
    eh = hashlib.sha256(json.dumps(sorted((k, sorted(v)) for k, v in (extras or {}).items())).encode()).hexdigest()[:16]
    key = store_key(ex.identity(), dh, eh)
    root = out_root / f"{name}-{key}"
    if (root / "meta.json").exists():
        return root
    root.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    nshards = (len(docs) + shard_size - 1) // shard_size
    todo = sum(1 for si in range(nshards) if not (root / f"shard-{si:05d}.npz").exists())
    computed = 0
    for si in range(nshards):
        p = root / f"shard-{si:05d}.npz"
        if p.exists():
            continue
        if min_free_gb and shutil.disk_usage(out_root).free < min_free_gb * 2**30:
            raise DiskLow(f"< {min_free_gb} GB free under {out_root} before shard {si} of {name}")
        feats = ex.docs_features(docs[si * shard_size:(si + 1) * shard_size], extras)
        tmp = root / f".tmp-shard-{si:05d}.npz"
        np.savez(tmp, **_concat(feats))
        os.replace(tmp, p)
        print(f"[features {name}] shard {si + 1}/{nshards} {ex.report} {time.time() - t0:.0f}s", flush=True)
        computed += 1
        if progress is not None:
            progress(computed, todo, time.time() - t0)
    if label_texts:
        write_label_vectors(root, ex, label_texts)
    meta = {"identity": ex.identity(), "docs_hash": dh, "extras_hash": eh, "n_docs": len(docs), "name": name,
            "report": ex.report, "hidden_size": ex.hidden_size, "seconds": round(time.time() - t0, 1),
            "n_shards": nshards}
    (root / "meta.json").write_text(json.dumps(meta, indent=2))
    return root


def write_label_vectors(root: Path, ex: Extractor, texts: list[str]) -> None:
    p = root / "labels.npz"
    have = {}
    if p.exists():
        z = np.load(p, allow_pickle=True)
        have = dict(zip(z["texts"].tolist(), z["vecs"]))
    new = sorted(set(texts) - set(have))
    if new:
        have.update(zip(new, ex.embed_texts(new)))
    ks = sorted(have)
    tmp = root / ".tmp-labels.npz"
    np.savez(tmp, texts=np.array(ks, dtype=object), vecs=np.stack([have[k] for k in ks]))
    os.replace(tmp, p)


def load_label_vectors(root: Path) -> dict[str, np.ndarray]:
    z = np.load(root / "labels.npz", allow_pickle=True)
    return dict(zip(z["texts"].tolist(), z["vecs"]))


def iter_shards(root: Path, fields: tuple[str, ...] | None = None):
    for p in sorted(root.glob("shard-*.npz")):
        z = np.load(p, allow_pickle=True)
        yield {k: z[k] for k in (fields or z.files)}


def load_store(root: Path, fields: tuple[str, ...] | None = None) -> dict:
    """Concatenate all shards; doc offsets are rebased."""
    parts = list(iter_shards(root, fields))
    if not parts:
        raise FileNotFoundError(f"no shards in {root}")
    out = {}
    for k in parts[0]:
        if k == "offsets":
            base, offs = 0, []
            for p in parts:
                offs.append(p["offsets"][:-1] + base); base += int(p["offsets"][-1])
            out[k] = np.concatenate(offs + [np.array([base])])
        else:
            out[k] = np.concatenate([p[k] for p in parts])
    return out


# ------------------------------------------------------------------ training extras

def training_extras(docs: list[Doc], seed: int, heldout_nodes: set[str]) -> dict[str, list]:
    """Gold spans (every raw label except the loader scaffold) plus one boundary-shifted copy
    of each gold span (label "__shift__", a NONE target). Held-out spans are kept with their
    label so the trainer can exclude them and every candidate overlapping them."""
    rng = random.Random(seed)
    out = {}
    for d in docs:
        rows = []
        for s in d.spans:
            if s.label_raw == "_scaffold" or not s.label_raw:
                continue
            rows.append((s.start, s.end, s.label_raw.lower()))
            txt = d.text
            opts = []
            if s.end - s.start > 1:
                opts += [(s.start + 1, s.end), (s.start, s.end - 1)]
            if s.start > 0:
                opts.append((s.start - 1, s.end))
            if s.end < len(txt):
                opts.append((s.start, s.end + 1))
            # shift to the next/previous word boundary so the shift changes tokens
            ws = txt.rfind(" ", 0, max(0, s.start - 1))
            if ws >= 0:
                opts.append((ws + 1, s.end))
            we = txt.find(" ", s.end + 1)
            if we > 0:
                opts.append((s.start, we))
            opts = [(a, b) for a, b in opts if b > a and txt[a:b].strip() and (a, b) != (s.start, s.end)]
            if opts:
                a, b = rng.choice(opts)
                rows.append((a, b, "__shift__"))
        out[d.doc_id] = rows
    return out


def select_training_docs(docs: list[Doc], *, per_label: int = 4000, background_frac: float = 0.05,
                         max_docs: int = 60000, seed: int = 0) -> list[Doc]:
    """Deterministic label-balanced subset: a doc is kept while any of its raw labels is under
    quota, plus a hash-fixed background fraction for hard negatives."""
    order = sorted(docs, key=lambda d: hashlib.sha256(f"{seed}:{d.doc_id}".encode()).hexdigest())
    counts: dict[str, int] = {}
    out = []
    for d in order:
        labs = [s.label_raw.lower() for s in d.spans if s.label_raw and s.label_raw != "_scaffold"]
        need = any(counts.get(l, 0) < per_label for l in labs)
        bg = int(hashlib.sha256(f"bg:{d.doc_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < background_frac
        if need or bg:
            out.append(d)
            for l in labs:
                counts[l] = counts.get(l, 0) + 1
        if len(out) >= max_docs:
            break
    return out


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--docs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--labels", nargs="*", default=[], help="label-set yaml files whose texts to embed")
    a = ap.parse_args(argv)
    from .labels import LabelSet
    texts = sorted({t for p in a.labels for ls in [LabelSet.from_yaml(Path(p))] for t in ls.texts()})
    print(extract(a.model, list(read_jsonl(a.docs)), a.out, name=a.name, label_texts=texts))


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
