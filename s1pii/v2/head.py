"""v2 level 2: typing a candidate span against a label set L supplied at inference.

    P(k | span, L) = softmax_k( [ scale * cos(S(span), U(l_k)) ]_{k in L} ; scale * cos(S(span), n) + b )

S is an MLP over [h_i; h_j; mean h; v1 type marginals; logit P_b], U an MLP over the label
text embedding (frozen v1 encoder, mean-pooled), n a learned NONE vector: "not an entity of
any class in L". Trained on cached features only (``features.py``), so it takes minutes.

Training targets (per batch a shared label set L_b is sampled):
* gold spans -> their raw label; labels compatible with the gold (ancestor, descendant or
  synonym in ``labels.TREE``) are masked out of L_b for that item, never used as negatives;
* NONE from (a) gold label dropped from L_b (with its compatible labels), (b) boundary-shifted
  gold spans, (c) hard negatives: candidates with P_b >= 0.05 overlapping no gold span;
* held-out labels (C3) never appear: their spans are removed, candidates overlapping them
  are removed from the hard negatives, and their texts never enter L_b.
Label texts vary per batch: name only, name + description, or a paraphrase.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from . import labels as LB
from .features import KIND_EXTRA, load_store, load_label_vectors

HEAD_VERSION = "v2-head-0.1"
NONE_TARGET = -1


@dataclass
class HeadConfig:
    dim: int = 512
    dropout: float = 0.1
    epochs: float = 4.0
    batch: int = 512
    lr: float = 1e-3
    weight_decay: float = 0.01
    min_labels: int = 8
    max_labels: int = 48
    p_gold_drop: float = 0.15
    p_name_only: float = 0.3
    p_paraphrase: float = 0.2
    hard_neg_min_pb: float = 0.05
    hard_neg_ratio: float = 0.5        # at most this many hard negatives per gold item
    shift_ratio: float = 0.5           # at most this many boundary shifts per gold item
    val_frac: float = 0.02
    seed: int = 1
    extra: dict = field(default_factory=dict)


def _mlp(i: int, d: int, p: float) -> nn.Module:
    return nn.Sequential(nn.Linear(i, d), nn.GELU(), nn.Dropout(p), nn.Linear(d, d))


class TypingHead(nn.Module):
    def __init__(self, hidden: int, n_v1: int = 9, cfg: HeadConfig | None = None):
        super().__init__()
        cfg = cfg or HeadConfig()
        self.hidden, self.n_v1 = hidden, n_v1
        self.span = _mlp(3 * hidden + n_v1 + 1, cfg.dim, cfg.dropout)
        self.label = _mlp(hidden, cfg.dim, cfg.dropout)
        self.none = nn.Parameter(torch.randn(cfg.dim) * 0.02)
        self.none_bias = nn.Parameter(torch.zeros(()))
        self.log_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def span_vec(self, rep: torch.Tensor, v1: torch.Tensor, pb: torch.Tensor) -> torch.Tensor:
        lp = torch.logit(pb.clamp(1e-4, 1 - 1e-4)).unsqueeze(1)
        return F.normalize(self.span(torch.cat([rep.float(), v1.float(), lp], 1)), dim=-1)

    def label_vec(self, lv: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.label(lv.float()), dim=-1)

    def logits(self, s: torch.Tensor, l: torch.Tensor) -> torch.Tensor:
        """(B, |L| + 1); the last column is NONE."""
        sc = self.log_scale.exp()
        none = (s @ F.normalize(self.none, dim=0)).unsqueeze(1) * sc + self.none_bias
        return torch.cat([sc * (s @ l.T), none], 1)


# ------------------------------------------------------------------ training data

@dataclass
class Items:
    rep: np.ndarray        # (N, 3H) float16
    v1: np.ndarray         # (N, 9)
    pb: np.ndarray         # (N,)
    y: np.ndarray          # (N,) index into ``labels`` or NONE_TARGET
    doc: np.ndarray        # (N,) doc index (for the validation split)
    kind: np.ndarray       # (N,) 0 gold, 1 shift, 2 hard negative
    labels: list[str]      # training label vocabulary (raw names, lowercased)
    start: np.ndarray | None = None   # char spans and doc ids (for end-to-end fine-tuning)
    end: np.ndarray | None = None
    doc_ids: np.ndarray | None = None


def build_items(store: Path, heldout_nodes: set[str], cfg: HeadConfig) -> tuple[Items, dict]:
    z = load_store(store)
    offs, starts, ends, kinds, labs, pb = z["offsets"], z["start"], z["end"], z["kind"], z["label"], z["pb"]
    rng = np.random.default_rng(cfg.seed)
    vocab = sorted({l for l in labs.tolist() if l and l != "__shift__" and l.lower() in LB.NATIVE
                    and LB.node_of(l) not in heldout_nodes})
    vix = {l: i for i, l in enumerate(vocab)}
    gold_idx, shift_idx, neg_idx, y_gold, stats = [], [], [], [], {"heldout_removed": 0, "unknown_label": 0,
                                                                     "shift_equal_gold": 0, "neg_overlap_removed": 0,
                                                                     "neg_teacher_zone_removed": 0}
    # teacher zones: spans of families the doc's source does not annotate (no NONE there)
    zf = Path(store) / "no_negative_zones.json"
    zones = json.loads(zf.read_text()) if zf.exists() else {}
    doc_ids = z["doc_ids"].tolist()
    for di in range(len(offs) - 1):
        a, b = int(offs[di]), int(offs[di + 1])
        zd = zones.get(doc_ids[di], [])
        gold = [(int(starts[r]), int(ends[r]), labs[r]) for r in range(a, b)
                if kinds[r] == KIND_EXTRA and labs[r] and labs[r] != "__shift__"]
        gset = {(s, e) for s, e, _ in gold}
        for r in range(a, b):
            if kinds[r] == KIND_EXTRA:
                l = labs[r]
                if l == "__shift__":
                    if (int(starts[r]), int(ends[r])) in gset:
                        stats["shift_equal_gold"] += 1; continue
                    # a shift overlapping a held-out span would teach NONE on held-out text
                    if any(LB.node_of(g) in heldout_nodes and min(ge, ends[r]) > max(gs, starts[r]) for gs, ge, g in gold):
                        stats["heldout_removed"] += 1; continue
                    shift_idx.append(r)
                elif l.lower() not in LB.NATIVE:
                    stats["unknown_label"] += 1
                elif LB.node_of(l) in heldout_nodes:
                    stats["heldout_removed"] += 1
                else:
                    gold_idx.append(r); y_gold.append(vix[l])
            elif pb[r] >= cfg.hard_neg_min_pb:
                s, e = int(starts[r]), int(ends[r])
                if any(min(ge, e) > max(gs, s) for gs, ge, _ in gold):
                    stats["neg_overlap_removed"] += 1; continue
                if any(min(ze, e) > max(zs, s) for zs, ze in zd):
                    stats["neg_teacher_zone_removed"] += 1; continue
                neg_idx.append(r)
    ng = len(gold_idx)
    shift_idx = rng.permutation(shift_idx)[:int(cfg.shift_ratio * ng)].tolist()
    neg_idx = rng.permutation(neg_idx)[:int(cfg.hard_neg_ratio * ng)].tolist()
    rows = np.array(gold_idx + shift_idx + neg_idx, dtype=np.int64)
    doc_of = np.searchsorted(offs, rows, side="right") - 1
    y = np.array(y_gold + [NONE_TARGET] * (len(shift_idx) + len(neg_idx)), dtype=np.int64)
    kind = np.array([0] * ng + [1] * len(shift_idx) + [2] * len(neg_idx), dtype=np.int8)
    stats.update({"gold": ng, "shift": len(shift_idx), "hard_neg": len(neg_idx), "labels": len(vocab),
                  "per_label": {l: int((y == i).sum()) for i, l in enumerate(vocab)}})
    return Items(z["rep"][rows], z["v1"][rows], pb[rows], y, doc_of, kind, vocab,
                 starts[rows], ends[rows], z["doc_ids"][doc_of]), stats


def label_text_variants(name: str) -> dict[str, list[str]]:
    lab = LB.native_label(name)
    return {"name": [lab.text(False)], "desc": [lab.text(True)], "para": list(LB.paraphrases(name))}


def all_training_texts(vocab: list[str]) -> list[str]:
    return sorted({t for n in vocab for v in label_text_variants(n).values() for t in v})


# ------------------------------------------------------------------ training

def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_head(store: Path, out: Path, heldout_nodes: set[str], cfg: HeadConfig | None = None,
               device: str | None = None, log=print) -> Path:
    cfg = cfg or HeadConfig()
    device = device or _device()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    items, stats = build_items(store, heldout_nodes, cfg)
    if len(items.y) == 0:
        raise RuntimeError("no training items")
    lvec = load_label_vectors(store)
    vocab = items.labels
    variants = [label_text_variants(n) for n in vocab]
    missing = [t for v in variants for ts in v.values() for t in ts if t not in lvec]
    if missing:
        raise KeyError(f"label vectors missing for {len(missing)} texts, e.g. {missing[:3]}")
    compat = np.array([[LB.compatible(a, b) for b in vocab] for a in vocab], dtype=bool)

    docs = np.unique(items.doc)
    val_docs = set(rng.choice(docs, max(1, int(cfg.val_frac * len(docs))), replace=False).tolist())
    is_val = np.array([d in val_docs for d in items.doc])
    tr = np.nonzero(~is_val)[0]; va = np.nonzero(is_val)[0]

    H = items.rep.shape[1] // 3
    head = TypingHead(H, items.v1.shape[1], cfg).to(device)
    rep = torch.as_tensor(items.rep, device=device)
    v1 = torch.as_tensor(items.v1, device=device); pb = torch.as_tensor(items.pb, device=device)
    y_all = torch.as_tensor(items.y, device=device)
    steps = max(1, int(cfg.epochs * math.ceil(len(tr) / cfg.batch)))
    opt = torch.optim.AdamW(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps, pct_start=0.05)
    compat_t = torch.as_tensor(compat, device=device)

    def label_matrix(ids: np.ndarray, mode: str | None = None) -> torch.Tensor:
        vecs = []
        for i in ids:
            v = variants[i]
            m = mode
            if m is None:
                u = rng.random()
                m = "name" if u < cfg.p_name_only else ("para" if u < cfg.p_name_only + cfg.p_paraphrase else "desc")
            vecs.append(lvec[v[m][rng.integers(len(v[m]))]])
        return torch.as_tensor(np.stack(vecs), device=device)

    def batch_loss(idx: np.ndarray, train: bool, mode: str | None = None, full_vocab: bool = False):
        y = y_all[idx]
        golds = np.unique(items.y[idx][items.y[idx] >= 0])
        if full_vocab:
            L = np.arange(len(vocab))
        else:
            m = int(rng.integers(cfg.min_labels, cfg.max_labels + 1))
            others = np.setdiff1d(np.arange(len(vocab)), golds)
            extra = rng.choice(others, max(0, min(len(others), m - len(golds))), replace=False)
            L = np.concatenate([golds, extra]).astype(np.int64)
        pos = torch.full((len(vocab),), -1, dtype=torch.long, device=device)
        pos[torch.as_tensor(L, device=device)] = torch.arange(len(L), device=device)
        Lt = torch.as_tensor(L, device=device)
        yy = y.clone()
        has = yy >= 0
        # mask labels compatible with the gold (except the gold itself)
        mask = torch.zeros(len(idx), len(L) + 1, dtype=torch.bool, device=device)
        if has.any():
            c = compat_t[yy[has]][:, Lt]                                   # (b, |L|)
            c &= Lt.unsqueeze(0) != yy[has].unsqueeze(1)
            mask[has, :len(L)] = c
        if train and cfg.p_gold_drop > 0 and has.any():
            drop = has & (torch.rand(len(idx), device=device) < cfg.p_gold_drop)
            if drop.any():
                mask[drop, :len(L)] |= compat_t[yy[drop]][:, Lt]
                yy = torch.where(drop, torch.full_like(yy, NONE_TARGET), yy)
        target = torch.where(yy >= 0, pos[yy.clamp(min=0)], torch.full_like(yy, len(L)))
        s = head.span_vec(rep[idx], v1[idx], pb[idx])
        lg = head.logits(s, head.label_vec(label_matrix(L, mode)))
        lg = lg.masked_fill(mask, -1e4)
        return F.cross_entropy(lg, target), lg, target

    log(json.dumps({"head_items": {k: v for k, v in stats.items() if k != "per_label"}, "steps": steps,
                    "train": len(tr), "val": len(va)}))
    t0 = time.time()
    head.train()
    order = rng.permutation(tr)
    pos_ = 0
    for step in range(steps):
        if pos_ + cfg.batch > len(order):
            order = rng.permutation(tr); pos_ = 0
        idx = order[pos_:pos_ + cfg.batch]; pos_ += cfg.batch
        loss, _, _ = batch_loss(idx, True)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % 200 == 0 or step + 1 == steps:
            log(json.dumps({"step": step + 1, "loss": round(float(loss.detach()), 4), "sec": round(time.time() - t0, 1)}))
    head.eval()
    val = evaluate_items(head, items, va, variants, lvec, device) if len(va) else {}
    log(json.dumps({"val": val}))
    return save_head(head, out, cfg, {"items": stats, "val": val, "vocab": vocab, "store": str(store),
                                      "store_meta": json.loads((store / "meta.json").read_text()),
                                      "heldout_nodes": sorted(heldout_nodes)})


@torch.no_grad()
def evaluate_items(head: TypingHead, items: Items, idx: np.ndarray, variants, lvec, device) -> dict:
    """Typing accuracy on held-back training-distribution items against the full training
    vocabulary (compatible labels masked), per text variant; NONE recall on negatives."""
    vocab = items.labels
    compat = np.array([[LB.compatible(a, b) for b in vocab] for a in vocab], dtype=bool)
    out = {}
    s = head.span_vec(torch.as_tensor(items.rep[idx], device=device), torch.as_tensor(items.v1[idx], device=device),
                      torch.as_tensor(items.pb[idx], device=device))
    y = items.y[idx]
    for mode in ("name", "desc"):
        l = torch.as_tensor(np.stack([lvec[variants[i][mode][0]] for i in range(len(vocab))]), device=device)
        lg = head.logits(s, head.label_vec(l)).cpu().numpy()
        g = y >= 0
        m = np.zeros_like(lg, dtype=bool)
        m[g, :len(vocab)] = compat[y[g]] & (np.arange(len(vocab))[None, :] != y[g][:, None])
        lg[m] = -1e4
        pred = lg.argmax(1)
        pred = np.where(pred == len(vocab), NONE_TARGET, pred)
        out[mode] = {"typing_acc": float((pred[g] == y[g]).mean()) if g.any() else None,
                     "none_recall": float((pred[~g] == NONE_TARGET).mean()) if (~g).any() else None}
    return out


def save_head(head: TypingHead, out: Path, cfg: HeadConfig, info: dict) -> Path:
    import os, shutil
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    torch.save(head.state_dict(), tmp / "head.pt")
    sha = hashlib.sha256((tmp / "head.pt").read_bytes()).hexdigest()
    (tmp / "head_manifest.json").write_text(json.dumps(
        {"head_version": HEAD_VERSION, "config": asdict(cfg), "hidden": head.hidden, "n_v1": head.n_v1,
         "head_sha256": sha, **info}, indent=2, default=str))
    shutil.rmtree(out, ignore_errors=True)
    os.replace(tmp, out)
    return out


def load_head(path: Path, device: str | None = None) -> tuple[TypingHead, dict]:
    man = json.loads((Path(path) / "head_manifest.json").read_text())
    cfg = HeadConfig(**{k: v for k, v in man["config"].items() if k in HeadConfig.__dataclass_fields__})
    h = TypingHead(man["hidden"], man["n_v1"], cfg)
    h.load_state_dict(torch.load(Path(path) / "head.pt", map_location="cpu"))
    return h.to(device or "cpu").eval(), man


def main(argv=None) -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=float, default=4.0)
    a = ap.parse_args(argv)
    print(train_head(a.store, a.out, LB.heldout_nodes(), HeadConfig(seed=a.seed, epochs=a.epochs)))


if __name__ == "__main__":
    import sys
    main(sys.argv[1:])
