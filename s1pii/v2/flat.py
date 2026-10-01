"""Ablation C (no hierarchy): a flat label-conditioned CRF.

Emissions for tag (kind, k) at token t are <W_kind h_t, U l_k> * scale, the O emission is a
linear function of h_t, and transitions are shared across labels at the level of tag kinds
(O/B/I/E/S), so one CRF handles any label set: nt = |L|. The encoder is the frozen v1
encoder. P(sensitive under L) = sum over sensitive k of the exact segment marginal of
(span, k). One seed, on a fixed training subset (cut order in PLAN_v2).

Training targets for a batch label set L_b: gold spans whose label is in L_b get its tags;
gold spans whose label is absent from L_b but compatible with a label in L_b (ancestor,
descendant, synonym) are ANY; other gold spans are O (not an entity of any class in L);
held-out spans are ANY.
"""
from __future__ import annotations

import functools
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ..schema import Doc, Span
from ..model.crf import CRF, constraint_masks, span_logprobs, tag, NEG
from ..model.encode import tokenize_doc, token_windows, special_ids, Example, ANY, _trim
from . import labels as LB

KINDS = "OBIES"


def kind_index(k: int) -> int:
    return 0 if k == 0 else 1 + (k - 1) % 4


@functools.lru_cache(maxsize=64)
def _kind_map(nt: int) -> torch.Tensor:
    return torch.tensor([kind_index(k) for k in range(1 + 4 * nt)])


class FlatCRF(CRF):
    """CRF whose K x K potentials are expanded from 5 x 5 kind-level parameters."""

    def __init__(self, nt: int, theta: torch.Tensor, start: torch.Tensor, end: torch.Tensor):
        nn.Module.__init__(self)
        self.nt, self.k = nt, 1 + 4 * nt
        self._theta, self._start, self._end = theta, start, end
        tr, st, en = _masks(nt)
        self.tr_mask, self.st_mask, self.en_mask = tr, st, en
        self.o_exit_bias = 0.0

    def potentials(self):
        km = _kind_map(self.nt).to(self._theta.device)
        T = self._theta[km][:, km]
        S = self._start[km]; E = self._end[km]
        dev = T.device
        T = torch.where(self.tr_mask.to(dev), T, torch.full_like(T, NEG))
        S = torch.where(self.st_mask.to(dev), S, torch.full_like(S, NEG))
        E = torch.where(self.en_mask.to(dev), E, torch.full_like(E, NEG))
        return T, S, E


@functools.lru_cache(maxsize=64)
def _masks(nt: int):
    return constraint_masks(nt)


class FlatModel(nn.Module):
    def __init__(self, hidden: int, dim: int = 256):
        super().__init__()
        self.kind = nn.Linear(hidden, 4 * dim)
        self.label = nn.Sequential(nn.Linear(hidden, dim), nn.GELU(), nn.Linear(dim, dim))
        self.o = nn.Linear(hidden, 1)
        self.theta = nn.Parameter(torch.zeros(5, 5)); self.start = nn.Parameter(torch.zeros(5)); self.end = nn.Parameter(torch.zeros(5))
        self.log_scale = nn.Parameter(torch.tensor(math.log(5.0)))
        self.dim = dim

    def emissions(self, h: torch.Tensor, lv: torch.Tensor) -> torch.Tensor:
        """h (B, L, H), lv (nt, H) -> (B, L, 1 + 4 nt) in the crf.py tag order."""
        B, L, _ = h.shape
        kv = F.normalize(self.kind(h).view(B, L, 4, self.dim), dim=-1)
        u = F.normalize(self.label(lv), dim=-1)                                  # (nt, d)
        sc = torch.einsum("blkd,nd->blnk", kv, u) * self.log_scale.exp()         # (B, L, nt, 4)
        return torch.cat([self.o(h), sc.reshape(B, L, -1)], -1)

    def crf(self, nt: int) -> FlatCRF:
        return FlatCRF(nt, self.theta, self.start, self.end)


def flat_allowed(d: Doc, tok, L: list[str], heldout: set[str]):
    """Token ids, offsets and a per-token allowed-tag matrix for label list ``L``."""
    enc = tok(d.text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    offs = np.array([_trim(d.text, a, b) for a, b in enc["offset_mapping"]], dtype=np.int64).reshape(-1, 2)
    n, nt = len(enc["input_ids"]), len(L)
    K = 1 + 4 * nt
    allowed = np.zeros((n, K), dtype=bool); allowed[:, 0] = True
    st, en = offs[:, 0], offs[:, 1]; vis = en > st
    li = {l: i for i, l in enumerate(L)}
    for s in d.spans:
        raw = (s.label_raw or "").lower()
        if not raw or raw == "_scaffold":
            continue
        hit = np.nonzero(vis & (st < s.end) & (en > s.start))[0]
        if len(hit) == 0:
            continue
        hit = np.arange(hit[0], hit[-1] + 1)
        if raw in LB.NATIVE and LB.node_of(raw) in heldout:
            allowed[hit] = True; continue
        if raw in li:
            t = li[raw]
            row = np.zeros((len(hit), K), dtype=bool)
            if len(hit) == 1:
                row[0, tag("S", t)] = True
            else:
                row[0, tag("B", t)] = True; row[1:-1, tag("I", t)] = True; row[-1, tag("E", t)] = True
            allowed[hit] = row
        elif any(LB.compatible(raw, l) for l in L if raw in LB.NATIVE):
            allowed[hit] = True
    return list(enc["input_ids"]), allowed


def train_flat(v1_dir: Path, docs: list[Doc], out: Path, heldout: set[str], *, seed: int = 1, steps: int = 3000,
               batch: int = 8, max_len: int = 512, max_labels: int = 24, device: str | None = None, log=print) -> Path:
    from ..model.train import load_exported
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    v1, tok, man = load_exported(v1_dir)
    enc = v1.encoder.to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    H = enc.config.hidden_size
    fm = FlatModel(H).to(device)
    opt = torch.optim.AdamW(fm.parameters(), lr=1e-3, weight_decay=0.01)
    vocab = sorted({s.label_raw.lower() for d in docs for s in d.spans
                    if s.label_raw and s.label_raw.lower() in LB.NATIVE and LB.node_of(s.label_raw) not in heldout})
    texts = {n: LB.native_label(n).text(True) for n in vocab}
    pre, suf = special_ids(tok)
    size = max_len - len(pre) - len(suf)

    @torch.no_grad()
    def embed(ts):
        e = tok(ts, padding=True, return_tensors="pt", return_special_tokens_mask=True, truncation=True, max_length=64)
        sp = e.pop("special_tokens_mask")
        h = enc(input_ids=e["input_ids"].to(device), attention_mask=e["attention_mask"].to(device)).last_hidden_state.float()
        m = (e["attention_mask"].bool() & ~sp.bool()).to(device).unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp(min=1)
    lvec_all = embed([texts[n] for n in vocab]); lix = {n: i for i, n in enumerate(vocab)}
    t0 = time.time()
    for step in range(steps):
        ds = [docs[i] for i in rng.choice(len(docs), batch, replace=False)]
        golds = sorted({s.label_raw.lower() for d in ds for s in d.spans if s.label_raw and s.label_raw.lower() in lix})
        m = int(rng.integers(4, max_labels + 1))
        others = [n for n in vocab if n not in golds]
        L = golds + list(rng.choice(others, max(0, min(len(others), m - len(golds))), replace=False))
        rows = []
        for d in ds:
            ids, allowed = flat_allowed(d, tok, L, heldout)
            if not ids:
                continue
            a = int(rng.integers(0, max(1, len(ids) - size + 1)))
            b = min(len(ids), a + size)
            al = allowed[a:b].copy()
            for edge in (0, b - a - 1):                  # a span cut by the crop is unknown
                if (a > 0 and edge == 0) or (b < len(ids) and edge == b - a - 1):
                    if not al[edge, 0] or al[edge].sum() > 1:
                        al[edge] = True
            rows.append((pre + ids[a:b] + suf, al))
        if not rows:
            continue
        Lin = max(len(r[0]) for r in rows); Lt = max(len(r[1]) for r in rows)
        pad = tok.pad_token_id or 0
        ids = torch.full((len(rows), Lin), pad, dtype=torch.long); att = torch.zeros_like(ids)
        allowed = torch.ones(len(rows), Lt, 1 + 4 * len(L), dtype=torch.bool); mask = torch.zeros(len(rows), Lt, dtype=torch.bool)
        for k, (x, al) in enumerate(rows):
            ids[k, :len(x)] = torch.as_tensor(x); att[k, :len(x)] = 1
            allowed[k, :len(al)] = torch.as_tensor(al); mask[k, :len(al)] = True
        with torch.no_grad():
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.autocast("cpu", enabled=False)
            with ctx:
                h = enc(input_ids=ids.to(device), attention_mask=att.to(device)).last_hidden_state.float()
            h = h[:, len(pre):len(pre) + Lt]
        em = fm.emissions(h, lvec_all[[lix[n] for n in L]])
        loss = fm.crf(len(L)).nll(em, mask.to(device), allowed.to(device))
        if not torch.isfinite(loss) or loss > 8e3:     # overlapping gold with no valid path (NEG = -1e4)
            continue
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (step + 1) % 100 == 0 or step + 1 == steps:
            log(json.dumps({"flat_step": step + 1, "loss": round(float(loss), 4), "sec": round(time.time() - t0, 1)}))
    out.mkdir(parents=True, exist_ok=True)
    torch.save(fm.state_dict(), out / "flat.pt")
    import hashlib
    sha = hashlib.sha256((out / "flat.pt").read_bytes()).hexdigest()
    (out / "flat_manifest.json").write_text(json.dumps({"v1": man["weights_sha256"], "v1_dir": str(v1_dir), "steps": steps,
                                                        "weights_sha256": sha,
                                                        "seed": seed, "vocab": vocab, "heldout_nodes": sorted(heldout)}, indent=2))
    return out


@torch.no_grad()
def predict_flat(v1_dir: Path, flat_dir: Path, docs: list[Doc], L: LB.LabelSet, *, system: str, max_len: int = 1024,
                 floor: float = 0.01, device: str | None = None) -> dict[str, list[Span]]:
    from ..model.train import load_exported
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    v1, tok, _ = load_exported(v1_dir)
    enc = v1.encoder.to(device).eval()
    fm = FlatModel(enc.config.hidden_size).to(device)
    fm.load_state_dict(torch.load(flat_dir / "flat.pt", map_location=device)); fm.eval()
    e = tok(L.texts(), padding=True, return_tensors="pt", return_special_tokens_mask=True, truncation=True, max_length=64)
    sp = e.pop("special_tokens_mask")
    h = enc(input_ids=e["input_ids"].to(device), attention_mask=e["attention_mask"].to(device)).last_hidden_state.float()
    m = (e["attention_mask"].bool() & ~sp.bool()).to(device).unsqueeze(-1).float()
    lv = (h * m).sum(1) / m.sum(1).clamp(min=1)
    nt = len(L.labels)
    crf = fm.crf(nt)
    sens = np.array(L.sensitive_mask())
    canon = [l.resolved_canonical() for l in L.labels]
    pre, suf = special_ids(tok)
    size = max_len - len(pre) - len(suf); stride = size // 2
    out = {}
    for d in docs:
        td = tokenize_doc(d, tok)
        best = {}
        for a, b in (token_windows(len(td.ids), size, stride) if td.ids else []):
            ids = torch.as_tensor([pre + td.ids[a:b] + suf], device=device)
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.autocast("cpu", enabled=False)
            with ctx:
                hh = enc(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state.float()
            em = fm.emissions(hh[:, len(pre):len(pre) + b - a], lv).double()
            msk = torch.ones(1, b - a, dtype=torch.bool, device=device)
            c64 = FlatCRF(nt, fm.theta.double(), fm.start.double(), fm.end.double())
            alpha, logz = c64._forward(em, msk); beta = c64._backward(em, msk)
            n = b - a
            agg: dict = {}
            for i, j, t, p in span_logprobs(c64, em[0].cpu().numpy(), alpha[0].cpu().numpy(), beta[0].cpu().numpy(),
                                            float(logz[0]), n, floor=0.002):
                if (i == 0 and a > 0) or (j == n - 1 and b < len(td.ids)):
                    continue
                v = agg.setdefault((i, j), np.zeros(nt)); v[t] += p
            for (i, j), v in agg.items():
                ps = float(v[sens].sum())
                if ps < floor:
                    continue
                cs, ce = int(td.offsets[a + i][0]), int(td.offsets[a + j][1])
                k = int(v.argmax())
                if (cs, ce) not in best or ps > best[(cs, ce)].score:
                    best[(cs, ce)] = Span(d.doc_id, cs, ce, canon[k], label_raw=L.labels[k].name, score=min(1.0, ps),
                                          source=system, surface=d.text[cs:ce])
        out[d.doc_id] = sorted(best.values(), key=lambda s: (s.start, s.end))
    return out
