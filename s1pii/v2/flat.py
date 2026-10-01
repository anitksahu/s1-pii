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
FLAT_VERSION = "v2.1-flat-0.1"


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


BGE_MODEL = "BAAI/bge-base-en-v1.5"


class LabelEncoder:
    """Label-text embedding for the flat CRF.
    * ``v1``: the frozen v1 encoder, mean-pooled over non-special tokens (the v2 behaviour);
    * ``bge``: a frozen sentence encoder (bge-base-en-v1.5, CLS pooling + L2, bge's recipe) at a
      pinned revision recorded in the manifest (v2.1)."""

    def __init__(self, kind: str, enc=None, tok=None, device: str = "cpu", model: str = BGE_MODEL,
                 revision: str | None = None, normalize: bool | None = None, max_length: int | None = None):
        """``normalize``/``max_length`` default to the v2 behaviour for ``v1`` (raw mean pool, 64
        tokens) and to bge's recipe for ``bge`` (L2, 128); the go/no-go M1 arm sets v1 to (True,
        128) so that only the encoder differs between arms."""
        self.kind, self.device, self.model = kind, device, model
        self.normalize = (kind == "bge") if normalize is None else normalize
        self.max_length = (128 if kind == "bge" else 64) if max_length is None else max_length
        if kind == "bge":
            from transformers import AutoModel, AutoTokenizer
            if revision is None:
                from huggingface_hub import HfApi
                revision = HfApi().model_info(model).sha
            self.revision = revision
            self.tok = AutoTokenizer.from_pretrained(model, revision=revision)
            self.enc = AutoModel.from_pretrained(model, revision=revision).to(device).eval()
        elif kind == "v1":
            self.revision, self.enc, self.tok, self.model = None, enc, tok, "v1"
        else:
            raise ValueError(kind)
        self.dim = self.enc.config.hidden_size

    def spec(self) -> dict:
        return {"kind": self.kind, "model": self.model, "revision": self.revision, "dim": self.dim,
                "normalize": self.normalize, "max_length": self.max_length}

    @classmethod
    def from_spec(cls, spec: dict | None, enc, tok, device: str):
        spec = spec or {"kind": "v1"}
        return cls(spec["kind"], enc, tok, device, spec.get("model", BGE_MODEL), spec.get("revision"),
                   spec.get("normalize"), spec.get("max_length"))

    @torch.no_grad()
    def __call__(self, texts: list[str]) -> torch.Tensor:
        out = []
        for i in range(0, len(texts), 64):
            e = self.tok(texts[i:i + 64], padding=True, return_tensors="pt", return_special_tokens_mask=True,
                         truncation=True, max_length=self.max_length)
            sp = e.pop("special_tokens_mask")
            h = self.enc(input_ids=e["input_ids"].to(self.device),
                         attention_mask=e["attention_mask"].to(self.device)).last_hidden_state.float()
            if self.kind == "bge":
                v = h[:, 0]
            else:
                m = (e["attention_mask"].bool() & ~sp.bool()).to(self.device).unsqueeze(-1).float()
                v = (h * m).sum(1) / m.sum(1).clamp(min=1)
            out.append(F.normalize(v, dim=-1) if self.normalize else v)
        return torch.cat(out) if out else torch.zeros(0, self.dim, device=self.device)


_GENERIC = {"number", "id", "code", "name", "address", "type", "level", "info", "information", "of", "the", "and", "or",
            "a", "an", "date", "status", "identifier", "details", "data"}


def _mentions(text: str, name: str) -> bool:
    """True if the text contains ANY distinctive word of the label name (case-insensitive; generic
    words such as "number" or "code" do not count unless the name has no other word)."""
    import re
    words = set(re.findall(r"[a-z0-9]+", text.lower()))
    nw = re.findall(r"[a-z0-9]+", name.lower().replace("_", " "))
    key = [w for w in nw if w not in _GENERIC] or nw
    return any(w in words for w in key)


def label_text_choices(name: str) -> dict[str, list[str]]:
    """``name``: texts with the label name (bare name, "name: description", and paraphrases or
    the description if they mention it); ``other``: texts that do not mention the name, used for
    name dropout (may be empty: then dropout falls back to the name texts)."""
    from .head import label_text_variants
    v = label_text_variants(name)
    desc_only = LB.NATIVE[name.lower()][1]
    pool = sorted({t for t in [*v["name"], *v["desc"], *v["para"], desc_only] if t})
    other = [t for t in pool if not _mentions(t, name)]
    return {"name": [t for t in pool if t not in other], "other": other}


class FlatModel(nn.Module):
    def __init__(self, hidden: int, dim: int = 256, label_dim: int | None = None, label_linear: bool = False):
        super().__init__()
        ld = label_dim or hidden
        self.kind = nn.Linear(hidden, 4 * dim)
        self.label = (nn.Linear(ld, dim) if label_linear else
                      nn.Sequential(nn.Linear(ld, dim), nn.GELU(), nn.Linear(dim, dim)))
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
               batch: int = 8, max_len: int = 512, max_labels: int = 24, min_labels: int = 4, device: str | None = None,
               label_encoder: str = "v1", text_mode: str = "desc", p_name_drop: float = 0.4, label_linear: bool = False,
               label_revision: str | None = None, label_normalize: bool | None = None, label_max_length: int | None = None,
               log=print) -> Path:
    """``text_mode='desc'`` (v2): every label is the name + description text. ``'sample'`` (v2.1):
    per step and label, with probability ``p_name_drop`` a paraphrase or description without the
    bare name, otherwise uniformly the name, a paraphrase or the description."""
    from ..model.train import load_exported
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    v1, tok, man = load_exported(v1_dir)
    enc = v1.encoder.to(device).eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    H = enc.config.hidden_size
    lenc = LabelEncoder(label_encoder, enc, tok, device, revision=label_revision, normalize=label_normalize,
                        max_length=label_max_length)
    fm = FlatModel(H, label_dim=lenc.dim, label_linear=label_linear).to(device)
    opt = torch.optim.AdamW(fm.parameters(), lr=1e-3, weight_decay=0.01)
    vocab = sorted({s.label_raw.lower() for d in docs for s in d.spans
                    if s.label_raw and s.label_raw.lower() in LB.NATIVE and LB.node_of(s.label_raw) not in heldout})
    pre, suf = special_ids(tok)
    size = max_len - len(pre) - len(suf)
    lix = {n: i for i, n in enumerate(vocab)}
    if text_mode == "desc":
        choices = {n: {"name": [LB.native_label(n).text(True)], "other": [LB.native_label(n).text(True)]} for n in vocab}
    elif text_mode == "sample":
        choices = {n: label_text_choices(n) for n in vocab}
    else:
        raise ValueError(text_mode)
    all_texts = sorted({t for c in choices.values() for v in c.values() for t in v})
    tvec = dict(zip(all_texts, lenc(all_texts)))

    def label_vecs(L):
        out_ = []
        for n in L:
            c = choices[n]
            if text_mode == "desc":
                t = c["name"][0]
            elif c["other"] and rng.random() < p_name_drop:
                t = c["other"][int(rng.integers(len(c["other"])))]
            else:
                pool = c["name"] + c["other"]
                t = pool[int(rng.integers(len(pool)))]
            out_.append(tvec[t])
        return torch.stack(out_)
    skipped = 0
    out.mkdir(parents=True, exist_ok=True)
    (out / "train_log.jsonl").write_text("")
    t0 = time.time()
    for step in range(steps):
        ds = [docs[i] for i in rng.choice(len(docs), batch, replace=False)]
        golds = sorted({s.label_raw.lower() for d in ds for s in d.spans if s.label_raw and s.label_raw.lower() in lix})
        m = int(rng.integers(min_labels, max_labels + 1))
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
        em = fm.emissions(h, label_vecs(L))
        loss = fm.crf(len(L)).nll(em, mask.to(device), allowed.to(device))
        if not torch.isfinite(loss) or loss > 8e3:     # overlapping gold with no valid path (NEG = -1e4)
            skipped += 1
            continue
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if (step + 1) % 100 == 0 or step + 1 == steps:
            rec = json.dumps({"flat_step": step + 1, "loss": round(float(loss.detach()), 4), "skipped": skipped,
                              "sec": round(time.time() - t0, 1)})
            log(rec)
            with open(out / "train_log.jsonl", "a") as tf:
                tf.write(rec + "\n")
    torch.save(fm.state_dict(), out / "flat.pt")
    import hashlib
    sha = hashlib.sha256((out / "flat.pt").read_bytes()).hexdigest()
    (out / "flat_manifest.json").write_text(json.dumps({"v1": man["weights_sha256"], "v1_dir": str(v1_dir), "steps": steps,
                                                        "weights_sha256": sha,
                                                        "seed": seed, "vocab": vocab, "heldout_nodes": sorted(heldout),
                                                        "label_encoder": lenc.spec(), "text_mode": text_mode,
                                                        "p_name_drop": p_name_drop, "label_linear": label_linear,
                                                        "min_labels": min_labels, "max_labels": max_labels,
                                                        "skipped_steps": skipped, "flat_version": FLAT_VERSION}, indent=2))
    return out


@torch.no_grad()
def predict_flat(v1_dir: Path, flat_dir: Path, docs: list[Doc], L: LB.LabelSet, *, system: str, max_len: int = 1024,
                 floor: float = 0.01, score: str = "sensitive", label_floor: float = 0.002, max_span: int = 64,
                 device: str | None = None) -> dict[str, list[Span]]:
    """``score='sensitive'``: P(sensitive under L) = summed exact segment marginals of the sensitive
    labels, label = argmax (v2). ``score='typed'`` (v2.1, C3): every label decoded, label = argmax
    marginal, score = that label's marginal (non-sensitive labels such as IGNORE quasi-identifiers
    are kept). ``label_floor`` prunes per-label segment marginals before they are summed."""
    from ..model.train import load_exported
    if score not in ("sensitive", "typed"):
        raise ValueError(score)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    v1, tok, _ = load_exported(v1_dir)
    enc = v1.encoder.to(device).eval()
    fman = json.loads((flat_dir / "flat_manifest.json").read_text())
    lenc = LabelEncoder.from_spec(fman.get("label_encoder"), enc, tok, device)
    fm = FlatModel(enc.config.hidden_size, label_dim=lenc.dim, label_linear=fman.get("label_linear", False)).to(device)
    fm.load_state_dict(torch.load(flat_dir / "flat.pt", map_location=device)); fm.eval()
    lv = lenc(L.texts())
    nt = len(L.labels)
    c64 = FlatCRF(nt, fm.theta.double(), fm.start.double(), fm.end.double())
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
            alpha, logz = c64._forward(em, msk); beta = c64._backward(em, msk)
            n = b - a
            agg: dict = {}
            for i, j, t, p in span_logprobs(c64, em[0].cpu().numpy(), alpha[0].cpu().numpy(), beta[0].cpu().numpy(),
                                            float(logz[0]), n, floor=label_floor, max_len=max_span):
                if (i == 0 and a > 0) or (j == n - 1 and b < len(td.ids)):
                    continue
                v = agg.setdefault((i, j), np.zeros(nt)); v[t] += p
            for (i, j), v in agg.items():
                k = int(v.argmax())
                sc = float(v[sens].sum()) if score == "sensitive" else float(v[k])
                if sc < floor:
                    continue
                cs, ce = int(td.offsets[a + i][0]), int(td.offsets[a + j][1])
                if (cs, ce) not in best or sc > best[(cs, ce)].score:
                    best[(cs, ce)] = Span(d.doc_id, cs, ce, canon[k], label_raw=L.labels[k].name, score=min(1.0, sc),
                                          source=system, surface=d.text[cs:ce])
        out[d.doc_id] = sorted(best.values(), key=lambda s: (s.start, s.end))
    return out
