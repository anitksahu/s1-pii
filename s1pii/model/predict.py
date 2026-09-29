"""S1 inference: token windows with 50% overlap -> CRF span marginals -> character spans.

Same output contract as the GLiNER adapters (``predict_docs`` returns canonical Spans and a
report; ``config`` is part of the cache key), so ``s1pii.adapters.run`` drives S1 too.
Spans whose tokens touch an interior window edge are dropped from that window; with 50%
overlap every span shorter than the stride is fully inside some window. Identical
(start, end, type) spans across windows keep the maximum probability.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..schema import Doc, Span, CANONICAL_TYPES
from .encode import tokenize_doc, token_windows, special_ids, Example
from .s1 import S1Model, collate, gather_tokens
from .crf import span_logprobs
from .postprocess import validator_spans, propagate

PREDICTOR_VERSION = "s1-predict-v0.1"


@dataclass
class S1Report:
    windows: int = 0
    raw: int = 0
    kept: int = 0
    dropped_edge: int = 0
    validator: int = 0
    propagated: int = 0
    repaired: int = 0
    dropped_unanchored: int = 0
    dropped_unknown_label: int = 0
    dropped_bad_offsets: int = 0
    window_splits: int = 0
    unknown_labels: dict = field(default_factory=dict)

    def merge(self, o) -> None:
        for k, v in o.__dict__.items():
            if isinstance(v, int):
                setattr(self, k, getattr(self, k, 0) + v)


class S1Predictor:
    """``predict_docs`` interface shared with ``adapters.base.Adapter``."""

    def __init__(self, model: S1Model, tokenizer, *, revision: str, max_len: int = 1024, floor: float = 0.01,
                 max_span_tokens: int = 64, validators: bool = True, propagation: bool = True,
                 device: str | None = None, batch_size: int = 8, system: str = "s1"):
        self.model, self.tokenizer = model, tokenizer
        self.revision, self.max_len, self.floor = revision, max_len, floor
        self.max_span_tokens, self.validators, self.propagation = max_span_tokens, validators, propagation
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size, self.system = batch_size, system
        self.model.to(self.device).eval()
        import copy
        self.crf = copy.deepcopy(self.model.crf).cpu().double().eval()   # decoding in float64 on CPU
        self.pre, self.suf = special_ids(tokenizer)
        self.size = max_len - len(self.pre) - len(self.suf)
        self.stride = max(1, self.size // 2)
        import transformers
        self.versions = {"torch": torch.__version__, "transformers": transformers.__version__}

    def config(self) -> dict:
        return {"predictor_version": PREDICTOR_VERSION, "max_len": self.max_len, "floor": self.floor,
                "max_span_tokens": self.max_span_tokens, "validators": self.validators,
                "propagation": self.propagation, "stride": self.stride, "o_exit_bias": self.crf.o_exit_bias}

    @torch.no_grad()
    def _emissions(self, exs: list[Example]) -> list[np.ndarray]:
        out = []
        for i in range(0, len(exs), self.batch_size):
            chunk = exs[i:i + self.batch_size]
            b = collate(chunk, self.tokenizer.pad_token_id or 0, with_targets=False)
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if self.device == "cuda" else torch.autocast("cpu", enabled=False)
            with ctx:
                em = self.model.emissions(b["input_ids"].to(self.device), b["attention_mask"].to(self.device))
            em = gather_tokens(em.float(), b["n_prefix"].to(self.device), int(b["tok_len"].max()))
            for j, e in enumerate(chunk):
                out.append(em[j, :e.n_tok].double().cpu().numpy())
        return out

    def predict_docs(self, docs: list[Doc], batch_size: int | None = None):
        if batch_size:
            self.batch_size = batch_size
        rep = S1Report()
        preds: dict[str, list[Span]] = {}
        crf = self.crf
        for d in docs:
            td = tokenize_doc(d, self.tokenizer)
            best: dict[tuple, Span] = {}
            wins = token_windows(len(td.ids), self.size, self.stride) if td.ids else []
            exs = [Example(d.doc_id, self.pre + td.ids[a:b] + self.suf, len(self.pre), b - a,
                           np.zeros(b - a, dtype=np.int64), a) for a, b in wins]
            rep.windows += len(exs)
            ems = self._emissions(exs)
            for (a, b), em in zip(wins, ems):
                n = b - a
                emt = torch.from_numpy(em).unsqueeze(0)
                alpha, beta, logz = crf.marginals(emt, torch.ones(1, n, dtype=torch.bool))
                spans = span_logprobs(crf, em, alpha[0].numpy(), beta[0].numpy(), float(logz[0]), n,
                                      floor=self.floor, max_len=self.max_span_tokens)
                for i, j, t, p in spans:
                    rep.raw += 1
                    if (i == 0 and a > 0) or (j == n - 1 and b < len(td.ids)):
                        rep.dropped_edge += 1
                        continue
                    cs, ce = int(td.offsets[a + i][0]), int(td.offsets[a + j][1])
                    if ce <= cs:
                        rep.dropped_bad_offsets += 1
                        continue
                    key = (cs, ce, CANONICAL_TYPES[t])
                    sp = Span(d.doc_id, cs, ce, CANONICAL_TYPES[t], label_raw="crf", score=min(1.0, p),
                              source=self.system, surface=d.text[cs:ce])
                    if key not in best or sp.score > best[key].score:
                        best[key] = sp
            out = list(best.values())
            if self.validators:
                v = validator_spans(d, source=self.system + ":validator")
                rep.validator += len(v)
                for s in v:
                    key = (s.start, s.end, s.label_canonical)
                    if key not in best or s.score > best[key].score:
                        best[key] = s
                out = list(best.values())
            if self.propagation:
                before = len(out)
                out = propagate(d, out)
                rep.propagated += len(out) - before
            preds[d.doc_id] = sorted(out, key=lambda s: (s.start, s.end, s.label_canonical))
        rep.kept = sum(len(v) for v in preds.values())
        return preds, rep


def weights_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    for f in sorted(Path(path).glob("*.safetensors")) + sorted(Path(path).glob("*.bin")) + sorted(Path(path).glob("s1_head.pt")):
        h.update(f.name.encode()); h.update(f.read_bytes())
    return h.hexdigest()
