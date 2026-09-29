"""Adapter core shared by every system: config, windowing, stitching, output normalization.

An adapter turns a ``Doc`` into canonical ``Span`` predictions:
1. Split the text into word windows with 50% overlap (``windows``). Window boundaries are
   character offsets into the original text, so no offset remapping is needed beyond
   adding the window start.
2. Call the model backend on each window (batched when the backend supports it) with the
   frozen query labels and the common emission floor.
3. Normalize the backend's raw output (``normalize_gliner`` / ``normalize_gliner2``) into
   (start, end, label, score, surface) tuples, re-anchor each against the surface the
   model reported, map the native label to a canonical type, and shift by the window start.
4. Stitch windows: identical (start, end, type) spans keep their maximum score; different
   overlapping spans are all kept (metrics take the per-character maximum).
Every repaired or dropped span is counted in the adapter report.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Any, Iterable, Protocol, Sequence

import yaml

from ..schema import Doc, Span, CANONICAL_TYPES
from ..eval.align import reanchor

ADAPTER_VERSION = "adapters-v0.2"
# GLiNER's own word splitter (words, hyphen/underscore compounds, single punctuation marks).
# Windows are sized in these tokens, so punctuation-heavy text (JSON, logs) gets shorter windows.
_TOK = re.compile(r"\w+(?:[-_]\w+)*|\S")


@lru_cache(maxsize=None)
def load_config(name: str = "baselines.yaml") -> dict:
    return yaml.safe_load(resources.files("s1pii.configs").joinpath(name).read_text())


def query_labels(system: str) -> dict[str, str]:
    """native query label -> canonical type (frozen with the prereg).

    Uniform coverage rule (prereg v0): every system is queried with the canonical label set
    of ``labels.yaml`` (every canonical type, with descriptions where the backend accepts
    them); PII-trained models additionally get their native label names from
    ``baselines.yaml``. So no system is structurally blind to a canonical type."""
    cfg = load_config()["systems"][system]
    canon = {k: v["type"] for k, v in load_config("labels.yaml").items()}
    labels = {**dict(cfg.get("labels", {})), **canon}
    bad = {k: v for k, v in labels.items() if v not in CANONICAL_TYPES}
    if bad:
        raise ValueError(f"{system}: labels map to non-canonical types {bad}")
    return labels


def label_descriptions(system: str) -> dict[str, str] | None:
    """Descriptions for the canonical labels, for backends that accept label descriptions
    (``use_descriptions: true`` in baselines.yaml). Native labels are passed by name."""
    cfg = load_config()["systems"][system]
    if not cfg.get("use_descriptions"):
        return None
    return {k: v["description"] for k, v in load_config("labels.yaml").items()}


def windows(text: str, words: int = 150, stride: int = 75) -> list[tuple[int, int]]:
    """Character windows covering ``words`` GLiNER word-tokens each, advancing ``stride``.
    Every character of the text is covered; the last window ends at len(text)."""
    bounds = [(m.start(), m.end()) for m in _TOK.finditer(text)]
    if len(bounds) <= words:
        return [(0, len(text))]
    out, i = [], 0
    while True:
        j = min(i + words, len(bounds))
        start = 0 if i == 0 else bounds[i][0]
        end = len(text) if j == len(bounds) else bounds[j - 1][1]
        out.append((start, end))
        if j == len(bounds):
            break
        i += stride
    return out


@dataclass
class RawEnt:
    start: int
    end: int
    label: str
    score: float
    surface: str | None


def normalize_gliner(out: Iterable[dict]) -> list[RawEnt]:
    """gliner ``predict_entities`` output: [{start, end, text, label, score}]."""
    ents = []
    for e in out or []:
        ents.append(RawEnt(int(e["start"]), int(e["end"]), str(e["label"]), float(e["score"]), e.get("text")))
    return ents


def normalize_gliner2(out: Any) -> list[RawEnt]:
    """gliner2 ``extract_entities(..., include_confidence=True, include_spans=True)`` output:
    {'entities': {label: [{'text', 'confidence', 'start', 'end'}]}}. Also accepts a list of
    {label, start, end, text, confidence|score} for forward compatibility."""
    ents = []
    if isinstance(out, dict):
        groups = out.get("entities", out)
        for label, items in groups.items():
            for e in items or []:
                if not isinstance(e, dict) or "start" not in e:
                    raise ValueError(f"gliner2 output for {label!r} lacks spans; call with include_spans=True")
                ents.append(RawEnt(int(e["start"]), int(e["end"]), str(label),
                                   float(e.get("confidence", e.get("score", 1.0))), e.get("text")))
    elif isinstance(out, list):
        for e in out:
            ents.append(RawEnt(int(e["start"]), int(e["end"]), str(e["label"]),
                               float(e.get("confidence", e.get("score", 1.0))), e.get("text")))
    else:
        raise TypeError(f"unexpected gliner2 output type {type(out).__name__}")
    return ents


class Backend(Protocol):
    def predict(self, texts: Sequence[str], labels: list[str], threshold: float) -> list[list[RawEnt]]: ...
    # optional: def fits(self, text: str, labels: list[str]) -> bool  (context-limit check)


class WindowTooLong(RuntimeError):
    pass


def fit_windows(text: str, spans: list[tuple[int, int]], backend, labels: list[str],
                min_tokens: int = 8) -> tuple[list[tuple[int, int]], int]:
    """Split any window the backend reports as over its context limit (label prompt
    included) into overlapping halves until it fits. Returns (windows, n_splits). Raises
    ``WindowTooLong`` if a window cannot be made to fit, so truncation is never silent."""
    fits = getattr(backend, "fits", None)
    if fits is None:
        return spans, 0
    out, splits, stack = [], 0, list(reversed(spans))
    while stack:
        a, b = stack.pop()
        if fits(text[a:b], labels):
            out.append((a, b)); continue
        toks = [(a + m.start(), a + m.end()) for m in _TOK.finditer(text[a:b])]
        if len(toks) <= min_tokens:
            raise WindowTooLong(f"window [{a},{b}) exceeds the model context even at {len(toks)} tokens")
        half, quarter = len(toks) // 2, len(toks) // 4
        left = (a, toks[half - 1][1])
        right = (toks[max(0, half - quarter)][0], b)
        splits += 1
        stack.extend([right, left])
    return sorted(out), splits


@dataclass
class AdapterReport:
    windows: int = 0
    raw: int = 0
    kept: int = 0
    repaired: int = 0
    dropped_unanchored: int = 0
    dropped_unknown_label: int = 0
    dropped_bad_offsets: int = 0
    window_splits: int = 0
    unknown_labels: dict = field(default_factory=dict)

    def merge(self, o: "AdapterReport") -> None:
        for k in ("windows", "raw", "kept", "repaired", "dropped_unanchored", "dropped_unknown_label",
                  "dropped_bad_offsets", "window_splits"):
            setattr(self, k, getattr(self, k) + getattr(o, k))
        for k, v in o.unknown_labels.items():
            self.unknown_labels[k] = self.unknown_labels.get(k, 0) + v


class Adapter:
    def __init__(self, system: str, backend: Backend, *, floor: float | None = None,
                 words: int | None = None, stride: int | None = None):
        cfg = load_config()
        self.system = system
        self.backend = backend
        self.labels = query_labels(system)
        self.floor = cfg["emission_floor"] if floor is None else floor
        self.words = words or cfg["window"]["words"]
        self.stride = stride or cfg["window"]["stride"]

    def config(self) -> dict:
        return {"labels": self.labels, "floor": self.floor, "words": self.words, "stride": self.stride,
                "descriptions_used": bool(getattr(self.backend, "descriptions", None)),
                "adapter_version": ADAPTER_VERSION}

    def predict_docs(self, docs: Sequence[Doc], batch_size: int = 16) -> tuple[dict[str, list[Span]], AdapterReport]:
        jobs = []  # (doc, window_start, window_text)
        qlabels = list(self.labels)
        n_splits = 0
        for d in docs:
            ws, k = fit_windows(d.text, windows(d.text, self.words, self.stride), self.backend, qlabels)
            n_splits += k
            for a, b in ws:
                jobs.append((d, a, d.text[a:b]))
        rep = AdapterReport(windows=len(jobs), window_splits=n_splits)
        best: dict[str, dict[tuple, Span]] = {d.doc_id: {} for d in docs}
        for i in range(0, len(jobs), batch_size):
            chunk = jobs[i:i + batch_size]
            outs = self.backend.predict([t for _, _, t in chunk], qlabels, self.floor)
            if len(outs) != len(chunk):
                raise ValueError(f"backend returned {len(outs)} results for {len(chunk)} windows")
            for (doc, off, wtext), ents in zip(chunk, outs):
                for e in ents:
                    rep.raw += 1
                    typ = self.labels.get(e.label)
                    if typ is None:
                        rep.dropped_unknown_label += 1
                        rep.unknown_labels[e.label] = rep.unknown_labels.get(e.label, 0) + 1
                        continue
                    if not (0 <= e.start < e.end <= len(wtext)):
                        rep.dropped_bad_offsets += 1
                        continue
                    if e.score < self.floor:
                        continue
                    sp = Span(doc.doc_id, off + e.start, off + e.end, typ, label_raw=e.label,
                              score=min(1.0, max(0.0, e.score)), source=self.system,
                              surface=e.surface if e.surface is not None else None)
                    fixed = reanchor(doc.text, sp)
                    if fixed is None:
                        rep.dropped_unanchored += 1
                        continue
                    if fixed is not sp:
                        rep.repaired += 1
                    key = (fixed.start, fixed.end, fixed.label_canonical)
                    cur = best[doc.doc_id].get(key)
                    if cur is None or fixed.score > cur.score:
                        best[doc.doc_id][key] = fixed
        preds = {did: sorted(v.values(), key=lambda s: (s.start, s.end, s.label_canonical)) for did, v in best.items()}
        rep.kept = sum(len(v) for v in preds.values())
        if rep.dropped_unknown_label:
            raise ValueError(f"{self.system}: backend returned labels that were not queried "
                             f"{rep.unknown_labels}; the label configuration is wrong")
        return preds, rep
