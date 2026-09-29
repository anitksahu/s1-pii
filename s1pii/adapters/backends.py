"""Model backends for the GLiNER families. Imported lazily inside each baseline's own
environment (``scripts/make_env.sh``), so ``gliner`` and ``gliner2`` never share a venv.

* Weights are fetched with ``snapshot_download`` at an explicit revision (ONNX duplicates
  skipped); the resolved commit SHA is ``backend.revision`` and is part of the cache key.
* ``fits(text, labels)`` checks the window plus label prompt against the model's context
  limits, so the adapter splits windows instead of letting the model truncate silently.
* The zero-shot description protocol is verified once at init; if the installed package
  cannot take descriptions, construction fails rather than silently changing the protocol.
"""
from __future__ import annotations

import re
from contextlib import nullcontext
from typing import Sequence

from .base import RawEnt, normalize_gliner, normalize_gliner2

_TOK = re.compile(r"\w+(?:[-_]\w+)*|\S")
SAFETY = 0.9


def _snapshot(model_id: str, revision: str | None) -> tuple[str, str]:
    from huggingface_hub import snapshot_download, HfApi
    sha = HfApi().model_info(model_id, revision=revision).sha
    path = snapshot_download(model_id, revision=sha, ignore_patterns=["*.onnx", "onnx/*", "*.msgpack", "*.h5"])
    return path, sha


def _torch():
    try:
        import torch
        return torch
    except ImportError:
        return None


def _device() -> str:
    t = _torch()
    return "cuda" if t is not None and t.cuda.is_available() else "cpu"


def _infer_ctx():
    t = _torch()
    return t.inference_mode() if t is not None else nullcontext()


def _versions(**pkgs) -> dict:
    t = _torch()
    return {**pkgs, "torch": getattr(t, "__version__", None),
            "cuda": getattr(getattr(t, "version", None), "cuda", None) if t else None}


def _find_tokenizer(model):
    for path in ("data_processor.transformer_tokenizer", "tokenizer", "processor.tokenizer",
                 "data_processor.tokenizer"):
        obj = model
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "encode"):
            return obj
    return None


class _Fits:
    """Context check: GLiNER word-tokens (text + labels) <= max_words and subword tokens
    (text + labels) <= max_subwords, both with a safety margin."""

    def _init_limits(self, max_words: int | None, max_subwords: int | None):
        self.max_words = max_words
        self.max_subwords = max_subwords
        self.tokenizer = _find_tokenizer(self.model)

    def prompt_text(self, labels: list[str]) -> str:
        """Everything sent alongside the window: label names plus descriptions if used."""
        q = self._query(labels) if hasattr(self, "_query") else labels
        return " ".join(f"{k} {v}" for k, v in q.items()) if isinstance(q, dict) else " ".join(q)

    def fits(self, text: str, labels: list[str]) -> bool:
        prompt = self.prompt_text(labels)
        words = len(_TOK.findall(text)) + len(_TOK.findall(prompt)) + len(labels)
        if self.max_words and words > SAFETY * self.max_words:
            return False
        if self.tokenizer is not None and self.max_subwords:
            n = len(self.tokenizer.encode(text + " " + prompt, add_special_tokens=True))
            if n > SAFETY * self.max_subwords:
                return False
        return True


class GlinerBackend(_Fits):
    """``gliner`` package (NVIDIA GLiNER-PII). Runs with ``flat_ner=False`` so overlapping
    candidates are kept, matching S1, which emits every overlapping span marginal (prereg)."""
    descriptions = None

    def __init__(self, model_id: str, revision: str | None = None):
        from gliner import GLiNER
        import gliner
        path, self.revision = _snapshot(model_id, revision)
        self.model = GLiNER.from_pretrained(path)
        self.device = _device()
        self.model.to(self.device)
        self.model.eval()
        cfg = getattr(self.model, "config", None)
        self._init_limits(getattr(cfg, "max_len", 384), getattr(cfg, "max_length", None) or 512)
        self.versions = _versions(gliner=getattr(gliner, "__version__", "unknown"))

    def predict(self, texts: Sequence[str], labels: list[str], threshold: float) -> list[list[RawEnt]]:
        m = self.model
        with _infer_ctx():
            if hasattr(m, "inference"):
                outs = m.inference(list(texts), labels, threshold=threshold, flat_ner=False)
            elif hasattr(m, "batch_predict_entities"):
                outs = m.batch_predict_entities(list(texts), labels, threshold=threshold, flat_ner=False)
            else:
                outs = [m.predict_entities(t, labels, threshold=threshold, flat_ner=False) for t in texts]
        return [normalize_gliner(o) for o in outs]


class Gliner2Backend(_Fits):
    """``gliner2`` package (GLiNER2-PII and GLiNER2.5). ``descriptions`` maps canonical query
    labels to natural-language descriptions; native labels are described by their name."""

    def __init__(self, model_id: str, revision: str | None = None, descriptions: dict[str, str] | None = None):
        from gliner2 import GLiNER2
        import gliner2
        path, self.revision = _snapshot(model_id, revision)
        self.model = GLiNER2.from_pretrained(path)
        self.device = _device()
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()
        self.descriptions = descriptions
        cfg = getattr(self.model, "config", None)
        self._init_limits(getattr(cfg, "max_len", None), getattr(cfg, "max_length", None) or 512)
        self.versions = _versions(gliner2=getattr(gliner2, "__version__", "unknown"))
        if self.descriptions:
            try:   # verify once that descriptions are accepted; fail loudly otherwise
                with _infer_ctx():
                    self.model.extract_entities("Contact Jane at jane@example.com.", {"email address": "an email"},
                                                threshold=0.5, include_confidence=True, include_spans=True)
            except TypeError as e:
                raise RuntimeError("installed gliner2 does not accept label descriptions; the preregistered "
                                   "zero-shot protocol cannot run with this version") from e

    def _query(self, labels: list[str]):
        if not self.descriptions:
            return labels
        return {l: self.descriptions.get(l, l.replace("_", " ")) for l in labels}

    def predict(self, texts: Sequence[str], labels: list[str], threshold: float) -> list[list[RawEnt]]:
        m, q = self.model, self._query(labels)
        with _infer_ctx():
            if hasattr(m, "batch_extract_entities"):
                try:
                    outs = m.batch_extract_entities(list(texts), q, threshold=threshold,
                                                    include_confidence=True, include_spans=True)
                except TypeError:
                    outs = [m.extract_entities(t, q, threshold=threshold, include_confidence=True,
                                               include_spans=True) for t in texts]
            else:
                outs = [m.extract_entities(t, q, threshold=threshold, include_confidence=True,
                                           include_spans=True) for t in texts]
        return [normalize_gliner2(o) for o in outs]


def make_backend(system: str):
    from .base import load_config, label_descriptions
    cfg = load_config()["systems"][system]
    if cfg["backend"] == "gliner":
        return GlinerBackend(cfg["model_id"], cfg.get("revision"))
    if cfg["backend"] == "gliner2":
        return Gliner2Backend(cfg["model_id"], cfg.get("revision"), label_descriptions(system))
    raise ValueError(f"unknown backend {cfg['backend']!r}")
