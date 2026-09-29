"""Model backends for the GLiNER families. Imported lazily inside each baseline's own
environment (``scripts/make_env.sh``), so ``gliner`` and ``gliner2`` never share a venv.

Weights are fetched with ``snapshot_download`` at an explicit revision; the resolved commit
SHA is exposed as ``backend.revision`` and recorded in the prediction cache meta.
"""
from __future__ import annotations

import os
from typing import Sequence

from .base import RawEnt, normalize_gliner, normalize_gliner2


def _snapshot(model_id: str, revision: str | None) -> tuple[str, str]:
    from huggingface_hub import snapshot_download, HfApi
    sha = HfApi().model_info(model_id, revision=revision).sha
    path = snapshot_download(model_id, revision=sha)
    return path, sha


def _device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class GlinerBackend:
    """``gliner`` package (NVIDIA GLiNER-PII)."""

    def __init__(self, model_id: str, revision: str | None = None):
        from gliner import GLiNER
        import gliner
        path, self.revision = _snapshot(model_id, revision)
        self.model = GLiNER.from_pretrained(path, local_files_only=True)
        self.device = _device()
        self.model.to(self.device)
        self.model.eval()
        self.versions = {"gliner": getattr(gliner, "__version__", "unknown")}

    def predict(self, texts: Sequence[str], labels: list[str], threshold: float) -> list[list[RawEnt]]:
        m = self.model
        if hasattr(m, "inference"):
            outs = m.inference(list(texts), labels, threshold=threshold, flat_ner=True)
        elif hasattr(m, "batch_predict_entities"):
            outs = m.batch_predict_entities(list(texts), labels, threshold=threshold, flat_ner=True)
        else:
            outs = [m.predict_entities(t, labels, threshold=threshold, flat_ner=True) for t in texts]
        return [normalize_gliner(o) for o in outs]


class Gliner2Backend:
    """``gliner2`` package (GLiNER2-PII and GLiNER2.5). ``descriptions`` maps query labels to
    natural-language descriptions for zero-shot use."""

    def __init__(self, model_id: str, revision: str | None = None, descriptions: dict[str, str] | None = None):
        from gliner2 import GLiNER2
        import gliner2
        path, self.revision = _snapshot(model_id, revision)
        self.model = GLiNER2.from_pretrained(path)
        self.device = _device()
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        self.descriptions = descriptions
        self.versions = {"gliner2": getattr(gliner2, "__version__", "unknown")}

    def _one(self, text: str, labels: list[str], threshold: float):
        query = {l: self.descriptions[l] for l in labels} if self.descriptions else labels
        try:
            return self.model.extract_entities(text, query, threshold=threshold,
                                               include_confidence=True, include_spans=True)
        except TypeError:
            if isinstance(query, dict):     # older API without descriptions
                return self.model.extract_entities(text, labels, threshold=threshold,
                                                   include_confidence=True, include_spans=True)
            raise

    def predict(self, texts: Sequence[str], labels: list[str], threshold: float) -> list[list[RawEnt]]:
        m = self.model
        if hasattr(m, "batch_extract_entities") and not self.descriptions:
            outs = m.batch_extract_entities(list(texts), labels, threshold=threshold,
                                            include_confidence=True, include_spans=True)
        else:
            outs = [self._one(t, labels, threshold) for t in texts]
        return [normalize_gliner2(o) for o in outs]


def make_backend(system: str):
    from .base import load_config, label_descriptions
    cfg = load_config()["systems"][system]
    if cfg["backend"] == "gliner":
        return GlinerBackend(cfg["model_id"], cfg.get("revision"))
    if cfg["backend"] == "gliner2":
        return Gliner2Backend(cfg["model_id"], cfg.get("revision"), label_descriptions(system))
    raise ValueError(f"unknown backend {cfg['backend']!r}")
