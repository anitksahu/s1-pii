"""Conditional stage B: a type-agnostic level-1 CRF (O/B/I/E/S, nt = 1) that also detects
quasi-identifiers, warm-started from the v1 encoder.

Runs only if gate B fires (``gates.py``). Targets per token:
* every gold span with a raw label (PII and quasi-identifier alike) is an entity;
* held-out spans (C3) are ANY, so level 1 is not trained on them;
* per-source type inventory: a teacher span whose families the source does not annotate
  becomes ANY, never O. Teacher (frozen, identical for both variants, never trained on
  Nemotron): spaCy ``en_core_web_lg`` at a pinned version; GPE/LOC/FAC -> location nodes,
  ORG -> organization, DATE -> date, NORP -> attribute. Its name and version and the hash of
  its ANY masks are written into the export manifest.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..schema import Doc
from ..model.encode import Example, token_windows, special_ids, target_is_valid, ANY, _trim
from ..model.crf import tag
from . import labels as LB

TEACHER = {"package": "en_core_web_lg", "version": "3.8.0"}
TEACHER_NODES = {"GPE": {"city", "state", "county", "country"}, "LOC": {"location", "country", "state"},
                 "FAC": {"street_address", "location"}, "ORG": {"organization"}, "DATE": {"date", "date_of_birth"},
                 "NORP": {"race_ethnicity", "religious_belief", "political_view"}}


def source_inventory(docs: list[Doc], source_of: dict[str, str]) -> dict[str, set[str]]:
    inv: dict[str, set[str]] = {}
    for d in docs:
        s = inv.setdefault(source_of[d.doc_id], set())
        for sp in d.spans:
            if sp.label_raw and sp.label_raw != "_scaffold" and sp.label_raw.lower() in LB.NATIVE:
                s.add(LB.node_of(sp.label_raw))
    return inv


def teacher_spans(docs: list[Doc], n_process: int = 4) -> dict[str, list[tuple[int, int, str]]]:
    import spacy
    nlp = spacy.load(TEACHER["package"], disable=["parser", "lemmatizer", "tagger", "attribute_ruler"])
    if nlp.meta.get("version") != TEACHER["version"]:
        raise RuntimeError(f"teacher version {nlp.meta.get('version')} != pinned {TEACHER['version']}")
    out = {}
    for d, doc in zip(docs, nlp.pipe((d.text for d in docs), batch_size=64, n_process=n_process)):
        out[d.doc_id] = [(e.start_char, e.end_char, e.label_) for e in doc.ents if e.label_ in TEACHER_NODES]
    return out


def teacher_for(docs: list[Doc], source_of: dict[str, str], inv: dict[str, set[str]] | None = None) -> dict:
    """Teacher spans only for docs of sources that do not annotate every teacher family."""
    inv = inv or source_inventory(docs, source_of)
    need = {s for s, nodes in inv.items() if any(not (fam <= nodes) for fam in TEACHER_NODES.values())}
    todo = [d for d in docs if source_of[d.doc_id] in need]
    return teacher_spans(todo) if todo else {}


def any_mask_spans(d: Doc, teacher: list[tuple[int, int, str]], inventory: set[str]) -> list[tuple[int, int]]:
    """Teacher spans of families the source does not fully annotate, not overlapping gold."""
    gold = [(s.start, s.end) for s in d.spans if s.label_raw and s.label_raw != "_scaffold"]
    out = []
    for a, b, lab in teacher:
        if TEACHER_NODES[lab] <= inventory:
            continue
        if any(min(b, ge) > max(a, gs) for gs, ge in gold):
            continue
        out.append((a, b))
    return out


def l1_target(d: Doc, tok, heldout: set[str], any_spans: list[tuple[int, int]]):
    enc = tok(d.text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    offs = np.array([_trim(d.text, a, b) for a, b in enc["offset_mapping"]], dtype=np.int64).reshape(-1, 2)
    n = len(enc["input_ids"])
    target = np.zeros(n, dtype=np.int64); span_of = np.full(n, -1, dtype=np.int64)
    st, en = offs[:, 0], offs[:, 1]
    vis = en > st
    for s in d.spans:                                             # scaffold and held-out: ANY
        if s.label_raw == "_scaffold" or (s.label_raw and LB.node_of(s.label_raw) in heldout):
            target[vis & (st < s.end) & (en > s.start)] = ANY
    for a, b in any_spans:
        target[vis & (st < b) & (en > a)] = ANY
    ents = sorted([s for s in d.spans if s.label_raw and s.label_raw != "_scaffold"
                   and LB.node_of(s.label_raw) not in heldout], key=lambda s: (s.start, -s.end))
    for si, s in enumerate(ents):
        hit = np.nonzero(vis & (st < s.end) & (en > s.start) & (span_of < 0) & (target != ANY))[0]
        if len(hit) == 0:
            continue
        hit = np.arange(hit[0], hit[-1] + 1)
        if (span_of[hit] >= 0).any():
            target[hit] = ANY; continue                           # nested/overlapping gold: unknown
        if len(hit) == 1:
            target[hit[0]] = tag("S", 0)
        else:
            target[hit[0]] = tag("B", 0); target[hit[1:-1]] = tag("I", 0); target[hit[-1]] = tag("E", 0)
        span_of[hit] = si
    return list(enc["input_ids"]), target, span_of


def l1_examples(docs: list[Doc], tok, heldout: set[str], any_by_doc: dict[str, list], max_len: int = 1024) -> tuple[list[Example], dict]:
    pre, suf = special_ids(tok)
    size = max_len - len(pre) - len(suf)
    out, bad = [], 0
    for d in docs:
        ids, target, span_of = l1_target(d, tok, heldout, any_by_doc.get(d.doc_id, []))
        if not ids:
            continue
        for a, b in token_windows(len(ids), size, size):
            t = target[a:b].copy(); so = span_of[a:b]
            for edge in {int(so[0]) if a > 0 else -1, int(so[-1]) if b < len(ids) else -1}:
                if edge >= 0:
                    t[so == edge] = ANY
            if not target_is_valid(t, nt=1):
                bad += 1; continue
            out.append(Example(d.doc_id, np.asarray(pre + ids[a:b] + suf, dtype=np.int32), len(pre), b - a, t, a))
    return out, {"invalid_windows": bad}


def train_level1(v1_dir: Path, docs: list[Doc], source_of: dict[str, str], out: Path, heldout: set[str], *,
                 seed: int, epochs: float = 1.0,
                 teacher: dict[str, list] | None = None, mirror: Path | None = None, max_steps: int | None = None,
                 max_len: int = 1024, device: str | None = None, **cfg_kw) -> Path:
    """Warm start from the v1 encoder; new 5-tag emission head and CRF."""
    from ..model.train import TrainConfig, train, load_exported
    from ..model.s1 import S1Model
    v1, tok, man = load_exported(v1_dir)
    model = S1Model(v1.encoder, v1.encoder.config.hidden_size, man["config"].get("dropout", 0.1), num_types=1)
    if hasattr(model.encoder, "gradient_checkpointing_enable"):
        model.encoder.gradient_checkpointing_enable()
    inv = source_inventory(docs, source_of)
    if teacher is None:
        teacher = teacher_for(docs, source_of, inv)
    any_by_doc = {d.doc_id: any_mask_spans(d, teacher.get(d.doc_id, []), inv.get(source_of[d.doc_id], set())) for d in docs}
    ex, rep = l1_examples(docs, tok, heldout, any_by_doc, max_len=max_len)
    mh = hashlib.sha256(json.dumps(sorted((k, v) for k, v in any_by_doc.items() if v)).encode()).hexdigest()[:16]
    cfg = TrainConfig(variant=man["config"]["variant"], seed=seed, epochs=epochs, max_len=max_len, **cfg_kw,
                      backbone=man["config"]["backbone"], backbone_revision=man["config"].get("backbone_revision"),
                      extra={"level1": True, "warm_start": man["weights_sha256"], "teacher": TEACHER,
                             "teacher_any_mask_hash": mh, "heldout_nodes": sorted(heldout),
                             "source_inventory": {k: sorted(v) for k, v in inv.items()}, **rep})
    return train(cfg, out, mirror=mirror, model=model, tokenizer=tok, examples=ex,
                 data_manifest={"source_counts": man.get("data", {}).get("source_counts"), "level1": True},
                 max_steps=max_steps, device=device)
