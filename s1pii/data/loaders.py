"""Dataset loaders. Each returns ``list[Doc]`` with gold spans mapped to the canonical taxonomy.

All loaders are fail-closed: schema surprises and unmapped labels raise with an actionable
message, and every document passes ``validate_doc`` before it is returned. Heavy
dependencies (``datasets``) are imported lazily so the harness and tests run without them.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable

from ..schema import Doc, Span, validate_doc, IGNORE, NOT_PII
from .. import taxonomy as tx
from .manifest import require_allowed

CACHE_DIR = Path(os.environ.get("S1PII_CACHE", Path.home() / ".cache" / "s1pii"))

TAB_URL = "https://raw.githubusercontent.com/NorskRegnesentral/text-anonymization-benchmark/master/echr_{split}.json"


class SchemaError(ValueError):
    pass


def _require(record: dict, keys: Iterable[str], dataset: str) -> None:
    missing = [k for k in keys if k not in record]
    if missing:
        raise SchemaError(f"{dataset}: record missing fields {missing}; has {sorted(record)}")


def _hf_load(repo: str, split: str, config: str | None = None, revision: str | None = None):
    try:
        import datasets
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install datasets (see requirements-colab.txt)") from e
    kwargs = {"split": split}
    if revision:
        kwargs["revision"] = revision
    return datasets.load_dataset(repo, config, **kwargs) if config else datasets.load_dataset(repo, **kwargs)


def _dedupe(doc: Doc) -> Doc:
    """Drop exact duplicate gold spans (same offsets and canonical label), e.g. the same
    mention annotated by several TAB annotators. Overlapping spans with different offsets
    are kept; per-character precedence in the metrics resolves them."""
    seen, keep = set(), []
    for s in doc.spans:
        k = (s.start, s.end, s.label_canonical)
        if k not in seen:
            seen.add(k); keep.append(s)
    return doc if len(keep) == len(doc.spans) else replace(doc, spans=tuple(keep))


def _finish(docs: list[Doc]) -> list[Doc]:
    docs = [_dedupe(d) for d in docs]
    for d in docs:
        validate_doc(d)
    ids = [d.doc_id for d in docs]
    if len(ids) != len(set(ids)):
        raise SchemaError("duplicate doc_id")
    return docs


# ---------------------------------------------------------------- TAB (ECHR)

def tab_from_records(records: list[dict], split: str, tier: str = "direct") -> list[Doc]:
    """Union over annotators; per-character precedence (PII > IGNORE > NOT_PII) is applied
    by the metrics, so overlapping annotator spans are kept as-is."""
    docs = []
    for r in records:
        _require(r, ["doc_id", "text", "annotations"], "tab")
        text, did = r["text"], r["doc_id"]
        spans = []
        for ann_name, ann in r["annotations"].items():
            for m in ann.get("entity_mentions", []):
                lab = tx.map_tab(m["entity_type"], m["identifier_type"], tier)
                spans.append(Span(
                    doc_id=did, start=int(m["start_offset"]), end=int(m["end_offset"]),
                    label_canonical=lab, label_raw=f'{m["entity_type"]}/{m["identifier_type"]}',
                    source=f"gold:{ann_name}", surface=m.get("span_text"),
                ))
        docs.append(Doc(doc_id=did, text=text, spans=tuple(spans), dataset=f"tab_{tier}",
                        split=split, cluster_id=did, meta={"task": r.get("task", "")}))
    return _finish(docs)


def load_tab(split: str = "test", tier: str = "direct") -> list[Doc]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"echr_{split}.json"
    if not path.exists():
        urllib.request.urlretrieve(TAB_URL.format(split=split), path)
    return tab_from_records(json.loads(path.read_text()), split, tier)


# ---------------------------------------------------------------- SPY (token format)

def spy_from_records(records: Iterable[dict], domain: str, split: str) -> list[Doc]:
    docs = []
    for i, r in enumerate(records):
        _require(r, ["tokens", "trailing_whitespace", "ent_tags"], "spy")
        toks, ws, tags = r["tokens"], r["trailing_whitespace"], r["ent_tags"]
        if not (len(toks) == len(ws) == len(tags)):
            raise SchemaError(f"spy record {i}: tokens/whitespace/tags length mismatch")
        pieces, starts, pos = [], [], 0
        for t, w in zip(toks, ws):
            starts.append(pos)
            pieces.append(t + (" " if w else ""))
            pos += len(t) + (1 if w else 0)
        text = "".join(pieces)
        if any(("{" in t and "}" in t) for t, g in zip(toks, tags) if g != "O"):
            raise SchemaError("spy: entity tokens look like unfilled placeholders; load via the "
                              "dataset's Faker-filling loader first")
        did = f"spy_{domain}_{split}_{i}"
        spans, cur = [], None
        for j, (t, g) in enumerate(zip(toks, tags)):
            if g.startswith("B-") or (g.startswith("I-") and (cur is None or cur[2] != g[2:])):
                if cur:
                    spans.append(cur)
                cur = [starts[j], starts[j] + len(t), g[2:]]
            elif g.startswith("I-"):
                cur[1] = starts[j] + len(t)
            else:
                if cur:
                    spans.append(cur)
                cur = None
        if cur:
            spans.append(cur)
        out = tuple(Span(doc_id=did, start=s, end=e, label_canonical=tx.map_label("spy", lab),
                         label_raw=lab, surface=text[s:e]) for s, e, lab in spans)
        docs.append(Doc(doc_id=did, text=text, spans=out, dataset=f"spy_{domain}", split=split,
                        cluster_id=did))
    return _finish(docs)


def load_spy(domain: str, split: str = "train", config: str | None = None) -> list[Doc]:
    """SPY ships one split per domain; it is used only as a test set here."""
    config = config or {"medical": "medical_consultations", "legal": "legal_questions"}[domain]
    try:
        ds = _hf_load("mks-logic/SPY", split, config)
    except Exception as e:
        import datasets
        raise SchemaError(f"SPY config {config!r} failed ({e}); available: "
                          f"{datasets.get_dataset_config_names('mks-logic/SPY')}") from e
    return spy_from_records(ds, domain, "test")


# ---------------------------------------------------------------- PII-TRACE (conversations)

def pii_trace_from_records(records: Iterable[dict], split: str) -> list[Doc]:
    docs = []
    for r in records:
        _require(r, ["id", "turns", "spans"], "pii_trace")
        did = f'pii_trace_{r["id"]}'
        parts, base, pos = [], {}, 0
        for t in sorted(r["turns"], key=lambda x: x["turn"]):
            for role, prefix in (("user", "User: "), ("assistant", "Assistant: ")):
                msg = t.get(role)
                if msg is None:
                    continue
                parts.append(prefix)
                pos += len(prefix)
                base[(int(t["turn"]), role)] = pos
                parts.append(msg + "\n")
                pos += len(msg) + 1
        text = "".join(parts)
        spans = []
        for s in r["spans"]:
            key = (int(s["turn"]), s["source"])
            if key not in base:
                raise SchemaError(f"{did}: span refers to missing turn/source {key}")
            off = base[key]
            spans.append(Span(doc_id=did, start=off + int(s["start"]), end=off + int(s["end"]),
                              label_canonical=tx.map_label("pii_trace", s["label"]),
                              label_raw=s["label"], surface=s.get("text")))
        docs.append(Doc(doc_id=did, text=text, spans=tuple(spans), dataset="pii_trace",
                        split=split, cluster_id=did))
    return _finish(docs)


def load_pii_trace(split: str = "train") -> list[Doc]:
    """The public release has a single split; it is used only as a test set."""
    return pii_trace_from_records(_hf_load("perplexity-ai/PII-TRACE", split, "conversations"), "test")


# ---------------------------------------------------------------- JSON-span datasets

def _json_spans(value) -> list[dict]:
    return json.loads(value) if isinstance(value, str) else list(value)


def spans_dataset_from_records(records: Iterable[dict], dataset: str, split: str, *, text_key: str,
                               spans_key: str, id_key: str | None, cluster_key: str | None,
                               meta_keys: tuple[str, ...] = ()) -> list[Doc]:
    docs, unmapped = [], set()
    for i, r in enumerate(records):
        _require(r, [text_key, spans_key], dataset)
        did = f"{dataset}_{split}_{r[id_key] if id_key else i}"
        text, spans = r[text_key], []
        for s in _json_spans(r[spans_key]):
            raw = s["label"]
            if raw not in tx.MAPS[dataset]:
                unmapped.add(raw)
                continue
            surface = s.get("text", s.get("value"))
            spans.append(Span(doc_id=did, start=int(s["start"]), end=int(s["end"]),
                              label_canonical=tx.MAPS[dataset][raw], label_raw=raw, surface=surface))
        cluster = f"{dataset}:{r[cluster_key]}" if cluster_key and r.get(cluster_key) else did
        docs.append(Doc(doc_id=did, text=text, spans=tuple(spans), dataset=dataset, split=split,
                        cluster_id=cluster, meta={k: r.get(k) for k in meta_keys}))
    if unmapped:
        raise tx.UnmappedLabelError(dataset, unmapped)
    return _finish(docs)


def load_nemotron(split: str = "test") -> list[Doc]:
    ds = _hf_load("nvidia/Nemotron-PII", split)
    # Documents generated from the same document_type/description share a template family.
    recs = ({**r, "_family": f'{r.get("domain")}|{r.get("document_type")}'} for r in ds)
    return spans_dataset_from_records(recs, "nemotron", split, text_key="text", spans_key="spans",
                                      id_key="uid", cluster_key="_family",
                                      meta_keys=("domain", "document_type", "document_format", "locale"))


def load_gretel(split: str = "test", language: str | None = "English") -> list[Doc]:
    ds = _hf_load("gretelai/synthetic_pii_finance_multilingual", split)
    recs = (r for r in ds if language is None or r.get("language") == language)
    recs = ({**r, "_family": f'{r.get("document_type")}|{r.get("expanded_type")}'} for r in recs)
    return spans_dataset_from_records(recs, "gretel", split, text_key="generated_text",
                                      spans_key="pii_spans", id_key="index", cluster_key="_family",
                                      meta_keys=("document_type", "language"))


def load_ai4privacy(split: str = "validation", language: str | None = "en") -> list[Doc]:
    require_allowed("ai4privacy", purpose="research")
    ds = _hf_load("ai4privacy/pii-masking-400k", split)
    recs = (r for r in ds if language is None or r.get("language") == language)
    return spans_dataset_from_records(recs, "ai4privacy", split, text_key="source_text",
                                      spans_key="privacy_mask", id_key=None, cluster_key=None,
                                      meta_keys=("language",))


def load_fresh_real(path: str | Path) -> list[Doc]:
    """Our own annotated set, canonical JSONL: {doc_id, text, spans:[{start,end,label}], cluster_id}."""
    docs = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        _require(r, ["doc_id", "text", "spans"], "fresh_real")
        spans = tuple(Span(doc_id=r["doc_id"], start=int(s["start"]), end=int(s["end"]),
                           label_canonical=tx.map_label("fresh_real", s["label"]), label_raw=s["label"],
                           surface=r["text"][int(s["start"]):int(s["end"])]) for s in r["spans"])
        docs.append(Doc(doc_id=r["doc_id"], text=r["text"], spans=spans, dataset="fresh_real",
                        split="test", cluster_id=r.get("cluster_id") or r["doc_id"],
                        meta={k: r.get(k) for k in ("source", "published_date")}))
    return _finish(docs)


# ---------------------------------------------------------------- registry and sampling

LOADERS: dict[str, Callable[..., list[Doc]]] = {
    "tab_direct": lambda split="test": load_tab(split, "direct"),
    "tab_quasi": lambda split="test": load_tab(split, "quasi"),
    "spy_medical": lambda split="test": load_spy("medical"),
    "spy_legal": lambda split="test": load_spy("legal"),
    "pii_trace": lambda split="test": load_pii_trace(),
    "nemotron": lambda split="test": load_nemotron(split),
    "gretel": lambda split="test": load_gretel(split),
    "ai4privacy": lambda split="validation": load_ai4privacy(split),
}


def load(name: str, split: str = "test", *, purpose: str = "eval") -> list[Doc]:
    require_allowed(name, purpose=purpose)
    return LOADERS[name](split=split)


def _h(salt: str, key: str) -> str:
    return hashlib.sha256(f"{salt}|{key}".encode()).hexdigest()


def sample(docs: list[Doc], n: int | None, salt: str = "s1pii-v0",
           strata: Callable[[Doc], str] | None = None) -> list[Doc]:
    """Deterministic sample: order by sha256(salt|doc_id) within each stratum, allocate
    proportionally (largest remainder). Sampling whole clusters is done upstream by passing
    cluster-level strata when needed."""
    if n is None or n >= len(docs):
        return sorted(docs, key=lambda d: _h(salt, d.doc_id))
    groups: dict[str, list[Doc]] = {}
    for d in docs:
        groups.setdefault(strata(d) if strata else "_", []).append(d)
    total = len(docs)
    quotas = {k: n * len(v) / total for k, v in groups.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    rest = n - sum(alloc.values())
    for k in sorted(quotas, key=lambda k: (-(quotas[k] - alloc[k]), k))[:rest]:
        alloc[k] += 1
    out = []
    for k in sorted(groups):
        out.extend(sorted(groups[k], key=lambda d: _h(salt, d.doc_id))[:alloc[k]])
    return out
