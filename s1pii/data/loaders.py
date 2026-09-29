"""Dataset loaders.

Two stages, so the census can inspect data before any mapping decision:

1. ``raw_<dataset>()`` yields ``RawDoc`` records: text plus spans with *raw* labels, built
   from the source format with all offset arithmetic done here.
2. ``build()`` maps raw labels through ``s1pii.taxonomy`` (fail-closed), validates the offset
   contract per record, quarantines bad records up to a reject budget, and returns ``Doc``s.

``load()`` adds the licence gate and a canonical JSONL snapshot: the first load writes
``$S1PII_DATA/snapshots/<name>-<split>.jsonl`` plus a ``.meta.json`` (source revision,
file hashes, dataset hash, library versions); later loads read the snapshot, so every run
scores byte-identical data. Heavy dependencies are imported lazily.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import random
import tempfile
import urllib.request
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable, Iterator

from ..schema import Doc, Span, OffsetError, validate_doc, IGNORE, read_jsonl, write_jsonl
from .. import taxonomy as tx
from .manifest import require_allowed, entry

DATA_DIR = Path(os.environ.get("S1PII_DATA", Path.home() / ".cache" / "s1pii"))

TAB_URL = "https://raw.githubusercontent.com/NorskRegnesentral/text-anonymization-benchmark/master/echr_{split}.json"


class SchemaError(ValueError):
    pass


class RejectBudgetExceeded(ValueError):
    pass


@dataclass
class RawDoc:
    doc_id: str
    text: str
    spans: list[dict]                      # {start, end, label, surface?, source?}
    cluster_id: str = ""
    meta: dict = field(default_factory=dict)
    ignore_ranges: list[tuple[int, int]] = field(default_factory=list)  # e.g. role prefixes


# ------------------------------------------------------------------ helpers

def rows(x) -> list[dict]:
    """Normalize a HF nested feature: a dict of lists becomes a list of dicts."""
    if x is None:
        return []
    if isinstance(x, dict):
        keys = list(x)
        n = len(x[keys[0]]) if keys else 0
        if any(len(x[k]) != n for k in keys):
            raise SchemaError(f"ragged nested feature: {[(k, len(x[k])) for k in keys]}")
        return [{k: x[k][i] for k in keys} for i in range(n)]
    return list(x)


def parse_spans(value, where: str) -> list[dict]:
    if isinstance(value, str):
        try:
            return rows(json.loads(value))
        except json.JSONDecodeError:
            try:
                return rows(ast.literal_eval(value))
            except (ValueError, SyntaxError) as e:
                raise SchemaError(f"{where}: spans field is neither JSON nor a Python literal") from e
    return rows(value)


def _require(record: dict, keys: Iterable[str], where: str) -> None:
    missing = [k for k in keys if k not in record]
    if missing:
        raise SchemaError(f"{where}: missing fields {missing}; record has {sorted(record)}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_download(url: str, dest: Path, expected_sha256: str | None = None) -> Path:
    if dest.exists() and (expected_sha256 is None or sha256_file(dest) == expected_sha256):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dest.parent, suffix=".part")
    os.close(fd)
    try:
        urllib.request.urlretrieve(url, tmp)
        got = sha256_file(Path(tmp))
        if expected_sha256 and got != expected_sha256:
            raise SchemaError(f"{url}: sha256 {got} != pinned {expected_sha256}; upstream changed")
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return dest


def _hf_load(repo: str, split: str, config: str | None, revision: str | None):
    try:
        import datasets
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install 's1pii[data]'") from e
    kw = {"split": split}
    if revision:
        kw["revision"] = revision
    return datasets.load_dataset(repo, config, **kw) if config else datasets.load_dataset(repo, **kw)


def _hf_revision(repo: str, revision: str | None) -> str | None:
    try:
        from huggingface_hub import HfApi
        return HfApi().dataset_info(repo, revision=revision).sha
    except Exception:
        return revision


# ------------------------------------------------------------------ raw readers

def raw_tab(records: list[dict], split: str) -> Iterator[RawDoc]:
    for i, r in enumerate(records):
        _require(r, ["doc_id", "text", "annotations"], f"tab[{i}]")
        spans = []
        for ann_name, ann in r["annotations"].items():
            for m in rows(ann.get("entity_mentions", [])):
                spans.append({"start": m["start_offset"], "end": m["end_offset"],
                              "label": f'{m["entity_type"]}/{m["identifier_type"]}',
                              "surface": m.get("span_text"), "source": f"gold:{ann_name}"})
        yield RawDoc(r["doc_id"], r["text"], spans, cluster_id=r["doc_id"],
                     meta={"task": r.get("task", ""), "n_annotators": len(r["annotations"])})


SPY_FILES = {"legal": "data/legal_questions_placeholders.jsonl",
             "medical": "data/medical_consultations_placeholders.jsonl"}
SPY_TYPES = ("EMAIL", "ID_NUM", "NAME", "PHONE_NUM", "ADDRESS", "URL", "USERNAME")


def spy_faker_fill(records: list[dict], seed: int = 0) -> list[dict]:
    """Deterministic re-implementation of SPY.py: each B- placeholder token is replaced by
    the words of a Faker entity of that type, one generated profile per record. Unlike the
    original (which shuffles with the global unseeded ``random``), every draw here comes from
    seeded generators, so the filled corpus is identical across runs for a given Faker version."""
    from faker import Faker
    fk = Faker("en_US")
    fk.seed_instance(seed)
    rng = random.Random(seed)
    funcs = {
        "EMAIL": [fk.ascii_email, fk.ascii_free_email], "NAME": [fk.name],
        "URL": [lambda: fk.uri(deep=1), lambda: fk.uri(deep=2)], "PHONE_NUM": [fk.phone_number],
        "ID_NUM": [fk.ripe_id, fk.msisdn, fk.ssn, fk.sbn9, fk.isbn10, fk.isbn13,
                   fk.credit_card_number, fk.aba, fk.bban, fk.iban],
        "ADDRESS": [fk.street_address], "USERNAME": [fk.user_name],
    }
    out = []
    for r in records:
        ws_key = "trailing_whitespaces" if "trailing_whitespaces" in r else "trailing_whitespace"
        _require(r, ["tokens", "ent_tags", ws_key], "spy")
        profile = {t: str(rng.choice(fs)()).split() for t, fs in funcs.items()}
        toks, tags, wss = [], [], []
        for tok, tag, ws in zip(r["tokens"], r["ent_tags"], r[ws_key]):
            if tag.startswith("B-") and tag[2:] in profile:
                words = profile[tag[2:]]
                for i, w in enumerate(words):
                    toks.append(w); tags.append(("B-" if i == 0 else "I-") + tag[2:])
                    wss.append(True if i < len(words) - 1 else bool(ws))
            else:
                toks.append(tok); tags.append(tag); wss.append(bool(ws))
        out.append({"tokens": toks, "ent_tags": tags, "trailing_whitespace": wss})
    return out


def raw_spy_tokens(records: Iterable[dict], domain: str) -> Iterator[RawDoc]:
    for i, r in enumerate(records):
        ws_key = "trailing_whitespace" if "trailing_whitespace" in r else "trailing_whitespaces"
        _require(r, ["tokens", "ent_tags", ws_key], f"spy[{i}]")
        toks, ws, tags = r["tokens"], r[ws_key], r["ent_tags"]
        if not (len(toks) == len(ws) == len(tags)):
            raise SchemaError(f"spy[{i}]: tokens/whitespace/tags length mismatch")
        if tags and not isinstance(tags[0], str):
            raise SchemaError(f"spy[{i}]: ent_tags must be strings (map ClassLabel ids first)")
        text, starts, pos = [], [], 0
        for t, w in zip(toks, ws):
            starts.append(pos)
            text.append(t + (" " if w else ""))
            pos += len(t) + (1 if w else 0)
        text = "".join(text)
        spans, cur = [], None
        for j, (t, g) in enumerate(zip(toks, tags)):
            typ = g[2:] if g[:2] in ("B-", "I-") else None
            if g.startswith("B-") or (g.startswith("I-") and (cur is None or cur["label"] != typ)):
                if cur:
                    spans.append(cur)
                cur = {"start": starts[j], "end": starts[j] + len(t), "label": typ}
            elif g.startswith("I-"):
                cur["end"] = starts[j] + len(t)
            else:
                if cur:
                    spans.append(cur)
                cur = None
        if cur:
            spans.append(cur)
        for s in spans:
            s["surface"] = text[s["start"]:s["end"]]
            if "{" in s["surface"] and "}" in s["surface"]:
                raise SchemaError(f"spy[{i}]: unfilled placeholder {s['surface']!r}")
        did = f"spy_{domain}_{i}"
        yield RawDoc(did, text, spans, cluster_id=did)


def raw_pii_trace(records: Iterable[dict]) -> Iterator[RawDoc]:
    for i, r in enumerate(records):
        _require(r, ["id", "turns", "spans"], f"pii_trace[{i}]")
        did = f'pii_trace_{r["id"]}'
        parts, base, ignore, pos = [], {}, [], 0
        for t in sorted(rows(r["turns"]), key=lambda x: int(x["turn"])):
            for role, prefix in (("user", "User: "), ("assistant", "Assistant: ")):
                msg = t.get(role)
                if msg is None:
                    continue
                parts.append(prefix)
                ignore.append((pos, pos + len(prefix)))       # scaffolding, not source text
                pos += len(prefix)
                base[(int(t["turn"]), role)] = pos
                parts.append(msg + "\n")
                pos += len(msg) + 1
        text = "".join(parts)
        spans = []
        for s in rows(r["spans"]):
            key = (int(s["turn"]), s["source"])
            if key not in base:
                raise SchemaError(f"{did}: span refers to missing turn/source {key}")
            spans.append({"start": base[key] + int(s["start"]), "end": base[key] + int(s["end"]),
                          "label": s["label"], "surface": s.get("text")})
        yield RawDoc(did, text, spans, cluster_id=did, ignore_ranges=ignore)


def raw_json_spans(records: Iterable[dict], dataset: str, *, text_key: str, spans_key: str,
                   id_fn: Callable[[dict, int], str], cluster_fn: Callable[[dict], str] | None,
                   meta_keys: tuple[str, ...] = (), casefold_surface: bool = False) -> Iterator[RawDoc]:
    """``casefold_surface``: the source's span ``text`` field is case-normalized for some labels
    (Nemotron-PII lowercases e.g. ``Black`` -> ``black``) while offsets are right. A surface that
    equals the offset slice up to case is replaced by the slice; any other mismatch is kept and
    rejected by the build step."""
    for i, r in enumerate(records):
        _require(r, [text_key, spans_key], f"{dataset}[{i}]")
        did = f"{dataset}_{id_fn(r, i)}"
        text = r[text_key]
        spans = []
        for s in parse_spans(r[spans_key], f"{dataset}[{i}]"):
            _require(s, ["start", "end", "label"], f"{dataset}[{i}].span")
            surf = s.get("text", s.get("value"))
            if surf is not None:
                surf = str(surf)
                seg = text[int(s["start"]):int(s["end"])] if isinstance(text, str) else None
                if casefold_surface and seg is not None and seg != surf and seg.casefold() == surf.casefold():
                    surf = seg
            spans.append({"start": s["start"], "end": s["end"], "label": s["label"], "surface": surf})
        cl = cluster_fn(r) if cluster_fn else None
        yield RawDoc(did, text, spans, cluster_id=f"{dataset}:{cl}" if cl else did,
                     meta={k: r.get(k) for k in meta_keys})


def _text_id(r: dict, i: int, key: str) -> str:
    return hashlib.sha256(r[key].encode()).hexdigest()[:16]


# ------------------------------------------------------------------ build (map + validate)

def build(raws: Iterable[RawDoc], dataset: str, split: str, mapper: Callable[[str], str], *,
          strict: bool = True, max_reject_rate: float = 0.001) -> tuple[list[Doc], dict]:
    docs, rejects, unmapped, seen = [], [], {}, set()
    n = 0
    for rd in raws:
        n += 1
        try:
            spans = []
            for s in rd.spans:
                try:
                    lab = mapper(s["label"])
                except tx.UnmappedLabelError:
                    unmapped[s["label"]] = unmapped.get(s["label"], 0) + 1
                    continue
                spans.append(Span(doc_id=rd.doc_id, start=s["start"], end=s["end"], label_canonical=lab,
                                  label_raw=str(s["label"]), source=s.get("source", "gold"),
                                  surface=s.get("surface")))
            spans += [Span(doc_id=rd.doc_id, start=a, end=b, label_canonical=IGNORE,
                           label_raw="_scaffold", source="loader") for a, b in rd.ignore_ranges]
            doc = _dedupe(Doc(rd.doc_id, rd.text, tuple(spans), dataset, split, rd.cluster_id or rd.doc_id, rd.meta))
            validate_doc(doc)
            if doc.doc_id in seen:
                raise SchemaError(f"duplicate doc_id {doc.doc_id!r}")
            seen.add(doc.doc_id)
            docs.append(doc)
        except (OffsetError, SchemaError, KeyError, TypeError, ValueError) as e:
            rejects.append({"doc_id": rd.doc_id, "error": f"{type(e).__name__}: {e}"[:300]})
    if unmapped:
        raise tx.UnmappedLabelError(dataset, unmapped)
    report = {"dataset": dataset, "split": split, "records": n, "docs": len(docs),
              "rejected": len(rejects), "rejects": rejects[:50]}
    if rejects and (strict and len(rejects) / max(n, 1) > max_reject_rate):
        raise RejectBudgetExceeded(f"{dataset}/{split}: {len(rejects)}/{n} records rejected "
                                   f"(budget {max_reject_rate:.2%}); first: {rejects[:3]}")
    return docs, report


def _dedupe(doc: Doc) -> Doc:
    """Drop exact duplicate gold spans (same offsets and canonical label), e.g. one mention
    annotated by several TAB annotators. Overlapping spans with different offsets are kept."""
    seen, keep = set(), []
    for s in doc.spans:
        k = (s.start, s.end, s.label_canonical)
        if k not in seen:
            seen.add(k); keep.append(s)
    return doc if len(keep) == len(doc.spans) else replace(doc, spans=tuple(keep))


# ------------------------------------------------------------------ dataset recipes

def _tab(split: str, tier: str):
    e = entry(f"tab_{tier}")
    path = atomic_download(TAB_URL.format(split=split), DATA_DIR / "raw" / f"echr_{split}.json",
                           (e.get("sha256") or {}).get(split))
    records = json.loads(path.read_text())
    return raw_tab(records, split), (lambda raw: tx.map_tab(*raw.split("/"), tier=tier)), \
        {"source_sha256": sha256_file(path)}


def _spy(domain: str, seed: int = 0):
    from huggingface_hub import hf_hub_download
    import faker
    e = entry(f"spy_{domain}")
    path = hf_hub_download("mks-logic/SPY", SPY_FILES[domain], repo_type="dataset", revision=e.get("revision"))
    recs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    filled = spy_faker_fill(recs, seed)
    return raw_spy_tokens(filled, domain), (lambda raw: tx.map_label("spy", raw)), \
        {"source_sha256": sha256_file(Path(path)), "faker_version": faker.VERSION, "faker_seed": seed}


def _pii_trace():
    e = entry("pii_trace")
    ds = _hf_load("perplexity-ai/PII-TRACE", "train", "conversations", e.get("revision"))
    return raw_pii_trace(ds), (lambda raw: tx.map_label("pii_trace", raw)), \
        {"hf_revision": _hf_revision("perplexity-ai/PII-TRACE", e.get("revision"))}


def _nemotron(split: str):
    e = entry("nemotron")
    ds = _hf_load("nvidia/Nemotron-PII", split, None, e.get("revision"))
    raws = raw_json_spans(ds, "nemotron", text_key="text", spans_key="spans",
                          # every uid ships twice (one document per locale, different text):
                          # the id carries the locale, and meta["uid"] groups the twins
                          id_fn=lambda r, i: f'{r.get("uid") or _text_id(r, i, "text")}-{r.get("locale")}',
                          cluster_fn=lambda r: f'{r.get("domain")}|{r.get("document_type")}',
                          meta_keys=("uid", "domain", "document_type", "document_format", "locale"),
                          casefold_surface=True)
    return raws, (lambda raw: tx.map_label("nemotron", raw)), \
        {"hf_revision": _hf_revision("nvidia/Nemotron-PII", e.get("revision"))}


def _gretel(split: str, language: str = "English"):
    e = entry("gretel")
    ds = _hf_load("gretelai/synthetic_pii_finance_multilingual", split, None, e.get("revision"))
    recs = (r for r in ds if r.get("language") == language)
    raws = raw_json_spans(recs, "gretel", text_key="generated_text", spans_key="pii_spans",
                          id_fn=lambda r, i: str(r.get("index", _text_id(r, i, "generated_text"))),
                          cluster_fn=lambda r: f'{r.get("document_type")}|{r.get("expanded_type")}',
                          meta_keys=("document_type", "language"))
    return raws, (lambda raw: tx.map_label("gretel", raw)), \
        {"hf_revision": _hf_revision("gretelai/synthetic_pii_finance_multilingual", e.get("revision"))}


def _ai4privacy(split: str, language: str = "en"):
    e = entry("ai4privacy")
    ds = _hf_load("ai4privacy/pii-masking-400k", split, None, e.get("revision"))
    recs = (r for r in ds if r.get("language") == language)
    raws = raw_json_spans(recs, "ai4privacy", text_key="source_text", spans_key="privacy_mask",
                          id_fn=lambda r, i: str(r.get("id") or _text_id(r, i, "source_text")),
                          cluster_fn=None, meta_keys=("language",))
    return raws, (lambda raw: tx.map_label("ai4privacy", raw)), \
        {"hf_revision": _hf_revision("ai4privacy/pii-masking-400k", e.get("revision"))}


def _fresh_real(path: str | None = None):
    path = Path(path or os.environ.get("S1PII_FRESH_REAL", DATA_DIR / "fresh_real.jsonl"))
    recs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    raws = (RawDoc(r["doc_id"], r["text"], [{**s, "surface": r["text"][s["start"]:s["end"]]} for s in r["spans"]],
                   cluster_id=r.get("cluster_id") or r["doc_id"],
                   meta={k: r.get(k) for k in ("source", "published_date")}) for r in recs)
    return raws, (lambda raw: tx.map_label("fresh_real", raw)), {"source_sha256": sha256_file(path)}


# name -> (recipe(split) -> (raws, mapper, provenance), default split, splits available)
RECIPES: dict[str, tuple[Callable, str]] = {
    "tab_direct": (lambda split: _tab(split, "direct"), "test"),
    "tab_quasi": (lambda split: _tab(split, "quasi"), "test"),
    "spy_medical": (lambda split: _spy("medical"), "all"),
    "spy_legal": (lambda split: _spy("legal"), "all"),
    "pii_trace": (lambda split: _pii_trace(), "all"),
    "nemotron": (lambda split: _nemotron(split), "test"),
    "gretel": (lambda split: _gretel(split), "test"),
    "ai4privacy": (lambda split: _ai4privacy(split), "validation"),
    "fresh_real": (lambda split: _fresh_real(), "all"),
}


def snapshot_paths(name: str, split: str) -> tuple[Path, Path]:
    base = DATA_DIR / "snapshots" / f"{name}-{split}"
    return base.with_suffix(".jsonl"), base.with_suffix(".meta.json")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _valid_snapshot(snap: Path, meta_path: Path) -> list[Doc] | None:
    """Return the snapshot's docs only if its meta exists (written last), matches the current
    taxonomy version and the recomputed dataset hash; otherwise None (rebuild)."""
    from ..ledger import dataset_hash
    if not (snap.exists() and meta_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text())
        if meta.get("taxonomy") != tx.MAP_VERSION:
            return None
        if hashlib.sha256(snap.read_bytes()).hexdigest() != meta.get("content_sha256"):
            return None
        docs = read_jsonl(snap)
        return docs if dataset_hash(docs) == meta.get("dataset_hash") else None
    except (json.JSONDecodeError, KeyError, TypeError, OffsetError):
        return None


# Census-backed exceptions to the 0.1% reject budget. Gretel finance EN has records whose
# span offsets point past the end of a truncated text (5/2962 test, 68/25948 train in the
# Colab census); those whole records are rejected and listed in the snapshot meta.
REJECT_BUDGET = {"gretel": 0.005}


def load(name: str, split: str | None = None, *, purpose: str = "eval", refresh: bool = False,
         strict: bool = True) -> list[Doc]:
    """Licence-gated, snapshotted load. ``split='all'`` datasets ship one split; carve
    calibration/test with ``calib_test_split``. Snapshots are written atomically (data first,
    meta last) and are reused only if they validate against the meta and taxonomy version."""
    require_allowed(name, purpose=purpose)
    recipe, default = RECIPES[name]
    split = split or default
    snap, meta_path = snapshot_paths(name, split)
    if not refresh:
        cached = _valid_snapshot(snap, meta_path)
        if cached is not None:
            return cached
    raws, mapper, prov = recipe(split)
    docs, report = build(raws, name, split, mapper, strict=strict,
                         max_reject_rate=REJECT_BUDGET.get(name, 0.001))
    from ..ledger import dataset_hash
    if meta_path.exists():
        meta_path.unlink()
    body = "".join(d.to_json() + "\n" for d in docs)
    _atomic_write_text(snap, body)
    meta = {"name": name, "split": split, "taxonomy": tx.MAP_VERSION, "dataset_hash": dataset_hash(docs),
            "content_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(), "report": report, **prov}
    _atomic_write_text(meta_path, json.dumps(meta, indent=2))
    return docs


def snapshot_meta(name: str, split: str) -> dict:
    return json.loads(snapshot_paths(name, split)[1].read_text())


# ------------------------------------------------------------------ splits and sampling

def _h(salt: str, key: str) -> str:
    return hashlib.sha256(f"{salt}|{key}".encode()).hexdigest()


def calib_test_split(docs: list[Doc], calib_frac: float = 0.2, salt: str = "s1pii-calib-v0") -> tuple[list[Doc], list[Doc]]:
    """Hash-fixed split at the cluster level: whole clusters go to calibration/dev or test."""
    clusters = sorted({d.cluster_id for d in docs}, key=lambda c: _h(salt, c))
    k = max(1, int(round(calib_frac * len(clusters))))
    cal = set(clusters[:k])
    return [d for d in docs if d.cluster_id in cal], [d for d in docs if d.cluster_id not in cal]


def sample(docs: list[Doc], n: int | None, salt: str = "s1pii-v0",
           strata: Callable[[Doc], str] | None = None) -> list[Doc]:
    """Deterministic stratified sample: order by sha256(salt|doc_id) within each stratum and
    allocate proportionally (largest remainder)."""
    if n is None or n >= len(docs):
        return sorted(docs, key=lambda d: _h(salt, d.doc_id))
    groups: dict[str, list[Doc]] = {}
    for d in docs:
        groups.setdefault(strata(d) if strata else "_", []).append(d)
    quotas = {k: n * len(v) / len(docs) for k, v in groups.items()}
    alloc = {k: int(q) for k, q in quotas.items()}
    for k in sorted(quotas, key=lambda k: (-(quotas[k] - alloc[k]), k))[:n - sum(alloc.values())]:
        alloc[k] += 1
    out = []
    for k in sorted(groups):
        out.extend(sorted(groups[k], key=lambda d: _h(salt, d.doc_id))[:alloc[k]])
    return out
