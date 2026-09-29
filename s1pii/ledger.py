"""Append-only results ledger and prediction cache.

Every ledger row carries git SHA, harness version, seed and the full config hash. The
ledger is JSONL (append is atomic per line); ``export_parquet`` produces the table view.
Predictions are cached as JSONL keyed by (system, revision, adapter version, dataset hash,
config hash) so metrics can be recomputed without a GPU.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

from . import __version__
from .schema import Doc, Span

RESULTS = Path(os.environ.get("S1PII_RESULTS", "results"))


def git_sha() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.call(["git", "diff", "--quiet"], stderr=subprocess.DEVNULL) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def stable_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def dataset_hash(docs: Sequence[Doc]) -> str:
    h = hashlib.sha256()
    for d in sorted(docs, key=lambda d: d.doc_id):
        h.update(d.doc_id.encode()); h.update(hashlib.sha256(d.text.encode()).digest())
    return h.hexdigest()[:16]


def append(row: dict, path: Path | None = None) -> dict:
    path = path or RESULTS / "ledger.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    full = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(), "git_sha": git_sha(),
            "harness_version": __version__, **row}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(full, default=str) + "\n")
    return full


def export_parquet(path: Path | None = None, out: Path | None = None) -> Path:
    import pandas as pd
    path = path or RESULTS / "ledger.jsonl"
    out = out or RESULTS / "ledger.parquet"
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    pd.json_normalize(rows).to_parquet(out)
    return out


def cache_key(system: str, revision: str, adapter_version: str, ds_hash: str, config: dict) -> str:
    return stable_hash([system, revision, adapter_version, ds_hash, config])


def cache_path(key: str, root: Path | None = None) -> Path:
    return (root or RESULTS / "predictions") / f"{key}.jsonl"


def write_predictions(preds_by_doc: dict[str, list[Span]], path: Path, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps({"_meta": meta}) + "\n")
        for did, spans in preds_by_doc.items():
            f.write(json.dumps({"doc_id": did, "spans": [s.to_dict() for s in spans]}, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_predictions(path: Path) -> tuple[dict, dict[str, list[Span]]]:
    meta, out = {}, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if "_meta" in r:
            meta = r["_meta"]; continue
        out[r["doc_id"]] = [Span.from_dict(s) for s in r["spans"]]
    return meta, out
