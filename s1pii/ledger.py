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


REPO = Path(__file__).resolve().parents[1]


class UnknownProvenance(RuntimeError):
    pass


def git_sha() -> str:
    """Commit of the installed code: the repo checkout this module lives in, else the commit
    recorded by a ``pip install git+...`` (PEP 610 direct_url.json), else $S1PII_GIT_SHA."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, stderr=subprocess.DEVNULL).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO,
                                        stderr=subprocess.DEVNULL).decode().strip() != ""
        return sha + ("-dirty" if dirty else "")
    except Exception:
        pass
    try:
        from importlib.metadata import distribution
        raw = distribution("s1pii").read_text("direct_url.json")
        if raw:
            vcs = json.loads(raw).get("vcs_info", {})
            if vcs.get("commit_id"):
                return vcs["commit_id"]
    except Exception:
        pass
    return os.environ.get("S1PII_GIT_SHA", "unknown")


def stable_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:16]


def dataset_hash(docs: Sequence[Doc]) -> str:
    h = hashlib.sha256()
    for d in sorted(docs, key=lambda d: d.doc_id):
        h.update(d.doc_id.encode()); h.update(hashlib.sha256(d.text.encode()).digest())
    return h.hexdigest()[:16]


def append(row: dict, path: Path | None = None, headline: bool = False) -> dict:
    """Append one row with a single O_APPEND write followed by fsync. Headline rows refuse
    unknown or dirty code provenance."""
    path = path or RESULTS / "ledger.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    sha = git_sha()
    if headline and (sha == "unknown" or sha.endswith("-dirty")):
        raise UnknownProvenance(f"headline results need clean, known code provenance (got {sha!r})")
    full = {"ts": dt.datetime.now(dt.timezone.utc).isoformat(), "git_sha": sha,
            "harness_version": __version__, "headline": headline, **row}
    data = (json.dumps(full, default=str) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)
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
    """Atomic write: unique temp file in the same directory, fsync, then rename."""
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".part")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps({"_meta": meta}) + "\n")
            for did, spans in preds_by_doc.items():
                f.write(json.dumps({"doc_id": did, "spans": [s.to_dict() for s in spans]}, ensure_ascii=False) + "\n")
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def read_predictions(path: Path) -> tuple[dict, dict[str, list[Span]]]:
    meta, out = {}, {}
    for line in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if "_meta" in r:
            meta = r["_meta"]; continue
        out[r["doc_id"]] = [Span.from_dict(s) for s in r["spans"]]
    return meta, out
