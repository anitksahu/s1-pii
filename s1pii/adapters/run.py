"""Sharded, resumable prediction runner. Runs inside a baseline's own venv:

    envs/gliner2/bin/python -m s1pii.adapters.run --system gliner2_pii \
        --docs $S1PII_DATA/snapshots/tab_direct-test.jsonl --out results/predictions

Input is a canonical snapshot JSONL written by the main environment (``loaders.load``), so
baseline venvs need no dataset libraries. Documents are sorted by doc_id and split into
shards; each shard is written atomically and skipped on resume. When all shards exist
they are merged into ``<out>/<cache_key>.jsonl`` whose first line is the run meta
(system, resolved revision, package versions, adapter config, dataset hash, report).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

from ..schema import read_jsonl, Span
from ..ledger import cache_key, dataset_hash, write_predictions, read_predictions, git_sha
from .base import Adapter, ADAPTER_VERSION


def _merge(total: dict, part: dict) -> None:
    for k, v in part.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            total[k] = total.get(k, 0) + v
        elif isinstance(v, dict):
            d = total.setdefault(k, {})
            for kk, vv in v.items():
                d[kk] = d.get(kk, 0) + vv


def run(system: str, docs_path: Path, out: Path, *, backend=None, predictor=None, shard_size: int = 200,
        batch_size: int = 16, limit: int | None = None) -> Path:
    """``predictor`` (e.g. ``S1Predictor``) replaces the GLiNER adapter when given: any object
    with ``predict_docs``, ``config``, ``revision`` and ``versions``."""
    docs = sorted(read_jsonl(docs_path), key=lambda d: d.doc_id)
    if limit:
        docs = docs[:limit]
    if predictor is not None:
        adapter, src = predictor, predictor
    else:
        if backend is None:
            from .backends import make_backend
            backend = make_backend(system)
        adapter, src = Adapter(system, backend), backend
    revision = getattr(src, "revision", "unknown")
    ds_hash = dataset_hash(docs)
    versions = dict(getattr(src, "versions", {}))
    key = cache_key(system, revision, ADAPTER_VERSION, ds_hash, {**adapter.config(), "versions": versions})
    final = out / f"{key}.jsonl"
    if final.exists():
        return final
    shard_dir = out / f"{key}.shards-{shard_size}"
    shard_dir.mkdir(parents=True, exist_ok=True)
    report, t0 = {}, time.time()
    for si in range(0, len(docs), shard_size):
        path = shard_dir / f"shard-{si // shard_size:05d}.jsonl"
        shard = docs[si:si + shard_size]
        if path.exists():
            meta, got = read_predictions(path)
            if set(got) == {d.doc_id for d in shard}:
                _merge(report, meta.get("report", {}))
                continue
            path.unlink()                         # stale or partial shard: recompute
        preds, rep = adapter.predict_docs(shard, batch_size)
        write_predictions(preds, path, {"report": dict(rep.__dict__), "n_docs": len(shard)})
        _merge(report, dict(rep.__dict__))
        print(f"[{system}] shard {si // shard_size + 1}/{(len(docs) + shard_size - 1) // shard_size} "
              f"docs={len(shard)} kept={rep.kept}", flush=True)
    merged: dict[str, list[Span]] = {}
    for path in sorted(shard_dir.glob("shard-*.jsonl")):
        merged.update(read_predictions(path)[1])
    missing = {d.doc_id for d in docs} - set(merged)
    if missing:
        raise RuntimeError(f"{len(missing)} docs missing from shards, e.g. {sorted(missing)[:3]}")
    meta = {"system": system, "revision": revision, "adapter_version": ADAPTER_VERSION,
            "dataset_hash": ds_hash, "docs_path": str(docs_path), "n_docs": len(docs),
            "config": adapter.config(), "report": report, "code_sha": git_sha(),
            "versions": {**versions, "python": platform.python_version()},
            "seconds": round(time.time() - t0, 1)}
    write_predictions({d.doc_id: merged[d.doc_id] for d in docs}, final, meta)
    return final


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--docs", required=True, type=Path)
    ap.add_argument("--out", default=Path("results/predictions"), type=Path)
    ap.add_argument("--shard-size", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    path = run(a.system, a.docs, a.out, shard_size=a.shard_size, batch_size=a.batch_size, limit=a.limit)
    print(json.dumps({"predictions": str(path)}))


if __name__ == "__main__":
    main(sys.argv[1:])
