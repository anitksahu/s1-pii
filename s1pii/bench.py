"""Benchmark driver: materialize calibration/test splits, score prediction files, log results.

    python -m s1pii.bench materialize tab_direct spy_medical spy_legal pii_trace nemotron
    python -m s1pii.bench score --system gliner2_pii --dataset tab_direct \
        --calib results/predictions/<key-calib>.jsonl --test results/predictions/<key-test>.jsonl

Splits (prereg v0): TAB dev/test; Nemotron and Gretel test, with calibration drawn from a
hash-fixed 2% slice of train that S1 never trains on (``dev_slice``); single-split sets
(SPY, PII-TRACE, Fresh-Real) are split at the cluster level with ``calib_test_split``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from .schema import Doc, write_jsonl, read_jsonl
from .data import loaders as L
from .eval.evaluate import evaluate, tune_threshold
from .ledger import read_predictions, append, dataset_hash
from .adapters.base import load_config

DEV_SLICE_SALT = "s1pii-devslice-v0"
DEV_SLICE_FRAC = 0.02


def in_dev_slice(doc_id: str) -> bool:
    h = int(hashlib.sha256(f"{DEV_SLICE_SALT}|{doc_id}".encode()).hexdigest()[:8], 16)
    return h / 0xFFFFFFFF < DEV_SLICE_FRAC


def _group(d: Doc) -> str:
    """Near-duplicate group: Nemotron-PII ships each uid twice (one per locale); both twins
    must fall on the same side of the dev/train cut."""
    return (d.meta or {}).get("uid") or d.doc_id


def dev_slice(train: list[Doc]) -> tuple[list[Doc], list[Doc]]:
    """(dev, remaining_train). The dev slice is excluded from every S1 training run."""
    return [d for d in train if in_dev_slice(_group(d))], [d for d in train if not in_dev_slice(_group(d))]


def splits(name: str) -> tuple[list[Doc], list[Doc]]:
    """(calibration/dev, test) for a benchmark, per the prereg."""
    if name in ("tab_direct", "tab_quasi"):
        return L.load(name, "dev"), L.load(name, "test")
    if name in ("nemotron", "gretel"):
        return dev_slice(L.load(name, "train", purpose="eval"))[0], L.load(name, "test")
    return L.calib_test_split(L.load(name))


def split_paths(name: str) -> dict[str, Path]:
    base = L.DATA_DIR / "splits"
    return {p: base / f"{name}-{p}.jsonl" for p in ("calib", "test")}


def _split_meta(name: str) -> Path:
    return L.DATA_DIR / "splits" / f"{name}.meta.json"


def _split_valid(name: str) -> bool:
    paths, mp = split_paths(name), _split_meta(name)
    if not (mp.exists() and all(p.exists() for p in paths.values())):
        return False
    try:
        meta = json.loads(mp.read_text())
        return all(hashlib.sha256(paths[k].read_bytes()).hexdigest() == meta[k]["content_sha256"] for k in paths)
    except (KeyError, json.JSONDecodeError):
        return False


def materialize(name: str, refresh: bool = False) -> dict[str, str]:
    """Write calib/test split files atomically (data first, meta with content hashes last);
    reuse them only if they validate against the meta."""
    paths = split_paths(name)
    if refresh or not _split_valid(name):
        calib, test = splits(name)
        mp = _split_meta(name)
        if mp.exists():
            mp.unlink()
        meta = {}
        for part, docs in (("calib", calib), ("test", test)):
            body = "".join(d.to_json() + "\n" for d in docs)
            L._atomic_write_text(paths[part], body)
            meta[part] = {"n_docs": len(docs), "dataset_hash": dataset_hash(docs),
                          "content_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest()}
        L._atomic_write_text(mp, json.dumps(meta, indent=2))
    return {k: str(v) for k, v in paths.items()}


def score(system: str, dataset: str, calib_pred: Path | None, test_pred: Path, *,
          default_threshold: float | None = None, headline: bool = False, seed: int = 0,
          n_boot_ci: int = 2000) -> dict:
    paths = split_paths(dataset)
    test_docs = read_jsonl(paths["test"])
    meta, preds = read_predictions(test_pred)
    if meta.get("dataset_hash") and meta["dataset_hash"] != dataset_hash(test_docs):
        raise ValueError(f"{test_pred} was produced on different test docs than {paths['test']}")
    if headline:                       # prereg: flagged clusters removed for every system
        from .c0 import audited_test_docs
        test_docs = audited_test_docs(dataset)
        keep = {d.doc_id for d in test_docs}
        preds = {k: v for k, v in preds.items() if k in keep}
    if default_threshold is None:
        default_threshold = load_config()["systems"].get(system, {}).get("default_threshold", 0.5)
    dev_t = None
    if calib_pred is not None:
        cmeta, cpreds = read_predictions(calib_pred)
        calib_docs = read_jsonl(paths["calib"])
        if cmeta.get("dataset_hash") and cmeta["dataset_hash"] != dataset_hash(calib_docs):
            raise ValueError(f"{calib_pred} was produced on different calibration docs")
        for k in ("system", "revision", "adapter_version", "config"):
            if cmeta.get(k) != meta.get(k):
                raise ValueError(f"calibration and test predictions differ in {k!r}: the dev threshold "
                                 f"must come from the same model run configuration")
        dev_t = tune_threshold(calib_docs, cpreds, dataset, over_budget=0.01)
    res = evaluate(test_docs, preds, dataset=dataset, default_threshold=default_threshold,
                   dev_threshold=dev_t, n_boot_ci=n_boot_ci, seed=seed)
    row = append({"kind": "eval", "system": system, "dataset": dataset, "prediction_meta": meta,
                  "test_path": str(test_pred), "calib_path": str(calib_pred) if calib_pred else None,
                  "dev_threshold": dev_t, "result": res}, headline=headline)
    return row


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("materialize"); m.add_argument("datasets", nargs="+")
    s = sub.add_parser("score")
    s.add_argument("--system", required=True); s.add_argument("--dataset", required=True)
    s.add_argument("--calib", type=Path); s.add_argument("--test", type=Path, required=True)
    s.add_argument("--headline", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "materialize":
        for n in a.datasets:
            print(n, json.dumps(materialize(n)))
    else:
        r = score(a.system, a.dataset, a.calib, a.test, headline=a.headline)
        print(json.dumps({k: r["result"][k] for k in ("pauc", "pauc_ci", "reachable_max_over")}, default=str))


if __name__ == "__main__":
    main(sys.argv[1:])
