"""Predict with an exported S1 model through the shared sharded runner.

    python -m s1pii.model.run_s1 --model /content/drive/MyDrive/s1pii/models/all-s1/final \
        --docs $S1PII_DATA/splits/tab_direct-test.jsonl --out $S1PII_RESULTS/predictions --system s1_all_seed1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..adapters.run import run
from .predict import S1Predictor
from .train import load_exported


def predictor(model_dir: Path, *, system: str, validators: bool = True, propagation: bool = True,
              floor: float = 0.01, max_len: int = 1024) -> S1Predictor:
    model, tok, manifest = load_exported(model_dir)
    p = S1Predictor(model, tok, revision=manifest["weights_sha256"], max_len=max_len, floor=floor,
                    validators=validators, propagation=propagation, system=system)
    p.versions.update({"train_version": manifest["train_version"], "variant": manifest["config"]["variant"],
                       "seed": manifest["config"]["seed"], "train_code_sha": manifest.get("code_sha")})
    return p


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--docs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--system", required=True)
    ap.add_argument("--no-validators", action="store_true")
    ap.add_argument("--no-propagation", action="store_true")
    ap.add_argument("--shard-size", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    pr = predictor(a.model, system=a.system, validators=not a.no_validators, propagation=not a.no_propagation)
    path = run(a.system, a.docs, a.out, predictor=pr, shard_size=a.shard_size, batch_size=a.batch_size, limit=a.limit)
    print(json.dumps({"predictions": str(path)}))


if __name__ == "__main__":
    main(sys.argv[1:])
