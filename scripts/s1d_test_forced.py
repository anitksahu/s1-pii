#!/usr/bin/env python3
"""CPU-only per-label audit of saved S1-D dev and test probabilities.

This script never constructs a model. It rebuilds the frozen questions, reads only completed
evaluation caches, and reports ordinary/forced-choice behavior on gold spans.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from s1pii.s1d import labels as H
from s1pii.s1d import run as R


TEST_CACHE_NAMES = {
    "prompted-0.6B": "prompted-0.6B-test.json",
    "prompted-1.7B": "prompted-1.7B-test.json",
    "prompted-4B": "prompted-4B-test.json",
    "1.7B-s1": "trained-1.7B-1.7B-s1-shared-test.json",
    "1.7B-s2": "trained-1.7B-1.7B-s2-shared-test.json",
    "4B-s1": "trained-4B-4B-s1-shared-test.json",
    "4B-s2": "trained-4B-4B-s2-shared-test.json",
}
DEV_CACHE_NAMES = {
    "prompted-4B": "prompted-4B-dev.json",
    "1.7B-s1": "trained-1.7B-1.7B-s1-shared-dev.json",
    "1.7B-s2": "trained-1.7B-1.7B-s2-shared-dev.json",
    "4B-s1": "trained-4B-4B-s1-shared-dev.json",
    "4B-s2": "trained-4B-4B-s2-shared-dev.json",
}


def _probabilities(path: Path, count: int, fingerprint: str) -> np.ndarray:
    value = json.loads(path.read_text())
    probabilities = value.get("probabilities")
    if value.get("version") != R._EVAL_CACHE_VERSION:
        raise ValueError(f"{path}: evaluation cache version is not {R._EVAL_CACHE_VERSION}")
    if value.get("completed") != value.get("total"):
        raise ValueError(f"{path}: evaluation cache is incomplete")
    if value.get("fingerprint") != fingerprint:
        raise ValueError(f"{path}: fingerprint does not match the rebuilt questions and saved system")
    if not isinstance(probabilities, list) or len(probabilities) != count \
            or any(row is None for row in probabilities):
        raise ValueError(f"{path}: expected {count} complete probability rows")
    result = np.asarray(probabilities, dtype=np.float64)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError(f"{path}: probabilities must be a finite matrix")
    return result


def per_label(rows, probabilities: np.ndarray, wanted: list[str]) -> dict:
    if len(rows) != len(probabilities):
        raise ValueError("question/probability count mismatch")
    option_names = [option.name for option in rows[0].question.options]
    if any([option.name for option in row.question.options] != option_names for row in rows):
        raise ValueError("questions do not share the frozen option set")
    if probabilities.shape[1] != len(option_names):
        raise ValueError("probability width does not match the frozen option set")
    not_pii = option_names.index(H.NOT_PII)
    prediction = probabilities.argmax(1)
    forced = probabilities.copy(); forced[:, not_pii] = -np.inf
    forced_prediction = forced.argmax(1)
    result = {}
    for raw in wanted:
        display = raw.replace("_", " ")
        indices = [i for i, row in enumerate(rows) if not row.hard_negative
                   and row.question.options[row.target].name == display]
        if not indices:
            raise ValueError(f"no gold questions for frozen label {raw}")
        top = Counter(option_names[int(forced_prediction[i])] for i in indices)
        result[raw] = {
            "gold_count": len(indices),
            "accuracy": float(np.mean([prediction[i] == rows[i].target for i in indices])),
            "forced_choice_accuracy": float(np.mean(
                [forced_prediction[i] == rows[i].target for i in indices])),
            "not_pii_rate_on_gold": float(np.mean([prediction[i] == not_pii for i in indices])),
            "top_3_forced_predictions": [
                {"option": option, "count": count, "share": count / len(indices)}
                for option, count in top.most_common(3)
            ],
        }
    return result


def _print_table(split: str, systems: dict) -> None:
    print(f"\n{split.upper()}")
    print(f"{'system':16s} {'label':28s} {'n':>6s} {'acc':>7s} {'forced':>7s} {'NP/gold':>7s} top forced")
    for system, labels in systems.items():
        for label, row in labels.items():
            top = ",".join(f"{item['option']}:{item['share']:.2f}"
                           for item in row["top_3_forced_predictions"])
            print(f"{system:16s} {label:28s} {row['gold_count']:6d} {row['accuracy']:7.3f} "
                  f"{row['forced_choice_accuracy']:7.3f} {row['not_pii_rate_on_gold']:7.3f} {top}")


def _prompted_fingerprint(rows, model_id: str, config: dict) -> str:
    identity = {"kind": "prompted", "model": model_id,
                "revision": config["models"][model_id]["revision"],
                "prompt_semantics": R._PROMPTED_SEMANTICS_VERSION,
                "state_tokens": config["state_tokens"], "scoring": "answer_code_distribution"}
    return R._eval_fingerprint(rows, identity)


def _trained_fingerprint(rows, model_id: str, run_dir: Path, config: dict) -> str:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    identity = {"kind": "trained", "model": model_id,
                "revision": config["models"][model_id]["revision"], "layout": "shared",
                "packing": {"state_tokens": 512, "stride": 384, "branches_per_window": 64,
                            "window_batch_size": 8},
                "run_hashes": manifest.get("hashes", {}),
                "step": (run_dir / "done").read_text().strip()}
    return R._eval_fingerprint(rows, identity)


def _fingerprints(rows, names: dict[str, str], config: dict, root: Path) -> dict[str, str]:
    model_ids = {model_id.rsplit("-", 1)[-1]: model_id for model_id in config["models"]}
    result = {}
    for name in names:
        if name.startswith("prompted-"):
            size = name.removeprefix("prompted-")
            result[name] = _prompted_fingerprint(rows, model_ids[size], config)
        else:
            size = name.split("-s", 1)[0]
            result[name] = _trained_fingerprint(
                rows, model_ids[size], root / "models" / "stage1-cov" / name, config)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path,
                        default=Path("/content/drive/MyDrive/s1pii/s1d"))
    parser.add_argument("--output", type=Path,
                        default=Path("docs/results_s1d/s1d_per_label.json"))
    parser.add_argument("--data", type=Path, default=None,
                        help="S1PII_DATA directory; defaults to the environment")
    args = parser.parse_args()
    if args.data is not None:
        os.environ["S1PII_DATA"] = str(args.data)
    if not os.environ.get("S1PII_DATA"):
        raise RuntimeError("S1PII_DATA must point at the local benchmark data")
    config = yaml.safe_load((Path(R.__file__).resolve().parents[1] / "configs" / "s1d.yaml").read_text())
    ctx = {"root": args.root, "dry": False, "config": config, "stage": "s1d_test"}
    _test_docs, test_rows = R._test_questions(
        ctx, descriptions=True, require_candidate_cache=True)
    _dev_docs, dev_rows = R._dev_questions(
        ctx, descriptions=True, require_candidate_cache=True)
    heldout = H.load()
    cache_root = args.root / "stores" / "stage1-dev-eval"
    fingerprints = {"test": _fingerprints(test_rows, TEST_CACHE_NAMES, config, args.root),
                    "dev": _fingerprints(dev_rows, DEV_CACHE_NAMES, config, args.root)}
    test_result = json.loads((args.root / "stores" / "s1d_test-test_eval.json").read_text())
    if test_result.get("questions") != len(test_rows):
        raise ValueError("completed s1d_test result does not match rebuilt test questions")
    splits = {
        "test": {
            name: per_label(test_rows, _probabilities(
                                cache_root / filename, len(test_rows), fingerprints["test"][name]),
                            heldout["test_labels"])
            for name, filename in TEST_CACHE_NAMES.items()
        },
        "dev": {
            name: per_label(dev_rows, _probabilities(
                                cache_root / filename, len(dev_rows), fingerprints["dev"][name]),
                            heldout["dev_labels"])
            for name, filename in DEV_CACHE_NAMES.items()
        },
    }
    output = {
        "version": 1,
        "source": {"root": str(args.root), "probabilities": "completed evaluation caches",
                   "model_inference": False, "threshold_fitting": False},
        "splits": splits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    R._atomic_json(args.output, output)
    for split, systems in splits.items():
        _print_table(split, systems)
    print("\nwrote", args.output)


if __name__ == "__main__":
    main()
