#!/usr/bin/env python3
"""Reconstruct Stage 1 pilot optimizer batches without loading a GPU model."""
from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import islice
from pathlib import Path

# Direct execution sets sys.path[0] to scripts/, so make the repository package importable
# without relying on an editable installation in the active Colab runtime.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import yaml

os.environ.setdefault("S1PII_DATA", "/content/s1pii_data")

from s1pii.s1d import run as R
from s1pii.s1d.data import (pack_training_questions, question_set_hash,
                            training_batch_composition)
from s1pii.s1d.model import prepare_tokenizer
from s1pii.s1d.train import token_batches


def window_summary(steps: list[dict], checkpoint: int, width: int,
                   not_pii_p95: float, dominant_p95: float) -> dict:
    rows = [row for row in steps if checkpoint - width < row["step"] <= checkpoint]
    not_pii = [row["not_pii_target_share"] for row in rows if row["not_pii_target_share"] is not None]
    dominant = [row["dominant_target_share"] for row in rows if row["dominant_target_share"] is not None]
    return {"steps": [row["step"] for row in rows],
            "mean_not_pii_target_share": float(np.mean(not_pii)) if not_pii else None,
            "max_not_pii_target_share": max(not_pii, default=None),
            "mean_dominant_target_share": float(np.mean(dominant)) if dominant else None,
            "max_dominant_target_share": max(dominant, default=None),
            "high_not_pii_steps": [row["step"] for row in rows
                                    if row["not_pii_target_share"] is not None
                                    and row["not_pii_target_share"] > not_pii_p95],
            "high_dominant_steps": [row["step"] for row in rows
                                     if row["dominant_target_share"] is not None
                                     and row["dominant_target_share"] > dominant_p95]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/content/drive/MyDrive/s1pii/s1d"))
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)
    root = args.root; out = args.out or root / "stores" / "stage1-pilot-batch-audit.json"
    legacy = root / "stores" / "stage1_pilot-paired_optimizer-v1.json"
    result_path = legacy if legacy.exists() else root / "stores" / "stage1_pilot-paired_optimizer.json"
    pilot_result = json.loads(result_path.read_text())
    if pilot_result.get("implementation_version") != 1:
        raise RuntimeError(f"batch audit requires the completed 330-step v1 pilot: {result_path}")
    cfg = yaml.safe_load((Path(R.__file__).resolve().parents[1] / "configs" / "s1d.yaml").read_text())
    spec = cfg["stage1_pilot"]; data_seed = int(spec["data_seed"])
    ctx = {"root": root, "dry": False, "config": cfg}
    questions = R._training_questions(ctx, data_seed=data_seed, require_candidate_cache=True)
    selected = R._repeat_rows(questions, int(spec["windows"]), data_seed)
    selected_hash = question_set_hash(selected)
    recorded = {condition["training"]["hashes"]["questions"]
                for pair in pilot_result["runs"].values()
                for condition in pair["conditions"].values()}
    if recorded != {selected_hash}:
        raise RuntimeError(f"reconstructed question hash {selected_hash} != pilot manifests {recorded}")

    sizes = {pair.split("-init", 1)[0] for pair in pilot_result["runs"]}
    if len(sizes) != 1:
        raise RuntimeError(f"audit expects one pilot model size, found {sorted(sizes)}")
    size = sizes.pop()
    model_id = next(model for model in cfg["models"] if model.endswith(size))
    revision = cfg["models"][model_id]["revision"]
    manifests = [condition["training"] for pair in pilot_result["runs"].values()
                 for condition in pair["conditions"].values()]
    expected_budget = int(cfg["training"]["token_budget"])
    if {manifest["hashes"]["revision"] for manifest in manifests} != {revision}:
        raise RuntimeError("pilot tokenizer revision does not match the pinned audit revision")
    if {int(manifest["config"]["token_budget"]) for manifest in manifests} != {expected_budget}:
        raise RuntimeError("pilot token budget does not match the audit configuration")
    if {int(manifest["config"]["data_seed"]) for manifest in manifests} != {data_seed}:
        raise RuntimeError("pilot batch-order seed does not match the audit data seed")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    prepare_tokenizer(tokenizer)

    def build(index):
        packed, target = pack_training_questions(
            [selected[index]], tokenizer, layout="shared", device=torch.device("cpu"),
            make_block_mask=False, keep_dense_mask=False)[0]
        return packed, target, selected[index]

    packed = R._LazyPackedRows(len(selected), build)
    batches = token_batches(packed, expected_budget,
                            length=lambda item: len(item[0].input_ids), seed=data_seed)
    max_steps = int(pilot_result["max_steps"])
    steps = []
    for step, batch in enumerate(islice(batches, max_steps), 1):
        steps.append({"step": step, **training_batch_composition([item[2] for item in batch])})
    if len(steps) != max_steps:
        raise RuntimeError(f"only reconstructed {len(steps)}/{max_steps} optimizer batches")

    not_pii_values = [row["not_pii_target_share"] for row in steps
                      if row["not_pii_target_share"] is not None]
    dominant_values = [row["dominant_target_share"] for row in steps
                       if row["dominant_target_share"] is not None]
    not_pii_p95 = float(np.percentile(not_pii_values, 95))
    dominant_p95 = float(np.percentile(dominant_values, 95))
    for row in steps:
        row["high_not_pii_share"] = (row["not_pii_target_share"] is not None
                                      and row["not_pii_target_share"] > not_pii_p95)
        row["high_dominant_share"] = (row["dominant_target_share"] is not None
                                       and row["dominant_target_share"] > dominant_p95)

    checkpoints = []
    for pair, pair_result in pilot_result["runs"].items():
        for condition, condition_result in pair_result["conditions"].items():
            for checkpoint in condition_result["checkpoints"]:
                row = {"pair": pair, "condition": condition, "step": checkpoint["step"],
                       "loss": checkpoint["loss"], "collapsed": checkpoint["collapsed"],
                       "collapse_reasons": checkpoint.get("collapse_reasons", [])}
                for width in (10, 25):
                    row[f"previous_{width}"] = window_summary(
                        steps, checkpoint["step"], width, not_pii_p95, dominant_p95)
                checkpoints.append(row)

    output = {"version": 1, "cpu_only": True, "data_seed": data_seed,
              "question_set_sha256": selected_hash, "model_id": model_id,
              "tokenizer_revision": revision, "batch_seed": data_seed,
              "token_budget": expected_budget,
              "max_steps": max_steps,
              "definitions": {"window": "checkpoint-width < batch_step <= checkpoint",
                              "dominant_target": "canonical Choice target only",
                              "mean_option_count": "all questions in the optimizer batch"},
              "percentiles": {"not_pii_target_share_p95": not_pii_p95,
                              "dominant_target_share_p95": dominant_p95},
              "steps": steps, "checkpoints": checkpoints}
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(out.suffix + ".part"); part.write_text(json.dumps(output, indent=2)); os.replace(part, out)
    print("pair\tcondition\tstep\tcollapsed\tnotpii10\tdominant10\thighN10\thighD10\t"
          "notpii25\tdominant25\thighN25\thighD25")
    def value(number):
        return "NA" if number is None else f"{number:.3f}"

    for row in checkpoints:
        a, b = row["previous_10"], row["previous_25"]
        print(f"{row['pair']}\t{row['condition']}\t{row['step']}\t{row['collapsed']}\t"
              f"{value(a['mean_not_pii_target_share'])}\t{value(a['mean_dominant_target_share'])}\t"
              f"{a['high_not_pii_steps']}\t{a['high_dominant_steps']}\t"
              f"{value(b['mean_not_pii_target_share'])}\t{value(b['mean_dominant_target_share'])}\t"
              f"{b['high_not_pii_steps']}\t{b['high_dominant_steps']}")
    print("wrote", out)


if __name__ == "__main__":
    main()
