"""Resumable Stage 0/1/2 command runner used by the Colab chain."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
import urllib.request
from pathlib import Path

import torch
import yaml

from ..schema import Doc, Span, OTHER_PII
from ..ledger import append
from . import labels
from .data import assert_no_heldout_leakage, generate_questions
from .train import GPUCapReached, GPUHours


UNITS = {
    "stage0": ("label_draw", "census", "revisions", "proposer_all", "proposer_no_nemotron",
               "prompted_probe", "kev_baseline", "latency"),
    "stage1": ("train_sizes", "layout_ablation", "dev_eval"),
    "stage2": ("train_final", "test_inference", "comparators"),
}


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part"); tmp.write_text(json.dumps(value, indent=2, sort_keys=True)); os.replace(tmp, path)


def _dry_docs() -> list[Doc]:
    rows = []
    for i, (text, raw) in enumerate((("Email Ada at ada@example.test", "email"),
                                     ("Call Jo on 555-0102", "phone_number"))):
        surface = text.split()[-1]; start = text.rfind(surface)
        rows.append(Doc(f"dry-{i}", text, (Span(f"dry-{i}", start, len(text), OTHER_PII,
                                                      raw, surface=surface),), "synthetic", "train", f"dry-{i}"))
    return rows


def _unit_label_draw(ctx, out):
    src = labels.CONFIG; dst = ctx["root"] / "s1d_heldout.yaml"
    if not dst.exists(): shutil.copy2(src, dst)
    cfg = labels.load(src)
    append({"unit": "s1d_label_draw", "seed": cfg["seed"], "rule": cfg["rule"],
            "config_sha": cfg["draw_sha256"]}, path=ctx["root"] / "ledger.jsonl")
    if not ctx["dry"]:
        labels.record_nearest_trained_neighbours(ctx["root"] / "ledger.jsonl", cfg)
    return {"sha": cfg["draw_sha256"]}


def _unit_census(ctx, out):
    cfg = labels.load(ctx["root"] / "s1d_heldout.yaml")
    if ctx["dry"]:
        counts = {name: cfg["min_spans"] for name in cfg["test_labels"]}
        return {"counts": counts, "fallback": [], "failed": [], "passed": True, "dry_docs": 2}
    data = Path(os.environ["S1PII_DATA"])
    candidates = list(data.rglob("*nemotron*test*.jsonl"))
    if not candidates:
        raise FileNotFoundError("Nemotron test JSONL not found below S1PII_DATA")
    from ..schema import read_jsonl
    spans = (s for p in candidates for d in read_jsonl(p) for s in d.spans)
    return labels.census(spans, config=cfg, ledger_path=ctx["root"] / "ledger.jsonl")


def _unit_revisions(ctx, out):
    models = ("Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B")
    external = ctx["config"]["external"]
    if ctx["dry"]:
        return {"implementation_version": 2,
                "models": {m: "local-dry" for m in models}, "external": "local-dry"}
    from huggingface_hub import HfApi
    pinned = {m: ctx["config"]["models"][m]["revision"] for m in models}
    pinned.update({external["kev"]["model_id"]: external["kev"]["revision"],
                   external["kev"]["base_id"]: external["kev"]["base_revision"],
                   external["proposer"]["model_id"]: external["proposer"]["revision"],
                   external["sentence_encoder"]["model_id"]: external["sentence_encoder"]["revision"]})
    resolved = {}
    for model in pinned:
        resolved[model] = HfApi().model_info(model, revision=pinned[model]).sha
        if resolved[model] != pinned[model]:
            raise RuntimeError(f"revision did not resolve exactly: {model}@{pinned[model]}")
    kev_sha = external["kev"]["git_revision"]
    with urllib.request.urlopen(f"https://api.github.com/repos/jaredpalmer/kev/commits/{kev_sha}") as response:
        github_sha = json.loads(response.read())["sha"]
    if github_sha != kev_sha:
        raise RuntimeError("Kev git revision did not resolve exactly")
    result = {"implementation_version": 2, "models": resolved, "kev_git": github_sha}
    append({"unit": "s1d_revisions", **result}, path=ctx["root"] / "ledger.jsonl")
    return result


def _unit_proposer(ctx, out, variant):
    docs = _dry_docs() if ctx["dry"] else []
    if not ctx["dry"]:
        from .proposer import train_and_gate
        return train_and_gate(variant, ctx["root"], cap_hours=ctx["config"]["stages"]["stage0"]["cap_a100_hours"],
                              control_path=ctx["root"] / "CONTROL", config=ctx["config"])
    # Verify the one-type path really optimizes for one step without a pretrained model.
    from ..model.s1 import S1Model
    class Encoder(torch.nn.Module):
        def __init__(self): super().__init__(); self.emb = torch.nn.Embedding(32, 8)
        def forward(self, input_ids, attention_mask):
            return type("Output", (), {"last_hidden_state": self.emb(input_ids)})
    model = S1Model(Encoder(), 8, dropout=0, num_types=1)
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    allowed = torch.ones(1, 3, 5, dtype=torch.bool)
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "tok_mask": torch.ones(1, 3, dtype=torch.bool),
             "allowed": allowed, "n_prefix": torch.tensor([0]), "tok_len": torch.tensor([3])}
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3); loss = model(batch); loss.backward(); opt.step()
    from .proposer import gate_g1
    gate = gate_g1({"dry": [(0, 1)]}, {"dry": [(0, 1, 0.9)]})
    return {"implementation_version": 2, "variant": variant, "docs": len(docs),
            "one_step_loss": float(loss.detach()), "gate_g1": gate}


def _unit_questions(ctx, out, name):
    if not ctx["dry"]:
        from .stage0 import kev_baseline, prompted_probe
        control = ctx["root"] / "CONTROL"
        if name == "prompted_probe":
            return prompted_probe(ctx["root"], ctx["config"], control)
        if name == "kev_baseline":
            return kev_baseline(ctx["root"], ctx["config"], control)
        raise ValueError(f"unknown Stage 0 question unit {name!r}")
    q = generate_questions(_dry_docs(), variant="all-sources", seed=0,
                           ledger_path=ctx["root"] / "ledger.jsonl")
    if q: assert_no_heldout_leakage(q)
    choice = [row for row in q if row.question.type == "choice"]
    option_count = len(choice[0].question.options) if choice else 0
    chance = 1 / option_count if option_count else 0.0
    score = chance * 2.1
    return {"implementation_version": 2, "unit": name, "questions": len(q),
            "options": option_count, "chance": chance,
            "accuracy": score, "macro_accuracy": score, "per_label_accuracy": {},
            "majority_class_baseline": chance, "status": "dry-complete"}


def _unit_latency(ctx, out):
    if not ctx["dry"]:
        from .stage0 import latency_benchmark
        return latency_benchmark(ctx["root"], ctx["config"], ctx["root"] / "CONTROL")
    from .latency import benchmark
    return benchmark(ctx["config"], dry=True)


def _unimplemented(ctx, _out, name):
    if ctx["dry"]:
        return {"status": "dry-complete", "unit": name}
    raise NotImplementedError(f"{name} is intentionally unavailable until its approved stage is implemented")


def _append_stop_rule_once(root: Path) -> None:
    path = root / "ledger.jsonl"
    if path.exists():
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row.get("unit") == "s1d_stage0_stop_rule" and row.get("version") == 1:
                return
    append({"unit": "s1d_stage0_stop_rule", "version": 1,
            "statistic": "macro_accuracy_over_dev_labels", "threshold": "2 * chance",
            "chance": "1 / len(options)",
            "fires_when": "both_prompted_and_kev_are_strictly_below_threshold"}, path=path)


def _stage0_should_stop(stores: Path, dry: bool) -> bool:
    probe = json.loads((stores / "stage0-prompted_probe.json").read_text())
    kev = json.loads((stores / "stage0-kev_baseline.json").read_text())
    chance = probe.get("chance", kev.get("chance"))
    return (not dry and chance is not None and
            probe.get("macro_accuracy") is not None and kev.get("macro_accuracy") is not None and
            probe["macro_accuracy"] < 2 * chance and kev["macro_accuracy"] < 2 * chance)


def _cache_is_current(root: Path, stores: Path, stage: str, unit: str, dry: bool) -> bool:
    result_path = stores / f"{stage}-{unit}.json"
    if not (stores / f"{stage}-{unit}.done").exists() or not result_path.exists():
        return False
    if stage != "stage0":
        # No production implementation exists yet, so never accept a legacy
        # placeholder marker as completion. Dry runs may resume their stubs.
        return dry
    if unit == "census":
        return True
    if unit == "label_draw":
        if dry:
            return True
        ledger = root / "ledger.jsonl"
        return ledger.exists() and any(
            json.loads(line).get("unit") == "s1d_label_neighbours"
            for line in ledger.read_text().splitlines() if line.strip())
    try:
        return json.loads(result_path.read_text()).get("implementation_version") == 2
    except (OSError, ValueError):
        return False


def run(stage: str, root: Path, dry: bool = False) -> int:
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "s1d.yaml").read_text())
    root.mkdir(parents=True, exist_ok=True)
    (root / "PHASE").write_text("CPU")
    if stage == "stage0":
        _append_stop_rule_once(root)
    if cfg["stages"][stage].get("requires_approval") and not (root / f"APPROVED_{stage}").exists():
        raise PermissionError(f"{root / ('APPROVED_' + stage)} is required")
    ctx = {"root": root, "dry": dry, "config": cfg}
    meter = GPUHours(root / "gpu_hours.jsonl", cfg["stages"][stage]["cap_a100_hours"], stage)
    try:
        meter.reserve(0.0)
    except GPUCapReached:
        return 3
    handlers = {
        "label_draw": _unit_label_draw, "census": _unit_census, "revisions": _unit_revisions,
        "proposer_all": lambda c, o: _unit_proposer(c, o, "all-sources"),
        "proposer_no_nemotron": lambda c, o: _unit_proposer(c, o, "no-nemotron"),
        "prompted_probe": lambda c, o: _unit_questions(c, o, "prompted_probe"),
        "kev_baseline": lambda c, o: _unit_questions(c, o, "kev_baseline"),
        "latency": _unit_latency,
        "train_sizes": lambda c, o: _unimplemented(c, o, "train_sizes"),
        "layout_ablation": lambda c, o: _unimplemented(c, o, "layout_ablation"),
        "dev_eval": lambda c, o: _unimplemented(c, o, "dev_eval"),
        "train_final": lambda c, o: _unimplemented(c, o, "train_final"),
        "test_inference": lambda c, o: _unimplemented(c, o, "test_inference"),
        "comparators": lambda c, o: _unimplemented(c, o, "comparators"),
    }
    stores = root / "stores"; stores.mkdir(exist_ok=True)
    try:
        for unit in UNITS[stage]:
            (root / "PHASE").write_text("CPU")
            if (root / "CONTROL").exists() and (root / "CONTROL").read_text().strip().upper() == "STOP":
                return 4
            if stage == "stage0" and unit == "latency" and _stage0_should_stop(stores, dry):
                return 5
            done = stores / f"{stage}-{unit}.done"
            if _cache_is_current(root, stores, stage, unit, dry):
                continue
            if not dry and unit not in ("label_draw", "census", "revisions"):
                meter.reserve(float(cfg["stages"][stage].get("unit_estimates", {}).get(unit, 0)))
            result = handlers[unit](ctx, stores / f"{stage}-{unit}.json")
            if not isinstance(result, dict) or result.get("status") in {"configured", "entrypoint-ready"}:
                raise RuntimeError(f"{unit} did not produce a complete result")
            _atomic_json(stores / f"{stage}-{unit}.json", result)
            done.write_text(time.strftime("%FT%TZ", time.gmtime()))
    except GPUCapReached:
        return 3
    except InterruptedError:
        return 4
    if stage == "stage1":
        result = json.loads((stores / "stage1-dev_eval.json").read_text())
        trained, prompted = result.get("trained_accuracy"), result.get("prompted_accuracy")
        if None not in (trained, prompted) and trained < prompted:
            return 5
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("stage", choices=UNITS); ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args(argv)
    raise SystemExit(run(args.stage, args.root, os.environ.get("S1D_DRY") == "1"))


if __name__ == "__main__": main()
