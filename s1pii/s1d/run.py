"""Resumable Stage 0/1/2 command runner used by the Colab chain."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import urllib.request
from collections import OrderedDict, defaultdict
from pathlib import Path

import torch
import yaml

from ..schema import Doc, Span, OTHER_PII
from ..ledger import append
from ..v2 import labels as v2
from . import labels
from .data import assert_no_heldout_leakage, generate_questions
from .train import GPUHours


UNITS = {
    "stage0": ("label_draw", "census", "revisions", "prompted_probe", "kev_baseline",
               "proposer_all", "proposer_no_nemotron", "latency"),
    "stage1": ("train_sizes", "layout_ablation", "dev_eval"),
    "stage1_pilot": ("paired_optimizer",),
    "stage1_pilot_4b": ("stable_optimizer",),
    "stage2": ("train_final", "test_inference", "comparators"),
    "comparators1": ("decision20",),
}

_TRAINING_SEMANTICS_VERSION = 2
_PILOT_SEMANTICS_VERSION = 4


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
    return benchmark(ctx["config"], dry=True, root=ctx["root"])


def _stage1_docs(dry: bool) -> list[Doc]:
    if dry:
        cfg = labels.load(); names = cfg["dev_labels"][:2]
        docs = []
        for i, name in enumerate(names):
            surface = f"secret{i}"
            text = f"A record contains {surface} and ordinary text"
            start = text.index(surface)
            docs.append(Doc(f"stage1-dry-{i}", text,
                            (Span(f"stage1-dry-{i}", start, start + len(surface), OTHER_PII,
                                  name, surface=surface),),
                            "synthetic_conv", "calib", f"stage1-dry-{i}"))
        return docs
    from ..schema import read_jsonl
    from .. import bench
    return read_jsonl(bench.split_paths("nemotron")["calib"])


# Bump when the candidate-mining code (floor, overlap filter, scoring, cap policy) changes, so
# both the cache key and the stored payload reject candidates produced by an older miner.
_CANDIDATE_MINING_VERSION = 1


def _require_end_to_end_latency(latency: dict, dry: bool) -> None:
    """Selection must not run against model-only or bare-encoder latency evidence."""
    if not dry and not latency.get("proposer_included"):
        raise RuntimeError("stage0-latency.json predates the end-to-end (proposer-included) latency "
                           "benchmark; re-run Stage 0 latency before Stage 1 selection")


def _proposer_weights_sha(path: Path) -> str:
    """The proposer identity used both to build S1Predictor and to key the candidate cache.

    Read from the same ``s1_manifest.json`` that ``load_exported`` consumes, so the cache can
    never silently reuse candidates mined from different proposer weights."""
    manifest = json.loads((path / "s1_manifest.json").read_text())
    weights = manifest.get("weights_sha256")
    if not weights:
        raise RuntimeError(f"proposer manifest at {path} is missing weights_sha256")
    return weights


def _candidate_cache_path(root: Path, docs: list[Doc], weights_sha: str, cache_name: str) -> Path:
    identity = hashlib.sha256(json.dumps({
        "documents": [doc.doc_id for doc in docs],
        "weights": weights_sha,
        "floor": 0.01,
        "mining_version": _CANDIDATE_MINING_VERSION,
    }, sort_keys=True).encode()).hexdigest()[:16]
    return root / "stores" / f"stage1-{cache_name}-candidates-{identity}.json"


def _proposer_candidates(root: Path, docs: list[Doc], dry: bool, *,
                         cache_name: str | None = None,
                         require_cache: bool = False) -> dict[str, list[Span]]:
    if dry:
        output = {}
        for doc in docs:
            surface = doc.text.split()[0]
            start = doc.text.index(surface)
            output[doc.doc_id] = [Span(doc.doc_id, start, start + len(surface), OTHER_PII,
                                       "candidate", score=0.5, surface=surface)]
        return output
    from ..model.train import load_exported
    from ..model.predict import S1Predictor
    path = root / "models" / "proposer-no-nemotron" / "final"
    weights_sha = _proposer_weights_sha(path)
    cache_path = _candidate_cache_path(root, docs, weights_sha, cache_name) if cache_name else None
    if cache_path is not None and cache_path.exists():
        payload = json.loads(cache_path.read_text())
        if payload.get("mining_version") == _CANDIDATE_MINING_VERSION:
            print(f"stage1 proposer candidates: loaded {payload['count']} from {cache_path.name}", flush=True)
            return {doc_id: [Span.from_dict(row) for row in rows]
                    for doc_id, rows in payload["candidates"].items()}
        print(f"stage1 proposer candidates: recomputing; {cache_path.name} has mining_version "
              f"{payload.get('mining_version')} != {_CANDIDATE_MINING_VERSION}", flush=True)
    if require_cache:
        expected = cache_path if cache_path is not None else "a named candidate cache"
        raise FileNotFoundError(f"CPU-only audit requires the existing proposer cache: {expected}")
    meter = GPUHours(root / "gpu_hours.jsonl", 12, "stage1")
    meter.reserve(0.0)
    last_accounted = time.monotonic()
    # Mining interleaves CPU tokenisation/CRF decoding with low-utilisation proposer forwards,
    # so it stays in the CPU phase: marking it GPU would let the idle-GPU watchdog kill a
    # legitimately slow pass. Wall-clock is still metered against the stage-1 A100 budget.
    (root / "PHASE").write_text("CPU")
    model, tokenizer, _manifest = load_exported(path, device="cuda")
    predictor = S1Predictor(model, tokenizer, revision=weights_sha, max_len=512,
                            floor=0.01, validators=False, propagation=False, device="cuda")
    output = {}
    try:
        control = root / "CONTROL"
        for first in range(0, len(docs), 64):
            if control.exists() and control.read_text().strip().upper() == "STOP":
                raise InterruptedError("STOP requested")
            batch = docs[first:first + 64]
            chunk, _ = predictor.predict_docs(batch)
            for doc in batch:
                output[doc.doc_id] = sorted(
                    (span for span in chunk.get(doc.doc_id, ())
                     if not any(span.start < gold.end and span.end > gold.start for gold in doc.spans)),
                    key=lambda span: span.score, reverse=True)
            now = time.monotonic()
            meter.record("stage1-candidate-mining", now - last_accounted, stage="stage1",
                         device="cuda", documents=min(first + len(batch), len(docs)))
            last_accounted = now
            print(f"stage1 proposer candidates: {min(first + len(batch), len(docs))}/{len(docs)} documents",
                  flush=True)
    finally:
        del predictor, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "mining_version": _CANDIDATE_MINING_VERSION,
                   "count": sum(map(len, output.values())),
                   "candidates": {doc_id: [span.to_dict() for span in rows]
                                  for doc_id, rows in output.items()}}
        part = cache_path.with_suffix(".part")
        part.write_text(json.dumps(payload, sort_keys=True)); part.replace(cache_path)
    return output


def _repeat_rows(rows, count: int, seed: int):
    if not rows:
        raise ValueError("Stage 1 generated no training questions")
    order = list(range(len(rows))); __import__("random").Random(seed).shuffle(order)
    return [rows[order[i % len(order)]] for i in range(count)]


class _LazyPackedRows:
    """Small LRU around deterministic packing; avoids retaining thousands of BlockMasks."""
    def __init__(self, count: int, builder, cache_size: int = 4):
        self.count, self.builder, self.cache_size = count, builder, cache_size
        self.cache = OrderedDict()

    def __len__(self): return self.count

    def __getitem__(self, index):
        if not 0 <= index < self.count: raise IndexError(index)
        if index not in self.cache:
            self.cache[index] = self.builder(index)
            while len(self.cache) > self.cache_size: self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]


def _run_manifest_hashes(run_dir: Path) -> dict | None:
    try:
        return json.loads((run_dir / "manifest.json").read_text()).get("hashes", {})
    except (OSError, ValueError):
        return None


def _archive_stale_run(run_dir: Path) -> Path | None:
    """Move a stale/partial run aside under a deterministic, non-clobbering name.

    Never removes an existing archive: if the base name is taken (two different stale contents
    share a questions hash), append ``-1``, ``-2``, ... so every archive stays recoverable."""
    if not run_dir.exists():
        return None
    hashes = _run_manifest_hashes(run_dir) or {}
    tag = str(hashes.get("questions", "nohash"))[:12]
    base = run_dir.parent / f"{run_dir.name}.stale-{tag}"
    dest, n = base, 1
    while dest.exists():
        dest = run_dir.parent / f"{base.name}-{n}"
        n += 1
    shutil.move(str(run_dir), str(dest))
    return dest


def _reuse_or_reset_run(run_dir: Path, expected: dict) -> bool:
    """Decide whether a prior run may be reused as-is; reset mismatches aside.

    Returns True only for a *completed* run whose manifest matches every expected hash. A
    matching but incomplete run is left in place so ``train`` resumes its own checkpoint. Any
    run whose manifest is absent or mismatched (completed or checkpoint-only) is archived
    recoverably, so ``_score_trained_run`` only ever sees the clean canonical model. Idempotent:
    once a clean matching run is written, later calls reuse it and archive nothing."""
    hashes = _run_manifest_hashes(run_dir)
    matches = hashes is not None and all(hashes.get(k) == v for k, v in expected.items())
    if matches:
        return (run_dir / "done").exists()
    if run_dir.exists() and any((run_dir / name).exists()
                                for name in ("done", "checkpoint.pt", "manifest.json")):
        _archive_stale_run(run_dir)
    return False


def _trainable_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(name.encode()); digest.update(str(value.dtype).encode())
        digest.update(str(tuple(value.shape)).encode()); digest.update(raw)
    return digest.hexdigest()


def _build_stage1_model(ctx, model_id: str, init_seed: int):
    from .model import S1DModel, apply_lora, prepare_tokenizer
    from .train import seed_everything
    seed_everything(init_seed)
    revision = "local-dry" if ctx["dry"] else ctx["config"]["models"][model_id]["revision"]
    if ctx["dry"]:
        from .latency import _DryTokenizer, _tiny_model
        tokenizer, model = _DryTokenizer(), _tiny_model()
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = S1DModel.from_pretrained(model_id, revision)
    token_rows = prepare_tokenizer(tokenizer, model); apply_lora(model, token_rows, rank=32)
    return tokenizer, model


def _train_stage1_run(ctx, questions, *, model_id: str, seed: int, layout: str,
                      window_count: int, run_dir: Path, branches_per_window: int = 1,
                      data_seed: int | None = None, init_seed: int | None = None,
                      order_seed: int | None = None, optimizer_condition: str = "old",
                      initial_state_path: Path | None = None, max_steps: int | None = None,
                      callback=None, callback_every: int | None = None,
                      callback_every_microbatches: int | None = None,
                      step_callback=None,
                      gradient_accumulation: int = 1,
                      unit_suffix: str | None = None,
                      lora_learning_rate: float | None = None,
                      pointer_learning_rate: float | None = None,
                      token_learning_rate: float | None = None,
                      extra_hashes: dict[str, str] | None = None) -> dict:
    from .data import pack_layout_ablation_questions, pack_training_questions, question_set_hash
    from .train import TrainConfig, train
    dry = ctx["dry"]
    data_seed = seed if data_seed is None else data_seed
    init_seed = seed if init_seed is None else init_seed
    order_seed = seed if order_seed is None else order_seed
    revision = "local-dry" if dry else ctx["config"]["models"][model_id]["revision"]
    token_budget = 4096 if dry else int(ctx["config"]["training"]["token_budget"])
    selected_count = min(window_count, 2 if branches_per_window > 1 else 8) if dry else window_count
    selected = _repeat_rows(questions, selected_count if branches_per_window == 1 else len(questions), data_seed)
    # The manifest hashes decide reuse/resume: a completed run is reused only when its questions,
    # revision and layout all match; mismatches (including a stale checkpoint) are archived aside.
    expected_hashes = {"questions": question_set_hash(selected), "revision": revision, "layout": layout,
                       "training_semantics": str(_TRAINING_SEMANTICS_VERSION),
                       "token_budget": str(token_budget)}
    expected_hashes.update(extra_hashes or {})
    if _reuse_or_reset_run(run_dir, expected_hashes):
        cached = json.loads((run_dir / "manifest.json").read_text()) | {"status": "cached"}
        checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
        steps = int(checkpoint["step"])
        warmup = (max(1, round(steps * float(cached["config"]["warmup_fraction"])))
                  if cached["config"]["optimizer_condition"] == "stable" else 0)
        cached.update(steps=steps, optimizer_updates=steps,
                      microbatches=int(checkpoint.get("microbatches_seen", 0)),
                      questions_seen=int(checkpoint.get("questions_seen", 0)),
                      warmup_updates=warmup)
        if initial_state_path is not None:
            cached["initial_state_sha256"] = expected_hashes["initial_state"]
        return cached
    tokenizer, model = _build_stage1_model(ctx, model_id, init_seed)
    loaded_initial_hash = None
    if initial_state_path is not None:
        initial = torch.load(initial_state_path, map_location="cpu", weights_only=False)
        state = initial["trainable_model"]
        result = model.load_state_dict(state, strict=False)
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        if result.unexpected_keys or trainable - set(state):
            raise RuntimeError(f"pilot initial state mismatch: missing trainable={sorted(trainable-set(state))[:3]} "
                               f"unexpected={result.unexpected_keys[:3]}")
        loaded_initial_hash = _trainable_digest({name: value for name, value in model.state_dict().items()
                                                 if name in trainable})
        if loaded_initial_hash != initial["sha256"]:
            raise RuntimeError("pilot initial state changed while loading")
    device = torch.device("cpu" if dry else "cuda")
    if branches_per_window > 1 and not dry:
        packed = _LazyPackedRows(selected_count, lambda index: pack_layout_ablation_questions(
            selected, tokenizer, windows=1, seed=seed + index, layout=layout,
            branches_per_window=branches_per_window, device=device, make_block_mask=False,
            keep_dense_mask=True, window_offset=index)[0])
    elif branches_per_window > 1:
        packed = pack_layout_ablation_questions(selected, tokenizer, windows=selected_count, seed=seed,
                                                layout=layout, branches_per_window=branches_per_window,
                                                device=device, make_block_mask=False)
    elif not dry:
        def build_one(index):
            item = pack_training_questions(
                [selected[index]], tokenizer, layout=layout, device=device, make_block_mask=False,
                keep_dense_mask=True)[0]
            return (*item, selected[index]) if initial_state_path is not None else item
        packed = _LazyPackedRows(selected_count, build_one)
    else:
        packed = pack_training_questions(selected, tokenizer, layout=layout, device=device,
                                         make_block_mask=not dry)
        if initial_state_path is not None:
            packed = [(*item, row) for item, row in zip(packed, selected)]
    pilot = ctx["config"].get("stage1_pilot", {})
    stage_name = ctx.get("stage", "stage1_pilot") if initial_state_path is not None else "stage1"
    unit = f"{model_id}-s{seed}-{layout}"
    if initial_state_path is not None:
        unit += f"-{unit_suffix or optimizer_condition}"
    config = TrainConfig(seed=seed, data_seed=data_seed, init_seed=init_seed, order_seed=order_seed,
                         token_budget=token_budget, epochs=1,
                         stage=stage_name, unit=unit,
                         cap_hours=ctx["config"]["stages"][stage_name]["cap_a100_hours"],
                         control_path=str(ctx["root"] / "CONTROL"),
                         checkpoint_every=int(pilot.get("checkpoint_every_updates", callback_every or 100)),
                         optimizer_condition=optimizer_condition,
                         lora_learning_rate=float(lora_learning_rate if lora_learning_rate is not None
                                                  else pilot.get("lora_learning_rate", 2e-4)),
                         pointer_learning_rate=float(pointer_learning_rate if pointer_learning_rate is not None
                                                     else pilot.get("pointer_learning_rate", 5e-5)),
                         token_learning_rate=float(token_learning_rate if token_learning_rate is not None
                                                   else pilot.get("token_learning_rate", 2e-4)),
                         warmup_fraction=float(pilot.get("warmup_fraction", 0.06)),
                         gradient_clip=(float(pilot.get("gradient_clip", 1.0))
                                        if optimizer_condition == "stable" else None),
                         gradient_accumulation=gradient_accumulation,
                         max_steps=max_steps)
    (ctx["root"] / "PHASE").write_text("GPU")
    try:
        result = train(model, packed, config, run_dir, ctx["root"] / "gpu_hours.jsonl",
                       hashes=expected_hashes, checkpoint_callback=callback,
                       callback_every=callback_every,
                       callback_every_microbatches=callback_every_microbatches,
                       step_callback=step_callback)
    finally:
        (ctx["root"] / "PHASE").write_text("CPU")
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result | ({"initial_state_sha256": loaded_initial_hash} if loaded_initial_hash else {})


def _training_questions(ctx, seed: int | None = None, *, data_seed: int | None = None,
                        require_candidate_cache: bool = False):
    from .data import generate_questions
    if data_seed is None:
        data_seed = 0 if seed is None else seed
    if ctx["dry"]:
        docs = _dry_docs(); candidates = _proposer_candidates(ctx["root"], docs, True)
    else:
        from ..model.train import TrainConfig as V1Config, training_docs
        docs, _ = training_docs(V1Config(variant="no-nemotron", seed=data_seed))
        counts = dict(__import__("collections").Counter(doc.dataset for doc in docs))
        print(f"stage1 training documents: {len(docs)} source_counts={counts}", flush=True)
        candidates = _proposer_candidates(ctx["root"], docs, False, cache_name="training",
                                          require_cache=require_candidate_cache)
    return generate_questions(docs, seed=data_seed, variant="no-nemotron", min_options=2, max_options=64,
                              hard_negative_hook=lambda doc: candidates.get(doc.doc_id, ()),
                              ledger_path=ctx["root"] / "ledger.jsonl")


def _unit_train_sizes(ctx, _out):
    models = list(ctx["config"]["models"])
    results = {}; models_root = ctx["root"] / "models" / "stage1"
    for seed in (1, 2):
        questions = _training_questions(ctx, seed)
        for model_id in models:
            size = model_id.rsplit("-", 1)[-1]
            key = f"{size}-s{seed}"
            results[key] = _train_stage1_run(ctx, questions, model_id=model_id, seed=seed,
                                             layout="shared", window_count=16000,
                                             run_dir=models_root / key)
    return {"implementation_version": 6, "variant": "no-nemotron", "windows_per_run": 16000,
            "training_semantics": _TRAINING_SEMANTICS_VERSION,
            "token_budget": ctx["config"]["training"]["token_budget"],
            "seeds": [1, 2], "runs": results}


def _dev_questions(ctx, *, descriptions: bool = True):
    from .data import span_evaluation_questions
    docs = _stage1_docs(ctx["dry"])
    candidates = _proposer_candidates(ctx["root"], docs, ctx["dry"], cache_name="calibration")
    rows = span_evaluation_questions(docs, candidates, labels=labels.load()["dev_labels"], seed=0,
                                     descriptions=descriptions, require_equal_negatives=False)
    if not rows or not any(row.hard_negative for row in rows):
        raise ValueError("Stage 1 dev questions require gold and not-PII candidate rows")
    return docs, rows


def _gretel_heldout_docs(ctx) -> list[Doc]:
    """The deterministic 2% Gretel dev slice that every training run excludes.

    ``training_docs`` trains on ``dev_slice(...)[1]``; this returns ``dev_slice(...)[0]``, so the
    calibration/sanity spans are genuinely held out yet in-domain for the no-nemotron models."""
    if ctx["dry"]:
        return _dry_docs()
    from .. import bench
    from ..data import loaders as L
    return bench.dev_slice(L.load("gretel", "train", purpose="eval"))[0]


def _seen_labels(config: dict) -> list[str]:
    """Trained-vocabulary option labels, with every held-out dev/test label removed so the
    calibration set can never leak a held-out name, and restricted to the frozen 56-way option
    set so every gold target is expressible by both the trained and the prompted scorer."""
    held = set(config["dev_labels"]) | set(config["test_labels"])
    option_names = {label.name for label in v2.c3_label_set(False).labels}
    return [name for name in labels.training_vocabulary(config)
            if name not in held and name in option_names]


def _seen_questions(ctx, *, descriptions: bool = True):
    """Held-out in-domain Gretel calibration questions over trained-vocabulary labels only."""
    from .data import span_evaluation_questions
    config = labels.load()
    seen = _seen_labels(config)
    docs = _gretel_heldout_docs(ctx)
    candidates = _proposer_candidates(ctx["root"], docs, ctx["dry"], cache_name="gretel-dev")
    return span_evaluation_questions(docs, candidates, labels=seen, seed=1, descriptions=descriptions,
                                     require_equal_negatives=False, pii_only=True)


def _probability_metrics(rows, probabilities, *, temperature: float = 1.0):
    from .infer import document_bootstrap, expected_calibration_error
    p = torch.as_tensor(probabilities, dtype=torch.float32)
    p = (p.clamp_min(1e-30).log() / temperature).softmax(-1).numpy()
    targets = [row.target for row in rows]; pred = p.argmax(1)
    gold = [i for i, row in enumerate(rows) if not row.hard_negative]
    by_label = defaultdict(list)
    for i in gold:
        by_label[rows[i].question.options[rows[i].target].name].append(float(pred[i] == targets[i]))
    macro = float(sum(map(__import__("numpy").mean, by_label.values())) / len(by_label))
    negative = [i for i, row in enumerate(rows) if row.hard_negative]
    none_accuracy = float(__import__("numpy").mean([pred[i] == targets[i] for i in negative]))
    by_doc = defaultdict(list)
    for i in gold: by_doc[rows[i].doc_id].append(float(pred[i] == targets[i]))
    return {"macro_accuracy": macro, "not_pii_accuracy": none_accuracy,
            "ece": expected_calibration_error(p, targets),
            "doc_bootstrap": document_bootstrap(dict(by_doc), seed=0, samples=100 if len(rows) < 20 else 2000),
            "probabilities": p.tolist()}


def _sanity_metrics(rows, probabilities, *, temperature: float = 1.0) -> dict:
    """In-domain accuracy of a trained model on held-out Gretel gold spans.

    Accurate here means a Nemotron dev-label failure is genuine cross-domain generalization;
    inaccurate here means the evaluation/packing is broken. Reported separately from the
    Nemotron dev result so the two are never conflated. Handles a gold-only set (no proposer
    negatives) without NaNs: ``not_pii_accuracy`` is ``None`` when there are no negative rows."""
    import numpy as np
    total = len(rows)
    gold = [i for i, row in enumerate(rows) if not row.hard_negative]
    negative = [i for i, row in enumerate(rows) if row.hard_negative]
    if not gold:
        return {"questions": total, "gold_questions": 0, "negative_questions": len(negative),
                "macro_accuracy": None, "accuracy": None, "not_pii_rate": None,
                "not_pii_accuracy": None, "per_label_accuracy": {}, "prediction_distribution": {},
                "labels": []}
    p = torch.as_tensor(probabilities, dtype=torch.float32)
    p = (p.clamp_min(1e-30).log() / temperature).softmax(-1).numpy()
    pred = p.argmax(1)
    options = rows[0].question.options
    not_pii_index = next((i for i, option in enumerate(options) if option.name == labels.NOT_PII), None)
    by_label = defaultdict(list)
    for i in gold:
        by_label[rows[i].question.options[rows[i].target].name].append(float(pred[i] == rows[i].target))
    distribution: dict[str, int] = {}
    for i in gold:
        name = options[int(pred[i])].name
        distribution[name] = distribution.get(name, 0) + 1
    macro = float(np.mean([np.mean(values) for values in by_label.values()]))
    accuracy = float(np.mean([pred[i] == rows[i].target for i in gold]))
    not_pii_rate = (float(np.mean([pred[i] == not_pii_index for i in gold]))
                    if not_pii_index is not None else None)
    not_pii_accuracy = (float(np.mean([pred[i] == rows[i].target for i in negative])) if negative else None)
    return {"questions": total, "gold_questions": len(gold), "negative_questions": len(negative),
            "macro_accuracy": macro, "accuracy": accuracy, "not_pii_rate": not_pii_rate,
            "not_pii_accuracy": not_pii_accuracy,
            "per_label_accuracy": {name: float(np.mean(values)) for name, values in by_label.items()},
            "prediction_distribution": distribution, "labels": sorted(by_label)}


def _collapse_flag(metrics: dict, thresholds: dict) -> tuple[bool, list[str]]:
    gold = max(1, int(metrics["gold_questions"])); distribution = metrics["prediction_distribution"]
    dominant = max(distribution.values(), default=0) / gold
    reasons = []
    if metrics["accuracy"] < float(thresholds["gold_accuracy_below"]):
        reasons.append("gold_accuracy")
    if metrics["not_pii_rate"] > float(thresholds["not_pii_rate_above"]):
        reasons.append("not_pii_rate")
    if dominant > float(thresholds["single_option_rate_above"]):
        reasons.append("single_option_rate")
    return bool(reasons), reasons


def _prepare_pilot_eval_tokenizer(tokenizer):
    """Register the reserved packing tokens on the tokenizer used by checkpoint evaluation."""
    from .model import prepare_tokenizer
    prepare_tokenizer(tokenizer)
    return tokenizer


def _pilot_probe_rows(rows, count: int = 32):
    """Deterministic balanced probe, preferring document-dense rows to minimize windows."""
    def take(candidates, n):
        by_doc = defaultdict(list)
        for row in candidates:
            by_doc[row.doc_id].append(row)
        output = []
        for _doc_id, doc_rows in sorted(by_doc.items(), key=lambda item: (-len(item[1]), item[0])):
            output.extend(doc_rows[:max(0, n - len(output))])
            if len(output) >= n:
                break
        return output
    if count < 2 or count % 2:
        raise ValueError("pilot probe size must be a positive even number")
    half = count // 2
    gold = take([row for row in rows if not row.hard_negative], half)
    negative = take([row for row in rows if row.hard_negative], half)
    if len(gold) != half or len(negative) != half:
        raise ValueError(f"pilot probe needs {half} gold and {half} not-PII rows; "
                         f"found {len(gold)} and {len(negative)}")
    return gold + negative


def _pilot_probe_metrics(rows, probabilities) -> dict:
    if len(rows) != len(probabilities) or not rows:
        raise ValueError("pilot probe rows and probabilities must have the same nonzero length")
    picks = []
    not_pii_picks = 0
    not_pii_probabilities = []
    for row, values in zip(rows, probabilities):
        p = torch.as_tensor(values, dtype=torch.float32)
        if p.numel() != len(row.question.options):
            raise ValueError("pilot probe probability width does not match its question")
        pred = int(p.argmax())
        not_pii = next(i for i, option in enumerate(row.question.options)
                       if option.name == labels.NOT_PII)
        picks.append(row.question.options[pred].name)
        not_pii_picks += pred == not_pii
        not_pii_probabilities.append(float(p[not_pii]))
    counts = __import__("collections").Counter(picks)
    dominant, dominant_count = counts.most_common(1)[0]
    return {"questions": len(rows),
            "not_pii_pick_rate": not_pii_picks / len(rows),
            "dominant_option": dominant,
            "dominant_option_share": dominant_count / len(rows),
            "mean_not_pii_probability": sum(not_pii_probabilities) / len(rows)}


def _unit_stage1_pilot(ctx, out, *, confirm_4b: bool = False):
    """Paired equal-exposure stability pilot with frozen data, initialization, and RNG."""
    from .train import _trainable_state
    spec = ctx["config"]["stage1_pilot"]
    data_seed = int(spec["data_seed"]); order_seed = int(spec["order_seed"])
    sanity_every = 8 if ctx["dry"] else int(spec["sanity_every_microbatches"])
    sizes = ["4B"] if confirm_4b else ["1.7B"]
    model_ids = {model_id.rsplit("-", 1)[-1]: model_id for model_id in ctx["config"]["models"]}
    questions = _training_questions(ctx, data_seed=data_seed)
    sanity_rows = _seen_questions(ctx, descriptions=True)
    probe_count = 4 if ctx["dry"] else int(spec.get("probe_questions", 32))
    probe_rows = _pilot_probe_rows(sanity_rows, probe_count)
    from .data import question_set_hash, training_batch_composition
    probe_hash = question_set_hash(probe_rows)
    conditions = {
        "stable": {"gradient_accumulation": 1,
                   "lora_learning_rate": float(spec["lora_learning_rate"]),
                   "pointer_learning_rate": float(spec["pointer_learning_rate"]),
                   "token_learning_rate": float(spec["token_learning_rate"])},
        "stable_acc8": {"gradient_accumulation": 8,
                        "lora_learning_rate": float(spec["lora_learning_rate"]),
                        "pointer_learning_rate": float(spec["pointer_learning_rate"]),
                        "token_learning_rate": float(spec["token_learning_rate"])},
        "stable_lr5": {"gradient_accumulation": 1,
                       "lora_learning_rate": 5e-5,
                       "pointer_learning_rate": 5e-5,
                       "token_learning_rate": 5e-5},
    }
    if confirm_4b:
        conditions = {"stable": conditions["stable"]}
    optimizer_common = {"warmup_fraction": float(spec["warmup_fraction"]),
                        "gradient_clip": float(spec["gradient_clip"]),
                        "checkpoint_every_updates": int(spec["checkpoint_every_updates"])}
    output = {"implementation_version": 2, "pilot_semantics": _PILOT_SEMANTICS_VERSION,
              "mode": "4b_final_confirmation" if confirm_4b else "1.7b_optimizer_pilot",
              "data_seed": data_seed, "order_seed": order_seed,
              "windows": int(spec["windows"]), "full_selection": True,
              "sanity_every_microbatches": sanity_every,
              "conditions": conditions, "optimizer_common": optimizer_common,
              "probe": {"questions": len(probe_rows), "question_set_sha256": probe_hash,
                        "timing": "immediately after each optimizer update"},
              "fixed_rule": ("healthy final checkpoint in both init seeds" if confirm_4b else
                             "no collapsed sanity checkpoint after warmup, in both init seeds"),
              "selection_rule": "final_checkpoint" if confirm_4b else "no_post_warmup_collapse",
              "collapse_thresholds": spec["collapse"], "sizes": sizes, "runs": {}}
    try:
        previous = json.loads(out.read_text())
        if previous.get("implementation_version") != output["implementation_version"]:
            legacy = out.with_name(f"{out.stem}-v{previous.get('implementation_version', 'unknown')}.json")
            if not legacy.exists():
                shutil.copy2(out, legacy)
        identity = ("implementation_version", "pilot_semantics", "mode", "data_seed", "order_seed",
                    "windows", "full_selection", "sanity_every_microbatches", "conditions",
                    "optimizer_common", "probe", "fixed_rule", "selection_rule",
                    "collapse_thresholds", "sizes")
        if all(previous.get(key) == output.get(key) for key in identity):
            output = previous
    except (OSError, ValueError):
        pass
    pilot_root = ctx["root"] / "models" / ("stage1-pilot-4b-v1" if confirm_4b
                                             else "stage1-pilot-v2")
    initial_root = ctx["root"] / "models" / "stage1-pilot" / "initial"
    for size in sizes:
        model_id = model_ids[size]
        if ctx["dry"]:
            from .latency import _DryTokenizer
            eval_tokenizer = _prepare_pilot_eval_tokenizer(_DryTokenizer())
        else:
            from transformers import AutoTokenizer
            eval_tokenizer = _prepare_pilot_eval_tokenizer(AutoTokenizer.from_pretrained(
                model_id, revision=ctx["config"]["models"][model_id]["revision"]))
        for init_seed in map(int, spec["init_seeds"]):
            pair = f"{size}-init{init_seed}"
            revision = "local-dry" if ctx["dry"] else ctx["config"]["models"][model_id]["revision"]
            init_path = (initial_root /
                         f"v{_TRAINING_SEMANTICS_VERSION}-{revision[:12]}-{pair}.pt")
            if not init_path.exists():
                _tokenizer, initial_model = _build_stage1_model(ctx, model_id, init_seed)
                initial_state = _trainable_state(initial_model)
                payload = {"version": 1, "model_id": model_id, "revision": revision,
                           "training_semantics": _TRAINING_SEMANTICS_VERSION, "init_seed": init_seed,
                           "trainable_model": initial_state,
                           "sha256": _trainable_digest(initial_state)}
                init_path.parent.mkdir(parents=True, exist_ok=True)
                part = init_path.with_suffix(".part"); torch.save(payload, part); os.replace(part, init_path)
                del initial_model
                if torch.cuda.is_available(): torch.cuda.empty_cache()
            initial = torch.load(init_path, map_location="cpu", weights_only=False)
            if (initial.get("model_id"), initial.get("revision"), initial.get("init_seed")) != \
                    (model_id, revision, init_seed):
                raise RuntimeError(f"pilot initial-state metadata mismatch at {init_path}")
            pair_output = output["runs"].setdefault(
                pair, {"initial_state_sha256": initial["sha256"], "conditions": {}})
            if pair_output["initial_state_sha256"] != initial["sha256"]:
                raise RuntimeError("saved pilot output refers to a different initial state")
            for condition, condition_spec in conditions.items():
                condition_output = pair_output["conditions"].setdefault(condition, {"checkpoints": []})
                checkpoints = condition_output["checkpoints"]
                optimizer_spec = {
                    "condition": condition, "full_selection": True,
                    "sanity_every_microbatches": sanity_every,
                    **optimizer_common,
                    **condition_spec,
                }
                optimizer_sha = hashlib.sha256(json.dumps(
                    optimizer_spec, sort_keys=True).encode()).hexdigest()
                step_identity = {"pilot_semantics": _PILOT_SEMANTICS_VERSION, "pair": pair,
                                 "condition": condition, "initial_state": initial["sha256"],
                                 "probe": probe_hash, "optimizer": optimizer_sha,
                                 "data_seed": data_seed, "order_seed": order_seed,
                                 "windows": int(spec["windows"])}
                step_fingerprint = hashlib.sha256(json.dumps(
                    step_identity, sort_keys=True).encode()).hexdigest()
                step_log = (ctx["root"] / "stores" / "stage1-pilot-step-logs" /
                            f"v{_PILOT_SEMANTICS_VERSION}-{pair}-{condition}-"
                            f"{step_fingerprint[:12]}.jsonl")
                step_log.parent.mkdir(parents=True, exist_ok=True)
                logged_steps = set()
                if step_log.exists():
                    previous_steps = [json.loads(line) for line in step_log.read_text().splitlines()
                                      if line.strip()]
                    if any(row.get("run_fingerprint") != step_fingerprint for row in previous_steps):
                        raise RuntimeError(f"pilot step-log identity mismatch at {step_log}")
                    logged_steps = {row["step"] for row in previous_steps}
                condition_output["step_log"] = str(step_log.relative_to(ctx["root"]))
                probe_cache = {}
                step_handle = step_log.open("a", encoding="utf-8", buffering=1)

                def log_step(model, step, stats, batch, *, pair=pair, condition=condition,
                             logged_steps=logged_steps, probe_cache=probe_cache,
                             step_handle=step_handle):
                    if step in logged_steps:
                        return
                    model.eval()
                    probabilities = _score_pilot_probe(model, eval_tokenizer, probe_rows, probe_cache)
                    composition = training_batch_composition([item[2] for item in batch])
                    row = {"version": 1, "pilot_semantics": _PILOT_SEMANTICS_VERSION,
                           "run_fingerprint": step_fingerprint,
                           "pair": pair, "condition": condition, "step": step,
                           **stats, "batch": composition,
                           "probe": _pilot_probe_metrics(probe_rows, probabilities)}
                    step_handle.write(json.dumps(row, sort_keys=True) + "\n")
                    logged_steps.add(step)

                def evaluate(model, step, stats, *, pair=pair, condition=condition,
                             checkpoints=checkpoints, initial_sha=initial["sha256"]):
                    if any(row["step"] == step for row in checkpoints):
                        return
                    model.eval()
                    identity = {"kind": "stage1-pilot", "pair": pair, "condition": condition,
                                "step": step, "questions_seen": stats["questions_seen"],
                                "initial_state_sha256": initial_sha,
                                "pilot_semantics": _PILOT_SEMANTICS_VERSION}
                    fingerprint = _eval_fingerprint(sanity_rows, identity)
                    probabilities = _score_loaded_trained(
                        ctx, model, eval_tokenizer, sanity_rows, layout="shared",
                        key=f"pilot-v{_PILOT_SEMANTICS_VERSION}-{pair}-{condition}-step{step}", split="sanity",
                        fingerprint=fingerprint)
                    metrics = _sanity_metrics(sanity_rows, probabilities)
                    collapsed, reasons = _collapse_flag(metrics, spec["collapse"])
                    checkpoints.append({"step": step, **stats, "metrics": metrics,
                                        "collapsed": collapsed, "collapse_reasons": reasons})
                    _atomic_json(out, output)

                run_dir = pilot_root / pair / condition
                try:
                    result = _train_stage1_run(
                        ctx, questions, model_id=model_id, seed=data_seed, data_seed=data_seed,
                        init_seed=init_seed, order_seed=order_seed, layout="shared",
                        window_count=int(spec["windows"]), run_dir=run_dir,
                        optimizer_condition="stable", unit_suffix=condition, initial_state_path=init_path,
                        max_steps=None, callback=evaluate,
                        callback_every_microbatches=sanity_every,
                        step_callback=log_step,
                        gradient_accumulation=int(condition_spec["gradient_accumulation"]),
                        lora_learning_rate=float(condition_spec["lora_learning_rate"]),
                        pointer_learning_rate=float(condition_spec["pointer_learning_rate"]),
                        token_learning_rate=float(condition_spec["token_learning_rate"]),
                        extra_hashes={"pilot_condition": condition, "initial_state": initial["sha256"],
                                      "order_seed": str(order_seed),
                                      "pilot_semantics": str(_PILOT_SEMANTICS_VERSION),
                                      "optimizer_spec": optimizer_sha})
                finally:
                    step_handle.close()
                condition_output["training"] = result
                eligible = [row for row in checkpoints if row["step"] > result["warmup_updates"]]
                condition_output["fixed_after_warmup"] = bool(eligible) and not any(
                    row["collapsed"] for row in eligible)
                if result["initial_state_sha256"] != initial["sha256"]:
                    raise RuntimeError("paired conditions did not load the registered initial state")
                _atomic_json(out, output)
    if confirm_4b:
        output["condition_outcomes"] = {
            "stable": {"healthy_final_both_seeds": all(
                not output["runs"][pair]["conditions"]["stable"]["checkpoints"][-1]["collapsed"]
                for pair in output["runs"] if pair.startswith("4B-init"))}}
    else:
        output["condition_outcomes"] = {
            condition: {"fixed_both_seeds": all(
                output["runs"][pair]["conditions"][condition].get("fixed_after_warmup", False)
                for pair in output["runs"] if pair.startswith("1.7B-init"))}
            for condition in conditions
        }
    _atomic_json(out, output)
    return output


def _inference_probabilities(model, window, device: torch.device, *, bf16: bool):
    """Run pointer inference with the same mixed-precision contract used for training."""
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
        return model(window).probabilities[0].float().cpu().tolist()


_EVAL_CACHE_VERSION = 2


def _eval_fingerprint(rows, identity: dict) -> str:
    from .data import question_set_hash
    value = {"version": _EVAL_CACHE_VERSION, "questions": question_set_hash(rows), **identity}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _eval_cache_path(ctx, key: str, split: str) -> Path:
    safe = key.replace("/", "_").replace(":", "_")
    return ctx["root"] / "stores" / "stage1-dev-eval" / f"{safe}-{split}.json"


def _read_eval_cache(path: Path, fingerprint: str, count: int):
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    probabilities = value.get("probabilities")
    if (value.get("version") != _EVAL_CACHE_VERSION or value.get("fingerprint") != fingerprint
            or not isinstance(probabilities, list) or len(probabilities) != count):
        return None
    return value


def _write_eval_cache(path: Path, fingerprint: str, probabilities, completed: int, total: int) -> None:
    _atomic_json(path, {"version": _EVAL_CACHE_VERSION, "fingerprint": fingerprint,
                        "completed": completed, "total": total, "probabilities": probabilities})


def _check_eval_stop(ctx) -> None:
    control = ctx["root"] / "CONTROL"
    if control.exists() and control.read_text().strip().upper() == "STOP":
        raise InterruptedError("STOP requested")


def _evaluation_window_specs(rows, tokenizer, *, layout: str):
    """Group questions sharing one state slice into real multi-branch inference windows."""
    from .data import question_pack_parts
    encoded = {}
    groups = OrderedDict()
    for index, row in enumerate(rows):
        document_key = (row.doc_id, row.state)
        if document_key not in encoded:
            encoded[document_key] = tokenizer(row.state, add_special_tokens=False,
                                              return_offsets_mapping=True, truncation=False)
        state, options, branch = question_pack_parts(row, tokenizer, encoded=encoded[document_key])
        groups.setdefault((row.doc_id, state, options), []).append((index, branch))
    limit = 64 if layout == "shared" else 16
    specs = []
    for (_doc_id, state, options), entries in groups.items():
        for first in range(0, len(entries), limit):
            chunk = entries[first:first + limit]
            specs.append((state, options, tuple(branch for _index, branch in chunk),
                          tuple(index for index, _branch in chunk)))
    return specs


def _score_pilot_probe(model, tokenizer, rows, cache: dict):
    """Score the fixed per-step probe in memory, without disk I/O or RNG side effects."""
    from .packer import pack_window
    device = next(model.parameters()).device
    if "packed" not in cache:
        specs = _evaluation_window_specs(rows, tokenizer, layout="shared")
        packed = [pack_window(tokenizer, state, options, branches, state_tokens=512,
                              layout="shared", device=device, make_block_mask=False,
                              keep_dense_mask=True)
                  for state, options, branches, _indices in specs]
        cache.update(specs=specs, packed=packed)
    probabilities = [None] * len(rows)
    by_length = OrderedDict()
    for index, packed in enumerate(cache["packed"]):
        by_length.setdefault(len(packed.input_ids), []).append((index, packed))
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                         enabled=device.type == "cuda"):
        outputs = [None] * len(cache["packed"])
        for group in by_length.values():
            indices, windows = zip(*group)
            group_outputs = (model.forward_many(list(windows)) if hasattr(model, "forward_many")
                             else [model(window) for window in windows])
            for index, output in zip(indices, group_outputs):
                outputs[index] = output.probabilities.float().cpu().tolist()
    for (_state, _options, _branches, indices), values in zip(cache["specs"], outputs):
        for row_index, row_values in zip(indices, values):
            probabilities[row_index] = row_values
    if any(values is None for values in probabilities):
        raise RuntimeError("pilot probe scoring left an incomplete row")
    return probabilities


def _score_loaded_trained(ctx, model, tokenizer, rows, *, layout: str, key: str, split: str,
                          fingerprint: str):
    from .packer import pack_window
    path = _eval_cache_path(ctx, key, split)
    specs = _evaluation_window_specs(rows, tokenizer, layout=layout)
    cached = _read_eval_cache(path, fingerprint, len(rows))
    probabilities = cached["probabilities"] if cached else [None] * len(rows)
    completed = min(int(cached.get("completed", 0)), len(specs)) if cached else 0
    if completed == len(specs) and all(row is not None for row in probabilities):
        print(f"{key} {split}: cached {len(rows)} questions in {len(specs)} windows", flush=True)
        return probabilities
    device = next(model.parameters()).device
    meter = GPUHours(ctx["root"] / "gpu_hours.jsonl", 0, "stage1")
    last_accounted = time.monotonic()
    batch_size = 8 if layout == "shared" else 1
    with torch.no_grad():
        for first in range(completed, len(specs), batch_size):
            _check_eval_stop(ctx)
            chunk = specs[first:first + batch_size]
            packed_rows = [pack_window(tokenizer, state, options, branches, state_tokens=512,
                                       layout=layout, device=device, make_block_mask=False,
                                       keep_dense_mask=True)
                           for state, options, branches, _indices in chunk]
            by_length = OrderedDict()
            for local_index, packed in enumerate(packed_rows):
                by_length.setdefault(len(packed.input_ids), []).append((local_index, packed))
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                chunk_outputs = [None] * len(packed_rows)
                for same_length in by_length.values():
                    local_indices, windows = zip(*same_length)
                    if hasattr(model, "forward_many"):
                        outputs = model.forward_many(list(windows))
                    else:  # small test doubles
                        outputs = [model(window) for window in windows]
                    for local_index, output in zip(local_indices, outputs):
                        chunk_outputs[local_index] = output.probabilities.float().cpu().tolist()
            for (_state, _options, _branches, indices), batch in zip(chunk, chunk_outputs):
                for row_index, values in zip(indices, batch):
                    probabilities[row_index] = values
            done = first + len(chunk)
            if done % 32 == 0 or done == len(specs):
                now = time.monotonic()
                meter.record(f"dev-eval-{key}-{split}", now - last_accounted, stage="stage1",
                             device=str(device), windows=done, total_windows=len(specs))
                last_accounted = now
                _write_eval_cache(path, fingerprint, probabilities, done, len(specs))
                print(f"{key} {split}: {done}/{len(specs)} windows, {len(rows)} questions", flush=True)
    return probabilities


def _score_trained_sets(ctx, row_sets: dict[str, list], *, model_id: str, run_dir: Path,
                        layout: str = "shared"):
    from .model import S1DModel, apply_lora, prepare_tokenizer
    revision = "local-dry" if ctx["dry"] else ctx["config"]["models"][model_id]["revision"]
    manifest = json.loads((run_dir / "manifest.json").read_text()) if (run_dir / "manifest.json").exists() else {}
    identity = {"kind": "trained", "model": model_id, "revision": revision, "layout": layout,
                "packing": {"state_tokens": 512, "stride": 384,
                            "branches_per_window": 64 if layout == "shared" else 16,
                            "window_batch_size": 8 if layout == "shared" else 1},
                "run_hashes": manifest.get("hashes", {}),
                "step": (run_dir / "done").read_text().strip() if (run_dir / "done").exists() else "dry"}
    key = f"trained-{model_id.rsplit('-', 1)[-1]}-{run_dir.name}-{layout}"
    fingerprints = {split: _eval_fingerprint(rows, identity) for split, rows in row_sets.items()}
    output = {}
    missing = {}
    for split, rows in row_sets.items():
        cached = _read_eval_cache(_eval_cache_path(ctx, key, split), fingerprints[split], len(rows))
        if cached and cached.get("completed") == cached.get("total") \
                and all(row is not None for row in cached["probabilities"]):
            output[split] = cached["probabilities"]
        else:
            missing[split] = rows
    if not missing:
        return output
    if ctx["dry"]:
        from .latency import _DryTokenizer, _tiny_model
        tokenizer, model = _DryTokenizer(), _tiny_model()
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = S1DModel.from_pretrained(model_id, revision)
    indices = prepare_tokenizer(tokenizer, model); apply_lora(model, indices, rank=32)
    checkpoint = run_dir / "checkpoint.pt"
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["trainable_model"],
                              strict=False)
    device = torch.device("cpu" if ctx["dry"] else "cuda"); model.to(device).eval()
    (ctx["root"] / "PHASE").write_text("GPU" if device.type == "cuda" else "CPU")
    try:
        for split, rows in missing.items():
            output[split] = _score_loaded_trained(ctx, model, tokenizer, rows, layout=layout,
                                                  key=key, split=split,
                                                  fingerprint=fingerprints[split])
    finally:
        (ctx["root"] / "PHASE").write_text("CPU")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for split, probabilities in output.items():
        if any(row is None for row in probabilities):
            raise RuntimeError(f"{key} {split}: incomplete probability cache")
    return output


def _score_trained_run(ctx, rows, *, model_id: str, run_dir: Path, layout: str = "shared"):
    return _score_trained_sets(ctx, {"rows": rows}, model_id=model_id,
                               run_dir=run_dir, layout=layout)["rows"]


# Bumped whenever the prompted probe's prompt/scoring semantics change, so the eval cache
# rejects probabilities produced by the old (option-less, raw-label-likelihood) baseline. v2
# shows every option with its description under a unique numeric answer code, instructs exactly
# one code with /no_think, scores only the answer-code distribution, and uses the same
# 512-token span-centered window as the Stage 0 probe and the trained packer.
_PROMPTED_SEMANTICS_VERSION = 3


def _prompted_answer_codes(count: int) -> list[str]:
    """Unique, tokeniser-robust answer codes. A-Z cannot address 56 options; numeric codes can,
    and ``prompted_option_distributions`` scores multi-token codes (e.g. ``"10"``) correctly."""
    if count < 2:
        raise ValueError("at least two options are required")
    return [f"{i + 1:02d}" for i in range(count)]


def _prompted_choice_prompt(tokenizer, state: str, surface: str, options, codes) -> str:
    """Mirror the valid Stage 0 probe: list every option with its description under its answer
    code, instruct exactly one code, and disable thinking."""
    lines = []
    for code, option in zip(codes, options):
        name = option.name.replace("_", " ")
        lines.append(f"{code}. {name}: {option.description}" if option.description else f"{code}. {name}")
    text = ("Classify the marked span using exactly one option number. /no_think\n\n"
            f"Document:\n{state}\n\nSpan: {surface}\n\nOptions:\n" + "\n".join(lines)
            + "\n\nAnswer with the single option number.")
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
    return text + "\nAnswer:"


def _prompted_span_window(tokenizer, row, state_tokens: int) -> tuple[str, str]:
    """The same 512-token span-centered document window the Stage 0 probe uses, so the prompted
    baseline never scores against an unbounded full document."""
    from .stage0 import span_window
    start, end = row.source_span
    window = span_window(tokenizer, {"doc_id": row.doc_id, "state": row.state,
                                     "start": start, "end": end}, state_tokens)
    return window["state"], window["state"][window["start"]:window["end"]]


def _score_prompted_sets(ctx, row_sets: dict[str, list], model_id: str):
    if ctx["dry"]:
        return {split: [[1 / len(row.question.options)] * len(row.question.options) for row in rows]
                for split, rows in row_sets.items()}
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .infer import prompted_option_distributions
    revision = ctx["config"]["models"][model_id]["revision"]
    state_tokens = ctx["config"]["state_tokens"]
    identity = {"kind": "prompted", "model": model_id, "revision": revision,
                "prompt_semantics": _PROMPTED_SEMANTICS_VERSION, "state_tokens": state_tokens,
                "scoring": "answer_code_distribution"}
    key = f"prompted-{model_id.rsplit('-', 1)[-1]}"
    fingerprints = {split: _eval_fingerprint(rows, identity) for split, rows in row_sets.items()}
    output, missing = {}, {}
    for split, rows in row_sets.items():
        cached = _read_eval_cache(_eval_cache_path(ctx, key, split), fingerprints[split], len(rows))
        if cached and cached.get("completed") == len(rows) and all(p is not None for p in cached["probabilities"]):
            output[split] = cached["probabilities"]
        else:
            missing[split] = rows
    if not missing:
        return output
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16).cuda().eval()
    (ctx["root"] / "PHASE").write_text("GPU")
    meter = GPUHours(ctx["root"] / "gpu_hours.jsonl", 0, "stage1")
    try:
        for split, rows in missing.items():
            path = _eval_cache_path(ctx, key, split)
            cached = _read_eval_cache(path, fingerprints[split], len(rows))
            probabilities = cached["probabilities"] if cached else [None] * len(rows)
            completed = min(int(cached.get("completed", 0)), len(rows)) if cached else 0
            last_accounted = time.monotonic()
            prompt_batch_size = 8
            for first in range(completed, len(rows), prompt_batch_size):
                _check_eval_stop(ctx)
                chunk = rows[first:first + prompt_batch_size]
                options = chunk[0].question.options
                names = tuple(option.name for option in options)
                if any(tuple(o.name for o in row.question.options) != names for row in chunk):
                    raise ValueError("prompt batches require one shared option set")
                codes = _prompted_answer_codes(len(options))
                prompts = []
                for row in chunk:
                    state, surface = _prompted_span_window(tokenizer, row, state_tokens)
                    prompts.append(_prompted_choice_prompt(tokenizer, state, surface, options, codes))
                # Score only the answer-code distribution, never the raw option-name likelihood.
                batch = prompted_option_distributions(model, tokenizer, prompts, codes,
                                                       device="cuda", max_expanded_batch=16).tolist()
                probabilities[first:first + len(chunk)] = batch
                done = first + len(chunk)
                if done % 32 == 0 or done == len(rows):
                    now = time.monotonic()
                    meter.record(f"dev-eval-{key}-{split}", now - last_accounted, stage="stage1",
                                 device="cuda", questions=done, total_questions=len(rows))
                    last_accounted = now
                    _write_eval_cache(path, fingerprints[split], probabilities, done, len(rows))
                    print(f"{key} {split}: {done}/{len(rows)} questions", flush=True)
            output[split] = probabilities
    finally:
        (ctx["root"] / "PHASE").write_text("CPU")
        del model
        torch.cuda.empty_cache()
    for split, probabilities in output.items():
        if any(row is None for row in probabilities):
            raise RuntimeError(f"{key} {split}: incomplete probability cache")
    return output


def _score_prompted_base(ctx, rows, model_id: str):
    return _score_prompted_sets(ctx, {"rows": rows}, model_id)["rows"]


def _unit_layout_ablation(ctx, _out):
    _, rows = _dev_questions(ctx, descriptions=False)
    questions = _training_questions(ctx, 11)
    model_id = "Qwen/Qwen3-0.6B"; root = ctx["root"] / "models" / "stage1" / "layout"
    metrics = {}
    for layout in ("shared", "kev"):
        for seed in (1, 2):
            key = f"{layout}-s{seed}"
            run_dir = root / key
            _train_stage1_run(ctx, questions, model_id=model_id, seed=seed, layout=layout,
                              window_count=4000, run_dir=run_dir, branches_per_window=16)
            metrics[key] = _probability_metrics(rows, _score_trained_run(ctx, rows, model_id=model_id,
                                                                          run_dir=run_dir, layout=layout))
    differences = {}
    for row_index, row in enumerate(rows):
        shared = sum(int(max(range(len(metrics[f"shared-s{s}"]["probabilities"][row_index])),
                             key=metrics[f"shared-s{s}"]["probabilities"][row_index].__getitem__) == row.target)
                     for s in (1, 2)) / 2
        kev = sum(int(max(range(len(metrics[f"kev-s{s}"]["probabilities"][row_index])),
                          key=metrics[f"kev-s{s}"]["probabilities"][row_index].__getitem__) == row.target)
                  for s in (1, 2)) / 2
        differences.setdefault(row.doc_id, []).append(kev - shared)
    from .infer import document_bootstrap
    interval = document_bootstrap(differences, seed=17, samples=100 if ctx["dry"] else 2000, side="lower")
    return {"implementation_version": 6, "windows_per_run": 4000, "sampled_spans_per_window": 16,
            "metrics": metrics, "difference": interval,
            "stage2_layout": "kev" if interval["one_sided_lower_95"] > 0.02 else "shared"}


def _unit_dev_eval(ctx, _out):
    _, rows = _dev_questions(ctx, descriptions=True)
    # Held-out in-domain Gretel spans: temperatures are fitted here (never on Nemotron, which the
    # no-nemotron models never saw) and the sanity check below reports in-domain accuracy on them.
    calib_rows = _seen_questions(ctx, descriptions=True)
    models_root = ctx["root"] / "models" / "stage1"
    trained, prompted, sanity = {}, {}, {}
    from .infer import fit_temperatures

    def _temperature(probabilities):
        return fit_temperatures([torch.tensor(p).clamp_min(1e-30).log() for p in probabilities],
                                [row.target for row in calib_rows],
                                ["choice"] * len(calib_rows))["choice"]

    for model_id in ctx["config"]["models"]:
        size = model_id.rsplit("-", 1)[-1]
        prompted_scores = _score_prompted_sets(ctx, {"calib": calib_rows, "dev": rows}, model_id)
        prompted_temp = _temperature(prompted_scores["calib"])
        prompted[size] = _probability_metrics(rows, prompted_scores["dev"], temperature=prompted_temp)
        prompted[size]["temperature"] = prompted_temp
        for seed in (1, 2):
            key = f"{size}-s{seed}"
            run_dir = models_root / key
            scores = _score_trained_sets(ctx, {"calib": calib_rows, "dev": rows}, model_id=model_id,
                                         run_dir=run_dir)
            temperature = _temperature(scores["calib"])
            trained[key] = _probability_metrics(rows, scores["dev"], temperature=temperature)
            trained[key]["temperature"] = temperature
            sanity[key] = _sanity_metrics(calib_rows, scores["calib"], temperature=temperature)
    latency = json.loads((ctx["root"] / "stores" / "stage0-latency.json").read_text()) if not ctx["dry"] else {
        "proposer_included": True,
        "models": {name: {"p95_ms_per_window": 1.0} for name in ctx["config"]["models"]}}
    _require_end_to_end_latency(latency, ctx["dry"])
    pooled = {}
    for model_id in ctx["config"]["models"]:
        size = model_id.rsplit("-", 1)[-1]
        runs = [trained[f"{size}-s{s}"] for s in (1, 2)]
        pooled[size] = {key: sum(run[key] for run in runs) / 2 for key in ("macro_accuracy", "ece")}
        latency_row = latency["models"].get(model_id, latency["models"].get(size, {}))
        pooled[size]["p95_ms_per_window"] = latency_row.get("p95_ms_per_window", float("inf"))
    best_acc = max(row["macro_accuracy"] for row in pooled.values())
    best_ece = min(row["ece"] for row in pooled.values())
    order = [model.rsplit("-", 1)[-1] for model in ctx["config"]["models"]]
    best_size = max(order, key=lambda size: pooled[size]["macro_accuracy"])
    from .infer import document_bootstrap
    outcomes = {}
    for size in order:
        differences = defaultdict(list)
        for i, row in enumerate(rows):
            if row.hard_negative: continue
            def pooled_correct(which):
                return sum(int(max(range(len(trained[f"{which}-s{s}"]["probabilities"][i])),
                                   key=trained[f"{which}-s{s}"]["probabilities"][i].__getitem__) == row.target)
                           for s in (1, 2)) / 2
            differences[row.doc_id].append(pooled_correct(best_size) - pooled_correct(size))
        gap_ci = document_bootstrap(dict(differences), seed=23, samples=100 if ctx["dry"] else 2000,
                                    side="upper")
        outcomes[size] = {"accuracy_gap": best_acc - pooled[size]["macro_accuracy"],
                          "accuracy_gap_one_sided_upper_95": gap_ci["one_sided_upper_95"],
                          "ece_gap": pooled[size]["ece"] - best_ece,
                          "latency_ok": pooled[size]["p95_ms_per_window"] <= 300}
    eligible = [size for size in order if outcomes[size]["accuracy_gap_one_sided_upper_95"] <= 0.03
                and outcomes[size]["ece_gap"] <= 0.01 and outcomes[size]["latency_ok"]]
    chosen = eligible[0] if eligible else order[-1]
    trained_accuracy = pooled[chosen]["macro_accuracy"]
    prompted_accuracy = prompted[chosen]["macro_accuracy"]
    calibration = {"source": "gretel", "split": "dev_slice", "held_out_from_training": True,
                   "questions": len(calib_rows),
                   "gold_questions": sum(not row.hard_negative for row in calib_rows),
                   "labels": sorted({row.question.options[row.target].name
                                     for row in calib_rows if not row.hard_negative})}
    return {"implementation_version": 7, "split": "nemotron-calib", "questions": len(rows),
            "not_pii_questions": sum(row.hard_negative for row in rows),
            "trained": trained, "prompted": prompted, "pooled": pooled, "rule_outcomes": outcomes,
            "calibration": calibration, "in_domain_sanity": sanity,
            "chosen_size": chosen, "trained_accuracy": trained_accuracy,
            "prompted_accuracy": prompted_accuracy, "stop_rule": trained_accuracy < prompted_accuracy}


def _decision20_window_tokenizer(ctx) -> tuple[str, str]:
    """The pinned Qwen3 tokenizer whose 512-token windows dev_eval uses."""
    model_id = next(iter(ctx["config"]["models"]))
    return model_id, ("local-dry" if ctx["dry"] else ctx["config"]["models"][model_id]["revision"])


def _decision20_groups(ctx, rows):
    """(state window text, row indices) per dev_eval window, from the Qwen3 tokenizer windows."""
    if ctx["dry"]:
        from .latency import _DryTokenizer
        tokenizer = _DryTokenizer()
    else:
        from transformers import AutoTokenizer
        tok_id, tok_rev = _decision20_window_tokenizer(ctx)
        tokenizer = AutoTokenizer.from_pretrained(tok_id, revision=tok_rev)
    return [(state, indices) for state, _options, _branches, indices
            in _evaluation_window_specs(rows, tokenizer, layout="shared")]


def _score_decision20_sets(ctx, row_sets: dict[str, list], model_id: str, revision: str):
    """Probabilities per split for one Decision 2.0 model, resumable through the eval cache."""
    from .decision20 import load, score_groups
    tok_id, tok_rev = _decision20_window_tokenizer(ctx)
    identity = {"kind": "decision20", "model": model_id, "revision": revision, "state": "dev_eval-windows",
                "tokenizer": tok_id, "tokenizer_revision": tok_rev, "state_tokens": 512, "stride": 384,
                "branches_per_window": 64}
    key = "decision20-" + model_id.rsplit("/", 1)[-1]
    output, errors, missing = {}, {}, {}
    for split, rows in row_sets.items():
        fingerprint = _eval_fingerprint(rows, identity)
        cached = _read_eval_cache(_eval_cache_path(ctx, key, split), fingerprint, len(rows))
        if cached and cached.get("completed") == cached.get("total") and "errors" in cached:
            output[split], errors[split] = cached["probabilities"], cached["errors"]
        else:
            missing[split] = (rows, fingerprint)
    if not missing:
        return output, errors
    if ctx["dry"]:
        from types import SimpleNamespace

        def system_one(*, state, questions):
            return {"answers": {qid: {"type": "choice", "probabilities":
                                      {name: 1 / len(q["criteria"]) for name in q["criteria"]}}
                                for qid, q in questions.items()}}
        model = SimpleNamespace(system_one=system_one)
    else:
        model = load(model_id, revision)
    (ctx["root"] / "PHASE").write_text("CPU" if ctx["dry"] else "GPU")
    try:
        for split, (rows, fingerprint) in missing.items():
            path = _eval_cache_path(ctx, key, split)
            groups = _decision20_groups(ctx, rows)
            partial = _read_eval_cache(path, fingerprint, len(rows))
            start = min(int(partial.get("completed", 0)), len(groups)) if partial else 0

            def checkpoint(done, probabilities, split_errors, total=len(groups)):
                if done % 32 == 0 or done == total:
                    _atomic_json(path, {"version": _EVAL_CACHE_VERSION, "fingerprint": fingerprint,
                                        "completed": done, "total": total, "probabilities": probabilities,
                                        "errors": split_errors})
                    print(f"{key} {split}: {done}/{total} windows", flush=True)
                _check_eval_stop(ctx)
            probabilities, split_errors = score_groups(
                model, rows, groups, on_group=checkpoint, start=start,
                out=partial["probabilities"] if partial else None,
                errors=partial.get("errors") if partial else None)
            output[split], errors[split] = probabilities, split_errors
    finally:
        (ctx["root"] / "PHASE").write_text("CPU")
        del model
        if not ctx["dry"] and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return output, errors


def _unit_decision20(ctx, _out):
    """Zero-shot Decision 2.0 System One models on the exact dev_eval questions and windows."""
    from .decision20 import failed_distribution
    from .infer import fit_temperatures
    _, rows = _dev_questions(ctx, descriptions=True)
    seen_rows = _seen_questions(ctx, descriptions=True)
    results = {}
    for model_id, spec in ctx["config"]["external"]["decision20"].items():
        scores, errors = _score_decision20_sets(ctx, {"seen": seen_rows, "dev": rows}, model_id, spec["revision"])
        seen_kept = [(r, p) for r, p in zip(seen_rows, scores["seen"]) if p is not None]
        temperature = fit_temperatures([torch.tensor(p).clamp_min(1e-30).log() for _, p in seen_kept],
                                       [r.target for r, _ in seen_kept],
                                       ["choice"] * len(seen_kept))["choice"] if seen_kept else 1.0
        answered = sum(p is not None for p in scores["dev"])
        if answered == 0:
            raise RuntimeError(f"{model_id}: no valid answers on {len(rows)} dev questions: {errors['dev']}")
        full = [p if p is not None else failed_distribution(r) for r, p in zip(rows, scores["dev"])]
        metrics = _probability_metrics(rows, full, temperature=temperature)
        metrics.update({"revision": spec["revision"], "temperature": temperature,
                        "answered": answered, "questions": len(rows),
                        "dev_errors": errors["dev"], "seen_errors": errors["seen"]})
        results[model_id] = metrics
    return {"implementation_version": 2, "split": "nemotron-calib", "state": "dev_eval 512-token windows",
            "questions": len(rows), "not_pii_questions": sum(row.hard_negative for row in rows),
            "models": results}


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
        if stage in {"stage1_pilot", "stage1_pilot_4b"}:
            try:
                result = json.loads(result_path.read_text())
            except (OSError, ValueError):
                return False
            if result.get("implementation_version") != 2:
                return False
            confirm_4b = stage == "stage1_pilot_4b"
            conditions = {"stable"} if confirm_4b else {"stable", "stable_acc8", "stable_lr5"}
            pairs = {"4B-init1", "4B-init2"} if confirm_4b else {"1.7B-init1", "1.7B-init2"}
            expected_mode = "4b_final_confirmation" if confirm_4b else "1.7b_optimizer_pilot"
            if result.get("mode") not in ({expected_mode} if confirm_4b else {None, expected_mode}):
                return False
            if set(result.get("runs", {})) != pairs \
                    or set(result.get("condition_outcomes", {})) != conditions:
                return False
            expected_questions = min(int(result.get("windows", 0)), 8) if dry \
                else int(result.get("windows", 0))
            for pair in pairs:
                rows = result["runs"][pair].get("conditions", {})
                if set(rows) != conditions:
                    return False
                for condition in conditions:
                    row = rows[condition]
                    training = row.get("training", {})
                    checkpoints = row.get("checkpoints", [])
                    if (int(training.get("questions_seen", -1)) != expected_questions
                            or not checkpoints
                            or int(checkpoints[-1].get("questions_seen", -1)) != expected_questions
                            or "fixed_after_warmup" not in row):
                        return False
            return True
        if stage == "comparators1":
            try:
                return json.loads(result_path.read_text()).get("implementation_version") == 2
            except (OSError, ValueError):
                return False
        if stage == "stage1":
            # The data-template repair invalidates both training units and the resulting eval.
            required = {"train_sizes": 6, "layout_ablation": 6, "dev_eval": 7}.get(unit, 6)
            try:
                return json.loads(result_path.read_text()).get("implementation_version") == required
            except (OSError, ValueError):
                return False
        # Stage 2 still has no production implementation. Dry runs may resume its stubs.
        return dry
    if unit == "census":
        return True
    if unit == "latency":
        # Bumped when the benchmark became a genuinely measured end-to-end (exported proposer +
        # decision model) pass; older model-only or bare-encoder caches re-run.
        try:
            return json.loads(result_path.read_text()).get("implementation_version") == 4
        except (OSError, ValueError):
            return False
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
    stage_cfg = cfg["stages"][stage]
    if dry and root.name != "s1d_dry":
        root = root.parent / "s1d_dry"
    root.mkdir(parents=True, exist_ok=True)
    (root / "PHASE").write_text("CPU")
    if stage == "stage0":
        _append_stop_rule_once(root)
    if stage_cfg.get("requires_approval") and not (root / f"APPROVED_{stage}").exists():
        raise PermissionError(f"{root / ('APPROVED_' + stage)} is required")
    ctx = {"root": root, "dry": dry, "config": cfg, "stage": stage}
    meter = GPUHours(root / "gpu_hours.jsonl", stage_cfg["cap_a100_hours"], stage)
    meter.record("accounting_active", 0.0, stage=stage, device="n/a",
                 note="GPU-hour accounting enabled; execution is uncapped")
    handlers = {
        "label_draw": _unit_label_draw, "census": _unit_census, "revisions": _unit_revisions,
        "proposer_all": lambda c, o: _unit_proposer(c, o, "all-sources"),
        "proposer_no_nemotron": lambda c, o: _unit_proposer(c, o, "no-nemotron"),
        "prompted_probe": lambda c, o: _unit_questions(c, o, "prompted_probe"),
        "kev_baseline": lambda c, o: _unit_questions(c, o, "kev_baseline"),
        "latency": _unit_latency,
        "train_sizes": _unit_train_sizes,
        "paired_optimizer": _unit_stage1_pilot,
        "stable_optimizer": lambda c, o: _unit_stage1_pilot(c, o, confirm_4b=True),
        "layout_ablation": _unit_layout_ablation,
        "dev_eval": _unit_dev_eval,
        "decision20": _unit_decision20,
        "train_final": lambda c, o: _unimplemented(c, o, "train_final"),
        "test_inference": lambda c, o: _unimplemented(c, o, "test_inference"),
        "comparators": lambda c, o: _unimplemented(c, o, "comparators"),
    }
    stores = root / "stores"; stores.mkdir(exist_ok=True)
    try:
        for unit in UNITS[stage]:
            (root / "PHASE").write_text("CPU")
            (root / "CURRENT").write_text(f"{unit} starting {time.strftime('%FT%TZ', time.gmtime())}")
            if (root / "CONTROL").exists() and (root / "CONTROL").read_text().strip().upper() == "STOP":
                return 4
            if stage == "stage0" and unit == "proposer_all" and _stage0_should_stop(stores, dry):
                return 5
            done = stores / f"{stage}-{unit}.done"
            if _cache_is_current(root, stores, stage, unit, dry):
                (root / "CURRENT").write_text(f"{unit} cached {time.strftime('%FT%TZ', time.gmtime())}")
                continue
            # The v1 pilot marker predates the v2 result and must not survive while v2 is
            # running: after an interruption it could otherwise authenticate partial output.
            if stage in {"stage1_pilot", "stage1_pilot_4b"}:
                done.unlink(missing_ok=True)
            if not dry and unit not in ("label_draw", "census", "revisions"):
                meter.reserve(float(cfg["stages"][stage].get("unit_estimates", {}).get(unit, 0)))
            result = handlers[unit](ctx, stores / f"{stage}-{unit}.json")
            if not isinstance(result, dict) or result.get("status") in {"configured", "entrypoint-ready"}:
                raise RuntimeError(f"{unit} did not produce a complete result")
            _atomic_json(stores / f"{stage}-{unit}.json", result)
            done.write_text(time.strftime("%FT%TZ", time.gmtime()))
            (root / "CURRENT").write_text(f"{unit} done {time.strftime('%FT%TZ', time.gmtime())}")
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
