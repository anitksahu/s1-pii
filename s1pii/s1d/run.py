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
from . import labels
from .data import assert_no_heldout_leakage, generate_questions
from .train import GPUHours


UNITS = {
    "stage0": ("label_draw", "census", "revisions", "prompted_probe", "kev_baseline",
               "proposer_all", "proposer_no_nemotron", "latency"),
    "stage1": ("train_sizes", "layout_ablation", "dev_eval"),
    "stage2": ("train_final", "test_inference", "comparators"),
    "comparators1": ("decision20",),
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
                         cache_name: str | None = None) -> dict[str, list[Span]]:
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


def _train_stage1_run(ctx, questions, *, model_id: str, seed: int, layout: str,
                      window_count: int, run_dir: Path, branches_per_window: int = 1) -> dict:
    from .data import pack_layout_ablation_questions, pack_training_questions, question_set_hash
    from .model import S1DModel, apply_lora, prepare_tokenizer
    from .train import TrainConfig, seed_everything, train
    dry = ctx["dry"]
    revision = "local-dry" if dry else ctx["config"]["models"][model_id]["revision"]
    selected_count = min(window_count, 2 if branches_per_window > 1 else 8) if dry else window_count
    selected = _repeat_rows(questions, selected_count if branches_per_window == 1 else len(questions), seed)
    # The manifest hashes decide reuse/resume: a completed run is reused only when its questions,
    # revision and layout all match; mismatches (including a stale checkpoint) are archived aside.
    expected_hashes = {"questions": question_set_hash(selected), "revision": revision, "layout": layout}
    if _reuse_or_reset_run(run_dir, expected_hashes):
        return json.loads((run_dir / "manifest.json").read_text()) | {"status": "cached"}
    seed_everything(seed)
    if dry:
        from .latency import _DryTokenizer, _tiny_model
        tokenizer, model = _DryTokenizer(), _tiny_model()
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = S1DModel.from_pretrained(model_id, revision)
    token_rows = prepare_tokenizer(tokenizer, model); apply_lora(model, token_rows, rank=32)
    device = torch.device("cpu" if dry else "cuda")
    if branches_per_window > 1 and not dry:
        packed = _LazyPackedRows(selected_count, lambda index: pack_layout_ablation_questions(
            selected, tokenizer, windows=1, seed=seed + index, layout=layout,
            branches_per_window=branches_per_window, device=device, make_block_mask=True,
            keep_dense_mask=False, window_offset=index)[0])
    elif branches_per_window > 1:
        packed = pack_layout_ablation_questions(selected, tokenizer, windows=selected_count, seed=seed,
                                                layout=layout, branches_per_window=branches_per_window,
                                                device=device, make_block_mask=False)
    elif not dry:
        packed = _LazyPackedRows(selected_count, lambda index: pack_training_questions(
            [selected[index]], tokenizer, layout=layout, device=device, make_block_mask=True,
            keep_dense_mask=False)[0])
    else:
        packed = pack_training_questions(selected, tokenizer, layout=layout, device=device,
                                         make_block_mask=not dry)
    config = TrainConfig(seed=seed, token_budget=4096 if dry else 8192, epochs=1,
                         stage="stage1", unit=f"{model_id}-s{seed}-{layout}",
                         cap_hours=ctx["config"]["stages"]["stage1"]["cap_a100_hours"],
                         control_path=str(ctx["root"] / "CONTROL"), checkpoint_every=100)
    (ctx["root"] / "PHASE").write_text("GPU")
    try:
        result = train(model, packed, config, run_dir, ctx["root"] / "gpu_hours.jsonl",
                       hashes=expected_hashes)
    finally:
        (ctx["root"] / "PHASE").write_text("CPU")
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return result


def _training_questions(ctx, seed: int):
    from .data import generate_questions
    if ctx["dry"]:
        docs = _dry_docs(); candidates = _proposer_candidates(ctx["root"], docs, True)
    else:
        from ..model.train import TrainConfig as V1Config, training_docs
        docs, _ = training_docs(V1Config(variant="no-nemotron", seed=seed))
        counts = dict(__import__("collections").Counter(doc.dataset for doc in docs))
        print(f"stage1 training documents: {len(docs)} source_counts={counts}", flush=True)
        candidates = _proposer_candidates(ctx["root"], docs, False, cache_name="training")
    return generate_questions(docs, seed=seed, variant="no-nemotron", min_options=2, max_options=64,
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
    return {"implementation_version": 5, "variant": "no-nemotron", "windows_per_run": 16000,
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


def _seen_questions(ctx, *, descriptions: bool = True):
    from .data import span_evaluation_questions
    if ctx["dry"]:
        docs = _dry_docs()
    else:
        docs = _stage1_docs(False)
    held = labels.load()
    seen = [name for name in labels.training_vocabulary(held)
            if name not in set(held["dev_labels"] + held["test_labels"])]
    return span_evaluation_questions(docs, {}, labels=seen, seed=1, descriptions=descriptions,
                                     require_equal_negatives=False)


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


def _inference_probabilities(model, window, device: torch.device, *, bf16: bool):
    """Run pointer inference with the same mixed-precision contract used for training."""
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
        return model(window).probabilities[0].float().cpu().tolist()


def _score_trained_run(ctx, rows, *, model_id: str, run_dir: Path, layout: str = "shared"):
    from .data import pack_training_questions
    from .model import S1DModel, apply_lora, prepare_tokenizer
    if ctx["dry"]:
        from .latency import _DryTokenizer, _tiny_model
        tokenizer, model = _DryTokenizer(), _tiny_model()
    else:
        from transformers import AutoTokenizer
        revision = ctx["config"]["models"][model_id]["revision"]
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        model = S1DModel.from_pretrained(model_id, revision)
    indices = prepare_tokenizer(tokenizer, model); apply_lora(model, indices, rank=32)
    checkpoint = run_dir / "checkpoint.pt"
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["trainable_model"],
                              strict=False)
    device = torch.device("cpu" if ctx["dry"] else "cuda"); model.to(device).eval()
    probs = []
    with torch.no_grad():
        for row in rows:
            window, _ = pack_training_questions([row], tokenizer, layout=layout, device=device,
                                                make_block_mask=not ctx["dry"],
                                                keep_dense_mask=ctx["dry"])[0]
            probs.append(_inference_probabilities(model, window, device, bf16=not ctx["dry"]))
    del model
    return probs


def _score_prompted_base(ctx, rows, model_id: str):
    if ctx["dry"]:
        return [[1 / len(row.question.options)] * len(row.question.options) for row in rows]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .infer import prompted_option_distribution
    revision = ctx["config"]["models"][model_id]["revision"]
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16).cuda().eval()
    result = []
    for row in rows:
        options = [option.name for option in row.question.options]
        prompt = ("Classify the marked span. /no_think\nDocument:\n" + row.state +
                  f"\nSpan: {row.state[row.source_span[0]:row.source_span[1]]}\nAnswer:")
        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=False)
        result.append(prompted_option_distribution(model, tokenizer, prompt, options, device="cuda").tolist())
    del model; torch.cuda.empty_cache()
    return result


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
    return {"implementation_version": 5, "windows_per_run": 4000, "sampled_spans_per_window": 16,
            "metrics": metrics, "difference": interval,
            "stage2_layout": "kev" if interval["one_sided_lower_95"] > 0.02 else "shared"}


def _unit_dev_eval(ctx, _out):
    _, rows = _dev_questions(ctx, descriptions=True)
    seen_rows = _seen_questions(ctx, descriptions=True)
    models_root = ctx["root"] / "models" / "stage1"
    trained, prompted = {}, {}
    from .infer import fit_temperatures
    for model_id in ctx["config"]["models"]:
        size = model_id.rsplit("-", 1)[-1]
        prompted_seen = _score_prompted_base(ctx, seen_rows, model_id) if seen_rows else []
        prompted_temp = fit_temperatures([torch.tensor(p).clamp_min(1e-30).log() for p in prompted_seen],
                                         [row.target for row in seen_rows], ["choice"] * len(seen_rows))["choice"]
        prompted[size] = _probability_metrics(rows, _score_prompted_base(ctx, rows, model_id),
                                               temperature=prompted_temp)
        prompted[size]["temperature"] = prompted_temp
        for seed in (1, 2):
            key = f"{size}-s{seed}"
            run_dir = models_root / key
            seen_probabilities = _score_trained_run(ctx, seen_rows, model_id=model_id,
                                                    run_dir=run_dir) if seen_rows else []
            temperature = fit_temperatures([torch.tensor(p).clamp_min(1e-30).log() for p in seen_probabilities],
                                           [row.target for row in seen_rows],
                                           ["choice"] * len(seen_rows))["choice"]
            trained[key] = _probability_metrics(rows, _score_trained_run(
                ctx, rows, model_id=model_id, run_dir=run_dir), temperature=temperature)
            trained[key]["temperature"] = temperature
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
    return {"implementation_version": 5, "split": "nemotron-calib", "questions": len(rows),
            "not_pii_questions": sum(row.hard_negative for row in rows),
            "trained": trained, "prompted": prompted, "pooled": pooled, "rule_outcomes": outcomes,
            "chosen_size": chosen, "trained_accuracy": trained_accuracy,
            "prompted_accuracy": prompted_accuracy, "stop_rule": trained_accuracy < prompted_accuracy}


def _unit_decision20(ctx, _out):
    """Zero-shot Decision 2.0 System One models on the exact dev_eval questions."""
    from .decision20 import failed_distribution, load, score_rows
    from .infer import fit_temperatures
    _, rows = _dev_questions(ctx, descriptions=True)
    seen_rows = _seen_questions(ctx, descriptions=True)
    results = {}
    for model_id, spec in ctx["config"]["external"]["decision20"].items():
        if ctx["dry"]:
            from types import SimpleNamespace

            def system_one(*, state, questions):
                return {"answers": {qid: {"type": "choice", "probabilities":
                                          {name: 1 / len(q["criteria"]) for name in q["criteria"]}}
                                    for qid, q in questions.items()}}
            model = SimpleNamespace(system_one=system_one)
        else:
            model = load(model_id, spec["revision"])
        seen_probs, seen_errors = score_rows(model, seen_rows)
        dev_probs, dev_errors = score_rows(model, rows)
        del model
        if not ctx["dry"] and torch.cuda.is_available():
            torch.cuda.empty_cache()
        seen_kept = [(r, p) for r, p in zip(seen_rows, seen_probs) if p is not None]
        temperature = fit_temperatures([torch.tensor(p).clamp_min(1e-30).log() for _, p in seen_kept],
                                       [r.target for r, _ in seen_kept],
                                       ["choice"] * len(seen_kept))["choice"] if seen_kept else 1.0
        answered = sum(p is not None for p in dev_probs)
        if answered == 0:
            raise RuntimeError(f"{model_id}: no valid answers on {len(rows)} dev questions: {dev_errors}")
        full = [p if p is not None else failed_distribution(r) for r, p in zip(rows, dev_probs)]
        metrics = _probability_metrics(rows, full, temperature=temperature)
        metrics.update({"revision": spec["revision"], "temperature": temperature,
                        "answered": answered, "questions": len(rows),
                        "dev_errors": dev_errors, "seen_errors": seen_errors})
        results[model_id] = metrics
    return {"implementation_version": 1, "split": "nemotron-calib", "questions": len(rows),
            "not_pii_questions": sum(row.hard_negative for row in rows), "models": results}


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
        if stage == "comparators1":
            try:
                return json.loads(result_path.read_text()).get("implementation_version") == 1
            except (OSError, ValueError):
                return False
        if stage == "stage1":
            try:
                return json.loads(result_path.read_text()).get("implementation_version") == 5
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
    ctx = {"root": root, "dry": dry, "config": cfg}
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
