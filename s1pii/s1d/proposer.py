"""Class-agnostic (nt=1) span proposer built from the v1 CRF path."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import torch

from ..schema import Doc, IGNORE, PERSON
from ..model.encode import train_examples
from ..model.s1 import S1Model, collate
from .labels import excluded, load, semantic_exclusions


def proposer_doc(doc: Doc, config: dict | None = None,
                 exclusions: dict[str, set[str]] | None = None) -> Doc:
    """Map every observed PII span to one type; held-out spans become partial labels."""
    cfg = config or load()
    spans = []
    for span in doc.spans:
        if span.label_raw and excluded(span.label_raw, cfg, exclusions=exclusions):
            spans.append(replace(span, label_canonical=IGNORE))
        elif span.is_pii:
            # PERSON is canonical index zero, hence BIOES ids 1..4. Its name never
            # reaches the proposer: this is simply the single anonymous CRF type.
            spans.append(replace(span, label_canonical=PERSON))
        else:
            spans.append(span)
    return replace(doc, spans=tuple(spans))


def examples(docs, tokenizer, *, config: dict | None = None,
             exclusions: dict[str, set[str]] | None = None, **kwargs):
    cfg = config or load()
    exclusions = semantic_exclusions(cfg) if exclusions is None else exclusions
    return train_examples([proposer_doc(d, cfg, exclusions) for d in docs],
                          tokenizer, num_types=1, **kwargs)


def _cached_examples(docs, tokenizer, *, cache: Path, config: dict, clock,
                     control_path: Path, chunk_size: int = 512):
    """Build resumable chunks so progress, STOP, and the cap are checked frequently."""
    if cache.exists():
        rows = torch.load(cache, map_location="cpu", weights_only=False)
        print(f"proposer examples: loaded {len(rows)} rows from legacy cache {cache.name}", flush=True)
        clock.tick(substep="load-example-cache", examples=len(rows))
        return rows
    chunk_dir = cache.with_suffix("")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    exclusions = semantic_exclusions(config)
    rows = []
    total = len(docs)
    for first in range(0, total, chunk_size):
        if control_path.exists() and control_path.read_text().strip().upper() == "STOP":
            raise InterruptedError("STOP requested")
        last = min(first + chunk_size, total)
        path = chunk_dir / f"{first:08d}-{last:08d}.pt"
        if path.exists():
            part = torch.load(path, map_location="cpu", weights_only=False)
        else:
            part = examples(docs[first:last], tokenizer, max_len=512,
                            config=config, exclusions=exclusions)
            tmp = path.with_suffix(".part")
            torch.save(part, tmp); tmp.replace(path)
        rows.extend(part)
        print(f"proposer examples: {last}/{total} documents, {len(rows)} rows", flush=True)
        clock.tick(substep="data-and-tokenization", documents=last, examples=len(rows))
    return rows


def proposer_model(encoder, hidden: int, dropout: float = 0.1) -> S1Model:
    return S1Model(encoder, hidden, dropout=dropout, num_types=1)


def proposer_collate(rows, pad_id: int, with_targets: bool = True):
    return collate(rows, pad_id, with_targets=with_targets, k=5)


def gate_g1(gold_by_window: dict[str, list[tuple[int, int]]],
            candidates_by_window: dict[str, list[tuple[int, int, float]]], limit: int = 64) -> dict:
    """Non-blocking exact-boundary recall gate at a fixed candidate cap."""
    hit = total = 0
    for window, gold in gold_by_window.items():
        proposed = {(a, b) for a, b, _ in sorted(candidates_by_window.get(window, ()),
                                                 key=lambda row: row[2], reverse=True)[:limit]}
        total += len(gold); hit += sum((a, b) in proposed for a, b in gold)
    return {"exact_boundary_recall": hit / total if total else 0.0, "hits": hit,
            "gold": total, "candidate_cap": limit, "blocking": False}


def train_and_gate(variant: str, root: Path, *, cap_hours: float, control_path: Path,
                   seed: int = 1, dry_limit: int | None = None,
                   config: dict | None = None) -> dict:
    """Train/resume one real nt=1 proposer and measure its non-blocking calibration gate."""
    if not torch.cuda.is_available():
        raise NotImplementedError("real proposer training requires the Stage 0 CUDA runtime")
    from transformers import AutoModel, AutoTokenizer
    from .. import bench
    from ..ledger import stable_hash
    from ..model.encode import tokenize_doc, token_windows
    from ..model.predict import S1Predictor
    from ..model.train import (TrainConfig as V1Config, build_optimizer,
                               export as export_v1, training_docs)
    from ..schema import read_jsonl
    from .stage0 import UnitClock, set_phase
    from .train import token_batches

    if config is None:
        import yaml
        config = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "s1d.yaml").read_text())
    spec = config["external"]["proposer"]
    model_id, revision = spec["model_id"], spec["revision"]
    clock = UnitClock(root, config, f"proposer-{variant}")
    set_phase(root, "CPU")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    cfg = V1Config(backbone=model_id, backbone_revision=revision, variant=variant, seed=seed,
                   max_len=512, epochs=1, bf16=True, gradient_checkpointing=True)
    print(f"proposer {variant}: loading training documents", flush=True)
    docs, source_counts = training_docs(cfg)
    if dry_limit:
        docs = docs[:dry_limit]
    data_sha = stable_hash([{"id": doc.doc_id, "text": doc.text,
                             "spans": [(s.start, s.end, s.label_raw) for s in doc.spans]} for doc in docs])
    cache = Path(root) / "stores" / f"proposer-examples-{variant}-{data_sha[:16]}.pt"
    cache.parent.mkdir(parents=True, exist_ok=True)
    rows = _cached_examples(docs, tokenizer, cache=cache, config=load(), clock=clock,
                            control_path=control_path)
    batches = list(token_batches(rows, cfg.token_budget, length=lambda x: len(x.input_ids), seed=seed))
    if not batches:
        raise ValueError("proposer generated no training batches")
    clock.tick(substep="batching", examples=len(rows), batches=len(batches))
    encoder = AutoModel.from_pretrained(model_id, revision=revision, dtype=torch.float32)
    if any(parameter.dtype != torch.float32 for parameter in encoder.parameters()):
        raise RuntimeError("proposer master weights must remain FP32")
    encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    set_phase(root, "GPU")
    model = proposer_model(encoder, encoder.config.hidden_size).cuda().train()
    optimizer, scheduler = build_optimizer(model, cfg, len(batches))
    clock.tick(substep="model-load")
    unit_dir = Path(root) / "models" / f"proposer-{variant}"
    unit_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = unit_dir / "resume.pt"
    start = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        # Round-one checkpoints used bf16 masters and had no scheduler state;
        # they cannot be resumed as the corrected optimizer trajectory.
        if saved.get("format_version") == 2:
            model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"])
            scheduler.load_state_dict(saved["scheduler"]); start = saved["step"]
    step = 0
    for batch_rows in batches:
        step += 1
        if step <= start:
            continue
        if control_path.exists() and control_path.read_text().strip().upper() == "STOP":
            raise InterruptedError("STOP requested")
        batch = proposer_collate(batch_rows, tokenizer.pad_token_id or 0)
        batch = {key: value.cuda() for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(batch)
        loss.backward(); optimizer.step(); scheduler.step()
        clock.tick(substep="train", step=step)
        if step % 100 == 0:
            torch.save({"format_version": 2, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "step": step}, checkpoint)
    torch.save({"format_version": 2, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "step": step}, checkpoint)
    final = export_v1(model, tokenizer, cfg, unit_dir / "final",
                      {"source_counts": source_counts, "examples": len(rows)}, step)
    clock.tick(substep="checkpoint-and-export")
    set_phase(root, "CPU")
    calibration = read_jsonl(bench.split_paths("nemotron")["calib"])
    held = load()
    wanted = set(held["dev_labels"] + held["test_labels"])
    evaluation_docs = [doc for doc in calibration
                       if any(s.label_raw.lower() in wanted for s in doc.spans)]
    predictor = S1Predictor(model.eval(), tokenizer, revision=revision, max_len=512, floor=0.0,
                            validators=False, propagation=False, max_span_tokens=64)
    predicted = {}
    set_phase(root, "GPU")
    for first in range(0, len(evaluation_docs), 16):
        if control_path.exists() and control_path.read_text().strip().upper() == "STOP":
            raise InterruptedError("STOP requested")
        chunk, _ = predictor.predict_docs(evaluation_docs[first:first + 16])
        predicted.update(chunk)
        clock.tick(substep="gate-inference", documents=len(chunk))
    set_phase(root, "CPU")
    gold_by_window, candidates_by_window = {}, {}
    for doc in evaluation_docs:
        td = tokenize_doc(doc, tokenizer)
        for a, b in token_windows(len(td.ids), predictor.size, predictor.stride):
            if a >= b:
                continue
            char_a, char_b = int(td.offsets[a][0]), int(td.offsets[b - 1][1])
            key = f"{doc.doc_id}:{a}:{b}"
            gold_spans = [(s.start, s.end) for s in doc.spans
                          if s.label_raw.lower() in wanted and s.start >= char_a and s.end <= char_b]
            if not gold_spans:
                continue
            gold_by_window[key] = gold_spans
            candidates_by_window[key] = [(s.start, s.end, s.score) for s in predicted.get(doc.doc_id, ())
                                         if s.start >= char_a and s.end <= char_b]
    gate = gate_g1(gold_by_window, candidates_by_window, 64)
    result = {"implementation_version": 2, "variant": variant, "model": str(final),
              "backbone_revision": revision,
              "source_counts": source_counts, "steps": step, "gate_g1": gate,
              "data_sha": data_sha, "master_dtype": "float32", "autocast": "bfloat16",
              "optimizer": "v1-separated", "scheduler": "v1-linear-warmup-decay"}
    (unit_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    clock.tick(substep="decode-and-result")
    return result
