"""Class-agnostic (nt=1) span proposer built from the v1 CRF path."""
from __future__ import annotations

from dataclasses import replace
import json
import time
from pathlib import Path

import torch

from ..schema import Doc, IGNORE, PERSON
from ..model.encode import train_examples
from ..model.s1 import S1Model, collate
from .labels import excluded, load


def proposer_doc(doc: Doc, config: dict | None = None) -> Doc:
    """Map every observed PII span to one type; held-out spans become partial labels."""
    cfg = config or load()
    spans = []
    for span in doc.spans:
        if span.label_raw and excluded(span.label_raw, cfg):
            spans.append(replace(span, label_canonical=IGNORE))
        elif span.is_pii:
            # PERSON is canonical index zero, hence BIOES ids 1..4. Its name never
            # reaches the proposer: this is simply the single anonymous CRF type.
            spans.append(replace(span, label_canonical=PERSON))
        else:
            spans.append(span)
    return replace(doc, spans=tuple(spans))


def examples(docs, tokenizer, **kwargs):
    return train_examples([proposer_doc(d) for d in docs], tokenizer, num_types=1, **kwargs)


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
                   seed: int = 1, dry_limit: int | None = None) -> dict:
    """Train/resume one real nt=1 proposer and measure its non-blocking calibration gate."""
    if not torch.cuda.is_available():
        raise NotImplementedError("real proposer training requires the Stage 0 CUDA runtime")
    from transformers import AutoModel, AutoTokenizer
    from .. import bench
    from ..ledger import stable_hash
    from ..model.encode import tokenize_doc, token_windows
    from ..model.predict import S1Predictor
    from ..model.train import TrainConfig as V1Config, export as export_v1, training_docs
    from ..schema import read_jsonl
    from .train import GPUHours, token_batches

    model_id = "answerdotai/ModernBERT-large"
    from huggingface_hub import HfApi
    revision = HfApi().model_info(model_id).sha
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    cfg = V1Config(backbone=model_id, backbone_revision=revision, variant=variant, seed=seed,
                   max_len=512, epochs=1, bf16=True, gradient_checkpointing=True)
    docs, source_counts = training_docs(cfg)
    if dry_limit:
        docs = docs[:dry_limit]
    rows = examples(docs, tokenizer, max_len=512)
    encoder = AutoModel.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16)
    encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = proposer_model(encoder, encoder.config.hidden_size).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_encoder)
    unit_dir = Path(root) / "models" / f"proposer-{variant}"
    unit_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = unit_dir / "resume.pt"
    start = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model"]); optimizer.load_state_dict(saved["optimizer"]); start = saved["step"]
    meter = GPUHours(Path(root) / "gpu_hours.jsonl", cap_hours, "stage0")
    accounted = time.monotonic(); step = 0
    for batch_rows in token_batches(rows, cfg.token_budget, length=lambda x: len(x.input_ids), seed=seed):
        step += 1
        if step <= start:
            continue
        if control_path.exists() and control_path.read_text().strip().upper() == "STOP":
            raise InterruptedError("STOP requested")
        meter.reserve(0.0)
        batch = proposer_collate(batch_rows, tokenizer.pad_token_id or 0)
        batch = {key: value.cuda() for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(batch)
        loss.backward(); optimizer.step()
        now = time.monotonic()
        meter.record(f"proposer-{variant}", now - accounted, stage="stage0", step=step)
        accounted = now
        if step % 100 == 0:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, checkpoint)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step}, checkpoint)
    final = export_v1(model, tokenizer, cfg, unit_dir / "final",
                      {"source_counts": source_counts, "examples": len(rows)}, step)
    calibration = read_jsonl(bench.split_paths("nemotron")["calib"])
    held = load()
    wanted = set(held["dev_labels"] + held["test_labels"])
    evaluation_docs = [doc for doc in calibration
                       if any(s.label_raw.lower() in wanted for s in doc.spans)]
    predictor = S1Predictor(model.eval(), tokenizer, revision=revision, max_len=512, floor=0.0,
                            validators=False, propagation=False, max_span_tokens=64)
    predicted = {}
    accounted = time.monotonic()
    for first in range(0, len(evaluation_docs), 16):
        if control_path.exists() and control_path.read_text().strip().upper() == "STOP":
            raise InterruptedError("STOP requested")
        meter.reserve(0.0)
        chunk, _ = predictor.predict_docs(evaluation_docs[first:first + 16])
        predicted.update(chunk)
        now = time.monotonic()
        meter.record(f"proposer-{variant}-gate", now - accounted, stage="stage0",
                     documents=len(chunk)); accounted = now
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
    result = {"variant": variant, "model": str(final), "backbone_revision": revision,
              "source_counts": source_counts, "steps": step, "gate_g1": gate,
              "data_sha": stable_hash([doc.doc_id for doc in docs])}
    (unit_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    return result
