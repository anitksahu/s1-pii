"""Worst-case S1-D latency benchmark used by Stage 0 and its CPU dry run."""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from .model import S1DModel, apply_lora, prepare_tokenizer
from .packer import BranchText, SPECIAL_TOKENS, pack_window


class _DryTokenizer:
    """Small word tokenizer: enough API for PEFT and packing, with no downloads."""

    def __init__(self):
        self.vocab = {token: i + 1 for i, token in enumerate(SPECIAL_TOKENS)}
        self.pad_token_id = 0
        self.unk_token_id = 127

    def __len__(self):
        return 128

    def add_special_tokens(self, value):
        before = len(self.vocab)
        for token in value["additional_special_tokens"]:
            self.vocab.setdefault(token, len(self.vocab) + 1)
        return len(self.vocab) - before

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, self.unk_token_id)

    def __call__(self, text, **_kwargs):
        words = str(text).split()
        return {"input_ids": [8 + (sum(map(ord, word)) % 112) for word in words]}


def _p95(values: list[float]) -> float:
    if not values:
        raise ValueError("latency benchmark produced no measurements")
    return sorted(values)[max(0, math.ceil(0.95 * len(values)) - 1)]


def _workload(tokenizer, *, device, make_block_mask: bool):
    # 510 body tokens plus delimiters gives the specified 512-token state block.
    state = " ".join(f"state{i}" for i in range(510))
    options = [f"option {i}: described personal-information category number {i}" for i in range(56)]
    branches = [BranchText(f"span{i}", "one two three four five six seven eight") for i in range(64)]
    return pack_window(tokenizer, state, options, branches, state_tokens=512,
                       make_block_mask=make_block_mask, device=device)


def _tiny_model():
    from transformers import Qwen3Config
    cfg = Qwen3Config(vocab_size=128, hidden_size=16, intermediate_size=32,
                      num_hidden_layers=1, num_attention_heads=2,
                      num_key_value_heads=2, head_dim=8,
                      max_position_embeddings=4096, use_cache=False,
                      attention_dropout=0)
    return S1DModel.from_config(cfg)


def _one_window_text(tokenizer, size: int) -> str:
    """Largest entity-dense word prefix that still tokenizes within a single window.

    Built against the real proposer tokenizer (not a whitespace assumption): the returned text
    has ``<= size`` content tokens and is near-capacity, so the predictor produces exactly one
    near-full window. The caller additionally asserts the measured window count."""
    base = ("Contact John Smith at john.smith@example.com or call +1 555 010 2034 "
            "on 2021-04-17 regarding account 4929-8381-2847 in Boston Massachusetts. ")
    words = (base * 60).split()

    def n_tokens(k: int) -> int:
        return len(tokenizer(" ".join(words[:k]), add_special_tokens=False)["input_ids"])

    if n_tokens(len(words)) <= size:
        return " ".join(words)
    lo, hi, best = 1, len(words), 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if n_tokens(mid) <= size:
            best = mid; lo = mid + 1
        else:
            hi = mid - 1
    return " ".join(words[:best])


@torch.no_grad()
def _proposer_window_latency_ms(root, config: dict, *, dry: bool, warmup: int, repeats: int,
                                control_path: Path | None = None, phase_callback=None) -> dict:
    """Genuinely measured per-window proposer latency.

    Production path loads the exported trained proposer at
    ``<root>/models/proposer-no-nemotron/final`` and times the full
    ``S1Predictor.predict_docs`` pipeline on a worst-case ~512-token document: encoder
    emissions, the trained head, the float64 CRF forward/backward, ``span_logprobs`` candidate
    extraction, and the ``<=64`` candidates-per-window selection. The plan's end-to-end budget
    includes this, so its p95 is added to every candidate decision model. The dry harness does
    an explicit download-free CPU measurement with the same schema."""
    if dry:
        if phase_callback:
            phase_callback("CPU")
        x = torch.ones(16, 16)
        samples = []
        for _ in range(max(1, repeats)):
            start = time.perf_counter()
            for _ in range(64):
                x = torch.tanh(x @ x)
            samples.append((time.perf_counter() - start) * 1000)
        return {"mode": "dry", "p95_ms_per_window": _p95(samples),
                "p95_ms_per_document": _p95(samples), "candidates_per_window": 0,
                "windows": 1, "window_tokens": 0, "window_capacity_tokens": 0, "repeats": repeats}
    from ..model.train import load_exported
    from ..model.predict import S1Predictor
    from ..schema import Doc
    path = Path(root) / "models" / "proposer-no-nemotron" / "final"
    weights_sha = json.loads((path / "s1_manifest.json").read_text())["weights_sha256"]
    if phase_callback:
        phase_callback("CPU")
    model, tokenizer, _m = load_exported(path, device="cuda")
    predictor = S1Predictor(model, tokenizer, revision=weights_sha, max_len=512,
                            floor=0.01, validators=False, propagation=False, device="cuda")
    # A near-capacity one-window document, sized against the real tokenizer and predictor.size.
    # Capture the capacity before the finally deletes the predictor, so it survives the return.
    capacity = predictor.size
    text = _one_window_text(tokenizer, capacity)
    window_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    doc = Doc("latency-proposer", text, (), "synthetic", "test", "latency-proposer")
    for _ in range(warmup):
        predictor.predict_docs([doc])
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if phase_callback:
        phase_callback("GPU")
    samples, candidate_count, windows = [], 0, 0
    try:
        for _ in range(repeats):
            if control_path and control_path.exists() and control_path.read_text().strip().upper() == "STOP":
                raise InterruptedError("STOP requested")
            start = time.perf_counter()
            preds, rep = predictor.predict_docs([doc])
            windows = int(getattr(rep, "windows", 0))
            # Per-window == per-document is valid only for a single window; reject anything else so
            # a multi-window duration is never mislabelled as per-window.
            if windows != 1:
                raise RuntimeError(
                    f"proposer latency workload spanned {windows} windows ({window_tokens} tokens, "
                    f"capacity {capacity}); per-window timing requires exactly one window")
            # With one window, top-64 on that window's candidates is the applicable selection.
            candidate_count = len(sorted(preds.get(doc.doc_id, ()),
                                         key=lambda s: s.score, reverse=True)[:64])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
    finally:
        del predictor, model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {"mode": "measured-exported", "p95_ms_per_window": _p95(samples),
            "p95_ms_per_document": _p95(samples), "candidates_per_window": candidate_count,
            "windows": windows, "window_tokens": window_tokens,
            "window_capacity_tokens": capacity, "repeats": repeats, "weights_sha256": weights_sha}


@torch.no_grad()
def benchmark(config: dict, *, dry: bool = False, root=None, control_path: Path | None = None,
              warmup: int | None = None, repeats: int | None = None,
              phase_callback=None) -> dict:
    """Measure merged random-LoRA models with a single prebuilt worst-case mask."""
    if not dry and not torch.cuda.is_available():
        raise NotImplementedError("the production latency benchmark requires CUDA")
    from transformers import AutoTokenizer

    specs = {"tiny-random": {"revision": "local"}} if dry else config["models"]
    warmup = 1 if warmup is None and dry else (3 if warmup is None else warmup)
    repeats = 1 if repeats is None and dry else (20 if repeats is None else repeats)
    device = torch.device("cpu" if dry else "cuda")
    proposer = _proposer_window_latency_ms(root, config, dry=dry, warmup=warmup, repeats=repeats,
                                           control_path=control_path, phase_callback=phase_callback)
    proposer_p95 = proposer["p95_ms_per_window"]
    results = {}
    for model_id, spec in specs.items():
        if phase_callback:
            phase_callback("CPU")
        if control_path and control_path.exists() and control_path.read_text().strip().upper() == "STOP":
            raise InterruptedError("STOP requested")
        if dry:
            tokenizer, model = _DryTokenizer(), _tiny_model()
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=spec["revision"])
            model = S1DModel.from_pretrained(model_id, spec["revision"])
        token_rows = prepare_tokenizer(tokenizer, model)
        apply_lora(model, token_rows, rank=32)
        model.backbone = model.backbone.merge_and_unload()
        if phase_callback:
            phase_callback("GPU")
        model = model.to(device).eval()
        if not dry:
            model = model.bfloat16()
        packed = _workload(tokenizer, device=device, make_block_mask=not dry)
        for _ in range(warmup):
            model(packed)
        if device.type == "cuda":
            torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            if control_path and control_path.exists() and control_path.read_text().strip().upper() == "STOP":
                raise InterruptedError("STOP requested")
            start = time.perf_counter()
            output = model(packed)
            if device.type == "cuda":
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        if output.probabilities.shape != (64, 56):
            raise RuntimeError("latency workload did not produce 64-by-56 probabilities")
        model_p95 = _p95(samples)
        end_to_end = model_p95 + proposer_p95
        results[model_id] = {
            "revision": spec["revision"],
            "model_p95_ms_per_window": model_p95,
            "proposer_p95_ms_per_window": proposer_p95,
            "p95_ms_per_window": end_to_end,
            "p95_ms_per_document": end_to_end, "samples_ms": samples,
            "true_tokens": packed.true_length, "bucket_tokens": len(packed.input_ids),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"implementation_version": 4, "batch_size": 1, "branches": 64, "options": 56,
            "state_tokens": 512, "warmup_excluded": True, "proposer_included": True,
            "proposer_p95_ms_per_window": proposer_p95, "proposer": proposer,
            "prebuilt_block_masks": not dry, "bf16": not dry,
            "lora": "random-r32-merged", "dry": dry, "models": results}
