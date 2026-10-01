"""Worst-case S1-D latency benchmark used by Stage 0 and its CPU dry run."""
from __future__ import annotations

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


@torch.no_grad()
def benchmark(config: dict, *, dry: bool = False, control_path: Path | None = None,
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
        p95 = _p95(samples)
        results[model_id] = {
            "revision": spec["revision"], "p95_ms_per_window": p95,
            "p95_ms_per_document": p95, "samples_ms": samples,
            "true_tokens": packed.true_length, "bucket_tokens": len(packed.input_ids),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return {"implementation_version": 2, "batch_size": 1, "branches": 64, "options": 56,
            "state_tokens": 512, "warmup_excluded": True,
            "prebuilt_block_masks": not dry, "bf16": not dry,
            "lora": "random-r32-merged", "dry": dry, "models": results}
