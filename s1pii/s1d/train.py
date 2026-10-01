"""Resumable S1-D training primitives with token and GPU-hour budgets."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch

from ..ledger import git_sha


class GPUCapReached(RuntimeError):
    pass


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def token_batches(rows: Sequence, token_budget: int, length=lambda x: len(x.input_ids), *,
                  seed: int = 0, shuffle: bool = True) -> Iterator[list]:
    """Deterministic batches whose summed packed-token count never exceeds the budget."""
    order = list(range(len(rows)))
    if shuffle:
        random.Random(seed).shuffle(order)
    batch, used = [], 0
    for i in order:
        n = int(length(rows[i]))
        if n > token_budget:
            raise ValueError(f"one example ({n} tokens) exceeds token budget {token_budget}")
        if batch and used + n > token_budget:
            yield batch; batch, used = [], 0
        batch.append(rows[i]); used += n
    if batch:
        yield batch


@dataclass
class TrainConfig:
    seed: int = 0
    token_budget: int = 8192
    learning_rate: float = 2e-4
    epochs: int = 1
    stage: str = "stage0"
    unit: str = "train"
    cap_hours: float = 5.0
    estimated_hours: float = 0.0
    bf16: bool = True
    checkpoint_every: int = 100
    control_path: str = ""


class GPUHours:
    def __init__(self, path: Path, cap: float, stage: str | None = None):
        self.path, self.cap, self.stage = Path(path), float(cap), stage

    def used(self) -> float:
        if not self.path.exists():
            return 0.0
        rows = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
        return sum(float(row.get("hours", 0)) for row in rows
                   if self.stage is None or row.get("stage", self.stage) == self.stage)

    def reserve(self, estimate: float) -> None:
        if self.used() + estimate > self.cap + 1e-12:
            raise GPUCapReached(f"GPU-hour cap {self.cap:g} would be exceeded")

    def record(self, unit: str, seconds: float, **extra) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"unit": unit, "hours": seconds / 3600.0, "seconds": seconds, **extra}
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, (json.dumps(row, sort_keys=True) + "\n").encode()); os.fsync(fd)
        finally:
            os.close(fd)


def _versions() -> dict[str, str]:
    out = {"python": __import__("platform").python_version()}
    for name in ("torch", "transformers", "peft", "accelerate"):
        try: out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: out[name] = "missing"
    return out


def manifest(config: TrainConfig, hashes: dict[str, str], model) -> dict:
    return {"git_sha": git_sha(), "config": asdict(config), "hashes": dict(hashes),
            "versions": _versions(), "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def _trainable_state(model) -> dict[str, torch.Tensor]:
    trainable = {name for name, param in model.named_parameters() if param.requires_grad}
    return {name: value.detach().cpu() for name, value in model.state_dict().items() if name in trainable}


def save_checkpoint(model, optimizer, step: int, out: Path, meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "checkpoint.pt.part"
    torch.save({"trainable_model": _trainable_state(model), "optimizer": optimizer.state_dict(), "step": step}, tmp)
    os.replace(tmp, out / "checkpoint.pt")
    (out / "manifest.json").write_text(json.dumps(meta, indent=2, sort_keys=True))


def train(model, packed_rows: Sequence, config: TrainConfig, out: Path, gpu_hours_path: Path,
          *, hashes: dict[str, str] | None = None) -> dict:
    """Train/resume pointer CE. Each row is ``(PackedWindow, target_index)``."""
    seed_everything(config.seed)
    meter = GPUHours(gpu_hours_path, config.cap_hours, config.stage); meter.reserve(config.estimated_hours)
    model.enable_training_memory_features()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device).train()
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config.learning_rate)
    start_step = 0
    checkpoint = out / "checkpoint.pt"
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["trainable_model"], strict=False)
        optimizer.load_state_dict(state["optimizer"]); start_step = state["step"]
    meta = manifest(config, hashes or {}, model)
    last_accounted = time.monotonic(); step = 0
    for epoch in range(config.epochs):
        for batch in token_batches(packed_rows, config.token_budget,
                                   length=lambda x: len(x[0].input_ids), seed=config.seed + epoch):
            if config.control_path and Path(config.control_path).exists() \
                    and Path(config.control_path).read_text().strip().upper() == "STOP":
                raise InterruptedError("STOP requested")
            meter.reserve(0.0)
            step += 1
            if step <= start_step: continue
            optimizer.zero_grad(set_to_none=True)
            losses = []
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=config.bf16 and device.type == "cuda"):
                for packed, target in batch:
                    target = torch.as_tensor(target if isinstance(target, (list, tuple)) else [target])
                    losses.append(model(packed, labels=target).loss)
                loss = torch.stack(losses).mean()
            loss.backward(); optimizer.step()
            now = time.monotonic()
            meter.record(config.unit, now - last_accounted, stage=config.stage, device=str(device), step=step)
            last_accounted = now
            if step % config.checkpoint_every == 0:
                save_checkpoint(model, optimizer, step, out, meta)
    save_checkpoint(model, optimizer, step, out, meta)
    (out / "done").write_text(str(step))
    return {**meta, "steps": step}
