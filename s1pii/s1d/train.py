"""Resumable S1-D training primitives with token and GPU-hour budgets."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import random
import time
from collections import OrderedDict
from dataclasses import asdict, dataclass
from itertools import islice
from pathlib import Path
from typing import Callable, Iterable, Iterator, Sequence

import numpy as np
import torch

from ..ledger import git_sha


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
    data_seed: int | None = None
    init_seed: int | None = None
    order_seed: int | None = None
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
    optimizer_condition: str = "old"
    lora_learning_rate: float = 2e-4
    pointer_learning_rate: float = 5e-5
    token_learning_rate: float = 2e-4
    warmup_fraction: float = 0.06
    gradient_clip: float | None = None
    gradient_accumulation: int = 1
    max_steps: int | None = None

    def seeds(self) -> tuple[int, int, int]:
        """Resolve legacy ``seed`` to all three streams unless explicitly separated."""
        return (self.seed if self.data_seed is None else self.data_seed,
                self.seed if self.init_seed is None else self.init_seed,
                self.seed if self.order_seed is None else self.order_seed)


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
        """Retained for callers; GPU-hour totals are accounting-only and never stop work."""
        return

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


def _rng_state() -> dict:
    state = {"python": random.getstate(), "numpy": np.random.get_state(),
             "torch": torch.random.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(state: dict) -> None:
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.random.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(model, optimizer, step: int, out: Path, meta: dict, scheduler=None, *,
                    microbatches_seen: int = 0, questions_seen: int = 0) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "checkpoint.pt.part"
    torch.save({"trainable_model": _trainable_state(model), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "rng": _rng_state(), "step": step, "microbatches_seen": microbatches_seen,
                "questions_seen": questions_seen}, tmp)
    os.replace(tmp, out / "checkpoint.pt")
    (out / "manifest.json").write_text(json.dumps(meta, indent=2, sort_keys=True))


def _stable_optimizer(model, config: TrainConfig, device: torch.device):
    groups = {"lora": [], "pointer": [], "tokens": []}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" in name:
            group = "lora"
        elif name.startswith(("decision_projection.", "option_projection.", "pointer_bias")):
            group = "pointer"
        elif "trainable_tokens" in name:
            group = "tokens"
        else:
            raise ValueError(f"stable optimizer has no parameter group for {name}")
        groups[group].append(parameter)
    rates = {"lora": config.lora_learning_rate, "pointer": config.pointer_learning_rate,
             "tokens": config.token_learning_rate}
    missing = [name for name, parameters in groups.items() if not parameters]
    if missing:
        raise ValueError(f"stable optimizer parameter groups are empty: {missing}")
    optimizer = torch.optim.AdamW([{"params": groups[name], "lr": rates[name], "name": name}
                                   for name in ("lora", "pointer", "tokens")],
                                  fused=device.type == "cuda")
    return optimizer


def _linear_schedule(optimizer, *, total_steps: int, warmup_fraction: float):
    warmup = max(1, round(total_steps * warmup_fraction))
    def scale(step):
        if step < warmup:
            return float(step + 1) / warmup
        return max(0.0, float(total_steps - step) / max(1, total_steps - warmup))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def _grad_norm(parameters) -> float:
    norms = [parameter.grad.detach().float().norm(2) for parameter in parameters
             if parameter.requires_grad and parameter.grad is not None]
    return float(torch.stack(norms).norm(2)) if norms else 0.0


def optimizer_batches(microbatches: Iterable[list], accumulation: int) -> Iterator[list[list]]:
    """Group consecutive token-budget microbatches into one optimizer update."""
    if accumulation < 1:
        raise ValueError("gradient accumulation must be positive")
    iterator = iter(microbatches)
    while group := list(islice(iterator, accumulation)):
        yield group


def question_weighted_mean(values: Sequence, counts: Sequence[int]):
    """Mean of microbatch means, weighted by their question counts."""
    if len(values) != len(counts) or not values or any(count < 1 for count in counts):
        raise ValueError("values and positive question counts must have the same nonzero length")
    total = sum(counts)
    return sum(value * (count / total) for value, count in zip(values, counts))


def _item_question_count(item) -> int:
    target = item[1]
    return len(target) if isinstance(target, (list, tuple)) else 1


def train(model, packed_rows: Sequence, config: TrainConfig, out: Path, gpu_hours_path: Path,
          *, hashes: dict[str, str] | None = None,
          checkpoint_callback: Callable[[object, int, dict], None] | None = None,
          callback_every: int | None = None,
          callback_every_microbatches: int | None = None,
          step_callback: Callable[[object, int, dict, Sequence], None] | None = None) -> dict:
    """Train/resume pointer CE. Each row is ``(PackedWindow, target_index)``."""
    data_seed, _init_seed, order_seed = config.seeds()
    if callback_every_microbatches and callback_every_microbatches % config.gradient_accumulation:
        raise ValueError("exposure checkpoints must align with optimizer-update boundaries")
    seed_everything(order_seed)
    meter = GPUHours(gpu_hours_path, config.cap_hours, config.stage); meter.reserve(config.estimated_hours)
    model.enable_training_memory_features()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device).train()
    if config.optimizer_condition == "stable":
        optimizer = _stable_optimizer(model, config, device)
    elif config.optimizer_condition == "old":
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=config.learning_rate, fused=device.type == "cuda")
    else:
        raise ValueError(f"unknown optimizer condition {config.optimizer_condition!r}")
    microbatch_count = sum(1 for _ in token_batches(
        packed_rows, config.token_budget, length=lambda x: len(x[0].input_ids), seed=data_seed))
    update_count = ((microbatch_count + config.gradient_accumulation - 1)
                    // config.gradient_accumulation) * config.epochs
    total_steps = min(update_count, config.max_steps) if config.max_steps is not None else update_count
    scheduler = None
    if config.optimizer_condition == "stable":
        scheduler = _linear_schedule(optimizer, total_steps=total_steps,
                                     warmup_fraction=config.warmup_fraction)
    start_step = 0
    checkpoint = out / "checkpoint.pt"
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(state["trainable_model"], strict=False)
        optimizer.load_state_dict(state["optimizer"]); start_step = state["step"]
        if scheduler is not None and state.get("scheduler") is not None:
            scheduler.load_state_dict(state["scheduler"])
        if state.get("rng") is not None:
            _set_rng_state(state["rng"])
    meta = manifest(config, hashes or {}, model)
    warmup_steps = max(1, round(total_steps * config.warmup_fraction)) if scheduler else 0
    last_accounted = time.monotonic(); step = 0; stats = None
    microbatches_seen = 0; questions_seen = 0; last_callback_step = 0
    for epoch in range(config.epochs):
        microbatches = token_batches(packed_rows, config.token_budget,
                                     length=lambda x: len(x[0].input_ids), seed=data_seed + epoch)
        for update_batches in optimizer_batches(microbatches, config.gradient_accumulation):
            if config.control_path and Path(config.control_path).exists() \
                    and Path(config.control_path).read_text().strip().upper() == "STOP":
                raise InterruptedError("STOP requested")
            meter.reserve(0.0)
            step += 1
            flat_batch = [item for batch in update_batches for item in batch]
            update_questions = sum(_item_question_count(item) for item in flat_batch)
            microbatches_seen += len(update_batches); questions_seen += update_questions
            if step <= start_step:
                continue
            optimizer.zero_grad(set_to_none=True)
            microbatch_losses = []
            microbatch_questions = []
            for batch in update_batches:
                if config.control_path and Path(config.control_path).exists() \
                        and Path(config.control_path).read_text().strip().upper() == "STOP":
                    raise InterruptedError("STOP requested")
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=config.bf16 and device.type == "cuda"):
                    can_batch = (hasattr(model, "forward_many")
                                 and all(item[0].dense_mask.numel() for item in batch))
                    if can_batch:
                        groups = OrderedDict()
                        for item in batch:
                            packed, target = item[:2]
                            groups.setdefault(len(packed.input_ids), []).append((packed, target))
                        losses = []
                        counts = []
                        for group in groups.values():
                            outputs = model.forward_many([packed for packed, _target in group],
                                                         targets=[target for _packed, target in group])
                            losses.extend(output.loss for output in outputs)
                            counts.extend(_item_question_count(item) for item in group)
                    else:
                        losses = []
                        counts = []
                        for item in batch:
                            packed, target = item[:2]
                            target = torch.as_tensor(target if isinstance(target, (list, tuple)) else [target])
                            losses.append(model(packed, labels=target).loss)
                            counts.append(_item_question_count(item))
                    loss = question_weighted_mean(losses, counts)
                batch_questions = sum(counts)
                (loss * (batch_questions / update_questions)).backward()
                microbatch_losses.append(float(loss.detach()))
                microbatch_questions.append(batch_questions)
            loss_value = float(question_weighted_mean(microbatch_losses, microbatch_questions))
            grad_norm = _grad_norm(model.parameters())
            learning_rates = {str(group.get("name", "all")): float(group["lr"])
                              for group in optimizer.param_groups}
            if config.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            now = time.monotonic()
            meter.record(config.unit, now - last_accounted, stage=config.stage, device=str(device), step=step)
            last_accounted = now
            stats = {"loss": loss_value, "grad_norm": grad_norm,
                     "learning_rates": learning_rates,
                     "question_count": update_questions, "questions_seen": questions_seen,
                     "microbatch_count": len(update_batches),
                     "microbatches_seen": microbatches_seen}
            if step == 1 or step % 25 == 0:
                print(f"{config.unit}: step {step} loss {stats['loss']:.6f}", flush=True)
            if step_callback is not None:
                state = _rng_state()
                try:
                    step_callback(model, step, stats, flat_batch)
                finally:
                    _set_rng_state(state); model.train()
            if step % config.checkpoint_every == 0:
                save_checkpoint(model, optimizer, step, out, meta, scheduler,
                                microbatches_seen=microbatches_seen, questions_seen=questions_seen)
            should_callback = ((callback_every and step % callback_every == 0)
                               or (callback_every_microbatches
                                   and microbatches_seen % callback_every_microbatches == 0))
            if checkpoint_callback is not None and should_callback:
                state = _rng_state()
                try:
                    checkpoint_callback(model, step, stats)
                finally:
                    _set_rng_state(state); model.train()
                last_callback_step = step
            if config.max_steps is not None and step >= config.max_steps:
                break
        if config.max_steps is not None and step >= config.max_steps:
            break
    if checkpoint_callback is not None and stats is not None and last_callback_step != step:
        state = _rng_state()
        try:
            checkpoint_callback(model, step, stats)
        finally:
            _set_rng_state(state); model.train()
    save_checkpoint(model, optimizer, step, out, meta, scheduler,
                    microbatches_seen=microbatches_seen, questions_seen=questions_seen)
    (out / "done").write_text(str(step))
    return {**meta, "steps": step, "optimizer_updates": step,
            "microbatches": microbatches_seen, "questions_seen": questions_seen,
            "warmup_updates": warmup_steps}
