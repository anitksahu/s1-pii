"""S1-PII training: resumable, seeded, bf16, length-bucketed token-budget batches.

    python -m s1pii.model.train --variant all-sources --seed 1 --out /content/ckpt/all-s1 \
        --mirror /content/drive/MyDrive/s1pii/ckpt/all-s1

Variants (prereg v0): ``all-sources`` = Nemotron train + Gretel EN train + synthetic
conversations; ``no-nemotron`` drops Nemotron. The Nemotron/Gretel dev slices
(``bench.dev_slice``) are always excluded. Checkpoints hold model, optimizer, scheduler,
all RNG states and the sampler position; they are written to local disk atomically
(tmp dir + rename, DONE marker), copied to the Drive mirror in a background thread, and
only the last two are kept. ``--resume`` continues from the newest complete checkpoint in
``--out`` or, if empty, in ``--mirror``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import threading
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import torch

from .encode import Example, train_examples
from .s1 import S1Model, collate

TRAIN_VERSION = "s1-train-v0.1"


@dataclass
class TrainConfig:
    backbone: str = "answerdotai/ModernBERT-large"
    backbone_revision: str | None = None
    variant: str = "all-sources"
    seed: int = 1
    max_len: int = 1024
    token_budget: int = 16384          # padded tokens per micro-batch
    grad_accum: int = 2
    epochs: float = 2.0
    lr_encoder: float = 3e-5
    lr_head: float = 1e-3
    weight_decay: float = 0.01
    warmup_frac: float = 0.06
    dropout: float = 0.1
    synth_n: int = 20000
    synth_seed: int = 17                # fixed across training seeds: data is constant, init/order vary
    max_train_docs: int | None = None
    ckpt_every: int = 500
    log_every: int = 50
    gradient_checkpointing: bool = True
    bf16: bool = True
    extra: dict = field(default_factory=dict)


# ------------------------------------------------------------------ data

def training_docs(cfg: TrainConfig):
    from ..data import loaders as L
    from ..data.synth import generate
    from ..bench import dev_slice
    sources = {"synthetic_conv": generate(cfg.synth_n, seed=cfg.synth_seed)}
    if cfg.variant in ("all-sources",):
        sources["nemotron"] = dev_slice(L.load("nemotron", "train", purpose="train"))[1]
    elif cfg.variant != "no-nemotron":
        raise ValueError(f"unknown variant {cfg.variant!r}")
    sources["gretel"] = dev_slice(L.load("gretel", "train", purpose="train"))[1]
    docs = [d for v in sources.values() for d in v]
    if cfg.max_train_docs:
        rng = random.Random(cfg.seed)
        docs = rng.sample(docs, min(cfg.max_train_docs, len(docs)))
    return docs, {k: len(v) for k, v in sources.items()}


class BucketSampler:
    """Deterministic token-budget batches over length-sorted examples, shuffled per epoch.
    State = (epoch, position) so a resumed run continues exactly where it stopped."""

    def __init__(self, lengths: list[int], budget: int, seed: int):
        self.lengths, self.budget, self.seed = lengths, budget, seed
        self.epoch, self.pos = 0, 0
        self._build()

    def _build(self):
        rng = np.random.default_rng(self.seed + 1000 * self.epoch)
        order = np.argsort(np.array(self.lengths) + rng.random(len(self.lengths)) * 8, kind="stable")
        batches, cur, mx = [], [], 0
        for i in order:
            L = self.lengths[i]
            if cur and max(mx, L) * (len(cur) + 1) > self.budget:
                batches.append(cur); cur, mx = [], 0
            cur.append(int(i)); mx = max(mx, L)
        if cur:
            batches.append(cur)
        rng.shuffle(batches)
        self.batches = batches

    def __len__(self):
        return len(self.batches)

    def next(self) -> list[int]:
        if self.pos >= len(self.batches):
            self.epoch += 1; self.pos = 0; self._build()
        b = self.batches[self.pos]; self.pos += 1
        return b

    def state(self) -> dict:
        return {"epoch": self.epoch, "pos": self.pos}

    def load(self, st: dict) -> None:
        self.epoch, self.pos = st["epoch"], st["pos"]
        self._build()


# ------------------------------------------------------------------ checkpoints

def _rng_state() -> dict:
    st = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng(st: dict) -> None:
    random.setstate(st["python"]); np.random.set_state(st["numpy"]); torch.set_rng_state(st["torch"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


def save_checkpoint(out: Path, step: int, model, opt, sched, sampler, cfg: TrainConfig, mirror: Path | None,
                    keep: int = 2, threads: list | None = None) -> Path:
    tmp = out / f".tmp-step-{step:08d}"
    final = out / f"step-{step:08d}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                "sampler": sampler.state(), "rng": _rng_state(), "step": step, "config": asdict(cfg)},
               tmp / "state.pt")
    (tmp / "DONE").write_text(str(step))
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    for old in sorted(out.glob("step-*"))[:-keep]:
        shutil.rmtree(old, ignore_errors=True)
    if mirror is not None:
        def copy():
            mirror.mkdir(parents=True, exist_ok=True)
            dst_tmp = mirror / f".tmp-{final.name}"
            shutil.rmtree(dst_tmp, ignore_errors=True)
            shutil.copytree(final, dst_tmp)
            shutil.rmtree(mirror / final.name, ignore_errors=True)
            os.replace(dst_tmp, mirror / final.name)
            for old in sorted(mirror.glob("step-*"))[:-keep]:
                shutil.rmtree(old, ignore_errors=True)
        t = threading.Thread(target=copy, daemon=False)
        t.start()
        if threads is not None:
            threads.append(t)
    return final


def latest_checkpoint(*dirs: Path | None) -> Path | None:
    for d in dirs:
        if d is None or not d.exists():
            continue
        done = [p for p in sorted(d.glob("step-*")) if (p / "DONE").exists()]
        if done:
            return done[-1]
    return None


# ------------------------------------------------------------------ loop

def build_optimizer(model: S1Model, cfg: TrainConfig, total_steps: int):
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": cfg.lr_encoder}, {"params": head, "lr": cfg.lr_head}],
                            weight_decay=cfg.weight_decay)
    warm = max(1, int(cfg.warmup_frac * total_steps))

    def lr_lambda(s: int) -> float:          # linear warmup, then linear decay to 0
        if s < warm:
            return (s + 1) / warm
        return max(0.0, (total_steps - s) / max(1, total_steps - warm))

    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


def train(cfg: TrainConfig, out: Path, *, mirror: Path | None = None, resume: bool = True,
          model: S1Model | None = None, tokenizer=None, examples: list[Example] | None = None,
          data_manifest: dict | None = None, max_steps: int | None = None, device: str | None = None) -> Path:
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out.mkdir(parents=True, exist_ok=True)
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg.backbone, revision=cfg.backbone_revision)
    if examples is None:
        docs, counts = training_docs(cfg)
        examples = train_examples(docs, tokenizer, cfg.max_len)
        data_manifest = {"source_counts": counts, "n_examples": len(examples)}
    if model is None:
        model = S1Model.from_pretrained_encoder(cfg.backbone, cfg.backbone_revision, cfg.dropout,
                                                cfg.gradient_checkpointing)
    model.to(device)
    sampler = BucketSampler([len(e.input_ids) for e in examples], cfg.token_budget, cfg.seed)
    steps_per_epoch = math.ceil(len(sampler) / cfg.grad_accum)
    total = max_steps or max(1, int(cfg.epochs * steps_per_epoch))
    opt, sched = build_optimizer(model, cfg, total)
    step = 0
    ck = latest_checkpoint(out, mirror) if resume else None
    if ck is not None:
        st = torch.load(ck / "state.pt", map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        sampler.load(st["sampler"]); _set_rng(st["rng"]); step = st["step"]
        print(f"resumed from {ck} at step {step}", flush=True)
    log = open(out / "train_log.jsonl", "a")
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    use_bf16 = cfg.bf16 and device == "cuda"
    threads: list = []
    model.train()
    t0 = time.time()
    while step < total:
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(cfg.grad_accum):
            batch = collate([examples[i] for i in sampler.next()], pad)
            batch = {k: v.to(device) for k, v in batch.items()}
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_bf16 else torch.autocast("cpu", enabled=False)
            with ctx:
                loss = model(batch) / cfg.grad_accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step}")
            loss.backward()
            loss_acc += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); step += 1
        if step % cfg.log_every == 0 or step == total:
            log.write(json.dumps({"step": step, "loss": loss_acc, "lr": sched.get_last_lr()[0],
                                  "epoch": sampler.epoch, "sec": round(time.time() - t0, 1)}) + "\n"); log.flush()
            print(f"step {step}/{total} loss {loss_acc:.4f}", flush=True)
        if step % cfg.ckpt_every == 0 or step == total:
            save_checkpoint(out, step, model, opt, sched, sampler, cfg, mirror, threads=threads)
    for t in threads:
        t.join()
    return export(model, tokenizer, cfg, out / "final", data_manifest or {}, step)


def export(model: S1Model, tokenizer, cfg: TrainConfig, path: Path, data_manifest: dict, step: int) -> Path:
    from safetensors.torch import save_file
    from ..ledger import git_sha
    from .predict import weights_sha256
    path.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(state, str(path / "model.safetensors"))
    tokenizer.save_pretrained(path)
    if hasattr(model.encoder, "config"):
        model.encoder.config.save_pretrained(path / "encoder_config")
    manifest = {"train_version": TRAIN_VERSION, "config": asdict(cfg), "steps": step, "code_sha": git_sha(),
                "data": data_manifest, "weights_sha256": weights_sha256(path), "torch": torch.__version__}
    (path / "s1_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return path


def load_exported(path: str | Path, device: str | None = None):
    """(model, tokenizer, manifest) from an ``export`` directory."""
    from transformers import AutoConfig, AutoModel, AutoTokenizer
    from safetensors.torch import load_file
    path = Path(path)
    manifest = json.loads((path / "s1_manifest.json").read_text())
    enc = AutoModel.from_config(AutoConfig.from_pretrained(path / "encoder_config"))
    model = S1Model(enc, enc.config.hidden_size, manifest["config"].get("dropout", 0.1))
    res = model.load_state_dict(load_file(str(path / "model.safetensors")), strict=False)
    bad_missing = [k for k in res.missing_keys if ".pooler." not in k]   # unused pooler heads only
    if bad_missing or res.unexpected_keys:
        raise RuntimeError(f"export mismatch: missing {bad_missing[:5]}, unexpected {res.unexpected_keys[:5]}")
    tok = AutoTokenizer.from_pretrained(path)
    return model.to(device or "cpu").eval(), tok, manifest


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all-sources")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mirror", type=Path)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--max-train-docs", type=int)
    ap.add_argument("--max-steps", type=int)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--token-budget", type=int, default=16384)
    a = ap.parse_args(argv)
    cfg = TrainConfig(variant=a.variant, seed=a.seed, max_train_docs=a.max_train_docs, epochs=a.epochs,
                      token_budget=a.token_budget)
    print(train(cfg, a.out, mirror=a.mirror, resume=not a.no_resume, max_steps=a.max_steps))


if __name__ == "__main__":
    main()
