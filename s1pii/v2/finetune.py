"""Conditional stage A: end-to-end typing with the top encoder layers fine-tuned.

Runs only if gate A fires (``gates.py``). Warm start: a copy of the v1 encoder (the level-1
encoder itself stays frozen, so P_b is unchanged) plus the cached-feature typing head. The
top ``n_layers`` transformer layers and the final norm of the copy are trained jointly with
the head; spans are represented from a ``context``-token window centred on the span and
label texts are re-encoded by the same (trainable) encoder every step. Items, label
sampling, compatibility masks and gold dropping are exactly those of ``head.py``.

Export: ``typing_encoder.safetensors`` + encoder config + tokenizer + ``head/``. Features
for inference are then re-extracted with ``Extractor(..., typing_encoder=<export>)``.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..schema import Doc
from ..model.encode import tokenize_doc, special_ids
from . import labels as LB
from .head import HeadConfig, build_items, label_text_variants, load_head, save_head, NONE_TARGET

FT_VERSION = "v2-finetune-0.1"


@dataclass
class FinetuneConfig:
    n_layers: int = 4
    context: int = 512
    batch: int = 64
    lr_encoder: float = 2e-5
    lr_head: float = 3e-4
    max_steps: int = 6000
    epochs: float = 1.0
    seed: int = 1


def _trainable_top(encoder, n: int) -> list:
    for p in encoder.parameters():
        p.requires_grad_(False)
    layers = getattr(encoder, "layers", None) or getattr(getattr(encoder, "encoder", None), "layer", None)
    if layers is None:
        raise ValueError("cannot find transformer layers on this encoder")
    params = []
    for layer in list(layers)[-n:]:
        for p in layer.parameters():
            p.requires_grad_(True); params.append(p)
    for name in ("final_norm", "norm"):
        m = getattr(encoder, name, None)
        if m is not None:
            for p in m.parameters():
                p.requires_grad_(True); params.append(p)
    return params


def finetune(v1_dir: Path, store: Path, head_dir: Path, docs: list[Doc], out: Path, heldout_nodes: set[str],
             cfg: FinetuneConfig | None = None, device: str | None = None, log=print) -> Path:
    from ..model.train import load_exported
    cfg = cfg or FinetuneConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.seed); rng = np.random.default_rng(cfg.seed)
    model, tok, man = load_exported(v1_dir)
    enc = copy.deepcopy(model.encoder).to(device)
    del model
    head, hman = load_head(head_dir, device)
    hcfg = head.cfg
    items, stats = build_items(store, heldout_nodes, hcfg)
    vocab = items.labels
    if vocab != hman["vocab"]:
        raise ValueError("training items differ from the head's vocabulary")
    variants = [label_text_variants(n) for n in vocab]
    compat = torch.as_tensor(np.array([[LB.compatible(a, b) for b in vocab] for a in vocab]), device=device)
    by_id = {d.doc_id: d for d in docs}
    miss = sorted(set(items.doc_ids.tolist()) - set(by_id))
    if miss:
        raise KeyError(f"{len(miss)} training docs missing, e.g. {miss[:3]}")
    tok_cache: dict[str, object] = {}
    pre, suf = special_ids(tok)
    size = cfg.context - len(pre) - len(suf)
    params = _trainable_top(enc, cfg.n_layers)
    head.train()
    opt = torch.optim.AdamW([{"params": params, "lr": cfg.lr_encoder},
                             {"params": head.parameters(), "lr": cfg.lr_head}], weight_decay=0.01)
    N = len(items.y)
    steps = min(cfg.max_steps, max(1, int(cfg.epochs * math.ceil(N / cfg.batch))))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[cfg.lr_encoder, cfg.lr_head], total_steps=steps, pct_start=0.05)
    pad = tok.pad_token_id or 0

    def window(did: str, cs: int, ce: int):
        td = tok_cache.get(did)
        if td is None:
            td = tokenize_doc(by_id[did], tok)
            if len(tok_cache) > 20000:
                tok_cache.clear()
            tok_cache[did] = td
        st, en = td.offsets[:, 0], td.offsets[:, 1]
        hit = np.nonzero((st < ce) & (en > cs) & (en > st))[0]
        if len(hit) == 0:
            return None
        i, j = int(hit[0]), int(hit[-1])
        if j - i + 1 > size:
            return None
        a = max(0, min((i + j) // 2 - size // 2, len(td.ids) - size))
        b = min(len(td.ids), a + size)
        return pre + td.ids[a:b] + suf, i - a + len(pre), j - a + len(pre)

    def encode_labels(texts: list[str]) -> torch.Tensor:
        e = tok(texts, padding=True, return_tensors="pt", return_special_tokens_mask=True, truncation=True, max_length=64)
        sp = e.pop("special_tokens_mask")
        h = enc(input_ids=e["input_ids"].to(device), attention_mask=e["attention_mask"].to(device)).last_hidden_state.float()
        m = (e["attention_mask"].bool() & ~sp.bool()).to(device).unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp(min=1)

    t0 = time.time()
    order = rng.permutation(N); pos = 0
    for step in range(steps):
        if pos + cfg.batch > N:
            order = rng.permutation(N); pos = 0
        idx = order[pos:pos + cfg.batch]; pos += cfg.batch
        wins = [window(str(items.doc_ids[r]), int(items.start[r]), int(items.end[r])) for r in idx]
        keep = [k for k, w in enumerate(wins) if w is not None]
        idx = idx[keep]; wins = [wins[k] for k in keep]
        L_ = max(len(w[0]) for w in wins)
        ids = torch.full((len(wins), L_), pad, dtype=torch.long); att = torch.zeros_like(ids)
        for k, (w, _, _) in enumerate(wins):
            ids[k, :len(w)] = torch.as_tensor(w); att[k, :len(w)] = 1
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if device == "cuda" else torch.autocast("cpu", enabled=False)
        with ctx:
            h = enc(input_ids=ids.to(device), attention_mask=att.to(device)).last_hidden_state.float()
            reps = []
            for k, (_, i, j) in enumerate(wins):
                reps.append(torch.cat([h[k, i], h[k, j], h[k, i:j + 1].mean(0)]))
            rep = torch.stack(reps)
            y = torch.as_tensor(items.y[idx], device=device)
            golds = np.unique(items.y[idx][items.y[idx] >= 0])
            m = int(rng.integers(hcfg.min_labels, hcfg.max_labels + 1))
            others = np.setdiff1d(np.arange(len(vocab)), golds)
            Lids = np.concatenate([golds, rng.choice(others, max(0, min(len(others), m - len(golds))), replace=False)]).astype(np.int64)
            texts = []
            for i_ in Lids:
                u = rng.random()
                mode = "name" if u < hcfg.p_name_only else ("para" if u < hcfg.p_name_only + hcfg.p_paraphrase else "desc")
                texts.append(variants[i_][mode][rng.integers(len(variants[i_][mode]))])
            lv = encode_labels(texts)
            s = head.span_vec(rep, torch.as_tensor(items.v1[idx], device=device), torch.as_tensor(items.pb[idx], device=device))
            lg = head.logits(s, head.label_vec(lv))
        Lt = torch.as_tensor(Lids, device=device)
        posv = torch.full((len(vocab),), -1, dtype=torch.long, device=device)
        posv[Lt] = torch.arange(len(Lids), device=device)
        mask = torch.zeros_like(lg, dtype=torch.bool)
        has = y >= 0
        if has.any():
            c = compat[y[has]][:, Lt] & (Lt.unsqueeze(0) != y[has].unsqueeze(1))
            mask[has, :len(Lids)] = c
            drop = has & (torch.rand(len(y), device=device) < hcfg.p_gold_drop)
            if drop.any():
                mask[drop, :len(Lids)] |= compat[y[drop]][:, Lt]
                y = torch.where(drop, torch.full_like(y, NONE_TARGET), y)
        target = torch.where(y >= 0, posv[y.clamp(min=0)], torch.full_like(y, len(Lids)))
        loss = F.cross_entropy(lg.float().masked_fill(mask, -1e4), target)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params + list(head.parameters()), 1.0)
        opt.step(); sched.step()
        if (step + 1) % 100 == 0 or step + 1 == steps:
            log(json.dumps({"ft_step": step + 1, "of": steps, "loss": round(float(loss.detach()), 4), "sec": round(time.time() - t0, 1)}))
    head.eval(); enc.eval()
    return export_typing_encoder(enc, tok, head, hcfg, out, {"finetune": asdict(cfg), "v1": man["weights_sha256"],
                                                             "vocab": vocab, "items": stats,
                                                             "heldout_nodes": sorted(heldout_nodes)})


def export_typing_encoder(enc, tok, head, hcfg: HeadConfig, out: Path, info: dict) -> Path:
    from safetensors.torch import save_file
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in enc.state_dict().items()}, str(tmp / "typing_encoder.safetensors"))
    enc.config.save_pretrained(tmp / "encoder_config")
    tok.save_pretrained(tmp)
    save_head(head, tmp / "head", hcfg, {"vocab": info["vocab"], "finetuned": True, "heldout_nodes": info["heldout_nodes"]})
    sha = hashlib.sha256((tmp / "typing_encoder.safetensors").read_bytes()).hexdigest()
    (tmp / "typing_manifest.json").write_text(json.dumps({"version": FT_VERSION, "weights_sha256": sha, **info},
                                                         indent=2, default=str))
    shutil.rmtree(out, ignore_errors=True)
    os.replace(tmp, out)
    return out


def load_typing_encoder(path: Path, device: str):
    from transformers import AutoConfig, AutoModel
    from safetensors.torch import load_file
    path = Path(path)
    man = json.loads((path / "typing_manifest.json").read_text())
    enc = AutoModel.from_config(AutoConfig.from_pretrained(path / "encoder_config"))
    enc.load_state_dict(load_file(str(path / "typing_encoder.safetensors")))
    return enc.to(device).eval(), man
