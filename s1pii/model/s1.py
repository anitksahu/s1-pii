"""S1-PII model: encoder -> linear emissions over 37 BIOES tags -> constrained CRF."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .crf import CRF, K
from .encode import Example, allowed_matrix


class S1Model(nn.Module):
    def __init__(self, encoder: nn.Module, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, K)
        self.crf = CRF()

    @classmethod
    def from_pretrained_encoder(cls, name_or_path: str, revision: str | None = None, dropout: float = 0.1,
                                gradient_checkpointing: bool = False, attn_implementation: str | None = None):
        from transformers import AutoModel
        kw = {"revision": revision} if revision else {}
        if attn_implementation:
            kw["attn_implementation"] = attn_implementation
        enc = AutoModel.from_pretrained(name_or_path, **kw)
        if gradient_checkpointing:
            enc.gradient_checkpointing_enable()
        return cls(enc, enc.config.hidden_size, dropout)

    def emissions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.head(self.dropout(h)).float()

    def forward(self, batch: dict) -> torch.Tensor:
        em = self.emissions(batch["input_ids"], batch["attention_mask"])
        em = gather_tokens(em, batch["n_prefix"], batch["tok_len"].max().item())
        return self.crf.nll(em, batch["tok_mask"], batch["allowed"])


def gather_tokens(em: torch.Tensor, n_prefix: torch.Tensor, L: int) -> torch.Tensor:
    """Select the real-token emissions (skip the prefix special tokens) into (B, L, K)."""
    B = em.shape[0]
    idx = n_prefix.view(B, 1) + torch.arange(L, device=em.device).view(1, L)
    idx = idx.clamp(max=em.shape[1] - 1)
    return em.gather(1, idx.unsqueeze(-1).expand(B, L, em.shape[-1]))


def collate(examples: list[Example], pad_id: int, with_targets: bool = True) -> dict:
    B = len(examples)
    Lin = max(len(e.input_ids) for e in examples)
    Lt = max(e.n_tok for e in examples)
    ids = torch.full((B, Lin), pad_id, dtype=torch.long)
    att = torch.zeros((B, Lin), dtype=torch.long)
    tok_mask = torch.zeros((B, Lt), dtype=torch.bool)
    allowed = torch.ones((B, Lt, K), dtype=torch.bool)
    for i, e in enumerate(examples):
        ids[i, :len(e.input_ids)] = torch.tensor(e.input_ids)
        att[i, :len(e.input_ids)] = 1
        tok_mask[i, :e.n_tok] = True
        if with_targets:
            allowed[i, :e.n_tok] = torch.from_numpy(allowed_matrix(e.target))
    return {"input_ids": ids, "attention_mask": att, "tok_mask": tok_mask, "allowed": allowed,
            "n_prefix": torch.tensor([e.n_prefix for e in examples]),
            "tok_len": torch.tensor([e.n_tok for e in examples])}
