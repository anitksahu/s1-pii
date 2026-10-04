"""Shared-option and Kev-layout packing with branch-isolated attention."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

SPECIAL_TOKENS = ("<state>", "</state>", "<opt>", "</opt>", "<q>", "<decide>")


@dataclass
class PackedWindow:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: object
    dense_mask: torch.Tensor
    decide_indices: torch.Tensor
    option_indices: torch.Tensor
    option_ranges: tuple[tuple[int, int], ...]
    branch_ranges: tuple[tuple[int, int], ...]
    state_range: tuple[int, int]
    layout: str = "shared"
    question_ranges: tuple[tuple[int, int], ...] = ()
    true_length: int = 0

    def batch(self, device=None) -> dict:
        return {"input_ids": self.input_ids.unsqueeze(0).to(device),
                "position_ids": self.position_ids.unsqueeze(0).to(device),
                "attention_mask": self.attention_mask}


@dataclass(frozen=True)
class BranchText:
    span: str
    left_context: str = ""


def _ids(tokenizer, text: str, max_length: int | None = None) -> list[int]:
    out = tokenizer(text, add_special_tokens=False, truncation=max_length is not None,
                    max_length=max_length) if max_length else tokenizer(text, add_special_tokens=False)
    return list(out["input_ids"] if isinstance(out, dict) else out.input_ids)


def _special(tokenizer, token: str) -> int:
    value = tokenizer.convert_tokens_to_ids(token)
    if value is None or value == getattr(tokenizer, "unk_token_id", None):
        raise ValueError(f"tokenizer does not define reserved token {token}")
    return int(value)


def _dense_mask(n: int, state: tuple[int, int], options: Sequence[tuple[int, int]],
                branches: Sequence[tuple[int, int]]) -> torch.Tensor:
    mask = torch.zeros((n, n), dtype=torch.bool)
    sa, sb = state
    for q in range(sa, sb):
        mask[q, sa:q + 1] = True
    for a, b in options:
        mask[a:b, sa:sb] = True
        for q in range(a, b):
            mask[q, a:q + 1] = True
    for a, b in branches:
        mask[a:b, sa:sb] = True
        for oa, ob in options:
            mask[a:b, oa:ob] = True
        for q in range(a, b):
            mask[q, a:q + 1] = True
    return mask


def _kev_dense_mask(n: int, state: tuple[int, int], groups) -> torch.Tensor:
    mask = torch.zeros((n, n), dtype=torch.bool)
    sa, sb = state
    for q in range(sa, sb):
        mask[q, sa:q + 1] = True
    for question, options, decide in groups:
        qa, qb = question
        mask[qa:qb, sa:sb] = True
        for q in range(qa, qb):
            mask[q, qa:q + 1] = True
        for oa, ob in options:
            mask[oa:ob, sa:sb] = True
            mask[oa:ob, qa:qb] = True
            for q in range(oa, ob):
                mask[q, oa:q + 1] = True
        da, db = decide
        mask[da:db, sa:sb] = True
        mask[da:db, qa:qb] = True
        for oa, ob in options:
            mask[da:db, oa:ob] = True
        for q in range(da, db):
            mask[q, da:q + 1] = True
    return mask


def flex_block_mask(dense: torch.Tensor, device=None):
    """Build a FlexAttention BlockMask. A clear error is preferable to implicit masking."""
    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError as e:  # pragma: no cover - dependency guard
        raise RuntimeError("torch>=2.5 with FlexAttention is required") from e
    n = dense.shape[-1]
    dense = dense.to(device or dense.device)
    def allowed(_b, _h, q, kv):
        return dense[q, kv]
    return create_block_mask(allowed, B=1, H=None, Q_LEN=n, KV_LEN=n, device=str(dense.device))


def flex_batch_block_mask(dense: torch.Tensor, device=None):
    """Build one explicit BlockMask for a batch of equally padded packed windows."""
    try:
        from torch.nn.attention.flex_attention import create_block_mask
    except ImportError as e:  # pragma: no cover - dependency guard
        raise RuntimeError("torch>=2.5 with FlexAttention is required") from e
    if dense.ndim != 3 or dense.shape[-1] != dense.shape[-2]:
        raise ValueError("batched dense masks must have shape [batch, length, length]")
    dense = dense.to(device or dense.device)
    batch, length, _ = dense.shape
    def allowed(b, _h, q, kv):
        return dense[b, q, kv]
    return create_block_mask(allowed, B=batch, H=None, Q_LEN=length, KV_LEN=length,
                             device=str(dense.device))


def length_bucket(length: int, buckets=(512, 1024, 2048, 4096, 8192)) -> int:
    return next((b for b in buckets if length <= b), 1 << (length - 1).bit_length())


def pack_window(tokenizer, state: str, options: Sequence[str], branches: Sequence[str], *,
                state_tokens: int = 512, left_context_tokens: int = 8, make_block_mask: bool = True,
                layout: str = "shared", device=None, pad_to_bucket: bool = True,
                keep_dense_mask: bool = True) -> PackedWindow:
    if not 2 <= len(options) <= 255:
        raise ValueError("2-255 shared options are required")
    if layout not in ("shared", "kev"):
        raise ValueError("layout must be shared or kev")
    sid = {s: _special(tokenizer, s) for s in SPECIAL_TOKENS}
    state_body = _ids(tokenizer, state, state_tokens)
    state_ids = [sid["<state>"], *state_body, sid["</state>"]]
    opt_ids = [[sid["<opt>"], *_ids(tokenizer, o), sid["</opt>"]] for o in options]
    question_ids = []
    for text in branches:
        if isinstance(text, BranchText):
            context = _ids(tokenizer, text.left_context)[-left_context_tokens:]
            body = [*context, *_ids(tokenizer, text.span)]
        elif isinstance(text, tuple) and len(text) == 2:
            context = _ids(tokenizer, text[1])[-left_context_tokens:]
            body = [*context, *_ids(tokenizer, text[0])]
        else:
            body = _ids(tokenizer, str(text))
        question_ids.append([sid["<q>"], *body])
    ids, pos = list(state_ids), list(range(len(state_ids)))
    state_range = (0, len(ids)); option_ranges = []; branch_ranges = []; question_ranges = []; option_ends = []
    state_n, max_opt = len(state_ids), max(map(len, opt_ids))
    if layout == "shared":
        for seq in opt_ids:
            a = len(ids); ids += seq; option_ranges.append((a, len(ids))); option_ends.append(len(ids) - 1)
            pos += list(range(state_n, state_n + len(seq)))
        for question in question_ids:
            seq = [*question, sid["<decide>"]]
            a = len(ids); ids += seq; branch_ranges.append((a, len(ids)))
            question_ranges.append((a, len(ids) - 1))
            pos += list(range(state_n + max_opt, state_n + max_opt + len(seq)))
        dense_builder = lambda size: _dense_mask(size, state_range, option_ranges, branch_ranges)
    else:
        groups = []
        for question in question_ids:
            branch_start = len(ids)
            qa = len(ids); ids += question; question_range = (qa, len(ids)); question_ranges.append(question_range)
            pos += list(range(state_n, state_n + len(question)))
            local_opts = []
            for seq in opt_ids:
                a = len(ids); ids += seq; local_opts.append((a, len(ids))); option_ends.append(len(ids) - 1)
                pos += list(range(state_n + len(question), state_n + len(question) + len(seq)))
            da = len(ids); ids.append(sid["<decide>"]); decide_range = (da, len(ids))
            pos.append(state_n + len(question) + max_opt)
            branch_ranges.append((branch_start, len(ids)))
            option_ranges.extend(local_opts)
            groups.append((question_range, local_opts, decide_range))
        dense_builder = lambda size: _kev_dense_mask(size, state_range, groups)
    true_length = len(ids)
    packed_length = length_bucket(true_length) if pad_to_bucket else true_length
    pad_id = getattr(tokenizer, "pad_token_id", None)
    pad_id = 0 if pad_id is None else int(pad_id)
    ids += [pad_id] * (packed_length - true_length)
    pos += [0] * (packed_length - true_length)
    dense = dense_builder(packed_length)
    for i in range(true_length, packed_length):
        dense[i, i] = True
    block = flex_block_mask(dense, device=device) if make_block_mask else dense
    stored_dense = dense if keep_dense_mask else torch.empty(0, dtype=torch.bool)
    return PackedWindow(torch.tensor(ids, dtype=torch.long), torch.tensor(pos, dtype=torch.long), block, stored_dense,
                        torch.tensor([b - 1 for _, b in branch_ranges]), torch.tensor(option_ends),
                        tuple(option_ranges), tuple(branch_ranges), state_range, layout,
                        tuple(question_ranges), true_length)


def pack_separate(tokenizer, state: str, options: Sequence[str], branches: Sequence[str], **kwargs) -> list[PackedWindow]:
    return [pack_window(tokenizer, state, options, [b], **kwargs) for b in branches]
