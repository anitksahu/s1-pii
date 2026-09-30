"""Tokenization, BIOES targets with partial labels, and token windows.

* The whole document is tokenized once with character offsets (fast tokenizer, no special
  tokens). Offsets are trimmed of surrounding whitespace so BPE tokens that carry a leading
  space map to their visible characters.
* Targets are per-token *allowed tag sets* (a single tag for known gold; every tag where
  gold is unknown: IGNORE regions and spans cut by a window edge).
* Windows hold ``max_len - 2`` tokens plus the tokenizer's special tokens. Training uses
  ``stride`` = window size (no overlap) by default; inference uses 50% overlap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..schema import Doc, CANONICAL_TYPES, IGNORE
from .crf import tag, K

ANY = -1


@dataclass
class TokDoc:
    doc_id: str
    ids: list[int]
    offsets: np.ndarray        # (n, 2) trimmed char offsets
    target: np.ndarray         # (n,) tag id, or ANY (-1)
    span_of: np.ndarray        # (n,) index of the gold PII span covering the token, or -1


def _trim(text: str, a: int, b: int) -> tuple[int, int]:
    while a < b and text[a].isspace():
        a += 1
    while b > a and text[b - 1].isspace():
        b -= 1
    return a, b


def tokenize_doc(doc: Doc, tokenizer) -> TokDoc:
    enc = tokenizer(doc.text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    offs = np.array([_trim(doc.text, a, b) for a, b in enc["offset_mapping"]], dtype=np.int64).reshape(-1, 2)
    n = len(enc["input_ids"])
    target = np.zeros(n, dtype=np.int64)
    span_of = np.full(n, -1, dtype=np.int64)
    starts = offs[:, 0]; ends = offs[:, 1]
    visible = ends > starts
    for s in doc.spans:                                  # IGNORE first; PII overrides
        if s.label_canonical == IGNORE:
            hit = visible & (starts < s.end) & (ends > s.start)
            target[hit] = ANY
    pii = sorted(doc.pii_spans(), key=lambda s: (s.start, -s.end))
    for si, s in enumerate(pii):
        hit = np.nonzero(visible & (starts < s.end) & (ends > s.start) & (span_of < 0))[0]
        if len(hit) == 0:
            continue
        t = CANONICAL_TYPES.index(s.label_canonical)
        hit = np.arange(hit[0], hit[-1] + 1)          # whitespace-only tokens inside the span are I
        if len(hit) == 1:
            target[hit[0]] = tag("S", t)
        else:
            target[hit[0]] = tag("B", t)
            target[hit[1:-1]] = tag("I", t)
            target[hit[-1]] = tag("E", t)
        span_of[hit] = si
    return TokDoc(doc.doc_id, list(enc["input_ids"]), offs, target, span_of)


def token_windows(n: int, size: int, stride: int) -> list[tuple[int, int]]:
    if n <= size:
        return [(0, n)]
    out, a = [], 0
    while True:
        b = min(a + size, n)
        out.append((a, b))
        if b == n:
            return out
        a += stride


def window_target(td: TokDoc, a: int, b: int) -> np.ndarray:
    """Targets for tokens a..b-1; a gold span cut by the window edge becomes ANY inside it."""
    t = td.target[a:b].copy()
    so = td.span_of[a:b]
    for edge_span in {int(so[0]) if a > 0 else -1, int(so[-1]) if b < len(td.ids) else -1}:
        if edge_span >= 0:
            t[so == edge_span] = ANY
    return t


def allowed_matrix(target: np.ndarray, k: int = K) -> np.ndarray:
    m = np.zeros((len(target), k), dtype=bool)
    known = target >= 0
    m[np.nonzero(known)[0], target[known]] = True
    m[~known] = True
    return m


def special_ids(tokenizer) -> tuple[list[int], list[int]]:
    """Prefix/suffix special tokens the model expects around a sequence (e.g. [CLS] ... [SEP])."""
    probe = tokenizer("a", add_special_tokens=True)["input_ids"]
    core = tokenizer("a", add_special_tokens=False)["input_ids"]
    i = probe.index(core[0])
    return probe[:i], probe[i + len(core):]


@dataclass
class Example:
    doc_id: str
    input_ids: np.ndarray         # int32, with special tokens
    n_prefix: int
    n_tok: int                    # real tokens in the window
    target: np.ndarray            # (n_tok,)
    tok_start: int                # window start in doc tokens


def target_is_valid(target: np.ndarray, nt: int | None = None) -> bool:
    """True iff some valid BIOES path agrees with every known tag (ANY is a wildcard)."""
    from .crf import constraint_masks, NT
    nt = nt or NT
    tr, st, en = (m.numpy() for m in constraint_masks(nt))
    allowed = allowed_matrix(target, 1 + 4 * nt)
    reach = st & allowed[0]
    for i in range(1, len(target)):
        reach = (reach[:, None] & tr).any(0) & allowed[i]
        if not reach.any():
            return False
    return bool((reach & en).any())


class InvalidTarget(ValueError):
    pass


def train_examples(docs: list[Doc], tokenizer, max_len: int = 1024, stride: int | None = None) -> list[Example]:
    pre, suf = special_ids(tokenizer)
    size = max_len - len(pre) - len(suf)
    stride = stride or size
    out, bad = [], []
    for d in docs:
        td = tokenize_doc(d, tokenizer)
        if not td.ids:
            continue
        for a, b in token_windows(len(td.ids), size, stride):
            tgt = window_target(td, a, b)
            if not target_is_valid(tgt):
                bad.append(d.doc_id)
                continue
            out.append(Example(d.doc_id, np.asarray(pre + td.ids[a:b] + suf, dtype=np.int32), len(pre), b - a, tgt, a))
    if bad:
        raise InvalidTarget(f"{len(bad)} windows have no valid BIOES path, e.g. docs {bad[:3]}")
    return out
