"""Offset re-anchoring and word-granularity expansion for predictions."""
from __future__ import annotations

import re
from dataclasses import replace
from typing import Sequence

from ..schema import Span

_WORD = re.compile(r"\S+")


def word_bounds(text: str) -> list[tuple[int, int]]:
    return [(m.start(), m.end()) for m in _WORD.finditer(text)]


def expand_to_words(text: str, start: int, end: int, bounds: list[tuple[int, int]] | None = None) -> tuple[int, int]:
    """Expand [start, end) to cover every whitespace-delimited word it overlaps.

    Applied to predictions only, identically for every system, so subword and character
    maskers are compared at the same granularity. Partial masking across words (a card
    number split by spaces) is still visible to the span-exposure metric."""
    bounds = bounds if bounds is not None else word_bounds(text)
    s, e = start, end
    for ws, we in bounds:
        if we <= start:
            continue
        if ws >= end:
            break
        s, e = min(s, ws), max(e, we)
    return s, e


def reanchor(text: str, span: Span, window: int = 8) -> Span | None:
    """Repair small offset drift using the span's surface. Returns the fixed span, the span
    unchanged if already consistent, or None if it cannot be anchored."""
    if span.surface is None or text[span.start:span.end] == span.surface:
        return span
    surf = span.surface
    lo, hi = max(0, span.start - window), min(len(text), span.end + window)
    best = None
    idx = text.find(surf, lo, hi)
    while idx != -1:
        d = abs(idx - span.start)
        if best is None or d < best[0]:
            best = (d, idx)
        idx = text.find(surf, idx + 1, hi)
    if best is not None:
        return replace(span, start=best[1], end=best[1] + len(surf))
    stripped = surf.strip()
    if stripped and stripped != surf:
        return reanchor(text, replace(span, surface=stripped), window)
    return None


def reanchor_all(text: str, spans: Sequence[Span]) -> tuple[list[Span], dict]:
    out, repaired, dropped = [], 0, 0
    for s in spans:
        r = reanchor(text, s)
        if r is None:
            dropped += 1
            continue
        if r is not s:
            repaired += 1
        out.append(r)
    return out, {"repaired": repaired, "dropped": dropped, "total": len(spans)}
