"""Class-agnostic (nt=1) span proposer built from the v1 CRF path."""
from __future__ import annotations

from dataclasses import replace

from ..schema import Doc, IGNORE, PERSON
from ..model.encode import train_examples
from ..model.s1 import S1Model, collate
from .labels import excluded, load


def proposer_doc(doc: Doc, config: dict | None = None) -> Doc:
    """Map every observed PII span to one type; held-out spans become partial labels."""
    cfg = config or load()
    spans = []
    for span in doc.spans:
        if span.label_raw and excluded(span.label_raw, cfg):
            spans.append(replace(span, label_canonical=IGNORE))
        elif span.is_pii:
            # PERSON is canonical index zero, hence BIOES ids 1..4. Its name never
            # reaches the proposer: this is simply the single anonymous CRF type.
            spans.append(replace(span, label_canonical=PERSON))
        else:
            spans.append(span)
    return replace(doc, spans=tuple(spans))


def examples(docs, tokenizer, **kwargs):
    return train_examples([proposer_doc(d) for d in docs], tokenizer, num_types=1, **kwargs)


def proposer_model(encoder, hidden: int, dropout: float = 0.1) -> S1Model:
    return S1Model(encoder, hidden, dropout=dropout, num_types=1)


def proposer_collate(rows, pad_id: int, with_targets: bool = True):
    return collate(rows, pad_id, with_targets=with_targets, k=5)


def gate_g1(gold_by_window: dict[str, list[tuple[int, int]]],
            candidates_by_window: dict[str, list[tuple[int, int, float]]], limit: int = 64) -> dict:
    """Non-blocking exact-boundary recall gate at a fixed candidate cap."""
    hit = total = 0
    for window, gold in gold_by_window.items():
        proposed = {(a, b) for a, b, _ in sorted(candidates_by_window.get(window, ()),
                                                 key=lambda row: row[2], reverse=True)[:limit]}
        total += len(gold); hit += sum((a, b) in proposed for a, b in gold)
    return {"exact_boundary_recall": hit / total if total else 0.0, "hits": hit,
            "gold": total, "candidate_cap": limit, "blocking": False}
