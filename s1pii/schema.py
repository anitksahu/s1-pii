"""Canonical span schema and offset contract.

Offset contract
---------------
* Offsets are Python ``str`` indices (Unicode code points) into ``Doc.text`` exactly as the
  loader produced it. No normalization is applied after loading, so no offset remapping is
  ever needed. Loaders that build text (for example from tokens or conversation turns) own
  the offset arithmetic and are tested for it.
* ``start`` is inclusive, ``end`` exclusive, ``0 <= start < end <= len(text)``.
* Every gold span, and every predicted span that carries a surface, must satisfy
  ``text[start:end] == surface``. ``validate_doc`` enforces this for gold; adapters enforce
  it for predictions after re-anchoring (see ``s1pii.eval.align``).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable, Sequence
import json

# Canonical PII types (core-8 plus OTHER_PII). Metrics are type-agnostic by default.
PERSON = "PERSON"
ADDRESS = "ADDRESS"
EMAIL = "EMAIL"
PHONE = "PHONE"
URL = "URL"
DATE = "DATE"
ACCOUNT_NUMBER = "ACCOUNT_NUMBER"
SECRET = "SECRET"
OTHER_PII = "OTHER_PII"

CANONICAL_TYPES: tuple[str, ...] = (
    PERSON, ADDRESS, EMAIL, PHONE, URL, DATE, ACCOUNT_NUMBER, SECRET, OTHER_PII,
)
CORE8: tuple[str, ...] = CANONICAL_TYPES[:8]

# Non-PII outcomes a raw label can map to.
IGNORE = "IGNORE"    # excluded from both leak and over-redaction (e.g. quasi-identifiers)
NOT_PII = "NOT_PII"  # explicitly non-PII; masking it counts as over-redaction

ALL_TARGETS: tuple[str, ...] = CANONICAL_TYPES + (IGNORE, NOT_PII)


class OffsetError(ValueError):
    """A span violates the offset contract."""


@dataclass(frozen=True)
class Span:
    doc_id: str
    start: int
    end: int
    label_canonical: str
    label_raw: str = ""
    score: float = 1.0
    source: str = "gold"
    surface: str | None = None

    def __post_init__(self) -> None:
        import operator
        if isinstance(self.start, bool) or isinstance(self.end, bool):
            raise OffsetError(f"{self.doc_id}: offsets must be integers, not bool")
        try:
            object.__setattr__(self, "start", operator.index(self.start))
            object.__setattr__(self, "end", operator.index(self.end))
        except TypeError as e:
            raise OffsetError(f"{self.doc_id}: offsets must be integers") from e
        if self.start < 0 or self.end <= self.start:
            raise OffsetError(f"{self.doc_id}: bad span [{self.start},{self.end})")
        if self.label_canonical not in ALL_TARGETS:
            raise OffsetError(f"{self.doc_id}: unknown canonical label {self.label_canonical!r}")
        if not (0.0 <= float(self.score) <= 1.0):
            raise OffsetError(f"{self.doc_id}: score {self.score} outside [0,1]")

    @property
    def is_pii(self) -> bool:
        return self.label_canonical in CANONICAL_TYPES

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Span":
        return Span(**d)


@dataclass(frozen=True)
class Doc:
    doc_id: str
    text: str
    spans: tuple[Span, ...] = ()
    dataset: str = ""
    split: str = ""
    cluster_id: str = ""      # unit for the cluster bootstrap (conversation, template, source doc)
    meta: dict = field(default_factory=dict)

    def pii_spans(self) -> list[Span]:
        return [s for s in self.spans if s.is_pii]

    def ignore_spans(self) -> list[Span]:
        return [s for s in self.spans if s.label_canonical == IGNORE]

    def to_json(self) -> str:
        return json.dumps({
            "doc_id": self.doc_id, "text": self.text, "dataset": self.dataset,
            "split": self.split, "cluster_id": self.cluster_id, "meta": self.meta,
            "spans": [s.to_dict() for s in self.spans],
        }, ensure_ascii=False)

    @staticmethod
    def from_json(line: str) -> "Doc":
        d = json.loads(line)
        return Doc(
            doc_id=d["doc_id"], text=d["text"], dataset=d.get("dataset", ""),
            split=d.get("split", ""), cluster_id=d.get("cluster_id", ""),
            meta=d.get("meta", {}), spans=tuple(Span.from_dict(s) for s in d["spans"]),
        )


def check_span(text: str, span: Span) -> None:
    if span.end > len(text):
        raise OffsetError(f"{span.doc_id}: span end {span.end} beyond text length {len(text)}")
    if span.surface is not None and text[span.start:span.end] != span.surface:
        raise OffsetError(
            f"{span.doc_id}: surface mismatch at [{span.start},{span.end}): "
            f"text has {text[span.start:span.end]!r}, span says {span.surface!r}"
        )


def validate_doc(doc: Doc) -> Doc:
    """Enforce the offset contract on every span of a document. Returns the doc."""
    if not doc.doc_id:
        raise OffsetError("doc_id is required")
    for s in doc.spans:
        if s.doc_id != doc.doc_id:
            raise OffsetError(f"span doc_id {s.doc_id!r} != doc {doc.doc_id!r}")
        check_span(doc.text, s)
    return doc


def validate_predictions(doc: Doc, preds: Sequence[Span]) -> None:
    for s in preds:
        if s.doc_id != doc.doc_id:
            raise OffsetError(f"prediction doc_id {s.doc_id!r} != doc {doc.doc_id!r}")
        check_span(doc.text, s)


def write_jsonl(docs: Iterable[Doc], path) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for d in docs:
            f.write(d.to_json() + "\n")
            n += 1
    return n


def read_jsonl(path) -> list[Doc]:
    with open(path, encoding="utf-8") as f:
        return [Doc.from_json(line) for line in f if line.strip()]
