"""Question generation for S1-D with semantic held-out exclusion."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from ..ledger import append
from ..schema import Doc, Span
from ..v2 import labels as v2
from . import labels as heldout
from .schema import Option, Question, QuestionType


@dataclass(frozen=True)
class TrainingQuestion:
    doc_id: str
    state: str
    question: Question
    target: int
    source_span: tuple[int, int] | None = None
    hard_negative: bool = False


def _raw(span: Span) -> str:
    return span.label_raw.strip().lower()


def _option(raw: str, rng: random.Random) -> Option:
    node, description, paraphrases = v2.NATIVE[raw]
    name = rng.choice((raw.replace("_", " "), *paraphrases))
    return Option(name, description)


def _overlaps(span: Span, held: Sequence[Span]) -> bool:
    return any(span.start < h.end and span.end > h.start for h in held)


def generate_questions(docs: Iterable[Doc], *, seed: int = 0, variant: str = "all-sources",
                       min_options: int = 2, max_options: int = 64,
                       hard_negative_hook: Callable[[Doc], Iterable[Span]] | None = None,
                       config: dict | None = None, ledger_path: Path | None = None) -> list[TrainingQuestion]:
    """Generate span Choice plus document Noul/Score questions from v1 documents."""
    if variant not in ("all-sources", "no-nemotron"):
        raise ValueError("unknown source variant")
    if not 2 <= min_options <= max_options <= 64:
        raise ValueError("training option sets must be between 2 and 64")
    cfg = config or heldout.load()
    rng = random.Random(seed)
    evaluation_only = set(cfg.get("never_train_native_names", ()))
    vocabulary = [n for n in sorted(v2.NATIVE) if not heldout.excluded(n, cfg) and n not in evaluation_only]
    output: list[TrainingQuestion] = []
    for doc in docs:
        if variant == "no-nemotron" and doc.dataset == "nemotron":
            continue
        forbidden = [s for s in doc.spans if s.label_raw and heldout.excluded(_raw(s), cfg)]
        positives = [s for s in doc.pii_spans() if _raw(s) in v2.NATIVE and _raw(s) not in evaluation_only
                     and s not in forbidden]
        # Native OOD names are evaluation-only even if an accidentally mixed source arrives.
        if doc.dataset in cfg.get("never_train_native_sets", ()):
            positives = []
        for span in positives:
            n = rng.randint(min_options, min(max_options, len(vocabulary) + 1))
            distractors = [x for x in vocabulary if not v2.compatible(x, _raw(span))]
            rng.shuffle(distractors)
            raws = [_raw(span), *distractors[:max(0, n - 2)]]
            opts = [_option(x, rng) for x in raws] + [Option(heldout.NOT_PII, "the span is not PII")]
            target_name = opts[0].name
            rng.shuffle(opts)
            q = Question(QuestionType.CHOICE, "Classify the marked span.", "Use its meaning and context.",
                         tuple(opts), id=f"{doc.doc_id}:{span.start}:{span.end}", span=(span.start, span.end))
            output.append(TrainingQuestion(doc.doc_id, doc.text, q,
                                           next(i for i, o in enumerate(opts) if o.name == target_name),
                                           (span.start, span.end)))
        if hard_negative_hook:
            for span in hard_negative_hook(doc):
                if _overlaps(span, forbidden) or _overlaps(span, positives):
                    continue
                distractors = vocabulary[:max(1, min(max_options - 1, 7))]
                opts = [_option(x, rng) for x in distractors] + [Option(heldout.NOT_PII)]
                q = Question("choice", "Classify the marked span.", options=tuple(opts),
                             id=f"{doc.doc_id}:hn:{span.start}:{span.end}", span=(span.start, span.end))
                output.append(TrainingQuestion(doc.doc_id, doc.text, q, len(opts) - 1,
                                               (span.start, span.end), True))
        contains = bool(positives)
        nq = Question("noul", "Does this document contain personal information?", id=f"{doc.doc_id}:noul")
        output.append(TrainingQuestion(doc.doc_id, doc.text, nq, int(contains)))
        levels = tuple(Option(str(i), value=float(i)) for i in range(5))
        sq = Question("score", "Rate the document's PII sensitivity.", options=levels, id=f"{doc.doc_id}:score")
        output.append(TrainingQuestion(doc.doc_id, doc.text, sq, min(4, len(positives))))
    digest = question_set_hash(output)
    append({"unit": "s1d_questions", "seed": seed, "variant": variant,
            "questions": len(output), "question_set_sha256": digest}, path=ledger_path)
    return output


def question_set_hash(questions: Sequence[TrainingQuestion]) -> str:
    rows = []
    for q in questions:
        d = asdict(q)
        d["question"]["type"] = str(q.question.type.value)
        rows.append(d)
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def assert_no_heldout_leakage(questions: Sequence[TrainingQuestion], config: dict | None = None) -> None:
    cfg = config or heldout.load()
    forbidden = {str(x).lower().replace("_", " ") for key in ("names", "paraphrases")
                 for x in cfg["exclusions"][key]}
    for row in questions:
        option_text = {o.name.lower().replace("_", " ") for o in row.question.options}
        if option_text & forbidden:
            raise AssertionError(f"held-out option leaked into {row.question.id}")
        if row.source_span and row.target < len(row.question.options):
            target = row.question.options[row.target].name.lower().replace("_", " ")
            if target in forbidden:
                raise AssertionError(f"held-out target leaked into {row.question.id}")
