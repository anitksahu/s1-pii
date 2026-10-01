"""Question generation for S1-D with semantic held-out exclusion."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from ..ledger import append
from ..schema import CANONICAL_TYPES, Doc, Span
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
    forbidden_spans: tuple[tuple[int, int], ...] = ()


def _raw(span: Span) -> str:
    return span.label_raw.strip().lower()


def _option(raw: str, rng: random.Random) -> Option:
    node, description, paraphrases = v2.NATIVE[raw]
    forbidden = {x.lower().replace("_", " ") for x in CANONICAL_TYPES}
    texts = [x for x in (raw.replace("_", " "), *paraphrases)
             if x.lower().replace("_", " ") not in forbidden]
    name = rng.choice(texts or [description])
    return Option(name, description)


def _overlaps(span: Span, held: Sequence[Span]) -> bool:
    return any(span.start < h.end and span.end > h.start for h in held)


def generate_questions(docs: Iterable[Doc], *, seed: int = 0, variant: str = "all-sources",
                       min_options: int = 2, max_options: int = 64,
                       hard_negative_hook: Callable[[Doc], Iterable[Span]] | None = None,
                       config: dict | None = None, ledger_path: Path | None = None,
                       remove_parent_labels: bool = False) -> list[TrainingQuestion]:
    """Generate span Choice plus document Noul/Score questions from v1 documents."""
    if variant not in ("all-sources", "no-nemotron"):
        raise ValueError("unknown source variant")
    if not 2 <= min_options <= max_options <= 64:
        raise ValueError("training option sets must be between 2 and 64")
    cfg = config or heldout.load()
    rng = random.Random(seed)
    evaluation_only = heldout.ood_only_names(cfg)
    vocabulary = heldout.training_vocabulary(cfg, remove_parents=remove_parent_labels)
    if ledger_path is not None:
        heldout.record_nearest_trained_neighbours(ledger_path, cfg, remove_parents=remove_parent_labels)
    output: list[TrainingQuestion] = []
    for doc in docs:
        if variant == "no-nemotron" and doc.dataset == "nemotron":
            continue
        if doc.dataset in cfg.get("never_train_native_sets", ()):
            continue
        forbidden = [s for s in doc.spans if s.label_raw and heldout.excluded(_raw(s), cfg)]
        forbidden_ranges = tuple((s.start, s.end) for s in forbidden)
        positives = [s for s in doc.pii_spans() if _raw(s) in v2.NATIVE and _raw(s) not in evaluation_only
                     and s not in forbidden]
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
                                           (span.start, span.end), False, forbidden_ranges))
        if hard_negative_hook:
            for span in hard_negative_hook(doc):
                if _overlaps(span, doc.spans):
                    continue
                distractors = vocabulary[:max(1, min(max_options - 1, 7))]
                opts = [_option(x, rng) for x in distractors] + [Option(heldout.NOT_PII)]
                q = Question("choice", "Classify the marked span.", options=tuple(opts),
                             id=f"{doc.doc_id}:hn:{span.start}:{span.end}", span=(span.start, span.end))
                output.append(TrainingQuestion(doc.doc_id, doc.text, q, len(opts) - 1,
                                               (span.start, span.end), True, forbidden_ranges))
        if forbidden:
            continue
        contains = bool(positives)
        nq = Question("noul", "Does this document contain personal information?", id=f"{doc.doc_id}:noul")
        output.append(TrainingQuestion(doc.doc_id, doc.text, nq, int(contains), forbidden_spans=forbidden_ranges))
        levels = tuple(Option(str(i), value=float(i)) for i in range(5))
        sq = Question("score", "Rate the document's PII sensitivity.", options=levels, id=f"{doc.doc_id}:score")
        output.append(TrainingQuestion(doc.doc_id, doc.text, sq, min(4, len(positives)),
                                       forbidden_spans=forbidden_ranges))
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
    exclusions = heldout.semantic_exclusions(cfg)
    forbidden = {str(x).lower().replace("_", " ")
                 for key in ("names", "paraphrases", "replication_labels",
                             "replication_synonyms_all_sources", "replication_nodes")
                 for x in exclusions[key]}
    for row in questions:
        option_text = {o.name.lower().replace("_", " ") for o in row.question.options}
        if option_text & forbidden:
            raise AssertionError(f"held-out option leaked into {row.question.id}")
        if row.source_span and row.target < len(row.question.options):
            target = row.question.options[row.target].name.lower().replace("_", " ")
            if target in forbidden:
                raise AssertionError(f"held-out target leaked into {row.question.id}")
        if row.source_span and any(row.source_span[0] < b and row.source_span[1] > a
                                   for a, b in row.forbidden_spans):
            raise AssertionError(f"held-out span used by {row.question.id}")


def pack_training_questions(questions: Sequence[TrainingQuestion], tokenizer, *, stride: int = 384,
                            layout: str = "shared", device=None, make_block_mask: bool = True):
    """Convert generated questions to real 512-token state windows and short branches."""
    from .packer import BranchText, pack_window
    output = []
    for row in questions:
        encoded = tokenizer(row.state, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        ids = list(encoded["input_ids"])
        offsets = list(encoded.get("offset_mapping", ()))
        windows = [(a, min(a + 512, len(ids))) for a in range(0, max(1, len(ids)), stride)] or [(0, 0)]
        if windows and windows[-1][1] < len(ids):
            windows.append((max(0, len(ids) - 512), len(ids)))
        chosen = windows[0]
        span_text = row.question.instructions
        left = ""
        if row.source_span is not None:
            start, end = row.source_span
            span_text = row.state[start:end]
            if offsets:
                for window in windows:
                    wa, wb = window
                    if wa < wb and offsets[wa][0] <= start and offsets[wb - 1][1] >= end:
                        chosen = window; break
                first = next((i for i, (_a, b) in enumerate(offsets) if b > start), 0)
                ca = offsets[max(chosen[0], first - 8)][0]
                left = row.state[ca:start]
        wa, wb = chosen
        if offsets and wa < wb:
            state = row.state[offsets[wa][0]:offsets[wb - 1][1]]
        else:
            state = row.state
        if row.source_span is None and row.question.criteria:
            span_text = f"{span_text} {row.question.criteria}"
        options = [o.text() for o in row.question.options]
        packed = pack_window(tokenizer, state, options, [BranchText(span_text, left)], state_tokens=512,
                             layout=layout, device=device, make_block_mask=make_block_mask)
        output.append((packed, row.target))
    return output
