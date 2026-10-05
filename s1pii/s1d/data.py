"""Question generation for S1-D with semantic held-out exclusion."""
from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

from ..ledger import append
from ..schema import CANONICAL_TYPES, Doc, Span
from ..v2 import labels as v2
from . import labels as heldout
from .schema import Option, Question, QuestionType

NOT_PII_DESCRIPTION = "the span is not PII"


@dataclass(frozen=True)
class TrainingQuestion:
    doc_id: str
    state: str
    question: Question
    target: int
    source_span: tuple[int, int] | None = None
    hard_negative: bool = False
    forbidden_spans: tuple[tuple[int, int], ...] = ()
    target_raw: str | None = None


def _raw(span: Span) -> str:
    return span.label_raw.strip().lower()


def _option(raw: str, rng: random.Random) -> Option:
    node, description, paraphrases = v2.NATIVE[raw]
    forbidden = {x.lower().replace("_", " ") for x in CANONICAL_TYPES}
    texts = [x for x in (raw.replace("_", " "), *paraphrases)
             if x.lower().replace("_", " ") not in forbidden]
    name = rng.choice(texts or [description])
    return Option(name, description)


def not_pii_option(*, descriptions: bool = True) -> Option:
    """The one canonical negative option used by training, calibration, and evaluation."""
    return Option(heldout.NOT_PII, NOT_PII_DESCRIPTION if descriptions else "")


def _choice_options(vocabulary: Sequence[str], rng: random.Random, *, min_options: int,
                    max_options: int, target_raw: str | None) -> tuple[tuple[Option, ...], int]:
    """Build indistinguishable positive/negative Choice option sets.

    ``target_raw=None`` denotes a hard negative, making NOT_PII the target. Both cases use the
    same option-count draw, option renderer, NOT_PII wording, and final shuffle.
    """
    n = rng.randint(min_options, min(max_options, len(vocabulary) + 1))
    if target_raw is None:
        distractors = list(vocabulary)
        rng.shuffle(distractors)
        raws = distractors[: n - 1]
    else:
        distractors = [raw for raw in vocabulary if not v2.compatible(raw, target_raw)]
        rng.shuffle(distractors)
        raws = [target_raw, *distractors[: max(0, n - 2)]]
    tagged = [(_option(raw, rng), target_raw is not None and i == 0) for i, raw in enumerate(raws)]
    tagged.append((not_pii_option(), target_raw is None))
    rng.shuffle(tagged)
    return (tuple(option for option, _target in tagged),
            next(i for i, (_option_, target) in enumerate(tagged) if target))


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
            opts, target = _choice_options(vocabulary, rng, min_options=min_options,
                                           max_options=max_options, target_raw=_raw(span))
            q = Question(QuestionType.CHOICE, "Classify the marked span.", "Use its meaning and context.",
                         opts, id=f"{doc.doc_id}:{span.start}:{span.end}", span=(span.start, span.end))
            output.append(TrainingQuestion(doc.doc_id, doc.text, q, target,
                                           (span.start, span.end), False, forbidden_ranges))
        if hard_negative_hook:
            for span in hard_negative_hook(doc):
                if _overlaps(span, doc.spans):
                    continue
                opts, target = _choice_options(vocabulary, rng, min_options=min_options,
                                               max_options=max_options, target_raw=None)
                q = Question("choice", "Classify the marked span.", "Use its meaning and context.", opts,
                             id=f"{doc.doc_id}:hn:{span.start}:{span.end}", span=(span.start, span.end))
                output.append(TrainingQuestion(doc.doc_id, doc.text, q, target,
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
        # Preserve every pre-coverage question hash. Coverage rows opt in to hashing the exact
        # canonical source-span label because some NATIVE descriptions are shared by aliases.
        if d["target_raw"] is None:
            del d["target_raw"]
        d["question"]["type"] = str(q.question.type.value)
        rows.append(d)
    return hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()


def training_batch_composition(rows: Sequence[TrainingQuestion]) -> dict:
    """Composition of one optimizer microbatch, using canonical Choice target labels."""
    description_to_raw = {description: raw for raw, (_node, description, _paraphrases) in v2.NATIVE.items()}
    kinds = Counter()
    targets = Counter()
    option_counts = []
    for row in rows:
        if row.question.type is QuestionType.NOUL:
            kind = "noul"
        elif row.question.type is QuestionType.SCORE:
            kind = "score"
        elif row.hard_negative:
            kind = "hard_negative"
        else:
            kind = "positive"
        kinds[kind] += 1
        option_counts.append(len(row.question.options))
        if row.question.type is QuestionType.CHOICE:
            option = row.question.options[row.target]
            target = (heldout.NOT_PII if option.name == heldout.NOT_PII
                      else description_to_raw.get(option.description, option.name))
            targets[target] += 1
    choice_count = sum(targets.values())
    dominant = targets.most_common(1)
    return {"question_count": len(rows),
            "counts": {name: kinds.get(name, 0)
                       for name in ("positive", "hard_negative", "noul", "score")},
            "choice_count": choice_count,
            "not_pii_target_share": targets.get(heldout.NOT_PII, 0) / choice_count if choice_count else None,
            "dominant_target_label": dominant[0][0] if dominant else None,
            "dominant_target_share": dominant[0][1] / choice_count if dominant else None,
            "mean_option_count": sum(option_counts) / len(option_counts) if option_counts else None}


def canonical_choice_target(row: TrainingQuestion) -> str | None:
    """Canonical raw target for Choice rows; document-level questions return ``None``."""
    if row.question.type is not QuestionType.CHOICE:
        return None
    if row.hard_negative:
        return heldout.NOT_PII
    if row.target_raw is not None:
        return row.target_raw
    description_to_raw = {description: raw for raw, (_node, description, _p) in v2.NATIVE.items()}
    option = row.question.options[row.target]
    raw = description_to_raw.get(option.description)
    if raw is None:
        raise ValueError(f"cannot recover canonical target for {row.question.id}")
    return raw


def _question_kind(row: TrainingQuestion) -> str:
    if row.question.type is QuestionType.NOUL:
        return "noul"
    if row.question.type is QuestionType.SCORE:
        return "score"
    return "hard_negative" if row.hard_negative else "positive"


def _weighted_without_replacement(rows: Sequence[TrainingQuestion], count: int, *, seed: int,
                                  exponent: float) -> list[TrainingQuestion]:
    """Deterministic PPS sampling; repeat only after the unique pool is exhausted."""
    if count <= 0:
        return []
    if not rows:
        raise ValueError("cannot sample positives from an empty pool")
    label_counts = Counter(canonical_choice_target(row) for row in rows)
    rng = random.Random(seed)
    output = []
    while len(output) < count:
        keyed = []
        for index, row in enumerate(rows):
            weight = label_counts[canonical_choice_target(row)] ** (-exponent)
            # Efraimidis-Spirakis exponential keys: smallest keys are sampled first.
            key = -math.log(max(rng.random(), 1e-300)) / weight
            keyed.append((key, index, row))
        keyed.sort(key=lambda item: (item[0], item[1]))
        output.extend(row for _key, _index, row in keyed[:count - len(output)])
    return output


def balanced_question_selection(rows: Sequence[TrainingQuestion], count: int, *, seed: int,
                                exponent: float = 0.5) -> list[TrainingQuestion]:
    """Balance positive targets while preserving every other kind's baseline count and row.

    The uniformly shuffled/repeated selection defines the exact hard-negative, Noul and Score
    shares. Positive slots are replaced by a weighted sample without replacement, where every
    positive row has weight ``label_count ** -exponent``.
    """
    if exponent < 0:
        raise ValueError("balance exponent must be non-negative")
    if not rows or count < 1:
        raise ValueError("balanced selection requires questions and a positive count")
    order = list(range(len(rows))); random.Random(seed).shuffle(order)
    baseline = [rows[order[i % len(order)]] for i in range(count)]
    positives = [row for row in rows if _question_kind(row) == "positive"]
    positive_count = sum(_question_kind(row) == "positive" for row in baseline)
    replacements = iter(_weighted_without_replacement(
        positives, positive_count, seed=seed, exponent=exponent))
    return [next(replacements) if _question_kind(row) == "positive" else row for row in baseline]


def coverage_composition(rows: Sequence[TrainingQuestion]) -> dict:
    """Coverage audit summary for an already selected question set."""
    kinds = Counter(_question_kind(row) for row in rows)
    positive_targets = Counter(canonical_choice_target(row) for row in rows
                               if _question_kind(row) == "positive")
    choice_targets = Counter(canonical_choice_target(row) for row in rows
                             if row.question.type is QuestionType.CHOICE)
    positive_total = sum(positive_targets.values())
    choice_total = sum(choice_targets.values())
    return {
        "questions": len(rows),
        "distinct_target_labels": len(positive_targets),
        "top_labels": [{"label": label, "count": n, "share": n / positive_total}
                       for label, n in positive_targets.most_common(5)],
        "not_pii_target_share": (choice_targets.get(heldout.NOT_PII, 0) / choice_total
                                 if choice_total else None),
        "counts": {name: kinds.get(name, 0)
                   for name in ("positive", "hard_negative", "noul", "score")},
    }


def assert_no_heldout_leakage(questions: Sequence[TrainingQuestion], config: dict | None = None) -> None:
    cfg = config or heldout.load()
    exclusions = heldout.semantic_exclusions(cfg)
    forbidden = {str(x).lower().replace("_", " ")
                 for key in ("names", "paraphrases", "replication_labels",
                             "replication_synonyms_all_sources", "replication_nodes")
                 for x in exclusions[key]}
    for row in questions:
        if row.target_raw and row.target_raw.lower().replace("_", " ") in forbidden:
            raise AssertionError(f"held-out canonical target leaked into {row.question.id}")
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


def question_pack_parts(row: TrainingQuestion, tokenizer, *, stride: int = 384, encoded=None):
    """Return the state slice, option text, and branch text used to pack one question."""
    from .packer import BranchText
    encoded = encoded or tokenizer(row.state, add_special_tokens=False,
                                   return_offsets_mapping=True, truncation=False)
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
    state = row.state[offsets[wa][0]:offsets[wb - 1][1]] if offsets and wa < wb else row.state
    if row.source_span is None and row.question.criteria:
        span_text = f"{span_text} {row.question.criteria}"
    return state, tuple(o.text() for o in row.question.options), BranchText(span_text, left)


def pack_training_questions(questions: Sequence[TrainingQuestion], tokenizer, *, stride: int = 384,
                            layout: str = "shared", device=None, make_block_mask: bool = True,
                            keep_dense_mask: bool = True):
    """Convert generated questions to real 512-token state windows and short branches."""
    from .packer import pack_window
    output = []
    for row in questions:
        state, options, branch = question_pack_parts(row, tokenizer, stride=stride)
        packed = pack_window(tokenizer, state, options, [branch], state_tokens=512,
                             layout=layout, device=device, make_block_mask=make_block_mask,
                             keep_dense_mask=keep_dense_mask)
        output.append((packed, row.target))
    return output


def evaluation_options(*, descriptions: bool = True) -> tuple[Option, ...]:
    """Frozen 55-way C3 option set plus the required not-PII option."""
    names = [label.name for label in v2.c3_label_set(False).labels]
    options = [Option(name.replace("_", " "), v2.NATIVE[name][1] if descriptions else "") for name in names]
    options.append(not_pii_option(descriptions=descriptions))
    return tuple(options)


def evaluation_option_index() -> dict[str, int]:
    """Canonical raw label to its fixed evaluation-option position."""
    names = [label.name for label in v2.c3_label_set(False).labels]
    return {name: i for i, name in enumerate([*names, heldout.NOT_PII])}


def span_evaluation_questions(docs: Sequence[Doc], candidates: dict[str, Sequence[Span]], *,
                              labels: Sequence[str], seed: int = 0,
                              descriptions: bool = True,
                              require_equal_negatives: bool = True,
                              pii_only: bool = False) -> list[TrainingQuestion]:
    """Gold-label questions plus an equal count of candidate spans disjoint from all gold.

    ``pii_only`` restricts gold spans to genuine PII (``Doc.pii_spans``), excluding IGNORE
    quasi-identifiers and NOT_PII; the in-domain calibration/sanity set uses it so temperature
    is fitted only on labels the model was actually trained to point at."""
    wanted = {name.lower() for name in labels}
    options = evaluation_options(descriptions=descriptions)
    option_index = evaluation_option_index()
    rows, negatives = [], []
    for doc in docs:
        gold_spans = doc.pii_spans() if pii_only else doc.spans
        for span in gold_spans:
            raw = _raw(span)
            if raw not in wanted:
                continue
            question = Question("choice", "Classify the marked span.", "Use its meaning and context.", options,
                                id=f"{doc.doc_id}:{span.start}:{span.end}", span=(span.start, span.end))
            rows.append(TrainingQuestion(doc.doc_id, doc.text, question, option_index[raw],
                                         (span.start, span.end)))
        for span in candidates.get(doc.doc_id, ()):
            if _overlaps(span, doc.spans):
                continue
            negatives.append((doc, span))
    rng = random.Random(seed); rng.shuffle(negatives)
    if require_equal_negatives and len(negatives) < len(rows):
        raise ValueError(f"need {len(rows)} non-overlapping proposer candidates, found {len(negatives)}")
    # Use every available proposer negative up to the gold count.  Evaluation metrics report
    # gold-label and not-PII accuracy separately, so a proposer yielding fewer candidates must
    # not make the entire evaluation impossible.  ``require_equal_negatives`` remains available
    # for callers that explicitly require a balanced set.
    for doc, span in negatives[:len(rows)]:
        question = Question("choice", "Classify the marked span.", "Use its meaning and context.", options,
                            id=f"{doc.doc_id}:negative:{span.start}:{span.end}", span=(span.start, span.end))
        rows.append(TrainingQuestion(doc.doc_id, doc.text, question, option_index[heldout.NOT_PII],
                                     (span.start, span.end), True))
    return rows


def pack_layout_ablation_questions(questions: Sequence[TrainingQuestion], tokenizer, *, windows: int,
                                   seed: int, layout: str, branches_per_window: int = 16,
                                   device=None, make_block_mask: bool = True,
                                   keep_dense_mask: bool = True, window_offset: int = 0):
    """Create identical fixed-option, 16-branch semantic windows for either layout."""
    from .packer import BranchText, pack_window
    options = evaluation_options(descriptions=False)
    option_names = [option.name for option in options]
    option_index = evaluation_option_index()
    description_to_raw = {description: name for name, (_node, description, _p) in v2.NATIVE.items()}
    by_doc = {}
    for row in questions:
        if row.source_span is None:
            continue
        target_option = row.question.options[row.target]
        target_name = heldout.NOT_PII if row.hard_negative else description_to_raw.get(
            target_option.description, target_option.name)
        if target_name not in option_index:
            continue
        by_doc.setdefault(row.doc_id, []).append((row, option_index[target_name]))
    if not by_doc:
        raise ValueError("layout ablation requires span questions")
    rng = random.Random(seed); docs = sorted(by_doc)
    output = []
    for index in range(windows):
        rows = by_doc[docs[(window_offset + index) % len(docs)]]
        chosen = [rows[rng.randrange(len(rows))] for _ in range(branches_per_window)]
        state = chosen[0][0].state; branches, targets = [], []
        for row, target in chosen:
            start, end = row.source_span
            branches.append(BranchText(state[start:end], state[max(0, start - 96):start]))
            targets.append(target)
        packed = pack_window(tokenizer, state, option_names, branches, state_tokens=512,
                             layout=layout, device=device, make_block_mask=make_block_mask,
                             keep_dense_mask=keep_dense_mask)
        output.append((packed, targets))
    return output
