"""Zero-shot Decision 2.0 (vllm-sr) System One models on the Stage 1 dev questions.

Each model answers the same 56-option span questions as `dev_eval`, through its own
`system_one(state=..., questions=...)` API: one call per document, one Choice question per span.
Answers with an error (e.g. `max_length_exceeded`) count as wrong (see `failed_distribution`) and are
also counted separately, so every model is scored on every row, as in `dev_eval`.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

CONTEXT_CHARS = 60
MAX_QUESTIONS_PER_CALL = 64


def instruction(row) -> str:
    start, end = row.source_span
    text = row.state
    left = text[max(0, start - CONTEXT_CHARS):start]
    right = text[end:end + CONTEXT_CHARS]
    return (f'What kind of information is the span "{text[start:end]}" in the document, '
            f'where it appears as: "...{left}[[{text[start:end]}]]{right}..."?')


def criteria(row) -> dict[str, str]:
    return {option.name: option.description or option.name for option in row.question.options}


def _probabilities(answer: dict, row) -> list[float] | None:
    if not isinstance(answer, dict) or answer.get("error"):
        return None
    probs = answer.get("probabilities")
    names = [option.name for option in row.question.options]
    if not isinstance(probs, dict) or set(probs) != set(names):
        return None
    return [float(probs[name]) for name in names]


def failed_distribution(row) -> list[float]:
    """Uniform over the wrong options: a failed answer is scored as incorrect, never as a free guess."""
    p = [1.0] * len(row.question.options); p[row.target] = 0.0
    return [x / sum(p) for x in p]


def score_rows(model, rows: Sequence) -> tuple[list[list[float] | None], dict]:
    """Probabilities per row in row order (None where the model returned an error)."""
    by_doc = defaultdict(list)
    for index, row in enumerate(rows):
        by_doc[(row.doc_id, row.state)].append(index)
    out: list[list[float] | None] = [None] * len(rows)
    errors = defaultdict(int)
    for (_doc_id, state), indices in by_doc.items():
        for start in range(0, len(indices), MAX_QUESTIONS_PER_CALL):
            chunk = indices[start:start + MAX_QUESTIONS_PER_CALL]
            questions = {f"q{i}": {"type": "choice", "instructions": instruction(rows[i]),
                                   "criteria": criteria(rows[i])} for i in chunk}
            answers = model.system_one(state=state, questions=questions)["answers"]
            for i in chunk:
                answer = answers.get(f"q{i}", {})
                out[i] = _probabilities(answer, rows[i])
                if out[i] is None:
                    errors[str(answer.get("error", "invalid_answer")) if isinstance(answer, dict)
                           else "invalid_answer"] += 1
    return out, dict(errors)


def load(model_id: str, revision: str):
    from transformers import AutoModel
    return AutoModel.from_pretrained(model_id, revision=revision, trust_remote_code=True)
