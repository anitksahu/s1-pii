"""Zero-shot Decision 2.0 (vllm-sr) System One models on the Stage 1 dev questions.

Each model answers the same 56-option span questions as `dev_eval`, through its own
`system_one(state=..., questions=...)` API: one call per 512-token state window (the windows the trained
S1-D models see), one Choice question per span in that window.
Answers with an error (e.g. `max_length_exceeded`) count as wrong (see `failed_distribution`) and are
also counted separately, so every model is scored on every row, as in `dev_eval`.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Sequence

CONTEXT_CHARS = 60
MAX_QUESTIONS_PER_CALL = 64


def instruction(row, state: str | None = None) -> str:
    """Name the span and quote up to CONTEXT_CHARS around it, never beyond the state window."""
    start, end = row.source_span
    text = row.state
    lo, hi = 0, len(text)
    if state is not None:
        w0 = text.rfind(state, 0, start + len(state))  # last window occurrence starting at or before the span
        if w0 >= 0 and end <= w0 + len(state):
            lo, hi = w0, w0 + len(state)
    left = text[max(lo, start - CONTEXT_CHARS):start]
    right = text[end:min(hi, end + CONTEXT_CHARS)]
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


def score_groups(model, rows: Sequence, groups: Sequence[tuple[str, Sequence[int]]], *,
                 on_group=None, start: int = 0, out=None, errors=None) -> tuple[list[list[float] | None], dict]:
    """Answer each group of questions with one `system_one` call on that group's state window.

    `groups` holds (state, row indices); the dev_eval windowing supplies them so every model sees
    the same 512-token state windows. `on_group(done, probabilities, errors)` lets the caller
    checkpoint; `start`, `out` and `errors` resume from such a checkpoint.
    """
    out = list(out) if out is not None else [None] * len(rows)
    errors = defaultdict(int, errors or {})
    for done, (state, indices) in enumerate(groups[start:], start + 1):
        for start in range(0, len(indices), MAX_QUESTIONS_PER_CALL):
            chunk = list(indices[start:start + MAX_QUESTIONS_PER_CALL])
            questions = {f"q{i}": {"type": "choice", "instructions": instruction(rows[i], state),
                                   "criteria": criteria(rows[i])} for i in chunk}
            answers = model.system_one(state=state, questions=questions)["answers"]
            for i in chunk:
                answer = answers.get(f"q{i}", {})
                out[i] = _probabilities(answer, rows[i])
                if out[i] is None:
                    errors[str(answer.get("error", "invalid_answer")) if isinstance(answer, dict)
                           else "invalid_answer"] += 1
        if on_group is not None:
            on_group(done, out, dict(errors))
    return out, dict(errors)


def score_rows(model, rows: Sequence) -> tuple[list[list[float] | None], dict]:
    """One call per full document state (used where no windowing is supplied)."""
    by_doc = defaultdict(list)
    for index, row in enumerate(rows):
        by_doc[(row.doc_id, row.state)].append(index)
    return score_groups(model, rows, [(state, indices) for (_doc, state), indices in by_doc.items()])


def load(model_id: str, revision: str):
    from transformers import AutoModel
    return AutoModel.from_pretrained(model_id, revision=revision, trust_remote_code=True)
