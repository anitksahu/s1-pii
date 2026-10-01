"""Public Jev/Kev-shaped request and response types for S1-PII-D."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable


class QuestionType(str, Enum):
    CHOICE = "choice"
    SCORE = "score"
    NOUL = "noul"


@dataclass(frozen=True)
class Option:
    name: str
    description: str = ""
    value: float | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("option names cannot be empty")

    def text(self) -> str:
        return f"{self.name}: {self.description}" if self.description else self.name


def _option(value: Option | str | dict) -> Option:
    if isinstance(value, Option):
        return value
    if isinstance(value, str):
        return Option(value)
    return Option(**value)


@dataclass(frozen=True)
class Question:
    type: QuestionType | str
    instructions: str
    criteria: str = ""
    options: tuple[Option, ...] = ()
    id: str = ""
    span: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        kind = QuestionType(self.type)
        object.__setattr__(self, "type", kind)
        object.__setattr__(self, "options", tuple(_option(o) for o in self.options))
        if not self.instructions.strip():
            raise ValueError("question instructions cannot be empty")
        if kind is QuestionType.NOUL:
            if self.options:
                raise ValueError("noul options are fixed; omit options")
            object.__setattr__(self, "options", (Option("no", value=0.0), Option("yes", value=1.0)))
        elif kind is QuestionType.SCORE:
            if not 2 <= len(self.options) <= 10:
                raise ValueError("score questions require 2-10 ordered levels")
            if any(o.value is None for o in self.options):
                object.__setattr__(self, "options", tuple(
                    Option(o.name, o.description, float(i)) for i, o in enumerate(self.options)
                ))
        elif not 2 <= len(self.options) <= 255:
            raise ValueError("choice questions require 2-255 options")
        if self.span is not None and not (0 <= self.span[0] < self.span[1]):
            raise ValueError("span must be a non-empty [start, end) range")


@dataclass(frozen=True)
class DecisionRequest:
    state: str
    questions: tuple[Question, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.state, str):
            raise TypeError("state must be text")
        object.__setattr__(self, "questions", tuple(
            q if isinstance(q, Question) else Question(**q) for q in self.questions
        ))
        if not self.questions:
            raise ValueError("at least one question is required")


@dataclass(frozen=True)
class Answer:
    question_id: str
    probabilities: dict[str, float]
    confidence: float
    argmax: str
    score: float | None = None
    abstained: bool = False


@dataclass(frozen=True)
class DecisionResponse:
    answers: tuple[Answer, ...] = field(default_factory=tuple)


def answer(question: Question, probabilities: Iterable[float], *, threshold: float | None = None) -> Answer:
    """Validate and turn one probability vector into the stable response schema."""
    ps = [float(p) for p in probabilities]
    if len(ps) != len(question.options) or any(not math.isfinite(p) or p < 0 for p in ps):
        raise ValueError("invalid probability vector")
    z = sum(ps)
    if z <= 0:
        raise ValueError("probabilities must have positive mass")
    ps = [p / z for p in ps]
    k = len(ps)
    entropy = -sum(p * math.log(p) for p in ps if p)
    confidence = 1.0 - entropy / math.log(k)
    best = max(range(k), key=ps.__getitem__)
    mean = None
    if question.type is QuestionType.SCORE:
        mean = sum(p * float(o.value) for p, o in zip(ps, question.options))
    return Answer(question.id, {o.name: p for o, p in zip(question.options, ps)}, confidence,
                  question.options[best].name, mean, threshold is not None and confidence < threshold)


def redaction_score(result: Answer, none_name: str = "not personal information") -> float:
    if none_name not in result.probabilities:
        raise KeyError(f"missing required option {none_name!r}")
    return 1.0 - result.probabilities[none_name]
