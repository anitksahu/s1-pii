"""Batched S1-D inference, seen-label calibration, and confidence abstention."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch

from .schema import DecisionResponse, Question, QuestionType, answer, redaction_score


@dataclass(frozen=True)
class Calibration:
    temperatures: dict[str, float]
    confidence_threshold: float


def _nll(log_t, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.cross_entropy(logits / log_t.exp(), targets)


def fit_temperatures(logits: Sequence[torch.Tensor], targets: Sequence[int], types: Sequence[str]) -> dict[str, float]:
    """Fit one positive scalar per question type; caller supplies seen-label rows only."""
    grouped = defaultdict(list)
    for z, y, kind in zip(logits, targets, types): grouped[QuestionType(kind).value].append((z.detach().float(), int(y)))
    result = {}
    for kind in QuestionType:
        rows = grouped.get(kind.value, [])
        if not rows:
            result[kind.value] = 1.0; continue
        z = torch.stack([r[0] for r in rows]); y = torch.tensor([r[1] for r in rows])
        log_t = torch.zeros((), requires_grad=True)
        opt = torch.optim.LBFGS([log_t], max_iter=50, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad(); loss = _nll(log_t, z, y); loss.backward(); return loss
        opt.step(closure)
        result[kind.value] = float(log_t.detach().exp().clamp(0.05, 20.0))
    return result


def fit_confidence_threshold(confidences: Sequence[float], correct: Sequence[bool], coverage: float = 0.9) -> float:
    """Fix c on seen calibration labels at the requested coverage."""
    if not 0 < coverage <= 1 or not confidences:
        raise ValueError("non-empty calibration confidences and coverage in (0,1] required")
    return float(np.quantile(np.asarray(confidences), max(0.0, 1.0 - coverage), method="lower"))


@torch.no_grad()
def predict(model, packed_windows, questions: Sequence[Sequence[Question]], calibration: Calibration | None = None,
            batch_size: int = 8) -> list[DecisionResponse]:
    """Run in bounded batches. BlockMasks differ per window, so forwards remain independent."""
    model.eval(); out = []
    for a in range(0, len(packed_windows), batch_size):
        for packed, qs in zip(packed_windows[a:a + batch_size], questions[a:a + batch_size]):
            logits = model(packed).logits
            answers = []
            for row, q in zip(logits, qs):
                t = calibration.temperatures.get(q.type.value, 1.0) if calibration else 1.0
                p = (row / t).softmax(-1).cpu().tolist()
                answers.append(answer(q, p, threshold=calibration.confidence_threshold if calibration else None))
            out.append(DecisionResponse(tuple(answers)))
    return out


def span_redaction_scores(response: DecisionResponse) -> list[float | None]:
    return [None if a.abstained else redaction_score(a) for a in response.answers]
