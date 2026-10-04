"""Batched S1-D inference, seen-label calibration, and confidence abstention."""
from __future__ import annotations

from collections import defaultdict
import copy
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


@torch.no_grad()
def prompted_option_distribution(model, tokenizer, prompt: str, options: Sequence[str], *, device=None) -> torch.Tensor:
    """Length-normalised option likelihoods with one prefix pass and one batched continuation pass."""
    if len(options) < 2:
        raise ValueError("at least two prompted options are required")
    device = device or next(model.parameters()).device
    prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"].to(device)
    prefix = model(input_ids=prompt_ids, use_cache=True, return_dict=True)
    option_ids = [tokenizer(option, add_special_tokens=False, return_tensors="pt")["input_ids"][0].to(device)
                  for option in options]
    if any(ids.numel() == 0 for ids in option_ids):
        raise ValueError("prompted options must tokenize to at least one token")
    first_logits = prefix.logits[:, -1].float().log_softmax(-1)[0]
    totals = [first_logits[ids[0]] for ids in option_ids]
    long = [index for index, ids in enumerate(option_ids) if ids.numel() > 1]
    if long and hasattr(prefix.past_key_values, "batch_repeat_interleave"):
        # Repeating a long-document KV cache for all 56 options can OOM the 4B model.  Eight
        # continuations keeps peak cache memory bounded while retaining GPU parallelism.
        for first in range(0, len(long), 8):
            chunk = long[first:first + 8]
            past = copy.deepcopy(prefix.past_key_values)
            past.batch_repeat_interleave(len(chunk))
            width = max(option_ids[index].numel() - 1 for index in chunk)
            pad = getattr(tokenizer, "pad_token_id", None)
            continuation_ids = torch.full((len(chunk), width), 0 if pad is None else int(pad),
                                          dtype=torch.long, device=device)
            for batch_index, option_index in enumerate(chunk):
                ids = option_ids[option_index]
                continuation_ids[batch_index, :ids.numel() - 1] = ids[:-1]
            continuation = model(input_ids=continuation_ids, past_key_values=past,
                                 use_cache=False, return_dict=True).logits.float().log_softmax(-1)
            for batch_index, option_index in enumerate(chunk):
                ids = option_ids[option_index]
                positions = torch.arange(ids.numel() - 1, device=device)
                totals[option_index] = totals[option_index] + continuation[batch_index, positions, ids[1:]].sum()
    else:
        # Compatibility fallback for legacy tuple caches and the tiny CPU test doubles.
        for option_index in long:
            ids = option_ids[option_index]
            continuation = model(input_ids=ids[:-1].unsqueeze(0),
                                 past_key_values=copy.deepcopy(prefix.past_key_values),
                                 use_cache=False, return_dict=True).logits[0].float().log_softmax(-1)
            positions = torch.arange(ids.numel() - 1, device=device)
            totals[option_index] = totals[option_index] + continuation[positions, ids[1:]].sum()
    scores = []
    for total, ids in zip(totals, option_ids):
        if ids.numel() == 0:  # guarded above; keeps static analyzers honest
            raise ValueError("prompted options must tokenize to at least one token")
        scores.append(total / ids.numel())
    return torch.stack(scores).softmax(0).cpu()


def expected_calibration_error(probabilities: Sequence[Sequence[float]], targets: Sequence[int],
                               bins: int = 15) -> float:
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(targets, dtype=int)
    if p.ndim != 2 or len(p) != len(y) or not len(y):
        raise ValueError("ECE requires aligned non-empty probability rows and targets")
    confidence, prediction = p.max(1), p.argmax(1)
    total = 0.0
    for lo in np.linspace(0, 1, bins, endpoint=False):
        hi = lo + 1 / bins
        mask = (confidence >= lo) & (confidence <= hi if hi >= 1 else confidence < hi)
        if mask.any():
            total += mask.mean() * abs((prediction[mask] == y[mask]).mean() - confidence[mask].mean())
    return float(total)


def document_bootstrap(values_by_doc: dict[str, Sequence[float]], *, seed: int = 0,
                       samples: int = 2000, side: str = "two-sided") -> dict[str, float]:
    """Cluster bootstrap over documents for scalar per-question values."""
    docs = sorted(values_by_doc)
    if not docs:
        raise ValueError("document bootstrap requires documents")
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(samples):
        chosen = rng.choice(docs, len(docs), replace=True)
        values = [value for doc in chosen for value in values_by_doc[doc]]
        draws.append(float(np.mean(values)))
    q = np.quantile(draws, [0.025, 0.05, 0.95, 0.975])
    result = {"mean": float(np.mean([v for values in values_by_doc.values() for v in values])),
              "lower_95": float(q[0]), "upper_95": float(q[3]),
              "one_sided_lower_95": float(q[1]), "one_sided_upper_95": float(q[2])}
    if side not in {"two-sided", "lower", "upper"}:
        raise ValueError("side must be two-sided, lower, or upper")
    return result
