"""v2 decisions from a feature store, a typing head and a label set L given at inference.

Headline score (the one quantity used for C0', C4 and the sweeps):

    P(sensitive under L) = P_b(span) * sum_{k in S(L)} P(k | span, L)

with S(L) the labels flagged sensitive and the softmax over L plus NONE. Typed output:
the argmax label k* over L (NONE excluded) with joint probability P_b * P(k* | span, L).
Hierarchical decision (``hier_decide``): back off from k* to the deepest ancestor node whose
summed probability over the labels of L under it is >= tau; below tau at the root the span is
typed "pii" (type abstention).

Everything here runs on CPU from cached features, so any number of label sets can be
scored without the GPU.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..schema import Span, read_jsonl
from ..ledger import write_predictions, dataset_hash, git_sha
from . import labels as LB
from .features import KIND_EXTRA, load_store, load_label_vectors, write_label_vectors
from .head import load_head, TypingHead

INFER_VERSION = "v2-infer-0.1"
FLOOR = 0.01


@torch.no_grad()
def label_probs(head: TypingHead, z: dict, lvecs: np.ndarray, device: str = "cpu", batch: int = 8192) -> np.ndarray:
    """(N, |L| + 1) probabilities for every row of a loaded store; last column NONE."""
    l = head.label_vec(torch.as_tensor(lvecs, device=device))
    out = []
    for s in range(0, len(z["pb"]), batch):
        sv = head.span_vec(torch.as_tensor(z["rep"][s:s + batch], device=device),
                           torch.as_tensor(z["v1"][s:s + batch], device=device),
                           torch.as_tensor(z["pb"][s:s + batch], device=device))
        out.append(torch.softmax(head.logits(sv, l).double(), 1).cpu().numpy())   # float64: no p = 1.0 ties
    return np.concatenate(out) if out else np.zeros((0, len(lvecs) + 1))


def sensitive_score(pb: np.ndarray, probs: np.ndarray, sensitive: list[bool]) -> np.ndarray:
    """P_b * sum over sensitive labels, computed as P_b * (1 - P(NONE) - P(non-sensitive)) in
    float64 so that near-1 scores keep their order instead of rounding to 1.0."""
    m = np.asarray(sensitive, dtype=bool)
    rest = probs[:, -1] + probs[:, :-1][:, ~m].sum(1)
    return pb.astype(np.float64) * np.clip(1.0 - rest, 0.0, 1.0)


def hier_decide(probs_row: np.ndarray, L: LB.LabelSet, tau: float) -> str:
    """Deepest node (a label name or a tree node) with probability mass >= tau."""
    p = probs_row[:-1]
    k = int(p.argmax())
    if p[k] >= tau:
        return L.labels[k].name
    nodes = [l.resolved_node() for l in L.labels]
    for anc in LB.ancestors(nodes[k])[1:]:
        mass = sum(p[i] for i, n in enumerate(nodes) if anc in LB.ancestors(n))
        if mass >= tau:
            return anc
    return "pii"


@dataclass
class Decisions:
    doc_ids: np.ndarray
    offsets: np.ndarray
    start: np.ndarray
    end: np.ndarray
    pb: np.ndarray
    probs: np.ndarray        # (N, |L|+1)
    p_sens: np.ndarray
    top: np.ndarray          # argmax label index over L


def decide(store: Path, head_dir: Path, L: LB.LabelSet, device: str = "cpu", z: dict | None = None,
           head=None) -> Decisions:
    z = z or load_store(store)
    keep = z["kind"] != KIND_EXTRA
    if not keep.all():                      # training stores: drop extras, rebase offsets
        idx = np.nonzero(keep)[0]
        offs = np.searchsorted(idx, z["offsets"])
        z = {**{k: z[k][idx] for k in ("start", "end", "pb", "v1", "rep", "kind", "label")},
             "doc_ids": z["doc_ids"], "offsets": offs}
    if head is None:
        head, _ = load_head(head_dir, device)
    texts = L.texts()
    lv = load_label_vectors(store)
    miss = [t for t in texts if t not in lv]
    if miss:
        raise KeyError(f"label texts not embedded in {store}: {miss[:3]}; run embed_labels first")
    probs = label_probs(head, z, np.stack([lv[t] for t in texts]), device)
    return Decisions(z["doc_ids"], z["offsets"], z["start"], z["end"], z["pb"], probs,
                     sensitive_score(z["pb"], probs, L.sensitive_mask()), probs[:, :-1].argmax(1))


def to_spans(D: Decisions, L: LB.LabelSet, system: str, texts: dict[str, str] | None = None,
             score: str = "sensitive", floor: float = FLOOR) -> dict[str, list[Span]]:
    """score='sensitive' -> P(sensitive under L) (redaction); 'typed' -> P_b * P(k*|span, L)."""
    out: dict[str, list[Span]] = {}
    canon = [l.resolved_canonical() for l in L.labels]
    for di, did in enumerate(D.doc_ids.tolist()):
        a, b = int(D.offsets[di]), int(D.offsets[di + 1])
        spans = []
        for r in range(a, b):
            k = int(D.top[r])
            sc = float(D.p_sens[r]) if score == "sensitive" else float(D.pb[r] * D.probs[r, k])
            if sc < floor:
                continue
            s, e = int(D.start[r]), int(D.end[r])
            spans.append(Span(did, s, e, canon[k], label_raw=L.labels[k].name, score=min(1.0, sc),
                              source=system, surface=texts[did][s:e] if texts else None))
        out[did] = sorted(spans, key=lambda x: (x.start, x.end, x.label_canonical))
    return out


def write(store: Path, head_dir: Path, L: LB.LabelSet, docs_path: Path, out_dir: Path, system: str,
          device: str = "cpu", score: str = "sensitive") -> Path:
    """Prediction file in the v1 ledger format, so bench.score, c0 and c4 read it unchanged."""
    docs = sorted(read_jsonl(docs_path), key=lambda d: d.doc_id)
    smeta = json.loads((store / "meta.json").read_text())
    if smeta["docs_hash"] != dataset_hash(docs):
        raise ValueError(f"store {store} was built on different docs than {docs_path}")
    D = decide(store, head_dir, L, device)
    preds = to_spans(D, L, system, {d.doc_id: d.text for d in docs}, score=score)
    hman = json.loads((head_dir / "head_manifest.json").read_text())
    ident = smeta["identity"]
    config = {"infer_version": INFER_VERSION, "score": score, "labels": L.name, "labels_hash": L.hash(), "floor": FLOOR,
              "features": ident["config"], "feature_version": ident["feature_version"], "validators": ident["config"]["validators"]}
    meta = {"system": system, "revision": f"{ident['v1_weights'][:16]}+{hman['head_sha256'][:16]}",
            "adapter_version": INFER_VERSION, "dataset_hash": smeta["docs_hash"], "docs_path": str(docs_path),
            "n_docs": len(docs), "config": config, "code_sha": git_sha(),
            "versions": {"variant": ident["v1_variant"], "seed": ident["v1_seed"], "head_version": hman["head_version"]},
            "report": smeta["report"]}
    from ..ledger import stable_hash
    path = out_dir / f"{stable_hash([system, meta['revision'], config, smeta['docs_hash']])}.jsonl"
    write_predictions({d.doc_id: preds.get(d.doc_id, []) for d in docs}, path, meta)
    return path


def embed_labels(store: Path, model_dir: Path, sets: list[LB.LabelSet], device: str | None = None) -> None:
    """Embed label texts into a store (GPU seconds; CPU works too)."""
    from .features import Extractor
    ex = Extractor(model_dir, device=device)
    write_label_vectors(store, ex, sorted({t for L in sets for t in L.texts()}))
