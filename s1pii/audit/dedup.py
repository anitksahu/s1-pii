"""Train/test leakage audit: MinHash near-duplicates and template skeletons.

* Near-duplicates: word 5-gram shingles, MinHash (128 perms), LSH at Jaccard >= 0.8.
* Skeletons: replace every gold span (PII or IGNORE) with ``<LABEL>``, digits with ``0``,
  collapse whitespace, lowercase. Identical skeleton hashes, or skeleton MinHash Jaccard
  >= 0.9, flag template reuse by synthetic generators.
Flagged test documents are removed from headline scoring and listed in the audit report.
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable, Sequence

from ..schema import Doc

NUM_PERM = 128


def shingles(text: str, n: int = 5) -> set[str]:
    toks = re.findall(r"\w+", text.lower())
    if len(toks) < n:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def skeleton(doc: Doc) -> str:
    text, out, pos = doc.text, [], 0
    for s in sorted(doc.spans, key=lambda s: (s.start, -s.end)):
        if s.start < pos:
            continue
        out.append(text[pos:s.start])
        out.append(f"<{s.label_canonical}>")
        pos = s.end
    out.append(text[pos:])
    sk = re.sub(r"\d", "0", "".join(out))
    return re.sub(r"\s+", " ", sk).strip().lower()


def skeleton_hash(doc: Doc) -> str:
    return hashlib.sha256(skeleton(doc).encode()).hexdigest()


def _minhash(sh: set[str]):
    from datasketch import MinHash
    m = MinHash(num_perm=NUM_PERM, seed=1)
    if sh:
        m.update_batch([s.encode() for s in sorted(sh)])
    return m


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def audit(train: Sequence[Doc], test: Sequence[Doc], text_threshold: float = 0.8,
          skeleton_threshold: float = 0.9) -> dict:
    from datasketch import MinHashLSH
    lsh_text = MinHashLSH(threshold=text_threshold, num_perm=NUM_PERM)
    lsh_skel = MinHashLSH(threshold=skeleton_threshold, num_perm=NUM_PERM)
    train_skel_hashes, tsh, tss = set(), {}, {}
    for d in train:
        sh = shingles(d.text)
        tsh[d.doc_id] = sh
        lsh_text.insert(d.doc_id, _minhash(sh))
        sk = skeleton(d)
        tss[d.doc_id] = shingles(sk)
        lsh_skel.insert(d.doc_id, _minhash(tss[d.doc_id]))
        train_skel_hashes.add(hashlib.sha256(sk.encode()).hexdigest())
    flagged: dict[str, dict] = {}
    for d in test:
        reasons = {}
        sh = shingles(d.text)
        near = sorted(k for k in lsh_text.query(_minhash(sh)) if jaccard(sh, tsh[k]) >= text_threshold)
        if near:
            reasons["near_duplicate_of"] = near[:5]
        sk = skeleton(d)
        if hashlib.sha256(sk.encode()).hexdigest() in train_skel_hashes:
            reasons["identical_skeleton"] = True
        ss = shingles(sk)
        tmpl = sorted(k for k in lsh_skel.query(_minhash(ss)) if jaccard(ss, tss[k]) >= skeleton_threshold)
        if tmpl:
            reasons["template_of"] = tmpl[:5]
        if reasons:
            flagged[d.doc_id] = {"cluster_id": d.cluster_id, **reasons}
    return {"n_train": len(train), "n_test": len(test), "n_flagged": len(flagged),
            "text_threshold": text_threshold, "skeleton_threshold": skeleton_threshold,
            "flagged": dict(sorted(flagged.items()))}


def drop_flagged(test: Iterable[Doc], report: dict) -> list[Doc]:
    """Drop every test document whose *cluster* contains a flagged document."""
    test = list(test)
    bad = {v.get("cluster_id") or k for k, v in report["flagged"].items()}
    return [d for d in test if (d.cluster_id or d.doc_id) not in bad and d.doc_id not in report["flagged"]]
