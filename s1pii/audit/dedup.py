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
    for s in sh:
        m.update(s.encode())
    return m


def audit(train: Sequence[Doc], test: Sequence[Doc], text_threshold: float = 0.8,
          skeleton_threshold: float = 0.9) -> dict:
    from datasketch import MinHashLSH
    lsh_text = MinHashLSH(threshold=text_threshold, num_perm=NUM_PERM)
    lsh_skel = MinHashLSH(threshold=skeleton_threshold, num_perm=NUM_PERM)
    train_skel_hashes = set()
    for d in train:
        lsh_text.insert(d.doc_id, _minhash(shingles(d.text)))
        sk = skeleton(d)
        lsh_skel.insert(d.doc_id, _minhash(shingles(sk)))
        train_skel_hashes.add(hashlib.sha256(sk.encode()).hexdigest())
    flagged: dict[str, dict] = {}
    for d in test:
        reasons = {}
        near = lsh_text.query(_minhash(shingles(d.text)))
        if near:
            reasons["near_duplicate_of"] = near[:5]
        sk = skeleton(d)
        if hashlib.sha256(sk.encode()).hexdigest() in train_skel_hashes:
            reasons["identical_skeleton"] = True
        tmpl = lsh_skel.query(_minhash(shingles(sk)))
        if tmpl:
            reasons["template_of"] = tmpl[:5]
        if reasons:
            flagged[d.doc_id] = reasons
    return {"n_train": len(train), "n_test": len(test), "n_flagged": len(flagged),
            "flagged": flagged}


def drop_flagged(test: Iterable[Doc], report: dict) -> list[Doc]:
    bad = set(report["flagged"])
    return [d for d in test if d.doc_id not in bad]
