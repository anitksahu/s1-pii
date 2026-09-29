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


class AuditIndex:
    """MinHash LSH over one training corpus, built once and queried by many test sets.
    Keeps only per-doc MinHash signatures and exact-Jaccard shingle sets for LSH hits."""

    def __init__(self, train: Sequence[Doc], text_threshold: float = 0.8, skeleton_threshold: float = 0.9):
        from datasketch import MinHashLSH
        self.text_threshold, self.skeleton_threshold = text_threshold, skeleton_threshold
        self.lsh_text = MinHashLSH(threshold=text_threshold, num_perm=NUM_PERM)
        self.lsh_skel = MinHashLSH(threshold=skeleton_threshold, num_perm=NUM_PERM)
        self.skel_hashes, self.tsh, self.tss = set(), {}, {}
        self.n_train = len(train)
        for d in train:
            sh = shingles(d.text)
            self.tsh[d.doc_id] = sh
            self.lsh_text.insert(d.doc_id, _minhash(sh))
            sk = skeleton(d)
            self.tss[d.doc_id] = shingles(sk)
            self.lsh_skel.insert(d.doc_id, _minhash(self.tss[d.doc_id]))
            self.skel_hashes.add(hashlib.sha256(sk.encode()).hexdigest())

    def query(self, test: Sequence[Doc]) -> dict:
        flagged: dict[str, dict] = {}
        for d in test:
            reasons = {}
            sh = shingles(d.text)
            near = sorted(k for k in self.lsh_text.query(_minhash(sh)) if jaccard(sh, self.tsh[k]) >= self.text_threshold)
            if near:
                reasons["near_duplicate_of"] = near[:5]
            sk = skeleton(d)
            if hashlib.sha256(sk.encode()).hexdigest() in self.skel_hashes:
                reasons["identical_skeleton"] = True
            ss = shingles(sk)
            tmpl = sorted(k for k in self.lsh_skel.query(_minhash(ss)) if jaccard(ss, self.tss[k]) >= self.skeleton_threshold)
            if tmpl:
                reasons["template_of"] = tmpl[:5]
            if reasons:
                flagged[d.doc_id] = {"cluster_id": d.cluster_id, **reasons}
        clusters = {d.cluster_id or d.doc_id for d in test}
        bad = {v["cluster_id"] or k for k, v in flagged.items()}
        dropped_docs = sum((d.cluster_id or d.doc_id) in bad for d in test)
        return {"n_train": self.n_train, "n_test": len(test), "n_flagged": len(flagged),
                "text_threshold": self.text_threshold, "skeleton_threshold": self.skeleton_threshold,
                "dropped_doc_frac": dropped_docs / max(1, len(test)),
                "dropped_cluster_frac": len(bad) / max(1, len(clusters)),
                "flagged": dict(sorted(flagged.items()))}


def audit(train: Sequence[Doc], test: Sequence[Doc], text_threshold: float = 0.8,
          skeleton_threshold: float = 0.9) -> dict:
    return AuditIndex(train, text_threshold, skeleton_threshold).query(test)


def drop_flagged(test: Iterable[Doc], report: dict) -> list[Doc]:
    """Drop every test document whose *cluster* contains a flagged document."""
    test = list(test)
    bad = {v.get("cluster_id") or k for k, v in report["flagged"].items()}
    return [d for d in test if (d.cluster_id or d.doc_id) not in bad and d.doc_id not in report["flagged"]]
