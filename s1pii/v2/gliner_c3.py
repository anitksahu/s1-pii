"""GLiNER predictions for C3 under the C3 label set (runs in a baseline venv).

    envs/gliner2/bin/python -m s1pii.v2.gliner_c3 --system gliner25_base_zeroshot \
        --docs $S1PII_DATA/splits/nemotron-test.jsonl --out $S1PII_RESULTS/predictions_v2

Output spans keep the queried raw label in ``label_raw`` (C3 scores typed matches on it).
GLiNER2 models receive name -> description; ``--names-only`` passes names (secondary).
The 55-label C3 prompt does not fit the model context with a text window, so labels are
queried in fixed chunks of ``CHUNK`` (sorted order) and the spans are merged; the chunking
is part of the adapter config (and the cache key).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..adapters.base import Adapter, load_config
from ..adapters.run import run
from .labels import c3_label_set


CHUNK = 12


class Chunked:
    """Runs one Adapter per label chunk and merges the spans (``predict_docs`` interface)."""

    def __init__(self, adapters: list[Adapter]):
        self.adapters = adapters
        self.revision = adapters[0].revision
        self.versions = {**adapters[0].versions, "label_chunk": CHUNK}

    def config(self) -> dict:
        return {"chunks": [a.config() for a in self.adapters]}

    def predict_docs(self, docs, batch_size: int = 16):
        merged, rep = {d.doc_id: [] for d in docs}, None
        for a in self.adapters:
            preds, r = a.predict_docs(docs, batch_size)
            for k, v in preds.items():
                merged[k].extend(v)
            if rep is None:
                rep = r
            else:
                rep.merge(r)
        for k in merged:
            merged[k].sort(key=lambda s: (s.start, s.end, s.label_raw))
        return merged, rep


def adapter(system: str, descriptions: bool = True):
    from ..adapters.backends import Gliner2Backend, GlinerBackend
    cfg = load_config()["systems"][system]
    L = c3_label_set(descriptions)
    if cfg["backend"] == "gliner2":
        be = Gliner2Backend(cfg["model_id"], cfg.get("revision"),
                            {l.name: (l.description if descriptions else l.name.replace("_", " ")) for l in L.labels})
    else:
        be = GlinerBackend(cfg["model_id"], cfg.get("revision"))
    out = []
    for i in range(0, len(L.labels), CHUNK):
        ad = Adapter(system, be)
        ad.labels = {l.name: l.resolved_canonical() for l in L.labels[i:i + CHUNK]}
        ad.revision = getattr(be, "revision", "unknown")
        ad.versions = {**getattr(be, "versions", {}), "c3_labels": L.hash()}
        out.append(ad)
    return Chunked(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True)
    ap.add_argument("--docs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--names-only", action="store_true")
    ap.add_argument("--batch-size", type=int, default=16)
    a = ap.parse_args(argv)
    name = f"{a.system}_c3" + ("_names" if a.names_only else "")
    ad = adapter(a.system, not a.names_only)
    print(json.dumps({"predictions": str(run(name, a.docs, a.out, predictor=ad, batch_size=a.batch_size))}))


if __name__ == "__main__":
    main(sys.argv[1:])
