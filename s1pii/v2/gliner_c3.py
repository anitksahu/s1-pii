"""GLiNER predictions for C3 under the C3 label set (runs in a baseline venv).

    envs/gliner2/bin/python -m s1pii.v2.gliner_c3 --system gliner25_base_zeroshot \
        --docs $S1PII_DATA/splits/nemotron-test.jsonl --out $S1PII_RESULTS/predictions_v2

Output spans keep the queried raw label in ``label_raw`` (C3 scores typed matches on it).
GLiNER2 models receive name -> description; ``--names-only`` passes names (secondary).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..adapters.base import Adapter, load_config
from ..adapters.run import run
from .labels import c3_label_set


def adapter(system: str, descriptions: bool = True) -> Adapter:
    from ..adapters.backends import Gliner2Backend, GlinerBackend
    cfg = load_config()["systems"][system]
    L = c3_label_set(descriptions)
    if cfg["backend"] == "gliner2":
        be = Gliner2Backend(cfg["model_id"], cfg.get("revision"),
                            {l.name: (l.description if descriptions else l.name.replace("_", " ")) for l in L.labels})
    else:
        be = GlinerBackend(cfg["model_id"], cfg.get("revision"))
    ad = Adapter(system, be)
    ad.labels = {l.name: l.resolved_canonical() for l in L.labels}
    ad.revision = getattr(be, "revision", "unknown")
    ad.versions = {**getattr(be, "versions", {}), "c3_labels": L.hash()}
    return ad


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
