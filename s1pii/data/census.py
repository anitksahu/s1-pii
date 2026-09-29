"""Label census: load each dataset and report unmapped raw labels (notebook 01, first step).

    python -m s1pii.data.census nemotron gretel spy_medical
"""
from __future__ import annotations

import sys
from collections import Counter

from .. import taxonomy as tx
from . import loaders


def census(name: str, split: str = "test") -> dict:
    try:
        docs = loaders.load(name, split)
    except tx.UnmappedLabelError as e:
        return {"dataset": name, "status": "unmapped", "labels": e.labels}
    c = Counter((s.label_raw, s.label_canonical) for d in docs for s in d.spans)
    return {"dataset": name, "status": "ok", "docs": len(docs),
            "labels": {f"{r} -> {k}": n for (r, k), n in c.most_common()}}


if __name__ == "__main__":
    for n in sys.argv[1:] or list(loaders.LOADERS):
        print(census(n))
