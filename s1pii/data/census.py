"""Label and offset census over raw records, before any mapping decision (notebook 01, step 1).

    python -m s1pii.data.census nemotron gretel spy_medical pii_trace tab_direct

Per dataset it reports every raw label with its count and current mapping (or UNMAPPED),
the surface-match rate, out-of-range spans, spans with whitespace at an edge or cutting a
word, overlapping spans, doc and cluster counts and text-length percentiles. Errors are
caught per dataset so one failure does not hide the others.
"""
from __future__ import annotations

import json
import sys
from collections import Counter

import numpy as np

from .. import taxonomy as tx
from . import loaders as L
from .manifest import require_allowed


def census_raws(raws, mapper) -> dict:
    labels, issues = Counter(), Counter()
    lengths, clusters, n = [], set(), 0
    for rd in raws:
        n += 1
        lengths.append(len(rd.text))
        clusters.add(rd.cluster_id)
        spans = sorted(rd.spans, key=lambda s: (int(s["start"]), int(s["end"])))
        last_end = -1
        for s in spans:
            a, b = int(s["start"]), int(s["end"])
            labels[str(s["label"])] += 1
            if not (0 <= a < b <= len(rd.text)):
                issues["out_of_range"] += 1
                continue
            seg = rd.text[a:b]
            if s.get("surface") is not None:
                issues["surface_checked"] += 1
                issues["surface_match"] += int(seg == s["surface"])
            if seg != seg.strip():
                issues["whitespace_edge"] += 1
            if (a > 0 and rd.text[a - 1].isalnum() and seg[:1].isalnum()) or \
               (b < len(rd.text) and rd.text[b].isalnum() and seg[-1:].isalnum()):
                issues["cuts_word"] += 1
            if a < last_end:
                issues["overlapping"] += 1
            last_end = max(last_end, b)
    mapping = {}
    for lab in labels:
        try:
            mapping[lab] = mapper(lab)
        except tx.UnmappedLabelError:
            mapping[lab] = "UNMAPPED"
    total_spans = sum(labels.values())
    return {
        "docs": n, "clusters": len(clusters), "spans": total_spans,
        "text_len_p50_p90_p99_max": [int(x) for x in np.percentile(lengths, [50, 90, 99, 100])] if lengths else [],
        "labels": {k: {"count": c, "maps_to": mapping[k]} for k, c in labels.most_common()},
        "unmapped": sorted(k for k, v in mapping.items() if v == "UNMAPPED"),
        "issues": dict(issues),
        "surface_match_rate": (issues["surface_match"] / issues["surface_checked"]) if issues["surface_checked"] else None,
    }


def census(name: str, split: str | None = None) -> dict:
    try:
        require_allowed(name, "eval")
        recipe, default = L.RECIPES[name]
        raws, mapper, prov = recipe(split or default)
        return {"dataset": name, "split": split or default, "status": "ok", **prov, **census_raws(raws, mapper)}
    except Exception as e:  # report and continue
        return {"dataset": name, "status": "error", "error": f"{type(e).__name__}: {e}"[:500]}


if __name__ == "__main__":
    for n in sys.argv[1:] or list(L.RECIPES):
        print(json.dumps(census(n), indent=2, default=str))
