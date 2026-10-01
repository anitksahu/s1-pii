"""Frozen S1-D label split and semantic training exclusions.

The draw consumes label names only. Descriptions are looked up only after the split has
been fixed, and no dataset is read by this module.
"""
from __future__ import annotations

import hashlib
import json
import random
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

import yaml

from .. import taxonomy
from ..ledger import append
from ..v2 import labels as v2

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "s1d_heldout.yaml"
V21_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "heldout_labels.yaml"
DEFAULT_SEED = 20261001
NOT_PII = "not personal information"
OOD_DATASETS = ("spy_legal", "spy_medical", "tab_direct", "pii_trace")
SYNTHETIC_RAW_LABELS = {
    "person", "address", "email", "phone", "url", "date", "account_number", "secret", "other_pii",
}
PARENT_LABELS = {
    "name", "person", "address", "date", "account_number", "secret", "other_pii", "unique_id",
}


def eligible_names() -> list[str]:
    old = yaml.safe_load(V21_CONFIG.read_text())
    excluded = {str(x).lower() for key in ("labels", "nodes", "synonyms_all_sources")
                for x in old.get(key, [])}
    return sorted(l.name for l in v2.c3_label_set(False).labels if l.name.lower() not in excluded)


def draw(seed: int = DEFAULT_SEED, *, dev: int = 5, test: int = 10) -> dict:
    pool = eligible_names()
    rng = random.Random(seed)
    chosen = rng.sample(pool, dev + test)
    dev_names, test_names = chosen[:dev], chosen[dev:]
    held = dev_names + test_names
    nodes = sorted({v2.node_of(n) for n in held})
    synonyms = sorted(n for n in v2.NATIVE if v2.node_of(n) in nodes)
    names = sorted({*held, *nodes, *synonyms})
    paraphrases = sorted({p for n in synonyms for p in v2.paraphrases(n)})
    canonical = sorted({v2.canonical_of(n) for n in held})
    frozen = {n: {"description": v2.NATIVE[n][1], "paraphrases": list(v2.paraphrases(n))} for n in held}
    ood_names = sorted({lab.name.lower() for dataset in OOD_DATASETS
                        for lab in v2.benchmark_label_set(dataset).labels})
    base = {
        "version": 1, "seed": seed,
        "rule": "sample 5 dev then 10 test names without replacement from sorted eligible names",
        "pool_sha256": hashlib.sha256("\n".join(pool).encode()).hexdigest(),
        "pool_size": len(pool), "dev_labels": dev_names, "test_labels": test_names,
        "replication_labels": list(yaml.safe_load(V21_CONFIG.read_text())["labels"]),
        "frozen": frozen,
        "exclusions": {"names": names, "paraphrases": paraphrases, "nodes": nodes,
                       "synonyms_all_sources": synonyms, "canonical_mappings": canonical},
        "never_train_native_sets": list(OOD_DATASETS),
        "never_train_native_names": ood_names,
        "min_spans": 300, "fallback_min_spans": 150,
    }
    base["draw_sha256"] = hashlib.sha256(json.dumps(base, sort_keys=True).encode()).hexdigest()
    return base


def freeze(path: Path = CONFIG, ledger_path: Path | None = None, seed: int = DEFAULT_SEED) -> dict:
    """Freeze the name-only draw and record it before a caller is allowed to read data."""
    if path.exists():
        return yaml.safe_load(path.read_text())
    result = draw(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(result, sort_keys=False))
    append({"unit": "s1d_label_draw", "seed": seed, "rule": result["rule"],
            "config_sha": result["draw_sha256"]}, path=ledger_path)
    return result


def load(path: Path = CONFIG) -> dict:
    result = yaml.safe_load(path.read_text())
    expected = result.pop("draw_sha256")
    actual = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    result["draw_sha256"] = expected
    if actual != expected:
        raise ValueError("s1d held-out config hash mismatch")
    return result


def split_labels(split: str, config: dict | None = None) -> v2.LabelSet:
    """Return a frozen dev/test/replication set without re-running the draw."""
    c = config or load()
    key = {"dev": "dev_labels", "test": "test_labels", "replication": "replication_labels"}.get(split)
    if key is None:
        raise ValueError("split must be dev, test, or replication")
    rows = []
    for name in c[key]:
        description = c.get("frozen", {}).get(name, {}).get("description", v2.NATIVE[name][1])
        rows.append(v2.Label(name, description, True, v2.canonical_of(name), v2.node_of(name)))
    return v2.LabelSet(f"s1d_{split}", rows)


def replication_set(config: dict | None = None) -> v2.LabelSet:
    return split_labels("replication", config)


def semantic_exclusions(config: dict | None = None) -> dict[str, set[str]]:
    c = config or load()
    out = {key: set(value) for key, value in c["exclusions"].items()}
    old = yaml.safe_load(V21_CONFIG.read_text())
    for key in ("labels", "nodes", "synonyms_all_sources"):
        out[f"replication_{key}"] = {str(x).lower() for x in old.get(key, ())}
    return out


def actual_training_raw_labels() -> set[str]:
    """Raw labels present in the three declared S1-D training sources."""
    return ({str(x).lower() for x in taxonomy.NEMOTRON}
            | {str(x).lower() for x in taxonomy.GRETEL}
            | SYNTHETIC_RAW_LABELS)


def ood_only_names(config: dict | None = None) -> set[str]:
    """Evaluation-native names that are absent from every actual training source."""
    c = config or load()
    return {str(x).lower() for x in c.get("never_train_native_names", ())} - actual_training_raw_labels()


def excluded(raw: str, config: dict | None = None) -> bool:
    c = config or load()
    exclusions = semantic_exclusions(c)
    x = raw.strip().lower().replace("_", " ")
    texts = [*c["exclusions"]["names"], *c["exclusions"]["paraphrases"]]
    return (any(x == str(t).lower().replace("_", " ") for t in texts)
            or v2.node_of(raw) in c["exclusions"]["nodes"]
            or raw.lower() in exclusions["replication_labels"]
            or v2.node_of(raw) in exclusions["replication_nodes"]
            or raw.lower() in exclusions["replication_synonyms_all_sources"])


def training_vocabulary(config: dict | None = None, *, remove_parents: bool = False) -> list[str]:
    """Fine option labels that may be trained; canonical type strings are never options."""
    c = config or load()
    raw = actual_training_raw_labels()
    names = sorted(n for n in raw if n in v2.NATIVE and not excluded(n, c) and n not in ood_only_names(c))
    if remove_parents:
        names = [n for n in names if n not in PARENT_LABELS]
    return names


def nearest_trained_neighbours(config: dict | None = None, *, remove_parents: bool = False) -> dict[str, dict]:
    """Audit nearest trainable label by separately comparing names and descriptions."""
    c = config or load(); trained = training_vocabulary(c, remove_parents=remove_parents)
    result = {}
    for test in c["test_labels"]:
        test_name = test.replace("_", " "); test_desc = v2.NATIVE[test][1]
        rows = []
        for candidate in trained:
            name_sim = SequenceMatcher(None, test_name, candidate.replace("_", " ")).ratio()
            desc_sim = SequenceMatcher(None, test_desc, v2.NATIVE[candidate][1]).ratio()
            rows.append(((name_sim + desc_sim) / 2, candidate, name_sim, desc_sim))
        score, neighbour, name_sim, desc_sim = max(rows)
        result[test] = {"label": neighbour, "name_similarity": round(name_sim, 6),
                        "description_similarity": round(desc_sim, 6), "mean_similarity": round(score, 6)}
    return result


def record_nearest_trained_neighbours(ledger_path: Path, config: dict | None = None,
                                      *, remove_parents: bool = False) -> dict[str, dict]:
    c = config or load(); rows = nearest_trained_neighbours(c, remove_parents=remove_parents)
    append({"unit": "s1d_label_neighbours", "config_sha": c["draw_sha256"],
            "parent_labels_removed": remove_parents, "test_labels": rows}, path=ledger_path)
    return rows


def census(spans: Iterable[object], *, config: dict | None = None, ledger_path: Path | None = None) -> dict:
    """Count test labels after the draw; fallback labels are explicitly ledgered."""
    c = config or load()
    counts = {n: 0 for n in c["test_labels"]}
    for span in spans:
        raw = getattr(span, "label_raw", span.get("label_raw", "") if isinstance(span, dict) else "").lower()
        if raw in counts:
            counts[raw] += 1
    fallback = sorted(n for n, count in counts.items() if c["fallback_min_spans"] <= count < c["min_spans"])
    failed = sorted(n for n, count in counts.items() if count < c["fallback_min_spans"])
    result = {"counts": counts, "fallback": fallback, "failed": failed, "passed": not failed}
    append({"unit": "s1d_census", "seed": c["seed"], "config_sha": c["draw_sha256"], **result},
           path=ledger_path)
    if failed:
        raise ValueError(f"test labels below fallback minimum: {failed}")
    return result
