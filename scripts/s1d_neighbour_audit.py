#!/usr/bin/env python3
"""CPU-only semantic-exclusion and remaining-neighbour audit for S1-D held-outs."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from s1pii.s1d import labels as H
from s1pii.v2 import labels as v2


def build_audit() -> dict:
    config = H.load()
    heldout = list(config["dev_labels"]) + list(config["test_labels"])
    exclusions = H.semantic_exclusions(config)
    candidate_training = sorted(H.actual_training_raw_labels() & set(v2.NATIVE))
    trained = set(H.training_vocabulary(config))
    nearest_config = copy.deepcopy(config)
    nearest_config["test_labels"] = heldout
    nearest = H.nearest_trained_neighbours(nearest_config)
    encoder = nearest.pop("_encoder")
    rows = {}
    for name in heldout:
        node = v2.node_of(name)
        semantic_candidates = [raw for raw in candidate_training if v2.node_of(raw) == node]
        excluded = [raw for raw in semantic_candidates
                    if H.excluded(raw, config, exclusions=exclusions)]
        remaining = [raw for raw in semantic_candidates if raw in trained]
        rows[name] = {
            "split": "dev" if name in config["dev_labels"] else "test",
            "semantic_node": node,
            "excluded_semantic_neighbours": excluded,
            "same_node_labels_remaining_in_training": remaining,
            "closest_trained_labels_remaining": nearest[name],
        }
    return {
        "version": 1,
        "definition": ("Excluded semantic neighbours are labels present in declared training sources "
                       "that share the held-out label's semantic node and are removed by "
                       "labels.semantic_exclusions. Closest remaining labels are produced by "
                       "labels.nearest_trained_neighbours."),
        "encoder": encoder,
        "labels": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("docs/results_s1d/s1d_neighbour_audit.json"))
    args = parser.parse_args()
    output = build_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))
    print("wrote", args.output)


if __name__ == "__main__":
    main()
