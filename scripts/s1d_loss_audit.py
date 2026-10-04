#!/usr/bin/env python3
"""CPU-only audit of S1-D Stage 1 loss lines.

The seed-matched reference is the smallest shared-layout run for that seed unless
``--healthy SEED=RUN`` overrides it. A departure is the first of N consecutive
reported points whose rolling-median loss exceeds both ``ratio * reference`` and
``reference + margin``. The rule and all parameters are emitted with the result.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

LOSS = re.compile(r"(?P<run>Qwen/Qwen3-(?P<size>[^:]+)-s(?P<seed>\d+)-(?P<layout>\w+)):\s+step\s+"
                  r"(?P<step>\d+)\s+loss\s+(?P<loss>[0-9.eE+-]+)")


def parse(paths):
    series = defaultdict(dict)
    metadata = {}
    for path in paths:
        for line in Path(path).read_text(errors="replace").splitlines():
            match = LOSS.search(line)
            if not match:
                continue
            run = match["run"]
            series[run][int(match["step"])] = float(match["loss"])
            metadata[run] = {"seed": int(match["seed"]), "size": match["size"],
                             "layout": match["layout"]}
    return {run: sorted(values.items()) for run, values in series.items()}, metadata


def smooth(values, window):
    return [(step, statistics.median([loss for _s, loss in values[max(0, i-window+1):i+1]]))
            for i, (step, _loss) in enumerate(values)]


def departure(values, reference, *, window, ratio, margin, consecutive):
    candidate = dict(smooth(values, window)); healthy = dict(smooth(reference, window))
    common = sorted(set(candidate) & set(healthy)); streak = []
    for step in common:
        if candidate[step] > healthy[step] * ratio and candidate[step] > healthy[step] + margin:
            streak.append(step)
            if len(streak) >= consecutive:
                return streak[-consecutive]
        else:
            streak.clear()
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="*", type=Path)
    ap.add_argument("--root", type=Path, default=Path("/content/drive/MyDrive/s1pii/s1d"))
    ap.add_argument("--steps", default="1,25,50,100,200,400,800,1200")
    ap.add_argument("--healthy", action="append", default=[], metavar="SEED=RUN")
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--ratio", type=float, default=3.0)
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--consecutive", type=int, default=3)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    paths = args.logs or sorted((args.root / "logs").glob("stage1*.log"))
    if not paths:
        ap.error("no Stage 1 logs found")
    series, metadata = parse(paths)
    if not series:
        ap.error("no Stage 1 loss lines found")
    override = {int(item.split("=", 1)[0]): item.split("=", 1)[1] for item in args.healthy}
    references = {}
    for seed in sorted({value["seed"] for value in metadata.values()}):
        candidates = [run for run, value in metadata.items()
                      if value["seed"] == seed and value["layout"] == "shared"]
        references[seed] = override.get(seed, sorted(candidates, key=lambda x: float(
            re.search(r"Qwen3-([0-9.]+)B", x).group(1)))[0])
    fixed = [int(value) for value in args.steps.split(",") if value]
    output = {"method": {"smoothing": f"rolling median, window={args.window}",
                          "departure": f"> {args.ratio}x reference and > reference + {args.margin}, "
                                       f"for {args.consecutive} consecutive reported points",
                          "references": references}, "runs": {}}
    for run in sorted(series):
        values = dict(series[run]); seed = metadata[run]["seed"]; reference = references[seed]
        output["runs"][run] = {**metadata[run],
            "fixed_steps": {str(step): values.get(step) for step in fixed},
            "last_step": max(values),
            "departure_step": None if run == reference else departure(
                series[run], series[reference], window=args.window, ratio=args.ratio,
                margin=args.margin, consecutive=args.consecutive)}
    text = json.dumps(output, indent=2, sort_keys=True)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True); args.json.write_text(text + "\n")


if __name__ == "__main__":
    main()
