"""Stage 1 diagnostic (read only on the real root; scratch writes go to /content/s1d_diag).

Run from the repo root on the Colab GPU after the Stage 1 run is stopped:
    python scripts/s1d_diag.py --runs 0.6B-s2 0.6B-s1

Answers four questions:
  0. What did the 16k training windows actually contain (positives, hard negatives, Noul, Score)?
     Does the regenerated set hash to the run manifest (so the composition is the real one)?
  1. Does the model work on held-out Gretel docs in the exact training format and path?
  2. Do the batched (forward_many) and multi-branch dev_eval paths match the training path?
  3. How much do option text (underscores, NOT_PII description) and the Nemotron domain move accuracy?
"""
from __future__ import annotations

import argparse
import collections
import dataclasses
import json
import os
import random
import time
from pathlib import Path

# Colab terminal processes do not inherit environment variables assigned by a
# notebook Python process.  Cell 1 always syncs the data to this location.
os.environ.setdefault("S1PII_DATA", "/content/s1pii_data")

import torch
import yaml

from s1pii import bench
from s1pii.data import loaders as L
from s1pii.s1d import data as D, labels as H, run as R
from s1pii.s1d.model import S1DModel, apply_lora, prepare_tokenizer
from s1pii.s1d.schema import Option, QuestionType
from s1pii.v2 import labels as v2

ap = argparse.ArgumentParser()
ap.add_argument("--root", type=Path, default=Path("/content/drive/MyDrive/s1pii/s1d"))
ap.add_argument("--scratch", type=Path, default=Path("/content/s1d_diag"))
ap.add_argument("--runs", nargs="+", default=["0.6B-s2", "0.6B-s1"])
ap.add_argument("--n", type=int, default=400)
ap.add_argument("--gretel-docs", type=int, default=400)
args = ap.parse_args()

ROOT, SCR, N = args.root, args.scratch, args.n
nemotron_calib = Path(os.environ["S1PII_DATA"]) / "splits" / "nemotron-calib.jsonl"
if not nemotron_calib.exists():
    raise FileNotFoundError(
        f"Missing {nemotron_calib}. Run Cell 1 of notebooks/06_s1d.ipynb to sync S1PII_DATA, then rerun."
    )
SCR.mkdir(parents=True, exist_ok=True)
link = SCR / "models" / "proposer-no-nemotron"
if not link.exists():
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(ROOT / "models" / "proposer-no-nemotron")
cfg = yaml.safe_load((Path(R.__file__).resolve().parents[1] / "configs" / "s1d.yaml").read_text())
held = H.load()
REAL = {"root": ROOT, "dry": False, "config": cfg}
SCRATCH = {"root": SCR, "dry": False, "config": cfg}
DEVICE = torch.device("cuda")
report = {}
DESC_TO_RAW = {desc: raw for raw, (_node, desc, _p) in v2.NATIVE.items()}


def kind(row):
    if row.question.type is QuestionType.NOUL:
        return "noul"
    if row.question.type is QuestionType.SCORE:
        return "score"
    return "hard_negative" if row.hard_negative else "positive"


def target_raw(row):
    option = row.question.options[row.target]
    return H.NOT_PII if option.name == H.NOT_PII else DESC_TO_RAW.get(option.description, option.name)


# ---------- 0. training composition (CPU) ----------
from s1pii.model.train import TrainConfig as V1Config, training_docs

trained_raws = {}
for seed in sorted({int(r.rsplit("-s", 1)[1]) for r in args.runs}):
    docs, sources = training_docs(V1Config(variant="no-nemotron", seed=seed))
    cands = R._proposer_candidates(ROOT, docs, False, cache_name="training")
    questions = D.generate_questions(
        docs,
        seed=seed,
        variant="no-nemotron",
        min_options=2,
        max_options=64,
        hard_negative_hook=lambda doc: cands.get(doc.doc_id, ()),
        ledger_path=SCR / "ledger.jsonl",
    )
    selected = R._repeat_rows(questions, 16000, seed)
    manifests = {
        r: json.loads((ROOT / "models" / "stage1" / r / "manifest.json").read_text()).get("hashes", {})
        for r in args.runs
        if r.endswith(f"-s{seed}")
    }
    digest = D.question_set_hash(selected)
    positives = [q for q in selected if kind(q) == "positive"]
    trained_raws[seed] = collections.Counter(target_raw(q) for q in positives)
    report[f"composition_s{seed}"] = {
        "sources": sources,
        "generated": dict(collections.Counter(map(kind, questions))),
        "selected_16k": dict(collections.Counter(map(kind, selected))),
        "unique_positive_questions": len({q.question.id for q in positives}),
        "positive_option_count_mean": sum(len(q.question.options) for q in positives) / max(1, len(positives)),
        "manifest_hash_match": {r: h.get("questions") == digest for r, h in manifests.items()},
        "positive_target_raws_top20": trained_raws[seed].most_common(20),
        "positive_target_raw_count": len(trained_raws[seed]),
    }
    print(json.dumps({f"composition_s{seed}": report[f"composition_s{seed}"]}, indent=1), flush=True)
    del docs, cands, questions, selected


# ---------- question sets ----------
def by_docs(rows, n, seed=0):
    """All rows from a random subset of documents, so multi-branch windows stay intact."""
    docs = collections.defaultdict(list)
    for row in rows:
        docs[row.doc_id].append(row)
    order = sorted(docs)
    random.Random(seed).shuffle(order)
    out = []
    for doc_id in order:
        if len(out) >= n:
            break
        out += docs[doc_id]
    return out


def restyle(rows, *, spaces=False, not_pii_description=None):
    out = []
    for row in rows:
        options = tuple(
            Option(o.name, o.description if not_pii_description is None else not_pii_description)
            if o.name == H.NOT_PII
            else Option(o.name.replace("_", " ") if spaces else o.name, o.description)
            for o in row.question.options
        )
        out.append(dataclasses.replace(row, question=dataclasses.replace(row.question, options=options)))
    return out


gretel_dev = sorted(bench.dev_slice(L.load("gretel", "train", purpose="train"))[0], key=lambda d: d.doc_id)
gretel_dev = gretel_dev[: args.gretel_docs]
gretel_cands = R._proposer_candidates(SCR, gretel_dev, False, cache_name="gretel-dev")
generated = D.generate_questions(
    gretel_dev,
    seed=7,
    variant="no-nemotron",
    min_options=2,
    max_options=64,
    hard_negative_hook=lambda doc: gretel_cands.get(doc.doc_id, ()),
    ledger_path=SCR / "ledger.jsonl",
)
rng = random.Random(0)
train_pos = [q for q in generated if kind(q) == "positive"]
train_hn = [q for q in generated if kind(q) == "hard_negative"]
train_pos = rng.sample(train_pos, min(N, len(train_pos)))
train_hn = rng.sample(train_hn, min(N, len(train_hn)))

c3 = {label.name for label in v2.c3_label_set(False).labels}
gretel_raws = {D._raw(s) for d in gretel_dev for s in d.spans}
seen_g = sorted(r for r in gretel_raws & c3 if not H.excluded(r, held))
dev_g = sorted(r for r in gretel_raws & c3 if r in set(held["dev_labels"]))
eval_g = by_docs(
    D.span_evaluation_questions(gretel_dev, gretel_cands, labels=seen_g, seed=0, require_equal_negatives=False), N
)
eval_g_dev = (
    by_docs(
        D.span_evaluation_questions(gretel_dev, {}, labels=dev_g, seed=0, require_equal_negatives=False), N
    )
    if dev_g
    else []
)
nemo_seen = by_docs(R._seen_questions(REAL), N)
nemo_dev = by_docs(R._dev_questions(REAL)[1], N)
seen_labels = {r.question.options[r.target].name for r in R._seen_questions(REAL)}
report["labels"] = {
    "gretel_seen_eval_labels": seen_g,
    "gretel_dev_labels": dev_g,
    "nemotron_seen_labels_never_trained": sorted(seen_labels - set().union(*map(set, trained_raws.values()))),
    "nemotron_seen_labels": sorted(seen_labels),
}
print(json.dumps({"labels": report["labels"]}, indent=1), flush=True)

SETS = {
    "A_train_fmt_pos": (train_pos, ("single", "batched_single")),
    "A_train_fmt_hardneg": (train_hn, ("single",)),
    "A_train_fmt_pos_undescribed_notpii": (restyle(train_pos, not_pii_description=""), ("single",)),
    "B_eval_fmt_gretel_seen": (eval_g, ("single", "batched_single", "dev_eval")),
    "C_eval_fmt_gretel_seen_trainstyle": (
        restyle(eval_g, spaces=True, not_pii_description="the span is not PII"),
        ("single", "dev_eval"),
    ),
    "D_eval_fmt_gretel_seen_undescribed_notpii": (restyle(eval_g, not_pii_description=""), ("single",)),
    "E_eval_fmt_gretel_devlabels": (eval_g_dev, ("single",)),
    "F_eval_fmt_nemotron_seen": (nemo_seen, ("single", "dev_eval")),
    "G_eval_fmt_nemotron_dev": (nemo_dev, ("single", "dev_eval")),
}


# ---------- scoring paths ----------
def score_single(model, tok, rows):
    """Exactly the training forward: one question, one window, B=1 BlockMask."""
    out = []
    for row in rows:
        packed, _ = D.pack_training_questions(
            [row], tok, device=DEVICE, make_block_mask=True, keep_dense_mask=False
        )[0]
        out.append(model(packed).probabilities[0].float().cpu())
    return out


def score_batched_single(model, tok, rows, batch=8):
    """Same single-question windows through forward_many (batched BlockMask)."""
    out = [None] * len(rows)
    for first in range(0, len(rows), batch):
        packs = [
            (
                first + i,
                D.pack_training_questions(
                    [row], tok, device=DEVICE, make_block_mask=False, keep_dense_mask=True
                )[0][0],
            )
            for i, row in enumerate(rows[first : first + batch])
        ]
        by_len = collections.defaultdict(list)
        for i, packed in packs:
            by_len[len(packed.input_ids)].append((i, packed))
        for group in by_len.values():
            for (i, _), result in zip(group, model.forward_many([p for _, p in group])):
                out[i] = result.probabilities[0].float().cpu()
    return out


def score_dev_eval(model, tok, rows, name):
    """The dev_eval code path itself (64-branch windows, batch of 8), cached only under scratch."""
    probs = R._score_loaded_trained(
        SCRATCH,
        model,
        tok,
        rows,
        layout="shared",
        key=f"diag-{name}",
        split="x",
        fingerprint=f"diag-{time.time()}",
    )
    return [torch.tensor(p) for p in probs]


PATHS = {"single": score_single, "batched_single": score_batched_single}


def summarize(rows, probs):
    gold = [0, 0]
    neg = [0, 0]
    picks_not_pii = 0
    picked = collections.Counter()
    for row, p in zip(rows, probs):
        names = [o.name for o in row.question.options]
        pred = int(p.argmax())
        hit = pred == row.target
        picks_not_pii += names[pred] == H.NOT_PII
        bucket = neg if row.hard_negative else gold
        bucket[0] += hit
        bucket[1] += 1
        if not row.hard_negative:
            picked[names[pred]] += 1
    return {
        "n": len(rows),
        "gold_acc": gold[0] / gold[1] if gold[1] else None,
        "not_pii_acc": neg[0] / neg[1] if neg[1] else None,
        "not_pii_pick_rate": picks_not_pii / max(1, len(rows)),
        "mean_max_prob": float(sum(p.max() for p in probs) / max(1, len(probs))),
        "top_picks_on_gold": picked.most_common(5),
    }


def compare(a, b):
    return {
        "max_abs_diff": float(max((x - y).abs().max() for x, y in zip(a, b))),
        "argmax_agree": sum(int(x.argmax() == y.argmax()) for x, y in zip(a, b)) / len(a),
    }


# ---------- run ----------
for run_name in args.runs:
    size = run_name.split("-s")[0]
    model_id = next(m for m in cfg["models"] if m.endswith(size))
    revision = cfg["models"][model_id]["revision"]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = S1DModel.from_pretrained(model_id, revision)
    apply_lora(model, prepare_tokenizer(tok, model), rank=32)
    state = torch.load(
        ROOT / "models" / "stage1" / run_name / "checkpoint.pt", map_location="cpu", weights_only=False
    )["trainable_model"]
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    result = model.load_state_dict(state, strict=False)
    lora_b = [v.float().abs().mean().item() for k, v in state.items() if "lora_B" in k]
    report[run_name] = {
        "checkpoint": {
            "saved": len(state),
            "trainable": len(trainable),
            "trainable_not_saved": len(trainable - set(state)),
            "unexpected": len(result.unexpected_keys),
            "lora_B_mean_abs": sum(lora_b) / max(1, len(lora_b)),
            "pointer_bias": float(model.pointer_bias),
        }
    }
    model.to(DEVICE).eval()
    print(json.dumps({run_name: report[run_name]}, indent=1), flush=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for set_name, (rows, paths) in SETS.items():
            if not rows:
                continue
            scored = {}
            for path in paths:
                t = time.time()
                scored[path] = (
                    score_dev_eval(model, tok, rows, f"{run_name}-{set_name}")
                    if path == "dev_eval"
                    else PATHS[path](model, tok, rows)
                )
                entry = summarize(rows, scored[path]) | {"seconds": round(time.time() - t, 1)}
                if path != "single":
                    entry["vs_single"] = compare(scored["single"], scored[path])
                report[run_name][f"{set_name}/{path}"] = entry
                print(run_name, f"{set_name}/{path}", json.dumps(entry), flush=True)
    del model
    torch.cuda.empty_cache()

(SCR / "diag.json").write_text(json.dumps(report, indent=1, default=str))
print("wrote", SCR / "diag.json")
