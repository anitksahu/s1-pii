"""Frozen S1-versus-GLiNER benchmark specified by s1_vs_gliner_prereg.md.

This unit performs inference and scoring only.  Candidate mining and model probabilities are
cached independently, so an interrupted Colab run resumes without repeating completed work.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .. import bench, c0, ledger
from ..eval import metrics as M
from ..eval.evaluate import compare, evaluate, tune_threshold, views
from ..schema import CANONICAL_TYPES, Doc, OTHER_PII, Span, read_jsonl
from ..v2 import labels as v2
from . import data as D
from . import labels as H
from .schema import Question

DATASETS = ("pii_trace", "tab_direct", "spy_legal", "spy_medical", "nemotron")
S1_SYSTEMS = ("s1_prompted_qwen3_4b", "s1d_1.7b_s1", "s1d_1.7b_s2", "s1d_4b_s1",
              "proposer_only")
GLINER_SYSTEMS = ("gliner2_pii", "nvidia_gliner_pii", "gliner25_base_zeroshot")
TRAINED = {"s1d_1.7b_s1": "1.7B-s1", "s1d_1.7b_s2": "1.7B-s2", "s1d_4b_s1": "4B-s1"}
IMPLEMENTATION_VERSION = 1


def _dry_split(dataset: str, split: str) -> list[Doc]:
    out = []
    for i, raw in enumerate(("email", "phone_number")):
        surface = "ada@example.test" if raw == "email" else "5550102"
        text = f"ordinary {surface}"
        start = text.index(surface)
        did = f"dry-{dataset}-{split}-{i}"
        out.append(Doc(did, text, (Span(did, start, len(text), v2.canonical_of(raw), raw,
                                         surface=surface),), dataset, split, did))
    return out


def _candidate_rows(docs: list[Doc], candidates: dict[str, list[Span]]) -> list[D.TrainingQuestion]:
    options = D.evaluation_options()
    target = D.evaluation_option_index()[H.NOT_PII]
    rows = []
    for doc in docs:
        for i, span in enumerate(candidates.get(doc.doc_id, ())):
            q = Question("choice", "Classify the marked span.", "Use its meaning and context.", options,
                         id=f"{doc.doc_id}:benchmark:{i}:{span.start}:{span.end}",
                         span=(span.start, span.end))
            rows.append(D.TrainingQuestion(doc.doc_id, doc.text, q, target,
                                           (span.start, span.end), True))
    return rows


def _predictions(docs: list[Doc], candidates: dict[str, list[Span]], rows, probabilities,
                 temperature: float, system: str) -> dict[str, list[Span]]:
    p, not_pii = __import__("s1pii.s1d.run", fromlist=["_test_calibrated_probabilities"]) \
        ._test_calibrated_probabilities(rows, probabilities, temperature)
    raws = [label.name for label in v2.c3_label_set(False).labels]
    out = {doc.doc_id: [] for doc in docs}
    offsets = {doc.doc_id: list(candidates.get(doc.doc_id, ())) for doc in docs}
    used = {doc.doc_id: 0 for doc in docs}
    for row, dist in zip(rows, p):
        index = used[row.doc_id]; used[row.doc_id] += 1
        candidate = offsets[row.doc_id][index]
        choice = dist.copy(); choice[not_pii] = -1
        raw = raws[int(choice.argmax())]
        out[row.doc_id].append(Span(row.doc_id, candidate.start, candidate.end,
                                    v2.canonical_of(raw), raw, float(1 - dist[not_pii]),
                                    system, candidate.surface))
    return out


def _proposer_predictions(docs: list[Doc], candidates: dict[str, list[Span]]) -> dict[str, list[Span]]:
    return {doc.doc_id: [replace(span, label_canonical=OTHER_PII, label_raw="candidate",
                                 source="proposer_only")
                         for span in candidates.get(doc.doc_id, ())]
            for doc in docs}


def _mean(values) -> float:
    clean = [float(v) for v in values if np.isfinite(v)]
    return float(np.mean(clean)) if clean else float("nan")


def _strict_report(docs: list[Doc], predictions: dict[str, list[Span]], dataset: str) -> dict:
    """Exact span report using the same admissibility and matching code as headline scoring."""
    vs = views(docs, predictions, dataset)
    per_type = {}
    for canonical in CANONICAL_TYPES:
        selected = [replace(v,
                            gold=[g for g in v.gold if g[2] == canonical],
                            preds=[p for p in v.preds if p.label_canonical == canonical])
                    for v in vs]
        per_type[canonical] = M.span_prf(selected, 0.5, typed=True, mode="strict")
    micro = M.span_prf(vs, 0.5, typed=True, mode="strict")
    macro = {name: _mean(row[name] for row in per_type.values())
             for name in ("precision", "recall", "f1")}
    return {"per_type": per_type, "micro": micro, "macro": macro}


def _score(docs, calib_docs, test_predictions, calib_predictions, dataset, *, dry: bool):
    threshold = tune_threshold(calib_docs, calib_predictions, dataset, over_budget=0.01)
    result = evaluate(docs, test_predictions, dataset=dataset, default_threshold=0.5,
                      dev_threshold=threshold, n_boot_ci=100 if dry else 2000, seed=0)
    result["dev_threshold"] = threshold
    result["strict_at_0_5"] = _strict_report(docs, test_predictions, dataset)
    return result


def _write_predictions(root: Path, system: str, dataset: str, split: str, docs, predictions) -> Path:
    path = root / "stores" / "s1-bench-predictions" / f"{system}-{dataset}-{split}.jsonl"
    ledger.write_predictions(predictions, path, {
        "system": system, "revision": IMPLEMENTATION_VERSION, "adapter_version": "s1-bench-v1",
        "dataset_hash": ledger.dataset_hash(docs), "docs_path": f"s1_bench:{dataset}:{split}",
        "config": {"floor": 0.01, "state_tokens": 512, "options": 56,
                   "keep_overlaps": True},
    })
    return path


def _baseline_predictions(dataset: str, test_docs: list[Doc], system: str, *, dry: bool,
                          proposer: dict[str, list[Span]]):
    if dry:
        # Deterministic stand-in: the dry run exercises identical metrics and bootstrap paths.
        return {doc.doc_id: list(proposer.get(doc.doc_id, ())) for doc in test_docs}
    path = bench.split_paths(dataset)["test"].resolve()
    index = c0.prediction_index()
    pred_path = index.get((system, str(path)))
    if pred_path is None:
        raise FileNotFoundError(f"missing frozen {system} predictions for {dataset}: {path}")
    all_docs = read_jsonl(path)
    _meta, predictions = c0._load_checked(pred_path, all_docs, {doc.doc_id for doc in test_docs})
    return predictions


def _baseline_metrics(dataset: str, system: str, final_results: dict, *, dry: bool,
                      docs, predictions):
    if not dry:
        row = dict(final_results[dataset]["systems"][system])
        # final_results preserves each model-card default (0.3 for NVIDIA).  This frozen
        # experiment additionally requires one common strict-span operating point at 0.5.
        row["strict_at_0_5"] = _strict_report(docs, predictions, dataset)
        return row
    row = evaluate(docs, predictions, dataset=dataset, default_threshold=0.5,
                   n_boot_ci=20, seed=0)
    row["strict_at_0_5"] = _strict_report(docs, predictions, dataset)
    return row


def _proposer_ceiling(docs, predictions, dataset):
    vs = views(docs, predictions, dataset)
    _clusters, hp, hn = M.cluster_histograms(vs)
    return {"maximum_achievable_character_recall": M.at_threshold(hp.sum(0), hn.sum(0), 0.0)["char_recall"],
            "candidates": sum(len(v) for v in predictions.values())}


def run_benchmark(ctx: dict, out: Path) -> dict:
    """Run the frozen five-benchmark inference/scoring protocol."""
    from . import run as R

    root, dry = ctx["root"], ctx["dry"]
    final_path = Path(__file__).resolve().parents[2] / "docs" / "final_results.json"
    final_results = {} if dry else json.loads(final_path.read_text())
    output = {
        "implementation_version": IMPLEMENTATION_VERSION,
        "training_or_tuning": False,
        "datasets": list(DATASETS),
        "systems": list((*S1_SYSTEMS, *GLINER_SYSTEMS)),
        "statistics": {"bootstrap_unit": "document_cluster", "samples": 100 if dry else 10_000,
                       "training_seed_variance_propagated": False,
                       "competitive_rule": "upper_95(pAUC S1 - pAUC best GLiNER) <= 0.02"},
        "benchmarks": {},
    }
    R._atomic_json(out, output)

    prepared = {}
    all_sets = {}
    for dataset in DATASETS:
        if dry:
            calib_docs = _dry_split(dataset, "calib")
            test_docs = _dry_split(dataset, "test")
        else:
            calib_docs = read_jsonl(bench.split_paths(dataset)["calib"])
            test_docs = c0.audited_test_docs(dataset)
        calib_candidates = R._proposer_candidates(
            root, calib_docs, dry, cache_name=f"s1-bench-{dataset}-calib-all",
            proposer_variant="no-nemotron", accounting_stage="s1_bench",
            exclude_gold_overlap=False)
        test_candidates = R._proposer_candidates(
            root, test_docs, dry, cache_name=f"s1-bench-{dataset}-test-all",
            proposer_variant="no-nemotron", accounting_stage="s1_bench",
            exclude_gold_overlap=False)
        calib_rows, test_rows = (_candidate_rows(calib_docs, calib_candidates),
                                 _candidate_rows(test_docs, test_candidates))
        prepared[dataset] = (calib_docs, test_docs, calib_candidates, test_candidates,
                             calib_rows, test_rows)
        all_sets[f"{dataset}-calib"] = calib_rows
        all_sets[f"{dataset}-test"] = test_rows

    if dry:
        prompted_temperature = 1.0
        trained_temperatures = {name: 1.0 for name in TRAINED.values()}
    else:
        prompted, trained_temperatures = R._saved_test_temperatures(ctx)
        prompted_temperature = prompted["4B"]
    model_ids = {name.rsplit("-", 1)[-1]: name for name in ctx["config"]["models"]}
    probability_sets = {
        "s1_prompted_qwen3_4b": (R._score_prompted_sets(ctx, all_sets, model_ids["4B"]),
                                  prompted_temperature)
    }
    for system, run_name in TRAINED.items():
        size = run_name.split("-s", 1)[0]
        run_dir = root / "models" / "stage1-cov" / run_name
        if not dry and not (run_dir / "checkpoint.pt").exists():
            raise FileNotFoundError(f"missing frozen stage1_cov checkpoint {run_dir / 'checkpoint.pt'}")
        probability_sets[system] = (
            R._score_trained_sets(ctx, all_sets, model_id=model_ids[size], run_dir=run_dir),
            trained_temperatures[run_name])

    for dataset in DATASETS:
        calib_docs, test_docs, calib_candidates, test_candidates, calib_rows, test_rows = prepared[dataset]
        proposer_calib = _proposer_predictions(calib_docs, calib_candidates)
        proposer_test = _proposer_predictions(test_docs, test_candidates)
        row = {"n_calibration_docs": len(calib_docs), "n_test_docs_audited": len(test_docs),
               "systems": {}, "comparisons": {},
               "proposer_ceiling": _proposer_ceiling(test_docs, proposer_test, dataset)}
        s1_predictions = {"proposer_only": proposer_test}
        row["systems"]["proposer_only"] = _score(
            test_docs, calib_docs, proposer_test, proposer_calib, dataset, dry=dry)
        _write_predictions(root, "proposer_only", dataset, "calib", calib_docs, proposer_calib)
        _write_predictions(root, "proposer_only", dataset, "test", test_docs, proposer_test)
        for system, (scores, temperature) in probability_sets.items():
            calib_pred = _predictions(calib_docs, calib_candidates, calib_rows,
                                      scores[f"{dataset}-calib"], temperature, system)
            test_pred = _predictions(test_docs, test_candidates, test_rows,
                                     scores[f"{dataset}-test"], temperature, system)
            s1_predictions[system] = test_pred
            row["systems"][system] = _score(
                test_docs, calib_docs, test_pred, calib_pred, dataset, dry=dry)
            _write_predictions(root, system, dataset, "calib", calib_docs, calib_pred)
            _write_predictions(root, system, dataset, "test", test_docs, test_pred)

        baseline_predictions = {}
        for system in GLINER_SYSTEMS:
            if (dataset, system) in c0.EXCLUDED:
                continue
            predictions = _baseline_predictions(dataset, test_docs, system, dry=dry,
                                                proposer=proposer_test)
            baseline_predictions[system] = predictions
            row["systems"][system] = _baseline_metrics(
                dataset, system, final_results, dry=dry, docs=test_docs, predictions=predictions)
        best = min(baseline_predictions,
                   key=lambda name: float(row["systems"][name]["pauc"]))
        boots = 100 if dry else 10_000
        for s1_name, prediction in s1_predictions.items():
            row["comparisons"][s1_name] = {}
            for baseline_name, baseline_prediction in baseline_predictions.items():
                comp = compare(test_docs, [prediction], [baseline_prediction], dataset=dataset,
                               n_boot=boots, seed=0)
                comp["document_level_only"] = True
                comp["training_seed_variance_propagated"] = False
                if baseline_name == best:
                    comp["competitive"] = comp["ci_high"] <= 0.02
                    comp["best_gliner"] = True
                row["comparisons"][s1_name][baseline_name] = comp
        row["best_gliner"] = best
        output["benchmarks"][dataset] = row
        R._atomic_json(out, output)
        print(f"s1_bench {dataset}: {len(test_docs)} audited test docs, "
              f"{sum(map(len, test_candidates.values()))} candidates", flush=True)
    return output
