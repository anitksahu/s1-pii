"""Measured Stage 0 probes. Nothing in this module fabricates a successful result."""
from __future__ import annotations

import gc
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import torch

from .. import bench
from ..schema import read_jsonl
from .labels import NOT_PII, load
from .train import GPUHours


LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def dev_gold_questions(config: dict | None = None) -> list[dict]:
    cfg = config or load(); wanted = set(cfg["dev_labels"])
    docs = read_jsonl(bench.split_paths("nemotron")["calib"])
    options = [*cfg["dev_labels"], NOT_PII]
    rows = []
    for doc in docs:
        for span in doc.spans:
            raw = span.label_raw.lower()
            if raw not in wanted:
                continue
            rows.append({"doc_id": doc.doc_id, "state": doc.text, "start": span.start, "end": span.end,
                         "surface": doc.text[span.start:span.end], "options": options,
                         "target": options.index(raw)})
    if not rows:
        raise ValueError("no dev-label gold spans found in the Nemotron calibration split")
    return rows


def _prompt(tokenizer, row: dict) -> str:
    cfg = load()
    choices = []
    for i, name in enumerate(row["options"]):
        description = "the span is not personal information" if name == NOT_PII else cfg["frozen"][name]["description"]
        choices.append(f"{LETTERS[i]}. {name.replace('_', ' ')}: {description}")
    text = ("Classify the marked span using exactly one option letter. /no_think\n\n"
            f"Document:\n{row['state']}\n\nSpan: {row['surface']}\n\nOptions:\n" + "\n".join(choices))
    messages = [{"role": "user", "content": text}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    return text + "\nAnswer:"


def _letter_ids(tokenizer, count: int) -> list[int]:
    ids = []
    for letter in LETTERS[:count]:
        pieces = tokenizer(letter, add_special_tokens=False)["input_ids"]
        if len(pieces) != 1:
            pieces = tokenizer(" " + letter, add_special_tokens=False)["input_ids"]
        if len(pieces) != 1:
            raise ValueError(f"option letter {letter!r} is not one token")
        ids.append(int(pieces[0]))
    return ids


@torch.no_grad()
def prompted_probe(root: Path, config: dict, control_path: Path, *, batch_size: int = 8) -> dict:
    if not torch.cuda.is_available():
        raise NotImplementedError("prompted probe requires the Stage 0 CUDA runtime")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows = dev_gold_questions(); by_model = {}
    meter = GPUHours(root / "gpu_hours.jsonl", config["stages"]["stage0"]["cap_a100_hours"], "stage0")
    for model_id, spec in config["models"].items():
        revision = spec["revision"]
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, padding_side="left")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16,
                                                     attn_implementation="sdpa").cuda().eval()
        letter_ids = _letter_ids(tokenizer, len(rows[0]["options"]))
        correct = 0; log_probs = []; accounted = time.monotonic()
        for start in range(0, len(rows), batch_size):
            if control_path.exists() and control_path.read_text().strip().upper() == "STOP":
                raise InterruptedError("STOP requested")
            meter.reserve(0.0)
            chunk = rows[start:start + batch_size]
            encoded = tokenizer([_prompt(tokenizer, row) for row in chunk], padding=True, truncation=True,
                                max_length=2048, return_tensors="pt").to("cuda")
            logits = model(**encoded).logits
            scores = logits[:, -1, letter_ids].float()
            lp = scores.log_softmax(-1).cpu()
            pred = lp.argmax(-1).tolist()
            correct += sum(p == row["target"] for p, row in zip(pred, chunk))
            log_probs.extend(lp.tolist())
            now = time.monotonic(); meter.record(f"prompt-{model_id.rsplit('/', 1)[-1]}", now - accounted,
                                                  stage="stage0", examples=len(chunk)); accounted = now
        by_model[model_id] = {"revision": revision, "accuracy": correct / len(rows),
                              "correct": correct, "questions": len(rows), "log_probs": log_probs}
        del model; gc.collect(); torch.cuda.empty_cache()
    chance = 1.0 / len(rows[0]["options"])
    return {"questions": len(rows), "options": len(rows[0]["options"]), "chance": chance,
            "accuracy": max(value["accuracy"] for value in by_model.values()), "models": by_model,
            "prompt": "option letters with /no_think"}


def _run_accounted(command: list[str], *, env: dict[str, str], meter: GPUHours,
                   unit: str, control_path: Path) -> None:
    process = subprocess.Popen(command, env=env)
    accounted = time.monotonic()
    try:
        while process.poll() is None:
            if control_path.exists() and control_path.read_text().strip().upper() == "STOP":
                raise InterruptedError("STOP requested")
            meter.reserve(0.0)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            meter.record(unit, now - accounted, stage="stage0"); accounted = now
        now = time.monotonic()
        if now > accounted:
            meter.record(unit, now - accounted, stage="stage0")
        if process.returncode:
            raise RuntimeError(f"{unit} exited with status {process.returncode}")
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        raise


def kev_baseline(root: Path, config: dict, control_path: Path) -> dict:
    """Run a pinned Kev command supplied by the Colab environment and validate its artifact."""
    command = os.environ.get("S1D_KEV_COMMAND", "").strip()
    if not command:
        raise NotImplementedError("set S1D_KEV_COMMAND to the pinned Kev baseline command")
    if control_path.exists() and control_path.read_text().strip().upper() == "STOP":
        raise InterruptedError("STOP requested")
    out = root / "eval" / "kev-stage0.json"; out.parent.mkdir(parents=True, exist_ok=True)
    rows = dev_gold_questions()
    questions = root / "eval" / "kev-stage0-questions.json"
    questions.write_text(json.dumps(rows, indent=2, sort_keys=True))
    env = {**os.environ, "S1D_KEV_OUTPUT": str(out), "S1D_KEV_QUESTIONS": str(questions)}
    meter = GPUHours(root / "gpu_hours.jsonl", config["stages"]["stage0"]["cap_a100_hours"], "stage0")
    _run_accounted(shlex.split(command), env=env, meter=meter, unit="kev-baseline",
                   control_path=control_path)
    if not out.exists():
        raise RuntimeError("Kev baseline command failed to write S1D_KEV_OUTPUT")
    result = json.loads(out.read_text())
    for key in ("accuracy", "chance", "questions", "revision"):
        if key not in result:
            raise ValueError(f"Kev result missing {key}")
    expected_chance = 1.0 / len(rows[0]["options"])
    if result["questions"] != len(rows) or abs(float(result["chance"]) - expected_chance) > 1e-12:
        raise ValueError("Kev result does not match the supplied dev question set")
    return result


def latency_benchmark(root: Path, config: dict, control_path: Path) -> dict:
    """Benchmark an already merged model; absence of one is an explicit incomplete unit."""
    artifact = root / "models" / "latency" / "benchmark.py"
    result_path = root / "eval" / "latency.json"
    if not artifact.exists():
        raise NotImplementedError(f"merged latency harness not found at {artifact}")
    if control_path.exists() and control_path.read_text().strip().upper() == "STOP":
        raise InterruptedError("STOP requested")
    env = {**os.environ, "S1D_ROOT": str(root), "S1D_LATENCY_OUTPUT": str(result_path)}
    meter = GPUHours(root / "gpu_hours.jsonl", config["stages"]["stage0"]["cap_a100_hours"], "stage0")
    _run_accounted([sys.executable, str(artifact)], env=env, meter=meter, unit="latency",
                   control_path=control_path)
    if not result_path.exists():
        raise RuntimeError("latency harness did not write its result")
    result = json.loads(result_path.read_text())
    required = {"dtype", "merged_lora", "batch_size", "branches", "options", "mask_prebuilt",
                "warmup_excluded", "p95_window_ms", "p95_document_ms"}
    if required - result.keys():
        raise ValueError(f"latency result missing {sorted(required - result.keys())}")
    expected = {"dtype": "bfloat16", "merged_lora": True, "batch_size": 1, "branches": 64,
                "options": 56, "mask_prebuilt": True, "warmup_excluded": True}
    if any(result[key] != value for key, value in expected.items()):
        raise ValueError("latency result does not describe the required benchmark protocol")
    return result
