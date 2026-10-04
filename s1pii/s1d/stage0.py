"""Measured Stage 0 probes. Nothing in this module fabricates a successful result."""
from __future__ import annotations

import gc
import json
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import torch

from .. import bench
from ..schema import read_jsonl
from .labels import NOT_PII, load
from .train import GPUHours


LETTERS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def set_phase(root: Path, phase: str) -> None:
    if phase not in {"CPU", "GPU"}:
        raise ValueError("phase must be CPU or GPU")
    (Path(root) / "PHASE").write_text(phase)


class UnitClock:
    """Incremental wall-clock accounting, including setup and teardown."""
    def __init__(self, root: Path, config: dict, unit: str):
        self.meter = GPUHours(Path(root) / "gpu_hours.jsonl",
                              config["stages"]["stage0"]["cap_a100_hours"], "stage0")
        self.unit, self.last = unit, time.monotonic()

    def tick(self, **extra) -> None:
        now = time.monotonic()
        self.meter.record(self.unit, now - self.last, stage="stage0", **extra)
        self.last = now
        # Account for elapsed allocation before checking the cap. Chunked callers
        # bound any overrun to one short chunk instead of one entire preprocessing job.
        self.meter.reserve(0.0)


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
                         "surface": doc.text[span.start:span.end], "label": raw, "options": options,
                         "target": options.index(raw)})
    if not rows:
        raise ValueError("no dev-label gold spans found in the Nemotron calibration split")
    return rows


def span_window(tokenizer, row: dict, max_tokens: int) -> dict:
    """Crop a token window containing the full gold span without truncating it."""
    encoded = tokenizer(row["state"], add_special_tokens=False, return_offsets_mapping=True,
                        truncation=False)
    offsets = list(encoded["offset_mapping"])
    touched = [i for i, (a, b) in enumerate(offsets) if a < row["end"] and b > row["start"]]
    if not touched:
        raise ValueError(f"span {row['doc_id']}:{row['start']}:{row['end']} has no tokenizer offsets")
    first, last = touched[0], touched[-1] + 1
    if last - first > max_tokens:
        raise ValueError(f"span {row['doc_id']} does not fit in {max_tokens} state tokens")
    begin = max(0, min(first - max(0, (max_tokens - (last - first)) // 2), len(offsets) - max_tokens))
    finish = min(len(offsets), begin + max_tokens)
    if not (begin <= first and last <= finish):
        raise ValueError(f"span {row['doc_id']} was not contained by its state window")
    char_begin = offsets[begin][0] if offsets else 0
    char_end = offsets[finish - 1][1] if offsets else len(row["state"])
    result = dict(row)
    result["state"] = row["state"][char_begin:char_end]
    result["start"] = row["start"] - char_begin
    result["end"] = row["end"] - char_begin
    result["window_start"] = row.get("window_start", 0) + char_begin
    result["state_tokens"] = finish - begin
    return result


def _description(name: str, config: dict | None = None) -> str:
    cfg = config or load()
    from .data import NOT_PII_DESCRIPTION
    return (NOT_PII_DESCRIPTION if name == NOT_PII
            else cfg["frozen"][name]["description"])


def _prompt(tokenizer, row: dict) -> str:
    choices = [f"{LETTERS[i]}. {name.replace('_', ' ')}: {_description(name)}"
               for i, name in enumerate(row["options"])]
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


def restricted_letter_log_probs(logits: torch.Tensor, letter_ids: list[int]) -> torch.Tensor:
    """Score only option letters at the final input position."""
    if logits.ndim != 3:
        raise ValueError("causal LM logits must have shape batch x sequence x vocabulary")
    return logits[:, -1, letter_ids].float().log_softmax(-1)


def accuracy_summary(rows: list[dict], predictions: list[int]) -> dict:
    if len(rows) != len(predictions) or not rows:
        raise ValueError("predictions must align with a non-empty question set")
    labels = list(dict.fromkeys(row["label"] for row in rows))
    by_label = {}
    for label in labels:
        pairs = [(row, pred) for row, pred in zip(rows, predictions) if row["label"] == label]
        by_label[label] = {"correct": sum(pred == row["target"] for row, pred in pairs), "questions": len(pairs)}
        by_label[label]["accuracy"] = by_label[label]["correct"] / by_label[label]["questions"]
    correct = sum(pred == row["target"] for row, pred in zip(rows, predictions))
    majority = max(Counter(row["target"] for row in rows).values()) / len(rows)
    return {"accuracy": correct / len(rows),
            "macro_accuracy": sum(value["accuracy"] for value in by_label.values()) / len(by_label),
            "per_label_accuracy": by_label, "majority_class_baseline": majority}


@torch.no_grad()
def prompted_probe(root: Path, config: dict, control_path: Path, *, batch_size: int = 8) -> dict:
    if not torch.cuda.is_available():
        raise NotImplementedError("prompted probe requires the Stage 0 CUDA runtime")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    raw_rows = dev_gold_questions(); by_model = {}; clock = UnitClock(root, config, "prompted-probe")
    for model_id, spec in config["models"].items():
        set_phase(root, "CPU")
        revision = spec["revision"]
        tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, padding_side="left")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        rows = [span_window(tokenizer, row, config["state_tokens"]) for row in raw_rows]
        prompts = [_prompt(tokenizer, row) for row in rows]
        lengths = [len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) for prompt in prompts]
        if max(lengths) > 2048:
            raise ValueError(f"prompt exceeds 2048 tokens after span windowing for {model_id}")
        clock.tick(model=model_id, substep="tokenize")
        set_phase(root, "GPU")
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.bfloat16,
                                                     attn_implementation="sdpa").cuda().eval()
        clock.tick(model=model_id, substep="load")
        letter_ids = _letter_ids(tokenizer, len(rows[0]["options"]))
        predictions, log_probs = [], []
        for start in range(0, len(rows), batch_size):
            if control_path.exists() and control_path.read_text().strip().upper() == "STOP":
                raise InterruptedError("STOP requested")
            chunk_prompts = prompts[start:start + batch_size]
            encoded = tokenizer(chunk_prompts, padding=True, return_tensors="pt")
            if encoded["input_ids"].shape[1] > 2048:
                raise ValueError("batched prompt was unexpectedly truncated")
            encoded = encoded.to("cuda")
            logits = model(**encoded, logits_to_keep=1).logits
            lp = restricted_letter_log_probs(logits, letter_ids).cpu()
            predictions.extend(lp.argmax(-1).tolist()); log_probs.extend(lp.tolist())
            clock.tick(model=model_id, substep="inference", examples=len(chunk_prompts))
        summary = accuracy_summary(rows, predictions)
        by_model[model_id] = {"revision": revision, "questions": len(rows), "log_probs": log_probs, **summary}
        del model; gc.collect(); torch.cuda.empty_cache(); clock.tick(model=model_id, substep="unload")
    best_id = max(by_model, key=lambda name: by_model[name]["macro_accuracy"])
    best = by_model[best_id]; chance = 1.0 / len(raw_rows[0]["options"])
    return {"implementation_version": 2, "questions": len(raw_rows),
            "options": len(raw_rows[0]["options"]), "chance": chance,
            "best_model": best_id, "models": by_model, "prompt": "option letters with /no_think", **
            {key: best[key] for key in ("accuracy", "macro_accuracy", "per_label_accuracy",
                                        "majority_class_baseline")}}


def _run_accounted(command: list[str], *, clock: UnitClock, control_path: Path,
                   cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    process = subprocess.Popen(command, cwd=cwd, env=env)
    try:
        while process.poll() is None:
            if control_path.exists() and control_path.read_text().strip().upper() == "STOP":
                raise InterruptedError("STOP requested")
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            clock.tick(substep="subprocess")
        if process.returncode:
            raise RuntimeError(f"command exited with status {process.returncode}: {command[0]}")
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill(); process.wait()
        raise


def _kev_runtime(root: Path, config: dict, control_path: Path, clock: UnitClock) -> tuple[Path, Path]:
    from huggingface_hub import snapshot_download
    spec = config["external"]["kev"]
    runtime = root / "models" / f"kev-runtime-{spec['git_revision'][:12]}"
    python = runtime / ".venv" / "bin" / "python"
    marker = runtime / "INSTALLED"
    runtime.mkdir(parents=True, exist_ok=True)
    set_phase(root, "CPU")
    if not marker.exists() or marker.read_text().strip() != spec["git_revision"]:
        uv = shutil.which("uv")
        if not uv:
            raise RuntimeError("uv is required to create the isolated Kev environment")
        if not python.exists():
            _run_accounted([uv, "venv", "--python", "3.12", str(runtime / ".venv")],
                           clock=clock, control_path=control_path)
        requirement = f"kev[serve] @ git+https://github.com/jaredpalmer/kev.git@{spec['git_revision']}"
        _run_accounted([uv, "pip", "install", "--python", str(python), requirement],
                       clock=clock, control_path=control_path)
        marker.write_text(spec["git_revision"])
    snapshot = root / "models" / f"kev-0.5b-{spec['revision'][:12]}"
    snapshot_download(spec["model_id"], revision=spec["revision"], local_dir=snapshot)
    clock.tick(substep="download")
    meta = torch.load(snapshot / "head.pt", map_location="cpu", weights_only=False)
    if meta.get("base") != spec["base_id"]:
        raise RuntimeError(f"Kev checkpoint base {meta.get('base')!r} differs from the configured base")
    if meta.get("base_revision") != spec["base_revision"]:
        meta["base_revision"] = spec["base_revision"]
        tmp = snapshot / "head.pt.part"; torch.save(meta, tmp); os.replace(tmp, snapshot / "head.pt")
    clock.tick(substep="pin-base")
    return python, snapshot


def _json_request(url: str, payload: dict | None = None, timeout: float = 30) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={"content-type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def predictions_from_probabilities(rows: list[dict], probability_maps: list[dict]) -> tuple[list[int], list[list[float]]]:
    """Validate and rescore upstream probability maps without trusting its chosen answer."""
    if len(rows) != len(probability_maps):
        raise ValueError("Kev probability rows do not align with questions")
    predictions, vectors = [], []
    for row, probabilities in zip(rows, probability_maps):
        if set(probabilities) != set(row["options"]):
            raise ValueError("Kev returned probabilities over a different option set")
        vector = [float(probabilities[name]) for name in row["options"]]
        predictions.append(max(range(len(vector)), key=vector.__getitem__))
        vectors.append(vector)
    return predictions, vectors


def kev_baseline(root: Path, config: dict, control_path: Path) -> dict:
    """Run the pinned upstream Kev server and rescore its probability vectors locally."""
    spec = config["external"]["kev"]; clock = UnitClock(root, config, "kev-baseline")
    python, snapshot = _kev_runtime(root, config, control_path, clock)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(spec["base_id"], revision=spec["base_revision"])
    # First apply the experiment's 512-token state policy, then Kev's stricter
    # training-time cap around the same (now local) span coordinates.
    rows = [span_window(tokenizer, span_window(tokenizer, row, config["state_tokens"]),
                        spec["state_tokens"]) for row in dev_gold_questions()]
    clock.tick(substep="tokenize")
    port = _free_port(); base_url = f"http://127.0.0.1:{port}"
    log_path = root / "logs" / "kev-server.log"; log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "KEV_CUDA_GRAPHS": "0", "KEV_FUSED": "0"}
    set_phase(root, "GPU" if torch.cuda.is_available() else "CPU")
    log = open(log_path, "a")
    server = subprocess.Popen([str(python), "-m", "kev.serve", "--run", str(snapshot), "--port", str(port)],
                              stdout=log, stderr=subprocess.STDOUT, env=env)
    probability_maps = []
    try:
        deadline = time.monotonic() + 900
        while True:
            if server.poll() is not None:
                raise RuntimeError(f"Kev server exited with status {server.returncode}; see {log_path}")
            if control_path.exists() and control_path.read_text().strip().upper() == "STOP":
                raise InterruptedError("STOP requested")
            try:
                _json_request(base_url + "/v1/models", timeout=2); break
            except (OSError, urllib.error.URLError):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Kev server did not become healthy")
                time.sleep(2); clock.tick(substep="server-load")
        for i, row in enumerate(rows):
            criteria = {name: _description(name) for name in row["options"]}
            payload = {"state": row["state"], "model": "kev-latest", "questions": {
                "pii": {"type": "choice",
                        "instructions": f"Classify the span {row['surface']!r} as personal information.",
                        "criteria": criteria}}}
            response = _json_request(base_url + "/v1/systemone", payload, timeout=120)
            probabilities = response["answers"]["pii"]["probabilities"]
            probability_maps.append(probabilities)
            clock.tick(substep="inference", example=i + 1)
    finally:
        if server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill(); server.wait()
        log.close(); clock.tick(substep="teardown")
    predictions, probability_rows = predictions_from_probabilities(rows, probability_maps)
    chance = 1.0 / len(rows[0]["options"])
    return {"implementation_version": 2, "questions": len(rows),
            "options": len(rows[0]["options"]), "chance": chance,
            "probabilities": probability_rows, "kev_git_sha": spec["git_revision"],
            "kev_hf_sha": spec["revision"], "base_sha": spec["base_revision"],
            "state_token_cap": spec["state_tokens"], **accuracy_summary(rows, predictions)}


def latency_benchmark(root: Path, config: dict, control_path: Path) -> dict:
    from .latency import benchmark
    clock = UnitClock(root, config, "latency")
    set_phase(root, "CPU")
    result = benchmark(config, dry=False, root=root, control_path=control_path,
                       phase_callback=lambda phase: set_phase(root, phase))
    clock.tick(substep="benchmark")
    result_path = root / "eval" / "latency.json"; result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    return result
