"""v2 chain on one GPU, resumable, with a hard A100-hour cap.

    python -m s1pii.v2.run_v2 cheap        --models $DRIVE/models --work /content/v2 --drive $DRIVE/v2
    python -m s1pii.v2.run_v2 gliner_c3    ...   (runs in the gliner2 venv through a subprocess)
    python -m s1pii.v2.run_v2 gates        ...
    python -m s1pii.v2.run_v2 conditional  ...   (B, then A, then C, as the gates and the cap allow)
    python -m s1pii.v2.run_v2 all          ...   (the four above in order)

Every unit writes a DONE marker and is skipped on resume. GPU wall time of every unit is
appended to ``<drive>/gpu_hours.jsonl``; a unit does not start if the cumulative total plus
its estimate exceeds the cap (``--cap``, A100-hours); the chain then stops and says why.
System names: ``s1v2-<tag>_<variant>-s<seed>`` with tag ``cheap``, ``B`` (level-1 retrained)
or ``A`` (typing encoder fine-tuned). The headline tag is the last stage that ran
(``<drive>/state.json``), fixed by the preregistered gates, not by test results.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .. import bench, c0, ledger
from ..schema import read_jsonl
from . import labels as LB

SEEDS = (1, 2, 3)
VARIANTS = ("all-sources", "no-nemotron")
BENCH = {"all-sources": ["pii_trace", "tab_direct", "spy_legal", "spy_medical", "nemotron"],
         "no-nemotron": ["nemotron"]}
# per-unit estimates (A100 hours); extraction units also project from their first shard and stop early
EST_H = {"train_store": 0.5, "head": 0.1, "bench_store": 0.15, "pred": 0.02, "gliner_c3": 1.0, "level1": 2.0,
         "finetune": 1.2, "flat": 1.5}
MIN_FREE_GB = 25.0


def system(tag: str, variant: str, seed: int) -> str:
    return f"s1v2-{tag}_{variant}-s{seed}"


def bench_system(tag: str, variant: str, seed: int) -> str:
    """Secondary C0': S1 under the benchmark's own label names and descriptions."""
    return f"s1v2bench-{tag}_{variant}-s{seed}"


def c3_system(tag: str, variant: str, seed: int, descriptions: bool = True) -> str:
    return f"s1v2c3{'' if descriptions else 'names'}-{tag}_{variant}-s{seed}"


class Chain:
    def __init__(self, models: Path, work: Path, drive: Path, cap: float, log=print):
        self.models, self.work, self.drive, self.cap, self.log = models, work, drive, cap, log
        for p in (work, drive):
            p.mkdir(parents=True, exist_ok=True)
        self.hours_path = drive / "gpu_hours.jsonl"
        self.state_path = drive / "state.json"
        self.pred_dir = ledger.RESULTS / "predictions_v2"
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------------------------------------------------------------- accounting
    def used(self) -> float:
        if not self.hours_path.exists():
            return 0.0
        return sum(json.loads(l)["hours"] for l in self.hours_path.read_text().splitlines() if l.strip())

    def phase(self, p: str) -> None:
        """GPU or CPU; the monitor applies its idle rule only in GPU phases."""
        (self.drive / "PHASE").write_text(p)

    def unit(self, name: str, kind: str, fn, gpu: bool = True, local: bool = False, stage: str | None = None):
        """Run ``fn`` once. Every unit's wall time counts against the cap (the whole runtime is
        billed), including failed attempts. ``local``: the result is a local store directory;
        if the runtime was lost, the marker is ignored and the store is rebuilt (``extract``
        resumes by shard), unless its stage was pruned on purpose."""
        done = self.drive / "done" / f"{name}.json"
        if done.exists():
            res = json.loads(done.read_text()).get("result")
            pruned = stage is not None and stage in self.state().get("pruned", [])
            if not local or pruned or (res and (Path(res) / "meta.json").exists()):
                return res
            self.log(f"[v2] {name}: local store missing (runtime lost); rebuilding")
        if self.used() + EST_H.get(kind, 0.1) > self.cap:
            raise CapReached(f"{name}: {self.used():.2f} h used + {EST_H.get(kind, 0.1)} h estimate > cap {self.cap} h")
        self.phase("GPU" if gpu else "CPU")
        t0 = time.time()
        self.log(f"[v2] start {name}")
        ok = False
        try:
            res = fn()
            ok = True
        finally:
            h = (time.time() - t0) / 3600
            with open(self.hours_path, "a") as f:
                f.write(json.dumps({"unit": name, "kind": kind, "hours": round(h, 4), "ok": ok, "t": time.time()}) + "\n")
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text(json.dumps({"result": str(res) if res is not None else None, "hours": h}))
        self.log(f"[v2] done {name} in {h * 60:.1f} min (total {self.used():.2f} h of {self.cap} h)")
        return str(res) if res is not None else None

    def progress(self, name: str):
        """Extraction callback: stop a unit whose rate over its shards projects past the cap."""
        base = self.used()
        def cb(done: int, total: int, elapsed: float):
            proj = base + elapsed / done * total / 3600
            if proj > self.cap:
                raise CapReached(f"{name}: projected {proj:.2f} h after {done}/{total} shards > cap {self.cap} h")
        return cb

    def extract(self, unit_name: str, *a, **k):
        from .features import extract
        return extract(*a, progress=self.progress(unit_name), min_free_gb=MIN_FREE_GB, **k)

    def prune(self, tag: str) -> None:
        """Delete a superseded stage's local stores once its predictions are written."""
        for p in (self.work / "stores").glob(f"*-{tag}-*"):
            shutil.rmtree(p, ignore_errors=True)
        st = self.state(); st.setdefault("pruned", [])
        if tag not in st["pruned"]:
            st["pruned"].append(tag)
        self.state_path.write_text(json.dumps(st, indent=2))

    def state(self) -> dict:
        return json.loads(self.state_path.read_text()) if self.state_path.exists() else {"headline": "cheap", "stages": []}

    def set_state(self, **kw) -> None:
        s = self.state(); s.update(kw)
        self.state_path.write_text(json.dumps(s, indent=2))

    # ---------------------------------------------------------------- data
    def training_docs(self, variant: str, seed: int):
        from ..model.train import TrainConfig, training_docs
        man = json.loads((self.models / f"{variant}-s{seed}" / "final" / "s1_manifest.json").read_text())
        c = man["config"]
        cfg = TrainConfig(variant=variant, seed=seed, synth_n=c["synth_n"], synth_seed=c["synth_seed"])
        docs, counts = training_docs(cfg)
        want = man.get("data", {}).get("source_counts")
        if want and want != counts:
            raise ValueError(f"{variant}-s{seed}: rebuilt sources {counts} != trained {want}")
        return docs, counts

    def selected(self, variant: str):
        """Label-balanced training subset, identical across seeds (data constant, as in v1)."""
        from .features import select_training_docs
        cache = self.__dict__.setdefault("_sel", {})
        if variant not in cache:
            docs, _ = self.training_docs(variant, 1)
            cache[variant] = select_training_docs(docs, per_label=3000, max_docs=60000, seed=0)
        return cache[variant]

    # ---------------------------------------------------------------- stores (lazy)
    def done(self, name: str) -> bool:
        return (self.drive / "done" / f"{name}.json").exists()

    def model_paths(self, tag: str, variant: str, seed: int) -> tuple[Path, Path | None]:
        """(level-1 model dir, typing encoder dir or None) for a stage."""
        if tag == "cheap":
            return self.models / f"{variant}-s{seed}" / "final", None
        if tag == "B":
            return self.drive / "level1" / f"{variant}-s{seed}" / "final", None
        if tag == "A":
            base = self.state().get("A_base", "cheap")
            return self.model_paths(base, variant, seed)[0], self.drive / "typing" / f"{variant}-s{seed}"
        raise ValueError(tag)

    def store(self, kind: str, tag: str, variant: str, seed: int, ds: str | None = None, split: str | None = None) -> Path:
        """Return a local feature store, rebuilding it (resumable by shard) only when a consumer
        with pending work asks for it; stores are never rebuilt just because a marker exists."""
        from .features import FeatureConfig, training_extras
        key = f"{tag}-{variant}-s{seed}"
        l1, typing = self.model_paths(tag, variant, seed)
        root = self.work / "stores"
        if kind == "train":
            unit = f"train_store-{key}"
            def fn():
                sel = self.selected(variant)
                ex = training_extras(sel, seed=0, heldout_nodes=LB.heldout_nodes())
                return self.extract(unit, l1, sel, root, name=f"train-{key}", extras=ex, label_texts=LB.all_texts(),
                                    cfg=FeatureConfig(store_min_pb=0.05), typing_encoder=typing)
            kind_ = "train_store"
        else:
            name = f"{ds}-{split}-{key}"
            unit = f"bench_store-{name}"
            docs_path = bench.split_paths(ds)[split]
            def fn():
                return self.extract(unit, l1, list(read_jsonl(docs_path)), root, name=name, label_texts=LB.all_texts(),
                                    typing_encoder=typing)
            kind_ = "bench_store"
        return Path(self.unit(unit, kind_, fn, local=True))

    # ---------------------------------------------------------------- one model
    def build(self, tag: str, variant: str, seed: int, l1_dir: Path | None = None, typing: Path | None = None) -> dict:
        from .head import HeadConfig, train_head
        from . import infer
        held = LB.heldout_nodes()
        key = f"{tag}-{variant}-s{seed}"
        _, typing = self.model_paths(tag, variant, seed)
        head_dir = self.drive / "heads" / key
        if typing is None:
            if not self.done(f"head-{key}"):
                ts = self.store("train", tag, variant, seed)
                zf = ts / "no_negative_zones.json"
                if not zf.exists():
                    zf.write_text(json.dumps(self.zones(variant)))
                self.unit(f"head-{key}", "head", lambda: train_head(ts, head_dir, held, HeadConfig(seed=seed), log=self.log))
        elif not head_dir.exists():
            shutil.copytree(typing / "head", head_dir)
        for ds in BENCH[variant]:
            for split in ("calib", "test"):
                docs_path = bench.split_paths(ds)[split]
                name = f"{ds}-{split}-{key}"
                jobs = []
                if ds == "nemotron" and variant == "no-nemotron" or ds != "nemotron":
                    # primary: the canonical strings every baseline received; secondary: benchmark labels
                    jobs.append((f"pred-{name}", LB.canonical_label_set(), system(tag, variant, seed), "sensitive"))
                    jobs.append((f"predbench-{name}", LB.load_benchmark_label_set(ds), bench_system(tag, variant, seed), "sensitive"))
                if ds == "nemotron":            # C3: primary no-nemotron, secondary all-sources; calib for thresholds
                    for desc in (True, False):
                        jobs.append((f"predc3-{name}-{desc}", LB.c3_label_set(desc), c3_system(tag, variant, seed, desc), "typed"))
                pending = [j for j in jobs if not self.done(j[0])]
                if not pending and not (ds == "nemotron" and split == "calib" and not (self.drive / f"gates-{tag}.json").exists()):
                    continue
                st = self.store("bench", tag, variant, seed, ds, split)
                for uname, L, sysn, score in pending:
                    self.unit(uname, "pred", lambda: infer.write(st, head_dir, L, docs_path, self.pred_dir, sysn,
                                                                 score=score, device=self.device))
        return {"head": str(head_dir)}

    # ---------------------------------------------------------------- stages
    def cheap(self):
        LB.load_heldout()
        res = {}
        for v in VARIANTS:
            for sd in SEEDS:
                res[f"{v}-s{sd}"] = self.build("cheap", v, sd, self.models / f"{v}-s{sd}" / "final")
        (self.drive / "cheap.json").write_text(json.dumps(res, indent=2))
        return res

    def gliner_c3(self, venv: Path):
        def run(split: str, names: bool):
            docs = bench.split_paths("nemotron")[split]
            cmd = [str(venv / "bin" / "python"), "-m", "s1pii.v2.gliner_c3", "--system", "gliner25_base_zeroshot",
                   "--docs", str(docs), "--out", str(self.pred_dir)] + (["--names-only"] if names else [])
            subprocess.run(cmd, check=True, env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])})
        for split in ("test", "calib"):
            self.unit(f"gliner_c3-{split}-desc", "gliner_c3", lambda: run(split, False))
        self.unit("gliner_c3-test-names", "gliner_c3", lambda: run("test", True))

    # ---------------------------------------------------------------- teacher (CPU)
    def teacher_path(self, variant: str) -> Path:
        return self.drive / f"teacher-{variant}.json"

    def teacher(self, variant: str) -> dict:
        """spaCy teacher spans over the variant's training docs, only for sources that do not
        annotate every teacher family (never Nemotron). CPU; launched in the background at the
        start of the chain so the GPU does not wait for it."""
        from .level1 import teacher_for
        p = self.teacher_path(variant)
        if not p.exists():
            try:
                docs, _ = self.training_docs(variant, 1)
                tmp = p.with_suffix(".part")
                tmp.write_text(json.dumps(teacher_for(docs, self._sources(variant))))
                os.replace(tmp, p)
            except BaseException:
                import traceback
                p.with_suffix(".failed").write_text(traceback.format_exc())
                raise
        return {k: [tuple(x) for x in v] for k, v in json.loads(p.read_text()).items()}

    def wait_teacher(self, variant: str, timeout: float = 1800) -> dict:
        """Wait for the background teacher; fail fast on its .failed file or a dead process.
        With no background process (pid file absent) the teacher runs here."""
        p = self.teacher_path(variant)
        pidf = self.drive / "teacher.pid"
        t0 = time.time()
        while not p.exists():
            if p.with_suffix(".failed").exists():
                raise RuntimeError(f"teacher failed:\n{p.with_suffix('.failed').read_text()[-2000:]}")
            if not pidf.exists():
                return self.teacher(variant)
            try:
                os.kill(int(pidf.read_text().strip()), 0)
            except (OSError, ValueError):
                if not p.exists():
                    raise RuntimeError("teacher process ended without writing its output")
            if time.time() - t0 > timeout:
                raise TimeoutError(f"teacher for {variant} not ready after {timeout} s")
            self.phase("CPU")
            time.sleep(10)
        return self.teacher(variant)

    def zones(self, variant: str) -> dict:
        """doc_id -> teacher spans of families its source does not annotate (no hard negatives)."""
        from .level1 import any_mask_spans, source_inventory
        cache = self.__dict__.setdefault("_zones", {})
        if variant not in cache:
            docs, _ = self.training_docs(variant, 1)
            src = self._sources(variant)
            inv = source_inventory(docs, src)
            t = self.wait_teacher(variant)
            sel = {d.doc_id for d in self.selected(variant)}
            cache[variant] = {d.doc_id: z for d in docs if d.doc_id in sel
                              for z in [any_mask_spans(d, t.get(d.doc_id, []), inv.get(src[d.doc_id], set()))] if z}
        return cache[variant]

    def gates(self, tag: str | None = None, reuse: bool = False):
        """Evaluate (or, with ``reuse``, load the frozen) gates for a stage."""
        from .gates import evaluate_gates
        tag = tag or self.state()["headline"]
        gp = self.drive / f"gates-{tag}.json"
        if reuse and gp.exists():
            return json.loads(gp.read_text())
        heads = [self.drive / "heads" / f"{tag}-{v}-s{sd}" for v in VARIANTS for sd in SEEDS]
        calib = bench.split_paths("nemotron")["calib"]
        stores = {f"{v}-s{sd}": (self.store("bench", tag, v, sd, "nemotron", "calib"), self.drive / "heads" / f"{tag}-{v}-s{sd}")
                  for v in VARIANTS for sd in SEEDS}
        g = evaluate_gates(heads, stores, calib)
        (self.drive / f"gates-{tag}.json").write_text(json.dumps(g, indent=2, default=str))
        ledger.append({"kind": "v2_gates", "tag": tag, **g})
        self.log(json.dumps({k: g[k] for k in ("head_val_acc_min", "heldout_typing_mean", "level1_recall_mean", "A_fires", "B_fires")}))
        return g

    def _store(self, name: str) -> Path:
        hits = sorted((self.work / "stores").glob(f"{name}-*"))
        hits = [h for h in hits if (h / "meta.json").exists()]
        if len(hits) != 1:
            raise FileNotFoundError(f"expected one store for {name}, found {hits}")
        return hits[0]

    def _gate_hash(self, tag: str) -> str:
        import hashlib
        return hashlib.sha256((self.drive / f"gates-{tag}.json").read_bytes()).hexdigest()

    def conditional(self):
        st = self.state()
        self.phase("CPU")
        g = self.gates("cheap", reuse=True)
        self.set_state(gates={**self.state().get("gates", {}), "cheap": self._gate_hash("cheap")})
        if g["B_fires"] and "B" not in st["stages"]:
            try:
                for v in VARIANTS:
                    docs, _ = self.training_docs(v, 1)
                    src = self._sources(v)
                    teacher = self.wait_teacher(v)
                    for sd in SEEDS:
                        def run_l1():
                            from .level1 import train_level1
                            local = self.work / "level1" / f"{v}-s{sd}"
                            fin = train_level1(self.models / f"{v}-s{sd}" / "final", docs, src, local, LB.heldout_nodes(),
                                               seed=sd, mirror=self.drive / "level1_ckpt" / f"{v}-s{sd}", teacher=teacher)
                            dst = self.drive / "level1" / f"{v}-s{sd}" / "final"
                            shutil.rmtree(dst, ignore_errors=True); dst.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copytree(fin, dst)
                            return dst
                        self.unit(f"level1-{v}-s{sd}", "level1", run_l1)
                for v in VARIANTS:
                    for sd in SEEDS:
                        self.build("B", v, sd)
                self.set_state(headline="B", stages=st["stages"] + ["B"])
                self.prune("cheap")
                g = self.gates("B", reuse=True)
                self.set_state(gates={**self.state().get("gates", {}), "B": self._gate_hash("B")})
            except CapReached as e:
                self.log(f"[v2] stage B stopped: {e}")
                self.set_state(stage_B_stopped=str(e))
        st = self.state()
        base = st["headline"]
        if g["A_fires"] and "A" not in st["stages"]:
            try:
                from .finetune import finetune
                self.set_state(A_base=base)
                for v in VARIANTS:
                    for sd in SEEDS:
                        key = f"{base}-{v}-s{sd}"
                        out = self.drive / "typing" / f"{v}-s{sd}"
                        if not self.done(f"finetune-{v}-s{sd}"):
                            l1 = self.model_paths(base, v, sd)[0]
                            ts = self.store("train", base, v, sd)
                            sel = self.selected(v)
                            self.unit(f"finetune-{v}-s{sd}", "finetune",
                                      lambda: finetune(l1, ts, self.drive / "heads" / key, sel, out, LB.heldout_nodes(), log=self.log))
                        self.build("A", v, sd)
                self.set_state(headline="A", stages=self.state()["stages"] + ["A"])
                self.prune(base)
            except CapReached as e:
                self.log(f"[v2] stage A stopped: {e}")
                self.set_state(stage_A_stopped=str(e))
        try:
            self.flat()
        except CapReached as e:
            self.log(f"[v2] ablation C skipped: {e}")
            self.set_state(stage_C_skipped=str(e))

    def _sources(self, variant: str) -> dict[str, str]:
        from ..data import loaders as L
        from ..data.synth import generate
        from ..model.train import TrainConfig
        man = json.loads((self.models / f"{variant}-s1" / "final" / "s1_manifest.json").read_text())
        c = man["config"]
        src = {d.doc_id: "synthetic_conv" for d in generate(c["synth_n"], seed=c["synth_seed"])}
        src.update({d.doc_id: "gretel" for d in bench.dev_slice(L.load("gretel", "train", purpose="train"))[1]})
        if variant == "all-sources":
            src.update({d.doc_id: "nemotron" for d in bench.dev_slice(L.load("nemotron", "train", purpose="train"))[1]})
        return src

    def ablations(self):
        """Cheap ablations on cached features, seed 1 of each variant: names-only label texts,
        and no NONE supervision from boundary shifts or hard negatives."""
        from .head import HeadConfig, train_head
        from . import infer
        held = LB.heldout_nodes()
        variants = {"names": dict(p_name_only=1.0, p_paraphrase=0.0), "nonone": dict(hard_neg_ratio=0.0, shift_ratio=0.0)}
        for v in VARIANTS:
            for ab, kw in variants.items():
                hd = self.drive / "heads" / f"abl{ab}-{v}-s1"
                if not self.done(f"head-abl{ab}-{v}"):
                    ts = self.store("train", "cheap", v, 1)
                    self.unit(f"head-abl{ab}-{v}", "head", lambda: train_head(ts, hd, held, HeadConfig(seed=1, **kw), log=self.log))
                for ds in BENCH[v]:
                    if (ds == "nemotron") != (v == "no-nemotron"):
                        continue
                    for split in ("calib", "test"):
                        uname = f"pred-abl{ab}-{ds}-{split}-{v}"
                        if self.done(uname):
                            continue
                        st = self.store("bench", "cheap", v, 1, ds, split)
                        L = LB.canonical_label_set()          # as the headline C0' system
                        if ab == "names":
                            L = LB.LabelSet(L.name, L.labels, descriptions=False)
                        self.unit(uname, "pred", lambda: infer.write(st, hd, L, bench.split_paths(ds)[split], self.pred_dir,
                                                                     f"s1v2abl{ab}_{v}-s1", device=self.device))

    def sweeps(self):
        """Preregistered sweeps on the headline stage's seed-1 test stores (rebuilt if lost)."""
        from . import evaluate as EV
        os.environ["S1PII_V2_STATE"] = str(self.state_path)
        tag = self.state()["headline"]
        for ds in EV.C4_SETS:
            self.store("bench", tag, c0.variant_for(ds), 1, ds, "test")
        self.phase("CPU")
        res = EV.sweeps(self.work, self.drive)
        bad = {k: v for k, v in res.items() if "problem" in v}
        if bad:
            raise RuntimeError(f"sweeps incomplete: {bad}")
        return res

    def flat(self, frac: float = 0.3):
        """Ablation C, seed 1 of each variant, on a fixed training subset."""
        from .flat import train_flat, predict_flat
        from ..ledger import write_predictions, dataset_hash
        import hashlib
        for v in VARIANTS:
            docs, _ = self.training_docs(v, 1)
            keep = [d for d in docs if int(hashlib.sha256(f"flat:{d.doc_id}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac]
            out = self.drive / "flat" / f"{v}-s1"
            v1 = self.models / f"{v}-s1" / "final"
            self.unit(f"flat-{v}", "flat", lambda: train_flat(v1, keep, out, LB.heldout_nodes(), log=self.log))
            for ds in BENCH[v]:
                if (ds == "nemotron") != (v == "no-nemotron"):
                    continue
                for split in ("calib", "test"):
                    p = bench.split_paths(ds)[split]
                    def run():
                        docs_ = sorted(read_jsonl(p), key=lambda d: d.doc_id)
                        L = LB.load_benchmark_label_set(ds)
                        preds = predict_flat(v1, out, docs_, L, system=f"s1v2flat_{v}-s1")
                        path = self.pred_dir / f"flat-{ds}-{split}-{v}.jsonl"
                        write_predictions(preds, path, {"system": f"s1v2flat_{v}-s1", "docs_path": str(p),
                                                        "dataset_hash": dataset_hash(docs_), "config": {"labels_hash": L.hash()},
                                                        "versions": {"variant": v, "seed": 1}})
                        return path
                    self.unit(f"flatpred-{ds}-{split}-{v}", "bench_store", run)


class CapReached(RuntimeError):
    pass


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["teacher", "cheap", "gliner_c3", "ablations", "gates", "conditional", "sweeps", "all"])
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--drive", type=Path, required=True)
    ap.add_argument("--cap", type=float, default=32.0)
    ap.add_argument("--gliner-venv", type=Path, default=Path("envs/gliner2"))
    a = ap.parse_args(argv)
    ch = Chain(a.models, a.work, a.drive, a.cap)
    try:
        if a.stage == "teacher":                     # CPU; the chain script runs this in the background
            for v in VARIANTS:
                ch.teacher(v)
            return
        if a.stage in ("cheap", "all"):
            ch.cheap()
        if a.stage in ("gliner_c3", "all"):
            ch.gliner_c3(a.gliner_venv)
        if a.stage in ("ablations", "all"):
            ch.ablations()
        if a.stage == "gates":
            ch.gates()
        if a.stage in ("conditional", "all"):
            ch.conditional()
        if a.stage in ("sweeps", "all"):
            ch.sweeps()
    except CapReached as e:
        print(f"[v2] STOPPED: {e}", flush=True)
        sys.exit(3)
    print(f"[v2] finished stage {a.stage}; GPU hours used {ch.used():.2f} of {a.cap}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
