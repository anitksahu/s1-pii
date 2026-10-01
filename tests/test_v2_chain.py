"""End-to-end dry run of the v2 GPU chain on CPU with tiny models and synthetic splits:
cheap path, ablations, gates, and the conditional stages B, A and C, including resume."""
import functools
import json
from dataclasses import dataclass

import pytest
import yaml

torch = pytest.importorskip("torch")

from s1pii.data.synth import generate
from s1pii.schema import write_jsonl
from s1pii.model.encode import train_examples
from s1pii.model.train import TrainConfig, train
from s1pii import bench, ledger
from s1pii.v2 import labels as LB, run_v2, features as FT, finetune as FTN, flat as FL, level1 as L1
from tests.test_model import tok, tiny_model  # noqa: F401


@pytest.fixture()
def setup(tok, tmp_path, monkeypatch):
    models = tmp_path / "models"
    docs = generate(60, seed=13)
    ex = train_examples(docs, tok, max_len=128)
    for v in run_v2.VARIANTS:
        for sd in run_v2.SEEDS:
            cfg = TrainConfig(backbone="tiny", seed=sd, variant=v, token_budget=4096, grad_accum=1, lr_encoder=5e-3,
                              lr_head=5e-2, ckpt_every=1000, gradient_checkpointing=False, bf16=False, synth_n=60,
                              synth_seed=13)
            train(cfg, models / f"{v}-s{sd}", model=tiny_model(tok, sd), tokenizer=tok, examples=ex, max_steps=4,
                  device="cpu", resume=False, data_manifest={"source_counts": {"synthetic_conv": 60}})
            (models / f"{v}-s{sd}" / "run").mkdir(exist_ok=True)
    # splits
    split_dir = tmp_path / "splits"; split_dir.mkdir()
    def split_paths(name):
        return {p: split_dir / f"{name}-{p}.jsonl" for p in ("calib", "test")}
    for i, ds in enumerate(["pii_trace", "tab_direct", "spy_legal", "spy_medical", "nemotron"]):
        for j, p in enumerate(("calib", "test")):
            write_jsonl(generate(6, seed=100 + 10 * i + j), split_paths(ds)[p])
    monkeypatch.setattr(bench, "split_paths", split_paths)
    held = tmp_path / "heldout.yaml"
    held.write_text(yaml.safe_dump({"labels": ["url"], "nodes": ["url"],
                                    "synonyms_all_sources": ["url", "private_url"]}))
    monkeypatch.setattr(LB, "heldout_path", lambda: held)
    monkeypatch.setattr(ledger, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(run_v2.Chain, "training_docs", lambda self, v, sd: (generate(60, seed=13), {"synthetic_conv": 60}))
    monkeypatch.setattr(run_v2, "MIN_FREE_GB", 0.0)
    monkeypatch.setattr(run_v2.Chain, "wait_teacher", lambda self, v, timeout=0: {})
    monkeypatch.setattr(run_v2.Chain, "_sources", lambda self, v: {d.doc_id: "synthetic_conv" for d in generate(60, seed=13)})

    @dataclass
    class SmallFC(FT.FeatureConfig):
        max_len: int = 128
    monkeypatch.setattr(FT, "FeatureConfig", SmallFC)
    orig_extract = FT.extract
    monkeypatch.setattr(FT, "extract", lambda *a, cfg=None, **k: orig_extract(*a, cfg=cfg or SmallFC(), **k))
    orig_ft = FTN.FinetuneConfig
    monkeypatch.setattr(FTN, "FinetuneConfig", lambda: orig_ft(n_layers=1, context=64, batch=16, max_steps=2))
    monkeypatch.setattr(FL, "train_flat", functools.partial(FL.train_flat, steps=2, batch=4, max_len=64, device="cpu"))
    monkeypatch.setattr(FL, "predict_flat", functools.partial(FL.predict_flat, max_len=128, device="cpu"))
    monkeypatch.setattr(L1, "teacher_for", lambda docs, src, inv=None: {})
    monkeypatch.setattr(L1, "train_level1", functools.partial(L1.train_level1, max_steps=2, max_len=128,
                                                              device="cpu", gradient_checkpointing=False, bf16=False,
                                                              token_budget=4096, grad_accum=1))
    return models, tmp_path


def test_chain_cheap_ablations_gates_conditional_and_resume(setup):
    models, tmp = setup
    ch = run_v2.Chain(models, tmp / "work", tmp / "drive", cap=100, log=lambda *_: None)
    ch.cheap()
    preds = sorted((ledger.RESULTS / "predictions_v2").glob("*.jsonl"))
    systems = {json.loads(p.read_text().splitlines()[0])["_meta"]["system"] for p in preds}
    assert {"s1v2-cheap_all-sources-s1", "s1v2-cheap_no-nemotron-s3", "s1v2c3-cheap_all-sources-s2",
            "s1v2c3names-cheap_all-sources-s1"} <= systems
    n_units = len(list((tmp / "drive" / "done").glob("*.json")))
    ch.cheap()                                                 # resume: nothing recomputed
    assert len(list((tmp / "drive" / "done").glob("*.json"))) == n_units
    import shutil
    shutil.rmtree(tmp / "work" / "stores")                     # runtime lost: local stores gone
    ch.cheap()
    assert len([p for p in (tmp / "work" / "stores").iterdir() if (p / "meta.json").exists()]) > 0
    ch.ablations()
    g = ch.gates("cheap")
    assert set(g) >= {"A_fires", "B_fires", "level1_recall_mean"}
    ch.conditional()
    ch.flat()
    st = ch.state()
    assert "cheap" in st["gates"]
    if g["B_fires"]:
        assert "B" in st["stages"] and "B" in st["gates"]
    systems = {json.loads(p.read_text().splitlines()[0])["_meta"]["system"] for p in (ledger.RESULTS / "predictions_v2").glob("*.jsonl")}
    assert "s1v2flat_all-sources-s1" in systems
    # CPU evaluation on the cheap path (baselines absent: listed as problems, no crash)
    from s1pii.v2 import evaluate as EV
    from s1pii import c0
    from s1pii.schema import read_jsonl
    import os
    os.environ["S1PII_V2_STATE"] = str(tmp / "drive" / "state.json")
    c0_aud = lambda ds: read_jsonl(bench.split_paths(ds)["test"])
    import pytest as _p
    mp = _p.MonkeyPatch()
    mp.setattr(c0, "audited_test_docs", c0_aud)
    mp.setattr(EV, "V2_PRED", ledger.RESULTS / "predictions_v2")
    sw = EV.sweeps(tmp / "work", tmp / "drive")
    assert all("size" in v for v in sw.values()), sw
    ab = EV.ablation_table()
    t = ab["table"]
    assert t["pii_trace"]["cheap"] is not None and t["pii_trace"]["names_only"] is not None
    assert t["pii_trace"]["headline"] is not None
    # flat rows only with a verified model hash; both label sets are predicted
    assert t["pii_trace"]["flat_bench"] is not None and t["pii_trace"]["flat_canonical"] is not None
    assert not ab["unverified_flat_files"]
    r4 = EV.c4(n_boot=50)
    assert r4["problems"] and r4["c4_holds"] is None
    mp.setattr(LB, "load_heldout", lambda: {"labels": ["url"], "nodes": ["url"], "synonyms_all_sources": ["url"]})
    r3 = EV.c3(None, n_boot=50)
    assert r3["s1v2"][0]["micro_f1"] >= 0 and r3["c3_holds"] is None and r3["primary"]
    assert r3["s1v2_tuned"][0] is not None and r3["s1v2_gold_typing"][0]["n"] >= 0
    r3b = EV.c3(None, n_boot=50, variant="all-sources")
    assert r3b["c3_holds"] is None and "caveat" in r3b
    mp.undo()
    hours = [json.loads(l) for l in (tmp / "drive" / "gpu_hours.jsonl").read_text().splitlines()]
    assert all(h["hours"] >= 0 for h in hours)
    if "B" in st["stages"]:
        assert "cheap" in st["pruned"] and not list((tmp / "work" / "stores").glob("*-cheap-*"))
    # runtime lost after the conditional stages: the full chain resumes without crashing and
    # rebuilds only the stores that pending work needs (here: the headline sweep stores)
    shutil.rmtree(tmp / "work" / "stores")
    n_hours = len((tmp / "drive" / "gpu_hours.jsonl").read_text().splitlines())
    ch2 = run_v2.Chain(models, tmp / "work", tmp / "drive", cap=100, log=lambda *_: None)
    ch2.cheap(); ch2.ablations(); ch2.conditional()
    assert len((tmp / "drive" / "gpu_hours.jsonl").read_text().splitlines()) == n_hours   # nothing recomputed
    sw = ch2.sweeps()
    assert set(sw) == {"tab_direct", "spy_medical", "spy_legal", "pii_trace", "nemotron"}
    rebuilt = [p.name for p in (tmp / "work" / "stores").iterdir()]
    assert all("-test-" in n and "-s1-" in n for n in rebuilt), rebuilt


def test_cap_stops_before_a_unit(setup):
    models, tmp = setup
    ch = run_v2.Chain(models, tmp / "work", tmp / "drive", cap=0.0, log=lambda *_: None)
    with pytest.raises(run_v2.CapReached):
        ch.cheap()
