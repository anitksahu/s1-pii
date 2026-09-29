import json

import pytest

from s1pii.schema import Doc, Span, PERSON, EMAIL, write_jsonl, read_jsonl
from s1pii import bench, c0, ledger
from s1pii.data import loaders as L
from s1pii.ledger import write_predictions, dataset_hash


def mkdocs(n, prefix):
    out = []
    for i in range(n):
        t = f"{prefix} note {i}: Alice{i} wrote to a{i}@x.org about item {i}."
        a = t.index("Alice"); e = t.index(f"a{i}@")
        out.append(Doc(f"{prefix}{i:03d}", t, (Span(f"{prefix}{i:03d}", a, a + len(f"Alice{i}"), PERSON),
                                               Span(f"{prefix}{i:03d}", e, e + len(f"a{i}@x.org"), EMAIL)),
                       cluster_id=f"{prefix}c{i}"))
    return out


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(ledger, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(ledger, "git_sha", lambda: "a" * 40)
    monkeypatch.setattr(c0, "HEADLINE", ["pii_trace", "spy_legal"])
    from s1pii.adapters import base as AB
    cfg = AB.load_config()
    pinned = {**cfg, "systems": {k: {**v, "revision": "rev1"} for k, v in cfg["systems"].items()}}
    monkeypatch.setattr(AB, "load_config", lambda name="baselines.yaml": pinned if name == "baselines.yaml" else cfg)
    models = tmp_path / "models"
    for v in ("all-sources", "no-nemotron"):
        (models / f"{v}-s1" / "final").mkdir(parents=True)
        (models / f"{v}-s1" / "final" / "s1_manifest.json").write_text(json.dumps({"config": {"variant": v}, "data": {"source_counts": {"k": 1}}}))
    pred_dir = tmp_path / "results" / "predictions"
    tests = {}
    for ds in c0.HEADLINE:
        cal, test = L.calib_test_split(mkdocs(40, ds[:3]), 0.2)
        p = bench.split_paths(ds); p["test"].parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(cal, p["calib"]); write_jsonl(test, p["test"])
        tests[ds] = test
    monkeypatch.setattr(c0, "training_sources_from_manifest",
                        lambda man: [tests["pii_trace"][0]] + mkdocs(5, "zz"))
    c0.run_audit(models)
    c0.c2_path().parent.mkdir(parents=True, exist_ok=True)
    c0.c2_path().write_text(json.dumps({"propagation": True}))

    def write(system, ds, spans_fn, variant=None, seed=None, fname=None, test=None):
        test = test if test is not None else tests[ds]
        preds = {d.doc_id: [Span(d.doc_id, s.start, s.end, s.label_canonical, score=0.9, source=system)
                            for s in spans_fn(d)] for d in test}
        meta = {"system": system, "docs_path": str(bench.split_paths(ds)["test"]), "dataset_hash": dataset_hash(tests[ds]),
                "revision": "rev1",
                "versions": {"variant": variant, "seed": seed},
                "config": {"validators": True, "propagation": True, "floor": 0.01, "o_exit_bias": 0.0}}
        write_predictions(preds, pred_dir / (fname or f"{system}-{ds}.jsonl"), meta)

    def full():
        for ds in c0.HEADLINE:
            for sd in (1, 2, 3):
                write(c0.s1_system("all-sources", sd), ds, lambda d: d.spans, "all-sources", sd)
            for b in c0.BASELINES:
                write(b, ds, lambda d: d.spans[:1])
    return {"write": write, "full": full, "tests": tests, "pred_dir": pred_dir}


def test_complete_family_decides(world):
    world["full"]()
    out = c0.decide(n_boot=2000, fresh_real_absent=True)
    assert out["problems"] == [] and out["expected_family"] == 6
    assert out["decision"]["datasets_won"] == ["pii_trace", "spy_legal"]
    assert len(c0.audited_test_docs("pii_trace")) < len(read_jsonl(bench.split_paths("pii_trace")["test"]))


def test_missing_baseline_voids_c0(world):
    world["full"]()
    for p in world["pred_dir"].glob("gliner2_pii-spy_legal.jsonl"):
        p.unlink()
    out = c0.decide(n_boot=100, fresh_real_absent=True)
    assert out["decision"]["c0_holds"] is None and any("missing gliner2_pii" in x for x in out["problems"])


def test_extra_seed_and_ablation_ignored_wrong_config_flagged(world):
    world["full"]()
    world["write"]("s1_all-sources-s4", "pii_trace", lambda d: d.spans, "all-sources", 4)
    world["write"]("s1_all-sources-s1-noval", "pii_trace", lambda d: [], "all-sources", 1)
    assert c0.decide(n_boot=100, fresh_real_absent=True)["problems"] == []
    world["write"](c0.s1_system("all-sources", 2), "spy_legal", lambda d: d.spans, "no-nemotron", 2)
    out = c0.decide(n_boot=100, fresh_real_absent=True)
    assert out["decision"]["c0_holds"] is None and any("variant/seed" in x for x in out["problems"])


def test_duplicates_and_stale_files_rejected(world):
    world["full"]()
    world["write"]("gliner2_pii", "pii_trace", lambda d: d.spans[:1], fname="gliner2_pii-pii_trace-old.jsonl")
    with pytest.raises(c0.DuplicatePredictions):
        c0.decide(n_boot=50, fresh_real_absent=True)
    (world["pred_dir"] / "gliner2_pii-pii_trace-old.jsonl").unlink()
    p = world["pred_dir"] / "gliner2_pii-spy_legal.jsonl"
    meta, preds = ledger.read_predictions(p)
    meta["dataset_hash"] = "stale"
    write_predictions(preds, p, meta)
    with pytest.raises(ValueError, match="different test docs"):
        c0.decide(n_boot=50, fresh_real_absent=True)


def test_missing_c2_voids_c0(world):
    world["full"]()
    c0.c2_path().unlink()
    assert c0.decide(n_boot=50, fresh_real_absent=True)["decision"]["c0_holds"] is None


def test_headline_ledger_refuses_dirty_code(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "RESULTS", tmp_path)
    monkeypatch.setattr(ledger, "git_sha", lambda: "abc-dirty")
    with pytest.raises(ledger.UnknownProvenance):
        ledger.append({"x": 1}, headline=True)


def test_fresh_real_fallback_refused_when_split_exists(world, monkeypatch):
    world["full"]()
    monkeypatch.setattr(c0, "HEADLINE", c0.HEADLINE + ["fresh_real"])
    p = bench.split_paths("fresh_real")["test"]; p.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(mkdocs(3, "fr"), p)
    out = c0.decide(n_boot=50, fresh_real_absent=True)
    assert out["decision"]["c0_holds"] is None and any("fresh-real-absent" in x for x in out["problems"])


def test_manifest_without_source_counts_rejected(monkeypatch):
    monkeypatch.setattr(c0, "generate", lambda n, seed: [])
    monkeypatch.setattr(c0.L, "load", lambda *a, **k: [])
    with pytest.raises(ValueError, match="source_counts"):
        c0.training_sources_from_manifest({"config": {"variant": "no-nemotron", "synth_n": 0, "synth_seed": 1}})
