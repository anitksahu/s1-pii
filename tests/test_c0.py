import json

import pytest

from s1pii.schema import Doc, Span, PERSON, EMAIL, write_jsonl
from s1pii import bench, c0, ledger
from s1pii.data import loaders as L
from s1pii.ledger import write_predictions, dataset_hash


def mkdocs(n, prefix):
    out = []
    for i in range(n):
        t = f"{prefix} note {i}: Alice{i} wrote to a{i}@x.org about item {i}."
        a = t.index("Alice"); e = t.index("a" + str(i) + "@")
        out.append(Doc(f"{prefix}{i:03d}", t, (Span(f"{prefix}{i:03d}", a, a + len(f"Alice{i}"), PERSON),
                                               Span(f"{prefix}{i:03d}", e, e + len(f"a{i}@x.org"), EMAIL)),
                       cluster_id=f"{prefix}c{i}"))
    return out


def test_c0_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(ledger, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(c0, "HEADLINE", ["pii_trace", "spy_legal"])
    monkeypatch.setattr(ledger, "git_sha", lambda: "a" * 40)
    pred_dir = tmp_path / "results" / "predictions"
    for ds in c0.HEADLINE:
        docs = mkdocs(40, ds[:3])
        cal, test = L.calib_test_split(docs, 0.2)
        p = bench.split_paths(ds); p["test"].parent.mkdir(parents=True, exist_ok=True)
        write_jsonl(cal, p["calib"]); write_jsonl(test, p["test"])
        # audit against unrelated training docs (nothing flagged) + one exact duplicate
        monkeypatch.setattr(c0, "training_sources", lambda v, **kw: [test[0]] + mkdocs(5, "zz"))
        c0.run_audit([ds])
        rep = json.loads(c0.audit_path(ds).read_text())
        assert rep["n_flagged"] >= 1
        def write(system, spans_fn, score):
            preds = {d.doc_id: [Span(d.doc_id, s.start, s.end, s.label_canonical, score=score, source=system)
                                for s in spans_fn(d)] for d in test}
            write_predictions(preds, pred_dir / f"{system}-{ds}.jsonl",
                              {"system": system, "docs_path": str(p["test"]), "dataset_hash": dataset_hash(test)})
        for seed in (1, 2, 3):
            write(f"s1_all-sources-s{seed}", lambda d: d.spans, 0.9)
        for b in c0.BASELINES:
            write(b, lambda d: d.spans[:1], 0.9)       # misses every email
    out = c0.decide(n_boot=300)
    assert out["missing"] == []
    assert out["decision"]["family_size"] == 6
    assert out["decision"]["datasets_won"] == ["pii_trace", "spy_legal"]
    from s1pii.schema import read_jsonl
    assert len(c0.audited_test_docs("pii_trace")) < len(read_jsonl(bench.split_paths("pii_trace")["test"]))


def test_headline_score_refuses_dirty_code(tmp_path, monkeypatch):
    monkeypatch.setattr(ledger, "RESULTS", tmp_path)
    monkeypatch.setattr(ledger, "git_sha", lambda: "abc-dirty")
    with pytest.raises(ledger.UnknownProvenance):
        ledger.append({"x": 1}, headline=True)
