import re

import pytest

from s1pii.schema import Doc, Span, PERSON, EMAIL, write_jsonl
from s1pii.adapters.base import (Adapter, windows, normalize_gliner, normalize_gliner2, RawEnt,
                                 query_labels, label_descriptions, load_config)
from s1pii.adapters import run as R
from s1pii.ledger import read_predictions


class FakeBackend:
    """Finds 'Alice' (person) and emails by regex; can drift offsets by +1 to exercise re-anchoring."""
    revision = "fake-rev"
    versions = {"fake": "1"}

    def __init__(self, drift=0, label_person="person", extra=None):
        self.drift, self.lp, self.extra, self.calls = drift, label_person, extra or [], 0

    def predict(self, texts, labels, threshold):
        self.calls += 1
        out = []
        for t in texts:
            ents = [RawEnt(m.start() + self.drift, m.end() + self.drift, self.lp, 0.9, m.group())
                    for m in re.finditer(r"Alice", t)]
            ents += [RawEnt(m.start(), m.end(), "email", 0.8, m.group()) for m in re.finditer(r"\S+@\S+\.\w+", t)]
            ents += self.extra
            out.append([e for e in ents if e.score >= threshold])
        return out


def test_windows_cover_and_overlap():
    text = " ".join(f"w{i}" for i in range(450))
    ws = windows(text, 200, 100)
    assert ws[0][0] == 0 and ws[-1][1] == len(text)
    covered = [False] * len(text)
    for a, b in ws:
        for i in range(a, b):
            covered[i] = True
    assert all(covered)
    assert windows("short text", 200, 100) == [(0, 10)]


def test_normalizers():
    g = normalize_gliner([{"start": 0, "end": 3, "text": "Bob", "label": "first_name", "score": 0.7}])
    assert (g[0].start, g[0].label, g[0].score) == (0, "first_name", 0.7)
    g2 = normalize_gliner2({"entities": {"email": [{"text": "a@b.c", "confidence": 0.9, "start": 4, "end": 9}]}})
    assert (g2[0].start, g2[0].end, g2[0].label) == (4, 9, "email")
    with pytest.raises(ValueError):
        normalize_gliner2({"entities": {"email": ["a@b.c"]}})


def test_config_labels_are_canonical():
    for sysname in load_config()["systems"]:
        assert query_labels(sysname)
    assert label_descriptions("gliner25_base_zeroshot")
    assert label_descriptions("gliner2_pii") is None


def test_adapter_windowed_stitch_and_reanchor():
    filler = " ".join(["lorem"] * 150)
    text = f"Alice wrote to x@y.org. {filler} Alice again {filler} end"
    d = Doc("d1", text)
    ad = Adapter("gliner2_pii", FakeBackend(drift=1), words=100, stride=50)
    preds, rep = ad.predict_docs([d], batch_size=4)
    ps = preds["d1"]
    assert [text[p.start:p.end] for p in ps if p.label_canonical == PERSON] == ["Alice", "Alice"]
    assert any(p.label_canonical == EMAIL and text[p.start:p.end] == "x@y.org" for p in ps)
    assert rep.repaired >= 2 and rep.windows > 1
    assert len(ps) == len({(p.start, p.end, p.label_canonical) for p in ps})   # stitched


def test_adapter_drops_unknown_labels_and_bad_offsets():
    d = Doc("d1", "Alice here")
    fb = FakeBackend(extra=[RawEnt(0, 5, "martian", 0.9, "Alice"), RawEnt(3, 99, "email", 0.9, None)])
    preds, rep = Adapter("gliner2_pii", fb).predict_docs([d])
    assert rep.dropped_unknown_label == 1 and rep.unknown_labels == {"martian": 1}
    assert rep.dropped_bad_offsets == 1
    assert [p.label_canonical for p in preds["d1"]] == [PERSON]


def test_runner_sharded_resume(tmp_path):
    docs = [Doc(f"d{i:03d}", f"Alice {i} mails a{i}@b.org") for i in range(25)]
    dp = tmp_path / "docs.jsonl"
    write_jsonl(docs, dp)
    fb = FakeBackend()
    out = R.run("gliner2_pii", dp, tmp_path / "preds", backend=fb, shard_size=10, batch_size=8)
    meta, preds = read_predictions(out)
    assert meta["revision"] == "fake-rev" and meta["n_docs"] == 25 and len(preds) == 25
    # resume: final exists -> no new backend calls
    calls = fb.calls
    assert R.run("gliner2_pii", dp, tmp_path / "preds", backend=fb, shard_size=10) == out and fb.calls == calls
    # partial shards: delete final and one shard -> only that shard recomputed
    out.unlink()
    shards = sorted((tmp_path / "preds").glob("*.shards/shard-*.jsonl"))
    shards[1].unlink()
    fb2 = FakeBackend()
    R.run("gliner2_pii", dp, tmp_path / "preds", backend=fb2, shard_size=10, batch_size=100)
    assert fb2.calls == 1


def test_bench_score_end_to_end(tmp_path, monkeypatch):
    from s1pii import bench, ledger
    from s1pii.data import loaders as L
    monkeypatch.setattr(L, "DATA_DIR", tmp_path)
    monkeypatch.setattr(ledger, "RESULTS", tmp_path / "results")
    docs = [Doc(f"d{i:03d}", f"Alice {i} mails a{i}@b.org", (Span(f"d{i:03d}", 0, 5, PERSON),
            Span(f"d{i:03d}", len(f"Alice {i} mails "), len(f"Alice {i} mails a{i}@b.org"), EMAIL)),
            cluster_id=f"c{i}") for i in range(40)]
    cal, test = L.calib_test_split(docs, 0.25)
    p = bench.split_paths("pii_trace")
    p["calib"].parent.mkdir(parents=True)
    write_jsonl(cal, p["calib"]); write_jsonl(test, p["test"])
    cpred = R.run("gliner2_pii", p["calib"], tmp_path / "preds", backend=FakeBackend())
    tpred = R.run("gliner2_pii", p["test"], tmp_path / "preds", backend=FakeBackend())
    row = bench.score("gliner2_pii", "pii_trace", cpred, tpred, n_boot_ci=100)
    assert row["result"]["pauc"] == pytest.approx(0.0) and row["dev_threshold"] is not None
    assert (tmp_path / "results" / "ledger.jsonl").exists()
    with pytest.raises(ValueError):
        bench.score("gliner2_pii", "pii_trace", None, cpred)       # wrong docs


def test_dev_slice_stable():
    from s1pii import bench
    docs = [Doc(f"n{i}", "x") for i in range(5000)]
    dev, rest = bench.dev_slice(docs)
    assert 50 < len(dev) < 150 and len(dev) + len(rest) == 5000
    assert bench.dev_slice(list(reversed(docs)))[0][0].doc_id in {d.doc_id for d in dev}
