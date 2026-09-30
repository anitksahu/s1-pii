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
    assert len(ws) == 4
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
    assert label_descriptions("nvidia_gliner_pii") is None
    assert set(query_labels("gliner25_base_zeroshot")) == set(load_config("labels.yaml"))


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


def test_adapter_bad_offsets_dropped_and_unknown_labels_fail():
    d = Doc("d1", "Alice here")
    preds, rep = Adapter("gliner2_pii", FakeBackend(extra=[RawEnt(3, 99, "email", 0.9, None)])).predict_docs([d])
    assert rep.dropped_bad_offsets == 1 and [p.label_canonical for p in preds["d1"]] == [PERSON]
    with pytest.raises(ValueError, match="martian"):
        Adapter("gliner2_pii", FakeBackend(extra=[RawEnt(0, 5, "martian", 0.9, "Alice")])).predict_docs([d])


def test_label_coverage_uniform():
    from s1pii import taxonomy as tx
    from s1pii.data.manifest import headline_datasets
    for sysname in load_config()["systems"]:
        covered = set(query_labels(sysname).values())
        for ds in headline_datasets():
            assert tx.admissible(ds) <= covered, (sysname, ds)


class LimitedBackend(FakeBackend):
    def __init__(self, max_tokens):
        super().__init__(); self.max_tokens, self.seen = max_tokens, []
    def fits(self, text, labels):
        from s1pii.adapters.base import _TOK
        return len(_TOK.findall(text)) <= self.max_tokens
    def predict(self, texts, labels, threshold):
        self.seen.extend(texts)
        return super().predict(texts, labels, threshold)


def test_windows_split_to_fit_json_and_never_truncate():
    import json as _j
    text = _j.dumps([{"name": "Alice", "id": i, "tags": ["a", "b"]} for i in range(60)])
    d = Doc("j", text)
    fb = LimitedBackend(40)
    preds, rep = Adapter("gliner2_pii", fb).predict_docs([d])
    from s1pii.adapters.base import _TOK
    assert rep.window_splits > 0 and all(len(_TOK.findall(t)) <= 40 for t in fb.seen)
    assert len([p for p in preds["j"] if p.label_canonical == PERSON]) == 60
    from s1pii.adapters.base import fit_windows, WindowTooLong
    with pytest.raises(WindowTooLong):
        fit_windows("x" * 50, [(0, 50)], LimitedBackend(0), [])


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
    shards = sorted((tmp_path / "preds").glob("*.shards-*/shard-*.jsonl"))
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


def test_calib_test_config_mismatch_rejected(tmp_path, monkeypatch):
    from s1pii import bench, ledger
    from s1pii.data import loaders as L
    monkeypatch.setattr(L, "DATA_DIR", tmp_path)
    monkeypatch.setattr(ledger, "RESULTS", tmp_path / "results")
    docs = [Doc(f"d{i:03d}", f"Alice {i}", (Span(f"d{i:03d}", 0, 5, PERSON),), cluster_id=f"c{i}") for i in range(20)]
    cal, test = L.calib_test_split(docs, 0.25)
    p = bench.split_paths("pii_trace"); p["calib"].parent.mkdir(parents=True)
    write_jsonl(cal, p["calib"]); write_jsonl(test, p["test"])
    class Other(FakeBackend):
        revision = "other-rev"
    cpred = R.run("gliner2_pii", p["calib"], tmp_path / "preds", backend=Other())
    tpred = R.run("gliner2_pii", p["test"], tmp_path / "preds", backend=FakeBackend())
    with pytest.raises(ValueError, match="revision"):
        bench.score("gliner2_pii", "pii_trace", cpred, tpred, n_boot_ci=50)


def test_materialize_atomic_and_validated(tmp_path, monkeypatch):
    from s1pii import bench
    from s1pii.data import loaders as L
    monkeypatch.setattr(L, "DATA_DIR", tmp_path)
    calls = []
    docs = [Doc(f"d{i}", "x", cluster_id=f"c{i}") for i in range(10)]
    monkeypatch.setattr(bench, "splits", lambda n: (calls.append(n), (docs[:2], docs[2:]))[1])
    bench.materialize("pii_trace"); bench.materialize("pii_trace")
    assert len(calls) == 1
    p = bench.split_paths("pii_trace")["test"]
    p.write_text(p.read_text()[:10])
    bench.materialize("pii_trace")
    assert len(calls) == 2


def test_fits_counts_descriptions():
    from s1pii.adapters.backends import _Fits
    class T:
        def encode(self, text, add_special_tokens=True):
            return text.split()
    class B(_Fits):
        def __init__(self, desc):
            self.model = None; self.descriptions = desc
            self.max_words, self.max_subwords, self.tokenizer = 1000, 60, T()
        def _query(self, labels):
            return {l: self.descriptions.get(l, l) for l in labels} if self.descriptions else labels
    text = " ".join(["w"] * 30)
    assert B(None).fits(text, ["email"])
    assert not B({"email": " ".join(["long"] * 40)}).fits(text, ["email"])


def test_nvidia_native_labels_match_nemotron_taxonomy():
    from s1pii import taxonomy as tx
    from s1pii.schema import CANONICAL_TYPES
    from s1pii.adapters.base import load_config
    native = load_config()["systems"]["nvidia_gliner_pii"]["labels"]
    assert native == {k: v for k, v in tx.NEMOTRON.items() if v in CANONICAL_TYPES}


def test_fit_windows_splits_a_single_huge_word_by_characters():
    from s1pii.adapters.base import fit_windows
    text = "cookie: " + "A" * 3000 + " end"
    class B:
        def fits(self, t, labels):
            return len(t) <= 500
    wins, n = fit_windows(text, [(0, len(text))], B(), ["x"])
    assert n > 0 and all(b - a <= 500 for a, b in wins)
    cov = set()
    for a, b in wins:
        cov.update(range(a, b))
    assert cov == set(range(len(text)))
