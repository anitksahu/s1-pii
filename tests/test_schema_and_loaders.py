import json

import numpy as np
import pytest

from s1pii.schema import Doc, Span, OffsetError, validate_doc, PERSON, IGNORE, NOT_PII, write_jsonl, read_jsonl
from s1pii.data import loaders as L
from s1pii.data.census import census_raws
from s1pii import taxonomy as tx
from s1pii.data.manifest import require_allowed, LicenceGateError


@pytest.mark.parametrize("text,surface", [
    ("Hi 😀 Zoë Ångström here", "Zoë Ångström"),
    ("客户 王小明 的电话", "王小明"),
    ("line1\r\nJohn Doe\r\n", "John Doe"),
    ("Café owner José Lopez", "José Lopez"),
])
def test_offset_roundtrip_unicode(text, surface, tmp_path):
    s = text.index(surface)
    d = Doc("d1", text, (Span("d1", s, s + len(surface), PERSON, surface=surface),))
    validate_doc(d)
    write_jsonl([d], tmp_path / "x.jsonl")
    back = read_jsonl(tmp_path / "x.jsonl")[0]
    assert back.text[back.spans[0].start:back.spans[0].end] == surface


def test_span_offset_types():
    assert Span("d", np.int64(1), np.int64(3), PERSON).start == 1
    with pytest.raises(OffsetError):
        Span("d", True, 3, PERSON)
    with pytest.raises(OffsetError):
        Span("d", 1.0, 3, PERSON)
    with pytest.raises(OffsetError):
        Span("d", 5, 5, PERSON)
    with pytest.raises(OffsetError):
        Span("d", 0, 2, "NOPE")
    with pytest.raises(OffsetError):
        Span("d", 0, 2, PERSON, score=1.5)


def test_surface_mismatch_rejected():
    with pytest.raises(OffsetError):
        validate_doc(Doc("d", "hello world", (Span("d", 0, 5, PERSON, surface="world"),)))


def test_rows_normalizer_and_parse_spans():
    assert L.rows({"a": [1, 2], "b": ["x", "y"]}) == [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    assert L.rows([{"a": 1}]) == [{"a": 1}]
    assert L.parse_spans("[{'start': 0, 'end': 1, 'label': 'x'}]", "t") == [{"start": 0, "end": 1, "label": "x"}]
    with pytest.raises(L.SchemaError):
        L.parse_spans("not json", "t")


def _build(raws, name, mapper, **kw):
    return L.build(raws, name, "test", mapper, **kw)


TAB_RECS = [{"doc_id": "x", "text": "Mr John Smith of Oslo, case 123/45.", "annotations": {
    "a1": {"entity_mentions": [
        {"start_offset": 3, "end_offset": 13, "span_text": "John Smith", "entity_type": "PERSON", "identifier_type": "DIRECT"},
        {"start_offset": 17, "end_offset": 21, "span_text": "Oslo", "entity_type": "LOC", "identifier_type": "QUASI"},
        {"start_offset": 28, "end_offset": 34, "span_text": "123/45", "entity_type": "CODE", "identifier_type": "NO_MASK"}]},
    "a2": {"entity_mentions": [
        {"start_offset": 3, "end_offset": 13, "span_text": "John Smith", "entity_type": "PERSON", "identifier_type": "DIRECT"},
        {"start_offset": 3, "end_offset": 7, "span_text": "John", "entity_type": "PERSON", "identifier_type": "QUASI"}]}}}]


def test_tab_tiers_and_dedupe():
    direct, _ = _build(L.raw_tab(TAB_RECS, "test"), "tab_direct", lambda r: tx.map_tab(*r.split("/"), tier="direct"))
    labs = sorted((s.start, s.end, s.label_canonical) for s in direct[0].spans)
    assert labs == [(3, 7, IGNORE), (3, 13, PERSON), (17, 21, IGNORE), (28, 34, NOT_PII)]
    quasi, _ = _build(L.raw_tab(TAB_RECS, "test"), "tab_quasi", lambda r: tx.map_tab(*r.split("/"), tier="quasi"))
    assert sum(s.is_pii for s in quasi[0].spans) == 3


def test_spy_faker_fill_deterministic_and_reconstruct():
    recs = [{"tokens": ["Hi", "I", "am", "NAME_PH", "email", "EMAIL_PH", "."],
             "trailing_whitespaces": [True, True, True, True, True, False, False],
             "ent_tags": ["O", "O", "O", "B-NAME", "O", "B-EMAIL", "O"]}] * 3
    f1, f2 = L.spy_faker_fill(recs, seed=0), L.spy_faker_fill(recs, seed=0)
    assert f1 == f2 and L.spy_faker_fill(recs, seed=1) != f1
    docs, rep = _build(L.raw_spy_tokens(f1, "medical"), "spy_medical", lambda r: tx.map_label("spy", r))
    d = docs[0]
    assert d.text.startswith("Hi I am ") and d.text.endswith(".")
    labs = [s.label_canonical for s in d.spans]
    assert labs == ["PERSON", "EMAIL"]
    assert " " not in d.text[d.spans[1].start:d.spans[1].end]


def test_spy_rejects_classlabel_ints_and_placeholders():
    bad = [{"tokens": ["a"], "trailing_whitespace": [False], "ent_tags": [3]}]
    with pytest.raises(L.SchemaError):
        list(L.raw_spy_tokens(bad, "legal"))
    ph = [{"tokens": ["{name}"], "trailing_whitespace": [False], "ent_tags": ["B-NAME"]}]
    with pytest.raises(L.SchemaError):
        list(L.raw_spy_tokens(ph, "legal"))


def test_pii_trace_offsets_prefix_ignore_and_hf_dict_of_lists():
    rec = {"id": "c1",
           "turns": {"turn": [0, 1], "user": ["I am Ravi.", "Mail ravi@a.io"], "assistant": ["Hi Ravi!", "Noted."]},
           "spans": {"label": ["private_person", "private_person", "private_email"], "turn": [0, 0, 1],
                     "source": ["user", "assistant", "user"], "start": [5, 3, 5], "end": [9, 7, 14],
                     "text": ["Ravi", "Ravi", "ravi@a.io"]}}
    docs, _ = _build(L.raw_pii_trace([rec]), "pii_trace", lambda r: tx.map_label("pii_trace", r))
    d = docs[0]
    assert [d.text[s.start:s.end] for s in d.pii_spans()] == ["Ravi", "Ravi", "ravi@a.io"]
    assert all(d.text[s.start:s.end] in ("User: ", "Assistant: ") for s in d.ignore_spans())


def test_json_spans_fail_closed_and_census():
    recs = [{"uid": "u1", "text": "Bob", "spans": json.dumps([{"start": 0, "end": 3, "label": "mystery"}])}]
    raws = lambda: L.raw_json_spans(recs, "nemotron", text_key="text", spans_key="spans",
                                    id_fn=lambda r, i: r["uid"], cluster_fn=None)
    with pytest.raises(tx.UnmappedLabelError) as e:
        _build(raws(), "nemotron", lambda r: tx.map_label("nemotron", r))
    assert e.value.labels == ["mystery"]
    c = census_raws(raws(), lambda r: tx.map_label("nemotron", r))
    assert c["unmapped"] == ["mystery"] and c["labels"]["mystery"]["count"] == 1


def test_reject_budget():
    recs = [{"uid": f"u{i}", "text": "Bob", "spans": [{"start": 0, "end": 3 if i else 9, "label": "first_name"}]}
            for i in range(10)]
    mk = lambda: L.raw_json_spans(recs, "nemotron", text_key="text", spans_key="spans",
                                  id_fn=lambda r, i: r["uid"], cluster_fn=None)
    with pytest.raises(L.RejectBudgetExceeded):
        _build(mk(), "nemotron", lambda r: tx.map_label("nemotron", r))
    docs, rep = _build(mk(), "nemotron", lambda r: tx.map_label("nemotron", r), max_reject_rate=0.2)
    assert len(docs) == 9 and rep["rejected"] == 1 and "u0" in rep["rejects"][0]["doc_id"]


def test_census_offset_issues():
    rd = L.RawDoc("d", "Call Bob now", [{"start": 4, "end": 8, "label": "first_name", "surface": " Bob"},
                                        {"start": 6, "end": 7, "label": "first_name"}])
    c = census_raws([rd], lambda r: tx.map_label("nemotron", r))
    assert c["issues"]["whitespace_edge"] == 1 and c["issues"]["cuts_word"] == 1 and c["issues"]["overlapping"] == 1


def test_taxonomy_targets_and_admissible():
    from s1pii.schema import ALL_TARGETS, CANONICAL_TYPES
    for name, table in tx.MAPS.items():
        assert set(table.values()) <= set(ALL_TARGETS), name
    for ds, allowed in tx.ADMISSIBLE.items():
        assert allowed <= set(CANONICAL_TYPES), ds
    from s1pii.data.manifest import manifest
    for ds, e in manifest().items():
        if "eval" in e.get("purposes", []):
            tx.admissible(ds)


def test_licence_gate(monkeypatch):
    require_allowed("nemotron", "train")
    with pytest.raises(LicenceGateError):
        require_allowed("tab_direct", "train")
    with pytest.raises(LicenceGateError):
        require_allowed("unknown_ds", "eval")
    monkeypatch.delenv("S1PII_ALLOW_RESEARCH", raising=False)
    with pytest.raises(LicenceGateError):
        require_allowed("ai4privacy", "eval")
    monkeypatch.setenv("S1PII_ALLOW_RESEARCH", "1")
    require_allowed("ai4privacy", "eval")


def test_calib_split_by_cluster_and_sampling():
    docs = [Doc(f"d{i}", "x", cluster_id=f"c{i // 3}", dataset="a" if i < 80 else "b") for i in range(99)]
    cal, test = L.calib_test_split(docs, 0.2)
    assert {d.cluster_id for d in cal}.isdisjoint({d.cluster_id for d in test})
    assert len(cal) + len(test) == 99 and len({d.cluster_id for d in cal}) == 7
    s1 = L.sample(docs, 10, strata=lambda d: d.dataset)
    assert [d.doc_id for d in s1] == [d.doc_id for d in L.sample(list(reversed(docs)), 10, strata=lambda d: d.dataset)]


def test_snapshot_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "DATA_DIR", tmp_path)
    calls = []
    def recipe(split):
        calls.append(split)
        return L.raw_tab(TAB_RECS, split), (lambda r: tx.map_tab(*r.split("/"), tier="direct")), {"source_sha256": "x"}
    monkeypatch.setitem(L.RECIPES, "tab_direct", (recipe, "test"))
    a = L.load("tab_direct")
    b = L.load("tab_direct")
    assert calls == ["test"] and [d.to_json() for d in a] == [d.to_json() for d in b]
    assert L.snapshot_meta("tab_direct", "test")["dataset_hash"]


def test_snapshot_invalidation(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "DATA_DIR", tmp_path)
    calls = []
    def recipe(split):
        calls.append(split)
        return L.raw_tab(TAB_RECS, split), (lambda r: tx.map_tab(*r.split("/"), tier="direct")), {}
    monkeypatch.setitem(L.RECIPES, "tab_direct", (recipe, "test"))
    L.load("tab_direct")
    snap, meta = L.snapshot_paths("tab_direct", "test")
    snap.write_text(snap.read_text()[:20])            # truncated snapshot
    L.load("tab_direct")
    assert len(calls) == 2
    monkeypatch.setattr(tx, "MAP_VERSION", "taxonomy-v9")
    L.load("tab_direct")
    assert len(calls) == 3
    meta.unlink()
    L.load("tab_direct")
    assert len(calls) == 4


def test_raw_json_spans_casefold_surface_and_numeric_surface():
    from s1pii.data.loaders import raw_json_spans
    recs = [{"text": "She is Black, pin 1234.", "spans": [
        {"start": 7, "end": 12, "label": "race_ethnicity", "text": "black"},
        {"start": 18, "end": 22, "label": "pin", "text": 1234}]}]
    rd = next(raw_json_spans(recs, "nemotron", text_key="text", spans_key="spans",
                             id_fn=lambda r, i: str(i), cluster_fn=None, casefold_surface=True))
    assert [s["surface"] for s in rd.spans] == ["Black", "1234"]
    rd = next(raw_json_spans(recs, "x", text_key="text", spans_key="spans",
                             id_fn=lambda r, i: str(i), cluster_fn=None))
    assert rd.spans[0]["surface"] == "black"      # without the flag the mismatch is kept (and rejected)


def test_dev_slice_keeps_locale_twins_together():
    from s1pii import bench
    from s1pii.schema import Doc
    docs = [Doc(f"n_{u}-{loc}", "x", (), cluster_id="c", meta={"uid": str(u)}) for u in range(400) for loc in ("us", "intl")]
    dev, rest = bench.dev_slice(docs)
    dev_u = {d.meta["uid"] for d in dev}
    assert dev and not dev_u & {d.meta["uid"] for d in rest}
