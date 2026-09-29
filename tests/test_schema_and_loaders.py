import json

import pytest

from s1pii.schema import Doc, Span, OffsetError, validate_doc, PERSON, IGNORE, NOT_PII, write_jsonl, read_jsonl
from s1pii.data import loaders as L
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
    p = tmp_path / "x.jsonl"
    write_jsonl([d], p)
    back = read_jsonl(p)[0]
    assert back.text[back.spans[0].start:back.spans[0].end] == surface


def test_surface_mismatch_rejected():
    with pytest.raises(OffsetError):
        validate_doc(Doc("d", "hello world", (Span("d", 0, 5, PERSON, surface="world"),)))


def test_bad_spans_rejected():
    with pytest.raises(OffsetError):
        Span("d", 5, 5, PERSON)
    with pytest.raises(OffsetError):
        Span("d", 0, 2, "NOPE")
    with pytest.raises(OffsetError):
        Span("d", 0, 2, PERSON, score=1.5)


def test_tab_mapping_tiers():
    recs = [{"doc_id": "x", "text": "Mr John Smith of Oslo, case 123/45.", "annotations": {
        "a1": {"entity_mentions": [
            {"start_offset": 3, "end_offset": 13, "span_text": "John Smith", "entity_type": "PERSON", "identifier_type": "DIRECT"},
            {"start_offset": 17, "end_offset": 21, "span_text": "Oslo", "entity_type": "LOC", "identifier_type": "QUASI"},
            {"start_offset": 28, "end_offset": 34, "span_text": "123/45", "entity_type": "CODE", "identifier_type": "NO_MASK"}]},
        "a2": {"entity_mentions": [
            {"start_offset": 3, "end_offset": 7, "span_text": "John", "entity_type": "PERSON", "identifier_type": "QUASI"}]}}}]
    direct = L.tab_from_records(recs, "test", "direct")[0]
    labs = sorted((s.start, s.label_canonical) for s in direct.spans)
    assert labs == [(3, IGNORE), (3, PERSON), (17, IGNORE), (28, NOT_PII)]
    quasi = L.tab_from_records(recs, "test", "quasi")[0]
    assert sum(s.is_pii for s in quasi.spans) == 3


def test_spy_token_reconstruction():
    rec = {"tokens": ["Call", "Anna", "Berg", "at", "anna@x.org", "."],
           "trailing_whitespace": [True, True, True, True, False, False],
           "ent_tags": ["O", "B-name", "I-name", "O", "B-email", "O"]}
    d = L.spy_from_records([rec], "medical", "test")[0]
    assert d.text == "Call Anna Berg at anna@x.org."
    assert [(d.text[s.start:s.end], s.label_canonical) for s in d.spans] == [("Anna Berg", "PERSON"), ("anna@x.org", "EMAIL")]


def test_spy_placeholder_detected():
    rec = {"tokens": ["{name}"], "trailing_whitespace": [False], "ent_tags": ["B-name"]}
    with pytest.raises(L.SchemaError):
        L.spy_from_records([rec], "legal", "test")


def test_pii_trace_offsets():
    rec = {"id": "c1", "turns": [{"turn": 0, "user": "I am Ravi.", "assistant": "Hi Ravi!"},
                                 {"turn": 1, "user": "Mail ravi@a.io", "assistant": "Noted."}],
           "spans": [{"label": "private_person", "turn": 0, "source": "user", "start": 5, "end": 9, "text": "Ravi"},
                     {"label": "private_person", "turn": 0, "source": "assistant", "start": 3, "end": 7, "text": "Ravi"},
                     {"label": "private_email", "turn": 1, "source": "user", "start": 5, "end": 14, "text": "ravi@a.io"}]}
    d = L.pii_trace_from_records([rec], "test")[0]
    assert [d.text[s.start:s.end] for s in d.spans] == ["Ravi", "Ravi", "ravi@a.io"]


def test_json_span_dataset_fail_closed():
    recs = [{"uid": "u1", "text": "Bob", "spans": json.dumps([{"start": 0, "end": 3, "label": "mystery"}])}]
    with pytest.raises(tx.UnmappedLabelError) as e:
        L.spans_dataset_from_records(recs, "nemotron", "test", text_key="text", spans_key="spans",
                                     id_key="uid", cluster_key=None)
    assert e.value.labels == ["mystery"]


def test_json_span_dataset_ok_and_cluster():
    recs = [{"uid": "u1", "fam": "f", "text": "Bob lives at 1 Main St",
             "spans": json.dumps([{"start": 0, "end": 3, "label": "first_name", "text": "Bob"},
                                  {"start": 13, "end": 22, "label": "street_address", "text": "1 Main St"}])}]
    d = L.spans_dataset_from_records(recs, "nemotron", "test", text_key="text", spans_key="spans",
                                     id_key="uid", cluster_key="fam")[0]
    assert d.cluster_id == "nemotron:f" and len(d.spans) == 2


def test_taxonomy_maps_total_over_targets():
    from s1pii.schema import ALL_TARGETS
    for name, table in tx.MAPS.items():
        assert set(table.values()) <= set(ALL_TARGETS), name
    for v in tx.TAB_ENTITY.values():
        assert v in ALL_TARGETS


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


def test_sampling_deterministic_and_stratified():
    docs = [Doc(f"d{i}", "x", dataset="a" if i < 80 else "b") for i in range(100)]
    s1 = L.sample(docs, 10, strata=lambda d: d.dataset)
    s2 = L.sample(list(reversed(docs)), 10, strata=lambda d: d.dataset)
    assert [d.doc_id for d in s1] == [d.doc_id for d in s2]
    assert sum(d.dataset == "a" for d in s1) == 8
