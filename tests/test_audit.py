from s1pii.schema import Doc, Span, PERSON
from s1pii.audit.dedup import audit, skeleton, drop_flagged


def mk(did, text, spans):
    return Doc(did, text, tuple(Span(did, s, e, PERSON) for s, e in spans))


def test_skeleton_masks_entities_and_digits():
    d = mk("a", "Dear Ann, your code is 1234.", [(5, 8)])
    assert skeleton(d) == "dear <person>, your code is 0000."


def test_audit_flags_near_dup_and_template():
    base = "The quarterly statement for account holder was mailed to the registered address on file today"
    train = [mk("t1", "Dear Ann Lee, " + base, [(5, 12)])]
    test = [mk("x1", "Dear Ann Lee, " + base, [(5, 12)]),                 # near duplicate
            mk("x2", "Dear Bob Stone, " + base, [(5, 14)]),              # same template, new entity
            mk("x3", "Completely different content about weather in the mountains and rivers flowing", [])]
    rep = audit(train, test)
    assert "x1" in rep["flagged"] and "x2" in rep["flagged"] and "x3" not in rep["flagged"]
    assert rep["flagged"]["x2"].get("identical_skeleton")
    assert [d.doc_id for d in drop_flagged(test, rep)] == ["x3"]
