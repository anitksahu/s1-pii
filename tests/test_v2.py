"""CPU tests for v2 (labels at inference, hierarchy, abstention) with a tiny BERT."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from s1pii.schema import Doc, Span, write_jsonl, PERSON, EMAIL
from s1pii.data.synth import generate
from s1pii.model.encode import train_examples
from s1pii.model.train import TrainConfig, train
from s1pii.v2 import labels as LB
from s1pii.v2.features import (FeatureConfig, extract, training_extras, load_store, select_training_docs,
                               KIND_EXTRA, Extractor)
from s1pii.v2.head import HeadConfig, train_head, load_head
from s1pii.v2 import infer
from tests.test_model import tok, tiny_model  # noqa: F401  (fixture)


@pytest.fixture(scope="module")
def v1_dir(tok, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("v1")
    docs = generate(80, seed=13)
    ex = train_examples(docs, tok, max_len=128)
    cfg = TrainConfig(backbone="tiny", seed=0, token_budget=4096, grad_accum=1, lr_encoder=5e-3, lr_head=5e-2,
                      ckpt_every=1000, log_every=50, gradient_checkpointing=False, bf16=False)
    return train(cfg, tmp / "run", model=tiny_model(tok), tokenizer=tok, examples=ex, max_steps=40, device="cpu",
                 resume=False)


@pytest.fixture(scope="module")
def heldout(tmp_path_factory):
    # hold out the synthetic "url" family for the tests
    return {"labels": ["url"], "nodes": ["url"], "synonyms_all_sources": sorted(x for x in LB.NATIVE if LB.node_of(x) == "url")}


FC = FeatureConfig(max_len=128, validators=False)


@pytest.fixture(scope="module")
def trained(v1_dir, heldout, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("v2")
    docs = generate(120, seed=21)
    ex = training_extras(docs, seed=0, heldout_nodes=set(heldout["nodes"]))
    store = extract(v1_dir, docs, tmp / "stores", name="train", extras=ex, label_texts=LB.all_texts(),
                    cfg=FeatureConfig(max_len=128, validators=False, store_min_pb=0.05))
    head = train_head(store, tmp / "head", set(heldout["nodes"]), HeadConfig(epochs=30, batch=128, dim=64), device="cpu",
                      log=lambda *_: None)
    return tmp, store, head, docs


def test_label_hierarchy_and_sets():
    assert LB.compatible("ssn", "unique_id") and not LB.compatible("ssn", "employee_id")
    assert LB.compatible("first_name", "name") and not LB.compatible("first_name", "last_name")
    assert LB.canonical_of("iban") == "ACCOUNT_NUMBER" and LB.canonical_of("my_custom_thing") == "OTHER_PII"
    ls = LB.benchmark_label_set("pii_trace")
    assert {l.name for l in ls.labels} >= {"private_person", "secret"} and all(l.sensitive for l in ls.labels)
    adhoc = LB.LabelSet.from_names(["employee_id", "badge number: staff badge code"])
    assert adhoc.labels[1].description == "staff badge code" and adhoc.hash() != ls.hash()


def test_select_heldout_rule_is_deterministic_and_vetoable():
    rng = np.random.default_rng(0)
    cache = {}
    def embed(xs):
        return np.stack([cache.setdefault(x, rng.normal(size=16)) for x in xs])
    counts = {r.lower(): 100 for r in LB.tx.NEMOTRON}
    a = LB.select_heldout(counts, embed, k=3)
    b = LB.select_heldout(counts, embed, k=3)
    assert a["labels"] == b["labels"] and len(a["labels"]) == 3
    v = LB.select_heldout(counts, embed, k=3, vetoes={a["labels"][0]: "test"})
    assert a["labels"][0] not in v["labels"]


def test_store_candidates_and_pb_exact(v1_dir, tmp_path):
    docs = generate(5, seed=3)
    store = extract(v1_dir, docs, tmp_path, name="t", cfg=FC, label_texts=["email"])
    z = load_store(store)
    assert len(z["doc_ids"]) == 5 and z["offsets"][-1] == len(z["pb"])
    assert (z["pb"] >= FC.floor - 1e-9).all() and (z["pb"] <= 1.0 + 1e-6).all()
    assert z["rep"].shape[1] == 3 * 32 and z["rep"].dtype == np.float16
    # P_b equals the sum over types of the v1 span marginals
    assert np.allclose(z["pb"], np.minimum(1.0, z["v1"].sum(1)), atol=1e-5)
    again = extract(v1_dir, docs, tmp_path, name="t", cfg=FC, label_texts=["email"])
    assert again == store                                      # cached by identity


def test_head_trains_and_types_seen_labels(trained, heldout):
    tmp, store, head, docs = trained
    man = json.loads((head / "head_manifest.json").read_text())
    assert "url" not in man["vocab"] and man["items"]["gold"] > 0 and man["items"]["hard_neg"] >= 0
    assert man["val"]["desc"]["typing_acc"] is not None


def test_decisions_formula_and_spans(trained, v1_dir, tmp_path):
    tmp, store, head, docs = trained
    test = generate(8, seed=77)
    st = extract(v1_dir, test, tmp_path, name="bench", cfg=FC, label_texts=LB.all_texts())
    L = LB.LabelSet("t", [LB.native_label("person"), LB.native_label("email"), LB.native_label("city", sensitive=False)])
    D = infer.decide(st, head, L)
    assert np.allclose(D.probs.sum(1), 1.0, atol=1e-5)
    manual = D.pb * D.probs[:, :2].sum(1)                          # city is not sensitive
    assert np.allclose(D.p_sens, manual)
    preds = infer.to_spans(D, L, "s", {d.doc_id: d.text for d in test})
    for d in test:
        for s in preds[d.doc_id]:
            assert d.text[s.start:s.end] == s.surface and s.score >= infer.FLOOR
            assert s.label_raw in {"person", "email", "city"}
    # hierarchical decision backs off to the parent when mass is split between siblings
    L2 = LB.LabelSet("h", [LB.native_label("first_name"), LB.native_label("last_name")])
    assert infer.hier_decide(np.array([0.45, 0.45, 0.10]), L2, 0.5) == "person_name"
    assert infer.hier_decide(np.array([0.8, 0.1, 0.1]), L2, 0.5) == "first_name"


def test_write_predictions_roundtrip(trained, v1_dir, tmp_path):
    tmp, store, head, _ = trained
    test = generate(6, seed=78)
    p = tmp_path / "docs.jsonl"
    write_jsonl(test, p)
    st = extract(v1_dir, test, tmp_path / "s", name="b", cfg=FC, label_texts=LB.all_texts())
    L = LB.benchmark_label_set("pii_trace")
    path = infer.write(st, head, L, p, tmp_path / "pred", "s1v2-cheap_all-sources-s1")
    from s1pii.ledger import read_predictions
    meta, preds = read_predictions(path)
    assert meta["config"]["labels_hash"] == L.hash() and set(preds) == {d.doc_id for d in test}
    typed = infer.write(st, head, LB.c3_label_set(), p, tmp_path / "pred", "c3", score="typed")
    assert read_predictions(typed)[0]["config"]["score"] == "typed"


def test_c4_band_and_selective_counts():
    from s1pii.v2.evaluate import selective_hist, sel_leak, sel_over
    from s1pii.eval import metrics as M
    d = Doc("d1", "Alice Smith paid 42", (Span("d1", 0, 11, PERSON),), cluster_id="c1")
    preds = {"d1": [Span("d1", 0, 5, PERSON, score=0.9), Span("d1", 6, 11, PERSON, score=0.3),
                    Span("d1", 17, 19, PERSON, score=0.95)]}
    vs = [M.view(d, preds["d1"])]
    lo, hi = 1 + M.k_of(0.2), 1 + M.k_of(0.5)
    hp, hn = selective_hist(vs, lo, hi, ["c1"])
    # "Smith" (0.3) is deferred; "Alice" masked; "42" (non-PII) masked -> over-redacted
    assert hp[0].tolist() == [5, 0, 10] and hn[0].tolist() == [6, 2, 6]
    assert sel_leak(hp.sum(0), hn.sum(0)) == 0 and sel_over(hp.sum(0), hn.sum(0)) == pytest.approx(2 / 6)


def test_c3_typed_prf_and_bootstrap():
    from s1pii.v2.evaluate import typed_prf, _f1_bootstrap
    docs = [Doc(f"d{i}", "mac 00:11:22 here", (Span(f"d{i}", 4, 12, "OTHER_PII", label_raw="mac_address"),),
                cluster_id=f"c{i}") for i in range(6)]
    good = {d.doc_id: [Span(d.doc_id, 4, 12, "OTHER_PII", label_raw="mac_address", score=0.9)] for d in docs}
    bad = {d.doc_id: [Span(d.doc_id, 4, 12, "OTHER_PII", label_raw="ipv4", score=0.9)] for d in docs}
    assert typed_prf(docs, good, {"mac_address"}, 0.5)["micro_f1"] == 1.0
    assert typed_prf(docs, bad, {"mac_address"}, 0.5)["micro_f1"] == 0.0
    r = _f1_bootstrap(docs, [good, good, good], bad, ["mac_address"], 0.5, "macro", 200, 0)
    assert r["diff"] == pytest.approx(1.0) and r["p_value"] < 0.05


def test_level1_targets_any_for_heldout_and_teacher(tok):
    from s1pii.v2.level1 import l1_target, any_mask_spans, source_inventory
    from s1pii.model.encode import ANY
    d = Doc("x", "Bob lives in Paris near www.a.com", (Span("x", 0, 3, PERSON, label_raw="person"),
                                                     Span("x", 24, 33, "URL", label_raw="url")))
    inv = source_inventory([d], {"x": "synthetic_conv"})
    anys = any_mask_spans(d, [(13, 18, "GPE")], inv["synthetic_conv"])
    assert anys == [(13, 18)]
    ids, target, _ = l1_target(d, tok, {"url"}, anys)
    enc = tok(d.text, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc["offset_mapping"]
    for k, (a, b) in enumerate(offs):
        if a >= 24:
            assert target[k] == ANY                 # held-out: unknown at level 1
        if 13 <= a < 18:
            assert target[k] == ANY                 # teacher family not annotated by the source
        if a == 0:
            assert target[k] in (1, 4)              # B or S of the single entity type


def test_level1_trains_nt1_and_extracts(v1_dir, tok, tmp_path):
    from s1pii.v2.level1 import train_level1
    docs = generate(30, seed=4)
    out = train_level1(v1_dir, docs, {d.doc_id: "synthetic_conv" for d in docs}, tmp_path / "l1", {"url"}, seed=1,
                       teacher={}, max_steps=3, max_len=128, device="cpu", gradient_checkpointing=False, bf16=False,
                       token_budget=4096, grad_accum=1)
    ex = Extractor(out, FC, "cpu")
    assert ex.crf.nt == 1
    f = ex.doc_features(generate(2, seed=9)[0])
    assert f["v1"].shape[1] == 1


def test_finetune_and_typing_encoder_extraction(trained, v1_dir, tmp_path, heldout):
    from s1pii.v2.finetune import finetune, FinetuneConfig
    tmp, store, head, docs = trained
    out = finetune(v1_dir, store, head, docs, tmp_path / "ft", set(heldout["nodes"]),
                   FinetuneConfig(n_layers=1, context=64, batch=16, max_steps=2), device="cpu", log=lambda *_: None)
    ex = Extractor(v1_dir, FC, "cpu", typing_encoder=out)
    assert ex.identity()["typing_encoder"]
    f = ex.doc_features(generate(2, seed=9)[1])
    assert f["rep"].shape[1] == 96
    h, man = load_head(out / "head")
    assert man["finetuned"]


def test_flat_ablation_trains_and_predicts(v1_dir, tmp_path):
    from s1pii.v2.flat import train_flat, predict_flat
    docs = generate(40, seed=6)
    out = train_flat(v1_dir, docs, tmp_path / "flat", {"url"}, steps=3, batch=4, max_len=64, device="cpu", log=lambda *_: None)
    L = LB.LabelSet("t", [LB.native_label("person"), LB.native_label("email")])
    preds = predict_flat(v1_dir, out, generate(3, seed=5), L, system="flat", max_len=128, device="cpu")
    for sp in (s for v in preds.values() for s in v):
        assert 0.01 <= sp.score <= 1.0 and sp.label_raw in {"person", "email"}


def test_batched_extraction_equals_one_doc_at_a_time(v1_dir):
    docs = generate(7, seed=31)
    a = Extractor(v1_dir, FeatureConfig(max_len=128, validators=True, batch_size=32), "cpu").docs_features(docs)
    ex1 = Extractor(v1_dir, FeatureConfig(max_len=128, validators=True, batch_size=1), "cpu")
    b = [ex1.doc_features(d) for d in docs]
    for x, y in zip(a, b):
        kx = sorted(zip(x["start"], x["end"], np.round(x["pb"], 5)))
        ky = sorted(zip(y["start"], y["end"], np.round(y["pb"], 5)))
        assert kx == ky


# ------------------------------------------------------------------ v2.1 fixes

def test_head_v21_config_and_legacy_load(tmp_path):
    import json, torch
    from s1pii.v2.head import TypingHead, HeadConfig, load_head, save_head
    h = TypingHead(8, 9, HeadConfig(dim=16))
    assert h.norm is not None and h.cfg.max_scale == 30.0
    with torch.no_grad():
        h.log_scale.fill_(10.0)                      # exp(10) >> 30: bounded at inference
        s = torch.nn.functional.normalize(torch.randn(2, 16), dim=-1)
        l = torch.nn.functional.normalize(torch.randn(3, 16), dim=-1)
        assert h.logits(s, l).abs().max() <= 30.0 + abs(float(h.none_bias)) + 1e-4
    save_head(h, tmp_path / "new", HeadConfig(dim=16), {"vocab": []})
    assert load_head(tmp_path / "new")[0].norm is not None
    # a head saved before v2.1 (no new fields in its manifest) loads exactly as it was trained
    old = TypingHead(8, 9, HeadConfig(dim=16, norm_inputs=False))
    save_head(old, tmp_path / "old", HeadConfig(dim=16, norm_inputs=False), {"vocab": []})
    m = json.loads((tmp_path / "old" / "head_manifest.json").read_text())
    for k in ("max_scale", "norm_inputs", "drop_noncand_gold"):
        m["config"].pop(k, None)
    (tmp_path / "old" / "head_manifest.json").write_text(json.dumps(m))
    lh, _ = load_head(tmp_path / "old")
    assert lh.norm is None and lh.cfg.max_scale is None


def test_sensitive_score_keeps_order_near_one():
    import numpy as np
    from s1pii.v2.infer import sensitive_score
    from s1pii.v2.features import VALIDATOR_PB
    probs = np.array([[1 - 1e-9, 0.0, 1e-9], [1 - 1e-12, 0.0, 1e-12]], dtype=np.float64)
    sc = sensitive_score(np.array([VALIDATOR_PB, VALIDATOR_PB]), probs, [True, False])
    assert (sc < 1.0).all() and sc[1] > sc[0]


def test_control_stop(tmp_path):
    from s1pii.v2 import run_v2
    ch = run_v2.Chain.__new__(run_v2.Chain)
    ch.drive = tmp_path
    ch.check_control()                               # no file: no stop
    (tmp_path / "CONTROL").write_text("SKIP_C\n")
    ch.check_control(); assert ch.control_has("SKIP_C")
    (tmp_path / "CONTROL").write_text("STOP")
    import pytest
    with pytest.raises(run_v2.StopRequested):
        ch.check_control()


def test_flat_v21_sampled_texts_typed_scores_and_legacy_manifest(v1_dir, tmp_path):
    import json
    from s1pii.v2.flat import train_flat, predict_flat, label_text_choices
    c = label_text_choices("person")
    assert c["name"] and c["other"] and not set(c["name"]) & set(c["other"])
    docs = generate(40, seed=6)
    out = train_flat(v1_dir, docs, tmp_path / "flat", {"url"}, steps=3, batch=4, max_len=64, device="cpu",
                     text_mode="sample", label_linear=True, max_labels=8, log=lambda *_: None)
    man = json.loads((out / "flat_manifest.json").read_text())
    assert man["label_encoder"]["kind"] == "v1" and man["text_mode"] == "sample" and "skipped_steps" in man
    # a non-sensitive label (IGNORE quasi-identifier) is decoded in typed mode, dropped in sensitive mode
    L = LB.LabelSet("t", [LB.native_label("person"), LB.native_label("email"), LB.native_label("occupation", sensitive=False)])
    assert L.sensitive_mask() == [True, True, False]
    docs2 = generate(3, seed=5)
    typed = predict_flat(v1_dir, out, docs2, L, system="flat", max_len=128, device="cpu", score="typed",
                         floor=1e-6, label_floor=1e-6)
    t_labels = {s.label_raw for v in typed.values() for s in v}
    assert "occupation" in t_labels
    assert (out / "train_log.jsonl").exists()
    from s1pii.v2.flat import _mentions
    for n in LB.NATIVE:                              # name dropout never shows the name
        assert not any(_mentions(t, n) for t in label_text_choices(n)["other"])
    deny = {"http_cookie": "cookie", "fax_number": "fax", "private_email": "email", "swift_bic": "swift",
            "bank_routing_number": "routing"}                # fixed denylist, independent of _mentions
    for n, w in deny.items():
        if n in LB.NATIVE:
            assert not any(w in t.lower() for t in label_text_choices(n)["other"]), (n, label_text_choices(n))
    # a v2 manifest (no label_encoder / label_linear fields) still loads as the v1-label MLP model
    out2 = train_flat(v1_dir, docs, tmp_path / "flat2", {"url"}, steps=2, batch=4, max_len=64, device="cpu", log=lambda *_: None)
    m2 = json.loads((out2 / "flat_manifest.json").read_text())
    for k in ("label_encoder", "label_linear", "text_mode"):
        m2.pop(k)
    (out2 / "flat_manifest.json").write_text(json.dumps(m2))
    predict_flat(v1_dir, out2, docs2[:1], L, system="flat", max_len=128, device="cpu")


def test_label_encoder_v1_matches_v2_pooling(v1_dir):
    import torch
    from s1pii.model.train import load_exported
    from s1pii.v2.flat import LabelEncoder
    m, tok, _ = load_exported(v1_dir)
    enc = m.encoder.eval()
    texts = ["person name", "email: an e-mail address"]
    with torch.no_grad():                         # the v2 inline embed() of flat.py
        e = tok(texts, padding=True, return_tensors="pt", return_special_tokens_mask=True, truncation=True, max_length=64)
        sp = e.pop("special_tokens_mask")
        h = enc(input_ids=e["input_ids"], attention_mask=e["attention_mask"]).last_hidden_state.float()
        mk = (e["attention_mask"].bool() & ~sp.bool()).unsqueeze(-1).float()
        ref = (h * mk).sum(1) / mk.sum(1).clamp(min=1)
    le = LabelEncoder("v1", enc, tok, "cpu")
    assert torch.allclose(le(texts), ref, atol=1e-6)
    n = LabelEncoder("v1", enc, tok, "cpu", normalize=True, max_length=128)(texts)
    assert torch.allclose(n.norm(dim=-1), torch.ones(2), atol=1e-5) and n.shape == ref.shape


def test_gonogo_decision_statistics():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("gng", pathlib.Path(__file__).parents[1] / "scripts" / "v21_gonogo.py")
    g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
    rows = ([{"doc": f"d{i}", "label": "a", "grp": "heldout", "match": True, "correct": i < 15} for i in range(20)]
            + [{"doc": f"e{i}", "label": "b", "grp": "heldout", "match": i < 10, "correct": i < 5} for i in range(20)]
            + [{"doc": "f", "label": "c", "grp": "heldout", "match": True, "correct": True}]
            + [{"doc": "s", "label": "x", "grp": "seen", "match": True, "correct": True}])
    s = g.summarize(rows)
    assert abs(s["heldout"]["macro_acc"] - (0.75 + 0.25) / 2) < 1e-9          # label c (n=1) excluded
    assert s["seen"]["acc"] == 1.0 and abs(g._macro(rows, ["a"]) - 0.75) < 1e-9


def test_predict_flat_skips_whitespace_only_spans(v1_dir, tmp_path, monkeypatch):
    """A segment made only of whitespace tokens trims to empty offsets; it must be skipped, not
    raised as an OffsetError (the v2.1 go/no-go eval crashed on one)."""
    from s1pii.schema import Doc
    from s1pii.v2 import flat as FL
    docs = generate(40, seed=6)
    out = FL.train_flat(v1_dir, docs, tmp_path / "flat", {"url"}, steps=2, batch=4, max_len=64, device="cpu", log=lambda *_: None)
    d = Doc("ws", "alpha beta gamma delta", [])
    real = FL.tokenize_doc

    def ws_tokenize(doc, tok):                       # make token 1 whitespace-only (empty trimmed offsets)
        td = real(doc, tok)
        td.offsets[1] = (td.offsets[1][0], td.offsets[1][0])
        return td
    monkeypatch.setattr(FL, "tokenize_doc", ws_tokenize)
    monkeypatch.setattr(FL, "span_logprobs", lambda crf, em, a, b, z, n, floor, max_len: [(i, i, 0, 0.9) for i in range(n)])
    L = LB.LabelSet("t", [LB.native_label("person")])
    pr = FL.predict_flat(v1_dir, out, [d], L, system="t", max_len=128, device="cpu", score="typed")
    assert pr["ws"] and all(s.end > s.start for s in pr["ws"])


def test_flatcrf_transitions_are_label_independent():
    """v21_diag ranks labels at a forced segment by emission sums; valid only if every
    transition/start/end potential is shared across labels (kind-level parameters)."""
    from s1pii.model.crf import tag
    from s1pii.v2.flat import FlatModel
    fm = FlatModel(8, dim=4)
    with torch.no_grad():
        fm.theta.normal_(); fm.start.normal_(); fm.end.normal_()
    T, S, E = fm.crf(5).potentials()
    for k in range(1, 5):
        for a, b in (("B", "I"), ("B", "E"), ("I", "I"), ("I", "E")):
            assert T[tag(a, k), tag(b, k)] == T[tag(a, 0), tag(b, 0)]
        for a in "BS":
            assert T[0, tag(a, k)] == T[0, tag(a, 0)] and S[tag(a, k)] == S[tag(a, 0)]
        for a in "ES":
            assert T[tag(a, k), 0] == T[tag(a, 0), 0] and E[tag(a, k)] == E[tag(a, 0)]
        for a, b in (("E", "B"), ("S", "S"), ("E", "S"), ("S", "B")):       # cross-label moves
            assert T[tag(a, k), tag(b, 0)] == T[tag(a, 0), tag(b, k)] == T[tag(a, 1 if k != 1 else 2), tag(b, k)]


def test_span_score_softmax_equals_crf_segment_conditional():
    """softmax over labels of v21_diag.span_scores == the CRF's exact segment marginals for that
    segment, normalized over labels (forward-backward via span_logprobs)."""
    import importlib.util, pathlib
    from s1pii.model.crf import span_logprobs
    from s1pii.v2.flat import FlatModel, FlatCRF
    spec = importlib.util.spec_from_file_location("v21diag", pathlib.Path(__file__).parents[1] / "scripts" / "v21_diag.py")
    g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
    torch.manual_seed(0)
    nt, n = 3, 7
    fm = FlatModel(8, dim=4)
    with torch.no_grad():
        fm.theta.normal_(); fm.start.normal_(); fm.end.normal_()
    em = torch.randn(1, n, 1 + 4 * nt, dtype=torch.float64)
    c = FlatCRF(nt, fm.theta.double(), fm.start.double(), fm.end.double())
    msk = torch.ones(1, n, dtype=torch.bool)
    alpha, logz = c._forward(em, msk); beta = c._backward(em, msk)
    segs = span_logprobs(c, em[0].numpy(), alpha[0].detach().numpy(), beta[0].detach().numpy(), float(logz[0]), n,
                         floor=1e-300, max_len=n)
    for i, j in ((2, 2), (1, 4), (0, 6)):
        p = np.array([sum(q for a, b, t, q in segs if (a, b, t) == (i, j, k)) for k in range(nt)])
        sc = g.span_scores(em[0].numpy(), i, j, nt)
        assert np.allclose(p / p.sum(), np.exp(sc - sc.max()) / np.exp(sc - sc.max()).sum(), atol=1e-8)


def test_v21_diag_end_to_end_on_fixtures(v1_dir, tmp_path, monkeypatch):
    import importlib.util, pathlib, shutil
    from s1pii import bench
    from s1pii.ledger import write_predictions, dataset_hash
    from s1pii.v2.flat import train_flat
    spec = importlib.util.spec_from_file_location("v21diag", pathlib.Path(__file__).parents[1] / "scripts" / "v21_diag.py")
    g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
    D = tmp_path / "drive"; G = D / "v21" / "gonogo"
    shutil.copytree(v1_dir, D / "models" / "no-nemotron-s1" / "final")
    docs = generate(40, seed=6)
    held = {"phone", "email"}
    for d in (D / "v2" / "flat" / "no-nemotron-s1", *(G / f"{m}-s{s}" for m in ("M1_v1labels", "M2_bgelabels") for s in (1, 2))):
        train_flat(D / "models" / "no-nemotron-s1" / "final", docs, d, held, steps=2, batch=4, max_len=64, device="cpu",
                   text_mode="sample", log=lambda *_: None)
    ev = generate(12, seed=8)
    write_jsonl(ev, G / "eval_docs.jsonl")
    calib = tmp_path / "nemotron-calib.jsonl"; write_jsonl(ev, calib)
    R = tmp_path / "r"
    # hybrid fixture: S1 finds every held-out span (label wrong), GLiNER types them correctly
    s1 = {d.doc_id: [Span(d.doc_id, x.start, x.end, "OTHER_PII", label_raw="person", score=0.9, source="s1")
                     for x in d.spans if x.label_raw in held] for d in ev}
    gl = {d.doc_id: [Span(d.doc_id, x.start, x.end, "OTHER_PII", label_raw=x.label_raw, score=0.9, source="gl")
                     for x in d.spans if x.label_raw in held] for d in ev}
    for sysn, pr in (("s1v2c3-A_no-nemotron-s1", s1), ("gliner25_base_zeroshot_c3", gl)):
        write_predictions(pr, R / "predictions_v2" / f"{sysn}.jsonl", {"system": sysn, "docs_path": str(calib),
                                                                     "dataset_hash": dataset_hash(ev)})
    monkeypatch.setenv("S1PII_RESULTS", str(R)); monkeypatch.setenv("S1PII_DATA", str(tmp_path))
    monkeypatch.setattr(bench, "split_paths", lambda name: {"calib": calib, "test": calib})
    labs = ["person", "account_number", "phone", "email", "date", "address", "url", "secret"]
    monkeypatch.setattr(g, "D", D); monkeypatch.setattr(g, "G", G)
    monkeypatch.setattr(LB, "load_heldout", lambda: {"labels": sorted(held), "nodes": sorted(held)})
    monkeypatch.setattr(LB, "c3_label_set", lambda desc=True: LB.LabelSet("t", [LB.native_label(x) for x in labs], desc))
    g.run()
    res = json.loads((G / "diag.json").read_text())
    assert res["exploratory"] is True and res["c3_reopenable"] is False
    assert res["n_heldout"] > 0 and set(res["models"]) == set(g.models())
    m = res["models"]["M1_v1labels-s1"]
    assert 0 <= m["heldout"]["top1"] <= 1 and m["heldout"]["median_rank"] >= 1
    assert set(m["prior_correction_acc"]) == {"none", "unseen_offset", "per_label_oracle"}
    assert "gap" in m["collapse"]["projected"] and "raw_heldout" in m["collapse"]["paraphrase_top1"]
    h = res["hybrid"]
    assert h["s1_match"] == 1.0 and h["hybrid_acc"] == 1.0 and h["gliner_acc"] == 1.0, h
