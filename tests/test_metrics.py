import numpy as np
import pytest

from s1pii.schema import Doc, Span, PERSON, EMAIL, DATE, IGNORE, ACCOUNT_NUMBER
from s1pii.eval import metrics as M
from s1pii.eval import bootstrap as B
from s1pii.eval import calibration as C
from s1pii.eval.align import expand_to_words, reanchor
from s1pii.eval.evaluate import evaluate, tune_threshold, compare, IncompletePredictions, views


def mk(text, gold, did="d"):
    return Doc(did, text, tuple(Span(did, s, e, lab) for s, e, lab in gold), cluster_id=did)


def P(did, s, e, lab=PERSON, score=1.0):
    return Span(did, s, e, lab, score=score, source="sys")


def hist(doc, preds, allowed=None):
    return M.view(doc, preds, allowed).hist()


def test_leak_over_hand_computed():
    d = mk("Ann met Bo!", [(0, 3, PERSON), (8, 10, PERSON)])       # 5 PII alnum, 3 non
    hp, hn = hist(d, [P("d", 0, 3, score=0.9), P("d", 4, 7, score=0.4)])
    r = M.at_threshold(hp, hn, 0.5)
    assert r["leak"] == pytest.approx(2 / 5) and r["over"] == 0.0
    r = M.at_threshold(hp, hn, 0.3)
    assert r["leak"] == pytest.approx(2 / 5) and r["over"] == pytest.approx(1.0)
    r = M.at_threshold(hp, hn, 0.4)          # >= semantics on the grid
    assert r["over"] == pytest.approx(1.0)


def test_ignore_excluded_everywhere():
    d = mk("Ann in Oslo", [(0, 3, PERSON), (7, 11, IGNORE)])
    hp, hn = hist(d, [P("d", 7, 11)])
    assert hp.sum() == 3 and hn.sum() == 2
    assert M.at_threshold(hp, hn, 0.5)["over"] == 0.0


def test_punctuation_whitespace_not_counted():
    hp, hn = hist(mk("a-b c", [(0, 3, PERSON)]), [])
    assert hp.sum() == 2 and hn.sum() == 1


def test_word_expansion_is_alnum_runs_and_json_safe():
    assert expand_to_words("call 4111-2222 now", 6, 8) == (5, 9)
    text = '{"name":"Smith","ssn":"123-45-6789","note":"routine"}'
    a = text.index("123"); b = a + len("123-45-6789")
    d = mk(text, [(a, b, ACCOUNT_NUMBER)])
    hp, hn = hist(d, [P("d", a, b, ACCOUNT_NUMBER)])
    r = M.at_threshold(hp, hn, 0.5)
    assert r["leak"] == 0.0 and r["over"] == 0.0


def test_admissible_filter():
    d = mk("Ann on 2020-01-01", [(0, 3, PERSON)])
    preds = [P("d", 0, 3), P("d", 7, 17, DATE)]
    hp, hn = hist(d, preds, frozenset({PERSON}))
    assert M.at_threshold(hp, hn, 0.5)["over"] == 0.0
    hp, hn = hist(d, preds, None)
    assert M.at_threshold(hp, hn, 0.5)["over"] > 0


def test_span_exposure_partial_card_and_merged_gold():
    d = mk("card 4111 1111 1111 1234", [(5, 24, ACCOUNT_NUMBER), (5, 14, ACCOUNT_NUMBER)])
    v = M.view(d, [P("d", 5, 19, ACCOUNT_NUMBER)])
    assert v.gold == [(5, 24, ACCOUNT_NUMBER)]
    assert M.span_exposure(v, 0.5) == (1, 1)


def test_curve_and_exact_pauc():
    hp = np.zeros(M.NBINS); hn = np.zeros(M.NBINS)
    hp[1 + 900] = 8; hp[0] = 2
    hn[1 + 500] = 10; hn[0] = 90
    leak, over = M.curve_from_hist(hp, hn)
    assert leak[900] == pytest.approx(0.2) and over[900] == 0.0
    assert leak[500] == pytest.approx(0.2) and over[500] == pytest.approx(0.1)
    assert (leak[-1], over[-1]) == (1.0, 0.0)
    assert M.pauc(leak, over, 0.05) == pytest.approx(0.2)


def test_pauc_step_exact():
    # points: (over=0, leak=1), (0.01, 0.5), (0.03, 0.1) -> area/0.05 = (0.01*1 + 0.02*0.5 + 0.02*0.1)/0.05
    leak = np.array([1.0, 0.5, 0.1]); over = np.array([0.0, 0.01, 0.03])
    assert M.pauc(leak, over, 0.05) == pytest.approx((0.01 + 0.01 + 0.002) / 0.05)
    assert M.pauc(np.array([1.0]), np.array([0.0])) == pytest.approx(1.0)


def test_batched_curve_matches_single():
    rng = np.random.default_rng(0)
    hp = rng.integers(0, 5, (3, M.NBINS)); hn = rng.integers(0, 5, (3, M.NBINS))
    lb, ob = M.curve_from_hist(hp, hn)
    pb = M.pauc(lb, ob)
    for i in range(3):
        l, o = M.curve_from_hist(hp[i], hn[i])
        assert np.allclose(lb[i], l) and np.allclose(ob[i], o) and pb[i] == pytest.approx(M.pauc(l, o))


def test_bootstrap_deterministic_identity_and_weights():
    rng = np.random.default_rng(1)
    hp = rng.integers(0, 3, (20, M.NBINS)).astype(float); hn = rng.integers(0, 3, (20, M.NBINS)).astype(float)
    r1 = B.paired_bootstrap(hp, hn, hp, hn, B.pauc_stat(), n_boot=300, seed=3, batch=128)
    r2 = B.paired_bootstrap(hp, hn, hp, hn, B.pauc_stat(), n_boot=300, seed=3, batch=128)
    assert r1 == r2 and r1["diff"] == 0 and r1["ci_low"] == 0 and r1["ci_high"] == 0
    W = B.cluster_weights(5, 7, seed=7)
    assert W.shape == (7, 5) and (W.sum(1) == 5).all()


def test_multi_seed_mean():
    rng = np.random.default_rng(2)
    h = lambda: (rng.integers(0, 3, (10, M.NBINS)).astype(float), rng.integers(0, 3, (10, M.NBINS)).astype(float))
    a = [h(), h(), h()]; b = [h()]
    r = B.paired_bootstrap_multi(a, b, B.pauc_stat(), n_boot=100, seed=0)
    assert r["a"] == pytest.approx(np.mean(r["a_per_seed"]))


def test_holm_and_c0():
    r = B.holm({"a": 0.01, "b": 0.04, "c": 0.03}, 0.05)
    assert r["a"]["reject"] and not r["c"]["reject"] and not r["b"]["reject"]
    assert r["a"]["p_holm"] == pytest.approx(0.03) and r["b"]["p_holm"] == pytest.approx(0.06)
    res = {"ds1|g1": {"p_value": 0.001, "diff": -0.1}, "ds1|g2": {"p_value": 0.001, "diff": -0.1},
           "ds2|g1": {"p_value": 0.001, "diff": 0.1}, "ds3|g1": {"p_value": 0.5, "diff": -0.1}}
    c0 = B.c0_decision(res, needed=1)
    assert c0["datasets_won"] == ["ds1"] and c0["c0_holds"] and c0["family_size"] == 4


def test_span_prf_order_independent_strict_partial_ignore():
    d = mk("Ann Lee met Bo in Oslo", [(0, 7, PERSON), (12, 14, PERSON), (18, 22, IGNORE)])
    preds = [P("d", 0, 3), P("d", 12, 14), P("d", 18, 22), P("d", 8, 11), P("d", 0, 7)]
    for order in (preds, list(reversed(preds))):
        v = M.view(d, order)
        s = M.span_prf([v], 0.5, typed=True, mode="strict")
        assert (s["tp"], s["fp"], s["fn"]) == (2, 2, 0)
        p = M.span_prf([v], 0.5, typed=False, mode="partial")
        assert (p["tp"], p["fp"], p["fn"]) == (2, 2, 0)


def test_consistency_variants():
    d = mk("Ann and Ann and Bob", [(0, 3, PERSON), (8, 11, PERSON), (16, 19, PERSON)])
    c = M.consistency(M.view(d, [P("d", 0, 3)]), 0.5)
    assert (c["all_masked"], c["groups"], c["cond_num"], c["cond_den"]) == (0, 1, 0, 1)
    c = M.consistency(M.view(d, []), 0.5)
    assert c["cond_den"] == 0
    c = M.consistency(M.view(d, [P("d", 0, 3), P("d", 8, 11)]), 0.5)
    assert c["all_masked"] == 1


def test_reanchor_drift():
    text = "Contact John Smith today"
    r = reanchor(text, Span("d", 9, 19, PERSON, surface="John Smith", source="s"))
    assert (r.start, r.end) == (8, 18)
    assert reanchor(text, Span("d", 0, 3, PERSON, surface="Zed", source="s")) is None


def test_calibration_one_to_one_and_cluster_ci():
    d = mk("Ann met Bo", [(0, 3, PERSON), (8, 10, PERSON)])
    v = M.view(d, [P("d", 0, 3, score=0.9), P("d", 0, 3, score=0.7), P("d", 4, 7, score=0.6)])
    c = C.candidates([v])
    assert c.targets.tolist() == [1, 0, 0] and c.uncovered_gold == 1 and c.total_gold == 2
    p = np.linspace(0.01, 0.99, 200); y = (np.random.default_rng(0).random(200) < p).astype(int)
    assert C.adaptive_ece(p, y) < 0.15
    cc = C.Candidates(p, y, ["PERSON"] * 200, [f"c{i % 20}" for i in range(200)])
    lo, hi = C.cluster_ci(cc, C.adaptive_ece, n_boot=100)
    assert lo <= C.adaptive_ece(p, y) <= hi + 0.05
    T = C.fit_temperature(np.log(p / (1 - p)) * 3, y)
    assert 2 < T < 4.5
    iso = C.TypeIsotonic(min_n=10).fit(cc)
    assert iso.transform(np.array([0.5]), ["PERSON"]).shape == (1,)


def test_evaluate_compare_tune_and_completeness():
    docs = [mk(f"Ann {i} met Bob", [(0, 3, PERSON), (len(f"Ann {i} met "), len(f"Ann {i} met Bob"), PERSON)], f"d{i}")
            for i in range(30)]
    good = {d.doc_id: [P(d.doc_id, s.start, s.end, score=0.9) for s in d.spans] for d in docs}
    bad = {d.doc_id: [P(d.doc_id, 0, 3, score=0.9)] for d in docs}
    r = evaluate(docs, good, dataset="pii_trace", default_threshold=0.5, dev_threshold=0.2, n_boot_ci=100)
    assert r["pauc"] == pytest.approx(0.0) and r["operating_points"]["default"]["span_exposure"] == 0
    assert "dev_tuned" in r["operating_points"]
    cmp = compare(docs, [good, good], [bad], dataset="pii_trace", n_boot=200)
    assert cmp["diff"] < 0 and cmp["ci_high"] < 0
    assert tune_threshold(docs, good, "pii_trace", 0.01) == 0.0
    assert tune_threshold(docs, {d.doc_id: [P(d.doc_id, 4, 5, score=0.9)] for d in docs}, "pii_trace", 0.0) == pytest.approx(0.901)
    with pytest.raises(IncompletePredictions):
        views(docs, {}, "pii_trace")


def test_span_prf_max_cardinality_tie():
    d = mk("John Smith x", [(0, 4, PERSON), (5, 10, PERSON)])
    for order in ([P("d", 0, 10), P("d", 5, 10)], [P("d", 5, 10), P("d", 0, 10)]):
        s = M.span_prf([M.view(d, order)], 0.5, typed=False, mode="partial")
        assert (s["tp"], s["fp"], s["fn"]) == (2, 0, 0)
