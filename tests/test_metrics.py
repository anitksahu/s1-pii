import numpy as np
import pytest

from s1pii.schema import Doc, Span, PERSON, EMAIL, IGNORE
from s1pii.eval import metrics as M
from s1pii.eval import bootstrap as B
from s1pii.eval import calibration as C
from s1pii.eval.align import expand_to_words, reanchor
from s1pii.eval.evaluate import evaluate, tune_threshold, compare


def mk(text, gold, did="d"):
    return Doc(did, text, tuple(Span(did, s, e, lab) for s, e, lab in gold), cluster_id=did)


def test_leak_over_hand_computed():
    # alnum: "Ann" (PII, 3) " " "met" (non, 3) " " "Bo" (PII, 2) "!" -> 5 PII, 3 non
    d = mk("Ann met Bo!", [(0, 3, PERSON), (8, 10, PERSON)])
    preds = [Span("d", 0, 3, PERSON, score=0.9, source="sys"), Span("d", 4, 7, PERSON, score=0.4, source="sys")]
    st = M.doc_stats(d, preds)
    r = M.at_threshold(st.hist_pii, st.hist_non, 0.5)
    assert r["leak"] == pytest.approx(2 / 5) and r["over"] == 0.0
    r = M.at_threshold(st.hist_pii, st.hist_non, 0.3)
    assert r["leak"] == pytest.approx(2 / 5) and r["over"] == pytest.approx(1.0)


def test_ignore_excluded_everywhere():
    d = mk("Ann in Oslo", [(0, 3, PERSON), (7, 11, IGNORE)])
    preds = [Span("d", 7, 11, PERSON, score=1.0, source="s")]
    st = M.doc_stats(d, preds)
    assert st.hist_pii.sum() == 3 and st.hist_non.sum() == 2  # "in" only
    r = M.at_threshold(st.hist_pii, st.hist_non, 0.5)
    assert r["over"] == 0.0 and r["leak"] == 1.0


def test_punctuation_and_whitespace_not_counted():
    d = mk("a-b c", [(0, 3, PERSON)])
    st = M.doc_stats(d, [])
    assert st.hist_pii.sum() == 2 and st.hist_non.sum() == 1


def test_word_granularity_expansion():
    assert expand_to_words("call 4111-2222 now", 7, 9) == (5, 14)
    d = mk("x 4111-2222 y", [(2, 11, PERSON)])
    st = M.doc_stats(d, [Span("d", 3, 5, PERSON, score=1.0, source="s")])
    assert M.at_threshold(st.hist_pii, st.hist_non, 0.5)["leak"] == 0.0


def test_span_exposure_partial_card():
    d = mk("card 4111 1111 1111 1234", [(5, 24, "ACCOUNT_NUMBER")])
    preds = [Span("d", 5, 19, "ACCOUNT_NUMBER", score=1.0, source="s")]
    assert M.span_exposure(d, preds, 0.5) == (1, 1)


def test_curve_and_pauc():
    hp = np.zeros(M.NBINS); hn = np.zeros(M.NBINS)
    hp[1 + 900] = 8; hp[0] = 2          # 8 PII chars scored 0.9, 2 unpredicted
    hn[1 + 500] = 10; hn[0] = 90        # 10 non-PII chars scored 0.5
    leak, over = M.curve_from_hist(hp, hn)
    assert leak[900] == pytest.approx(0.2) and over[900] == 0.0
    assert leak[500] == pytest.approx(0.2) and over[500] == pytest.approx(0.1)
    assert leak[-1] == 1.0 and over[-1] == 0.0
    assert M.pauc(leak, over, 0.05) == pytest.approx(0.2)
    assert M.best_leak_at(leak, over, 0.0) == pytest.approx(0.2)


def test_pauc_nothing_masked_is_one():
    hp = np.zeros(M.NBINS); hp[0] = 5
    hn = np.zeros(M.NBINS); hn[0] = 5
    leak, over = M.curve_from_hist(hp, hn)
    assert M.pauc(leak, over) == pytest.approx(1.0)


def test_batched_curve_matches_single():
    rng = np.random.default_rng(0)
    hp = rng.integers(0, 5, (3, M.NBINS)); hn = rng.integers(0, 5, (3, M.NBINS))
    lb, ob = M.curve_from_hist(hp, hn)
    for i in range(3):
        l, o = M.curve_from_hist(hp[i], hn[i])
        assert np.allclose(lb[i], l) and np.allclose(ob[i], o)


def test_bootstrap_deterministic_and_identity():
    rng = np.random.default_rng(1)
    hp = rng.integers(0, 3, (20, M.NBINS)).astype(float); hn = rng.integers(0, 3, (20, M.NBINS)).astype(float)
    r1 = B.paired_bootstrap(hp, hn, hp, hn, B.pauc_stat(), n_boot=200, seed=3)
    r2 = B.paired_bootstrap(hp, hn, hp, hn, B.pauc_stat(), n_boot=200, seed=3)
    assert r1 == r2 and r1["diff"] == 0 and r1["ci_low"] == 0 and r1["ci_high"] == 0


def test_bootstrap_weights_match_direct_resample():
    rng = np.random.default_rng(2)
    hp = rng.integers(0, 3, (5, M.NBINS)).astype(float)
    W = B.cluster_weights(5, 3, seed=7)
    assert np.allclose((W @ hp)[0], sum(W[0, c] * hp[c] for c in range(5)))
    assert W.sum(1).tolist() == [5, 5, 5]


def test_holm():
    r = B.holm({"a": 0.01, "b": 0.04, "c": 0.03}, 0.05)
    assert r["a"]["reject"] and not r["c"]["reject"] and not r["b"]["reject"]
    assert r["a"]["p_holm"] == pytest.approx(0.03)
    assert r["c"]["p_holm"] == pytest.approx(0.06)
    assert r["b"]["p_holm"] == pytest.approx(0.06)


def test_span_prf_strict_partial_and_ignore():
    d = mk("Ann Lee met Bo in Oslo", [(0, 7, PERSON), (12, 14, PERSON), (18, 22, IGNORE)])
    preds = {"d": [Span("d", 0, 3, PERSON, score=1, source="s"), Span("d", 12, 14, PERSON, score=1, source="s"),
                   Span("d", 18, 22, PERSON, score=1, source="s"), Span("d", 8, 11, PERSON, score=1, source="s")]}
    strict = M.span_prf([d], preds, 0.5, typed=True, mode="strict")
    assert (strict["tp"], strict["fp"], strict["fn"]) == (1, 2, 1)
    part = M.span_prf([d], preds, 0.5, typed=False, mode="partial")
    assert (part["tp"], part["fp"], part["fn"]) == (2, 1, 0)


def test_consistency():
    d = mk("Ann and Ann and Bob", [(0, 3, PERSON), (8, 11, PERSON), (16, 19, PERSON)])
    assert M.consistency(d, [Span("d", 0, 3, PERSON, score=1, source="s")], 0.5) == (0, 1)
    assert M.consistency(d, [Span("d", 0, 3, PERSON, score=1, source="s"), Span("d", 8, 11, PERSON, score=1, source="s")], 0.5) == (1, 1)


def test_reanchor_drift():
    text = "Contact John Smith today"
    s = Span("d", 9, 19, PERSON, surface="John Smith", source="s")
    r = reanchor(text, s)
    assert (r.start, r.end) == (8, 18)
    assert reanchor(text, Span("d", 0, 3, PERSON, surface="Zed", source="s")) is None


def test_calibration_basics():
    d = mk("Ann met Bo", [(0, 3, PERSON), (8, 10, PERSON)])
    preds = {"d": [Span("d", 0, 3, PERSON, score=0.9, source="s"), Span("d", 4, 7, PERSON, score=0.6, source="s")]}
    c = C.candidates([d], preds)
    assert c.targets.tolist() == [1, 0] and c.uncovered_gold == 1 and c.total_gold == 2
    assert C.brier(np.array([1.0, 0.0]), np.array([1, 0])) == 0.0
    p = np.linspace(0.01, 0.99, 200); y = (np.random.default_rng(0).random(200) < p).astype(int)
    assert C.adaptive_ece(p, y) < 0.15
    T = C.fit_temperature(np.log(p / (1 - p)) * 3, y)
    assert 2 < T < 4.5


def test_evaluate_and_compare_end_to_end():
    docs = [mk(f"Ann {i} met Bob", [(0, 3, PERSON), (len(f'Ann {i} met '), len(f'Ann {i} met Bob'), PERSON)], f"d{i}") for i in range(30)]
    good = {d.doc_id: [Span(d.doc_id, s.start, s.end, PERSON, score=0.9, source="g") for s in d.spans] for d in docs}
    bad = {d.doc_id: [Span(d.doc_id, 0, 3, PERSON, score=0.9, source="b")] for d in docs}
    r = evaluate(docs, good, default_threshold=0.5, n_boot_ci=100)
    assert r["pauc"] == pytest.approx(0.0) and r["operating_points"]["default"]["span_exposure"] == 0
    cmp = compare(docs, good, bad, n_boot=200)
    assert cmp["diff"] < 0 and cmp["ci_high"] < 0
    assert tune_threshold(docs, good, 0.01) == 0.0
