import itertools

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from s1pii.model.crf import CRF, K, NT, tag, tag_kind, constraint_masks, span_logprobs, NEG


def small_crf(seed=0):
    torch.manual_seed(seed)
    crf = CRF()
    with torch.no_grad():
        crf.trans.normal_(0, 0.5); crf.start.normal_(0, 0.5); crf.end.normal_(0, 0.5)
    return crf


def valid(path):
    tr, st, en = constraint_masks()
    return bool(st[path[0]]) and bool(en[path[-1]]) and all(bool(tr[a, b]) for a, b in zip(path, path[1:]))


def path_score(crf, em, path):
    T, S, E = crf.potentials()
    s = S[path[0]] + em[0, path[0]]
    for i in range(1, len(path)):
        s = s + T[path[i - 1], path[i]] + em[i, path[i]]
    return s + E[path[-1]]


# brute force over a reduced tag alphabet: O + tags of the first type only keeps it tractable
SUB = [0, tag("B", 0), tag("I", 0), tag("E", 0), tag("S", 0)]


def brute(crf, em, n):
    em = em.clone()
    mask_other = torch.ones(K, dtype=torch.bool); mask_other[SUB] = False
    em[:, mask_other] = NEG
    paths = [p for p in itertools.product(SUB, repeat=n) if valid(p)]
    scores = torch.stack([path_score(crf, em[:n], p) for p in paths]).double()
    return em, paths, scores


def test_tags_roundtrip_and_constraints():
    for t in range(NT):
        for kind in "BIES":
            assert tag_kind(tag(kind, t)) == (kind, t)
    assert K == 37
    assert valid([tag("B", 0), tag("I", 0), tag("E", 0), 0, tag("S", 1)])
    assert not valid([0, tag("I", 0)])
    assert not valid([tag("B", 0), 0])
    assert not valid([tag("B", 0), tag("E", 1)])


def test_logz_marginals_spans_viterbi_against_brute_force():
    crf = small_crf()
    n = 5
    em = torch.randn(1, n, K) * 0.8
    em_b, paths, scores = brute(crf, em[0], n)
    em_b = em_b.unsqueeze(0)
    mask = torch.ones(1, n, dtype=torch.bool)
    alpha, beta, logz = crf.marginals(em_b, mask)
    assert float(logz[0]) == pytest.approx(float(torch.logsumexp(scores, 0)), abs=1e-6)
    probs = torch.softmax(scores, 0)
    # unary marginals
    for pos in range(n):
        for k in SUB:
            want = sum(float(p) for p, path in zip(probs, paths) if path[pos] == k)
            got = float(torch.exp(alpha[0, pos, k] + beta[0, pos, k] - logz[0]))
            assert got == pytest.approx(want, abs=1e-6)
    # exact span probabilities
    spans = span_logprobs(crf, em_b[0].double().numpy(), alpha[0].numpy(), beta[0].numpy(), float(logz[0]), n, floor=1e-9)
    got = {(i, j, t): p for i, j, t, p in spans if t == 0}
    for i in range(n):
        for j in range(i, n):
            seg = [tag("S", 0)] if i == j else [tag("B", 0)] + [tag("I", 0)] * (j - i - 1) + [tag("E", 0)]
            want = sum(float(p) for p, path in zip(probs, paths) if list(path[i:j + 1]) == seg)
            assert got.get((i, j, 0), 0.0) == pytest.approx(want, abs=1e-6), (i, j)
    # viterbi = argmax path
    best = paths[int(torch.argmax(scores))]
    assert crf.viterbi(em_b, mask)[0] == list(best)


def test_partial_label_nll_against_brute_force():
    crf = small_crf(1)
    n = 4
    em = torch.randn(1, n, K)
    em_b, paths, scores = brute(crf, em[0], n)
    allowed = torch.zeros(1, n, K, dtype=torch.bool)
    allowed[0, 0, 0] = True
    allowed[0, 1, SUB] = True                      # unknown position: any tag
    allowed[0, 2, tag("B", 0)] = True
    allowed[0, 3, tag("E", 0)] = True
    loss = crf.nll(em_b.unsqueeze(0), torch.ones(1, n, dtype=torch.bool), allowed)
    ok = [s for s, p in zip(scores, paths) if p[0] == 0 and p[2] == tag("B", 0) and p[3] == tag("E", 0)]
    want = float(torch.logsumexp(scores, 0) - torch.logsumexp(torch.stack(ok), 0))
    assert float(loss) == pytest.approx(want, abs=1e-5)
    loss.backward()
    assert crf.trans.grad is not None and torch.isfinite(crf.trans.grad).all()


def test_variable_lengths_match_unpadded():
    crf = small_crf(2)
    em = torch.randn(2, 6, K)
    mask = torch.tensor([[1] * 6, [1] * 3 + [0] * 3], dtype=torch.bool)
    a, b, z = crf.marginals(em, mask)
    a1, b1, z1 = crf.marginals(em[1:2, :3], torch.ones(1, 3, dtype=torch.bool))
    assert float(z[1]) == pytest.approx(float(z1[0]), abs=1e-9)
    assert torch.allclose(a[1, :3], a1[0], atol=1e-9) and torch.allclose(b[1, :3], b1[0], atol=1e-9)
    assert len(crf.viterbi(em, mask)[1]) == 3
