"""Linear-chain CRF over BIOES tags with hard transition constraints, partial-label training,
Viterbi decoding, forward-backward marginals and exact span probabilities.

Tags: 0 = O; for canonical type t (index into ``CANONICAL_TYPES``): B = 1+4t, I = 2+4t,
E = 3+4t, S = 4+4t. Invalid transitions (O->I, B->B, I->O, ...) and invalid start/end tags
get a large negative score, so every decoded or summed path is a well-formed BIOES
sequence.

Span confidence (the S1 contribution): for a span of type t over tokens i..j,

    P(y_i = B_t, y_{i+1..j-1} = I_t, y_j = E_t | x)            (S_t if i == j)

computed exactly from the forward and backward lattices. This is the probability of the
segment, not a Viterbi path score. Partial labels: positions whose gold is unknown
(IGNORE regions, spans cut by a window edge) allow every tag; the loss is
logZ - logZ_allowed, the negative log of the total probability of all allowed paths.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..schema import CANONICAL_TYPES

NT = len(CANONICAL_TYPES)
K = 1 + 4 * NT
NEG = -1e4


def tag(kind: str, t: int) -> int:
    return 1 + 4 * t + "BIES".index(kind)


def tag_kind(k: int) -> tuple[str, int]:
    if k == 0:
        return "O", -1
    return "BIES"[(k - 1) % 4], (k - 1) // 4


def constraint_masks(nt: int = NT) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(allowed_trans (K,K) prev->cur, allowed_start (K,), allowed_end (K,)) as bool, K = 1 + 4 nt."""
    K = 1 + 4 * nt
    tr = torch.zeros(K, K, dtype=torch.bool)
    begin_like = [0] + [tag("B", t) for t in range(nt)] + [tag("S", t) for t in range(nt)]
    for prev in range(K):
        kp, tp = tag_kind(prev)
        if kp in ("O", "E", "S"):
            for c in begin_like:
                tr[prev, c] = True
        else:  # B or I
            tr[prev, tag("I", tp)] = True
            tr[prev, tag("E", tp)] = True
    start = torch.zeros(K, dtype=torch.bool)
    start[begin_like] = True
    end = torch.zeros(K, dtype=torch.bool)
    end[[0] + [tag("E", t) for t in range(nt)] + [tag("S", t) for t in range(nt)]] = True
    return tr, start, end


class CRF(nn.Module):
    """``nt`` entity types (default: the 9 canonical types; v2 level 1 uses nt = 1)."""

    def __init__(self, nt: int = NT):
        super().__init__()
        self.nt, self.k = nt, 1 + 4 * nt
        K = self.k
        self.trans = nn.Parameter(torch.zeros(K, K))
        self.start = nn.Parameter(torch.zeros(K))
        self.end = nn.Parameter(torch.zeros(K))
        tr, st, en = constraint_masks(nt)
        self.register_buffer("tr_mask", tr, persistent=False)
        self.register_buffer("st_mask", st, persistent=False)
        self.register_buffer("en_mask", en, persistent=False)
        self.o_exit_bias = 0.0   # recall knob: added to O->B/S and start->B/S (0 for headline)

    def potentials(self):
        T = torch.where(self.tr_mask, self.trans, torch.full_like(self.trans, NEG))
        S = torch.where(self.st_mask, self.start, torch.full_like(self.start, NEG))
        E = torch.where(self.en_mask, self.end, torch.full_like(self.end, NEG))
        if self.o_exit_bias:
            ent = torch.zeros(self.k, dtype=torch.bool, device=T.device)
            ent[[tag("B", t) for t in range(self.nt)] + [tag("S", t) for t in range(self.nt)]] = True
            T = T.clone(); S = S.clone()
            T[0, ent] += self.o_exit_bias
            S[ent] += self.o_exit_bias
        return T, S, E

    # ---------------------------------------------------------------- lattices
    def _forward(self, em: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """alpha (B,L,K) and logZ (B,). ``mask`` (B,L) is left-aligned (valid prefix)."""
        T, S, E = self.potentials()
        B, L, _ = em.shape
        alphas = [S + em[:, 0]]
        for t in range(1, L):
            nxt = torch.logsumexp(alphas[-1].unsqueeze(2) + T.unsqueeze(0), dim=1) + em[:, t]
            alphas.append(torch.where(mask[:, t:t + 1], nxt, alphas[-1]))
        alpha = torch.stack(alphas, 1)
        lengths = mask.long().sum(1)
        last = alpha[torch.arange(B, device=em.device), lengths - 1]
        return alpha, torch.logsumexp(last + E, dim=1)

    def _backward(self, em: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        T, S, E = self.potentials()
        B, L, _ = em.shape
        betas = [None] * L
        K = self.k
        betas[L - 1] = E.expand(B, K)
        for t in range(L - 2, -1, -1):
            nxt = torch.logsumexp(T.unsqueeze(0) + (em[:, t + 1] + betas[t + 1]).unsqueeze(1), dim=2)
            betas[t] = torch.where(mask[:, t + 1:t + 2], nxt, E.expand(B, K))
        return torch.stack(betas, 1)

    def nll(self, em: torch.Tensor, mask: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        """Mean over the batch of -log P(allowed paths). ``allowed`` (B,L,K) bool."""
        em_a = torch.where(allowed, em, torch.full_like(em, NEG))
        B = em.shape[0]
        _, logz_all = self._forward(torch.cat([em, em_a], 0), torch.cat([mask, mask], 0))   # one pass, 2B rows
        return (logz_all[:B] - logz_all[B:]).mean()

    @torch.no_grad()
    def marginals(self, em: torch.Tensor, mask: torch.Tensor):
        """(alpha, beta, logZ) for span probabilities; tensors in float64 on CPU."""
        em = em.double()
        alpha, logz = self._forward(em, mask)
        beta = self._backward(em, mask)
        return alpha, beta, logz

    @torch.no_grad()
    def viterbi(self, em: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        T, S, E = self.potentials()
        B, L, _ = em.shape
        score = S + em[:, 0]
        back = []
        for t in range(1, L):
            cand = score.unsqueeze(2) + T.unsqueeze(0)
            best, idx = cand.max(1)
            nxt = best + em[:, t]
            m = mask[:, t:t + 1]
            score = torch.where(m, nxt, score)
            back.append(torch.where(m, idx, torch.arange(self.k, device=em.device).expand(B, self.k)))
        lengths = mask.long().sum(1).tolist()
        out = []
        for b in range(B):
            n = lengths[b]
            last = int((score[b] + E).argmax())
            path = [last]
            for t in range(n - 1, 0, -1):
                path.append(int(back[t - 1][b, path[-1]]))
            out.append(path[::-1])
        return out


def span_logprobs(crf: CRF, em: np.ndarray, alpha: np.ndarray, beta: np.ndarray, logz: float,
                  n: int, floor: float = 0.01, max_len: int = 64) -> list[tuple[int, int, int, float]]:
    """Exact span probabilities for one sequence of length n. Returns (i, j, type, prob) for
    every span with prob >= floor and length <= max_len tokens. Starts are pruned by the B
    marginal (lossless: a span's probability never exceeds its start-tag marginal); ends are
    evaluated vectorized over j."""
    T, _, _ = (x.detach().double().cpu().numpy() for x in crf.potentials())
    lf = np.log(floor)
    out = []
    for t in range(crf.nt):
        b, i_, e, s = tag("B", t), tag("I", t), tag("E", t), tag("S", t)
        ps = alpha[:n, s] + beta[:n, s] - logz
        for i in np.nonzero(ps >= lf)[0]:
            out.append((int(i), int(i), t, float(np.exp(min(ps[i], 0.0)))))
        pb = alpha[:n, b] + beta[:n, b] - logz
        c = T[i_, i_] + em[:n, i_]
        pref = np.concatenate([[0.0], np.cumsum(c)])
        endscore = T[i_, e] + em[:n, e] + beta[:n, e]            # for j >= i + 2
        for i in np.nonzero(pb >= lf)[0]:
            i = int(i)
            hi = min(n, i + max_len)
            if i + 1 < hi:
                lp1 = alpha[i, b] + T[b, e] + em[i + 1, e] + beta[i + 1, e] - logz
                if lp1 >= lf:
                    out.append((i, i + 1, t, float(np.exp(min(lp1, 0.0)))))
            if i + 2 < hi:
                js = np.arange(i + 2, hi)
                lp = (alpha[i, b] + T[b, i_] + em[i + 1, i_] + (pref[js] - pref[i + 2]) + endscore[js] - logz)
                for j in js[lp >= lf]:
                    out.append((i, int(j), t, float(np.exp(min(lp[j - i - 2], 0.0)))))
    return out
