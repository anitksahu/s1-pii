import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from s1pii.schema import Doc, OTHER_PII, Span
from s1pii.s1d import labels as HL
from s1pii.s1d.data import assert_no_heldout_leakage, generate_questions
from s1pii.s1d.model import S1DModel, has_lm_head
from s1pii.s1d.packer import SPECIAL_TOKENS, pack_separate, pack_window
from s1pii.s1d.schema import Question, answer


class StubTokenizer:
    def __init__(self):
        self.vocab = {s: i + 1 for i, s in enumerate(SPECIAL_TOKENS)}
        self.unk_token_id = 0

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, 0)

    def __call__(self, text, **kwargs):
        ids = [8 + (ord(c) % 96) for c in text]
        if kwargs.get("truncation"):
            ids = ids[:kwargs["max_length"]]
        return {"input_ids": ids}


@pytest.fixture(scope="module")
def tiny():
    from transformers import Qwen3Config
    torch.manual_seed(7)
    cfg = Qwen3Config(vocab_size=128, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=2, head_dim=8,
                      max_position_embeddings=2048, use_cache=False, attention_dropout=0)
    return S1DModel.from_config(cfg).eval()


def packed(options=("person", "not personal information"), branches=("Ada", "555"), state="Ada called 555"):
    return pack_window(StubTokenizer(), state, options, branches)


def test_schema_choice_score_noul():
    with pytest.raises(ValueError): Question("choice", "x", options=("one",))
    score = Question("score", "rate", options=("low", "medium", "high"))
    got = answer(score, [0.2, 0.3, 0.5])
    assert got.argmax == "high" and got.score == pytest.approx(1.3)
    assert len(Question("noul", "is it PII?").options) == 2


def test_label_draw_deterministic_disjoint_and_frozen():
    a, b = HL.draw(), HL.draw()
    assert a == b and len(HL.eligible_names()) == 45
    assert len(a["dev_labels"]) == 5 and len(a["test_labels"]) == 10
    assert set(a["dev_labels"]).isdisjoint(a["test_labels"])
    assert a["min_spans"] == 300 and a["fallback_min_spans"] == 150
    assert a["replication_labels"] == HL.load()["replication_labels"]


def test_mask_has_no_cross_branch_edges():
    p = packed()
    a0, b0 = p.branch_ranges[0]; a1, b1 = p.branch_ranges[1]
    assert not p.dense_mask[a0:b0, a1:b1].any()
    assert not p.dense_mask[a1:b1, a0:b0].any()
    assert p.dense_mask[a0:b0, p.state_range[0]:p.state_range[1]].all()


def test_zero_cross_branch_gradient(tiny):
    p = packed()
    captured = {}
    def hook(_module, _inputs, output):
        output.retain_grad(); captured["embedding"] = output
    h = tiny.backbone.embed_tokens.register_forward_hook(hook)
    y = tiny(p).probabilities[0, 0]
    tiny.zero_grad(set_to_none=True); y.backward(); h.remove()
    a, b = p.branch_ranges[1]
    assert torch.count_nonzero(captured["embedding"].grad[0, a:b]) == 0


def test_branch_changes_with_state(tiny):
    a = tiny(packed(state="Ada called 555")).probabilities
    b = tiny(packed(state="A bank record says 555")).probabilities
    assert not torch.equal(a, b)


def test_option_permutation_equivariance_fp32(tiny):
    opts = ("person", "phone", "not personal information")
    a = tiny(packed(opts)).probabilities
    perm = [2, 0, 1]
    b = tiny(packed(tuple(opts[i] for i in perm))).probabilities
    inverse = [perm.index(i) for i in range(3)]
    assert torch.allclose(a, b[:, inverse], atol=1e-5, rtol=0)


def test_packed_equals_separate(tiny):
    tok = StubTokenizer(); state = "Ada called 555"; opts = ("person", "phone", "not personal information")
    branches = ("Ada", "555")
    together = tiny(pack_window(tok, state, opts, branches)).probabilities
    separate = torch.cat([tiny(x).probabilities for x in pack_separate(tok, state, opts, branches)])
    assert torch.allclose(together, separate, atol=1e-5, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FlexAttention bf16 is unavailable on CPU")
def test_bf16_permutation_and_separate_tolerances():
    from transformers import Qwen3Config
    cfg = Qwen3Config(vocab_size=128, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=2, head_dim=8,
                      max_position_embeddings=2048, use_cache=False, attention_dropout=0)
    model = S1DModel.from_config(cfg).cuda().bfloat16().eval(); tok = StubTokenizer()
    opts = ("person", "phone", "not personal information"); branches = tuple(f"q{i}" for i in range(200))
    p = pack_window(tok, "Ada called 555", opts, branches)
    a = model(p).probabilities.float(); perm = [2, 0, 1]
    b = model(pack_window(tok, "Ada called 555", tuple(opts[i] for i in perm), branches)).probabilities.float()
    inverse = [perm.index(i) for i in range(3)]; b = b[:, inverse]
    assert (a - b).abs().max() <= 2e-2
    assert (a.argmax(-1) == b.argmax(-1)).float().mean() >= 0.995
    separate = torch.cat([model(x).probabilities.float()
                          for x in pack_separate(tok, "Ada called 555", opts, branches)])
    assert (a - separate).abs().max() <= 2e-2
    assert (a.argmax(-1) == separate.argmax(-1)).float().mean() >= 0.995


def test_rope_receives_packed_positions_and_no_lm_head(tiny):
    p = packed(); seen = []
    def pre(_module, args, kwargs): seen.append(kwargs.get("position_ids", args[1] if len(args) > 1 else None).detach().cpu())
    handle = tiny.backbone.rotary_emb.register_forward_pre_hook(pre, with_kwargs=True)
    tiny(p); handle.remove()
    assert torch.equal(seen[0], p.position_ids.unsqueeze(0))
    assert not has_lm_head(tiny)


def test_heldout_spans_are_never_targets_or_hard_negatives(tmp_path):
    held = HL.load()["test_labels"][0]
    text = "secret@example.test ordinary"
    hs = Span("d", 0, 19, OTHER_PII, held, surface=text[:19])
    doc = Doc("d", text, (hs,), "nemotron", "train", "d")
    hard = lambda _d: [Span("d", 0, 19, OTHER_PII, "", surface=text[:19])]
    rows = generate_questions([doc], hard_negative_hook=hard, ledger_path=tmp_path / "ledger.jsonl")
    assert_no_heldout_leakage(rows)
    assert all(r.source_span != (0, 19) for r in rows)


def test_proposer_nt1_trains_one_step():
    from s1pii.model.s1 import S1Model
    class Encoder(torch.nn.Module):
        def __init__(self): super().__init__(); self.e = torch.nn.Embedding(16, 8)
        def forward(self, input_ids, attention_mask): return type("O", (), {"last_hidden_state": self.e(input_ids)})
    model = S1Model(Encoder(), 8, dropout=0, num_types=1)
    ids = torch.tensor([[1, 2, 3]]); allowed = torch.ones(1, 3, 5, dtype=torch.bool)
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "tok_mask": torch.ones(1, 3, dtype=torch.bool),
             "allowed": allowed, "n_prefix": torch.tensor([0]), "tok_len": torch.tensor([3])}
    opt = torch.optim.AdamW(model.parameters()); loss = model(batch); loss.backward(); opt.step()
    assert torch.isfinite(loss)


def test_64_branch_56_option_latency_smoke(tiny):
    p = pack_window(StubTokenizer(), "state", [f"o{i}" for i in range(56)], [f"q{i}" for i in range(64)])
    y = tiny(p).probabilities
    assert y.shape == (64, 56)


def test_stage0_dry_chain(tmp_path):
    env = {**os.environ, "DRIVE": str(tmp_path), "S1D_DRY": "1", "S1D_SKIP_INSTALL": "1",
           "PYTHON": sys.executable}
    run = subprocess.run(["bash", "scripts/s1d_chain.sh", "stage0"], env=env, capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr
    assert (tmp_path / "s1d" / "STATUS").read_text().startswith("DONE")
    assert len(list((tmp_path / "s1d" / "stores").glob("stage0-*.done"))) == 8
