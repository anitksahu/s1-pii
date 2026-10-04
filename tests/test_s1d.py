import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from s1pii.schema import Doc, OTHER_PII, Span
from s1pii.s1d import labels as HL
from s1pii.s1d.data import assert_no_heldout_leakage, generate_questions, pack_training_questions
from s1pii.s1d.model import S1DModel, apply_lora, has_lm_head, prepare_tokenizer
from s1pii.s1d.packer import BranchText, SPECIAL_TOKENS, pack_separate, pack_window
from s1pii.s1d.schema import Question, answer


class StubTokenizer:
    def __init__(self):
        self.vocab = {s: i + 1 for i, s in enumerate(SPECIAL_TOKENS)}
        self.unk_token_id = 0
        self.pad_token_id = 0

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, 0)

    def __call__(self, text, **kwargs):
        ids = [8 + (ord(c) % 96) for c in text]
        if kwargs.get("truncation"):
            ids = ids[:kwargs["max_length"]]
        out = {"input_ids": ids}
        if kwargs.get("return_offsets_mapping"):
            out["offset_mapping"] = [(i, i + 1) for i in range(len(ids))]
        return out

    def add_special_tokens(self, spec):
        for token in spec["additional_special_tokens"]:
            self.vocab.setdefault(token, len(self.vocab) + 1)
        return len(spec["additional_special_tokens"])

    def __len__(self):
        return 128


@pytest.fixture(scope="module")
def tiny():
    from transformers import Qwen3Config
    torch.manual_seed(7)
    cfg = Qwen3Config(vocab_size=128, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
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


def test_trainable_c3_vocabulary_and_ood_only_names_are_pinned():
    from s1pii.v2.labels import c3_label_set
    expected = {
        "account_number", "age", "biometric_identifier", "city", "company_name", "credit_debit_card",
        "date_of_birth", "device_identifier", "email", "employee_id", "employment_status", "gender",
        "health_plan_beneficiary_number", "ipv6", "license_plate", "mac_address", "national_id", "password",
        "phone_number", "pin", "sexuality", "ssn", "swift_bic", "tax_id", "time", "unique_id", "url",
        "user_name", "vehicle_identifier",
    }
    c3 = {row.name for row in c3_label_set(False).labels}
    assert set(HL.training_vocabulary()) & c3 == expected
    ood = {"code", "datetime", "dem", "id_num", "loc", "misc", "org", "phone_num", "private_address",
           "private_date", "private_email", "private_person", "private_phone", "private_url", "quantity", "username"}
    assert HL.ood_only_names() == ood
    assert set(HL.load()["replication_labels"]).isdisjoint(HL.training_vocabulary())
    assert set(HL.training_vocabulary(remove_parents=True)).isdisjoint(HL.PARENT_LABELS)


def test_replication_exclusions_are_parsed_once(monkeypatch):
    text = HL.V21_CONFIG.read_text()
    reads = []
    class Source:
        def read_text(self):
            reads.append(1)
            return text
    HL._replication_exclusions.cache_clear()
    monkeypatch.setattr(HL, "V21_CONFIG", Source())
    cfg = HL.load()
    for _ in range(20):
        HL.excluded("email", cfg)
    assert len(reads) == 1
    HL._replication_exclusions.cache_clear()


def test_nearest_trained_neighbours_are_complete_and_ledgered(tmp_path):
    def local_embeddings(texts):
        return torch.tensor([[1.0 + (sum(map(ord, text)) % 17), float(i + 1), 0.5]
                             for i, text in enumerate(texts)])
    rows = HL.record_nearest_trained_neighbours(tmp_path / "ledger.jsonl", embedder=local_embeddings)
    assert set(rows) == set(HL.load()["test_labels"])
    assert all(set(row) == {"name_and_paraphrases", "description"} for row in rows.values())
    assert all(len(row[kind]) == 3 for row in rows.values()
               for kind in ("name_and_paraphrases", "description"))
    assert all(item["label"] in HL.training_vocabulary() for row in rows.values()
               for kind in ("name_and_paraphrases", "description") for item in row[kind])
    ledger = json.loads((tmp_path / "ledger.jsonl").read_text().splitlines()[-1])
    assert ledger["test_labels"] == rows


def test_mask_has_no_cross_branch_edges():
    p = packed()
    a0, b0 = p.branch_ranges[0]; a1, b1 = p.branch_ranges[1]
    assert not p.dense_mask[a0:b0, a1:b1].any()
    assert not p.dense_mask[a1:b1, a0:b0].any()
    assert p.dense_mask[a0:b0, p.state_range[0]:p.state_range[1]].all()


def test_kev_mask_is_branch_local_and_options_see_own_question():
    p = pack_window(StubTokenizer(), "state", ("one", "two"),
                    (BranchText("span-a", "left-a"), BranchText("span-b", "left-b")), layout="kev")
    q0, q1 = p.question_ranges
    opts0, opts1 = p.option_ranges[:2], p.option_ranges[2:]
    for a, b in opts0:
        assert p.dense_mask[a:b, q0[0]:q0[1]].all()
        assert not p.dense_mask[a:b, q1[0]:q1[1]].any()
    d0, d1 = (p.decide_indices[0].item(), p.decide_indices[1].item())
    assert all(p.dense_mask[d0, a:b].all() for a, b in opts0)
    assert all(not p.dense_mask[d0, a:b].any() for a, b in opts1)
    assert all(p.dense_mask[d1, a:b].all() for a, b in opts1)


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


def test_flex_no_grad_matches_eager_and_mask_is_material():
    from transformers import Qwen3Config, Qwen3Model
    cfg = Qwen3Config(vocab_size=128, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=2, head_dim=8,
                      max_position_embeddings=2048, use_cache=False, attention_dropout=0)
    eager = S1DModel.from_config(cfg).eval()
    flex_cfg = Qwen3Config(**cfg.to_dict()); flex_cfg._attn_implementation = "flex_attention"
    flex = S1DModel(Qwen3Model(flex_cfg), 16).eval(); flex.load_state_dict(eager.state_dict())
    p = packed()
    try:
        with torch.no_grad():
            a = eager(p).probabilities
            b = flex(p).probabilities
    except (NotImplementedError, RuntimeError) as exc:
        if "CPU" in str(exc) or "platform" in str(exc):
            pytest.skip(f"FlexAttention CPU kernel unavailable: {exc}")
        raise
    with torch.no_grad():
        unmasked = eager.backbone(input_ids=p.input_ids.unsqueeze(0), position_ids=p.position_ids.unsqueeze(0),
                                  attention_mask=None, return_dict=True).last_hidden_state
        masked = eager(p, output_hidden_states=True).hidden_states
    assert torch.allclose(a, b, atol=1e-5, rtol=0)
    assert not torch.allclose(masked, unmasked[0])


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


def test_forward_many_matches_individual_windows(tiny):
    tok = StubTokenizer(); opts = ("person", "phone", "not personal information")
    windows = [pack_window(tok, state, opts, (branch,), make_block_mask=False)
               for state, branch in (("Ada called 555", "Ada"), ("Jo called 444", "444"))]
    with torch.no_grad():
        individual = [tiny(window).probabilities for window in windows]
        batched = [output.probabilities for output in tiny.forward_many(windows)]
    assert all(torch.allclose(a, b, atol=1e-5, rtol=0) for a, b in zip(individual, batched))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="batched FlexAttention requires CUDA")
def test_forward_many_flex_cuda_matches_individual_windows():
    from s1pii.s1d.packer import flex_block_mask
    from transformers import Qwen3Config
    cfg = Qwen3Config(vocab_size=128, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=2, head_dim=8,
                      max_position_embeddings=2048, use_cache=False, attention_dropout=0)
    model = S1DModel.from_config(cfg).cuda().bfloat16().eval(); tok = StubTokenizer()
    options = ("person", "phone", "not personal information")
    windows = [pack_window(tok, state, options, (branch,), device="cuda", make_block_mask=False)
               for state, branch in (("Ada called 555", "Ada"), ("Jo called 444", "444"))]
    with torch.no_grad():
        batched = [output.probabilities.float() for output in model.forward_many(windows)]
        individual = []
        for window in windows:
            window.attention_mask = flex_block_mask(window.dense_mask, device="cuda")
            individual.append(model(window).probabilities.float())
    assert all((a - b).abs().max() <= 2e-2 for a, b in zip(individual, batched))
    assert all((a.argmax(-1) == b.argmax(-1)).float().mean() >= 0.995
               for a, b in zip(individual, batched))


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


def test_lora_and_reserved_token_rows():
    from transformers import Qwen3Config
    cfg = Qwen3Config(vocab_size=128, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                      num_attention_heads=2, num_key_value_heads=2, head_dim=8, use_cache=False)
    model = S1DModel.from_config(cfg); tok = StubTokenizer()
    indices = prepare_tokenizer(tok, model)
    rows_before = model.backbone.get_input_embeddings().num_embeddings
    apply_lora(model, indices)
    trainable = [name for name, p in model.named_parameters() if p.requires_grad]
    assert rows_before == 128 and indices == [1, 2, 3, 4, 5, 6]
    assert any("lora_" in name for name in trainable)
    assert any("trainable_tokens" in name for name in trainable)
    assert not has_lm_head(model)


def test_heldout_spans_are_never_targets_or_hard_negatives(tmp_path):
    held = HL.load()["test_labels"][0]
    text = "secret@example.test ordinary"
    hs = Span("d", 0, 19, OTHER_PII, held, surface=text[:19])
    doc = Doc("d", text, (hs,), "nemotron", "train", "d")
    hard = lambda _d: [Span("d", 0, 19, OTHER_PII, "", surface=text[:19])]
    rows = generate_questions([doc], hard_negative_hook=hard, ledger_path=tmp_path / "ledger.jsonl")
    assert_no_heldout_leakage(rows)
    assert all(r.source_span != (0, 19) for r in rows)
    assert not [r for r in rows if r.question.type.value in ("noul", "score")]


def test_heldout_doc_has_no_document_targets_and_hard_negatives_avoid_all_gold(tmp_path):
    held = HL.load()["test_labels"][0]
    text = "held ordinary"
    hs = Span("h", 0, 4, OTHER_PII, held, surface="held")
    doc = Doc("h", text, (hs,), "nemotron", "train", "h")
    hard = lambda _d: [Span("h", 0, 4, OTHER_PII, "", surface="held"),
                       Span("h", 5, 13, OTHER_PII, "", surface="ordinary")]
    rows = generate_questions([doc], hard_negative_hook=hard, ledger_path=tmp_path / "ledger")
    assert not [r for r in rows if r.question.type.value in ("noul", "score")]
    assert all(r.source_span != (0, 4) for r in rows)


def test_training_question_windowing_uses_descriptions_and_left_context(tmp_path):
    text = "x" * 20 + "ada@example.test"
    span = Span("w", 20, len(text), OTHER_PII, "email", surface="ada@example.test")
    rows = generate_questions([Doc("w", text, (span,), "synthetic_conv", "train", "w")],
                              seed=3, ledger_path=tmp_path / "ledger")
    choice = [r for r in rows if r.question.type.value == "choice"]
    packed_rows = pack_training_questions(choice, StubTokenizer(), make_block_mask=False)
    assert packed_rows and packed_rows[0][0].true_length < len(packed_rows[0][0].input_ids)


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


def test_proposer_fp32_master_weights_under_autocast():
    from s1pii.s1d.proposer import proposer_model
    class Encoder(torch.nn.Module):
        def __init__(self): super().__init__(); self.e = torch.nn.Embedding(16, 8)
        def forward(self, input_ids, attention_mask): return type("O", (), {"last_hidden_state": self.e(input_ids)})
    model = proposer_model(Encoder(), 8, dropout=0)
    ids = torch.tensor([[1, 2, 3]]); allowed = torch.ones(1, 3, 5, dtype=torch.bool)
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
             "tok_mask": torch.ones(1, 3, dtype=torch.bool), "allowed": allowed,
             "n_prefix": torch.tensor([0]), "tok_len": torch.tensor([3])}
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert torch.isfinite(model(batch))
    assert {p.dtype for p in model.parameters()} == {torch.float32}


def test_stage1_inference_autocast_handles_bf16_activations_with_fp32_head():
    from s1pii.s1d.run import _inference_probabilities
    class MixedDtypeModel(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.head = torch.nn.Linear(4, 2)
        def forward(self, _window):
            logits = self.head(torch.ones(1, 4, dtype=torch.bfloat16))
            return type("Output", (), {"probabilities": logits.softmax(-1)})
    model = MixedDtypeModel().eval()
    with pytest.raises(RuntimeError, match="same dtype"):
        model(None)
    probabilities = _inference_probabilities(model, None, torch.device("cpu"), bf16=True)
    assert len(probabilities) == 2
    assert sum(probabilities) == pytest.approx(1.0, abs=5e-3)


def test_proposer_example_chunks_are_resumable(tmp_path, monkeypatch):
    from s1pii.s1d import proposer
    docs = [Doc(f"d{i}", "plain", (), "synthetic", "train", f"d{i}") for i in range(3)]
    class Clock:
        def __init__(self): self.ticks = []
        def tick(self, **values): self.ticks.append(values)
    cache = tmp_path / "examples.pt"
    clock = Clock()
    first = proposer._cached_examples(docs, StubTokenizer(), cache=cache, config=HL.load(),
                                      clock=clock, control_path=tmp_path / "CONTROL", chunk_size=1)
    assert len(first) == 3 and len(clock.ticks) == 3
    monkeypatch.setattr(proposer, "examples", lambda *a, **k: pytest.fail("cached chunk was rebuilt"))
    second = proposer._cached_examples(docs, StubTokenizer(), cache=cache, config=HL.load(),
                                       clock=Clock(), control_path=tmp_path / "CONTROL", chunk_size=1)
    assert [row.doc_id for row in second] == [row.doc_id for row in first]


def test_unit_clock_records_without_stopping_at_former_cap(tmp_path):
    from s1pii.s1d.stage0 import UnitClock
    from s1pii.s1d.train import GPUHours
    config = {"stages": {"stage0": {"cap_a100_hours": 0}}}
    clock = UnitClock(tmp_path, config, "bounded")
    clock.last -= 1
    clock.tick(substep="chunk")
    assert GPUHours(tmp_path / "gpu_hours.jsonl", 0, "stage0").used() > 0


def test_gpu_hours_are_accounting_only(tmp_path):
    from s1pii.s1d.stage0 import UnitClock
    from s1pii.s1d.train import GPUHours
    path = tmp_path / "gpu_hours.jsonl"
    path.write_text(json.dumps({"stage": "stage0", "hours": 6.225}) + "\n")
    meter = GPUHours(path, 5, "stage0")
    meter.reserve(100)
    clock = UnitClock(tmp_path, {"stages": {"stage0": {"cap_a100_hours": 5}}}, "override")
    clock.last -= 1
    clock.tick(substep="continues")
    assert meter.used() > 6.225


def test_all_stages_are_configured_uncapped():
    import yaml
    from s1pii.s1d import run as runner
    config = yaml.safe_load((Path(runner.__file__).parents[1] / "configs" / "s1d.yaml").read_text())
    assert all(stage.get("enforce_cap") is False for stage in config["stages"].values())


def test_gate_g1_caps_candidates_and_is_non_blocking():
    from s1pii.s1d.proposer import gate_g1
    candidates = [(i, i + 1, float(100 - i)) for i in range(70)]
    got = gate_g1({"w": [(0, 1), (63, 64), (64, 65)]}, {"w": candidates})
    assert got == {"exact_boundary_recall": pytest.approx(2 / 3), "hits": 2,
                   "gold": 3, "candidate_cap": 64, "blocking": False}


def test_long_probe_window_keeps_options_assistant_and_final_letter_scores():
    from s1pii.s1d.stage0 import _prompt, restricted_letter_log_probs, span_window
    class ChatTokenizer(StubTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["add_generation_prompt"] and not kwargs["enable_thinking"]
            return messages[0]["content"] + "\n<assistant>"
    tok = ChatTokenizer(); cfg = HL.load(); names = cfg["dev_labels"][:2]
    state = "x" * 800 + "TARGET" + "y" * 800
    row = {"doc_id": "long", "state": state, "start": 800, "end": 806,
           "surface": "TARGET", "label": names[0], "options": names, "target": 0}
    window = span_window(tok, row, 512); prompt = _prompt(tok, window)
    assert "TARGET" in window["state"] and "Options:" in prompt
    assert all(name.replace("_", " ") in prompt for name in names)
    assert prompt.endswith("<assistant>")
    logits = torch.zeros(1, 3, 12); logits[0, 0, 2] = 100; logits[0, -1, 3] = 5
    assert restricted_letter_log_probs(logits, [2, 3]).argmax(-1).item() == 1


def test_kev_scoring_uses_probability_argmax_and_validates_keys():
    from s1pii.s1d.stage0 import accuracy_summary, predictions_from_probabilities
    rows = [{"label": "a", "options": ["a", "none"], "target": 0},
            {"label": "b", "options": ["b", "none"], "target": 1}]
    predictions, vectors = predictions_from_probabilities(
        rows, [{"a": 0.9, "none": 0.1}, {"b": 0.2, "none": 0.8}])
    assert predictions == [0, 1] and vectors[1] == [0.2, 0.8]
    assert accuracy_summary(rows, predictions)["macro_accuracy"] == 1
    with pytest.raises(ValueError):
        predictions_from_probabilities(rows[:1], [{"wrong": 1.0}])


def test_64_branch_56_option_latency_smoke(tiny):
    p = pack_window(StubTokenizer(), "state", [f"o{i}" for i in range(56)], [f"q{i}" for i in range(64)])
    y = tiny(p).probabilities
    assert y.shape == (64, 56)


def test_latency_harness_tiny_cpu_dry():
    from s1pii.s1d.latency import benchmark
    result = benchmark({}, dry=True, warmup=0, repeats=1)
    assert result["branches"] == 64 and result["options"] == 56
    assert result["implementation_version"] == 4 and result["proposer_included"] is True
    assert result["proposer"]["mode"] == "dry"
    assert result["proposer_p95_ms_per_window"] == result["proposer"]["p95_ms_per_window"]
    row = result["models"]["tiny-random"]
    assert row["p95_ms_per_window"] > 0
    # End-to-end p95 is the decision model plus the proposer (0 in the dry harness).
    assert row["p95_ms_per_window"] == row["model_p95_ms_per_window"] + row["proposer_p95_ms_per_window"]


def test_stage0_dry_chain(tmp_path):
    env = {**os.environ, "DRIVE": str(tmp_path), "S1D_DRY": "1", "S1D_SKIP_INSTALL": "1",
           "PYTHON": sys.executable}
    run = subprocess.run(["bash", "scripts/s1d_chain.sh", "stage0"], env=env, capture_output=True, text=True)
    log_path = tmp_path / "s1d_dry" / "logs" / "stage0.log"
    log = log_path.read_text() if log_path.exists() else "<stage0 log was not created>"
    assert run.returncode == 0, run.stdout + run.stderr + "\n--- stage0.log ---\n" + log
    assert not (tmp_path / "s1d").exists()
    assert (tmp_path / "s1d_dry" / "STATUS").read_text().startswith("DONE")
    assert len(list((tmp_path / "s1d_dry" / "stores").glob("stage0-*.done"))) == 8
    probe = json.loads((tmp_path / "s1d_dry" / "stores" / "stage0-prompted_probe.json").read_text())
    assert probe["questions"] > 0
    assert probe["chance"] == pytest.approx(1 / probe["options"])


def test_colab_cpu_test_cell_hides_gpu_checks_imports_and_streams_failures(capsys):
    notebook = json.loads(Path("notebooks/06_s1d.ipynb").read_text())
    source = "".join(notebook["cells"][1]["source"])
    assert "'CUDA_VISIBLE_DEVICES': ''" in source
    assert "from transformers import BertModel, Qwen3Model" in source
    assert "'torchao'" in source
    assert "is_torchao_available();" in source
    assert "stdout=subprocess.PIPE" in source
    assert "stderr=subprocess.STDOUT" in source
    assert "for line in proc.stdout" in source
    assert "full output is above" in source
    assert "CalledProcessError" not in source

    chain = Path("scripts/s1d_chain.sh").read_text()
    assert "torchvision torchaudio torchtext torchao" in chain
    assert "is_torchao_available; is_torchao_available()" in chain
    assert "torch.cuda.is_available()" in chain
    assert "starting $STAGE" in chain

    monitor = Path("scripts/s1d_monitor.py").read_text()
    assert 'read(ROOT / "CURRENT")' in monitor
    assert "max(logs, key=lambda path: path.stat().st_mtime)" in monitor

    launch_source = "".join(notebook["cells"][2]["source"])
    assert "Runtime > Change runtime type > GPU" in launch_source
    assert "torch.cuda.get_device_name(0)" in launch_source
    assert "STARTING {stage}" in launch_source
    assert launch_source.index("STARTING {stage}") < launch_source.index("subprocess.Popen")
    assert launch_source.index("run_visible([sys.executable, '-c', cuda_probe])") < launch_source.index("subprocess.Popen")

    helper_source = source.split("\nrun_visible([", 1)[0]
    namespace = {"subprocess": subprocess}
    exec(compile(helper_source, "notebook-cell-2", "exec"), namespace)
    command = [
        sys.executable,
        "-c",
        "import sys; print('visible stdout'); print('visible stderr', file=sys.stderr); raise SystemExit(7)",
    ]
    with pytest.raises(RuntimeError, match="exit status 7; full output is above"):
        namespace["run_visible"](command)
    output = capsys.readouterr().out
    assert "visible stdout" in output
    assert "visible stderr" in output


@pytest.mark.skipif(torch.cuda.is_available(), reason="exercises the no-CUDA Stage 0 preflight")
def test_non_dry_chain_refuses_before_stage0_without_cuda(tmp_path):
    env = {**os.environ, "DRIVE": str(tmp_path), "S1D_SKIP_INSTALL": "1", "PYTHON": sys.executable}
    run = subprocess.run(["bash", "scripts/s1d_chain.sh", "stage0"], env=env, capture_output=True, text=True)
    root = tmp_path / "s1d"
    assert run.returncode == 19
    assert (root / "STATUS").read_text().startswith("FAILED gpu")
    assert "CUDA is unavailable" in (root / "logs" / "stage0.log").read_text()
    assert not list((root / "stores").glob("stage0-*.done"))


def test_non_dry_stage0_never_marks_placeholder_done(tmp_path, monkeypatch):
    from s1pii.s1d import run as runner
    monkeypatch.setattr(runner, "_unit_census", lambda c, o: {"passed": True})
    monkeypatch.setattr(runner, "_unit_revisions", lambda c, o: {"model": "revision"})
    with pytest.raises(NotImplementedError):
        runner.run("stage0", tmp_path, dry=False)
    assert not (tmp_path / "stores" / "stage0-proposer_all.done").exists()


@pytest.mark.parametrize("stage", ["stage2"])
def test_future_stage_placeholders_never_write_done(tmp_path, stage):
    from s1pii.s1d.run import run
    (tmp_path / f"APPROVED_{stage}").write_text("approved")
    stores = tmp_path / "stores"; stores.mkdir()
    first = {"stage2": "train_final"}[stage]
    (stores / f"{stage}-{first}.json").write_text(json.dumps({"status": "entrypoint-ready"}))
    (stores / f"{stage}-{first}.done").write_text("legacy")
    with pytest.raises(NotImplementedError):
        run(stage, tmp_path, dry=False)
    assert {p.name for p in stores.glob(f"{stage}-*.done")} == {f"{stage}-{first}.done"}


def test_stage0_macro_stop_rule_fires_before_latency(tmp_path, monkeypatch):
    from s1pii.s1d import run as runner
    assert runner.UNITS["stage0"] == ("label_draw", "census", "revisions", "prompted_probe",
                                      "kev_baseline", "proposer_all",
                                      "proposer_no_nemotron", "latency")
    monkeypatch.setattr(runner, "_unit_label_draw", lambda c, o: {"ok": True})
    monkeypatch.setattr(runner, "_unit_census", lambda c, o: {"passed": True})
    monkeypatch.setattr(runner, "_unit_revisions", lambda c, o: {"ok": True})
    monkeypatch.setattr(runner, "_unit_proposer", lambda c, o, v: {"variant": v})
    def questions(_ctx, _out, name):
        return {"chance": 0.2, "accuracy": 0.9, "macro_accuracy": 0.1,
                "unit": name}
    called = []
    monkeypatch.setattr(runner, "_unit_questions", questions)
    monkeypatch.setattr(runner, "_unit_latency", lambda c, o: called.append(True) or {"ok": True})
    assert runner.run("stage0", tmp_path, dry=False) == 5
    assert not called
    assert (tmp_path / "stores" / "stage0-kev_baseline.done").exists()
    assert not (tmp_path / "stores" / "stage0-proposer_all.done").exists()
    assert not (tmp_path / "stores" / "stage0-proposer_no_nemotron.done").exists()
    assert not (tmp_path / "stores" / "stage0-latency.done").exists()
    rule_rows = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()
                 if json.loads(line).get("unit") == "s1d_stage0_stop_rule"]
    assert len(rule_rows) == 1 and rule_rows[0]["statistic"] == "macro_accuracy_over_dev_labels"


def test_gpu_hours_never_stop_above_recorded_reference(tmp_path):
    from s1pii.s1d.train import GPUHours
    path = tmp_path / "gpu_hours.jsonl"
    path.write_text(json.dumps({"stage": "stage0", "hours": 100}) + "\n")
    GPUHours(path, 5, "stage0").reserve(100)


def test_prompted_option_scorer_reuses_prefix_and_returns_56_probabilities():
    from s1pii.s1d.infer import prompted_option_distribution
    class Tokenizer:
        def __call__(self, text, **kwargs):
            if text == "prompt": ids = [1, 2]
            else: ids = [3 + int(text.removeprefix("option"))]
            return {"input_ids": torch.tensor([ids])}
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.anchor = torch.nn.Parameter(torch.zeros(())); self.calls = 0
        def forward(self, input_ids, **kwargs):
            self.calls += 1
            logits = torch.arange(64, dtype=torch.float32).repeat(1, input_ids.shape[1], 1)
            return type("Output", (), {"logits": logits, "past_key_values": ((torch.zeros(1),),)})
    model = Model()
    probabilities = prompted_option_distribution(model, Tokenizer(), "prompt",
                                                  [f"option{i}" for i in range(56)])
    assert probabilities.shape == (56,)
    assert probabilities.sum().item() == pytest.approx(1.0)
    assert model.calls == 1


def test_prompted_option_scorer_batches_multitoken_continuations():
    from s1pii.s1d.infer import prompted_option_distribution
    class Cache:
        def batch_repeat_interleave(self, repeats): self.repeats = repeats
    class Tokenizer:
        pad_token_id = 0
        def __call__(self, text, **kwargs):
            ids = [1, 2] if text == "prompt" else [3] + [4] * int(text[-1])
            return {"input_ids": torch.tensor([ids])}
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.anchor = torch.nn.Parameter(torch.zeros(())); self.shapes = []
        def forward(self, input_ids, past_key_values=None, **kwargs):
            self.shapes.append(tuple(input_ids.shape))
            logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 16)
            return type("Output", (), {"logits": logits, "past_key_values": Cache()})
    model = Model()
    probabilities = prompted_option_distribution(model, Tokenizer(), "prompt", ["o0", "o1", "o2", "o3"])
    assert probabilities.tolist() == pytest.approx([0.25] * 4)
    assert model.shapes == [(1, 2), (3, 3)]


def test_prompted_option_continuation_batches_are_memory_bounded():
    from s1pii.s1d.infer import prompted_option_distribution
    class Cache:
        def batch_repeat_interleave(self, repeats): self.repeats = repeats
    class Tokenizer:
        pad_token_id = 0
        def __call__(self, text, **kwargs):
            ids = [1, 2] if text == "prompt" else [3, 4]
            return {"input_ids": torch.tensor([ids])}
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.anchor = torch.nn.Parameter(torch.zeros(())); self.batch_sizes = []
        def forward(self, input_ids, past_key_values=None, **kwargs):
            self.batch_sizes.append(input_ids.shape[0])
            logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 8)
            return type("Output", (), {"logits": logits, "past_key_values": Cache()})
    model = Model()
    prompted_option_distribution(model, Tokenizer(), "prompt", [f"option-{i}" for i in range(18)])
    assert model.batch_sizes == [1, 16, 2]


def test_prompted_scorer_batches_prompts_and_bounds_total_cache_expansion():
    from s1pii.s1d.infer import prompted_option_distributions
    class Cache:
        def batch_repeat_interleave(self, repeats): self.repeats = repeats
    class Tokenizer:
        pad_token_id = 0
        def __call__(self, text, **kwargs):
            ids = [1] * (2 + int(text[-1])) if text.startswith("prompt") else [3, 4]
            return {"input_ids": torch.tensor([ids])}
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.anchor = torch.nn.Parameter(torch.zeros(())); self.batch_sizes = []
        def forward(self, input_ids, past_key_values=None, **kwargs):
            self.batch_sizes.append(input_ids.shape[0])
            logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 8)
            return type("Output", (), {"logits": logits, "past_key_values": Cache()})
    model = Model()
    probabilities = prompted_option_distributions(
        model, Tokenizer(), [f"prompt{i}" for i in range(4)], [f"option-{i}" for i in range(18)],
        max_expanded_batch=16)
    assert probabilities.shape == (4, 18)
    assert model.batch_sizes == [4, 16, 16, 16, 16, 8]
    assert probabilities.sum(1).tolist() == pytest.approx([1.0] * 4)


def test_batched_prompted_scores_match_full_sequence_qwen_reference():
    from s1pii.s1d.infer import prompted_option_distributions
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(11)
    config = Qwen3Config(vocab_size=64, hidden_size=16, intermediate_size=32,
                         num_hidden_layers=1, num_attention_heads=2,
                         num_key_value_heads=2, head_dim=8, use_cache=True,
                         attention_dropout=0)
    model = Qwen3ForCausalLM(config).eval()
    mapping = {"p0": [5, 6, 7], "p1": [8, 9, 10, 11, 12],
               "a": [13], "bb": [14, 15], "ccc": [16, 17, 18]}
    class Tokenizer:
        pad_token_id = 0
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([mapping[text]])}
    tokenizer = Tokenizer(); prompts = ["p0", "p1"]; options = ["a", "bb", "ccc"]
    actual = prompted_option_distributions(model, tokenizer, prompts, options, max_expanded_batch=4)
    expected = []
    with torch.no_grad():
        for prompt in prompts:
            prompt_ids = mapping[prompt]; scores = []
            for option in options:
                option_ids = mapping[option]
                sequence = torch.tensor([prompt_ids + option_ids[:-1]])
                logits = model(input_ids=sequence, use_cache=False).logits[0].float().log_softmax(-1)
                positions = torch.arange(len(prompt_ids) - 1, len(prompt_ids) - 1 + len(option_ids))
                scores.append(logits[positions, torch.tensor(option_ids)].mean())
            expected.append(torch.stack(scores).softmax(0))
    assert torch.allclose(actual, torch.stack(expected), atol=1e-5, rtol=0)


def test_evaluation_windows_group_branches_by_document_and_preserve_order():
    from s1pii.s1d.data import span_evaluation_questions
    from s1pii.s1d.run import _evaluation_window_specs
    label = HL.load()["dev_labels"][0]
    text = "first and second"
    spans = (Span("d", 0, 5, OTHER_PII, label, surface="first"),
             Span("d", 10, 16, OTHER_PII, label, surface="second"))
    rows = span_evaluation_questions([Doc("d", text, spans, "nemotron", "calib", "d")], {},
                                     labels=[label], require_equal_negatives=False)
    specs = _evaluation_window_specs(rows, StubTokenizer(), layout="shared")
    assert len(specs) == 1
    _state, options, branches, indices = specs[0]
    assert len(options) == 56 and len(branches) == 2 and indices == (0, 1)


def test_grouped_evaluation_resumes_from_partial_window_cache(tmp_path):
    from s1pii.s1d.data import span_evaluation_questions
    from s1pii.s1d.run import (_eval_cache_path, _score_loaded_trained,
                               _write_eval_cache)
    label = HL.load()["dev_labels"][0]
    docs = []
    for index in range(2):
        text = f"secret{index}"
        span = Span(f"d{index}", 0, len(text), OTHER_PII, label, surface=text)
        docs.append(Doc(f"d{index}", text, (span,), "nemotron", "calib", f"d{index}"))
    rows = span_evaluation_questions(docs, {}, labels=[label], require_equal_negatives=False)
    ctx = {"root": tmp_path, "dry": True}
    key, split, fingerprint = "trained-test", "dev", "fingerprint"
    first = [1 / 56] * 56
    _write_eval_cache(_eval_cache_path(ctx, key, split), fingerprint,
                      [first, None], completed=1, total=2)
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.anchor = torch.nn.Parameter(torch.zeros(())); self.calls = 0
        def forward(self, packed):
            self.calls += 1
            probabilities = torch.full((len(packed.decide_indices), 56), 1 / 56)
            return type("Output", (), {"probabilities": probabilities})
    model = Model()
    probabilities = _score_loaded_trained(ctx, model, StubTokenizer(), rows, layout="shared",
                                          key=key, split=split, fingerprint=fingerprint)
    assert model.calls == 1
    assert probabilities[0] == first and probabilities[1] == pytest.approx(first)


def test_dev_question_builder_has_56_options_and_equal_not_pii_rows():
    from s1pii.s1d.data import span_evaluation_questions
    label = HL.load()["dev_labels"][0]
    text = "secret ordinary"
    gold = Span("d", 0, 6, OTHER_PII, label, surface="secret")
    candidate = Span("d", 7, 15, OTHER_PII, "candidate", surface="ordinary")
    rows = span_evaluation_questions([Doc("d", text, (gold,), "nemotron", "calib", "d")],
                                     {"d": [candidate]}, labels=[label])
    assert len(rows) == 2 and sum(row.hard_negative for row in rows) == 1
    assert all(len(row.question.options) == 56 for row in rows)
    assert rows[1].question.options[rows[1].target].name == "not personal information"


def test_dev_question_builder_uses_all_available_negatives_when_unbalanced():
    from s1pii.s1d.data import span_evaluation_questions
    label = HL.load()["dev_labels"][0]
    text = "gold1 gold2 candidate"
    gold = (Span("d", 0, 5, OTHER_PII, label, surface="gold1"),
            Span("d", 6, 11, OTHER_PII, label, surface="gold2"))
    candidate = Span("d", 12, 21, OTHER_PII, "candidate", surface="candidate")
    doc = Doc("d", text, gold, "nemotron", "calib", "d")
    with pytest.raises(ValueError, match="need 2 .* found 1"):
        span_evaluation_questions([doc], {"d": [candidate]}, labels=[label])
    rows = span_evaluation_questions([doc], {"d": [candidate]}, labels=[label],
                                     require_equal_negatives=False)
    assert len(rows) == 3
    assert sum(row.hard_negative for row in rows) == 1


def test_layout_ablation_uses_same_16_branch_windows_for_both_layouts(tmp_path):
    from s1pii.s1d.data import generate_questions, pack_layout_ablation_questions
    text = "email ada@example.test phone 555-0102"
    spans = (Span("d", 6, 22, OTHER_PII, "email", surface="ada@example.test"),
             Span("d", 29, 37, OTHER_PII, "phone_number", surface="555-0102"))
    questions = generate_questions([Doc("d", text, spans, "synthetic_conv", "train", "d")],
                                   seed=4, ledger_path=tmp_path / "ledger")
    shared = pack_layout_ablation_questions(questions, StubTokenizer(), windows=2, seed=9,
                                            layout="shared", make_block_mask=False)
    kev = pack_layout_ablation_questions(questions, StubTokenizer(), windows=2, seed=9,
                                         layout="kev", make_block_mask=False)
    assert all(len(targets) == 16 and len(packed.decide_indices) == 16
               and len(packed.option_indices) == (56 if packed.layout == "shared" else 16 * 56)
               for packed, targets in shared + kev)
    assert [target for _packed, target in shared] == [target for _packed, target in kev]


def test_stage1_reads_nemotron_calibration_never_test(tmp_path, monkeypatch):
    from s1pii import bench
    from s1pii.s1d.run import _stage1_docs
    calib = tmp_path / "calib.jsonl"; test = tmp_path / "test.jsonl"
    doc = Doc("calib-only", "plain", (), "nemotron", "calib", "calib-only")
    calib.write_text(doc.to_json() + "\n")
    test.write_text("this is deliberately not JSON\n")
    monkeypatch.setattr(bench, "split_paths", lambda _name: {"calib": calib, "test": test})
    assert [row.doc_id for row in _stage1_docs(False)] == ["calib-only"]


def test_stage1_proposer_weights_sha_requires_manifest(tmp_path):
    from s1pii.s1d.run import _proposer_weights_sha
    path = tmp_path / "final"; path.mkdir()
    (path / "s1_manifest.json").write_text(json.dumps({"weights_sha256": "deadbeef"}))
    assert _proposer_weights_sha(path) == "deadbeef"
    (path / "s1_manifest.json").write_text(json.dumps({"other": 1}))
    with pytest.raises(RuntimeError):
        _proposer_weights_sha(path)


def test_stage1_candidate_cache_round_trips_without_gpu(tmp_path):
    from s1pii.s1d.run import _CANDIDATE_MINING_VERSION, _candidate_cache_path, _proposer_candidates
    root = tmp_path
    path = root / "models" / "proposer-no-nemotron" / "final"; path.mkdir(parents=True)
    (path / "s1_manifest.json").write_text(json.dumps({"weights_sha256": "w-sha"}))
    docs = [Doc("d0", "alpha beta", (), "nemotron", "calib", "d0")]
    cache_path = _candidate_cache_path(root, docs, "w-sha", "calibration")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    span = Span("d0", 0, 5, OTHER_PII, "cand", score=0.7, surface="alpha")
    cache_path.write_text(json.dumps({"version": 1, "mining_version": _CANDIDATE_MINING_VERSION,
                                      "count": 1, "candidates": {"d0": [span.to_dict()]}}))
    # A cache hit returns before any model load, so no CUDA is required to reuse candidates.
    out = _proposer_candidates(root, docs, False, cache_name="calibration")
    assert [s.to_dict() for s in out["d0"]] == [span.to_dict()]
    # Identity is keyed on the proposer weights: a different sha maps to a different cache file.
    assert _candidate_cache_path(root, docs, "other-sha", "calibration") != cache_path


def test_stage1_candidate_mining_stays_cpu_and_meters_without_cap(tmp_path, monkeypatch):
    from s1pii.s1d import run as runner
    import s1pii.model.train as v1train
    import s1pii.model.predict as v1predict
    root = tmp_path
    path = root / "models" / "proposer-no-nemotron" / "final"; path.mkdir(parents=True)
    (path / "s1_manifest.json").write_text(json.dumps({"weights_sha256": "w-sha"}))
    (root / "PHASE").write_text("GPU")   # a prior GPU phase must be cleared by mining

    class _StubPredictor:
        def __init__(self, *a, **k):
            pass

        def predict_docs(self, batch):
            # Five non-overlapping candidates per doc proves there is no four-candidate cap.
            return ({d.doc_id: [Span(d.doc_id, 10 + 2 * i, 11 + 2 * i, OTHER_PII, "cand",
                                     score=0.5 + i / 100, surface="x") for i in range(5)]
                     for d in batch}, None)

    monkeypatch.setattr(v1train, "load_exported",
                        lambda *a, **k: (object(), object(), {"weights_sha256": "w-sha"}))
    monkeypatch.setattr(v1predict, "S1Predictor", _StubPredictor)
    docs = [Doc(f"d{i}", "plain text body", (), "nemotron", "calib", f"d{i}") for i in range(3)]
    out = runner._proposer_candidates(root, docs, False, cache_name="training")
    assert all(len(out[f"d{i}"]) == 5 for i in range(3))              # no per-doc cap
    assert (root / "PHASE").read_text() == "CPU"                      # idle-watchdog-safe phase
    assert [s.score for s in out["d0"]] == sorted((s.score for s in out["d0"]), reverse=True)
    rows = [json.loads(line) for line in (root / "gpu_hours.jsonl").read_text().splitlines() if line.strip()]
    assert any(r["unit"] == "stage1-candidate-mining" and r["stage"] == "stage1" for r in rows)


def test_stage1_candidate_cache_rejects_stale_mining_version(tmp_path, monkeypatch):
    from s1pii.s1d import run as runner
    import s1pii.model.train as v1train
    import s1pii.model.predict as v1predict
    root = tmp_path
    path = root / "models" / "proposer-no-nemotron" / "final"; path.mkdir(parents=True)
    (path / "s1_manifest.json").write_text(json.dumps({"weights_sha256": "w-sha"}))
    docs = [Doc("d0", "alpha beta", (), "nemotron", "calib", "d0")]
    cache_path = runner._candidate_cache_path(root, docs, "w-sha", "calibration")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    stale = Span("d0", 0, 5, OTHER_PII, "stale", score=0.9, surface="alpha")
    cache_path.write_text(json.dumps({"version": 1, "mining_version": runner._CANDIDATE_MINING_VERSION - 1,
                                      "count": 1, "candidates": {"d0": [stale.to_dict()]}}))

    class _StubPredictor:
        def __init__(self, *a, **k):
            pass

        def predict_docs(self, batch):
            return ({d.doc_id: [Span(d.doc_id, 1, 2, OTHER_PII, "fresh", score=0.3, surface="l")]
                     for d in batch}, None)

    monkeypatch.setattr(v1train, "load_exported",
                        lambda *a, **k: (object(), object(), {"weights_sha256": "w-sha"}))
    monkeypatch.setattr(v1predict, "S1Predictor", _StubPredictor)
    out = runner._proposer_candidates(root, docs, False, cache_name="calibration")
    assert [s.label_raw for s in out["d0"]] == ["fresh"]              # recomputed, not the stale cache
    reread = json.loads(cache_path.read_text())
    assert reread["mining_version"] == runner._CANDIDATE_MINING_VERSION   # rewritten at current version


def test_latency_selection_guard_requires_proposer_included():
    from s1pii.s1d.run import _require_end_to_end_latency
    _require_end_to_end_latency({"proposer_included": True}, False)      # ok
    _require_end_to_end_latency({}, True)                                # dry is exempt
    with pytest.raises(RuntimeError):
        _require_end_to_end_latency({"models": {}}, False)


def test_latency_proposer_dry_is_download_free_measurement():
    from s1pii.s1d.latency import _proposer_window_latency_ms
    result = _proposer_window_latency_ms(None, {}, dry=True, warmup=0, repeats=2)
    assert result["mode"] == "dry" and result["windows"] == 1
    assert result["p95_ms_per_window"] >= 0
    assert set(result) >= {"mode", "p95_ms_per_window", "p95_ms_per_document",
                           "candidates_per_window", "windows", "repeats"}


def test_latency_proposer_measures_exported_pipeline_not_bare_encoder(tmp_path, monkeypatch):
    from s1pii.s1d import latency as lat
    import s1pii.model.train as v1train
    import s1pii.model.predict as v1predict
    path = tmp_path / "models" / "proposer-no-nemotron" / "final"; path.mkdir(parents=True)
    (path / "s1_manifest.json").write_text(json.dumps({"weights_sha256": "w-sha"}))
    calls = {"load_exported": 0, "predict_docs": 0}

    class _WSTokenizer:
        def __call__(self, text, **k):
            return {"input_ids": text.split()}

    class _Report:
        def __init__(self, windows):
            self.windows = windows

    class _StubPredictor:
        size = 510

        def __init__(self, *a, **k):
            pass

        def predict_docs(self, docs):
            calls["predict_docs"] += 1
            d = docs[0]
            # 100 candidates proves the timed path applies the <=64-per-window selection; the
            # report says exactly one window, which the benchmark requires.
            spans = [Span(d.doc_id, 2 * i, 2 * i + 1, OTHER_PII, "crf", score=i / 200, surface="x")
                     for i in range(100)]
            return ({d.doc_id: spans}, _Report(1))

    def _stub_load_exported(p, device=None):
        calls["load_exported"] += 1
        assert Path(p) == path       # the exported trained proposer, not a hub encoder
        return object(), _WSTokenizer(), {"weights_sha256": "w-sha"}

    monkeypatch.setattr(v1train, "load_exported", _stub_load_exported)
    monkeypatch.setattr(v1predict, "S1Predictor", _StubPredictor)
    result = lat._proposer_window_latency_ms(tmp_path, {}, dry=False, warmup=0, repeats=2)
    # A bare-encoder proxy would never load_exported, run predict_docs, or cap to 64 candidates.
    assert result["mode"] == "measured-exported"
    assert result["candidates_per_window"] == 64 and result["windows"] == 1
    assert result["window_tokens"] <= result["window_capacity_tokens"] == 510   # one near-full window
    assert result["p95_ms_per_window"] >= 0 and result["weights_sha256"] == "w-sha"
    assert calls["load_exported"] == 1 and calls["predict_docs"] >= 2


def test_latency_one_window_text_fits_capacity_unlike_old_workload():
    from s1pii.s1d.latency import _one_window_text

    class _WSTokenizer:
        def __call__(self, text, **k):
            return {"input_ids": text.split()}

    tok, size = _WSTokenizer(), 510
    text = _one_window_text(tok, size)
    n = len(tok(text)["input_ids"])
    assert n <= size                 # exactly one window's worth of content tokens
    assert n >= size - 20            # near-capacity, not a tiny document
    # The previous hard-coded workload was >= 570 tokens and would have broken this invariant.
    assert n < 570


def test_latency_proposer_rejects_multi_window(tmp_path, monkeypatch):
    from s1pii.s1d import latency as lat
    import s1pii.model.train as v1train
    import s1pii.model.predict as v1predict
    path = tmp_path / "models" / "proposer-no-nemotron" / "final"; path.mkdir(parents=True)
    (path / "s1_manifest.json").write_text(json.dumps({"weights_sha256": "w-sha"}))

    class _WSTokenizer:
        def __call__(self, text, **k):
            return {"input_ids": text.split()}

    class _Report:
        windows = 2

    class _StubPredictor:
        size = 510

        def __init__(self, *a, **k):
            pass

        def predict_docs(self, docs):
            d = docs[0]
            return ({d.doc_id: [Span(d.doc_id, 0, 1, OTHER_PII, "crf", score=0.5, surface="x")]},
                    _Report())

    monkeypatch.setattr(v1train, "load_exported",
                        lambda *a, **k: (object(), _WSTokenizer(), {"weights_sha256": "w-sha"}))
    monkeypatch.setattr(v1predict, "S1Predictor", _StubPredictor)
    # The benchmark reads report.windows and must refuse to label a multi-window duration per-window.
    with pytest.raises(RuntimeError):
        lat._proposer_window_latency_ms(tmp_path, {}, dry=False, warmup=0, repeats=1)


def test_archive_stale_run_never_clobbers(tmp_path):
    from s1pii.s1d.run import _archive_stale_run
    run_dir = tmp_path / "0.6B-s1"

    def _make(marker):
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "manifest.json").write_text(json.dumps({"hashes": {"questions": "SAME"}}))
        (run_dir / "marker.txt").write_text(marker)

    _make("first")
    first = _archive_stale_run(run_dir)
    _make("second")            # same questions hash -> same base archive name, different content
    second = _archive_stale_run(run_dir)
    assert first != second
    assert (first / "marker.txt").read_text() == "first"     # original archive untouched
    assert (second / "marker.txt").read_text() == "second"   # second archived alongside


def _write_stage1_run(run_dir, hashes, *, done):
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps({"hashes": hashes}))
    (run_dir / "checkpoint.pt").write_text("ckpt")
    if done:
        (run_dir / "done").write_text("1")


def test_train_run_reuses_only_matching_completed_run(tmp_path):
    from s1pii.s1d.run import _reuse_or_reset_run
    exp = {"questions": "qh", "revision": "rev", "layout": "shared"}
    run_dir = tmp_path / "0.6B-s1"
    _write_stage1_run(run_dir, exp, done=True)
    assert _reuse_or_reset_run(run_dir, exp) is True
    assert _reuse_or_reset_run(run_dir, exp) is True          # idempotent
    assert list(tmp_path.glob("*.stale-*")) == []


def test_train_run_archives_completed_mismatch(tmp_path):
    from s1pii.s1d.run import _reuse_or_reset_run
    run_dir = tmp_path / "0.6B-s1"
    _write_stage1_run(run_dir, {"questions": "OLD", "revision": "rev", "layout": "shared"}, done=True)
    exp = {"questions": "NEW", "revision": "rev", "layout": "shared"}
    assert _reuse_or_reset_run(run_dir, exp) is False
    assert not run_dir.exists()                               # canonical cleared for a fresh run
    stale = list(tmp_path.glob("*.stale-*"))
    assert len(stale) == 1 and (stale[0] / "manifest.json").exists()   # recoverable


def test_train_run_archives_checkpoint_only_mismatch(tmp_path):
    from s1pii.s1d.run import _reuse_or_reset_run
    run_dir = tmp_path / "0.6B-s1"
    _write_stage1_run(run_dir, {"questions": "OLD", "revision": "rev", "layout": "shared"}, done=False)
    exp = {"questions": "NEW", "revision": "rev", "layout": "shared"}
    assert _reuse_or_reset_run(run_dir, exp) is False
    assert not run_dir.exists()
    assert len(list(tmp_path.glob("*.stale-*"))) == 1


def test_train_run_resumes_matching_incomplete_run(tmp_path):
    from s1pii.s1d.run import _reuse_or_reset_run
    exp = {"questions": "qh", "revision": "rev", "layout": "shared"}
    run_dir = tmp_path / "0.6B-s1"
    _write_stage1_run(run_dir, exp, done=False)              # matching but unfinished
    assert _reuse_or_reset_run(run_dir, exp) is False        # not a completed reuse
    assert (run_dir / "checkpoint.pt").exists()              # left in place for train() to resume
    assert list(tmp_path.glob("*.stale-*")) == []


def test_stage1_dry_chain_writes_all_units(tmp_path):
    root = tmp_path / "s1d_dry"; root.mkdir()
    (root / "APPROVED_stage1").write_text("approved")
    env = {**os.environ, "DRIVE": str(tmp_path), "S1D_DRY": "1", "S1D_SKIP_INSTALL": "1",
           "PYTHON": sys.executable}
    run = subprocess.run(["bash", "scripts/s1d_chain.sh", "stage1"], env=env,
                         capture_output=True, text=True)
    log = (root / "logs" / "stage1.log").read_text()
    assert run.returncode == 0, run.stdout + run.stderr + "\n" + log
    outputs = {name: json.loads((root / "stores" / f"stage1-{name}.json").read_text())
               for name in ("train_sizes", "layout_ablation", "dev_eval")}
    assert all(value["implementation_version"] == 5 for value in outputs.values())
    assert len(outputs["train_sizes"]["runs"]) == 6
    for seed in (1, 2):
        hashes = {outputs["train_sizes"]["runs"][f"{size}-s{seed}"]["hashes"]["questions"]
                  for size in ("0.6B", "1.7B", "4B")}
        assert len(hashes) == 1
    dev = outputs["dev_eval"]
    assert dev["split"] == "nemotron-calib" and dev["not_pii_questions"] > 0
    assert all(len(row) == 56 for result in dev["prompted"].values()
               for row in result["probabilities"])
    assert outputs["layout_ablation"]["stage2_layout"] in {"shared", "kev"}


def test_decision20_dry_chain_scores_dev_questions(tmp_path):
    root = tmp_path / "s1d_dry"; root.mkdir()
    env = {**os.environ, "DRIVE": str(tmp_path), "S1D_DRY": "1", "S1D_SKIP_INSTALL": "1",
           "PYTHON": sys.executable}
    run = subprocess.run(["bash", "scripts/s1d_chain.sh", "comparators1"], env=env,
                         capture_output=True, text=True)
    log = (root / "logs" / "comparators1.log").read_text()
    assert run.returncode == 0, run.stdout + run.stderr + "\n" + log
    out = json.loads((root / "stores" / "comparators1-decision20.json").read_text())
    assert out["split"] == "nemotron-calib" and out["not_pii_questions"] > 0
    assert set(out["models"]) == {"vllm-sr/Decision-2.0-Kai-0.6B", "vllm-sr/Decision-2.0-Eos-0.8B",
                                  "vllm-sr/Decision-2.0-Sol-2B", "vllm-sr/Decision-2.0-Nox-4B",
                                  "vllm-sr/Decision-2.0-Lux-9B"}
    for result in out["models"].values():
        assert result["answered"] == result["questions"] == out["questions"]
        assert all(len(row) == 56 for row in result["probabilities"])


def test_decision20_parser_excludes_errors_and_mismatched_options():
    from types import SimpleNamespace
    from s1pii.s1d.decision20 import criteria, instruction, score_rows
    from s1pii.s1d.schema import Option, Question
    options = (Option("email", "email address"), Option("not personal information", "not PII"))
    def row(i):
        text = f"write to a{i}@x.org now"
        start = text.index("a"); end = text.index(" now")
        return SimpleNamespace(doc_id="d", state=text, source_span=(start, end), target=0,
                               question=Question("choice", "Classify", "", options))
    rows = [row(0), row(1), row(2)]
    assert "[[a0@x.org]]" in instruction(rows[0])
    windowed = instruction(rows[0], "a0@x.org")
    assert '"...[[a0@x.org]]..."' in windowed  # no context quoted beyond the state window
    assert criteria(rows[0]) == {"email": "email address", "not personal information": "not PII"}
    calls = []
    def system_one(*, state, questions):
        calls.append(sorted(questions))
        return {"answers": {"q0": {"type": "choice", "probabilities": {"email": 0.9, "not personal information": 0.1}},
                            "q1": {"type": "choice", "error": "max_length_exceeded"},
                            "q2": {"type": "choice", "probabilities": {"email": 1.0}}}}
    probs, errors = score_rows(SimpleNamespace(system_one=system_one), rows)
    assert probs == [[0.9, 0.1], None, None]
    assert errors == {"max_length_exceeded": 1, "invalid_answer": 1}
    assert len(calls) == 3  # each row has its own state text, so one call per document state
    from s1pii.s1d.decision20 import failed_distribution
    failed = failed_distribution(rows[1])
    assert abs(sum(failed) - 1) < 1e-9 and failed[rows[1].target] == 0.0
    assert max(range(len(failed)), key=failed.__getitem__) != rows[1].target


def test_decision20_resume_keeps_prior_answers_and_errors():
    from types import SimpleNamespace
    from s1pii.s1d.decision20 import score_groups
    from s1pii.s1d.schema import Option, Question
    options = (Option("email", "email address"), Option("not personal information", "not PII"))
    rows = [SimpleNamespace(doc_id="d", state="mail a@x.org now", source_span=(5, 12), target=0,
                            question=Question("choice", "Classify", "", options)) for _ in range(3)]
    seen = []
    def system_one(*, state, questions):
        seen.append(sorted(questions))
        return {"answers": {q: {"type": "choice", "probabilities": {"email": 0.7, "not personal information": 0.3}}
                            for q in questions}}
    groups = [("s0", [0]), ("s1", [1]), ("s2", [2])]
    checkpoints = []
    probs, errors = score_groups(SimpleNamespace(system_one=system_one), rows, groups, start=1,
                                 out=[[0.1, 0.9], None, None], errors={"max_length_exceeded": 2},
                                 on_group=lambda done, p, e: checkpoints.append((done, e)))
    assert seen == [["q1"], ["q2"]]  # group 0 is not re-scored
    assert probs == [[0.1, 0.9], [0.7, 0.3], [0.7, 0.3]]
    assert errors == {"max_length_exceeded": 2}
    assert [done for done, _ in checkpoints] == [2, 3]


def test_decision20_groups_follow_dev_eval_windows():
    from s1pii.s1d import run as R
    ctx = {"dry": True, "root": None, "config": {}}
    _docs, rows = R._dev_questions({**ctx, "root": __import__("pathlib").Path("/tmp/s1d_dry_groups")})
    groups = R._decision20_groups(ctx, rows)
    covered = sorted(i for _state, indices in groups for i in indices)
    assert covered == list(range(len(rows)))
    for state, indices in groups:
        assert 0 < len(indices) <= 64
        assert all(state in rows[i].state for i in indices)
