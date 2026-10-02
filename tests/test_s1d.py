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


def test_unit_clock_records_then_stops_on_cap(tmp_path):
    from s1pii.s1d.stage0 import UnitClock
    from s1pii.s1d.train import GPUCapReached, GPUHours
    config = {"stages": {"stage0": {"cap_a100_hours": 0}}}
    clock = UnitClock(tmp_path, config, "bounded")
    clock.last -= 1
    with pytest.raises(GPUCapReached):
        clock.tick(substep="chunk")
    assert GPUHours(tmp_path / "gpu_hours.jsonl", 0, "stage0").used() > 0


def test_gpu_cap_override_preserves_accounting_and_does_not_stop(tmp_path, monkeypatch):
    from s1pii.s1d.stage0 import UnitClock
    from s1pii.s1d.train import GPUHours
    monkeypatch.setenv("S1D_CAP_OVERRIDE", "1")
    path = tmp_path / "gpu_hours.jsonl"
    path.write_text(json.dumps({"stage": "stage0", "hours": 6.225}) + "\n")
    meter = GPUHours(path, 5, "stage0")
    meter.reserve(100)
    clock = UnitClock(tmp_path, {"stages": {"stage0": {"cap_a100_hours": 5}}}, "override")
    clock.last -= 1
    clock.tick(substep="continues")
    assert meter.used() > 6.225


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
    assert result["models"]["tiny-random"]["p95_ms_per_window"] > 0


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
    assert "os.environ['S1D_CAP_OVERRIDE'] = '1'" in launch_source
    assert "CAP OVERRIDE: continuing" in launch_source
    assert "STARTING {stage}" in launch_source
    assert launch_source.index("S1D_CAP_OVERRIDE") < launch_source.index("subprocess.Popen")
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


@pytest.mark.parametrize("stage", ["stage1", "stage2"])
def test_future_stage_placeholders_never_write_done(tmp_path, stage):
    from s1pii.s1d.run import run
    (tmp_path / f"APPROVED_{stage}").write_text("approved")
    stores = tmp_path / "stores"; stores.mkdir()
    first = {"stage1": "train_sizes", "stage2": "train_final"}[stage]
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


def test_stage_cap_returns_exit_code_three(tmp_path):
    from s1pii.s1d.run import run
    (tmp_path / "gpu_hours.jsonl").write_text(json.dumps({"hours": 6}) + "\n")
    assert run("stage0", tmp_path, dry=False) == 3
