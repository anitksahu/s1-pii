"""CPU tests for S1 with a tiny BERT and a locally trained WordPiece tokenizer (no downloads)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from s1pii.schema import Doc, Span, PERSON, EMAIL, IGNORE
from s1pii.data.synth import generate
from s1pii.model.encode import tokenize_doc, train_examples, ANY, token_windows, window_target
from s1pii.model.crf import tag, tag_kind
from s1pii.model.s1 import S1Model
from s1pii.model.train import TrainConfig, train, latest_checkpoint, export, load_exported
from s1pii.model.predict import S1Predictor
from s1pii.model.postprocess import validator_spans, propagate, luhn_ok, iban_ok


@pytest.fixture(scope="module")
def tok(tmp_path_factory):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
    from transformers import PreTrainedTokenizerFast
    corpus = [d.text for d in generate(400, seed=3)]
    t = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    t.normalizer = normalizers.BertNormalizer(lowercase=False)
    t.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    t.train_from_iterator(corpus, trainers.WordPieceTrainer(vocab_size=800, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]"]))
    from tokenizers.processors import TemplateProcessing
    t.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    return PreTrainedTokenizerFast(tokenizer_object=t, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]", sep_token="[SEP]")


def tiny_model(tok, seed=0):
    from transformers import BertConfig, BertModel
    torch.manual_seed(seed)
    cfg = BertConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=256)
    return S1Model(BertModel(cfg, add_pooling_layer=False), 32, dropout=0.0)


def decode_targets(td):
    spans, cur = [], None
    for i, t in enumerate(td.target):
        if t < 0:
            continue
        k, ty = tag_kind(int(t))
        if k == "S":
            spans.append((i, i, ty))
        elif k == "B":
            cur = (i, ty)
        elif k == "E":
            spans.append((cur[0], i, ty)); cur = None
    return spans


def test_targets_match_gold_and_ignore(tok):
    d = generate(50, seed=5)[7]
    d = Doc(d.doc_id, d.text, d.spans + (Span(d.doc_id, 0, 4, IGNORE),), cluster_id="x")
    td = tokenize_doc(d, tok)
    got = {(int(td.offsets[i][0]), int(td.offsets[j][1])) for i, j, _ in decode_targets(td)}
    want = {(s.start, s.end) for s in d.pii_spans()}
    assert got == want
    assert (td.target[td.offsets[:, 1] <= 4] == ANY).all()


def test_window_cut_spans_become_any(tok):
    d = generate(50, seed=5)[3]
    td = tokenize_doc(d, tok)
    si = int(np.nonzero(td.span_of >= 0)[0][1])       # inside first gold span
    t = window_target(td, si, len(td.ids))
    assert t[0] == ANY


def test_training_loss_decreases_and_resume_is_exact(tok, tmp_path):
    docs = generate(60, seed=11)
    ex = train_examples(docs, tok, max_len=128)
    cfg = TrainConfig(backbone="tiny", seed=0, token_budget=2048, grad_accum=1, lr_encoder=3e-3, lr_head=3e-2,
                      ckpt_every=3, log_every=1, gradient_checkpointing=False, bf16=False)
    m1 = tiny_model(tok)
    train(cfg, tmp_path / "a", model=m1, tokenizer=tok, examples=ex, max_steps=6, device="cpu", resume=False)
    losses = [__import__("json").loads(l)["loss"] for l in (tmp_path / "a" / "train_log.jsonl").read_text().splitlines()]
    assert losses[-1] < losses[0]
    # 3 steps, then resume to 6 from a fresh process state == 6 straight
    m2 = tiny_model(tok)
    train(cfg, tmp_path / "b", model=m2, tokenizer=tok, examples=ex, max_steps=6, device="cpu", resume=False)
    ck = latest_checkpoint(tmp_path / "b")
    assert ck.name == "step-00000006"
    import shutil
    shutil.rmtree(ck)                                   # keep step 3 only
    m3 = tiny_model(tok, seed=99)                       # different init: must be overwritten by resume
    train(cfg, tmp_path / "b", model=m3, tokenizer=tok, examples=ex, max_steps=6, device="cpu", resume=True)
    for (k, a), (_, b) in zip(m2.state_dict().items(), m3.state_dict().items()):
        assert torch.allclose(a, b, atol=1e-6), k


def test_predictor_spans_valid_and_export_roundtrip(tok, tmp_path):
    docs = generate(80, seed=13)
    ex = train_examples(docs, tok, max_len=128)
    cfg = TrainConfig(backbone="tiny", seed=0, token_budget=4096, grad_accum=1, lr_encoder=5e-3, lr_head=5e-2,
                      ckpt_every=1000, log_every=50, gradient_checkpointing=False, bf16=False)
    m = tiny_model(tok)
    final = train(cfg, tmp_path / "run", model=m, tokenizer=tok, examples=ex, max_steps=60, device="cpu", resume=False)
    test_docs = generate(10, seed=99)
    pr = S1Predictor(m, tok, revision="t", max_len=64, device="cpu", validators=False, propagation=False)
    preds, rep = pr.predict_docs(test_docs)
    assert rep.windows >= len(test_docs) and rep.kept > 0
    for d in test_docs:
        for s in preds[d.doc_id]:
            assert d.text[s.start:s.end] == s.surface and 0.01 <= s.score <= 1.0
    # the trained tiny model should find some gold person names
    hits = sum(any(p.start == g.start and p.end == g.end for p in preds[d.doc_id] if p.score >= 0.5)
               for d in test_docs for g in d.pii_spans() if g.label_canonical == PERSON)
    assert hits > 0
    m2, tok2, man = load_exported(final)
    preds2, _ = S1Predictor(m2, tok2, revision="t", max_len=64, device="cpu", validators=False,
                            propagation=False).predict_docs(test_docs)
    assert {k: [(s.start, s.end, round(s.score, 6)) for s in v] for k, v in preds.items()} == \
           {k: [(s.start, s.end, round(s.score, 6)) for s in v] for k, v in preds2.items()}
    assert man["weights_sha256"]


def test_validators_and_propagation():
    assert luhn_ok("4111111111111111") and not luhn_ok("4111111111111112")
    assert iban_ok("GB82 WEST 1234 5698 7654 32")
    text = "Card 4111 1111 1111 1111, mail a@b.io, SSN 123-45-6789. Ms. Ada Byron called; Ada Byron again."
    d = Doc("v", text)
    labs = sorted((d.text[s.start:s.end], s.label_canonical) for s in validator_spans(d))
    assert ("4111 1111 1111 1111", "ACCOUNT_NUMBER") in labs and ("a@b.io", "EMAIL") in labs
    assert ("123-45-6789", "ACCOUNT_NUMBER") in labs
    a = text.index("Ada Byron")
    base = [Span("v", a, a + 9, PERSON, score=0.9, source="s1")]
    out = propagate(d, base)
    assert len(out) == 2 and out[1].label_raw == "propagated" and text[out[1].start:out[1].end] == "Ada Byron"


def test_s1_through_shared_runner(tok, tmp_path):
    from s1pii.adapters.run import run
    from s1pii.schema import write_jsonl
    from s1pii.ledger import read_predictions
    docs = generate(12, seed=21)
    dp = tmp_path / "d.jsonl"; write_jsonl(docs, dp)
    pr = S1Predictor(tiny_model(tok), tok, revision="w1", max_len=64, device="cpu")
    out = run("s1_test", dp, tmp_path / "p", predictor=pr, shard_size=5)
    meta, preds = read_predictions(out)
    assert meta["revision"] == "w1" and len(preds) == 12 and meta["config"]["validators"] is True


class NewlineTok:
    """Stub fast-tokenizer that emits standalone whitespace tokens (like ModernBERT's BPE)."""
    pad_token_id = 0
    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False, truncation=False):
        import re
        toks = [(m.start(), m.end()) for m in re.finditer(r"\n|[^\s]+| +", text)]
        ids = [5 + i for i in range(len(toks))]
        if add_special_tokens:
            return {"input_ids": [1] + ids + [2]}
        return {"input_ids": ids, "offset_mapping": toks}


def test_whitespace_tokens_inside_span_are_I_and_targets_valid():
    from s1pii.model.encode import target_is_valid, tokenize_doc
    from s1pii.schema import ADDRESS
    text = "Ship to 12 Main St\nSpringfield now"
    a, b = 8, len("Ship to 12 Main St\nSpringfield")
    d = Doc("n", text, (Span("n", a, b, ADDRESS),))
    td = tokenize_doc(d, NewlineTok())
    assert target_is_valid(td.target)
    ex = train_examples([d], NewlineTok(), max_len=64)
    assert len(ex) == 1
    bad = td.target.copy(); bad[bad == tag("I", 1)] = 0
    assert not target_is_valid(bad)


def test_build_examples_parallel_equals_serial_and_caches(tok, tmp_path, monkeypatch):
    import numpy as np
    from s1pii.model import train as T
    from s1pii.model.encode import train_examples
    from s1pii.data.synth import generate
    docs = generate(1200, seed=3)
    monkeypatch.setattr(T, "training_docs", lambda cfg: (docs, {"synthetic_conv": len(docs)}))
    monkeypatch.setenv("S1PII_EXAMPLE_CACHE", str(tmp_path))
    cfg = T.TrainConfig(backbone="local-test", max_len=128)
    ex, man = T.build_examples(cfg, tok, workers=3)
    ref = train_examples(docs, tok, 128)
    assert len(ex) == len(ref) and all(np.array_equal(a.input_ids, b.input_ids) and a.doc_id == b.doc_id for a, b in zip(ex, ref))
    assert T.examples_hash(ex) == T.examples_hash(ref) and man["n_examples"] == len(ref)
    ex2, _ = T.build_examples(cfg, tok, workers=3)
    assert T.examples_hash(ex2) == T.examples_hash(ex) and len(list(tmp_path.glob("*.pkl"))) == 1
