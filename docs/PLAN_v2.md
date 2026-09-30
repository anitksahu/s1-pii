# S1-PII v2: Jev-style properties on the same architecture

v1 lacked the three properties that make a model Jev-style: classes supplied at inference, hierarchical
typed decisions, and calibrated decisions with abstention. v2 adds them on the v1 architecture
(ModernBERT-large + constrained BIOES CRF with exact span marginals) and reuses the six trained v1 models.
It claims none of the properties until C3 and C4 hold.

## Model (cheap path, always run)
**Level 1, candidate spans.** The v1 CRF is reused frozen. P_b(span) is the exact probability that the
segment is an entity of any type: the sum over the 9 v1 types of the exact segment marginals (disjoint
events), each computed down to 0.001 (<= floor / 9, so no span with P_b >= 0.01 is lost). Candidates: P_b >= 0.01, at most 512 per window; drops are counted and any drop fails a headline
run. Validator hits enter with P_b = 1, as in v1.

**Level 2, typing against a label set L given at inference.** Label = "name: description" or a bare name,
embedded by the frozen v1 encoder (mean over real tokens). Span features = [h_i; h_j; mean h; v1 type
marginals; logit P_b]. Score = scale * cos(S(span), U(label)); softmax over L plus a learned NONE ("not an
entity of any class in L"). Trained on cached features (minutes).

**Decision.** With S(L) the labels flagged sensitive:
P(sensitive under L) = P_b(span) * sum over k in S(L) of P(k | span, L).
This one quantity is used for C0', C4, abstention and the sweeps. Typed output: k* = argmax over L with
P_b * P(k* | span, L).

**Hierarchy.** A frozen type tree (`s1pii/v2/labels.py`, e.g. ssn -> government_id -> identifier -> pii).
Training never uses a label as a negative for a span whose gold label is its ancestor, descendant or synonym.
Inference: `hier_decide` backs off from k* to the deepest ancestor whose summed probability is >= tau
(type-level abstention); user labels outside the tree hang under pii.

## Training data for the typing head
Label-balanced subset of each variant's v1 training sources (<= 3000 docs per raw label, 5% background,
<= 60k docs, identical across seeds). Items: gold spans with their raw label; NONE from (a) gold label
dropped from L_b with its compatible labels, (b) one boundary-shifted copy per gold span, (c) hard
negatives: candidates with P_b >= 0.05 overlapping no gold span and no teacher span of a family the doc's
source does not annotate (the teacher below, run on CPU in the background at chain start). Label texts per batch: name, name +
description, or paraphrase (`NATIVE` table, frozen). ai4privacy excluded. The synthetic conversations are
regenerated with synth-v0.2 (fixed DOB range) for the head; the v1 encoders are unchanged.

## Frozen inputs (prereg v1, hashed before any v2 GPU work)
- `s1pii/configs/benchmark_labels/<ds>.yaml`: every raw label of the benchmark that the taxonomy maps to PII,
  with description and sensitive = true; TAB keeps the DIRECT tier. The hash is checked on every C0' file.
- `s1pii/configs/heldout_labels.yaml`: 10 Nemotron leaf labels chosen by a frozen rule: rank Nemotron leaf
  labels by the maximum cosine similarity (sentence-transformers/all-MiniLM-L6-v2, revision resolved and
  recorded at selection) of any of their texts to any text of every other node's labels; take the 10 lowest
  with >= 50 spans in the Nemotron calibration split (test plays no part in choosing the classes). Nothing is prelisted; manual review can only veto (reason logged), and a veto takes
  the next label. Held-out nodes and their synonyms are excluded across all sources: their spans leave the
  typing data, candidates overlapping them leave the hard negatives, their texts never enter L_b.

## Conditional stages (code always shipped; run only if a preregistered gate fires)
Gates (`s1pii/v2/gates.py`) use only calibration data (Nemotron dev slice) and held-back training items.
- **B, level-1 quasi-identifier supervision.** Fires if, for either variant, the mean over its seeds of level-1 recall
  (exact-boundary candidate with P_b >= 0.5) on Nemotron calibration quasi-identifier and held-out gold spans
  is < 0.60. A type-agnostic CRF (nt = 1) is warm-started from each v1 encoder and trained 1 epoch: every
  gold span is an entity, held-out spans are ANY, and a teacher span becomes ANY in sources that do not
  annotate its families. Teacher, identical for both variants and never trained on Nemotron: spaCy
  en_core_web_lg 3.8.0, run only on sources missing a family (not Nemotron); name, version and the hash of
  its ANY masks are in each export manifest. Typing heads are then retrained on the new features.
- **A, typing encoder fine-tuning.** Fires (re-evaluated after B if B ran) if the minimum over the 6 heads of
  held-back typing accuracy (name + description) is < 0.90, or, for either variant, held-out typing accuracy
  on Nemotron calibration gold held-out spans (mean over its seeds) is < 0.50. A copy of the level-1 encoder
  trains its top 4 layers jointly with the head (512-token context, <= 6000 steps); level 1 is unchanged.
- The headline system is the last stage that ran, written to `state.json` together with the sha256 of every
  gates file. Every claim (C0', C3, C4) refuses to run unless `$S1PII_V2_STATE` points at that file and the
  gates files match their hashes; there is no override, and the state and gate hashes go into every headline
  ledger row. Earlier stages are reported as secondary.
- **C, flat label-conditioned CRF ablation** (no hierarchy; emissions <W_kind h, U l>, kind-level
  transitions, nt = |L|), seed 1, 30% training subset; runs if the cap leaves room.

## Evaluation
**C0' (redaction).** v1 protocol and win rule unchanged (pAUC on [0, 5%], paired cluster bootstrap over 3
seeds, Holm, 3 of 5 benchmarks with Fresh-Real absent); baselines are the v1 prediction files; NVIDIA
excluded on Nemotron; Nemotron uses no-nemotron. Primary: S1 scores P(sensitive under L) with L = exactly the
canonical `labels.yaml` strings and descriptions every baseline received (same information on both sides).
Secondary: S1 under each benchmark's own label names (more information than the baselines had; reported,
not a claim). v1 C0 is re-reported.

**C3 (class flexibility).** Primary: the no-nemotron variant (never saw Nemotron text or labels), 3 seeds,
on Nemotron test with L = every Nemotron raw label (name + description; names-only secondary). Strict typed
micro and macro F1 on held-out labels at threshold 0.5 after non-overlapping decoding. Comparator: GLiNER2.5
zero-shot with the same strings. Win rule: paired cluster bootstrap, Holm over micro and macro; C3 holds if
both are significant wins. Secondary: threshold-free typing accuracy on gold held-out spans, and per-system
thresholds tuned on Nemotron calibration using seen labels only (GLiNER2.5 is also run on calibration).
All-sources is reported as secondary with the caveat that it was trained on Nemotron train (same generator
and templates, held-out spans supervised under coarse v1 types; only the label names are unseen).
GLiNER2-PII reported only (may have been trained on these labels).

**C4 (calibration and abstention).** Unit: alphanumeric character (IGNORE excluded). Per system and
benchmark on the calibration split: t_hi = v1 dev threshold (over-redaction <= 1%); the deferral band
[t_lo, t_hi) grows down from t_hi while deferred characters stay <= 1 - coverage.
- C4a (selective abstention): selective leak at 95% coverage; paired cluster bootstrap (3 seeds), Holm over
  (baselines x benchmarks); a benchmark is won when S1 is significantly lower than every baseline; holds on
  >= 3 of 5.
- C4b (calibration): restricted to characters where a decision happens (covered by a candidate; counts
  reported per system); isotonic map fitted on calibration characters; test ECE with 15 equal-width bins; on all
  5 benchmarks the upper 95% cluster-bootstrap CI of S1's ECE (mean over seeds) is < 0.05, and S1's ECE is
  not significantly worse than any baseline's (paired bootstrap, Holm).
- "Calibrated abstention" is claimed only if C4a and C4b hold; C4a alone is reported as "selective
  abstention". Brier skill against the calibration base rate is reported. Realized test coverage is reported per cell and flagged when more than 1 point under nominal.
90% coverage, selective over-redaction and type-level abstention are descriptive. Deferred characters count
as redacted in C0' (defer = redact).

**Robustness sweeps (prereg, seed 1).** |L| padded with non-sensitive distractors to 3..60 labels, and three
phrasings (names, name + description, paraphrase): pAUC, mean |drift| of P(sensitive), NONE share.

**Ablations (seed 1).** Names-only label texts; no NONE from shifts or hard negatives; flat CRF (stage C);
cheap vs B vs A where they ran.

## Budget
Hard cap 32 A100-hours for all v2 GPU work, enforced per unit (`gpu_hours.jsonl`; a unit does not start if
used + estimate > cap; the chain then stops and records why). Estimates: cheap path about 5 h (6 training
stores 2.1, 6 heads 0.6, benchmark stores 2.6), GLiNER2.5 C3 about 2.3 h (test, test names-only, calibration), ablations 0.2, B about 11 h (6 level-1 runs plus re-extraction), A about 9 h,
C about 3 h. Order: cheap, GLiNER C3, ablations, gates, B, A, C. If the cap stops a conditional stage, the
headline is the last completed stage and the stop is recorded in state.json and the ledger.

## Colab operation
One tracked process chain (`scripts/v2_chain.sh`) writes STATUS to Drive; the monitor cell releases the
runtime on DONE, FAILED, STOPPED_CAP, or GPU utilization < 20% for 10 minutes; check-ins every 5 minutes.
Sweeps run before release (they read local stores); C0', C4, C3 and the ablation table run afterwards on a
CPU runtime from the prediction files on Drive.

## Implementation (done; 91 CPU tests pass, including an end-to-end dry run of the chain)
`s1pii/v2/`: labels.py, features.py, head.py, infer.py, gates.py, level1.py, finetune.py, flat.py,
gliner_c3.py, evaluate.py, run_v2.py; crf.py, encode.py and s1.py take num_types; c0.decide takes the
system name and meta check.
