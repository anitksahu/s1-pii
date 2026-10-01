# S1-PII v2.1 plan (judge-reviewed draft, then run: go/no-go NO-GO and exploratory failure analysis recorded below)

## Evidence from the v2 run (headline stage A; ~11.6 of 32 A100-h used; v2.1 fixes pushed at b6e644f)
- C0' primary holds 3/5 (spy_medical, spy_legal, pii_trace); Nemotron lost (0.44 vs 0.22-0.30 pAUC).
- C3 fails: S1 micro F1 ~0.03 vs GLiNER2.5 zero-shot 0.72. Per held-out label (A, no-nemotron): fax_number matched 0.99 / acc|match 0.88;
  bank_routing_number 0.85 / 0.00 (-> national_id); medical_record_number 0.66 / 0.02; blood_type, language, education_level,
  http_cookie ~0 matched; religion/politics/race 0.2-0.25 matched, 0 acc. GLiNER2.5: matched 0.85, acc|match 0.93.
  Post-floor match 0.35 although level-1 recall after B was 0.85 (all-sources) / 0.51 (no-nemotron) on calibration: the 0.01
  typed-score floor (P_b x P(k*)) removes candidates that lose to NONE. http_cookie: 102 of its gold spans are > 64 tokens.
- Head val acc 0.30 was only the all-sources cheap heads; no-nemotron heads 0.95; all heads after B >= 0.945.
- Label space probe (paraphrase/description -> label name top-1 over 55 Nemotron labels, 165 queries; held-out subset 30):
  v1 encoder 0.21 (0.27), v1+head 0.47 (0.23), A encoder 0.45 (0.47), A+head 0.53 (0.27), bge-small-en-v1.5 0.88 (0.93).
- C4: on SPY every system's 1%-budget calibration threshold is 1.0 or mask-nothing (v1 S1 saturates: 2% non-PII at 1.0);
  selective leak 1.0 there is degenerate for all systems. v2.1 code now marks such cells infeasible.
- Fair ablation, benchmark labels on both sides (pAUC test, seed 1; lower better):
  flat_bench vs A_bench vs v1: tab_direct .179/.332/.252, spy_medical .408/.539/.679, spy_legal .377/.491/.646,
  pii_trace .053/.175/.078, nemotron .315/.326/.309. Flat = frozen v1 encoder, 30% of training docs, 3000 steps, 1 seed.
  Canonical labels: A .262/.562/.514/.071/.457, cheap .293/.662/.639/.121/.424.

## Proposed v2.1 headline (not run: the go/no-go below was NO-GO): flat label-conditioned CRF (no P_b x P(k) factorization)
- Model: existing FlatModel (emission <W_kind h_t, U l_k>, kind-level shared transitions, exact segment marginals; any |L|).
- Label encoder: frozen sentence encoder (bge-base-en-v1.5; NOT MiniLM, which chose the held-out set), label texts as in v2
  (name / paraphrase / description sampling); U maps 768 -> 256.
- Encoder: copy of v1 encoder, top 4 layers fine-tuned (stage A showed +typing), lr 2e-5; head lr 1e-3.
- Training: 100% of each variant's training docs, label-set sampling as flat.py (gold labels in L_b tagged; compatible
  absent labels ANY; other gold O; held-out ANY; hard negatives implicit as O). 6000 steps, batch 8 docs x 512 tokens.
  Variants all-sources / no-nemotron x 3 seeds (6 runs).
- Inference: batched windows across docs; P(sensitive under L) = sum of exact segment marginals over sensitive labels;
  typed output argmax label, typed score = its marginal (no P_b product, so no NONE-driven floor loss).
- Claims re-run exactly as preregistered in v1 (C0' canonical primary / bench secondary; C3 primary no-nemotron vs GLiNER2.5,
  strict micro+macro F1 at 0.5, Holm; C4 with v2.1 infeasibility rule; C4b ECE). GLiNER/baseline predictions reused.
- Ablations: v2 A (existing), v2 flat frozen-encoder 30% (existing), v1, flat with v1-encoder labels (1 seed, if cap allows).
- Gates: none; single preregistered headline. Tag prereg-v2 before GPU.

## Cost (A100-h), cap 12
- training ~45 min x 6 = 4.5; batched prediction (C0' 5 sets x calib/test x 2 label sets, C3 2 label sets) ~1.0;
  ablation 0.75; smoke 0.1. Total ~6.5.

## Engineering
- Batched predict_flat (length-sorted windows across docs, encoder once per window, CRF built once).
- 3-min smoke on GPU first; pip wheelhouse on Drive; CONTROL file; phone alert (ntfy) on STATUS change; eval on CPU runtime.

## Known validity risks (for the judges)
- Held-out labels and Nemotron test were already looked at in v2 (C3 failed); reusing them after diagnosis is a forking-paths risk.
- C3 bar (beat GLiNER2.5 zero-shot) is very likely unreachable; should a weaker, preregistered target be added, and how labeled?
- Flat's advantage in v2 is 1 seed, 30% data, frozen encoder; fine-tuning + sentence-encoder labels are two changes at once.
- Training label vocabulary is ~35-70 native labels; zero-shot generalization may need a broad external type inventory
  (e.g. Pile-NER types), which adds data/licence questions.

## Go/no-go before any v2.1 GPU run (fixed before running; exploratory, calibration data only; judge-vetted)
Script: scripts/v21_gonogo.py. CPU runtime: `prep` (no-nemotron seed-1 training docs, the same 30% subset as the v2 flat
run; a stratified sample of Nemotron *calibration* docs, rarest held-out label first, >= 30 spans per held-out label or all
available, <= 300 docs; per-label n written before any GPU run) and the bge-base label probe (pins the HF revision).
GPU runtime (~0.4 A100-h): M1 (v1-encoder labels) and M2 (bge-base labels) x seeds 1, 2, frozen v1 encoder, 1000 steps.
The arms differ only in the label encoder: both L2-normalized, max 128 tokens, identical text sampling (name / paraphrase /
description, name dropout 0.4 to texts without the name), |L| 4-24. M0 = the v2 flat model (3000 steps) for reference.
Typed decoding over the 55-label C3 set (floor 1e-4). Provenance asserted (v1 sha, held-out nodes).
Primary statistic: macro typed accuracy over held-out labels with n >= 20 in the sample (at least 5 such labels, else
prep aborts; exact boundary and correct label over all gold spans; no conditioning on matches). INVALID (a sanity failure,
not a NO-GO) if M0's seen-label accuracy < 0.30, any arm skipped > 5% of its steps, or the evals used different samples.
Otherwise GO iff, on EACH seed: M2 macro >= M1 macro + 0.15 and >= 0.40; doc-level bootstrap 95% CI lower bound of the
macro difference > 0; M2 > M1 on a majority of those labels (raw counts); M2 held-out match rate >= M1 - 0.05; M2 accuracy
on seen labels (training vocabulary) >= M1 - 0.03 (a guard only: seen spans come from docs sampled for held-out labels).
Name dropout uses only texts containing none of the label name's distinctive words (14 of 88 labels have no such text, e.g.
http_cookie, swift_bic, private_email, and get no dropout); loss traces are saved with each model. A GO supports sentence-encoder labels only with
a frozen encoder at 30% data; the full run keeps a v1-label arm in the fine-tuned setting. Calibration docs read here are
also used for thresholds in the full plan; Nemotron test is untouched.

## Go/no-go outcome and failure analysis (fixed before running the analysis)
Outcome: NO-GO (sanity checks passed). Held-out typed macro acc M1 0.085 / 0.041, M2 0.009 / 0.010; CI of M2 - M1
below zero on both seeds; held-out spans proposed 0.81-0.85 (M0 0.69). C3 stays a preregistered negative.
Failure analysis: scripts/v21_diag.py on the go/no-go eval sample (Nemotron calibration), exploratory, CPU only;
`c3_reopenable: false`. Sections: A label rank at forced gold boundaries (softmax of emission sums equals the CRF
segment conditional, tested); B restricted label sets (held-out only, chance 0.1; gold + 4 seen, chance 0.2); C prior
correction on doc halves (none / one offset on labels outside the training vocabulary / per-label biases = ORACLE,
uses held-out gold); D label-map collapse (held-out->nearest seen minus seen->nearest other seen, raw vs mapped;
paraphrase top-1 held-out and seen); E supervised probe on frozen v1 span features (balanced accuracy, doc-grouped CV,
doc-bootstrap CI); F hybrid: v2 headline spans typed by the most-overlapping GLiNER2.5 span vs GLiNER2.5 alone.
Interpretation (held-out; P = probe balanced acc, R = held-out-only acc, A55 = 55-way top-1):
| Result | Implies | Next step |
|---|---|---|
| P < 0.6 | frozen v1 features cannot separate held-out types | stop zero-shot flat typing; go to F |
| P >= 0.6, R >= 0.5, A55 < 0.15, unseen offset gives held-out 55-way acc >= 0.3 (absolute) | prior toward seen labels | seen/unseen calibration or abstention |
| P >= 0.6, 0.3 <= R < 0.5 | partial signal, no single cause | report; no further GPU on flat typing |
| P >= 0.6, R < 0.3, collapse gap grows after the map | the label map collapses held-out labels | fixed label space / linear map near identity |
| P >= 0.6, R < 0.3, no collapse | training objective failure | rethink the label-conditioned head |
| F hybrid >= GLiNER + 0.10 | boundaries and typing separable | hybrid as a v2.2 candidate (system claim, fresh labels/test) |
| only the oracle C recovers | needs held-out supervision | not zero-shot; report, do not pursue |

## Failure analysis outcome (exploratory; C3 stays a preregistered negative)
Run: scripts/v21_diag.py at GitHub main 2ef29a6, CPU, 1699 gold spans (324 held-out) from the go/no-go calibration sample;
output $D/v21/gonogo/diag.json. Rows are assigned per arm; the table did not say how to aggregate across arms or seeds.

| Measure (held-out unless noted) | M0 | M1 s1 / s2 | M2 s1 / s2 |
|---|---|---|---|
| R, acc among held-out labels only (chance 0.1) | 0.278 | 0.262 / 0.309 | 0.160 / 0.210 |
| A55, 55-way top-1 | 0.012 | 0.090 / 0.040 | 0.009 / 0.009 |
| seen-label top-1 at gold boundaries | 0.655 | 0.515 / 0.536 | 0.537 / 0.606 |
| per-label oracle bias (C), held-out top-1 | 0.071 | 0.120 / 0.099 | 0.031 / 0.031 |
| collapse gap, raw -> mapped (D) | 0.001 -> 0.011 | 0.001 -> 0.001 / 0.003 | -0.137 -> -0.020 / -0.018 |
| paraphrase top-1 held-out / seen, raw -> mapped | 0.27/0.20 -> 0.23/0.19 | 0.27/0.20 -> 0.30/0.32, 0.20/0.36 | 0.93/0.88 -> 0.47/0.67, 0.37/0.68 |

- P (E): probe balanced accuracy 0.935 (doc-bootstrap CI 0.906 to 0.965, chance 0.1); seen 0.835. This is supervised
  linear separability of frozen v1 span features, not zero-shot transfer; fax, bank routing and MRN are likely
  partly surface format (per-class recall 1.0, low median ranks).
- M2 (bge): collapse row. The learned map degrades a well-separated label space, more for held-out labels (paraphrase
  0.93 -> 0.37 to 0.47) than seen ones (0.88 -> 0.67 to 0.68); both nearest-neighbour terms rise, so it is largely global
  compression. Correlational: no identity-map control was run.
- M0, M1: no collapse added by the map (gap stays about 0, which means no differential collapse; the 0.925 absolute
  cosine is anisotropy of mean-pooled MLM embeddings). The v1 label space is weak before the map (raw paraphrase 0.27).
  Row: training objective failure for M0 and M1 s1. M1 s2 (R 0.309) meets the partial row by the letter of the table;
  with no preregistered aggregation rule (s1 0.262, mean about 0.29), M1 is recorded as borderline between the two rows.
- Shared contributor: seen-label top-1 (0.52 to 0.66, plain accuracy) is well below the seen probe (balanced accuracy
  0.835, CI 0.798 to 0.869, 45 classes). This is consistent with the label-conditioned head underusing the encoder
  features in every arm, with map collapse as extra damage in M2. The metrics differ, so the gap is indicative only.
- Prior row excluded because R < 0.5 with only held-out labels in play. The single unseen offset (C) gave 0.0 to
  0.006 (held-out 55-way); it is fit on pooled rows that are mostly seen spans (its value was not logged), so it does
  not test the prior hypothesis.
- Oracle row excluded: per-label biases recover only 0.03 to 0.12.
- R is plain accuracy on imbalanced classes while P is balanced accuracy; the comparison is indicative only.
- F not testable for the flat models (their spans were not saved). It used v2 headline stage A spans at the 0.01
  floor: S1 match 0.293, hybrid 0.272, GLiNER2.5 match 0.787, acc 0.728. Typing given a match is about 0.93 for both.
  The F row is not met for this hybrid (0.272 vs GLiNER2.5 0.728).
  The go/no-go proposal rate (0.81 to 0.85) used floor 1e-4 and is a lenient rate, not boundary recall.
- Decision: no further GPU on zero-shot flat typing.
- Plan limitations: the collapse criterion had no numeric threshold, and the table had no cross-arm or cross-seed
  aggregation rule; rows above were assigned per arm after the fact.
