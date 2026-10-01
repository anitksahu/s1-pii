# S1-PII v2.1 plan (DRAFT for judge review; nothing run yet)

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

## Proposed v2.1 headline: flat label-conditioned CRF (no P_b x P(k) factorization)
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
| P < 0.6 | frozen v1 features cannot separate held-out types | stop zero-shot flat typing; go to F |
| P >= 0.6, R >= 0.5, A55 < 0.15, unseen offset gives held-out 55-way acc >= 0.3 (absolute) | prior toward seen labels | seen/unseen calibration or abstention |
| P >= 0.6, 0.3 <= R < 0.5 | partial signal, no single cause | report; no further GPU on flat typing |
| P >= 0.6, R < 0.3, collapse gap grows after the map | the label map collapses held-out labels | fixed label space / linear map near identity |
| P >= 0.6, R < 0.3, no collapse | training objective failure | rethink the label-conditioned head |
| F hybrid >= GLiNER + 0.10 | boundaries and typing separable | hybrid as a v2.2 candidate (system claim, fresh labels/test) |
| only the oracle C recovers | needs held-out supervision | not zero-shot; report, do not pursue |
