# S1 pipeline vs GLiNER on the five PII benchmarks

Preregistered from git SHA `83e86f566271ddd93e22b8d4e4a86f68e372910e` before implementation or GPU work.

## 1. Pipeline (what an S1 system is here)
- Span proposals: the Stage 0 `proposer-no-nemotron` model (class-agnostic BIOES tagger), floor 0.01, all
  candidates kept including overlaps, run on the audited test documents of PII-TRACE, TAB (direct IDs),
  SPY legal, SPY medical and the Nemotron-PII 10k test sample (the exact document sets behind
  `docs/final_results.json`).
- Decision: each candidate span is one Choice question over the same 56 options as dev_eval (55 fine labels
  plus "not personal information"), packed into 512-token state windows exactly as in dev_eval.
- Span score = 1 - P(not personal information). Span type = the canonical type (taxonomy map already used
  for Nemotron raw labels) of the highest-probability non-NOT_PII option.
- Predictions are written in the same prediction format the GLiNER baselines use and scored by the same
  code that produced `docs/final_results.json` (pAUC on [0, 5%], leak at 1%/2%/5% budgets, dev-tuned
  threshold on each benchmark's calibration split, strict micro/macro P/R/F1 at threshold 0.5, types a
  benchmark does not annotate dropped for every system).

## 2. Systems (fixed now)
- `s1_prompted_qwen3_4b`: prompted Qwen3-4B, zero-shot, numbered-code prompt as in dev_eval.
- `s1d_1.7b_s1`, `s1d_1.7b_s2`, `s1d_4b_s1`: stage1_cov final checkpoints, Gretel temperatures.
  `s1d_4b_s2` (collapsed) is not run; the README says so.
- `proposer_only`: the proposer's own span marginal as the score, no S1 decision (shows what the S1
  decision adds and the proposer recall ceiling).
- GLiNER2-PII, NVIDIA GLiNER-PII, GLiNER2.5: reused from `docs/final_results.json`, not re-run.
- Also report per benchmark the proposer's maximum achievable character recall (all candidates kept).

## 3. Statistics
- Paired cluster bootstrap over documents (10,000 resamples) of pAUC(S1 system) - pAUC(GLiNER model), per
  benchmark, using the existing bootstrap code. Report differences with 95% intervals. These are
  document-level intervals only; there is no seed resampling and the README must say so.
- "Competitive" on a benchmark := the upper 95% bound of pAUC(S1) - pAUC(best GLiNER on that benchmark)
  is at most 0.02. Reported per system per benchmark, descriptive, no Holm family, no win counts.

## 4. Known biases, stated in advance
- S1-D never trained on held-out labels (`first_name`, `street_address`, `date_time`, `last_name`,
  `country`, `state`, `county` and neighbours), so it is expected to under-score names, addresses and dates.
- Every S1 system is bounded by proposer recall.
- On Nemotron-PII the S1-D runs saw Nemotron training documents (all-sources); the proposer did not.
