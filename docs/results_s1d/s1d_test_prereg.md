# S1-D held-out test preregistration

Status: frozen before any `s1d_test` GPU inference.

## Outcome-blind amendment — 2026-10-06T15:55:42Z

The initial feasibility pass completed proposer mining before any system was scored. It found 20,042 eligible gold spans and 3,480 available non-overlapping `proposer-no-nemotron` candidates, making the originally specified equal count impossible. Outcomes remained unseen.

With explicit approval, the test now uses **all gold test-label spans and all available non-overlapping proposer negatives**, exactly as `dev_eval` does with `require_equal_negatives=False`. Gold and negative counts are reported by target label. No metric, system, calibration value, bootstrap rule, outcome statement, or other protocol element changed. This corrects the original brief's erroneous balance requirement, which contradicted the development protocol the test was intended to mirror.

## Protocol

- **Data:** the Nemotron-PII test sample defined by `bench.splits("nemotron")`.
- **Gold questions:** gold spans whose canonical labels are the 10 test labels frozen in `s1d_heldout.yaml`.
- **Negative questions:** all available `proposer-no-nemotron` candidates that overlap no gold span. Their target is `not personal information`; their count need not equal the gold count.
- **Options:** the same fixed 56 options used by development evaluation.
- **Systems:** prompted Qwen3-0.6B, Qwen3-1.7B, and Qwen3-4B; and the final `stage1_cov` checkpoints for 1.7B-s1, 1.7B-s2, 4B-s1, and 4B-s2.
- **Run status:** 1.7B-s1, 1.7B-s2, and 4B-s1 are the preregistered healthy S1-D runs. The collapsed 4B-s2 run is reported separately and is excluded from statements quantified over healthy runs.
- **Calibration:** use the already saved Gretel calibration temperatures. No temperature, threshold, parameter, prompt, model, or checkpoint is refit or selected on test data.
- **Inference only:** this stage performs no training or tuning.

## Metrics

For every system, report:

- macro accuracy over the 10 gold labels;
- forced-choice macro accuracy, excluding `not personal information` from the argmax on gold questions;
- not-PII accuracy on negative questions;
- not-PII prediction rate on gold questions;
- NOT_PII AUROC over gold and negative questions;
- per-label accuracy and gold count.

The result also reports gold and negative question counts by target label.

For every system other than prompted Qwen3-4B, report paired 95% document-clustered bootstrap intervals for differences versus prompted Qwen3-4B in macro accuracy, forced-choice macro accuracy, and NOT_PII AUROC. The bootstrap resamples source-document clusters with replacement, uses the same sampled clusters for both systems, uses 2,000 replicates with seed 20261006, and reports percentile 2.5% and 97.5% bounds. The point estimate, rather than whether the interval excludes zero, determines the outcome statements below.

Also write `labels.nearest_trained_neighbours` for the 10 frozen test labels as an audit; it is descriptive and cannot change inference or outcomes.

## Preregistered outcome statements

Each statement will be reported exactly as **HELD** or **FAILED**.

1. **Prompted Qwen3-4B forced-choice macro exceeds every healthy S1-D run.** HELD iff its point estimate is strictly greater than the point estimate of each of 1.7B-s1, 1.7B-s2, and 4B-s1.
2. **S1-D not-PII AUROC exceeds prompted Qwen3-4B for every healthy run.** HELD iff each of 1.7B-s1, 1.7B-s2, and 4B-s1 has a strictly higher point estimate than prompted Qwen3-4B.
3. **Both S1-D 1.7B runs exceed prompted Qwen3-1.7B on macro.** HELD iff both 1.7B-s1 and 1.7B-s2 have strictly higher macro-accuracy point estimates than prompted Qwen3-1.7B.

The complete result is written to `stores/s1d_test-test_eval.json`.
