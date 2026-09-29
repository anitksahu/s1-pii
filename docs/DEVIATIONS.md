# Deviations from implementation plan v2 (PDF)

| Plan v2 | Implementation | Reason |
| --- | --- | --- |
| Baselines: OPF, PII-Tracer, GLiNER2-PII, Presidio, regex | Latest GLiNER models only: GLiNER2-PII, NVIDIA GLiNER-PII, GLiNER2.5 base (zero-shot) | Author scope decision; GLiNER2.5 (Aug 2026) replaces GLiNER2 base as the latest zero-shot model |
| Offsets over NFC-normalized text | Offsets over text exactly as loaded; no normalization | Avoids offset remapping; enforced by surface checks |
| Unreachable operating points "marked unreachable" | Budgets beyond reach use the system's lowest reachable leak; systems with reachable over-redaction < 5% are flagged | M1 scientist review: conservative and never credits an unproduced operating point |
| Word granularity = whitespace words | Alphanumeric runs | M1 scientist review: whitespace words over-redact whole JSON/CSV records |
| Holm family 18 | 17 (NVIDIA GLiNER-PII excluded on Nemotron) | M1 scientist review |
| Seed handling unspecified in test | Mean over seeds under shared bootstrap weights | M1 scientist review |
| PII-TRACE LOBO model | Not needed | Public PII-TRACE release is a single 500-conversation split, used only as test |
| SPY loaded via HF script | Re-implemented seeded Faker fill from the raw placeholder files | `datasets>=4` refuses loading scripts; original fill is unseeded |
| Ledger as parquet | Append-only JSONL, exported to parquet | Atomic appends on Drive |
| S1 trains on PII-TRACE train, with a no-PII-TRACE LOBO variant | S1 never trains on PII-TRACE | The public release has a single 500-conversation split, used only as calibration/test |
| NVIDIA GLiNER-PII inference default `flat_ner=True` | `flat_ner=False` | M3 scientist review: keep overlapping candidates on both sides of the comparison |
| Taxonomy v0.1 | v0.2 adds labels found unmapped by the Colab census: Nemotron `api_key`, `http_cookie` (SECRET), `credit_debit_card`, `national_id`, `tax_id`, `unique_id` (ACCOUNT_NUMBER), `fax_number` (PHONE), `ipv6` (OTHER_PII), `language` (IGNORE); Gretel `account_pin` (SECRET). NVIDIA GLiNER-PII native query labels = every Nemotron label mapped to a PII type | Fail-closed census before any scoring |
| Surface must equal the offset slice exactly | Nemotron-PII span `text` is lowercased for some labels (e.g. `Black` vs `black`) while offsets are right: a case-only mismatch takes the slice; other mismatches are still rejected | Colab census: 1.5% of Nemotron test spans, all case-only |
| Reject budget 0.1% | Gretel 0.5%: records whose span offsets point past the end of a truncated text are rejected whole (5/2962 test, 68/25948 train) and listed in the snapshot meta | Colab census |
