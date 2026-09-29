# S1-PII

A System One span-level PII detector (ModernBERT-large + constrained BIOES CRF with exact span marginals) and a leakage-audited benchmark against the latest GLiNER PII models. Plan: `docs/` and `prereg/v0.md`; deviations: `docs/DEVIATIONS.md`.

## Run on Google Colab

Add a Colab secret `GH_TOKEN` (fine-grained, read-only on `anitksahu/s1-pii`). Open the notebooks from GitHub in order; each starts with the same setup cell (Drive mount, clone or pull, data sync).

| Notebook | GPU | What it does |
| --- | --- | --- |
| `00_env_and_drive_sync` | any | install, run tests |
| `01_build_data` | none | label/offset census, snapshots, calibration/test splits |
| `02_run_baselines` | L4/A100 | GLiNER2-PII, NVIDIA GLiNER-PII, GLiNER2.5 zero-shot in isolated venvs; sharded, resumable |
| `03_train_s1` | A100 (L4 ok) | train `all-sources` and `no-nemotron`, seeds 1 to 3; resumable, mirrored to Drive |
| `04_predict_and_score_s1` | L4/A100 | S1 predictions through the same runner; scoring |
| `05_audit_c0_report` | none | leakage audit, headline scoring, C0 decision, curves |

Before the first training run: commit `envs/*.lock`, the resolved dataset and model revisions, and tag `prereg-v0`.

## Local

```sh
pip install -e ".[dev,model,data]" && pytest -q
```
