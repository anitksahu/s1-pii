#!/usr/bin/env bash
# v2 GPU chain on one Colab A100 (resumable; every unit is skipped once done).
# Env from the setup cell: S1PII_DATA, S1PII_RESULTS (Drive). Status for the monitor cell:
# $D/v2/STATUS = RUNNING | DONE | FAILED <code> | STOPPED_CAP. The monitor releases the
# runtime on DONE, FAILED or STOPPED_CAP, and on GPU utilization < 20% for 10 minutes.
set -u
cd "$(dirname "$0")/.."
D=/content/drive/MyDrive/s1pii
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD" S1PII_EXAMPLE_CACHE="$D/examples"
L="$D/results/logs"; mkdir -p "$L" "$D/v2"
CAP=${CAP:-32}
echo RUNNING > "$D/v2/STATUS"; echo CPU > "$D/v2/PHASE"
fail() { echo "FAILED $1 $(date -u +%FT%TZ)" > "$D/v2/STATUS"; exit "$1"; }

# frozen inputs: committed and tagged prereg-v1 by scripts/v2_prep.sh (CPU runtime) before any GPU work
git tag --points-at HEAD | grep -qx prereg-v1 || { echo "HEAD is not tagged prereg-v1" >> "$L/v2.log"; fail 10; }
git ls-files --error-unmatch s1pii/configs/heldout_labels.yaml >/dev/null 2>&1 || fail 11
[ -z "$(git status --porcelain -- s1pii docs scripts)" ] || { git status --porcelain >> "$L/v2.log"; fail 12; }
sha256sum -c docs/frozen_inputs.sha256 >> "$L/v2.log" 2>&1 || fail 13

# gliner2 venv builds in the background (needed only at the GLiNER C3 stage, which waits for it)
rm -f envs/gliner2.ready envs/gliner2.failed
( if [ -x envs/gliner2/bin/python ] || bash scripts/make_env.sh gliner2 >> "$L/v2-env.log" 2>&1; then touch envs/gliner2.ready
  else touch envs/gliner2.failed; fi ) &
# spaCy only if the teacher outputs are missing (normally computed earlier on a CPU runtime)
if [ ! -f "$D/v2/teacher-all-sources.json" ] || [ ! -f "$D/v2/teacher-no-nemotron.json" ]; then
  python -c "import spacy" 2>/dev/null || pip install -q "spacy==3.8.*" >> "$L/v2.log" 2>&1 || fail 15
  python -c "import en_core_web_lg" 2>/dev/null || pip install -q \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.8.0/en_core_web_lg-3.8.0-py3-none-any.whl \
    >> "$L/v2.log" 2>&1 || fail 16
fi

# spaCy teacher on CPU in the background (only sources that do not annotate a family; never Nemotron);
# the GPU starts on feature extraction meanwhile and head training waits for the teacher file
rm -f "$D"/v2/teacher-*.failed
python -m s1pii.v2.run_v2 teacher --models "$D/models" --work /content/v2 --drive "$D/v2" >> "$L/v2-teacher.log" 2>&1 &
echo $! > "$D/v2/teacher.pid"
python -m s1pii.v2.run_v2 all --models "$D/models" --work /content/v2 --drive "$D/v2" --cap "$CAP" \
  --gliner-venv envs/gliner2 >> "$L/v2.log" 2>&1
rc=$?
if [ $rc -eq 3 ]; then echo "STOPPED_CAP $(date -u +%FT%TZ)" > "$D/v2/STATUS"; exit 3; fi
[ $rc -eq 0 ] || fail $rc      # "all" ends with the preregistered sweeps (they need the local stores)
echo "DONE $(date -u +%FT%TZ)" > "$D/v2/STATUS"
