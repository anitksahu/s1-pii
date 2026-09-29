#!/usr/bin/env bash
# Train every preregistered S1 run back to back on one Colab runtime (resumable: finished runs
# are skipped, interrupted runs resume from their newest checkpoint). Exports go to Drive.
set -u
cd "$(dirname "$0")/.."
D=/content/drive/MyDrive/s1pii
export PYTHONUNBUFFERED=1 PYTHONPATH="$PWD" S1PII_EXAMPLE_CACHE="$D/examples"
L="$D/results/logs"; mkdir -p "$L"
for v in all-sources no-nemotron; do
  for s in 1 2 3; do
    if [ -f "$D/models/$v-s$s/final/s1_manifest.json" ]; then continue; fi
    echo "START $v s$s $(date -u +%FT%TZ)" >> "$L/train-queue.log"
    if ! python -m s1pii.model.train --variant "$v" --seed "$s" --out "/content/ckpt/$v-s$s" \
         --mirror "$D/ckpt/$v-s$s" >> "$L/train-$v-s$s.log" 2>&1; then
      echo "FAILED $v s$s $(date -u +%FT%TZ)" >> "$L/train-queue.log"; exit 1
    fi
    mkdir -p "$D/models/$v-s$s" && rsync -a "/content/ckpt/$v-s$s/final/" "$D/models/$v-s$s/final/"
    echo "DONE $v s$s $(date -u +%FT%TZ)" >> "$L/train-queue.log"
  done
done
echo "ALL DONE $(date -u +%FT%TZ)" >> "$L/train-queue.log"
