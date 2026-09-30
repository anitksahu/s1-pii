#!/usr/bin/env bash
# v2 frozen inputs (run once, before any GPU work; CPU is enough, about 2 minutes):
# select the held-out labels from the Nemotron calibration split, hash the frozen inputs,
# commit, tag prereg-v1 and push (token in $S1PII_GH_TOKEN, never echoed). The selection is also
# kept on Drive, so a later runtime re-creates the identical commit if the push was not possible.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD"
D=/content/drive/MyDrive/s1pii; mkdir -p "$D/v2"
if git tag --points-at HEAD | grep -qx prereg-v1; then echo "already tagged prereg-v1"; exit 0; fi
# a later runtime reuses the exact prereg commit saved on Drive by the first prep (same SHA)
if [ -f "$D/v2/prereg-v1.bundle" ]; then
  git fetch -q "$D/v2/prereg-v1.bundle" "refs/tags/prereg-v1:refs/tags/prereg-v1"
  git checkout -q prereg-v1
  cmp -s docs/frozen_inputs.sha256 "$D/v2/frozen_inputs.sha256" || { echo "frozen hashes differ from Drive"; exit 1; }
  echo "checked out prereg-v1 $(git rev-parse --short HEAD) from the Drive bundle"; exit 0
fi
python -m s1pii.v2.labels freeze >/dev/null
git diff --quiet -- s1pii/configs/benchmark_labels || { echo "benchmark label sets differ from the commit"; exit 1; }
if [ ! -f s1pii/configs/heldout_labels.yaml ]; then
  if [ -f "$D/v2/heldout_labels.yaml" ]; then cp "$D/v2/heldout_labels.yaml" s1pii/configs/heldout_labels.yaml   # selected earlier: frozen
    # manual review (veto only; the next label in the stored ranking replaces it)
    grep -q "^vetoes:" s1pii/configs/heldout_labels.yaml || python -m s1pii.v2.labels veto s1pii/configs/heldout_labels.yaml \
      --veto "url=core PII type in every benchmark and in the canonical label set the baselines received; holding it out would remove URL typing from C0'"
    cp s1pii/configs/heldout_labels.yaml "$D/v2/heldout_labels.yaml"
  else python -m s1pii.v2.labels heldout && cp s1pii/configs/heldout_labels.yaml "$D/v2/heldout_labels.yaml"; fi
fi
sha256sum s1pii/configs/heldout_labels.yaml s1pii/configs/benchmark_labels/*.yaml s1pii/v2/labels.py > docs/frozen_inputs.sha256
git add s1pii/configs/heldout_labels.yaml docs/frozen_inputs.sha256
git -c user.name="Anit Kumar Sahu" -c user.email="anit.sahu@gmail.com" commit -q -m "prereg v1: held-out label set selected by the frozen rule on Nemotron calibration counts; frozen input hashes"
git tag prereg-v1
BASE=$(git rev-parse HEAD~1)
git bundle create -q "$D/v2/prereg-v1.bundle" prereg-v1 "^$BASE"
cp docs/frozen_inputs.sha256 "$D/v2/frozen_inputs.sha256"
# push with the token via GIT_ASKPASS so it never appears in argv, URLs or output
ASK=$(mktemp); trap 'rm -f "$ASK"' EXIT
printf '#!/bin/sh\ncase "$1" in Username*) echo x-access-token;; *) printf %%s "$S1PII_GH_TOKEN";; esac\n' > "$ASK"; chmod 700 "$ASK"
GIT_ASKPASS="$ASK" GIT_TERMINAL_PROMPT=0 git push -q origin HEAD:main prereg-v1 >/dev/null 2>&1 \
  && echo "pushed main and prereg-v1" || echo "PUSH FAILED (commit and tag exist locally; token not shown)"
