#!/usr/bin/env bash
# Resumable S1-D chain. Every unit caches under Drive and the monitor owns release.
set -u
cd "$(dirname "$0")/.."
D=${DRIVE:-/content/drive/MyDrive/s1pii}
if [ "${S1D_DRY:-0}" = 1 ]; then R="$D/s1d_dry"; else R="$D/s1d"; fi
L="$R/logs"; mkdir -p "$L" "$R/models" "$R/stores" "$R/eval"
STAGE=${1:-stage0}; LOG="$L/$STAGE.log"
COMPOSITION_ONLY=${2:-}
PYTHON=${PYTHON:-python}
status() { echo "$1 $(date -u +%FT%TZ)" > "$R/STATUS"; }
fail() { status "FAILED $1"; exit "${2:-1}"; }
echo "[$(date -u +%FT%TZ)] starting $STAGE" >> "$LOG"
status "RUNNING $STAGE"; echo CPU > "$R/PHASE"

if [ "${S1D_SKIP_INSTALL:-0}" != 1 ]; then
  pip uninstall -q -y torchvision torchaudio torchtext torchao >> "$LOG" 2>&1 || fail deps 16
  pip install -q -r envs/requirements-s1d.txt -e ".[data,model]" "faker==40.40.0" >> "$LOG" 2>&1 || fail deps 17
fi
"$PYTHON" -c "import torch, transformers, peft; from peft.import_utils import is_torchao_available; is_torchao_available(); from transformers import BertModel, Qwen3Model; import s1pii.s1d" >> "$LOG" 2>&1 || fail deps 18

if [ "${S1D_DRY:-0}" != 1 ]; then
  "$PYTHON" -c 'import torch; assert torch.cuda.is_available(), "CUDA is unavailable. In Colab select Runtime > Change runtime type > GPU, reconnect, and rerun Cells 1-3."; print("CUDA ready:", torch.cuda.get_device_name(0))' >> "$LOG" 2>&1 || fail gpu 19
fi

if [ -f "$R/CONTROL" ] && [ "$(tr '[:lower:]' '[:upper:]' < "$R/CONTROL")" = STOP ]; then status STOPPED_USER; exit 4; fi
echo CPU > "$R/PHASE"
if [ "$COMPOSITION_ONLY" = "--composition-only" ]; then
  "$PYTHON" -m s1pii.s1d.run "$STAGE" --root "$R" --composition-only >> "$LOG" 2>&1
else
  "$PYTHON" -m s1pii.s1d.run "$STAGE" --root "$R" >> "$LOG" 2>&1
fi
rc=$?
if [ "$rc" -eq 4 ]; then status STOPPED_USER; exit "$rc"; fi
if [ "$rc" -eq 5 ]; then status STOPPED_RULE; exit "$rc"; fi
[ "$rc" -eq 0 ] || fail "$STAGE" "$rc"
status DONE
