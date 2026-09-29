#!/usr/bin/env bash
# Create an isolated venv for one baseline family: gliner | gliner2 | s1
# Reuses the host torch/CUDA (Colab) via --system-site-packages; installs the family's
# package, then this repo without dependencies. If envs/<name>.lock exists it is installed
# exactly; otherwise the resolved versions are frozen into envs/<name>.lock (commit it).
set -euo pipefail
NAME=${1:?usage: make_env.sh gliner|gliner2|s1}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ENV="$ROOT/envs/$NAME"
python3 -m venv --system-site-packages "$ENV"
PIP="$ENV/bin/pip"
"$PIP" install -q --upgrade pip
if [ -f "$ROOT/envs/$NAME.lock" ]; then
  "$PIP" install -q -r "$ROOT/envs/$NAME.lock"
else
  "$PIP" install -q -r "$ROOT/envs/requirements-$NAME.txt"
  "$PIP" freeze --exclude-editable | grep -Ei "^(gliner|gliner2|transformers|tokenizers|huggingface.hub|safetensors|torch|datasketch|pyyaml|numpy|scipy|scikit.learn|sentencepiece|onnxruntime|accelerate)==" > "$ROOT/envs/$NAME.lock" || true
  echo "froze $ROOT/envs/$NAME.lock -- commit it"
fi
"$PIP" install -q --no-deps -e "$ROOT"
"$ENV/bin/python" -c "import s1pii; print('env', '$NAME', 'ok')"
