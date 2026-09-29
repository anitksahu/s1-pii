#!/usr/bin/env bash
# Create an isolated venv for one baseline family: gliner | gliner2 | s1
# Reuses the host torch/CUDA/numpy (Colab) via --system-site-packages and the host pip
# (Colab lacks ensurepip). If envs/<name>.lock exists it is installed exactly; otherwise the
# family's requirements are installed and the resolved family packages are frozen into
# envs/<name>.lock (host-provided torch/numpy/CUDA wheels are never pinned; their versions
# are recorded in every prediction meta instead). Commit the lock.
set -euo pipefail
NAME=${1:?usage: make_env.sh gliner|gliner2|s1}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ENV="$ROOT/envs/$NAME"
python3 -m venv --system-site-packages --without-pip "$ENV"
PY="$ENV/bin/python"
if [ -f "$ROOT/envs/$NAME.lock" ]; then
  "$PY" -m pip install -q -r "$ROOT/envs/$NAME.lock"
else
  "$PY" -m pip install -q -r "$ROOT/envs/requirements-$NAME.txt"
  # packages installed into the venv itself (not inherited from the host)
  SITE=$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
  "$PY" -m pip freeze --path "$SITE" --exclude-editable \
    | grep -viE "^(torch|torchvision|torchaudio|triton|numpy|nvidia-|jax|tensorflow)" > "$ROOT/envs/$NAME.lock"
  if [ ! -s "$ROOT/envs/$NAME.lock" ]; then echo "lock for $NAME is empty; aborting" >&2; exit 1; fi
  echo "froze $ROOT/envs/$NAME.lock -- commit it"
fi
"$PY" -m pip install -q --no-deps -e "$ROOT"
"$PY" -c "import s1pii; print('env', '$NAME', 'ok')"
