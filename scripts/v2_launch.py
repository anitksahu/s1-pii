"""Tonight's launch cell: %run scripts/v2_launch.py  (after the setup cell, on an A100 runtime)

prep (frozen inputs, prereg-v1 from the Drive bundle or created once) -> STATUS RUNNING ->
chain in its own session -> monitor (blocks this cell; releases the runtime at the end).
"""
import os
import subprocess
from pathlib import Path

from google.colab import userdata

REPO = Path("/content/s1-pii")
V2 = Path("/content/drive/MyDrive/s1pii/v2"); V2.mkdir(parents=True, exist_ok=True)
env = {**os.environ, "S1PII_GH_TOKEN": (userdata.get("GH_TOKEN") or "").strip()}
r = subprocess.run(["bash", "scripts/v2_prep.sh"], cwd=REPO, env=env, capture_output=True, text=True)
print((r.stdout + r.stderr).replace(env["S1PII_GH_TOKEN"] or "\0", "***")[-3000:])
if r.returncode != 0:
    raise SystemExit("prep failed; nothing launched")
(V2 / "STATUS").write_text("RUNNING (launched)")
env.pop("S1PII_GH_TOKEN")
subprocess.Popen(["bash", "scripts/v2_chain.sh"], cwd=REPO, env=env, start_new_session=True)
print("chain launched; monitor follows", flush=True)
exec(open(REPO / "scripts" / "v2_monitor.py").read())
