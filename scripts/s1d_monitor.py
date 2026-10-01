"""Monitor the Drive S1-D chain and release Colab on terminal/idle state."""
import json
import os
import subprocess
import time
from pathlib import Path

D = Path(os.environ.get("DRIVE", "/content/drive/MyDrive/s1pii")); ROOT = D / "s1d"


def read(path):
    try: return path.read_text().strip()
    except OSError: return ""


def gpu_util():
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True).stdout.split()[0])
    except Exception: return -1


def hours():
    path = ROOT / "gpu_hours.jsonl"
    try: return sum(float(json.loads(x).get("hours", 0)) for x in path.read_text().splitlines() if x.strip())
    except OSError: return 0.0


def release(message):
    print(message, flush=True)
    try:
        from google.colab import runtime
        runtime.unassign()
    except ImportError: pass


low = 0; started = time.time()
while True:
    status, phase, util = read(ROOT / "STATUS"), read(ROOT / "PHASE"), gpu_util()
    logs = sorted((ROOT / "logs").glob("*.log")) if (ROOT / "logs").exists() else []
    tail = read(logs[-1]).splitlines()[-1:] if logs else []
    stage = status.split()[1] if status.startswith("RUNNING ") and len(status.split()) > 1 else "stage0"
    caps = {"stage0": 5, "stage1": 12, "stage2": 23}
    print(time.strftime("%H:%M:%S"), status or "STARTING", phase or "-", f"gpu {util}%",
          f"hours {hours():.3f}/{caps.get(stage, 40)}", tail[0][:120] if tail else "", flush=True)
    if status.startswith(("DONE", "FAILED", "STOPPED_CAP", "STOPPED_USER", "STOPPED_RULE")):
        release(f"chain finished: {status}"); break
    low = low + 1 if phase == "GPU" and 0 <= util < 20 else 0
    if low >= 25:
        subprocess.run(["pkill", "-f", "s1pii.s1d.run"])
        (ROOT / "STATUS").write_text(f"FAILED idle-gpu {time.strftime('%FT%TZ', time.gmtime())}")
        release("GPU idle for 25 minutes; chain stopped"); break
    time.sleep(60)
