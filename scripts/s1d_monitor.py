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


def hours(stage):
    path = ROOT / "gpu_hours.jsonl"
    try:
        rows = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except (json.JSONDecodeError, TypeError):
                # Accounting is append-only and an interrupted Drive write can leave one
                # malformed record. Monitoring must continue; valid records remain counted.
                continue
        return sum(float(row.get("hours", 0)) for row in rows if row.get("stage", stage) == stage)
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
    logs = list((ROOT / "logs").glob("*.log")) if (ROOT / "logs").exists() else []
    latest = max(logs, key=lambda path: path.stat().st_mtime) if logs else None
    tail = read(latest).splitlines()[-1:] if latest else []
    current = read(ROOT / "CURRENT")
    fields = status.split()
    caps = {"stage0": "uncapped", "stage1": "uncapped",
            "stage1_pilot": "uncapped", "stage1_pilot_4b": "uncapped",
            "stage1_cov": "uncapped", "s1d_test": "uncapped", "s1_bench": "uncapped"}
    stage = fields[1] if len(fields) > 1 and fields[1] in caps else "stage0"
    print(time.strftime("%H:%M:%S"), status or "STARTING", phase or "-", f"gpu {util}%",
          f"hours {hours(stage):.3f}/{caps.get(stage, 'unknown')}", current,
          tail[0][:120] if tail else "", flush=True)
    if status.startswith(("DONE", "FAILED", "STOPPED_USER", "STOPPED_RULE")):
        # A just-launched background process may not have replaced the previous
        # terminal status yet. Confirm once before releasing the runtime.
        time.sleep(2)
        if read(ROOT / "STATUS") == status:
            release(f"chain finished: {status}"); break
        continue
    low = low + 1 if phase == "GPU" and 0 <= util < 20 else 0
    if low >= 25:
        subprocess.run(["pkill", "-f", "s1pii.s1d.run"])
        (ROOT / "STATUS").write_text(f"FAILED idle-gpu {time.strftime('%FT%TZ', time.gmtime())}")
        release("GPU idle for 25 minutes; chain stopped"); break
    time.sleep(60)
