"""Colab monitor for the v2 chain; run in a notebook cell: %run scripts/v2_monitor.py

Every 60 s: prints STATUS, PHASE, GPU utilization and the last log line. Releases the
runtime on DONE, FAILED or STOPPED_CAP, and when the chain is in a GPU phase with GPU
utilization < 20% for 10 consecutive minutes (the chain is then killed and STATUS set).
"""
import subprocess
import time
from pathlib import Path

D = Path("/content/drive/MyDrive/s1pii")
V2, LOG = D / "v2", D / "results" / "logs" / "v2.log"


def read(p):
    try:
        return p.read_text().strip()
    except OSError:
        return ""


def gpu_util():
    try:
        return int(subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True).stdout.split()[0])
    except Exception:
        return -1


def release(msg):
    print(msg, flush=True)
    from google.colab import runtime
    runtime.unassign()


low = 0
T0 = time.time()
while True:
    st, ph, u = read(V2 / "STATUS"), read(V2 / "PHASE"), gpu_util()
    try:
        fresh = (V2 / "STATUS").stat().st_mtime >= T0 - 5
    except OSError:
        fresh = False
    if not fresh:                    # a STATUS left by an earlier attempt: the chain has not written yet
        st = "STARTING"
    tail = read(LOG).splitlines()[-1:] if LOG.exists() else []
    print(time.strftime("%H:%M:%S"), st.split(" ")[0] or "-", ph or "-", f"gpu {u}%", (tail[0][:120] if tail else ""), flush=True)
    if st.startswith(("DONE", "FAILED", "STOPPED_CAP")):
        release(f"chain finished: {st}; releasing the runtime")
        break
    low = low + 1 if (ph == "GPU" and 0 <= u < 20) else 0
    if low >= 10:
        subprocess.run(["pkill", "-f", "s1pii.v2.run_v2"])
        (V2 / "STATUS").write_text(f"FAILED idle-gpu {time.strftime('%FT%TZ', time.gmtime())}")
        release("GPU idle for 10 minutes in a GPU phase; chain stopped; releasing the runtime")
        break
    time.sleep(60)
