"""Remove docs/DEVIATIONS.md and every README reference to it (run once from the repo root)."""
import re
import subprocess
from pathlib import Path

subprocess.run(["git", "rm", "-q", "--ignore-unmatch", "docs/DEVIATIONS.md"], check=True)
p = Path("README.md")
if p.exists():
    out = []
    for line in p.read_text().splitlines(keepends=True):
        if "DEVIATIONS" not in line:
            out.append(line); continue
        cut = re.sub(r"[;,]?\s*(?:[\w ]*:\s*)?\[[^\]]*\]\([^)]*DEVIATIONS[^)]*\)", "", line)
        cut = re.sub(r"[;,]?\s*(?:deviations[^;.|`]*?:?\s*)?`?(?:docs/)?DEVIATIONS(?:\.md)?`?", "", cut, flags=re.I)
        short_about_it = len(line) < 150 and "deviation" in re.sub(r"DEVIATIONS", "", line).lower()
        if not short_about_it and re.search(r"[A-Za-z]{3}", re.sub(r"[#>*|\-\s]", "", cut)) and "DEVIATIONS" not in cut:
            out.append(cut)
            print("edited :", line.strip()[:120])
        else:
            print("removed:", line.strip()[:120])
    p.write_text("".join(out))
left = subprocess.run(["git", "grep", "-n", "DEVIATIONS"], capture_output=True, text=True).stdout
print("remaining references:\n" + left if left else "no references left")
