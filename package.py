#!/usr/bin/env python3
"""Checks the submission and builds <team>_voiceops.zip:   python package.py my_team   [--force]"""
import os
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SKIP_DIRS = {"venv", ".venv", "env", "__pycache__", "dataset", "outputs", "clips", "test_audio", ".git", ".idea", ".vscode"}
SKIP_FILES = {".env", "eval_cache.json", "tickets_created.csv", ".DS_Store"}
SECRET = re.compile(r"AIza[0-9A-Za-z_\-]{30,}|sk-[A-Za-z0-9]{20,}")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
force = "--force" in sys.argv
team = re.sub(r"[^A-Za-z0-9_-]+", "_", args[0]) if args else "team"

files = []
for dp, dn, fn in os.walk(ROOT):
    dn[:] = [d for d in dn if d not in SKIP_DIRS and not d.startswith("Capstone_Project_Dataset")]
    for x in fn:
        if x in SKIP_FILES or x.endswith((".pyc", ".zip")):
            continue
        files.append(Path(dp) / x)

problems, notes = [], []


def need(cond, msg):
    (notes if cond else problems).append(("OK      " if cond else "MISSING ") + msg)


rel = {str(f.relative_to(ROOT)).replace("\\", "/") for f in files}
for req in ["README.md", ".env.example", "requirements.txt", "run.py", "report.pdf", "demo.mp4"]:
    need(req in rel, req)
need(any(r.startswith("src/") and r.endswith(".py") for r in rel), "src/ with code")
need(any(r.startswith("sample_outputs/") and not r.endswith("README.txt") for r in rel),
     "sample_outputs/ with real output from run.py (run it on your own audio clips)")
if (ROOT / "report.pdf").exists():
    b = (ROOT / "report.pdf").read_bytes()
    pages = len(re.findall(rb"/Type\s*/Page[^s]", b))
    need(pages <= 4, f"report.pdf is {pages} page(s) (max 4)")
if (ROOT / "demo.mp4").exists():
    need((ROOT / "demo.mp4").stat().st_size > 100_000, "demo.mp4 is not empty (keep it under 2 minutes)")
readme = (ROOT / "README.md").read_text(encoding="utf-8", errors="ignore") if (ROOT / "README.md").exists() else ""
need("python run.py" in readme, "README shows the run command")
need("github.com/" in readme and "<" not in re.findall(r"Repository:.*", readme)[0] if re.findall(r"Repository:.*", readme) else False,
     "README has your public GitHub URL on a 'Repository:' line")
if (ROOT / ".env.example").exists():
    filled = [l for l in (ROOT / ".env.example").read_text(encoding="utf-8", errors="ignore").splitlines()
              if "=" in l and not l.lstrip().startswith("#") and "KEY" in l.split("=")[0].upper() and l.split("=", 1)[1].strip()]
    need(not filled, ".env.example contains no keys")
for d in ("data", "kb", "practice"):
    need(not (ROOT / d).is_dir(), f"no dataset copy ({d}/ folder not present)")

# secret scan: the contents of your real .env must not appear in any file that goes into the zip
secrets = []
if (ROOT / ".env").exists():
    for line in (ROOT / ".env").read_text(encoding="utf-8", errors="ignore").splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            if "KEY" in k.upper() and len(v.strip()) >= 12:
                secrets.append(v.strip().strip('"\''))
for f in files:
    if f.suffix.lower() in {".py", ".md", ".txt", ".example", ".json", ".pdf", ".csv", ".sh", ".yml"} or f.name == ".env.example":
        t = f.read_bytes().decode("utf-8", errors="ignore")
        if SECRET.search(t) or any(s in t for s in secrets):
            problems.append(f"SECRET   an API key appears in {f.relative_to(ROOT)} - remove it before submitting")

print("\n".join(notes + problems))
if problems and not force:
    sys.exit("\nNot packaged. Fix the items above (or use --force to zip anyway).")
out = ROOT / f"{team}_voiceops.zip"
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(files):
        z.write(f, f.relative_to(ROOT))
print(f"\nCreated {out}  ({len(files)} files)")
