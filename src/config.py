"""Central configuration. Everything secret comes from environment variables."""
import os
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
try:  # optional .env support
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

# The dataset is NOT shipped in the submission. Point ZENTRO_DATASET at the extracted folder.
def _find_dataset():
    """ZENTRO_DATASET if set (an empty value counts as unset), else look in the usual places."""
    given = os.environ.get("ZENTRO_DATASET", "").strip().strip('"\'')
    if given:
        return Path(given).expanduser()
    for p in (ROOT / "dataset", ROOT / "Capstone_Project_Dataset_FTTT_2026", ROOT.parent / "Capstone_Project_Dataset_FTTT_2026",
              ROOT.parent / "dataset", Path.cwd() / "Capstone_Project_Dataset_FTTT_2026", Path.cwd() / "dataset"):
        if (p / "kb").is_dir() and (p / "data").is_dir():
            return p
    return ROOT / "dataset"


DATASET = _find_dataset()
KB_DIR = DATASET / "kb"
DATA_DIR = DATASET / "data"
PRACTICE_DIR = DATASET / "practice"

TODAY = date(2026, 10, 20)  # dataset reference date (never the system clock)

LLM_MODEL = os.environ.get("LLM_MODEL", "gemini-2.5-pro")
AUX_MODEL = os.environ.get("AUX_MODEL", LLM_MODEL)  # cheap helper calls (query rewrite, grading); separate quota
FALLBACKS = [m.strip() for m in os.environ.get("FALLBACK_MODELS", "").split(",") if m.strip()]  # tried if a model 404s / hits its daily quota
REWRITE = os.environ.get("REWRITE", "always")  # always | auto | never  (LLM query rewrite; 'auto' skips it for plain first-turn English)
TTS_MODEL = os.environ.get("GNANI_TTS_MODEL", "timbre-v2.5")
TTS_VOICE = os.environ.get("GNANI_TTS_VOICE", "Kaveri")
STT_LANGS = os.environ.get("GNANI_STT_LANGS", "en-IN,hi-IN")
TOP_K = int(os.environ.get("TOP_K", "6"))


def ensure_dataset():
    if not (KB_DIR.is_dir() and DATA_DIR.is_dir()):
        sys.exit(f"Dataset not found at {DATASET}. Set ZENTRO_DATASET to the extracted "
                 "Capstone_Project_Dataset_FTTT_2026 folder.")
    tools_dir = str(DATASET)
    if tools_dir not in sys.path:
        sys.path.insert(0, tools_dir)
