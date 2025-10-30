from pathlib import Path

CACHE_DIR = Path("data/cache")
ARTIFACTS_DIR = Path("artifacts")


def ensure_dirs():
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
