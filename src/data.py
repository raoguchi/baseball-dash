from __future__ import annotations
from datetime import date
from pathlib import Path
import pandas as pd
from pybaseball import statcast
from utils import CACHE_DIR


def default_window() -> tuple[str, str]:
    today = date.today()
    start = date(today.year if today.month >= 3 else today.year - 1, 3, 1)
    return start.isoformat(), today.isoformat()


def _cache_path(start: str, end: str) -> Path:
    return CACHE_DIR / f"statcast_{start}_{end}.parquet"


def load_statcast(start_date: str, end_date: str, force: bool = False) -> pd.DataFrame:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cp = _cache_path(start_date, end_date)
    if cp.exists() and not force:
        return pd.read_parquet(cp)
    df = statcast(start_dt=start_date, end_dt=end_date)
    if "pitch_type" in df.columns:
        df = df[df["pitch_type"].notna()]
    df.to_parquet(cp, index=False)
    return df
