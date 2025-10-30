from __future__ import annotations
import numpy as np
import pandas as pd

INCHES_PER_FOOT = 12.0


def infer_ivb_sign(df_raw: pd.DataFrame) -> int:
    """
    Data-driven IVB orientation: pick +1 or -1 so 'ride' is positive.
    Uses only df_raw['pfx_z'] (no hardcoding of pitch types).
    """
    if "pfx_z" not in df_raw.columns or df_raw["pfx_z"].dropna().empty:
        return -1
    med = df_raw["pfx_z"].median()
    return -1 if med < 0 else +1


def signed_arm_side(hb_in_raw: pd.Series, p_throws: pd.Series) -> pd.Series:
    """
    Convert Statcast pfx_x (catcher-right +) into 'arm-side positive' regardless of handedness.
    RHP → +pfx_x is arm-side ; LHP → -pfx_x is arm-side.
    """
    handed = p_throws.fillna("R").str.upper().str[0]
    sign = np.where(handed == "R", 1.0, -1.0)
    return -hb_in_raw * sign


def _safe_rate(num, den):
    return np.divide(
        num, den, out=np.full_like(num, np.nan, dtype=float), where=den > 0
    )


def engineer_pitch_features(df: pd.DataFrame, ivb_sign: int) -> pd.DataFrame:
    cols = [
        "pitch_type",
        "player_name",
        "game_date",
        "events",
        "description",
        "p_throws",
        "stand",
        "release_pos_x",
        "release_pos_z",
        "pfx_x",
        "pfx_z",
        "release_speed",
        "release_spin_rate",
        "plate_x",
        "plate_z",
        "zone",
    ]
    have = [c for c in cols if c in df.columns]
    df = df[have].copy()

    # outcomes
    df["is_called_strike"] = (df["description"] == "called_strike").astype(int)
    df["is_swing"] = (
        df["description"]
        .isin(["swinging_strike", "swinging_strike_blocked", "foul", "hit_into_play"])
        .astype(int)
    )
    df["is_whiff"] = (
        df["description"]
        .isin(["swinging_strike", "swinging_strike_blocked"])
        .astype(int)
    )
    df["is_in_play"] = (df["description"] == "hit_into_play").astype(int)
    df["is_gb"] = (
        df["events"]
        .isin(["groundout", "field_error", "single", "double", "triple"])
        .astype(int)
    )

    # movement (handedness-aware XY)
    df["hb_in_raw"] = df["pfx_x"] * INCHES_PER_FOOT
    df["ivb_in"] = ivb_sign * df["pfx_z"] * INCHES_PER_FOOT  # + = ride, − = drop
    df["hb_as_in"] = signed_arm_side(df["hb_in_raw"], df.get("p_throws"))

    grp = df.groupby(["player_name", "pitch_type", "p_throws"], as_index=False)
    agg = grp.agg(
        n=("pitch_type", "size"),
        velo=("release_speed", "mean"),
        spin=("release_spin_rate", "mean"),
        ivb_in=("ivb_in", "mean"),
        hb_as_in=("hb_as_in", "mean"),
        rel_height=("release_pos_z", "mean"),
        rel_side=("release_pos_x", "mean"),
        cs=("is_called_strike", "sum"),
        swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum"),
        inplay=("is_in_play", "sum"),
        gb=("is_gb", "sum"),
    )

    agg["csw"] = _safe_rate(agg["cs"] + agg["whiffs"], agg["n"])
    agg["whiff_rate"] = _safe_rate(agg["whiffs"], agg["swings"])
    agg["gb_rate"] = _safe_rate(agg["gb"], agg["inplay"])
    agg["zone_pct"] = _safe_rate(agg["cs"] + agg["inplay"], agg["n"])

    keep = [
        "player_name",
        "pitch_type",
        "p_throws",
        "n",
        "velo",
        "spin",
        "ivb_in",
        "hb_as_in",
        "rel_height",
        "rel_side",
        "csw",
        "whiff_rate",
        "gb_rate",
        "zone_pct",
    ]
    return agg[keep].dropna(subset=["velo", "ivb_in", "hb_as_in"])
