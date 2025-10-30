from __future__ import annotations
import numpy as np
import pandas as pd


def _mag_label(v, q25, q75, small="Subtle", mid="Moderate", big="Heavy"):
    if pd.isna(v):
        return mid
    if v >= q75:
        return big
    if v <= q25:
        return small
    return mid


def _vert_label(ivb):
    if pd.isna(ivb):
        return "Neutral"
    return "Ride" if ivb >= 0 else "Drop"


def _armside_from_raw_hb(hb_raw: float, throws: str) -> str:
    """Return 'Arm-Side' or 'Glove-Side' from raw HB (catcher view) and dominant throws.
    Statcast convention (catcher view): positive = to catcher’s left (3B side).
    Arm-side mapping commonly used:
      - RHP arm-side run → negative hb_raw
      - LHP arm-side run → positive hb_raw
    """
    if pd.isna(hb_raw) or throws not in ("R", "L"):
        return "Neutral"
    if (throws == "R" and hb_raw < 0) or (throws == "L" and hb_raw > 0):
        return "Arm-Side"
    return "Glove-Side"


def _infer_side_series(sub: pd.DataFrame) -> pd.Series:
    """Infer per-pitch side (Arm/Glove) robustly, using raw hb if available,
    else reconstruct a raw-ish value from hb_as_in and p_throws."""
    has_raw = "hb_in" in sub.columns
    if has_raw:
        hb_raw = sub["hb_in"]
    else:
        # Reconstruct raw-ish: if hb_as_in is arm-side-adjusted (positive toward arm-side),
        # then flip sign for RHP to get a catcher-view-like raw sign.
        # raw ≈ +hb_as for LHP, raw ≈ -hb_as for RHP
        if "hb_as_in" in sub.columns and "p_throws" in sub.columns:
            hb_raw = np.where(sub["p_throws"] == "L", sub["hb_as_in"], -sub["hb_as_in"])
            hb_raw = pd.Series(hb_raw, index=sub.index)
        else:
            return pd.Series(["Neutral"] * len(sub), index=sub.index)

    throws = sub["p_throws"].fillna(
        sub["p_throws"].mode().iloc[0] if not sub["p_throws"].mode().empty else "R"
    )
    return pd.Series(
        np.where(
            ((throws == "R") & (hb_raw < 0)) | ((throws == "L") & (hb_raw > 0)),
            "Arm-Side",
            "Glove-Side",
        ),
        index=sub.index,
    )


def xy_cluster_tags(df_with_clusters: pd.DataFrame) -> dict[int, str]:
    df = df_with_clusters.copy()

    # Quantiles for magnitude bucketing
    q_abs_ivb25 = np.nanquantile(np.abs(df["ivb_in"]), 0.25)
    q_abs_ivb75 = np.nanquantile(np.abs(df["ivb_in"]), 0.75)
    q_abs_hb25 = np.nanquantile(np.abs(df["hb_as_in"]), 0.25)
    q_abs_hb75 = np.nanquantile(np.abs(df["hb_as_in"]), 0.75)

    # Quality quantiles
    q_wh75 = np.nanquantile(df["whiff_rate"], 0.75)
    q_gb75 = np.nanquantile(df["gb_rate"], 0.75)
    q_zn75 = np.nanquantile(df["zone_pct"], 0.75)
    q_wh50 = np.nanquantile(df["whiff_rate"], 0.50)
    q_gb50 = np.nanquantile(df["gb_rate"], 0.50)
    q_zn50 = np.nanquantile(df["zone_pct"], 0.50)

    tags = {}
    for c, sub in df.groupby("cluster"):
        # Robust central tendency
        row = sub.median(numeric_only=True)

        # Dominant metadata
        dom_pt = (
            sub["pitch_type"].mode().iloc[0]
            if "pitch_type" in sub and not sub["pitch_type"].mode().empty
            else "Pitch"
        )
        dom_throw = (
            sub["p_throws"].mode().iloc[0]
            if "p_throws" in sub and not sub["p_throws"].mode().empty
            else "R"
        )

        # Robust side inference
        per_pitch_side = _infer_side_series(sub)
        side_counts = per_pitch_side.value_counts(dropna=False)
        side = side_counts.idxmax() if not side_counts.empty else "Neutral"

        # If nearly tied or Neutral, fall back to median raw
        if side in ("Neutral",) or (
            len(side_counts) > 1 and (side_counts.max() - side_counts.min()) <= 2
        ):
            # Use hb_raw median logic
            if "hb_in" in sub.columns:
                hb_raw_med = sub["hb_in"].median()
            else:
                # Reconstruct raw-ish median from hb_as_in + throws
                if "hb_as_in" in sub.columns:
                    hb_raw_med = sub.apply(
                        lambda r: (
                            r["hb_as_in"]
                            if r.get("p_throws", dom_throw) == "L"
                            else -r["hb_as_in"]
                        ),
                        axis=1,
                    ).median()
                else:
                    hb_raw_med = np.nan
            side = _armside_from_raw_hb(hb_raw_med, dom_throw)

        # Vertical shape from ivb sign (already handedness-invariant)
        vert = _vert_label(row.get("ivb_in", np.nan))

        # Magnitudes from absolute, handedness-invariant features
        mag_side = _mag_label(abs(row.get("hb_as_in", np.nan)), q_abs_hb25, q_abs_hb75)
        mag_vert = _mag_label(abs(row.get("ivb_in", np.nan)), q_abs_ivb25, q_abs_ivb75)

        # Flavor tags
        flavor = []
        if row.get("whiff_rate", 0) >= q_wh75:
            flavor.append("Whiff-First")
        if row.get("gb_rate", 0) >= q_gb75:
            flavor.append("Grounder-First")
        if row.get("zone_pct", 0) >= q_zn75:
            flavor.append("Strike-Throwing")
        if not flavor:
            diffs = {
                "Whiff-First": row.get("whiff_rate", 0) - q_wh50,
                "Grounder-First": row.get("gb_rate", 0) - q_gb50,
                "Strike-Throwing": row.get("zone_pct", 0) - q_zn50,
            }
            flavor.append(max(diffs, key=diffs.get))

        side_noun = (
            "Run"
            if side == "Arm-Side"
            else ("Sweep" if side == "Glove-Side" else "Run/Sweep")
        )
        vert_noun = (
            "Ride" if vert == "Ride" else ("Drop" if vert == "Drop" else "Ride/Drop")
        )
        shape = f"{side} • {mag_side} {side_noun}, {mag_vert} {vert_noun}"

        tags[c] = f"{dom_pt}: {shape} • " + " / ".join(flavor)

    return tags

