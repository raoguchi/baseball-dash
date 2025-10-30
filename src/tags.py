from __future__ import annotations
import numpy as np
import pandas as pd


def _mag_label(v, q25, q75, small="Subtle", mid="Moderate", big="Heavy"):
    if v >= q75:
        return big
    if v <= q25:
        return small
    return mid


def _side_label(hb_as):
    return "Arm-Side" if hb_as >= 0 else "Glove-Side"


def _vert_label(ivb):
    return "Ride" if ivb >= 0 else "Drop"


def xy_cluster_tags(df_with_clusters: pd.DataFrame) -> dict[int, str]:
    df = df_with_clusters.copy()

    q_abs_ivb25 = np.nanquantile(np.abs(df["ivb_in"]), 0.25)
    q_abs_ivb75 = np.nanquantile(np.abs(df["ivb_in"]), 0.75)
    q_abs_hb25 = np.nanquantile(np.abs(df["hb_as_in"]), 0.25)
    q_abs_hb75 = np.nanquantile(np.abs(df["hb_as_in"]), 0.75)

    q_wh75 = np.nanquantile(df["whiff_rate"], 0.75)
    q_gb75 = np.nanquantile(df["gb_rate"], 0.75)
    q_zn75 = np.nanquantile(df["zone_pct"], 0.75)
    q_wh50 = np.nanquantile(df["whiff_rate"], 0.50)
    q_gb50 = np.nanquantile(df["gb_rate"], 0.50)
    q_zn50 = np.nanquantile(df["zone_pct"], 0.50)

    tags = {}
    for c, sub in df.groupby("cluster"):
        row = sub.mean(numeric_only=True)
        dom_pt = (
            sub["pitch_type"].mode().iloc[0]
            if not sub["pitch_type"].mode().empty
            else "Pitch"
        )

        side = _side_label(row["hb_as_in"])
        vert = _vert_label(row["ivb_in"])
        mag_side = _mag_label(abs(row["hb_as_in"]), q_abs_hb25, q_abs_hb75)
        mag_vert = _mag_label(abs(row["ivb_in"]), q_abs_ivb25, q_abs_ivb75)

        flavor = []
        if row["whiff_rate"] >= q_wh75:
            flavor.append("Whiff-First")
        if row["gb_rate"] >= q_gb75:
            flavor.append("Grounder-First")
        if row["zone_pct"] >= q_zn75:
            flavor.append("Strike-Throwing")
        if not flavor:
            diffs = {
                "Whiff-First": row["whiff_rate"] - q_wh50,
                "Grounder-First": row["gb_rate"] - q_gb50,
                "Strike-Throwing": row["zone_pct"] - q_zn50,
            }
            flavor.append(max(diffs, key=diffs.get))

        side_noun = "Run" if side == "Arm-Side" else "Sweep"
        vert_noun = "Ride" if vert == "Ride" else "Drop"
        shape = f"{side} • {mag_side} {side_noun}, {mag_vert} {vert_noun}"
        tags[c] = f"{dom_pt}: {shape} • " + " / ".join(flavor)

    return tags
