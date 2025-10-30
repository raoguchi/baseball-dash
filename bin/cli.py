from __future__ import annotations
import argparse
from data import load_statcast, default_window
from featurize import infer_ivb_sign, engineer_pitch_features
from model import fit_kmeans, nearest_comps
from tags import xy_cluster_tags
from plots import movement_scatter_xy
from utils import ensure_dirs, ARTIFACTS_DIR
import plotly.io as pio


def main():
    parser = argparse.ArgumentParser(
        description="PitchXY: handedness-aware pitch archetypes"
    )
    parser.add_argument("--start", type=str, help="YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="YYYY-MM-DD")
    parser.add_argument("-k", type=int, default=8)
    parser.add_argument(
        "--pitcher", type=str, help='Filter pitcher by name (e.g. "Cole")'
    )
    parser.add_argument(
        "--save-html", action="store_true", help="Save plots to artifacts/"
    )
    parser.add_argument(
        "--force", action="store_true", help="Force re-download Statcast"
    )
    args = parser.parse_args()

    ensure_dirs()
    start, end = (
        (args.start, args.end) if (args.start and args.end) else default_window()
    )
    print(f"Window: {start} → {end}")

    df_raw = load_statcast(start, end, force=args.force)
    ivb_sign = infer_ivb_sign(df_raw)
    print(f"IVB sign inferred = {ivb_sign} (ride should be positive)")

    df_feat = engineer_pitch_features(df_raw, ivb_sign)
    df_fit, scaler, km, nn = fit_kmeans(df_feat, k=args.k)
    cluster_names = xy_cluster_tags(df_fit)
    df_fit["cluster_name"] = df_fit["cluster"].map(cluster_names)

    # Save artifacts
    feat_p = ARTIFACTS_DIR / "pitch_features.parquet"
    fit_p = ARTIFACTS_DIR / "pitch_features_clusters.parquet"
    df_feat.to_parquet(feat_p, index=False)
    df_fit.to_parquet(fit_p, index=False)
    print(f"Saved: {feat_p}, {fit_p}")

    # Optional pitcher card + comps
    if args.pitcher:
        sub = df_fit[
            df_fit["player_name"].str.contains(args.pitcher, case=False, na=False)
        ]
        if sub.empty:
            print(f"No pitcher matched '{args.pitcher}'")
        else:
            name = sub["player_name"].iloc[0]
            df_p = df_fit[df_fit["player_name"] == name].sort_values("pitch_type")
            print(f"\n=== Scouting Card: {name} ===")
            print(
                df_p[
                    [
                        "pitch_type",
                        "p_throws",
                        "n",
                        "velo",
                        "ivb_in",
                        "hb_as_in",
                        "csw",
                        "whiff_rate",
                        "gb_rate",
                        "zone_pct",
                        "cluster_name",
                    ]
                ].to_string(index=False)
            )
            for _, row in df_p.iterrows():
                comps = nearest_comps(
                    row, df_fit, scaler, nn, within_pitch_type=True, k=6
                )
                print(f"\nNearest comps — {row['pitch_type']} ({row['cluster_name']}):")
                print(comps.to_string(index=False))

    # Movement plot
    fig = movement_scatter_xy(df_fit, color="cluster_name")
    if args.save_html:
        out = ARTIFACTS_DIR / "movement_all.html"
        pio.write_html(fig, file=str(out), auto_open=False, include_plotlyjs="cdn")
        print(f"Saved plot: {out}")
