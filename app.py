import streamlit as st
import pandas as pd
from data import load_statcast, default_window
from featurize import infer_ivb_sign, engineer_pitch_features
from model import fit_kmeans, nearest_comps
from tags import xy_cluster_tags
from plots import movement_scatter_xy, radar_quality
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

st.set_page_config(page_title="PitchXY (Handedness-Aware)", layout="wide")
st.title("⚾ PitchXY — Handedness-Aware Pitch Archetypes & Scouting Cards")

with st.sidebar:
    st.header("Data Window")
    dstart, dend = default_window()
    start = st.text_input("Start YYYY-MM-DD", dstart)
    end = st.text_input("End YYYY-MM-DD", dend)
    k = st.slider("Clusters (k)", 5, 12, 8)
    force = st.checkbox("Force re-download", value=False)

df_raw = load_statcast(start, end, force=force)
if df_raw.empty:
    st.warning("No data for that window.")
    st.stop()

ivb_sign = infer_ivb_sign(df_raw)
df_feat = engineer_pitch_features(df_raw, ivb_sign)
df_fit, scaler, km, nn = fit_kmeans(df_feat, k=k)
cluster_names = xy_cluster_tags(df_fit)
df_fit["cluster_name"] = df_fit["cluster"].map(cluster_names)

pitcher = st.selectbox("Pitcher", sorted(df_fit["player_name"].unique()))
df_p = df_fit[df_fit["player_name"] == pitcher].sort_values("pitch_type")

tab1, tab2, tab3 = st.tabs(["Movement", "Scouting Card", "Comps"])

with tab1:
    view = st.radio("View", ["Selected pitcher", "All pitchers"], horizontal=True)
    if view == "Selected pitcher":
        st.subheader(f"Movement — {pitcher}")
        st.plotly_chart(
            movement_scatter_xy(df_p, color="pitch_type"), use_container_width=True
        )
    else:
        st.subheader("Movement — All pitchers (cluster context)")
        st.plotly_chart(
            movement_scatter_xy(df_fit, color="cluster_name"), use_container_width=True
        )


with tab2:
    st.subheader(f"Scouting Card — {pitcher}")
    st.dataframe(
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
        ]
    )
    for _, row in df_p.iterrows():
        st.markdown(f"### {row['pitch_type']} — {row['cluster_name']}")
        st.plotly_chart(radar_quality(row), use_container_width=True)

with tab3:
    for _, row in df_p.iterrows():
        st.markdown(f"#### {row['pitch_type']} comps")
        comps = nearest_comps(row, df_fit, scaler, nn, within_pitch_type=True, k=6)
        st.dataframe(comps)
