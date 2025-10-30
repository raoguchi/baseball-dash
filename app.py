# app.py
import os, sys
from datetime import datetime

# Ensure we can import from ./src even on HF Spaces
BASE_DIR = os.path.dirname(__file__)
sys.path.append(os.path.join(BASE_DIR, "src"))

import streamlit as st
import pandas as pd

# Your local modules
from data import load_statcast, default_window
from featurize import infer_ivb_sign, engineer_pitch_features
from model import fit_kmeans, nearest_comps
from tags import xy_cluster_tags
from plots import movement_scatter_xy, radar_quality

try:
    from huggingface_hub import hf_hub_download

    HF_HUB_OK = True
except Exception:
    HF_HUB_OK = False

st.set_page_config(page_title="PitchXY (Handedness-Aware)", layout="wide")
st.title("⚾ PitchXY — Handedness-Aware Pitch Archetypes & Scouting Cards")

# ---- Helpers


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def load_statcast_cached(start: str, end: str, force: bool = False) -> pd.DataFrame:
    """
    Cached wrapper around your loader. On Spaces, expensive network calls during
    app init are the #1 cause of infinite 'Starting...'. This keeps it fast.
    """
    return load_statcast(start, end, force=force)


@st.cache_data(show_spinner=False)
def load_sample_fallback() -> pd.DataFrame:
    """
    Optional: fallback sample data so the app is usable even if MLB/Statcast
    endpoints are rate limited / blocked in Spaces.
    - Put a small parquet or CSV in your Space repo: data/sample_statcast.parquet
    - Or host it under a HF Dataset repo and set SAMPLE_DATA_REPO, SAMPLE_DATA_FILE.
    """
    local_path = os.path.join(BASE_DIR, "data", "sample_statcast.parquet")
    if os.path.exists(local_path):
        return pd.read_parquet(local_path)

    # If not bundled locally, try HF Hub (if available)
    repo_id = os.getenv("SAMPLE_DATA_REPO", "").strip()
    file_name = os.getenv("SAMPLE_DATA_FILE", "sample_statcast.parquet").strip()
    if HF_HUB_OK and repo_id:
        path = hf_hub_download(repo_id=repo_id, filename=file_name, repo_type="dataset")
        return pd.read_parquet(path)

    # Give a tiny empty frame with expected columns to keep UI alive
    return pd.DataFrame(
        columns=[
            "game_date",
            "player_name",
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
            "cluster",
            "cluster_name",
            "x_mvt",
            "y_mvt",
        ]
    )


def safe_load_data(start: str, end: str, force: bool) -> pd.DataFrame:
    """
    Try cached real data first; if it errors or returns empty, fall back to a sample.
    """
    try:
        df = load_statcast_cached(start, end, force)
        # Basic sanity check – empty windows are common; handle gracefully
        if df is not None and not df.empty:
            return df
        st.info("No live data returned for that window — showing sample data instead.")
    except Exception as e:
        st.warning(f"Live data failed: {e}\nUsing sample data instead.")
    return load_sample_fallback()


# ---- Sidebar

with st.sidebar:
    st.header("Data Window")
    dstart, dend = default_window()
    start = st.text_input("Start YYYY-MM-DD", dstart)
    end = st.text_input("End YYYY-MM-DD", dend)
    k = st.slider("Clusters (k)", 5, 12, 8)
    force = st.checkbox("Force re-download (discouraged on Spaces)", value=False)
    st.caption("Tip: avoid 'Force re-download' on Spaces to keep startup snappy.")

# ---- Data pipeline

with st.spinner("Loading data…"):
    df_raw = safe_load_data(start, end, force)

if df_raw.empty:
    st.warning(
        "No data available (live and sample were both empty). "
        "Upload a small sample file to ./data/sample_statcast.parquet or set "
        "env vars SAMPLE_DATA_REPO + SAMPLE_DATA_FILE to a HF dataset."
    )
    st.stop()


# Feature engineering (cache stable steps)
@st.cache_data(show_spinner=False)
def _featurize(df_raw_in: pd.DataFrame):
    ivb_sign = infer_ivb_sign(df_raw_in)
    df_feat_local = engineer_pitch_features(df_raw_in, ivb_sign)
    return df_feat_local


df_feat = _featurize(df_raw)


@st.cache_data(show_spinner=False)
def _fit_model(df_feat_in: pd.DataFrame, k_val: int):
    df_fit_local, scaler, km, nn = fit_kmeans(df_feat_in, k=k_val)
    cluster_names_local = xy_cluster_tags(df_fit_local)
    df_fit_local = df_fit_local.copy()
    df_fit_local["cluster_name"] = df_fit_local["cluster"].map(cluster_names_local)
    return df_fit_local, scaler, km, nn


with st.spinner("Clustering & tagging…"):
    df_fit, scaler, km, nn = _fit_model(df_feat, k)

# ---- UI

pitcher = st.selectbox("Pitcher", sorted(df_fit["player_name"].dropna().unique()))
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
        ],
        use_container_width=True,
    )
    for _, row in df_p.iterrows():
        st.markdown(f"### {row['pitch_type']} — {row['cluster_name']}")
        st.plotly_chart(radar_quality(row), use_container_width=True)

with tab3:
    for _, row in df_p.iterrows():
        st.markdown(f"#### {row['pitch_type']} comps")
        comps = nearest_comps(row, df_fit, scaler, nn, within_pitch_type=True, k=6)
        st.dataframe(comps, use_container_width=True)

