from __future__ import annotations
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

ARCH_FEATURES = [
    "velo",
    "ivb_in",
    "hb_as_in",
    "rel_height",
    "rel_side",
    "spin",
    "csw",
    "whiff_rate",
    "gb_rate",
    "zone_pct",
]


def fit_kmeans(df_feat: pd.DataFrame, k: int = 8, random_state: int = 42):
    df = df_feat.dropna(subset=ARCH_FEATURES).copy()
    X = df[ARCH_FEATURES].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    labels = km.fit_predict(Xs)
    df["cluster"] = labels

    nn = NearestNeighbors(n_neighbors=6, metric="euclidean")
    nn.fit(Xs)
    return df, scaler, km, nn


def nearest_comps(
    row: pd.Series, df_fit: pd.DataFrame, scaler, nn, within_pitch_type=True, k=6
):
    xq = scaler.transform(row[ARCH_FEATURES].values.reshape(1, -1))
    dists, idxs = nn.kneighbors(xq, n_neighbors=k)
    comps = df_fit.iloc[idxs[0]].copy()
    if within_pitch_type:
        comps = comps[comps["pitch_type"] == row["pitch_type"]]
    cols = [
        "player_name",
        "pitch_type",
        "p_throws",
        "velo",
        "ivb_in",
        "hb_as_in",
        "whiff_rate",
        "gb_rate",
        "cluster_name",
    ]
    return comps[cols].head(k - 1)
