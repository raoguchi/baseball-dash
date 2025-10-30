from __future__ import annotations
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def movement_scatter_xy(
    df: pd.DataFrame, color="pitch_type", facet_by_handedness=False
):
    dfp = df.copy()
    if facet_by_handedness:
        fig = px.scatter(
            dfp,
            x="hb_as_in",
            y="ivb_in",
            color=color,
            facet_col="p_throws",
            hover_data=[
                "player_name",
                "pitch_type",
                "p_throws",
                "velo",
                "whiff_rate",
                "gb_rate",
                "csw",
            ],
        )
    else:
        fig = px.scatter(
            dfp,
            x="hb_as_in",
            y="ivb_in",
            color=color,
            hover_data=[
                "player_name",
                "pitch_type",
                "p_throws",
                "velo",
                "whiff_rate",
                "gb_rate",
                "csw",
            ],
        )
    fig.update_layout(
        xaxis_title="Horizontal: Arm-Side (+)  |  Glove-Side (−)",
        yaxis_title="Vertical: Ride (+)  |  Drop (−)",
        legend_title_text=color,
    )
    fig.add_hline(y=0, line_dash="dot")
    fig.add_vline(x=0, line_dash="dot")
    return fig


def radar_quality(row: pd.Series):
    cats = ["csw", "whiff_rate", "gb_rate", "zone_pct"]
    vals = [row[c] for c in cats]
    fig = go.Figure(data=go.Scatterpolar(r=vals, theta=cats, fill="toself"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])), showlegend=False
    )
    return fig
