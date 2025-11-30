import dash
from dash import html, dcc, Input, Output, callback
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from model import load_master

dash.register_page(__name__, path="/stress-ridge", name="Stress Ridge 3D")

# --- Data preparation -----------------------------------------------------

try:
    df = load_master()

    # Ensure datetime exists and is parsed
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        datetime_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        if datetime_cols:
            df["datetime"] = pd.to_datetime(df[datetime_cols[0]], errors="coerce")
        else:
            df["datetime"] = pd.NaT

    df = df.dropna(subset=["datetime"])
    df["hour"] = df["datetime"].dt.hour
    df["year"] = df["datetime"].dt.year

    # Price column
    if "standardized_price" in df.columns:
        PRICE_COL = "standardized_price"
    elif "charge" in df.columns:
        PRICE_COL = "charge"
    else:
        PRICE_COL = None

    # Margin / quantity-like column (RA margin proxy)
    MARGIN_COL = None
    for cand in ["ra_margin_mw", "ra_margin", "capacity_margin", "margin", "qty"]:
        if cand in df.columns:
            MARGIN_COL = cand
            break

    AVAILABLE_YEARS = sorted(df["year"].dropna().unique().tolist())
except Exception as e:
    print(f"Error loading data for Stress Ridge 3D: {e}")
    df = pd.DataFrame()
    PRICE_COL = None
    MARGIN_COL = None
    AVAILABLE_YEARS = []


def _empty_3d(message="No data available"):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
        ),
        annotations=[
            dict(
                text=message,
                x=0.5,
                y=0.5,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(color="#9ca3af"),
            )
        ],
        margin=dict(l=20, r=20, t=30, b=20),
    )
    return fig


# --- Layout ---------------------------------------------------------------

layout = html.Div(
    style={"maxWidth": "1200px", "margin": "16px auto 32px auto", "padding": "0 16px"},
    children=[
        html.Div(
            className="glass-card",
            children=[
                # Header
                html.Div(
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "center",
                        "marginBottom": "12px",
                    },
                    children=[
                        html.Div(
                            children=[
                                html.H2("Stress ridge – hour × margin"),
                                html.P(
                                    "A 3D view of how price or stress probability varies across hour of day "
                                    "and a margin/quantity dimension, by year.",
                                    className="text-muted",
                                    style={"fontSize": "0.9rem"},
                                ),
                            ]
                        ),
                    ],
                ),
                # Controls
                html.Div(
                    style={
                        "display": "flex",
                        "flexWrap": "wrap",
                        "gap": "18px",
                        "marginBottom": "12px",
                    },
                    children=[
                        html.Div(
                            style={"minWidth": "160px"},
                            children=[
                                html.Label(
                                    "Year",
                                    style={
                                        "fontSize": "0.8rem",
                                        "textTransform": "uppercase",
                                        "letterSpacing": "0.14em",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="ridge-year",
                                    options=[{"label": str(y), "value": y} for y in AVAILABLE_YEARS],
                                    value=AVAILABLE_YEARS[-1] if AVAILABLE_YEARS else None,
                                    clearable=False,
                                ),
                            ],
                        ),
                        html.Div(
                            style={"minWidth": "200px"},
                            children=[
                                html.Label(
                                    "Metric",
                                    style={
                                        "fontSize": "0.8rem",
                                        "textTransform": "uppercase",
                                        "letterSpacing": "0.14em",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="ridge-metric",
                                    options=[
                                        {"label": "Average price", "value": "price"},
                                        {"label": "Stress probability", "value": "prob"},
                                    ],
                                    value="prob",
                                    clearable=False,
                                ),
                            ],
                        ),
                        html.Div(
                            style={"minWidth": "220px"},
                            children=[
                                html.Label(
                                    "Stress threshold (percentile)",
                                    style={
                                        "fontSize": "0.8rem",
                                        "textTransform": "uppercase",
                                        "letterSpacing": "0.14em",
                                    },
                                ),
                                dcc.Slider(
                                    id="ridge-quantile",
                                    min=80,
                                    max=99,
                                    step=1,
                                    value=95,
                                    marks={80: "80", 90: "90", 95: "95", 99: "99"},
                                    tooltip={"placement": "bottom"},
                                ),
                            ],
                        ),
                    ],
                ),
                # 3D graph
                dcc.Graph(
                    id="stress-ridge-3d",
                    className="dash-graph",
                    style={"height": "480px"},
                ),
            ],
        )
    ],
)


# --- Callback -------------------------------------------------------------


@callback(
    Output("stress-ridge-3d", "figure"),
    Input("ridge-year", "value"),
    Input("ridge-metric", "value"),
    Input("ridge-quantile", "value"),
)
def update_stress_ridge(year, metric, quantile):
    if df.empty or PRICE_COL is None or MARGIN_COL is None:
        msg = "Missing data or margin/price columns. Check model.load_master and column names."
        return _empty_3d(msg)

    dff = df.copy()
    if year is not None:
        dff = dff[dff["year"] == year]

    dff = dff.dropna(subset=[PRICE_COL, MARGIN_COL, "hour"])
    if dff.empty:
        return _empty_3d("No data available for this selection")

    # Threshold for stress
    q = (quantile or 95) / 100.0
    threshold = dff[PRICE_COL].quantile(q)
    dff["is_stress"] = dff[PRICE_COL] >= threshold

    # Bin margin / quantity into ~15 bands
    num_bins = 15
    margin_vals = dff[MARGIN_COL]
    if margin_vals.nunique() > num_bins * 2:
        try:
            dff["margin_bin"] = pd.qcut(margin_vals, q=num_bins, duplicates="drop")
        except Exception:
            dff["margin_bin"] = pd.cut(margin_vals, bins=num_bins)
    else:
        dff["margin_bin"] = margin_vals

    dff = dff.dropna(subset=["margin_bin"])
    dff["margin_bin_label"] = dff["margin_bin"].astype(str)

    # Build pivot: rows = margin bins, cols = hour, values = metric
    if metric == "prob":
        pivot = dff.pivot_table(
            index="margin_bin_label",
            columns="hour",
            values="is_stress",
            aggfunc="mean",
        )
        z_label = "Stress probability"
    else:
        pivot = dff.pivot_table(
            index="margin_bin_label",
            columns="hour",
            values=PRICE_COL,
            aggfunc="mean",
        )
        z_label = "Average price"

    if pivot.empty:
        return _empty_3d("Not enough data to build surface")

    # X axis: hours
    hours = pivot.columns.values
    # Y axis: numeric index with tick labels as bin labels
    margin_labels = pivot.index.tolist()
    y_idx = np.arange(len(margin_labels))

    Z = pivot.values.astype(float)

    fig = go.Figure(
        data=[
            go.Surface(
                x=hours,
                y=y_idx,
                z=Z,
                colorscale="Magma",
                colorbar=dict(title=z_label),
                showscale=True,
                opacity=0.9,
            )
        ]
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=40, b=10),
        scene=dict(
            xaxis=dict(
                title="Hour of day",
                tickmode="linear",
                dtick=3,
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="#374151",
                color="#e5e7eb",
            ),
            yaxis=dict(
                title=MARGIN_COL,
                tickmode="array",
                tickvals=y_idx,
                ticktext=margin_labels,
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="#374151",
                color="#e5e7eb",
            ),
            zaxis=dict(
                title=z_label,
                backgroundcolor="rgba(0,0,0,0)",
                gridcolor="#374151",
                color="#e5e7eb",
            ),
            camera=dict(eye=dict(x=1.5, y=1.6, z=1.2)),
        ),
        title=dict(
            text=f"Stress Ridge – {z_label} vs hour and {MARGIN_COL} (Year {year})",
            x=0.5,
            xanchor="center",
        ),
    )

    return fig
