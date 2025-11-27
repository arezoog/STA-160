import dash
from dash import html, dcc, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from model import load_master

# Register page
dash.register_page(__name__, path="/scenarios", name="Scenario Lab")

# --- Data preparation ---
try:
    df = load_master()

    # Ensure datetime column exists and is parsed
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        # Fallback: try to infer a datetime-like column
        datetime_cols = [
            c
            for c in df.columns
            if "date" in c.lower() or "time" in c.lower()
        ]
        if datetime_cols:
            df["datetime"] = pd.to_datetime(
                df[datetime_cols[0]], errors="coerce"
            )
        else:
            df["datetime"] = pd.NaT

    df["date"] = df["datetime"].dt.date
    df["year"] = df["datetime"].dt.year

    # Choose a price column
    if "standardized_price" in df.columns:
        PRICE_COL = "standardized_price"
    elif "charge" in df.columns:
        PRICE_COL = "charge"
    else:
        PRICE_COL = None

    available_years = sorted(
        df["year"].dropna().unique().tolist()
    )
except Exception as e:
    print(f"Error loading data for Scenario Lab: {e}")
    df = pd.DataFrame()
    PRICE_COL = None
    available_years = []


def _empty_fig(message="No data available"):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
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
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


# --- Layout ---
layout = html.Div(
    # Outer container to keep content centered and not full-bleed
    style={
        "maxWidth": "1200px",
        "margin": "16px auto 32px auto",
    },
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
                                html.H2(
                                    "Scenario Lab – Top Stress Days"
                                ),
                                html.Div(
                                    "Explore which days are most stressed under different price thresholds, "
                                    "and drill down into the intraday pattern for any selected day.",
                                    className="text-muted",
                                    style={"fontSize": "0.85rem"},
                                ),
                            ]
                        ),
                    ],
                ),

                # Controls row
                html.Div(
                    style={
                        "display": "flex",
                        "gap": "16px",
                        "flexWrap": "wrap",
                        "marginBottom": "12px",
                    },
                    children=[
                        html.Div(
                            style={"minWidth": "220px"},
                            children=[
                                html.Label(
                                    "Year",
                                    style={
                                        "fontSize": "0.8rem",
                                        "textTransform": "uppercase",
                                        "letterSpacing": "0.12em",
                                    },
                                ),
                                dcc.Dropdown(
                                    id="scen-year",
                                    options=[
                                        {
                                            "label": str(y),
                                            "value": int(y),
                                        }
                                        for y in available_years
                                    ],
                                    value=(
                                        int(available_years[-1])
                                        if available_years
                                        else None
                                    ),
                                    placeholder="Select year",
                                ),
                            ],
                        ),
                        html.Div(
                            style={"minWidth": "260px"},
                            children=[
                                html.Label(
                                    "Price Threshold (percentile)",
                                    style={
                                        "fontSize": "0.8rem",
                                        "textTransform": "uppercase",
                                        "letterSpacing": "0.12em",
                                    },
                                ),
                                dcc.Slider(
                                    id="scen-quantile",
                                    min=80,
                                    max=99,
                                    step=1,
                                    value=95,
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": False,
                                    },
                                    marks={
                                        p: f"{p}%"
                                        for p in [80, 85, 90, 95, 99]
                                    },
                                ),
                            ],
                        ),
                        html.Div(
                            style={"minWidth": "220px"},
                            children=[
                                html.Label(
                                    "Top N days",
                                    style={
                                        "fontSize": "0.8rem",
                                        "textTransform": "uppercase",
                                        "letterSpacing": "0.12em",
                                    },
                                ),
                                dcc.Slider(
                                    id="scen-topn",
                                    min=5,
                                    max=30,
                                    step=1,
                                    value=10,
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": False,
                                    },
                                    marks={
                                        5: "5",
                                        10: "10",
                                        20: "20",
                                        30: "30",
                                    },
                                ),
                            ],
                        ),
                    ],
                ),

                # Graphs row
                html.Div(
                    style={
                        "display": "grid",
                        "gap": "16px",
                        "gridTemplateColumns": "minmax(0, 1.05fr) minmax(0, 1.25fr)",
                    },
                    children=[
                        html.Div(
                            children=[
                                html.H4(
                                    "Top Stress Days",
                                    style={"marginBottom": "4px"},
                                ),
                                html.Div(
                                    "Bars are ordered by the fraction of hours above the selected price threshold.",
                                    className="text-muted",
                                    style={
                                        "marginBottom": "6px",
                                        "fontSize": "0.8rem",
                                    },
                                ),
                                dcc.Graph(
                                    id="scen-top-days",
                                    className="dash-graph",
                                    style={"height": "360px"},
                                ),
                            ]
                        ),
                        html.Div(
                            children=[
                                html.H4(
                                    "Intraday Profile of Selected Day",
                                    style={"marginBottom": "4px"},
                                ),
                                html.Div(
                                    "Click a bar on the left to inspect that day's intraday price path. "
                                    "Red markers indicate hours above the threshold.",
                                    className="text-muted",
                                    style={
                                        "marginBottom": "6px",
                                        "fontSize": "0.8rem",
                                    },
                                ),
                                dcc.Graph(
                                    id="scen-day-detail",
                                    className="dash-graph",
                                    style={"height": "360px"},
                                ),
                            ]
                        ),
                    ],
                ),
            ],
        )
    ],
)


# --- Callbacks ---
@callback(
    Output("scen-top-days", "figure"),
    Output("scen-day-detail", "figure"),
    Input("scen-year", "value"),
    Input("scen-quantile", "value"),
    Input("scen-topn", "value"),
    Input("scen-top-days", "clickData"),
)
def update_scenario_lab(year, quantile, topn, clickData):
    if df.empty or PRICE_COL is None:
        return _empty_fig("No data available"), _empty_fig(
            "No data available"
        )

    dff = df.copy()
    if year is not None:
        dff = dff[dff["year"] == year]

    dff = dff.dropna(subset=["datetime", PRICE_COL, "date"])
    if dff.empty:
        return _empty_fig("No data for this year"), _empty_fig(
            "No data for this year"
        )

    # Compute threshold from chosen percentile
    q = (quantile or 95) / 100.0
    threshold = dff[PRICE_COL].quantile(q)

    dff["is_stress"] = dff[PRICE_COL] >= threshold

    daily = (
        dff.groupby("date")
        .agg(
            max_price=(PRICE_COL, "max"),
            avg_price=(PRICE_COL, "mean"),
            stress_hours=("is_stress", "sum"),
            total_hours=("is_stress", "size"),
        )
        .reset_index()
    )
    daily["stress_ratio"] = daily["stress_hours"] / daily["total_hours"]

    # Top N by stress_ratio (then max_price)
    daily_sorted = daily.sort_values(
        ["stress_ratio", "max_price"],
        ascending=[False, False],
    )
    top = daily_sorted.head(int(topn) if topn else 10)

    if top.empty:
        return _empty_fig(
            "No days exceed this threshold"
        ), _empty_fig("No days exceed this threshold")

    # Bar chart of top stress days
    fig_top = px.bar(
        top,
        x="date",
        y="stress_ratio",
        hover_data={
            "max_price": ":.2f",
            "stress_hours": True,
            "total_hours": True,
            "stress_ratio": ":.2f",
        },
        labels={
            "stress_ratio": "Share of hours above threshold",
            "date": "Date",
        },
    )
    fig_top.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=40),
    )
    fig_top.update_traces(marker_color="#22d3ee")

    # Determine which date to drill into
    if clickData and "points" in clickData and clickData["points"]:
        selected_date_str = clickData["points"][0]["x"]
    else:
        selected_date_str = str(top.iloc[0]["date"])

    try:
        selected_date = pd.to_datetime(selected_date_str).date()
    except Exception:
        selected_date = top.iloc[0]["date"]

    day_df = dff[dff["date"] == selected_date].sort_values("datetime")

    if day_df.empty:
        return fig_top, _empty_fig(
            "No intraday data for selected day"
        )

    # Intraday line + stress markers
    fig_detail = go.Figure()

    fig_detail.add_trace(
        go.Scatter(
            x=day_df["datetime"],
            y=day_df[PRICE_COL],
            mode="lines",
            name="Price",
            line=dict(color="#38bdf8", width=2),
        )
    )

    stress_df = day_df[day_df["is_stress"]]
    if not stress_df.empty:
        fig_detail.add_trace(
            go.Scatter(
                x=stress_df["datetime"],
                y=stress_df[PRICE_COL],
                mode="markers",
                name="Above threshold",
                marker=dict(color="#f97316", size=8),
            )
        )

    # Horizontal line for threshold
    fig_detail.add_hline(
        y=threshold,
        line=dict(color="#f97316", dash="dash"),
        annotation_text=f"Threshold ≈ {threshold:.2f}",
        annotation_position="top left",
    )

    fig_detail.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=30, b=40),
        xaxis_title="Time",
        yaxis_title="Price",
    )

    return fig_top, fig_detail
