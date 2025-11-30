import dash
from dash import html, dcc, Input, Output, callback
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from model import load_master

dash.register_page(__name__, path="/risk", name="Risk Analysis")

# --- Data preparation -----------------------------------------------------

try:
    df = load_master()

    # Ensure datetime column
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        dt_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        if dt_cols:
            df["datetime"] = pd.to_datetime(df[dt_cols[0]], errors="coerce")
        else:
            df["datetime"] = pd.NaT

    df = df.dropna(subset=["datetime"])
    df["year"] = df["datetime"].dt.year
    df["month_num"] = df["datetime"].dt.month

    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    df["month_cat"] = pd.Categorical(
        df["month_num"].map(lambda m: month_labels[m - 1] if 1 <= m <= 12 else None),
        categories=month_labels,
        ordered=True,
    )

    # Choose price column
    if "standardized_price" in df.columns:
        price_col = "standardized_price"
    elif "weighted_avg_price" in df.columns:
        price_col = "weighted_avg_price"
    elif "charge" in df.columns:
        price_col = "charge"
    else:
        price_col = None
except Exception as e:
    print(f"Error preparing risk data: {e}")
    df = pd.DataFrame()
    price_col = None


def _empty_fig(message="No data available"):
    fig = go.Figure()
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=20),
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
    )
    return fig


# --- Layout ---------------------------------------------------------------

layout = html.Div(
    style={"maxWidth": "1200px", "margin": "16px auto 32px auto", "padding": "0 16px"},
    children=[
        # 1. Calendar-style heatmap card
        html.Div(
            className="glass-card",
            style={"marginBottom": "16px"},
            children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                    children=[
                        html.Div(
                            children=[
                                html.H2("Stress grid – year vs month"),
                                html.P(
                                    "Average price by month and year. Darker tiles indicate higher average prices.",
                                    className="text-muted",
                                    style={"fontSize": "0.9rem"},
                                ),
                            ]
                        ),
                    ],
                ),
                # FIX: Constrain height
                dcc.Graph(id="risk-heatmap", className="dash-graph", style={"height": "450px"}),
            ],
        ),
        # 2. Seasonal distribution card
        html.Div(
            className="glass-card",
            children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between", "alignItems": "center"},
                    children=[
                        html.Div(
                            children=[
                                html.H3("Seasonal risk distribution", style={"marginTop": 0, "marginBottom": "4px"}),
                                html.P(
                                    "Violin plots show the shape of the price distribution by month. Wider sections "
                                    "represent more common price ranges.",
                                    className="text-muted",
                                    style={"fontSize": "0.9rem", "marginBottom": "0px"},
                                ),
                            ]
                        ),
                        dcc.Checklist(
                            id="violin-options",
                            options=[{"label": " Show individual points", "value": "points"}],
                            value=[],
                            style={"color": "#e5e7eb", "fontSize": "0.85rem"},
                        ),
                    ],
                ),
                # FIX: Constrain height
                dcc.Graph(id="risk-violin", className="dash-graph", style={"height": "500px"}),
            ],
        ),
    ],
)


# --- Callback -------------------------------------------------------------


@callback(
    Output("risk-heatmap", "figure"),
    Output("risk-violin", "figure"),
    Input("violin-options", "value"),
)
def update_risk_plots(options):
    if df.empty or price_col is None:
        empty = _empty_fig()
        return empty, empty

    # --- 1. Calendar-like heatmap ----------------------------------------
    pivot_df = df.pivot_table(
        index="year",
        columns="month_cat",
        values=price_col,
        aggfunc="mean",
    ).sort_index(ascending=True)

    fig_heat = px.imshow(
        pivot_df,
        aspect="auto",
        color_continuous_scale="Turbo",
        origin="lower",
    )
    fig_heat.update_traces(
        hovertemplate="<b>%{x} %{y}</b><br>Avg price: $%{z:.2f}<extra></extra>"
    )
    fig_heat.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=20, b=40),
        xaxis_title=None,
        yaxis=dict(title=None, type="category", dtick=1),
        coloraxis_colorbar=dict(title="Avg price"),
        autosize=True,
    )

    # --- 2. Violin plot ---------------------------------------------------
    show_points = "outliers" if "points" in (options or []) else False

    fig_violin = px.violin(
        df,
        x="month_cat",
        y=price_col,
        color="month_cat",
        points=show_points,
        box=True,
    )
    fig_violin.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=40, b=40),
        showlegend=False,
        xaxis_title=None,
        yaxis_title="Price distribution ($)",
        autosize=True,
    )
    fig_violin.update_traces(
        meanline_visible=True,
        line_color="white",
        opacity=0.8,
    )

    return fig_heat, fig_violin