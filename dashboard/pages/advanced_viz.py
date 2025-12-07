import dash
from dash import html, dcc
import plotly.express as px
import pandas as pd
import numpy as np

from model import load_master

dash.register_page(__name__, path="/animation", name="Market Evolution")

# ----------------------------------------------------------------------
# Data prep – try to be robust to column naming
# ----------------------------------------------------------------------
try:
    master = load_master()
    df = master.copy()

    # 1) Find a datetime column
    datetime_col = None
    for cand in ["datetime", "transaction_begin_date", "begin_date"]:
        if cand in df.columns:
            datetime_col = cand
            break

    if datetime_col is None:
        raise ValueError("No datetime-like column found for animation.")

    df[datetime_col] = pd.to_datetime(df[datetime_col])
    df["datetime"] = df[datetime_col]

    # 2) Create a monthly period for the animation frame
    df["period"] = df["datetime"].dt.to_period("M").dt.to_timestamp()

    # NEW: sort by period and create a label for the slider (YYYY-MM)
    df = df.sort_values("period")
    df["period_label"] = df["period"].dt.strftime("%Y-%m")

    # 3) Choose price column
    price_col = None
    for cand in [
        "standardized_price",
        "price",
        "clearing_price",
        "ra_price",
        "price_$",
    ]:
        if cand in df.columns:
            price_col = cand
            break
    if price_col is None:
        raise ValueError("No price column found for animation.")

    # 4) Choose quantity column
    qty_col = None
    for cand in ["qty", "quantity_mw", "quantity", "mw", "transacted_quantity"]:
        if cand in df.columns:
            qty_col = cand
            break
    if qty_col is None:
        raise ValueError("No quantity column found for animation.")

    # 5) Choose a charge / revenue column for bubble size
    charge_col = None
    for cand in ["charge", "total_charge", "revenue", "total_revenue"]:
        if cand in df.columns:
            charge_col = cand
            break

    if charge_col is not None:
        df["charge_abs"] = df[charge_col].abs()
    else:
        df["charge_abs"] = 1.0

    # 6) Keep only needed cols and drop missing
    df = df[
        ["datetime", "period", "period_label", qty_col, price_col, "charge_abs"]
    ].dropna()
    df = df.rename(columns={qty_col: "qty"})

    # 7) Ranges for axes
    max_qty = 10000
    max_price = 100000

except Exception as e:
    # Fall back to an empty frame if anything goes wrong
    df = pd.DataFrame()
    price_col = "standardized_price"
    max_qty = 1.0
    max_price = 1.0


# ----------------------------------------------------------------------
# Figure builder
# ----------------------------------------------------------------------
def create_animation():
    """Build the animated scatter with a single Play button and zoom enabled."""
    if df.empty:
        return {}

    fig = px.scatter(
        df,
        x="qty",
        y=price_col,
        animation_frame="period_label",  # use ordered YYYY-MM label
        animation_group="qty",
        size="charge_abs",
        color=price_col,
        color_continuous_scale="Magma",
        hover_data=["datetime"],
        range_x=[0, max_qty],
        range_y=[0, max_price],
        template="plotly_dark",
    )

    # Core styling
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=20, b=20),
        font=dict(color="#e5e7eb"),
        transition={"duration": 1000},
    )
    fig.update_xaxes(
        title="Transacted Quantity (MW)",
        gridcolor="#374151",
        zeroline=False,
    )
    fig.update_yaxes(
        title="Price ($)",
        gridcolor="#374151",
        zeroline=False,
    )

    # Remove the default Play/Pause controls added by Plotly Express
    fig.layout.updatemenus = ()

    # Add a single custom Play button + keep the slider
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=1.15,
                x=1.0,
                xanchor="right",
                yanchor="top",
                pad=dict(t=0, r=10),
                buttons=[
                    dict(
                        label="▶ Play Market History",
                        method="animate",
                        args=[
                            None,
                            dict(
                                frame=dict(duration=600, redraw=True),
                                fromcurrent=True,
                                mode="immediate",
                            ),
                        ],
                    )
                ],
            )
        ],
        sliders=[
            dict(
                currentvalue={
                    "prefix": "Month: ",
                    "font": {"size": 20, "color": "#ef4444"},
                },
                pad={"t": 50},
            )
        ],
    )

    return fig


# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------
layout = html.Div(
    className="page-container",
    children=[
        html.Div(
            className="glass-card",
            children=[
                html.H3(
                    "Market Evolution (Time-Lapse)",
                    style={
                        "marginTop": 0,
                        "marginBottom": "6px",
                        "fontSize": "1.1rem",
                    },
                ),
                html.P(
                    "Press 'Play' to watch several years of market data evolve. "
                    "Each bubble is one RA capacity trade.",
                    style={
                        "color": "#9ca3af",
                        "fontSize": "0.9rem",
                        "marginBottom": "8px",
                    },
                ),
                html.H4(
                    "What this page shows",
                    style={
                        "marginTop": "10px",
                        "marginBottom": "4px",
                        "fontSize": "1rem",
                    },
                ),
                html.P(
                    "This page lets you watch the California Resource Adequacy (RA) "
                    "capacity market evolve over time. Each bubble on the chart is a "
                    "single trade taken from the cleaned FERC EQR data."
                ),
                html.Ul(
                    style={"marginLeft": "20px", "lineHeight": "1.7"},
                    children=[
                        html.Li(
                            [
                                html.Strong("Horizontal axis (x-axis): "),
                                "trade size — how much capacity was transacted (MW).",
                            ]
                        ),
                        html.Li(
                            [
                                html.Strong("Vertical axis (y-axis): "),
                                "price of the trade in dollars.",
                            ]
                        ),
                        html.Li(
                            [
                                html.Strong("Bubble size: "),
                                "total dollar value of the trade (larger bubbles mean more money).",
                            ]
                        ),
                        html.Li(
                            [
                                html.Strong("Bubble color: "),
                                "relative price level — darker/brighter colors reflect higher prices.",
                            ]
                        ),
                    ],
                ),
                html.H4(
                    "How to interact with the graph",
                    style={
                        "marginTop": "16px",
                        "marginBottom": "4px",
                        "fontSize": "1rem",
                    },
                ),
                html.Ol(
                    style={"marginLeft": "20px", "lineHeight": "1.7"},
                    children=[
                        html.Li(
                            "Click the 'Play Market History' button above the slider to "
                            "run the animation. The chart will step through the data "
                            "month by month."
                        ),
                        html.Li(
                            "Use the time slider to jump to a specific month. The label "
                            "above the slider shows which month you are viewing."
                        ),
                        html.Li(
                            "Hover your mouse over a bubble to see details for that "
                            "trade, including its date, price, quantity, and total charge."
                        ),
                        html.Li(
                            "Use the zoom tools in the top-right corner of the chart or "
                            "your mouse scroll wheel to zoom in and out, then click and "
                            "drag to pan around. Click the home icon to reset the view."
                        ),
                        html.Li(
                            "Look for patterns as time moves forward—for example, months "
                            "where prices spike, volumes surge, or high prices coincide "
                            "with fewer trades."
                        ),
                    ],
                ),
            ],
        ),
        html.Div(
            className="glass-card",
            children=[
                dcc.Graph(
                    id="animation-graph",
                    figure=create_animation(),
                    style={"height": "75vh"},
                    config={
                        "displayModeBar": True,   # show zoom / pan controls
                        "scrollZoom": True,       # allow mouse-wheel zoom
                        # optional: remove tools you don't need
                        "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                    },
                )
            ],
        ),
    ],
)
