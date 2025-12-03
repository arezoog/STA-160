import dash
from dash import html, dcc
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

dash.register_page(__name__, path="/eda", name="Exploratory Analysis")

# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

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


# -------------------------------------------------------------------------
# Load EDA data from dashboard/data/EQR_output_EDA.csv
# -------------------------------------------------------------------------

try:
    # Assuming this file is dashboard/pages/eda.py
    # so parent (.. ) is the dashboard directory.
    BASE_DIR = Path(__file__).resolve().parents[1]
    DATA_PATH = BASE_DIR / "data" / "EQR_output_EDA.csv"

    df = pd.read_csv(DATA_PATH)

    # Ensure datetime types
    df["trade_date_year_mo"] = pd.to_datetime(df["trade_date_year_mo"])
    df["delivery_year_mo"] = pd.to_datetime(df["delivery_year_mo"])

except Exception as e:
    print(f"[EDA] Error loading EQR_output_EDA.csv: {e}")
    df = pd.DataFrame()


# -------------------------------------------------------------------------
# Build animated figure: trade month on x, delivery month on slider
# -------------------------------------------------------------------------

def build_delivery_animation_figure():
    """
    Animated dual-axis chart:
    - x-axis: trade_date_year_mo
    - left y-axis: weighted_avg_price
    - right y-axis: total_transacted_quantity
    - frames/slider: each frame is one delivery_year_mo
    """
    if df.empty:
        return _empty_fig("No EDA data available for animation.")

    local_df = df.copy()

    # Unique delivery months as 'YYYY-MM' labels
    month_labels_all = sorted(
        local_df["delivery_year_mo"].dt.strftime("%Y-%m").unique()
    )

    def process_month_data(month_str: str):
        """
        Filter to a single delivery month and:
        - sort by trade_date_year_mo
        - expand to full monthly range
        - interpolate weighted_avg_price
        - fill missing quantity with 0
        Returns a DataFrame indexed by trade_date_year_mo, or None if empty.
        """
        mask = local_df["delivery_year_mo"].dt.strftime("%Y-%m") == month_str
        g = local_df.loc[mask].copy()
        if g.empty:
            return None

        # Sort by trade date
        g = g.sort_values("trade_date_year_mo")

        # Full range of trade months (avoid gaps in the animation)
        full_range = pd.date_range(
            start=g["trade_date_year_mo"].min(),
            end=g["trade_date_year_mo"].max(),
            freq="MS",  # Month Start
        )

        g = g.set_index("trade_date_year_mo").reindex(full_range)
        g.index.name = "trade_date_year_mo"

        # Interpolate price and fill quantity
        g["weighted_avg_price"] = g["weighted_avg_price"].interpolate()
        g["total_transacted_quantity"] = g["total_transacted_quantity"].fillna(0)

        return g

    frames = []
    valid_month_labels = []

    for month_str in month_labels_all:
        g = process_month_data(month_str)
        if g is None:
            continue

        valid_month_labels.append(month_str)

        line_trace = go.Scatter(
            x=g.index,
            y=g["weighted_avg_price"],
            mode="lines",
            name="Weighted Avg Price",
            yaxis="y1",
        )

        bar_trace = go.Bar(
            x=g.index,
            y=g["total_transacted_quantity"],
            name="Total Transacted Quantity",
            opacity=0.4,
            yaxis="y2",
        )

        frames.append(
            go.Frame(
                name=month_str,
                data=[line_trace, bar_trace],
                layout=go.Layout(
                    title_text=f"Delivery Month: {month_str}",
                    xaxis=dict(
                        autorange=True,
                        tickformat="%Y-%m",
                    ),
                    yaxis=dict(autorange=True),
                    yaxis2=dict(autorange=True),
                ),
            )
        )

    if not frames:
        return _empty_fig("No frames were created – check EQR_output_EDA.csv.")

    initial_frame = frames[0]

    # Slider to move between delivery months
    slider_steps = []
    for month_str in valid_month_labels:
        slider_steps.append(
            {
                "label": month_str,
                "method": "animate",
                "args": [
                    [month_str],  # go to the frame with this name
                    {
                        "mode": "immediate",
                        "frame": {"duration": 0, "redraw": True},
                        "transition": {"duration": 0},
                    },
                ],
            }
        )

    sliders = [
        {
            "active": 0,
            "pad": {"t": 40},
            "x": 0.1,
            "y": -0.15,
            "len": 0.8,
            "steps": slider_steps,
        }
    ]

    # Optional Play button
    updatemenus = [
        {
            "type": "buttons",
            "showactive": False,
            "x": 0.1,
            "y": 1.12,
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {"duration": 700, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0},
                        },
                    ],
                }
            ],
        }
    ]

    fig = go.Figure(
        data=initial_frame.data,
        layout=go.Layout(
            title=dict(
                text="Price & quantity by trade month (animated by delivery month)",
                x=0.5,
            ),
            xaxis=dict(
                title="Trade Month (YYYY-MM)",
                tickformat="%Y-%m",
                autorange=True,
            ),
            yaxis=dict(
                title="Weighted Avg Price ($/kW-mo)",
                autorange=True,
            ),
            yaxis2=dict(
                title="Total Transacted Quantity (MW)",
                overlaying="y",
                side="right",
                autorange=True,
            ),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.6)",
            font=dict(color="#e5e7eb"),
            margin=dict(l=70, r=70, t=90, b=110),
            sliders=sliders,
            updatemenus=updatemenus,
        ),
        frames=frames,
    )

    return fig


# Build once at import (static dataset)
animated_fig = build_delivery_animation_figure()

# -------------------------------------------------------------------------
# Layout: ONLY the new EDA animated graph
# -------------------------------------------------------------------------

layout = html.Div(
    style={"maxWidth": "1200px", "margin": "16px auto 32px auto", "padding": "0 16px"},
    children=[
        html.Div(
            className="glass-card",
            children=[
                html.H2(
                    "Exploratory analysis: delivery-month animation",
                    style={"marginBottom": "6px"},
                ),
                html.P(
                    "Each frame shows a single delivery month. Use the slider or Play button "
                    "to see how trades build up across trade months.",
                    className="text-muted",
                    style={"fontSize": "0.9rem", "marginBottom": "12px"},
                ),
                dcc.Graph(
                    id="eda-animated-delivery",
                    className="dash-graph",
                    style={"height": "500px"},
                    figure=animated_fig,
                ),
            ],
        )
    ],
)
