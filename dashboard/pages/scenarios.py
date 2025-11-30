import dash
from dash import html, dcc, Input, Output, callback, no_update
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from model import load_master

dash.register_page(__name__, path="/scenarios", name="Scenario Lab")

# --- Data preparation -----------------------------------------------------

try:
    df = load_master()

    # Ensure datetime column exists and is parsed
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    else:
        datetime_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        if datetime_cols:
            df["datetime"] = pd.to_datetime(df[datetime_cols[0]], errors="coerce")
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

    available_years = sorted(df["year"].dropna().unique().tolist())
except Exception as e:
    print(f"Error loading data for Scenarios: {e}")
    df = pd.DataFrame()
    available_years = []
    PRICE_COL = None


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
        
        # --- Controls Section ---
        html.Div(
            className="glass-card",
            style={"marginBottom": "24px"},
            children=[
                html.Div(
                    style={"display": "flex", "flexWrap": "wrap", "gap": "24px", "alignItems": "flex-end"},
                    children=[
                        # Title
                        html.Div(
                            style={"flex": "1 1 300px"},
                            children=[
                                html.H2("Scenario Lab", style={"marginBottom": "8px"}),
                                html.P(
                                    "Simulate stress events. Select a year and define a 'Stress Threshold' price "
                                    "to identify high-risk days. Click the top chart to inspect hourly details.",
                                    className="text-muted",
                                    style={"fontSize": "0.9rem", "maxWidth": "500px"}
                                ),
                            ]
                        ),
                        # Controls
                        html.Div(
                            children=[
                                html.Label("Select Year", style={"color": "var(--accent-1)", "fontSize": "0.85rem"}),
                                dcc.Dropdown(
                                    id="scenario-year",
                                    options=[{"label": str(y), "value": y} for y in available_years],
                                    value=available_years[-1] if available_years else None,
                                    clearable=False,
                                    style={"width": "140px"}
                                )
                            ]
                        ),
                        html.Div(
                            style={"flex": "1 1 200px"},
                            children=[
                                html.Label("Stress Threshold ($)", style={"color": "var(--accent-2)", "fontSize": "0.85rem"}),
                                dcc.Slider(
                                    id="scenario-threshold",
                                    min=0,
                                    max=2000,
                                    step=50,
                                    value=500,
                                    marks={0: "$0", 500: "$500", 1000: "$1k", 1500: "$1.5k", 2000: "$2k"},
                                    tooltip={"placement": "bottom", "always_visible": True},
                                )
                            ]
                        ),
                    ]
                )
            ]
        ),

        # --- Top Chart: Daily Overview ---
        html.Div(
            className="glass-card",
            style={"marginBottom": "24px"},
            children=[
                html.H3("Daily Price Overview", style={"marginBottom": "10px"}),
                # FIX: Constrain height
                dcc.Graph(
                    id="scenarios-daily-overview", 
                    className="dash-graph", 
                    style={"height": "450px"}
                ),
                html.P(
                    "Click on any point above to see intraday details below.", 
                    style={"textAlign": "center", "color": "var(--text-muted)", "fontSize": "0.8rem", "marginTop": "8px"}
                )
            ]
        ),

        # --- Bottom Chart: Intraday Detail ---
        html.Div(
            className="glass-card",
            children=[
                html.H3("Intraday Stress Detail", style={"marginBottom": "10px"}),
                # FIX: Constrain height
                dcc.Graph(
                    id="scenarios-detail-view", 
                    className="dash-graph", 
                    style={"height": "450px"}
                ),
            ]
        ),
    ]
)


# --- Callbacks ------------------------------------------------------------

@callback(
    Output("scenarios-daily-overview", "figure"),
    Output("scenarios-detail-view", "figure"),
    Input("scenario-year", "value"),
    Input("scenario-threshold", "value"),
    Input("scenarios-daily-overview", "clickData"),
)
def update_scenario_graphs(year, threshold, click_data):
    if df.empty or not year or PRICE_COL is None:
        return _empty_fig(), _empty_fig()

    # 1. Filter by year
    dff = df[df["year"] == year].copy()
    if dff.empty:
        return _empty_fig(f"No data for {year}"), _empty_fig()

    # 2. Identify stress events (above threshold)
    dff["is_stress"] = dff[PRICE_COL] > threshold

    # 3. Aggregate to daily max/mean for the top chart
    daily = dff.groupby("date").agg({
        PRICE_COL: "max",
        "is_stress": "any"
    }).reset_index()
    daily.rename(columns={PRICE_COL: "max_price"}, inplace=True)

    # --- Top Figure: Daily Max Price ---
    fig_top = go.Figure()

    # Normal days
    normal = daily[~daily["is_stress"]]
    fig_top.add_trace(go.Scatter(
        x=normal["date"], 
        y=normal["max_price"],
        mode="markers",
        name="Normal Day",
        marker=dict(color="#94a3b8", size=6, opacity=0.6)
    ))

    # Stress days
    stress = daily[daily["is_stress"]]
    fig_top.add_trace(go.Scatter(
        x=stress["date"], 
        y=stress["max_price"],
        mode="markers",
        name="Stress Event",
        marker=dict(color="#ef4444", size=10, line=dict(width=2, color="#7f1d1d"))
    ))

    # Threshold line
    fig_top.add_hline(y=threshold, line_dash="dash", line_color="var(--accent-2)", annotation_text="Threshold")

    fig_top.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=20, b=40),
        xaxis_title="Date",
        yaxis_title="Max Price ($)",
        hovermode="x unified",
        clickmode="event+select",
        autosize=True
    )

    # --- Bottom Figure: Intraday Detail ---
    # Determine which date to show (from clickData or default to max price day)
    if click_data:
        clicked_date = click_data["points"][0]["x"]
        target_date = pd.to_datetime(clicked_date).date()
    else:
        # Default to the day with the highest price
        if not daily.empty:
            target_date = daily.sort_values("max_price", ascending=False).iloc[0]["date"]
        else:
            target_date = dff["date"].iloc[0]

    day_data = dff[dff["date"] == target_date].sort_values("datetime")
    
    if day_data.empty:
        fig_bottom = _empty_fig("No data for selected date")
    else:
        fig_bottom = go.Figure()
        
        # Line chart for the day
        fig_bottom.add_trace(go.Scatter(
            x=day_data["datetime"],
            y=day_data[PRICE_COL],
            mode="lines+markers",
            name="Price",
            line=dict(color="#22d3ee", width=3),
            marker=dict(size=6, color="#0ea5e9")
        ))

        # Highlight points above threshold
        above = day_data[day_data[PRICE_COL] > threshold]
        if not above.empty:
            fig_bottom.add_trace(go.Scatter(
                x=above["datetime"],
                y=above[PRICE_COL],
                mode="markers",
                name="Above Threshold",
                marker=dict(color="#ef4444", size=12, symbol="diamond")
            ))

        fig_bottom.update_layout(
            template="plotly_dark",
            title=f"Intraday Profile: {target_date}",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.5)",
            margin=dict(l=40, r=20, t=40, b=40),
            xaxis_title="Time of Day",
            yaxis_title="Price ($)",
            autosize=True
        )

    return fig_top, fig_bottom