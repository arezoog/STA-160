import dash
from dash import html, dcc, Input, Output, callback
import pandas as pd
import plotly.graph_objects as go

from model import get_forecast_dashboard_data

result = get_forecast_dashboard_data()
data = result["data"]
feature_cols = result["feature_cols"]
scaler = result["scaler"]
svr_model = result["svr_model"]      # MultiOutputRegressor(SVR)
rf_model = result["rf_model"]        # MultiOutputRegressor(RandomForest)

period_dt = pd.to_datetime(data["period"])
data_ui = data[period_dt.dt.year >= 2019]

# Dropdown options for base period
period_options = [
    {"label": pd.to_datetime(p).strftime("%b %Y"), "value": str(p)}
    for p in sorted(data_ui["period"].unique())
]

default_period = period_options[-1]["value"] if period_options else None

# Slider ranges / marks from empirical distribution
qty_series = data["total_transacted_quantity"]
qty_min = float(qty_series.quantile(0.05))
qty_max = float(qty_series.quantile(0.95))
qty_mid = (qty_min + qty_max) / 2
qty_default = float(qty_series.iloc[-1])

qty_marks = {
    int(qty_min): f"{int(qty_min):,}",
    int(qty_mid): f"{int(qty_mid):,}",
    int(qty_max): f"{int(qty_max):,}",
}

trades_series = data["num_trades"]
trades_min = float(trades_series.quantile(0.05))
trades_max = float(trades_series.quantile(0.95))
trades_mid = (trades_min + trades_max) / 2
trades_default = float(trades_series.iloc[-1])

trades_marks = {
    int(trades_min): f"{int(trades_min):,}",
    int(trades_mid): f"{int(trades_mid):,}",
    int(trades_max): f"{int(trades_max):,}",
}


# -----------------------------------------------------------
# Layout
# -----------------------------------------------------------
layout = html.Div(
    className="page-container",
    children=[
        # Header
        html.Div(
            className="page-header",
            children=[
                html.H2("Scenario Lab"),
                html.P(
                    "This page lets you stress-test the RA price models under different market conditions. "
                    "Start from a real historical month, then adjust key drivers like total quantity, number "
                    "of trades, and average price to see how two models (SVR and Random Forest) respond.",
                    className="text-muted",
                    style={"fontSize": "0.95rem"},
                ),
                html.Ul(
                    style={"marginLeft": "20px", "lineHeight": "1.7"},
                    children=[
                        html.Li(
                            [
                                html.Strong("Base month: "),
                                "the historical starting point for your scenario (e.g., Aug 2024).",
                            ]
                        ),
                        html.Li(
                            [
                                html.Strong("Total transacted quantity: "),
                                "represents how much RA capacity is traded in that month.",
                            ]
                        ),
                        html.Li(
                            [
                                html.Strong("Number of trades: "),
                                "how many individual deals occur in that month.",
                            ]
                        ),
                        html.Li(
                            [
                                html.Strong("Average standardized price multiplier: "),
                                "scales the typical price level up or down (e.g., 1.2x = 20% higher prices).",
                            ]
                        ),
                    ],
                ),
            ],
        ),

        # Two-column grid: left inputs, right outputs
        html.Div(
            className="scenario-grid",     # style in CSS as display:flex; gap: etc.
            children=[
                # LEFT: Controls column
                html.Div(
                    className="scenario-input-card",
                    children=[
                        html.H3("Scenario Inputs"),
                        html.P(
                            "Choose a base month, then adjust the sliders to define your scenario. "
                            "The ranges come from the actual historical distribution, so values near "
                            "the edges represent relatively extreme conditions.",
                            className="text-muted",
                            style={"fontSize": "0.9rem"},
                        ),

                        html.H4(
                            "1. Pick a base month",
                            style={
                                "marginTop": "10px",
                                "marginBottom": "4px",
                                "fontSize": "0.9rem",
                            },
                        ),
                        html.Label("Base Month"),
                        dcc.Dropdown(
                            id="scenario-period",
                            options=period_options,
                            value=default_period,
                            clearable=False,
                            className="dropdown-dark",
                        ),

                        html.Div(style={"height": "12px"}),

                        html.H4(
                            "2. Adjust market conditions",
                            style={
                                "marginTop": "10px",
                                "marginBottom": "4px",
                                "fontSize": "0.9rem",
                            },
                        ),
                        html.Label("Total Transacted Quantity (MWh, override)"),
                        dcc.Slider(
                            id="scenario-qty",
                            min=qty_min,
                            max=qty_max,
                            step=(qty_max - qty_min) / 50,
                            value=qty_default,
                            marks=qty_marks,
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                        html.Div(
                            id="scenario-qty-label",
                            className="slider-value-label",
                            style={"marginTop": "4px"},
                        ),

                        html.Div(style={"height": "18px"}),

                        html.Label("Number of Trades (override)"),
                        dcc.Slider(
                            id="scenario-trades",
                            min=trades_min,
                            max=trades_max,
                            step=5,
                            value=trades_default,
                            marks=trades_marks,
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                        html.Div(
                            id="scenario-trades-label",
                            className="slider-value-label",
                            style={"marginTop": "4px"},
                        ),

                        html.Div(style={"height": "18px"}),

                        html.Label("Average Standardized Price multiplier (x)"),
                        dcc.Slider(
                            id="scenario-price-mult",
                            min=0.8,
                            max=1.2,
                            step=0.01,
                            value=1.0,
                            marks={0.8: "0.8x", 1.0: "1.0x", 1.2: "1.2x"},
                            tooltip={"placement": "bottom", "always_visible": False},
                        ),
                        html.P(
                            "Values below 1.0x simulate a softer price environment, "
                            "while values above 1.0x simulate higher typical prices for the base month.",
                            className="text-muted",
                            style={"fontSize": "0.8rem", "marginTop": "6px"},
                        ),
                    ],
                ),

                # RIGHT: Summary + two side-by-side graphs
                html.Div(
                    className="scenario-output-card",
                    children=[
                        html.H3("Scenario Results"),
                        html.P(
                            "Once you set your scenario on the left, this panel shows how the SVR and "
                            "Random Forest models forecast prices for the next two months (t+1 and t+2). "
                            "You can compare these scenario forecasts to the original baseline prices "
                            "for that same base month.",
                            className="text-muted",
                            style={"fontSize": "0.9rem"},
                        ),
                        dcc.Loading(
                            type="default",
                            children=[
                                html.Div(id="scenario-summary"),
                                html.Div(style={"height": "16px"}),
                                html.Div(
                                    className="scenario-chart-row",  # style as flex row
                                    children=[
                                        dcc.Graph(
                                            id="scenario-chart-t1",
                                            config={"displayModeBar": False},
                                            style={"height": "350px"},
                                        ),
                                        dcc.Graph(
                                            id="scenario-chart-t2",
                                            config={"displayModeBar": False},
                                            style={"height": "350px"},
                                        ),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


@callback(
    Output("scenario-qty-label", "children"),
    Output("scenario-trades-label", "children"),
    Output("scenario-summary", "children"),
    Output("scenario-chart-t1", "figure"),
    Output("scenario-chart-t2", "figure"),
    Input("scenario-period", "value"),
    Input("scenario-qty", "value"),
    Input("scenario-trades", "value"),
    Input("scenario-price-mult", "value"),
)
def run_scenario(period_value, qty_value, trades_value, price_mult):
    # Empty figures helper
    def empty_fig():
        return go.Figure().update_layout(
            template="plotly_dark",
            xaxis_title="",
            yaxis_title="",
            margin=dict(l=40, r=10, t=30, b=40),
        )

    if period_value is None:
        return (
            "",
            "",
            "Select a base month to run a scenario.",
            empty_fig(),
            empty_fig(),
        )

    # Base row
    base_rows = data.loc[data["period"] == period_value]
    if base_rows.empty:
        return (
            "",
            "",
            f"No data found for period {period_value}.",
            empty_fig(),
            empty_fig(),
        )

    row = base_rows.iloc[-1].copy()

    # Baseline t+1/t+2 (actuals or labels)
    baseline_t1 = row.get("target_1", None)
    baseline_t2 = row.get("target_2", None)

    # Override scenario inputs
    row["total_transacted_quantity"] = qty_value
    row["num_trades"] = trades_value

    if "avg_std_price" in row.index and pd.notna(row["avg_std_price"]):
        row["avg_std_price"] = row["avg_std_price"] * price_mult

    # Build feature vector
    X_scenario = pd.DataFrame([row])[feature_cols]
    X_scaled = scaler.transform(X_scenario)

    svr_pred = svr_model.predict(X_scaled)[0]  # [t+1, t+2]
    rf_pred = rf_model.predict(X_scenario)[0]  # [t+1, t+2]

    svr_t1, svr_t2 = svr_pred[0], svr_pred[1]
    rf_t1, rf_t2 = rf_pred[0], rf_pred[1]

    # Slider labels
    qty_label = f"Current setting: {qty_value:,.0f} MWh"
    trades_label = f"Current setting: {trades_value:,.0f} trades"

    # Summary text + forecast cards
    summary = html.Div(
        children=[
            html.H3(
                f"Scenario results for {pd.to_datetime(period_value).strftime('%b %Y')}"
            ),
            html.P(
                f"Using total quantity = {qty_value:,.0f} MWh, "
                f"num trades = {trades_value:,.0f}, "
                f"price multiplier = {price_mult:.2f}x.",
                className="text-muted",
            ),
            html.Div(
                className="scenario-cards",
                children=[
                    html.Div(
                        className="scenario-card",
                        children=[
                            html.H4("SVR Forecast"),
                            html.P(f"t+1 price: {svr_t1:,.2f}"),
                            html.P(f"t+2 price: {svr_t2:,.2f}"),
                        ],
                    ),
                    html.Div(
                        className="scenario-card",
                        children=[
                            html.H4("Random Forest Forecast"),
                            html.P(f"t+1 price: {rf_t1:,.2f}"),
                            html.P(f"t+2 price: {rf_t2:,.2f}"),
                        ],
                    ),
                ],
            ),
            html.P(
                "Bars in the comparison charts below show how scenario forecasts compare to the baseline "
                "t+1 and t+2 prices. This helps you see whether your hypothetical conditions push the "
                "models toward higher or lower price expectations.",
                className="text-muted",
                style={"marginTop": "10px", "fontSize": "0.85rem"},
            ),
        ]
    )

    # --------- Build side-by-side charts ---------

    # Chart 1: t+1 comparison
    x1 = ["Baseline t+1", "SVR t+1", "RF t+1"]
    y1 = [
        baseline_t1 if pd.notna(baseline_t1) else None,
        svr_t1,
        rf_t1,
    ]
    colors1 = ["#888888", "#22d3ee", "#a855f7"]

    fig_t1 = go.Figure(
        data=[go.Bar(x=x1, y=y1, marker_color=colors1)]
    )
    fig_t1.update_layout(
        template="plotly_dark",
        title="t+1 Price: Baseline vs Scenario",
        xaxis_title="",
        yaxis_title="Price",
        margin=dict(l=40, r=10, t=40, b=50),
    )

    # Chart 2: t+2 comparison
    x2 = ["Baseline t+2", "SVR t+2", "RF t+2"]
    y2 = [
        baseline_t2 if pd.notna(baseline_t2) else None,
        svr_t2,
        rf_t2,
    ]
    colors2 = ["#888888", "#22d3ee", "#a855f7"]

    fig_t2 = go.Figure(
        data=[go.Bar(x=x2, y=y2, marker_color=colors2)]
    )
    fig_t2.update_layout(
        template="plotly_dark",
        title="t+2 Price: Baseline vs Scenario",
        xaxis_title="",
        yaxis_title="Price",
        margin=dict(l=40, r=10, t=40, b=50),
    )

    return qty_label, trades_label, summary, fig_t1, fig_t2
