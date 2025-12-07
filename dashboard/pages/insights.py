import dash
from dash import html, dcc, Input, Output, callback

# Import the figure builders from your existing pages
from .eda import build_delivery_animation_figure
from .risk import build_risk_heatmap
from .surface import build_surface_figure, YEAR_OPTIONS, DEFAULT_YEAR_VALUE

# This registers a new nav item "Market Insights"
dash.register_page(__name__, path="/insights", name="Market Analysis")


def layout():
    return html.Div(
        className="page-container",
        children=[
            # --- Page intro ---------------------------------------------------
            html.Div(
                className="glass-card",
                style={"marginBottom": "16px"},
                children=[
                    html.H2("Market Analysis Overview"),
                    html.P(
                        "This page gives three different lenses on the California Resource Adequacy (RA) "
                        "capacity market. Even without prior energy background, you can use these views to "
                        "see when trading activity happens, where prices tend to be higher, and how those "
                        "patterns change over time.",
                        className="text-muted",
                        style={"fontSize": "0.95rem"},
                    ),
                    html.Ul(
                        style={"marginLeft": "20px", "lineHeight": "1.7"},
                        children=[
                            html.Li(
                                [
                                    html.Strong("Market Flow Explorer – "),
                                    "shows how trades for a given delivery month build up across trade months.",
                                ]
                            ),
                            html.Li(
                                [
                                    html.Strong("Portfolio Stress Lens – "),
                                    "highlights months and years where average prices are elevated.",
                                ]
                            ),
                            html.Li(
                                [
                                    html.Strong("Market Price Topography – "),
                                    "shows a 3D surface of prices across months of the year and hours of the day.",
                                ]
                            ),
                        ],
                    ),
                ],
            ),

            # --- EDA animated chart ----------------------------------------
            html.Div(
                className="glass-card",
                children=[
                    html.Div(
                        className="card-header",
                        children=[
                            html.Div(
                                children=[
                                    html.H1(
                                        "Market Flow Explorer",
                                        className="gradient-text",
                                    ),
                                    html.P(
                                        "Animated view of trade volumes and prices for each delivery month, "
                                        "helping you see how forward curves fill in over time.",
                                        className="text-muted",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className="card-body",
                        children=[
                            html.H4(
                                "What this graph shows",
                                style={
                                    "marginTop": "4px",
                                    "marginBottom": "4px",
                                    "fontSize": "1rem",
                                },
                            ),
                            html.P(
                                "This graph shows how trading activity for a single delivery month builds up "
                                "across the months when trades are actually executed. Each frame focuses on one "
                                "delivery month and plots both the price and the total quantity traded over the "
                                "timeline of trade months.",
                                className="text-muted",
                                style={"fontSize": "0.9rem"},
                            ),
                            html.Ul(
                                style={"marginLeft": "20px", "lineHeight": "1.7"},
                                children=[
                                    html.Li(
                                        [
                                            html.Strong("Line: "),
                                            "weighted-average price of RA capacity as you move along the x-axis "
                                            "(trade month).",
                                        ]
                                    ),
                                    html.Li(
                                        [
                                            html.Strong("Bars: "),
                                            "total quantity of capacity traded in each trade month for that "
                                            "delivery month.",
                                        ]
                                    ),
                                    html.Li(
                                        [
                                            html.Strong("Frames / slider: "),
                                            "each frame represents a different delivery month (for example "
                                            "“2024-08” delivery).",
                                        ]
                                    ),
                                ],
                            ),
                            html.H4(
                                "How to interact",
                                style={
                                    "marginTop": "10px",
                                    "marginBottom": "4px",
                                    "fontSize": "1rem",
                                },
                            ),
                            html.Ol(
                                style={"marginLeft": "20px", "lineHeight": "1.7"},
                                children=[
                                    html.Li(
                                        "Use the slider or Play button under the chart to move through delivery "
                                        "months in order. Watch how some delivery months see early, steady trading "
                                        "while others have late surges."
                                    ),
                                    html.Li(
                                        "Look for delivery months where the price line spikes or where most volume "
                                        "only appears just before delivery—these can signal tighter conditions."
                                    ),
                                    html.Li(
                                        "Hover over the line or bars to see exact price and quantity values for "
                                        "specific trade months.",
                                    ),
                                ],
                            ),
                            dcc.Graph(
                                id="eda-animated-delivery-combined",
                                className="dash-graph",
                                style={"height": "480px"},
                                figure=build_delivery_animation_figure(),
                            ),
                        ],
                    ),
                ],
            ),

            # --- Risk heatmap ----------------------------------------------
            html.Div(
                className="glass-card",
                style={"marginTop": "16px"},
                children=[
                    html.Div(
                        className="card-header",
                        children=[
                            html.Div(
                                children=[
                                    html.H1(
                                        "Portfolio Stress Lens",
                                        className="gradient-text",
                                    ),
                                    html.P(
                                        "Calendar-style heatmap of average prices by year and month, "
                                        "highlighting when the market tends to be more expensive.",
                                        className="text-muted",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        className="card-body",
                        children=[
                            html.H4(
                                "What this heatmap shows",
                                style={
                                    "marginTop": "4px",
                                    "marginBottom": "4px",
                                    "fontSize": "1rem",
                                },
                            ),
                            html.P(
                                "Each tile on this heatmap is a year–month pair. The color of the tile reflects "
                                "the average price for that month in that year. Darker or hotter colors indicate "
                                "higher average prices, while lighter tiles suggest calmer, cheaper conditions.",
                                className="text-muted",
                                style={"fontSize": "0.9rem"},
                            ),
                            html.Ul(
                                style={"marginLeft": "20px", "lineHeight": "1.7"},
                                children=[
                                    html.Li(
                                        [
                                            html.Strong("Rows: "),
                                            "calendar years (earlier years at the top, later years at the bottom).",
                                        ]
                                    ),
                                    html.Li(
                                        [
                                            html.Strong("Columns: "),
                                            "months from January through December.",
                                        ]
                                    ),
                                    html.Li(
                                        [
                                            html.Strong("Tile color: "),
                                            "average RA capacity price—darker tiles point to more expensive months.",
                                        ]
                                    ),
                                ],
                            ),
                            html.H4(
                                "How to use it",
                                style={
                                    "marginTop": "10px",
                                    "marginBottom": "4px",
                                    "fontSize": "1rem",
                                },
                            ),
                            html.Ol(
                                style={"marginLeft": "20px", "lineHeight": "1.7"},
                                children=[
                                    html.Li(
                                        "Scan across a single row (one year) to see which months were relatively "
                                        "expensive or cheap in that year."
                                    ),
                                    html.Li(
                                        "Scan down a single column (one month across years) to check whether certain "
                                        "months, such as late summer, tend to be high priced over and over again."
                                    ),
                                    html.Li(
                                        "Look for years where many tiles are dark—these are periods of generally "
                                        "elevated prices that may warrant extra attention in planning.",
                                    ),
                                ],
                            ),
                            dcc.Graph(
                                id="risk-heatmap-combined",
                                className="dash-graph",
                                style={"height": "430px"},
                                figure=build_risk_heatmap(),
                            ),
                        ],
                    ),
                ],
            ),

            # --- 3D Price surface ------------------------------------------
            html.Div(
                className="glass-card",
                style={"marginTop": "16px"},
                children=[
                    html.H2("Market Price Topography"),
                    html.P(
                        "3D surface of average prices across month of year and hour of day. "
                        "Rotate the view to see where high-price ridges and low-price valleys form.",
                        className="text-muted",
                        style={"fontSize": "0.9rem"},
                    ),
                    html.H4(
                        "What this surface shows",
                        style={
                            "marginTop": "4px",
                            "marginBottom": "4px",
                            "fontSize": "1rem",
                        },
                    ),
                    html.P(
                        "This chart condenses many trades into a smooth \"terrain map\" of prices. "
                        "Each point on the surface corresponds to a specific month of the year and hour "
                        "of the day, colored and raised according to the average price at that time.",
                        className="text-muted",
                        style={"fontSize": "0.9rem"},
                    ),
                    html.Ul(
                        style={"marginLeft": "20px", "lineHeight": "1.7"},
                        children=[
                            html.Li(
                                [
                                    html.Strong("X-axis (Month): "),
                                    "month of the calendar year (1–12).",
                                ]
                            ),
                            html.Li(
                                [
                                    html.Strong("Y-axis (Hour): "),
                                    "hour of the day (0–23).",
                                ]
                            ),
                            html.Li(
                                [
                                    html.Strong("Height and color: "),
                                    "average price—higher peaks and brighter colors indicate higher prices.",
                                ]
                            ),
                        ],
                    ),
                    html.H4(
                        "How to interact",
                        style={
                            "marginTop": "10px",
                            "marginBottom": "4px",
                            "fontSize": "1rem",
                        },
                    ),
                    html.Ol(
                        style={"marginLeft": "20px", "lineHeight": "1.7"},
                        children=[
                            html.Li(
                                "Use the 'Year filter' dropdown to switch between seeing all years combined or "
                                "focusing on a single year."
                            ),
                            html.Li(
                                "Click and drag on the surface to rotate the view. Adjust the angle until high-price "
                                "ridges and low-price basins are easy to see."
                            ),
                            html.Li(
                                "Zoom in with your mouse or trackpad to inspect particular regions, such as summer "
                                "evenings or early-morning hours.",
                            ),
                            html.Li(
                                "Look for recurring patterns—for example, whether certain months and hours consistently "
                                "sit on higher terrain compared to others.",
                            ),
                        ],
                    ),
                    html.Div(
                        style={
                            "display": "flex",
                            "justifyContent": "flex-start",
                            "alignItems": "center",
                            "marginBottom": "0.75rem",
                            "gap": "0.75rem",
                        },
                        children=[
                            html.Span(
                                "Year filter:",
                                className="text-muted",
                                style={"fontSize": "0.85rem"},
                            ),
                            dcc.Dropdown(
                                id="insights-surface-year-dropdown",
                                options=YEAR_OPTIONS,
                                value=DEFAULT_YEAR_VALUE,
                                clearable=False,
                                className="dropdown-dark",
                                style={"width": "220px"},
                            ),
                        ],
                    ),
                    dcc.Graph(
                        id="price-surface-combined",
                        className="dash-graph",
                        style={"height": "70vh", "borderRadius": "16px"},
                        config={"displayModeBar": False},
                        figure=build_surface_figure(),
                    ),
                ],
            ),
        ],
    )


@callback(
    Output("price-surface-combined", "figure"),
    Input("insights-surface-year-dropdown", "value"),
)
def update_price_surface_combined(selected_year_value):
    year = None if selected_year_value in (None, -1) else int(selected_year_value)
    return build_surface_figure(year)
