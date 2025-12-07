# dashboard/pages/home.py

import dash
from dash import html, dcc

dash.register_page(__name__, path="/", name="Home")

layout = html.Div(
    className="page-container",
    children=[
        # HERO (shorter)
        html.Div(
            className="glass-card",
            style={"marginBottom": "32px", "textAlign": "center", "padding": "40px"},
            children=[
                html.H1("Grid Sentinel"),
                html.H3(
                    "Interactive RA capacity price explorer for California",
                    style={"color": "var(--accent-1)", "opacity": 0.9},
                ),
                html.P(
                    "Use this dashboard to explore historical RA prices, visualize market stress, "
                    "and run what-if scenarios on future capacity costs.",
                    className="text-muted",
                    style={"marginTop": "10px"},
                ),
                html.P(
                    [
                        "New to the project? ",
                        dcc.Link(
                            "Read the full introduction →",
                            href="/introduction",
                            style={"color": "var(--accent-2)"},
                        ),
                    ],
                    style={"marginTop": "8px", "fontSize": "0.9rem"},
                ),
            ],
        ),

        # QUICK NAV GRID
        html.Div(
            className="glass-card",
            style={"marginBottom": "24px", "padding": "24px"},
            children=[
                html.H2("Start Exploring", style={"textAlign": "center"}),
                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fit, minmax(230px, 1fr))",
                        "gap": "16px",
                        "marginTop": "16px",
                    },
                    children=[
                        # Card 1: Market Analysis
                        html.Div(
                            className="mini-card",
                            children=[
                                html.H4("Market Analysis"),
                                html.P(
                                    "See heatmaps, animated trade flows, and 3D price surfaces to "
                                    "understand when and where RA prices are elevated.",
                                    className="text-muted",
                                    style={"fontSize": "0.9rem"},
                                ),
                                dcc.Link(
                                    "Open Market Analysis →",
                                    href="/insights",
                                    className="nav-link",
                                ),
                            ],
                        ),
                        # Card 2: Forecast & Scenario
                        html.Div(
                            className="mini-card",
                            children=[
                                html.H4("Forecast & Scenario"),
                                html.P(
                                    "Compare forecasting models and run scenario tests that change "
                                    "volume, trades, and price levels for future months.",
                                    className="text-muted",
                                    style={"fontSize": "0.9rem"},
                                ),
                                dcc.Link(
                                    "Open Forecast Tools →",
                                    href="/models",
                                    className="nav-link",
                                ),
                            ],
                        ),
                        # Card 3: Market Evolution
                        html.Div(
                            className="mini-card",
                            children=[
                                html.H4("Market Evolution"),
                                html.P(
                                    "Watch a time-lapse of RA trades as bubbles moving through time, "
                                    "showing how price and quantity evolve.",
                                    className="text-muted",
                                    style={"fontSize": "0.9rem"},
                                ),
                                dcc.Link(
                                    "Play the time-lapse →",
                                    href="/animation",
                                    className="nav-link",
                                ),
                            ],
                        ),
                        # Card 4: Project Introduction
                        html.Div(
                            className="mini-card",
                            children=[
                                html.H4("Project Story"),
                                html.P(
                                    "Learn why California’s RA market is opaque, how we cleaned the data, "
                                    "and what questions this tool is designed to answer.",
                                    className="text-muted",
                                    style={"fontSize": "0.9rem"},
                                ),
                                dcc.Link(
                                    "Read the introduction →",
                                    href="/introduction",
                                    className="nav-link",
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)
