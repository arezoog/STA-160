import dash
from dash import html, dcc

dash.register_page(__name__, path="/", name="Home")

layout = html.Div(
    className="page-container",
    children=[

        # --- HERO SECTION ----------------------------------------------------
        html.Div(
            className="glass-card",
            style={"marginBottom": "32px", "textAlign": "center", "padding": "40px"},
            children=[
                html.H1("Illuminating the Price of Power"),
                html.H3(
                    "Uncovering hidden costs in California's opaque energy market",
                    style={"color": "var(--accent-1)", "opacity": "0.9"},
                ),
                # Launch Dashboard button removed – navigation now happens via navbar
            ],
        ),

        # --- THE PROBLEM -----------------------------------------------------
        html.Div(
            className="glass-card",
            style={"marginBottom": "32px"},
            children=[
                html.H2("The Problem: Buying 'Grid Insurance' in the Dark"),
                html.P(
                    "When you flip a light switch, you expect the power to be there. "
                    "Behind the scenes, grid operators have to make sure enough capacity "
                    "is available even during heat waves, wildfires, or plant outages. "
                    "In California this reliability obligation is handled through a "
                    "regulatory program called Resource Adequacy (RA)—essentially "
                    "'grid insurance' that ensures there will be power when it is most "
                    "needed."
                ),

                # Figure 1 (grid map) removed because the image never renders reliably

                html.H4("The California Challenge", style={"color": "var(--accent-2)"}),
                html.P(
                    "On the East Coast, many regions run centralized capacity auctions "
                    "where a single market-clearing price is published for each delivery "
                    "period. In California, by contrast, RA capacity is bought and sold "
                    "through bilateral contracts negotiated privately between utilities "
                    "and suppliers. Prices are scattered across thousands of contracts "
                    "and rarely visible beyond the two counterparties, leaving buyers "
                    "with little guidance about what constitutes a fair price."
                ),

                # --- IMAGE 2: MARKET STRUCTURE ---
                html.Div(
                    style={
                        "textAlign": "center",
                        "margin": "32px 0",
                        "padding": "20px",
                        "background": "rgba(0,0,0,0.2)",
                        "borderRadius": "12px",
                        "border": "1px dashed var(--accent-muted)",
                    },
                    children=[
                        html.Img(
                            src="assets/market_structure.png",
                            style={
                                "maxWidth": "100%",
                                "maxHeight": "400px",
                                "borderRadius": "8px",
                                "boxShadow": "0 4px 15px rgba(0,0,0,0.5)",
                            },
                            alt="Bilateral Market vs Centralized Auction",
                        ),
                        html.P(
                            "Figure 1: Opaque Bilateral Markets vs. Transparent Centralized Auctions.",
                            style={
                                "fontSize": "0.85rem",
                                "color": "var(--text-muted)",
                                "marginTop": "12px",
                            },
                        ),
                    ],
                ),

                html.Ul(
                    children=[
                        html.Li(
                            [
                                html.Strong("No Central Price: "),
                                "There is no single 'ticker symbol' for capacity.",
                            ]
                        ),
                        html.Li(
                            [
                                html.Strong("High Opacity: "),
                                "Buyers struggle to know if they are paying a fair price.",
                            ]
                        ),
                        html.Li(
                            [
                                html.Strong("The Risk: "),
                                "Utilities may overpay or fail to secure necessary resources.",
                            ]
                        ),
                    ],
                    style={"color": "var(--text-main)", "lineHeight": "1.8"},
                ),

                html.P(
                    "Grid Sentinel was built to reduce that uncertainty. By turning raw "
                    "FERC transaction filings into a consistent RA price index and "
                    "forward-looking forecasts, the project aims to give planners, "
                    "analysts, and students a clearer view of how much 'grid insurance' "
                    "really costs over time."
                ),
            ],
        ),

        # --- DATA & APPROACH -------------------------------------------------
        html.Div(
            className="glass-card",
            style={"marginBottom": "32px"},
            children=[
                html.H2("Our Approach"),
                html.P(
                    "To shed light on this opaque market, we extracted and cleaned "
                    "Resource Adequacy (RA) capacity transactions from millions of "
                    "records in the Federal Energy Regulatory Commission (FERC) Electric "
                    "Quarterly Reports (EQRs). After filtering to CAISO-delivered "
                    "capacity products and standardizing prices to dollars per "
                    "kW-month, we constructed a monthly time series of weighted-average "
                    "prices and traded quantities from 2019 through mid-2025."
                ),

                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 1fr",
                        "gap": "24px",
                        "marginTop": "24px",
                    },
                    children=[
                        # Left Col: The Challenge
                        html.Div(
                            [
                                html.H4("Dirty Data", style={"color": "#ef4444"}),
                                html.P(
                                    "Real-world energy data is notoriously noisy. We found "
                                    "clerical errors where prices were reported in the "
                                    "millions and quantities in the millions of kilowatts, "
                                    "obscuring the true market signal.",
                                    className="text-muted",
                                ),
                            ]
                        ),
                        # Right Col: The Solution
                        html.Div(
                            [
                                html.H4(
                                    "Rigorous Cleaning", style={"color": "var(--accent-1)"}
                                ),
                                html.P(
                                    "We built a pipeline to standardize units, remove "
                                    "invalid and non-monthly trades, and filter out the "
                                    "top 0.5% of extreme outliers. The result is a "
                                    "robust RA price index that can support meaningful "
                                    "visualization and forecasting.",
                                    className="text-muted",
                                ),
                            ]
                        ),
                    ],
                ),

                html.P(
                    "These cleaned data feed directly into the Grid Sentinel dashboard, "
                    "where users can explore market evolution, examine how prices and "
                    "volumes respond to grid conditions, and compare historical patterns "
                    "across delivery years.",
                    style={"marginTop": "20px"},
                ),
            ],
        ),

        # --- MODELING --------------------------------------------------------
        html.Div(
            className="glass-card",
            style={"marginBottom": "32px", "borderLeft": "4px solid var(--accent-2)"},
            children=[
                html.Div(
                    className="badge-pill",
                    children="Forecasting Engine",
                    style={"marginBottom": "12px", "display": "inline-block"},
                ),
                html.H2("Modeling & Forecasting"),
                html.P(
                    "Using this historical price index, we fitted several forecasting "
                    "models to estimate short-term RA capacity prices. Classical "
                    "time-series approaches such as SARIMA, SARIMAX with exogenous "
                    "market variables, and Holt–Winters exponential smoothing were "
                    "tested alongside machine-learning models including Support Vector "
                    "Regression, Random Forests, and Ridge regression. Models were "
                    "evaluated using rolling backtests to mimic real-world deployment, "
                    "focusing on forecast errors for 1–12-month-ahead horizons.",
                    style={"color": "var(--text-muted)"},
                ),
            ],
        ),

        # --- RESULTS ---------------------------------------------------------
        html.Div(
            className="glass-card",
            style={"borderLeft": "4px solid var(--accent-1)"},
            children=[
                html.Div(
                    className="badge-pill",
                    children="Insights",
                    style={"marginBottom": "12px", "display": "inline-block"},
                ),
                html.H2("Key Findings"),
                html.P(
                    "Across models, approaches that explicitly encode yearly seasonality "
                    "and market conditions—especially the SARIMAX model with exogenous "
                    "variables—produced the lowest forecast errors. Prices tend to rise "
                    "ahead of periods of grid stress and fall when supply is ample, and "
                    "high-price months are typically associated with lower traded "
                    "volumes. While no model can perfectly anticipate policy changes or "
                    "extreme weather events, the forecasts and interactive dashboard "
                    "provide a practical starting point for more informed Resource "
                    "Adequacy procurement decisions in California's bilateral market.",
                    style={"color": "var(--text-muted)"},
                ),
            ],
        ),
    ],
)
