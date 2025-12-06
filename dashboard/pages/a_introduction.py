# dashboard/pages/introduction.py

import dash
from dash import html

dash.register_page(
    __name__,
    path="/introduction",
    name="Introduction",
)

layout = html.Div(
    className="page-container",
    children=[
        # HERO
        html.Div(
            className="glass-card",
            style={"marginBottom": "32px", "textAlign": "center", "padding": "40px"},
            children=[
                html.H1("Illuminating the Price of Power"),
                html.H3(
                    "Uncovering hidden costs in California's opaque energy market",
                    style={"color": "var(--accent-1)", "opacity": "0.9"},
                ),
            ],
        ),

        # PROBLEM SECTION
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
                    style={"lineHeight": "1.8"},
                ),
            ],
        ),

        # DATA & APPROACH
        html.Div(
            className="glass-card",
            style={"marginBottom": "32px"},
            children=[
                html.H2("Data & Approach"),
                html.P(
                    "We extracted and cleaned RA capacity transactions from the FERC EQR "
                    "data, filtered to CAISO-delivered capacity, standardized prices to "
                    "$/kW-month, and constructed a monthly time series of "
                    "weighted-average prices and traded quantities from 2019 onward."
                ),
                html.P(
                    "These cleaned data feed directly into the Grid Sentinel dashboard, "
                    "where users can explore market evolution, examine how prices and "
                    "volumes respond to grid conditions, and compare historical patterns "
                    "across delivery years. The year–month heatmap is built from this "
                    "series by averaging prices for each delivery month and year."
                ),
                html.P(
                    "Some tiles in the heatmap show an average price of zero. These do "
                    "not mean that RA capacity was free; they indicate months where, "
                    "after filtering and outlier trimming, no qualifying trades remain "
                    "in the dataset. We keep those months in the grid and fill missing "
                    "values with 0 so users can still see where data is absent."
                ),
            ],
        ),

        # MODELING & RESULTS SUMMARY
        html.Div(
            className="glass-card",
            style={"marginBottom": "32px"},
            children=[
                html.H2("Modeling & Results in Brief"),
                html.P(
                    "After constructing a cleaned RA capacity dataset, we modeled the "
                    "monthly weighted-average price as the main series to forecast. We "
                    "first fit classical time-series models, including SARIMA and "
                    "Holt–Winters with yearly seasonality, to capture trend and the "
                    "strong annual cycle in capacity prices. We then extended this to a "
                    "SARIMAX model with exogenous variables such as total standardized "
                    "quantity, total transaction charge, average trade price, number of "
                    "trades, and the share of long-term contracts, allowing the model "
                    "to respond to changing market conditions."
                ),
                html.P(
                    "In parallel, we developed machine-learning models on the same price "
                    "index. Support Vector Regression with an RBF kernel and a Random "
                    "Forest regressor used lagged prices, calendar features, and "
                    "transaction aggregates to make one- and two-month-ahead forecasts. "
                    "All models were evaluated using rolling backtests that train only "
                    "on past data and repeatedly forecast the next month, mirroring how "
                    "the tools would be used in practice."
                ),
                html.P(
                    "Across experiments, the SARIMAX model with exogenous variables "
                    "achieved the lowest MAE and RMSE, reliably capturing the overall "
                    "level and annual pattern of RA prices. Holt–Winters and baseline "
                    "SARIMA performed well on smooth seasonal behavior, while tuned SVR "
                    "and Random Forest delivered competitive short-term forecasts but "
                    "struggled with rare price spikes."
                ),
                html.P(
                    "The scenario tools in the dashboard apply these models at the level "
                    "of a representative RA contract in each month: baseline t+1 values "
                    "reflect the historical next-period price for that contract, while "
                    "the SVR and Random Forest forecasts show what next month's price "
                    "could be for a similar contract under the user’s chosen volume, "
                    "trade-count, and price conditions."
                ),
            ],
        ),
    ],
)
