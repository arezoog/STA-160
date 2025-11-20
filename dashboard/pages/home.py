import dash
from dash import html

dash.register_page(__name__, path="/", name="Home")


layout = html.Div(
    style={"paddingTop": "8px"},
    children=[
        # Intro card
        html.Div(
            style={
                "backgroundColor": "#020617",
                "borderRadius": "16px",
                "padding": "18px 20px",
                "boxShadow": "0 10px 30px rgba(0,0,0,0.45)",
                "border": "1px solid #1f2937",
                "marginBottom": "16px",
            },
            children=[
                html.H3(
                    "About this dashboard",
                    style={"marginTop": 0, "marginBottom": "6px", "fontSize": "1.1rem"},
                ),
                html.P(
                    [
                        "This interactive dashboard is part of a Resource Adequacy project using ",
                        html.Span("FERC EQR transaction data (2019–2025)", style={"fontWeight": 600}),
                        ". The goal is to explore when the market enters a high-price, stress-like regime ",
                        "and how that risk changes with different modeling choices.",
                    ],
                    style={"color": "#d1d5db", "fontSize": "0.9rem", "marginBottom": "6px"},
                ),
                html.P(
                    "Under the hood, the app fits a logistic regression model to predict whether a trade "
                    "falls in the upper tail of standardized prices (for example, above the 95th percentile). "
                    "The Dashboard page lets you choose which years to train on, how to define a “stress” "
                    "event, and then visualize out-of-sample performance.",
                    style={"color": "#9ca3af", "fontSize": "0.86rem"},
                ),
            ],
        ),

        # Two-column section
        html.Div(
            style={
                "display": "grid",
                "gridTemplateColumns": "minmax(0, 1.4fr) minmax(0, 1.6fr)",
                "gap": "16px",
                "flexWrap": "wrap",
            },
            children=[
                html.Div(
                    style={
                        "backgroundColor": "#020617",
                        "borderRadius": "16px",
                        "padding": "16px 18px",
                        "boxShadow": "0 10px 30px rgba(0,0,0,0.45)",
                        "border": "1px solid #1f2937",
                        "fontSize": "0.86rem",
                    },
                    children=[
                        html.Div(
                            "What this dashboard shows",
                            style={"fontWeight": 600, "marginBottom": "8px"},
                        ),
                        html.Ul(
                            style={"paddingLeft": "18px", "color": "#d1d5db"},
                            children=[
                                html.Li(
                                    "A probability model for high-price (stress) events, trained on "
                                    "EQR transactions from selected years."
                                ),
                                html.Li(
                                    "Out-of-sample predicted probabilities over time, to see when the "
                                    "model thinks the system is more stressed."
                                ),
                                html.Li(
                                    "ROC curve and summary metrics (accuracy, AUC, Brier score) to judge "
                                    "discrimination and calibration."
                                ),
                                html.Li(
                                    "A scenario explorer where you plug in quantity, total charge, and "
                                    "month to get a predicted stress probability."
                                ),
                            ],
                        ),
                        html.P(
                            "In your written report, you can reference this page as an overview of the data "
                            "source, modeling goal, and how to read the dashboard.",
                            style={"color": "#9ca3af", "marginTop": "8px"},
                        ),
                    ],
                ),

                html.Div(
                    style={
                        "backgroundColor": "#020617",
                        "borderRadius": "16px",
                        "padding": "16px 18px",
                        "boxShadow": "0 10px 30px rgba(0,0,0,0.45)",
                        "border": "1px solid #1f2937",
                        "fontSize": "0.86rem",
                    },
                    children=[
                        html.Div(
                            "How to use the Dashboard page",
                            style={"fontWeight": 600, "marginBottom": "8px"},
                        ),
                        html.Ol(
                            style={"paddingLeft": "20px", "color": "#d1d5db", "marginBottom": "8px"},
                            children=[
                                html.Li("Click the Dashboard link at the top of the page."),
                                html.Li(
                                    "Select one or more years (e.g., 2022–2025). "
                                    "The model trains only on those years."
                                ),
                                html.Li(
                                    "Choose a test-set fraction and an event percentile (e.g., 95th). "
                                    "Higher percentiles correspond to rarer, more extreme price spikes."
                                ),
                                html.Li(
                                    "Click “Train / Refresh model” to fit the logistic regression "
                                    "with those settings. Metrics and plots will update."
                                ),
                                html.Li(
                                    "Use the scenario inputs on the right side of the Dashboard page "
                                    "to plug in Qty, Charge, and Month and read the predicted stress "
                                    "probability."
                                ),
                            ],
                        ),
                        html.P(
                            "This tool is for exploratory analysis and communication, not for real-time "
                            "operations or trading decisions.",
                            style={"color": "#9ca3af"},
                        ),
                    ],
                ),
            ],
        ),
    ],
)

