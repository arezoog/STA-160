import os
import dash
from dash import html, dcc

# Initialize multi-page Dash app
app = dash.Dash(
    __name__,
    use_pages=True,
    title="Resource Adequacy – Predictive Dashboard",
    suppress_callback_exceptions=True,
)

# Expose server for deployment (Render, etc.)
server = app.server


def serve_layout():
    return html.Div(
        id="app",
        children=[
            # Track URL for multipage routing
            dcc.Location(id="url"),

            # ===== TOP NAVBAR =====
            html.Nav(
                className="navbar",
                children=html.Div(
                    className="navbar-inner",
                    children=[
                        # Left: title + subtitle
                        html.Div(
                            children=[
                                html.Div(
                                    "Resource Adequacy – Predictive Dashboard",
                                    className="nav-title",
                                ),
                                html.Div(
                                    "Explore how different data windows and event definitions affect a probabilistic model of high-price (stress) events.",
                                    className="text-muted",
                                ),
                            ]
                        ),
                        # Center/Right: nav links
                        html.Div(
                            className="nav-links",
                            children=[
                                dcc.Link("Home", href="/", className="nav-link"),
                                dcc.Link(
                                    "Market Evolution",
                                    href="/animation",  # matches advanced_viz.py
                                    className="nav-link",
                                ),
                                dcc.Link(
                                    "Dashboard",
                                    href="/dashboard",
                                    className="nav-link",
                                ),
                                dcc.Link(
                                    "Exploratory Analysis",
                                    href="/eda",  # matches eda.py
                                    className="nav-link",
                                ),
                                dcc.Link(
                                    "Risk Analysis",
                                    href="/risk",  # matches risk.py
                                    className="nav-link",
                                ),
                                dcc.Link(
                                    "Scenario Lab",
                                    href="/scenarios",  # matches scenarios.py
                                    className="nav-link",
                                ),
                                dcc.Link(
                                    "3D Market Terrain",
                                    href="/surface",  # matches surface.py
                                    className="nav-link",
                                ),
                            ],
                        ),
                        # Right: pill badge
                        html.Div(
                            className="badge-pill",
                            children=["EQR 2019–2025"],
                        ),
                    ],
                ),
            ),

            # ===== MAIN CONTENT =====
            html.Main(
                className="app-container",
                children=[
                    dash.page_container,
                    html.Footer(
                        className="text-muted",
                        children=html.Div(
                            "Prototype dashboard for STA 160 – CAISO Resource Adequacy stress analysis.",
                            style={
                                "textAlign": "center",
                                "padding": "24px 0 32px 0",
                                "fontSize": "0.8rem",
                            },
                        ),
                    ),
                ],
            ),
        ],
    )


# Use a callable layout (good practice)
app.layout = serve_layout

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=True)
