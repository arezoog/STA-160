import os

import dash
from dash import html, dcc

# Enable Dash Pages (multi-page support)
app = dash.Dash(
    __name__,
    use_pages=True,  # 👈 this turns on dash.register_page / page_container
    title="RA Predictive Dashboard",
    suppress_callback_exceptions=True,
)

app.layout = html.Div(
    style={
        "minHeight": "100vh",
        "backgroundColor": "#020617",
        "padding": "24px",
        "fontFamily": "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        "color": "#e5e7eb",
    },
    children=[
        html.Div(
            style={"maxWidth": "1300px", "margin": "0 auto"},
            children=[
                # Header
                html.Div(
                    style={
                        "display": "flex",
                        "justifyContent": "space-between",
                        "alignItems": "baseline",
                        "marginBottom": "12px",
                    },
                    children=[
                        html.Div(
                            children=[
                                html.H2(
                                    "Resource Adequacy – Predictive Dashboard",
                                    style={"margin": 0, "fontWeight": 600, "letterSpacing": "0.02em"},
                                ),
                                html.P(
                                    "Explore how different data windows and event definitions affect a "
                                    "probabilistic model of high-price (stress) events.",
                                    style={"margin": "6px 0 0 0", "color": "#9ca3af", "fontSize": "0.9rem"},
                                ),
                            ]
                        ),
                        html.Div(
                            "EQR 2019–2025",
                            style={
                                "padding": "4px 10px",
                                "borderRadius": "999px",
                                "border": "1px solid #374151",
                                "fontSize": "0.75rem",
                                "color": "#9ca3af",
                            },
                        ),
                    ],
                ),

                # Navigation bar – links to all registered pages
                html.Div(
                    [
                        dcc.Link(
                            page["name"],
                            href=page["path"],
                            style={
                                "marginRight": "16px",
                                "textDecoration": "none",
                                "color": "#e5e7eb",
                            },
                        )
                        for page in dash.page_registry.values()
                    ]
                ),
                html.Hr(style={"borderColor": "#1f2937"}),

                # This is where each page's `layout` shows up
                dash.page_container,

                # Footer
                html.Div(
                    "Prototype dashboard – not for operational decision-making.",
                    style={
                        "marginTop": "12px",
                        "fontSize": "0.7rem",
                        "color": "#6b7280",
                        "textAlign": "right",
                    },
                ),
            ],
        )
    ],
)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    # Dash creates its own Flask server internally (app.server)
    app.run(host="0.0.0.0", port=port, debug=False)

