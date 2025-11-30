import dash
from dash import html, dcc

dash.register_page(__name__, path="/", name="Home")

layout = html.Div(
    style={"maxWidth": "1000px", "margin": "40px auto", "padding": "0 16px"},
    children=[
        
        # --- HERO SECTION ----------------------------------------------------
        html.Div(
            className="glass-card",
            style={"marginBottom": "32px", "textAlign": "center", "padding": "40px"},
            children=[
                html.H1("Illuminating the Price of Power"),
                html.H3("Uncovering hidden costs in California's opaque energy market", 
                        style={"color": "var(--accent-1)", "opacity": "0.9"}),
                
                html.Div(
                    style={"marginTop": "24px"},
                    children=[
                        dcc.Link(
                            html.Button("Launch Dashboard", className="primary-button"),
                            href="/dashboard"
                        ),
                    ]
                )
            ]
        ),

        # --- THE PROBLEM -----------------------------------------------------
        html.Div(
            className="glass-card",
            style={"marginBottom": "32px"},
            children=[
                html.H2("The Problem: Buying 'Grid Insurance' in the Dark"),
                html.P(
                    "When you flip a light switch, you expect the power to be there. "
                    "But how does the grid guarantee reliability during a heatwave or emergency? "
                    "The answer is a regulatory framework called Resource Adequacy (RA)."
                ),
                
                # --- IMAGE 1: GRID MAP ---
                html.Div(
                    style={
                        "textAlign": "center",
                        "margin": "32px 0",
                        "padding": "20px",
                        "background": "rgba(0,0,0,0.2)",
                        "borderRadius": "12px",
                        "border": "1px dashed var(--accent-muted)"
                    },
                    children=[
                        html.Img(
                            src="assets/Electricity_grid_simple-_North_America.svg.png", 
                            style={
                                "maxWidth": "100%", 
                                "maxHeight": "350px", 
                                "borderRadius": "8px",
                                "boxShadow": "0 4px 15px rgba(0,0,0,0.5)"
                            },
                            alt="North American Electric Grid Map"
                        ),
                        html.P(
                            "Figure 1: The North American Electric Grid Interconnections.",
                            style={"fontSize": "0.85rem", "color": "var(--text-muted)", "marginTop": "12px"}
                        )
                    ]
                ),

                html.H4("The California Challenge", style={"color": "var(--accent-2)"}),
                html.P(
                    "Unlike the East Coast, where capacity is traded in transparent centralized auctions, "
                    "California operates a Bilateral Market. Buyers and suppliers negotiate contracts privately "
                    "behind closed doors."
                ),

                # --- IMAGE 2: MARKET STRUCTURE ---
                html.Div(
                    style={
                        "textAlign": "center",
                        "margin": "32px 0",
                        "padding": "20px",
                        "background": "rgba(0,0,0,0.2)",
                        "borderRadius": "12px",
                        "border": "1px dashed var(--accent-muted)"
                    },
                    children=[
                        # NOTE: Ensure you renamed your uploaded file 'Chatgpt#27.png' to 'market_structure.png'
                        html.Img(
                            src="assets/market_structure.png", 
                            style={
                                "maxWidth": "100%", 
                                "maxHeight": "400px", 
                                "borderRadius": "8px",
                                "boxShadow": "0 4px 15px rgba(0,0,0,0.5)"
                            },
                            alt="Bilateral Market vs Centralized Auction"
                        ),
                        html.P(
                            "Figure 2: Opaque Bilateral Markets vs. Transparent Centralized Auctions.",
                            style={"fontSize": "0.85rem", "color": "var(--text-muted)", "marginTop": "12px"}
                        )
                    ]
                ),

                html.Ul(
                    children=[
                        html.Li([html.Strong("No Central Price: "), "There is no 'ticker symbol' for capacity."]),
                        html.Li([html.Strong("High Opacity: "), "Buyers struggle to know if they are paying a fair price."]),
                        html.Li([html.Strong("The Risk: "), "Utilities may overpay or fail to secure necessary resources."]),
                    ],
                    style={"color": "var(--text-main)", "lineHeight": "1.8"}
                ),
            ]
        ),

        # --- DATA & APPROACH -------------------------------------------------
        html.Div(
            className="glass-card",
            style={"marginBottom": "32px"},
            children=[
                html.H2("Our Approach"),
                html.P(
                    "To shed light on this opaque market, we analyzed over 33,000 transaction records "
                    "from the Federal Energy Regulatory Commission (FERC)."
                ),

                html.Div(
                    style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "24px", "marginTop": "24px"},
                    children=[
                        # Left Col: The Challenge
                        html.Div([
                            html.H4("Dirty Data", style={"color": "#ef4444"}),
                            html.P("Real-world energy data is notoriously noisy. We found clerical errors where prices "
                                   "were reported in the millions, skewing the market signal.", className="text-muted")
                        ]),
                        # Right Col: The Solution
                        html.Div([
                            html.H4("Rigorous Cleaning", style={"color": "var(--accent-1)"}),
                            html.P("We built a pipeline to standardize units and filter out the top 0.5% of extreme outliers, "
                                   "resulting in a robust price index.", className="text-muted")
                        ])
                    ]
                )
            ]
        ),

        # --- MODELING (PLACEHOLDER) ------------------------------------------
        html.Div(
            className="glass-card",
            style={"marginBottom": "32px", "borderLeft": "4px solid var(--accent-2)"},
            children=[
                html.Div(className="badge-pill", children="In Progress", style={"marginBottom": "12px", "display": "inline-block"}),
                html.H2("Modeling Methodology"),
                html.P(
                    "We are currently developing a time-series model to forecast future capacity prices based on "
                    "historical seasonal trends and grid stress events.",
                    style={"fontStyle": "italic", "color": "var(--text-muted)"}
                ),
            ]
        ),

        # --- RESULTS (PLACEHOLDER) -------------------------------------------
        html.Div(
            className="glass-card",
            style={"borderLeft": "4px solid var(--accent-1)"},
            children=[
                html.Div(className="badge-pill", children="Pending", style={"marginBottom": "12px", "display": "inline-block"}),
                html.H2("Key Findings"),
                html.P(
                    "Early results indicate a strong correlation between heatwave events and bilateral price spikes. "
                    "Full predictive results will be published here upon completion.",
                    style={"fontStyle": "italic", "color": "var(--text-muted)"}
                ),
            ]
        ),
    ]
)