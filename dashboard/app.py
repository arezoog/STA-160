import os
import dash
from dash import html, dcc

app = dash.Dash(
    __name__, 
    use_pages=True, 
    suppress_callback_exceptions=True
)

# Expose the server for deployment (e.g., Gunicorn)
server = app.server

app.layout = html.Div([
    
    # --- NAVBAR START ---
    html.Nav(
        className="navbar",
        children=[
            html.Div(
                className="navbar-inner",
                children=[
                    # 1. LOGO / BRAND (Left Side)
                    dcc.Link(
                        "TABS",
                        href="/",
                        className= "nav-brand",
                    ),

                    # 2. NAVIGATION TABS (Right Side)
                    # We wrap these in 'nav-links' to apply the Flexbox gap from CSS
                    html.Div(
                        className="nav-links",
                        children=[
                            dcc.Link(
                                page['name'], 
                                href=page["relative_path"], 
                                className="nav-link"
                            )
                            for page in dash.page_registry.values()
                        ]
                    )
                ]
            )
        ]
    ),
    # --- NAVBAR END ---

    # Main Content Area
    dash.page_container
])

if __name__ == "__main__":
    # Use os.environ.get to configure the port and debug mode
    # This allows the app to run locally or in cloud environments (like Render/Heroku)
    # without changing the code.
    port = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("DEBUG", "True").lower() == "true"
    
    app.run(debug=False, host="0.0.0.0", port=port)