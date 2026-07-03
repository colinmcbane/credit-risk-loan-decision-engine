"""
phase7_dashboard/app.py

Phase 7 — Interactive Credit Risk Dashboard
Credit Risk & Loan Decision Engine
Author: Colin McBane

A five-tab Dash dashboard providing:
    Tab 1 — Model Performance (AUC, KS, Gini, ROC curve, confusion matrix)
    Tab 2 — Portfolio Overview (5,000 scored loans from Phase 4)
    Tab 3 — Synthetic Test Analysis (4 AI source comparison)
    Tab 4 — Fairness Analysis (ECOA disparate impact results)
    Tab 5 — Live Loan Application (real-time scoring + email delivery)

Usage:
    python3 phase7_dashboard/app.py

Then open: http://127.0.0.1:8050
"""

import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output

from phase7_dashboard.tabs import (
    tab_model,
    tab_portfolio,
    tab_synthetic,
    tab_fairness,
    tab_live,
    tab_about,
)

# ── Initialize app ────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
    suppress_callback_exceptions=True,
    title="Credit Risk Decision Engine",
)

server = app.server  # Required for Render deployment

# ── Layout ────────────────────────────────────────────────────────────────────
app.layout = dbc.Container(
    fluid=True,
    children=[

        # ── Header ───────────────────────────────────────────────────────────
        dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        html.H1(
                            "Credit Risk & Loan Decision Engine",
                            className="text-white mb-1",
                            style={"fontWeight": "700", "fontSize": "1.8rem"},
                        ),
                        html.P(
                            "SBA 7(a) LightGBM Champion Model | "
                            "ECOA Compliant | SR 11-7 Documented",
                            className="text-white-50 mb-0",
                            style={"fontSize": "0.9rem"},
                        ),
                    ],
                    style={"padding": "20px 0"},
                ),
                width=12,
            ),
            style={"backgroundColor": "#2C3E50", "marginBottom": "0px"},
        ),

        # ── Tabs ─────────────────────────────────────────────────────────────
        dbc.Row(
            dbc.Col(
                dbc.Tabs(
                    id="main-tabs",
                    active_tab="tab-model",
                    children=[
                        dbc.Tab(label="Model Performance",   tab_id="tab-model"),
                        dbc.Tab(label="Portfolio Overview",  tab_id="tab-portfolio"),
                        dbc.Tab(label="Synthetic Analysis",  tab_id="tab-synthetic"),
                        dbc.Tab(label="Fairness Analysis",   tab_id="tab-fairness"),
                        dbc.Tab(label="Live Application",    tab_id="tab-live"),
                        dbc.Tab(label="About & Sources",     tab_id="tab-about"),
                    ],
                    style={"marginTop": "10px"},
                ),
                width=12,
            ),
        ),

        # ── Tab content ───────────────────────────────────────────────────────
        dbc.Row(
            dbc.Col(
                html.Div(id="tab-content", style={"padding": "20px 0"}),
                width=12,
            ),
        ),
    ],
)


# ── Tab routing callback ──────────────────────────────────────────────────────
@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "active_tab"),
)
def render_tab(active_tab):
    """Route to the correct tab layout based on active tab selection."""
    if active_tab == "tab-model":
        return tab_model.layout()
    elif active_tab == "tab-portfolio":
        return tab_portfolio.layout()
    elif active_tab == "tab-synthetic":
        return tab_synthetic.layout()
    elif active_tab == "tab-fairness":
        return tab_fairness.layout()
    elif active_tab == "tab-live":
        return tab_live.layout()
    elif active_tab == "tab-about":
        return tab_about.layout()
    return html.P("Tab not found.")


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=8050)