"""
phase7_dashboard/tabs/tab_fairness.py

Tab 4 — Fairness Analysis
Displays ECOA disparate impact testing results from Phase 5
across geography, business age, industry, and loan size dimensions.

Data sources:
    - outputs/fairness/fairness_summary.csv
    - outputs/fairness/equalized_odds_by_state.csv
    - outputs/fairness/equalized_odds_by_industry.csv
    - outputs/fairness/equalized_odds_by_loan_size.csv
"""

import os
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import html, dcc

# ── Data loading ──────────────────────────────────────────────────────────────
BASE_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FAIRNESS_DIR = os.path.join(BASE_DIR, "outputs", "fairness")


def load_fairness_summary() -> pd.DataFrame:
    return pd.read_csv(os.path.join(FAIRNESS_DIR, "fairness_summary.csv"))


def load_state() -> pd.DataFrame:
    return pd.read_csv(os.path.join(FAIRNESS_DIR, "equalized_odds_by_state.csv"))


def load_industry() -> pd.DataFrame:
    return pd.read_csv(os.path.join(FAIRNESS_DIR, "equalized_odds_by_industry.csv"))


def load_loan_size() -> pd.DataFrame:
    return pd.read_csv(os.path.join(FAIRNESS_DIR, "equalized_odds_by_loan_size.csv"))


# ── Chart builders ────────────────────────────────────────────────────────────
def build_summary_table(df: pd.DataFrame) -> dbc.Table:
    """Build fairness summary table with flag indicators."""
    df = df.fillna("N/A")
    header = html.Thead(html.Tr([
        html.Th(col) for col in df.columns
    ]))

    rows = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            style = {}
            if col == "Overall Flag":
                if str(val) == "False":
                    style = {"color": "#2E7D32", "fontWeight": "700"}
                elif str(val) == "True":
                    style = {"color": "#C62828", "fontWeight": "700"}
            cells.append(html.Td(str(val), style=style))
        rows.append(html.Tr(cells))

    return dbc.Table(
        [header, html.Tbody(rows)],
        bordered=True,
        hover=True,
        striped=True,
        responsive=True,
        size="sm",
    )


def build_tpr_chart(df: pd.DataFrame, title: str) -> go.Figure:
    """Build True Positive Rate comparison chart by group."""
    colors = ["#E74C3C" if flag else "#2ECC71"
              for flag in df["tpr_flag"]]

    fig = go.Figure(go.Bar(
        x=df["group"].astype(str),
        y=df["tpr"],
        marker_color=colors,
        text=[f"{v:.1%}" for v in df["tpr"]],
        textposition="outside",
    ))

    if "ref_tpr" in df.columns:
        ref_tpr = df["ref_tpr"].iloc[0]
        fig.add_hline(
            y=ref_tpr * 0.80,
            line_dash="dash",
            line_color="#E74C3C",
            annotation_text="4/5ths Threshold",
            annotation_position="top right",
        )

    fig.update_layout(
        title=f"{title} — True Positive Rate",
        yaxis_title="True Positive Rate (Recall)",
        yaxis=dict(range=[0, 1.15], fixedrange=True),
        xaxis=dict(fixedrange=True),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=480,
        margin=dict(t=60, b=80),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, tickangle=45)
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    return fig


def build_fpr_chart(df: pd.DataFrame, title: str) -> go.Figure:
    """Build False Positive Rate comparison chart by group."""
    colors = ["#E74C3C" if flag else "#2ECC71"
              for flag in df["fpr_flag"]]

    fig = go.Figure(go.Bar(
        x=df["group"].astype(str),
        y=df["fpr"],
        marker_color=colors,
        text=[f"{v:.1%}" for v in df["fpr"]],
        textposition="outside",
    ))

    fig.update_layout(
        title=f"{title} — False Positive Rate",
        yaxis_title="False Positive Rate",
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=480,
        margin=dict(t=60, b=80),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, tickangle=45)
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    return fig


# ── Layout ────────────────────────────────────────────────────────────────────
def layout():
    """Return the full Fairness Analysis tab layout."""
    try:
        summary_df  = load_fairness_summary()
        state_df    = load_state()
        industry_df = load_industry()
        loan_df     = load_loan_size()
    except FileNotFoundError as e:
        return html.Div([
            html.H3("Fairness Analysis"),
            dbc.Alert(
                f"Fairness data not found: {e}. "
                "Ensure Phase 5 completed successfully.",
                color="danger"
            ),
        ])

    passes = (summary_df["Overall Flag"] == False).sum()
    fails  = (summary_df["Overall Flag"] == True).sum()

    return html.Div([

        html.H3("Fairness Analysis", className="mb-2 mt-2"),
        html.P(
            "ECOA Regulation B disparate impact testing using equalized odds. "
            "Red bars indicate groups flagged for disparate impact relative "
            "to the reference group.",
            style={"color": "#666", "fontSize": "0.9rem", "marginBottom": "32px"},
        ),

        # ── Summary cards ─────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H2("4/5ths Rule", style={"color": "#2C3E50", "fontWeight": "700"}),
                html.P("ECOA Test Applied"),
            ], className="p-4"), style={"textAlign": "center", "borderTop": "4px solid #2C3E50"}), width=4),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H2(f"{passes}", style={"color": "#2ECC71", "fontWeight": "700"}),
                html.P("Dimensions Clean (No Flag)"),
            ], className="p-4"), style={"textAlign": "center", "borderTop": "4px solid #2ECC71"}), width=4),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H2(f"{fails}", style={"color": "#E74C3C", "fontWeight": "700"}),
                html.P("Dimensions Flagged for Review"),
            ], className="p-4"), style={"textAlign": "center", "borderTop": "4px solid #E74C3C"}), width=4),
        ], className="mb-5 g-3"),

        # ── Summary table ─────────────────────────────────────────────────
        html.H5("Fairness Test Summary", className="mt-4 mb-3"),
        build_summary_table(summary_df),

        html.Hr(className="my-5"),

        # ── State charts ──────────────────────────────────────────────────
        html.H5("Equalized Odds by Group", className="mt-2 mb-3"),
        html.P(
            "Red bars indicate groups flagged for disparate impact. "
            "TPR = True Positive Rate (default detection). "
            "FPR = False Positive Rate (wrongful denial rate).",
            style={"color": "#666", "fontSize": "0.85rem", "marginBottom": "24px"},
        ),

        dbc.Row([
            dbc.Col(dcc.Graph(figure=build_tpr_chart(state_df, "By State"), config={"displayModeBar": False}), width=6),
            dbc.Col(dcc.Graph(figure=build_fpr_chart(state_df, "By State"), config={"displayModeBar": False}), width=6),
        ], className="mb-5"),

        # ── Industry charts ───────────────────────────────────────────────
        dbc.Row([
            dbc.Col(dcc.Graph(figure=build_tpr_chart(industry_df, "By Industry"), config={"displayModeBar": False}), width=6),
            dbc.Col(dcc.Graph(figure=build_fpr_chart(industry_df, "By Industry"), config={"displayModeBar": False}), width=6),
        ], className="mb-5"),

        # ── Loan size chart + regulatory context ──────────────────────────
        dbc.Row([
            dbc.Col(dcc.Graph(figure=build_tpr_chart(loan_df, "By Loan Size"), config={"displayModeBar": False}), width=6),
            dbc.Col([
                html.H5("Regulatory Context", className="mb-3"),
                dbc.Alert(
                    [
                        html.Strong("ECOA Regulation B "),
                        "(12 CFR Part 1002) prohibits credit discrimination "
                        "based on race, color, religion, national origin, sex, "
                        "marital status, age, or receipt of public assistance. ",
                        html.Br(), html.Br(),
                        html.Strong("Equalized Odds: "),
                        "A fair model should have similar True Positive Rates "
                        "and False Positive Rates across all demographic groups. "
                        "Red bars indicate groups where the model performs "
                        "significantly differently from the reference group. ",
                        html.Br(), html.Br(),
                        html.Strong("SR 11-7: "),
                        "Federal Reserve guidance requires model risk governance "
                        "documentation including bias testing and limitations "
                        "disclosure.",
                    ],
                    color="info",
                    style={"fontSize": "0.85rem"},
                ),
            ], width=6),
        ], className="mb-5"),

    ])