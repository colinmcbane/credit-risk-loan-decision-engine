"""
phase7_dashboard/tabs/tab_portfolio.py

Tab 2 — Portfolio Overview
Displays decision distribution, predicted probability histogram,
and top SHAP features for the 5,000 scored test loans from Phase 4.

Data sources:
    - outputs/phase6_decisions.csv  (Phase 6 decision output)
    - data/processed/shap_values_sample.csv  (Phase 4 SHAP values)
"""

import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import html, dcc

# ── Data loading ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_decisions() -> pd.DataFrame:
    """Load Phase 6 decision output."""
    path = os.path.join(BASE_DIR, "outputs", "phase6_decisions.csv")
    return pd.read_csv(path)


def load_shap() -> pd.DataFrame:
    """Load Phase 4 SHAP values sample."""
    path = os.path.join(BASE_DIR, "data", "processed", "shap_values_sample.csv")
    return pd.read_csv(path)


# ── Chart builders ────────────────────────────────────────────────────────────
def build_decision_pie(df: pd.DataFrame) -> go.Figure:
    """Build decision distribution pie chart."""
    counts = df["decision"].value_counts()

    colors = {
        "Approved":                    "#2ECC71",
        "Denied":                      "#E74C3C",
        "Approved - Unfavorable Terms": "#F39C12",
    }

    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        marker_colors=[colors.get(l, "#888") for l in counts.index],
        hole=0.4,
        textinfo="label+percent",
        textfont_size=13,
    ))

    fig.update_layout(
        title="Decision Distribution (5,000 Test Loans)",
        height=460,
        margin=dict(t=60, b=30, l=20, r=20),
        paper_bgcolor="white",
        showlegend=False,
    )
    return fig


def build_prob_histogram(df: pd.DataFrame) -> go.Figure:
    """Build predicted probability distribution histogram."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=df["predicted_prob"],
        nbinsx=50,
        marker_color="#3498DB",
        opacity=0.8,
        name="All Loans",
    ))

    # Add threshold lines
    fig.add_vline(x=0.30, line_dash="dash", line_color="#F39C12",
                  annotation_text="Unfavorable Threshold (0.30)",
                  annotation_position="top right")
    fig.add_vline(x=0.50, line_dash="dash", line_color="#E74C3C",
                  annotation_text="Denial Threshold (0.50)",
                  annotation_position="top right")

    fig.update_layout(
        title="Predicted Default Probability Distribution",
        xaxis_title="Predicted Default Probability",
        yaxis_title="Number of Loans",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=460,
        margin=dict(t=60, b=50),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    return fig


def build_shap_importance(shap_df: pd.DataFrame) -> go.Figure:
    """Build mean absolute SHAP value feature importance chart."""
    feature_cols = [c for c in shap_df.columns
                    if c not in ["predicted_prob", "actual_default"]]

    mean_shap = shap_df[feature_cols].abs().mean().sort_values(ascending=True)

    # Clean up feature names for display
    label_map = {
        "term_months":              "Term Length (Months)",
        "interest_rate":            "Interest Rate",
        "loan_amount":              "Loan Amount",
        "sba_guarantee_pct":        "SBA Guarantee %",
        "business_age_mature":      "Business Age: Mature",
        "business_age_startup":     "Business Age: Startup",
        "business_age_new":         "Business Age: New",
        "business_age_established": "Business Age: Established",
        "loan_size_bucket_micro":   "Loan Size: Micro",
        "loan_size_bucket_small":   "Loan Size: Small",
        "loan_size_bucket_medium":  "Loan Size: Medium",
        "loan_size_bucket_large":   "Loan Size: Large",
        "jobs_supported":           "Jobs Supported",
        "naics_sector_45":          "Industry: Retail",
        "naics_sector_48":          "Industry: Transportation",
        "naics_sector_52":          "Industry: Finance",
        "naics_sector_62":          "Industry: Healthcare",
        "naics_sector_71":          "Industry: Arts & Entertainment",
        "borr_state_CA":            "State: California",
        "borr_state_FL":            "State: Florida",
        "borr_state_MN":            "State: Minnesota",
        "borr_state_NJ":            "State: New Jersey",
        "borr_state_NY":            "State: New York",
        "borr_state_TX":            "State: Texas",
        "borr_state_WA":            "State: Washington",
        "borr_state_WI":            "State: Wisconsin",
    }

    labels = [label_map.get(f, f) for f in mean_shap.index]

    fig = go.Figure(go.Bar(
        x=mean_shap.values,
        y=labels,
        orientation="h",
        marker_color="#2C3E50",
        opacity=0.85,
    ))

    fig.update_layout(
        title="Top SHAP Feature Importance (Mean |SHAP Value|)",
        xaxis_title="Mean Absolute SHAP Value",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=630,
        margin=dict(t=60, b=50, l=210),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eee")
    fig.update_yaxes(showgrid=False)
    return fig


def build_decision_by_prob_band(df: pd.DataFrame) -> go.Figure:
    """Build decision breakdown by probability band."""
    bins   = [0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]
    labels = ["0-10%", "10-20%", "20-30%", "30-40%", "40-50%",
              "50-60%", "60-70%", "70-80%", "80-90%", "90-100%"]

    df = df.copy()
    df["prob_band"] = pd.cut(df["predicted_prob"], bins=bins, labels=labels)
    band_counts = df.groupby(["prob_band", "decision"]).size().unstack(fill_value=0)

    fig = go.Figure()
    colors = {
        "Approved":                    "#2ECC71",
        "Approved - Unfavorable Terms": "#F39C12",
        "Denied":                      "#E74C3C",
    }

    for decision in ["Approved", "Approved - Unfavorable Terms", "Denied"]:
        if decision in band_counts.columns:
            fig.add_trace(go.Bar(
                name=decision,
                x=band_counts.index.astype(str),
                y=band_counts[decision],
                marker_color=colors.get(decision, "#888"),
            ))

    fig.update_layout(
        title="Loan Count by Probability Band and Decision",
        xaxis_title="Predicted Default Probability Band",
        yaxis_title="Number of Loans",
        barmode="stack",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=460,
        margin=dict(t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    return fig


# ── Layout ────────────────────────────────────────────────────────────────────
def layout():
    """Return the full Portfolio Overview tab layout."""
    try:
        decisions_df = load_decisions()
        shap_df      = load_shap()
    except FileNotFoundError as e:
        return html.Div([
            html.H3("Portfolio Overview"),
            dbc.Alert(f"Data file not found: {e}", color="danger"),
        ])

    total    = len(decisions_df)
    approved = (decisions_df["decision"] == "Approved").sum()
    denied   = (decisions_df["decision"] == "Denied").sum()
    unfav    = (decisions_df["decision"] == "Approved - Unfavorable Terms").sum()
    mean_pd  = decisions_df["predicted_prob"].mean()

    return html.Div([

        # ── Section header ────────────────────────────────────────────────
        html.H3("Portfolio Overview", className="mb-2 mt-2"),
        html.P(
            f"5,000 out-of-sample test loans scored by LightGBM champion model | "
            f"Mean predicted default probability: {mean_pd:.2%}",
            style={"color": "#666", "fontSize": "0.9rem", "marginBottom": "32px"},
        ),

        # ── Summary cards ─────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H2(f"{total:,}", style={"color": "#2C3E50", "fontWeight": "700"}),
                html.P("Total Loans Scored"),
            ], className="p-4"), style={"textAlign": "center", "borderTop": "4px solid #2C3E50"}), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H2(f"{approved:,}", style={"color": "#2ECC71", "fontWeight": "700"}),
                html.P(f"Approved ({approved/total*100:.1f}%)"),
            ], className="p-4"), style={"textAlign": "center", "borderTop": "4px solid #2ECC71"}), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H2(f"{denied:,}", style={"color": "#E74C3C", "fontWeight": "700"}),
                html.P(f"Denied ({denied/total*100:.1f}%)"),
            ], className="p-4"), style={"textAlign": "center", "borderTop": "4px solid #E74C3C"}), width=3),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H2(f"{unfav:,}", style={"color": "#F39C12", "fontWeight": "700"}),
                html.P(f"Unfavorable Terms ({unfav/total*100:.1f}%)"),
            ], className="p-4"), style={"textAlign": "center", "borderTop": "4px solid #F39C12"}), width=3),
        ], className="mb-5 g-3"),

        # ── Charts row 1 ─────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(dcc.Graph(figure=build_decision_pie(decisions_df), config={"displayModeBar": False, "staticPlot": True}), width=5),
            dbc.Col(dcc.Graph(figure=build_prob_histogram(decisions_df), config={"displayModeBar": False, "staticPlot": True}), width=7),
        ], className="mb-5"),

        # ── Charts row 2 ─────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(dcc.Graph(figure=build_decision_by_prob_band(decisions_df), config={"displayModeBar": False, "staticPlot": True}), width=6),
            dbc.Col(dcc.Graph(figure=build_shap_importance(shap_df), config={"displayModeBar": False, "staticPlot": True}), width=6),
        ], className="mb-5"),

    ])