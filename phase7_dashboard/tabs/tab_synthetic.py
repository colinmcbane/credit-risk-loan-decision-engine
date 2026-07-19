"""
phase7_dashboard/tabs/tab_synthetic.py

Tab 3 — Synthetic Test Analysis
Compares model performance across 4 AI-generated test datasets
(Gemini, ChatGPT, Claude, Copilot) with known risk tier profiles.

Data sources:
    - outputs/synthetic_scoring_results.csv  (Phase 6 synthetic scoring)
"""

import os
import pandas as pd
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
from dash import html, dcc

# ── Data loading ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_synthetic() -> pd.DataFrame:
    """Load synthetic scoring results."""
    path = os.path.join(BASE_DIR, "outputs", "synthetic_scoring_results.csv")
    return pd.read_csv(path)


# ── Chart builders ────────────────────────────────────────────────────────────
SOURCE_COLORS = {
    "Gemini":  "#4285F4",
    "ChatGPT": "#10A37F",
    "Claude":  "#CC785C",
    "Copilot": "#7B2FBE",
}


def build_decision_comparison(df: pd.DataFrame) -> go.Figure:
    """Build decision distribution comparison across AI sources."""
    fig = go.Figure()

    decision_colors = {
        "Approved":                    "#2ECC71",
        "Approved - Unfavorable Terms": "#F39C12",
        "Denied":                      "#E74C3C",
    }

    for decision in ["Approved", "Approved - Unfavorable Terms", "Denied"]:
        values = []
        for source in SOURCE_COLORS:
            subset = df[df["source"] == source]
            count  = (subset["decision"] == decision).sum()
            pct    = count / len(subset) * 100 if len(subset) > 0 else 0
            values.append(pct)

        fig.add_trace(go.Bar(
            name=decision,
            x=list(SOURCE_COLORS.keys()),
            y=values,
            marker_color=decision_colors.get(decision, "#888"),
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
        ))

    fig.update_layout(
        title="Decision Distribution by AI Source (%)",
        yaxis_title="Percentage of Applicants",
        yaxis=dict(range=[0, 110]),
        barmode="group",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=460,
        margin=dict(t=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    return fig


def build_mean_pd_comparison(df: pd.DataFrame) -> go.Figure:
    """Build mean predicted default probability by AI source."""
    sources   = list(SOURCE_COLORS.keys())
    mean_pds  = [df[df["source"] == s]["predicted_prob"].mean() for s in sources]

    fig = go.Figure(go.Bar(
        x=sources,
        y=mean_pds,
        marker_color=list(SOURCE_COLORS.values()),
        text=[f"{v:.1%}" for v in mean_pds],
        textposition="outside",
    ))

    fig.add_hline(
        y=0.50, line_dash="dash", line_color="#E74C3C",
        annotation_text="Denial (50%)",
        annotation_position="top left",
    )
    fig.add_hline(
        y=0.30, line_dash="dash", line_color="#F39C12",
        annotation_text="Unfavorable (30%)",
        annotation_position="top left",
    )

    fig.update_layout(
        title="Mean Predicted Default Probability by AI Source",
        yaxis_title="Mean Predicted Default Probability",
        yaxis=dict(range=[0, 1.1]),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=460,
        margin=dict(t=60, b=50),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    return fig


def build_prob_distribution(df: pd.DataFrame) -> go.Figure:
    """Build predicted probability distribution per AI source."""
    fig = go.Figure()

    for source, color in SOURCE_COLORS.items():
        subset = df[df["source"] == source]["predicted_prob"]
        fig.add_trace(go.Box(
            y=subset,
            name=source,
            marker_color=color,
            boxmean=True,
        ))

    fig.add_hline(y=0.50, line_dash="dash", line_color="#E74C3C",
                  annotation_text="Denial (50%)",
                  annotation_position="top left")
    fig.add_hline(y=0.30, line_dash="dash", line_color="#F39C12",
                  annotation_text="Unfavorable (30%)",
                  annotation_position="top left")

    fig.update_layout(
        title="Predicted Probability Distribution by AI Source",
        yaxis_title="Predicted Default Probability",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=460,
        margin=dict(t=60, b=50),
        showlegend=False,
    )
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    return fig


def build_risk_tier_table(df: pd.DataFrame) -> dbc.Table:
    """Build risk tier accuracy table across all AI sources."""
    tiers = [
        ("Low Risk",      0,  15, "Approved"),
        ("Moderate Risk", 15, 35, None),
        ("High Risk",     35, 50, "Denied"),
    ]

    header = html.Thead(html.Tr([
        html.Th("Risk Tier"),
        html.Th("Expected Decision"),
        html.Th("Gemini"),
        html.Th("ChatGPT"),
        html.Th("Claude"),
        html.Th("Copilot"),
    ]))

    rows = []
    for tier_name, start, end, expected in tiers:
        cells = [html.Td(tier_name), html.Td(expected or "Mixed")]
        for source in SOURCE_COLORS:
            subset = df[df["source"] == source].iloc[start:end]
            if expected and len(subset) > 0:
                correct = (subset["decision"] == expected).sum()
                pct     = correct / len(subset) * 100
                color   = "#2E7D32" if pct >= 70 else "#E65100"
                cells.append(html.Td(
                    f"{correct}/{len(subset)} ({pct:.0f}%)",
                    style={"color": color, "fontWeight": "600"}
                ))
            elif len(subset) > 0:
                mean_pd = subset["predicted_prob"].mean()
                cells.append(html.Td(f"Mean PD: {mean_pd:.2%}"))
            else:
                cells.append(html.Td("N/A"))
        rows.append(html.Tr(cells))

    return dbc.Table(
        [header, html.Tbody(rows)],
        bordered=True,
        hover=True,
        striped=True,
        responsive=True,
        size="sm",
    )


def build_source_summary_cards(df: pd.DataFrame) -> list:
    """Build summary cards for each AI source."""
    cards = []
    for source, color in SOURCE_COLORS.items():
        subset  = df[df["source"] == source]
        mean_pd = subset["predicted_prob"].mean()
        denied  = (subset["decision"] == "Denied").sum()
        total   = len(subset)

        cards.append(dbc.Col(
            dbc.Card(dbc.CardBody([
                html.H4(source, style={"color": color, "fontWeight": "700"}),
                html.Hr(style={"borderColor": color}),
                html.P(f"Applicants: {total}", style={"marginBottom": "4px"}),
                html.P(f"Mean PD: {mean_pd:.1%}", style={"marginBottom": "4px"}),
                html.P(f"Denied: {denied} ({denied/total*100:.1f}%)",
                       style={"marginBottom": "4px", "color": "#E74C3C",
                              "fontWeight": "600"}),
            ], className="p-4"), style={"borderTop": f"4px solid {color}"}),
            width=3,
        ))
    return cards


# ── Layout ────────────────────────────────────────────────────────────────────
def layout():
    """Return the full Synthetic Test Analysis tab layout."""
    try:
        df = load_synthetic()
    except FileNotFoundError:
        return html.Div([
            html.H3("Synthetic Test Analysis"),
            dbc.Alert("synthetic_scoring_results.csv not found. "
                      "Run score_synthetic_applicants.py first.", color="danger"),
        ])

    total = len(df)

    return html.Div([

        # ── Section header ────────────────────────────────────────────────
        html.H3("Synthetic Test Analysis", className="mb-2 mt-2"),
        html.P(
            f"Out-of-sample validation using {total} synthetic loan applicants "
            f"generated by 4 AI systems with known risk tier profiles "
            f"(15 low risk / 20 moderate risk / 15 high risk per source)",
            style={"color": "#666", "fontSize": "0.9rem", "marginBottom": "32px"},
        ),

        # ── Source summary cards ──────────────────────────────────────────
        dbc.Row(build_source_summary_cards(df), className="mb-5 g-3"),

        # ── Charts row 1 ─────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(dcc.Graph(figure=build_decision_comparison(df), config={"displayModeBar": False, "staticPlot": True}), width=6),
            dbc.Col(dcc.Graph(figure=build_mean_pd_comparison(df), config={"displayModeBar": False, "staticPlot": True}), width=6),
        ], className="mb-5"),

        # ── Charts row 2 ─────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(dcc.Graph(figure=build_prob_distribution(df), config={"displayModeBar": False, "staticPlot": True}), width=6),
            dbc.Col([
                html.H5("Risk Tier Accuracy", className="mt-2 mb-3"),
                html.P(
                    "Each AI source generated 50 applicants with known risk profiles. "
                    "This table shows how accurately the champion model classified "
                    "each tier.",
                    style={"color": "#666", "fontSize": "0.85rem",
                           "marginBottom": "15px"},
                ),
                build_risk_tier_table(df),
                dbc.Alert(
                    "Key finding: Significant variance in risk profile distribution "
                    "across AI sources given identical prompts — Gemini generated the "
                    "most extreme profiles (88% denial rate) while Claude generated "
                    "the most conservative (16% denial rate).",
                    color="info",
                    style={"fontSize": "0.8rem", "marginTop": "15px"},
                ),
            ], width=6),
        ], className="mb-5"),

    ])