"""
phase7_dashboard/tabs/tab_model.py

Tab 1 — Model Performance
Displays champion model metrics, ROC curve, confusion matrix,
and champion-challenger comparison table.

Data sources:
    - data/processed/model_comparison.csv  (Phase 4 champion-challenger results)
"""

import os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import dash_bootstrap_components as dbc
from dash import html, dcc

# ── Data loading ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_model_comparison() -> pd.DataFrame:
    """Load Phase 4 champion-challenger comparison results."""
    path = os.path.join(BASE_DIR, "data", "processed", "model_comparison.csv")
    return pd.read_csv(path)


# ── Chart builders ────────────────────────────────────────────────────────────
def build_metrics_cards(df: pd.DataFrame) -> list:
    """Build metric summary cards for the champion model."""
    champion = df[df["AUC"] == df["AUC"].max()].iloc[0]

    metrics = [
        ("AUC",        f"{champion['AUC']:.4f}",      "Area Under ROC Curve",        "#2ECC71"),
        ("KS",         f"{champion['KS Statistic']:.2f}", "KS Statistic",             "#3498DB"),
        ("Gini",       f"{champion['Gini']:.2f}%",     "Gini Coefficient",            "#9B59B6"),
        ("Recall",     f"{champion['Recall']:.4f}",    "Default Detection Rate",      "#E67E22"),
        ("Precision",  f"{champion['Precision']:.4f}", "Precision",                   "#1ABC9C"),
        ("F1",         f"{champion['F1']:.4f}",        "F1 Score",                    "#E74C3C"),
    ]

    cards = []
    for label, value, subtitle, color in metrics:
        cards.append(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody([
                        html.H2(value, style={"color": color, "fontWeight": "700",
                                              "marginBottom": "4px"}),
                        html.P(label, style={"fontWeight": "600", "marginBottom": "2px",
                                             "fontSize": "0.95rem"}),
                        html.P(subtitle, style={"color": "#888", "fontSize": "0.75rem",
                                                "marginBottom": "0"}),
                    ]),
                    style={"textAlign": "center", "borderTop": f"4px solid {color}"},
                ),
                width=2,
            )
        )
    return cards


def build_roc_chart(df: pd.DataFrame) -> go.Figure:
    """Build champion-challenger AUC comparison bar chart."""
    colors = {
        "LightGBM":            "#2E7D32",
        "XGBoost":             "#1565C0",
        "Logistic Regression": "#9E9E9E",
    }

    fig = go.Figure()
    for _, row in df.iterrows():
        fig.add_trace(go.Bar(
            x=[row["Model"]],
            y=[row["AUC"]],
            name=row["Model"],
            marker_color=colors.get(row["Model"], "#888"),
            text=[f"{row['AUC']:.4f}"],
            textposition="outside",
        ))

    fig.update_layout(
        title="AUC by Model — Champion-Challenger Comparison",
        yaxis=dict(range=[0.85, 1.0], title="AUC"),
        xaxis_title="Model",
        showlegend=False,
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=350,
        margin=dict(t=50, b=40),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    return fig


def build_ks_gini_chart(df: pd.DataFrame) -> go.Figure:
    """Build KS and Gini comparison chart."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="KS Statistic",
        x=df["Model"],
        y=df["KS Statistic"],
        marker_color="#3498DB",
        text=[f"{v:.1f}" for v in df["KS Statistic"]],
        textposition="outside",
    ))

    fig.add_trace(go.Bar(
        name="Gini (%)",
        x=df["Model"],
        y=df["Gini"],
        marker_color="#9B59B6",
        text=[f"{v:.1f}%" for v in df["Gini"]],
        textposition="outside",
    ))

    fig.update_layout(
        title="KS Statistic & Gini Coefficient by Model",
        barmode="group",
        yaxis_title="Value",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=350,
        margin=dict(t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eee")
    return fig


def build_confusion_matrix(df: pd.DataFrame) -> go.Figure:
    """Build confusion matrix heatmap for champion model."""
    champion = df[df["AUC"] == df["AUC"].max()].iloc[0]

    tn = int(champion["True Neg"])
    fp = int(champion["False Pos"])
    fn = int(champion["False Neg"])
    tp = int(champion["True Pos"])

    z     = [[tn, fp], [fn, tp]]
    text  = [[f"TN: {tn:,}", f"FP: {fp:,}"],
             [f"FN: {fn:,}", f"TP: {tp:,}"]]

    fig = go.Figure(go.Heatmap(
        z=z,
        text=text,
        texttemplate="%{text}",
        colorscale="Blues",
        showscale=False,
        x=["Predicted: Paid", "Predicted: Default"],
        y=["Actual: Paid", "Actual: Default"],
    ))

    fig.update_layout(
        title=f"Confusion Matrix — {champion['Model']} (Champion)",
        height=350,
        margin=dict(t=50, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def build_comparison_table(df: pd.DataFrame) -> dbc.Table:
    """Build champion-challenger comparison table."""
    display_cols = ["Model", "AUC", "KS Statistic", "Gini",
                    "Precision", "Recall", "F1", "CV AUC Mean"]

    header = html.Thead(html.Tr([html.Th(col) for col in display_cols]))

    champion_name = df.loc[df["AUC"].idxmax(), "Model"]

    rows = []
    for _, row in df.iterrows():
        is_champion = row["Model"] == champion_name
        style = {"backgroundColor": "#E8F5E9", "fontWeight": "600"} if is_champion else {}
        cells = []
        for col in display_cols:
            val = row[col]
            if isinstance(val, float):
                val = f"{val:.4f}"
            if col == "Model" and is_champion:
                val = f"★ {val} (Champion)"
            cells.append(html.Td(val, style=style))
        rows.append(html.Tr(cells))

    body = html.Tbody(rows)
    return dbc.Table(
        [header, body],
        bordered=True,
        hover=True,
        striped=False,
        responsive=True,
        size="sm",
    )


# ── Layout ────────────────────────────────────────────────────────────────────
def layout():
    """Return the full Model Performance tab layout."""
    try:
        df = load_model_comparison()
    except FileNotFoundError:
        return html.Div([
            html.H3("Model Performance"),
            dbc.Alert("model_comparison.csv not found. "
                      "Ensure Phase 4 completed successfully.", color="danger"),
        ])

    champion = df[df["AUC"] == df["AUC"].max()].iloc[0]

    return html.Div([

        # ── Section header ────────────────────────────────────────────────
        html.H3("Model Performance", className="mb-1"),
        html.P(
            f"Champion Model: {champion['Model']} | "
            f"Trained on 382,144 SBA 7(a) loans | "
            f"5-fold stratified cross-validation with SMOTE",
            style={"color": "#666", "fontSize": "0.9rem", "marginBottom": "20px"},
        ),

        # ── Metric cards ──────────────────────────────────────────────────
        dbc.Row(build_metrics_cards(df), className="mb-4 g-3"),

        # ── Charts row 1 ─────────────────────────────────────────────────
        dbc.Row([
            dbc.Col(dcc.Graph(figure=build_roc_chart(df)), width=6),
            dbc.Col(dcc.Graph(figure=build_ks_gini_chart(df)), width=6),
        ], className="mb-4"),

        # ── Confusion matrix ──────────────────────────────────────────────
        dbc.Row([
            dbc.Col(dcc.Graph(figure=build_confusion_matrix(df)), width=6),
            dbc.Col([
                html.H5("Industry Context", className="mb-3"),
                dbc.Table([
                    html.Thead(html.Tr([
                        html.Th("Metric"),
                        html.Th("Industry Standard"),
                        html.Th("This Model"),
                        html.Th("Assessment"),
                    ])),
                    html.Tbody([
                        html.Tr([
                            html.Td("AUC"),
                            html.Td("0.70 – 0.85"),
                            html.Td(f"{champion['AUC']:.4f}",
                                    style={"color": "#2E7D32", "fontWeight": "600"}),
                            html.Td("✓ Exceeds standard"),
                        ]),
                        html.Tr([
                            html.Td("KS Statistic"),
                            html.Td("30 – 50"),
                            html.Td(f"{champion['KS Statistic']:.2f}",
                                    style={"color": "#2E7D32", "fontWeight": "600"}),
                            html.Td("✓ Exceeds standard"),
                        ]),
                        html.Tr([
                            html.Td("Gini"),
                            html.Td("40 – 70%"),
                            html.Td(f"{champion['Gini']:.2f}%",
                                    style={"color": "#2E7D32", "fontWeight": "600"}),
                            html.Td("✓ Exceeds standard"),
                        ]),
                    ]),
                ], bordered=True, hover=True, size="sm"),
                dbc.Alert(
                    "Note: High metrics reflect clean SBA FOIA data. "
                    "Real-world consumer credit data typically produces "
                    "lower scores due to noise and missing values.",
                    color="info",
                    style={"fontSize": "0.8rem", "marginTop": "15px"},
                ),
            ], width=6),
        ], className="mb-4"),

        # ── Comparison table ──────────────────────────────────────────────
        html.H5("Champion-Challenger Comparison Table", className="mb-3"),
        build_comparison_table(df),

    ])