"""
phase7_dashboard/tabs/tab_live.py

Tab 5 — Live Loan Application
Interactive form for real-time loan scoring using the Phase 4
LightGBM champion model. Denied or unfavorable applications
trigger Gemini-generated ECOA adverse action letters delivered
via Gmail SMTP.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
from dash import html, dcc, Input, Output, State, callback

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from utils.scorer import preprocess_and_scale
from utils.decision_classifier import classify_decision, extract_reason_codes
from utils.letter_generator import initialize_gemini, generate_letter
from utils.email_sender import send_all_emails

# ── Configuration ─────────────────────────────────────────────────────────────
BASE_DIR            = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CHAMPION_MODEL_PATH = os.path.join(BASE_DIR, "models", "champion_model.pkl")

FEATURE_COLUMNS = [
    "term_months", "interest_rate", "loan_amount", "sba_guarantee_pct",
    "business_age_mature", "loan_size_bucket_micro", "naics_sector_62",
    "loan_size_bucket_large", "borr_state_FL", "business_age_startup",
    "borr_state_TX", "jobs_supported", "business_age_established",
    "loan_size_bucket_small", "borr_state_CA", "loan_size_bucket_medium",
    "business_age_new", "naics_sector_48", "borr_state_WI", "borr_state_NJ",
    "borr_state_NY", "naics_sector_71", "naics_sector_52", "borr_state_WA",
    "naics_sector_45", "borr_state_MN",
]

DECISION_COLORS = {
    "Approved":                    "#2ECC71",
    "Approved - Unfavorable Terms": "#F39C12",
    "Denied":                      "#E74C3C",
}

# Load model once at module level
try:
    CHAMPION_MODEL = joblib.load(CHAMPION_MODEL_PATH)
except Exception:
    CHAMPION_MODEL = None


def build_input_row(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert form inputs into the 26-column feature DataFrame
    the champion model expects.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Single-row DataFrame with raw form values.

    Returns
    -------
    pd.DataFrame
        Scaled 26-column feature DataFrame ready for scoring.
    """
    return preprocess_and_scale(raw_df)


# ── Layout ────────────────────────────────────────────────────────────────────
def layout():
    """Return the full Live Loan Application tab layout."""
    return html.Div([

        html.H3("Live Loan Application", className="mb-1"),
        html.P(
            "Submit a loan application for real-time scoring by the LightGBM "
            "champion model. Denied or unfavorable applications will receive "
            "an ECOA-compliant adverse action letter via email.",
            style={"color": "#666", "fontSize": "0.9rem", "marginBottom": "25px"},
        ),

        dbc.Row([

            # ── Application form ──────────────────────────────────────────
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Loan Application Form",
                                           className="mb-0")),
                    dbc.CardBody([

                        # Email
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Applicant Email Address"),
                                dbc.Input(
                                    id="input-email",
                                    type="email",
                                    placeholder="applicant@example.com",
                                ),
                                dbc.FormText(
                                    "Adverse action letter will be sent here "
                                    "if application is denied or flagged."
                                ),
                            ]),
                        ], className="mb-3"),

                        # Loan amount and term
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Loan Amount ($)"),
                                dbc.Input(
                                    id="input-loan-amount",
                                    type="number",
                                    placeholder="e.g. 250000",
                                    min=5000,
                                    max=5000000,
                                ),
                            ], width=6),
                            dbc.Col([
                                dbc.Label("Loan Term (Months)"),
                                dbc.Input(
                                    id="input-term-months",
                                    type="number",
                                    placeholder="e.g. 120",
                                    min=12,
                                    max=300,
                                ),
                            ], width=6),
                        ], className="mb-3"),

                        # Interest rate and SBA guarantee
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Interest Rate (%)"),
                                dbc.Input(
                                    id="input-interest-rate",
                                    type="number",
                                    placeholder="e.g. 6.5",
                                    min=1.0,
                                    max=15.0,
                                    step=0.1,
                                ),
                                dbc.FormText("Enter as percentage, e.g. 6.5 for 6.5%"),
                            ], width=6),
                            dbc.Col([
                                dbc.Label("SBA Guarantee (%)"),
                                dbc.Input(
                                    id="input-sba-guarantee",
                                    type="number",
                                    placeholder="e.g. 75",
                                    min=50,
                                    max=85,
                                    step=1,
                                ),
                                dbc.FormText("Typical range: 50% – 85%"),
                            ], width=6),
                        ], className="mb-3"),

                        # Jobs supported
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Jobs Supported"),
                                dbc.Input(
                                    id="input-jobs",
                                    type="number",
                                    placeholder="e.g. 10",
                                    min=1,
                                    max=500,
                                ),
                                dbc.FormText("Total jobs created + retained"),
                            ], width=6),
                        ], className="mb-3"),

                        # Business age
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Business Age"),
                                dbc.Select(
                                    id="input-business-age",
                                    options=[
                                        {"label": "Startup (Opening with loan funds)",
                                         "value": "startup"},
                                        {"label": "New (2 years or less)",
                                         "value": "new"},
                                        {"label": "Established (2–10 years)",
                                         "value": "established"},
                                        {"label": "Mature (10+ years)",
                                         "value": "mature"},
                                    ],
                                    placeholder="Select business age...",
                                ),
                            ]),
                        ], className="mb-3"),

                        # Industry
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Industry Sector"),
                                dbc.Select(
                                    id="input-industry",
                                    options=[
                                        {"label": "Healthcare (NAICS 62)",
                                         "value": "62"},
                                        {"label": "Transportation (NAICS 48)",
                                         "value": "48"},
                                        {"label": "Finance & Insurance (NAICS 52)",
                                         "value": "52"},
                                        {"label": "Arts & Entertainment (NAICS 71)",
                                         "value": "71"},
                                        {"label": "Retail Trade (NAICS 45)",
                                         "value": "45"},
                                        {"label": "Other", "value": "other"},
                                    ],
                                    placeholder="Select industry...",
                                ),
                            ]),
                        ], className="mb-3"),

                        # State
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Borrower State"),
                                dbc.Select(
                                    id="input-state",
                                    options=[
                                        {"label": "California", "value": "CA"},
                                        {"label": "Florida",    "value": "FL"},
                                        {"label": "Minnesota",  "value": "MN"},
                                        {"label": "New Jersey", "value": "NJ"},
                                        {"label": "New York",   "value": "NY"},
                                        {"label": "Texas",      "value": "TX"},
                                        {"label": "Washington", "value": "WA"},
                                        {"label": "Wisconsin",  "value": "WI"},
                                        {"label": "Other State", "value": "other"},
                                    ],
                                    placeholder="Select state...",
                                ),
                            ]),
                        ], className="mb-3"),

                        # Submit button
                        dbc.Button(
                            "Submit Application",
                            id="submit-button",
                            color="primary",
                            size="lg",
                            className="w-100 mt-2",
                        ),

                    ]),
                ]),
            ], width=5),

            # ── Decision output ───────────────────────────────────────────
            dbc.Col([
                html.Div(id="decision-output"),
            ], width=7),

        ]),

    ])


# ── Callback ──────────────────────────────────────────────────────────────────
@callback(
    Output("decision-output", "children"),
    Input("submit-button", "n_clicks"),
    State("input-email", "value"),
    State("input-loan-amount", "value"),
    State("input-term-months", "value"),
    State("input-interest-rate", "value"),
    State("input-sba-guarantee", "value"),
    State("input-jobs", "value"),
    State("input-business-age", "value"),
    State("input-industry", "value"),
    State("input-state", "value"),
    prevent_initial_call=True,
)
def score_application(n_clicks, email, loan_amount, term_months,
                      interest_rate, sba_guarantee, jobs,
                      business_age, industry, state):
    """Score a submitted loan application and return the decision."""

    # ── Validate inputs ───────────────────────────────────────────────────
    missing = []
    if not email:           missing.append("Email Address")
    if not loan_amount:     missing.append("Loan Amount")
    if not term_months:     missing.append("Loan Term")
    if not interest_rate:   missing.append("Interest Rate")
    if not sba_guarantee:   missing.append("SBA Guarantee")
    if not jobs:            missing.append("Jobs Supported")
    if not business_age:    missing.append("Business Age")
    if not industry:        missing.append("Industry Sector")
    if not state:           missing.append("Borrower State")

    if missing:
        return dbc.Alert(
            f"Please fill in all required fields: {', '.join(missing)}",
            color="warning",
        )

    if CHAMPION_MODEL is None:
        return dbc.Alert("Champion model not loaded. Check models/ directory.",
                         color="danger")

    # ── Build feature row ─────────────────────────────────────────────────
    row = {f: 0 for f in FEATURE_COLUMNS}

    # Continuous features — interest rate stored as percentage
    row["loan_amount"]       = float(loan_amount)
    row["term_months"]       = float(term_months)
    row["interest_rate"]     = float(interest_rate)   # already in % e.g. 6.5
    row["sba_guarantee_pct"] = float(sba_guarantee) / 100.0
    row["jobs_supported"]    = float(jobs)

    # Business age one-hot
    age_map = {
        "startup":     "business_age_startup",
        "new":         "business_age_new",
        "established": "business_age_established",
        "mature":      "business_age_mature",
    }
    if business_age in age_map:
        row[age_map[business_age]] = 1

    # Industry one-hot
    industry_map = {
        "62": "naics_sector_62",
        "48": "naics_sector_48",
        "52": "naics_sector_52",
        "71": "naics_sector_71",
        "45": "naics_sector_45",
    }
    if industry in industry_map:
        row[industry_map[industry]] = 1

    # State one-hot
    state_map = {
        "CA": "borr_state_CA", "FL": "borr_state_FL",
        "MN": "borr_state_MN", "NJ": "borr_state_NJ",
        "NY": "borr_state_NY", "TX": "borr_state_TX",
        "WA": "borr_state_WA", "WI": "borr_state_WI",
    }
    if state in state_map:
        row[state_map[state]] = 1

    # Loan size bucket
    amount = float(loan_amount)
    if amount < 50000:
        row["loan_size_bucket_micro"] = 1
    elif amount < 350000:
        row["loan_size_bucket_small"] = 1
    elif amount < 2000000:
        row["loan_size_bucket_medium"] = 1
    else:
        row["loan_size_bucket_large"] = 1

    # ── Scale and score ───────────────────────────────────────────────────
    raw_df    = pd.DataFrame([row])
    scaled_df = preprocess_and_scale(raw_df)
    X         = scaled_df[FEATURE_COLUMNS]

    predicted_prob = float(CHAMPION_MODEL.predict_proba(X)[0, 1])
    decision       = classify_decision(predicted_prob)
    reason_codes   = extract_reason_codes(
        pd.Series(scaled_df.iloc[0])
    ) if decision != "Approved" else []

    color = DECISION_COLORS.get(decision, "#888")

    # ── Send email if adverse action required ─────────────────────────────
    email_status = ""
    if decision in ("Denied", "Approved - Unfavorable Terms"):
        try:
            from dotenv import load_dotenv
            load_dotenv()
            gemini_client = initialize_gemini()
            letter_text   = generate_letter(
                applicant_id=email,
                decision=decision,
                predicted_prob=predicted_prob,
                reason_codes=reason_codes,
                client=gemini_client,
            )
            letter_results = {
                email: {
                    "letter_path": None,
                    "letter_text": letter_text,
                    "decision":    decision,
                    "status":      "generated",
                }
            }
            # Write letter to temp file for email sender
            import tempfile
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False
            ) as tmp:
                tmp.write(letter_text)
                tmp_path = tmp.name

            letter_results[email]["letter_path"] = tmp_path
            send_all_emails(
                letter_results,
                recipient_address=email,
                dry_run=False,
            )
            email_status = f"✓ Adverse action letter sent to {email}"
        except Exception as e:
            email_status = f"⚠ Email not sent: {str(e)[:80]}"

    # ── Build result card ─────────────────────────────────────────────────
    result_children = [
        dbc.Card([
            dbc.CardHeader(
                html.H4("Application Decision",
                        style={"color": "white"},
                        className="mb-0"),
                style={"backgroundColor": color},
            ),
            dbc.CardBody([

                html.H2(
                    decision,
                    style={"color": color, "fontWeight": "700",
                           "fontSize": "2rem", "marginBottom": "10px"},
                ),

                html.H4(
                    f"Predicted Default Probability: {predicted_prob:.1%}",
                    style={"color": "#444", "marginBottom": "20px"},
                ),

                html.Hr(),

                # Risk gauge
                html.H6("Risk Score", className="mb-2"),
                dbc.Progress(
                    value=predicted_prob * 100,
                    color="danger" if predicted_prob >= 0.50
                    else "warning" if predicted_prob >= 0.30
                    else "success",
                    style={"height": "25px", "marginBottom": "20px"},
                    label=f"{predicted_prob:.1%}",
                ),

                # Reason codes
                html.Div([
                    html.H6("Adverse Action Reason Codes", className="mb-2"),
                    html.Ul([
                        html.Li(reason, style={"marginBottom": "5px"})
                        for reason in reason_codes
                    ]),
                ]) if reason_codes else html.Div(),

                html.Hr() if reason_codes else html.Div(),

                # Decision thresholds reference
                dbc.Row([
                    dbc.Col([
                        html.Small("Decision Thresholds:", className="text-muted"),
                        html.Br(),
                        html.Small("< 30% → Approved", style={"color": "#2ECC71"}),
                        html.Br(),
                        html.Small("30–50% → Unfavorable Terms",
                                   style={"color": "#F39C12"}),
                        html.Br(),
                        html.Small("≥ 50% → Denied", style={"color": "#E74C3C"}),
                    ], width=6),
                    dbc.Col([
                        html.Small("Application Summary:", className="text-muted"),
                        html.Br(),
                        html.Small(f"Loan: ${float(loan_amount):,.0f} "
                                   f"@ {float(interest_rate):.1f}%"),
                        html.Br(),
                        html.Small(f"Term: {term_months} months | "
                                   f"SBA: {sba_guarantee}%"),
                        html.Br(),
                        html.Small(f"Business: {business_age} | "
                                   f"State: {state}"),
                    ], width=6),
                ]),

                # Email status
                html.Div(
                    dbc.Alert(email_status, color="success"
                              if email_status.startswith("✓") else "warning",
                              className="mt-3"),
                ) if email_status else html.Div(),

            ]),
        ]),
    ]

    return html.Div(result_children)