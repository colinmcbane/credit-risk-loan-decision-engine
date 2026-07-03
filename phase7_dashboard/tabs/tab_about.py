"""
phase7_dashboard/tabs/tab_about.py

Tab 6 — About & Sources
Project overview, methodology, regulatory references,
data sources, and technical citations.
"""

import dash_bootstrap_components as dbc
from dash import html


def layout():
    """Return the full About & Sources tab layout."""
    return html.Div([

        html.H3("About & Sources", className="mb-2 mt-2"),
        html.P(
            "Full documentation of the Credit Risk & Loan Decision Engine — "
            "methodology, regulatory compliance, data sources, and citations.",
            style={"color": "#666", "fontSize": "0.9rem", "marginBottom": "32px"},
        ),

        # ── Project Overview ──────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Project Overview", className="mb-0")),
                    dbc.CardBody([
                        html.P(
                            "The Credit Risk & Loan Decision Engine is an end-to-end "
                            "machine learning pipeline built on 382,144 historical SBA "
                            "7(a) loan records. It trains a LightGBM champion model, "
                            "performs ECOA-compliant fairness testing, and deploys an "
                            "AI-powered adverse action letter generation system.",
                            className="mb-3",
                        ),
                        dbc.Row([
                            dbc.Col([
                                html.H6("Author", className="text-muted mb-1"),
                                html.P("Colin McBane", className="mb-3"),
                                html.H6("Institution", className="text-muted mb-1"),
                                html.P("NC State University", className="mb-3"),
                                html.H6("Major", className="text-muted mb-1"),
                                html.P(
                                    "Applied Mathematics (Financial Mathematics) "
                                    "& Economics",
                                    className="mb-3"
                                ),
                            ], width=6),
                            dbc.Col([
                                html.H6("Champion Model", className="text-muted mb-1"),
                                html.P("LightGBM (AUC: 0.9667)", className="mb-3"),
                                html.H6("Training Data", className="text-muted mb-1"),
                                html.P("382,144 SBA 7(a) loans (2010–2019)", className="mb-3"),
                                html.H6("GitHub", className="text-muted mb-1"),
                                html.A(
                                    "github.com/colinmcbane/credit-risk-loan-decision-engine",
                                    href="https://github.com/colinmcbane/credit-risk-loan-decision-engine",
                                    target="_blank",
                                ),
                            ], width=6),
                        ]),
                    ], className="p-4"),
                ]),
            ], width=12),
        ], className="mb-5"),

        # ── Pipeline Phases ───────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Pipeline Architecture", className="mb-0")),
                    dbc.CardBody([
                        dbc.Table([
                            html.Thead(html.Tr([
                                html.Th("Phase"),
                                html.Th("Description"),
                                html.Th("Key Output"),
                            ])),
                            html.Tbody([
                                html.Tr([
                                    html.Td("Phase 1 — SQL Cleaning"),
                                    html.Td("Data ingestion and cleaning of raw SBA FOIA data using SQLite"),
                                    html.Td("382,144 clean loan records"),
                                ]),
                                html.Tr([
                                    html.Td("Phase 2 — EDA"),
                                    html.Td("Exploratory data analysis in R (Quarto) and Python"),
                                    html.Td("Distribution analysis, correlation matrix, default rate trends"),
                                ]),
                                html.Tr([
                                    html.Td("Phase 3 — Feature Engineering"),
                                    html.Td("One-hot encoding, log transforms, z-score scaling, train/test split"),
                                    html.Td("26-feature matrix, scaler parameters"),
                                ]),
                                html.Tr([
                                    html.Td("Phase 4 — Modeling"),
                                    html.Td("Champion-challenger framework: LightGBM vs XGBoost vs Logistic Regression"),
                                    html.Td("LightGBM champion (AUC 0.9667, KS 83.63, Gini 93.34%)"),
                                ]),
                                html.Tr([
                                    html.Td("Phase 5 — Fairness Analysis"),
                                    html.Td("ECOA disparate impact testing using 4/5ths rule and equalized odds"),
                                    html.Td("SR 11-7 model card, fairness audit across 4 dimensions"),
                                ]),
                                html.Tr([
                                    html.Td("Phase 6 — AI Decision Engine"),
                                    html.Td("SHAP reason code extraction, Gemini letter generation, Gmail SMTP delivery"),
                                    html.Td("ECOA-compliant adverse action letters for 1,051 flagged loans"),
                                ]),
                                html.Tr([
                                    html.Td("Phase 7 — Dashboard"),
                                    html.Td("Interactive Dash dashboard with live loan application form"),
                                    html.Td("This deployed web application"),
                                ]),
                            ]),
                        ], bordered=True, hover=True, striped=True,
                           responsive=True, size="sm"),
                    ], className="p-4"),
                ]),
            ], width=12),
        ], className="mb-5"),

        # ── Data Sources & Regulatory References ──────────────────────────
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Data Sources", className="mb-0")),
                    dbc.CardBody([
                        dbc.ListGroup([
                            dbc.ListGroupItem([
                                html.H6("FOIA - 7(a) FY2010-FY2019", className="mb-1"),
                                html.P(
                                    "SBA 7(a) loan records from fiscal year 2010 through 2019. "
                                    "Primary historical training dataset.",
                                    className="mb-1 text-muted",
                                    style={"fontSize": "0.85rem"},
                                ),
                                html.A(
                                    "data.sba.gov/dataset/7-a-504-foia",
                                    href="https://data.sba.gov/dataset/7-a-504-foia",
                                    target="_blank",
                                ),
                            ]),
                            dbc.ListGroupItem([
                                html.H6("FOIA - 7(a) FY2020-Present", className="mb-1"),
                                html.P(
                                    "SBA 7(a) loan records from fiscal year 2020 to present. "
                                    "Combined with FY2010-FY2019 to form the full 382,144 loan corpus.",
                                    className="mb-1 text-muted",
                                    style={"fontSize": "0.85rem"},
                                ),
                                html.A(
                                    "data.sba.gov/dataset/7-a-504-foia",
                                    href="https://data.sba.gov/dataset/7-a-504-foia",
                                    target="_blank",
                                ),
                            ]),
                            dbc.ListGroupItem([
                                html.H6("7(a) & 504 FOIA Data Dictionary", className="mb-1"),
                                html.P(
                                    "Official SBA data dictionary defining all field names, "
                                    "definitions, and valid values used in the FOIA datasets.",
                                    className="mb-1 text-muted",
                                    style={"fontSize": "0.85rem"},
                                ),
                                html.A(
                                    "data.sba.gov/dataset/7-a-504-foia",
                                    href="https://data.sba.gov/dataset/7-a-504-foia",
                                    target="_blank",
                                ),
                            ]),
                            dbc.ListGroupItem([
                                html.H6("Synthetic Validation Datasets",
                                        className="mb-1"),
                                html.P(
                                    "200 synthetic loan applicants generated by 4 AI "
                                    "systems (Gemini, ChatGPT, Claude, Copilot) for "
                                    "out-of-sample model validation with known risk "
                                    "tier profiles.",
                                    className="mb-1 text-muted",
                                    style={"fontSize": "0.85rem"},
                                ),
                                html.Span("Generated for this project — not publicly available",
                                          style={"fontSize": "0.85rem", "color": "#888"}),
                            ]),
                        ], flush=True),
                    ], className="p-4"),
                ]),
            ], width=6),

            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Regulatory References", className="mb-0")),
                    dbc.CardBody([
                        dbc.ListGroup([
                            dbc.ListGroupItem([
                                html.H6("ECOA Regulation B — 12 CFR Part 1002",
                                        className="mb-1"),
                                html.P(
                                    "Equal Credit Opportunity Act implementing regulation. "
                                    "Governs adverse action notice requirements and "
                                    "prohibits credit discrimination.",
                                    className="mb-1 text-muted",
                                    style={"fontSize": "0.85rem"},
                                ),
                                html.A(
                                    "consumerfinance.gov/rules-policy/regulations/1002",
                                    href="https://www.consumerfinance.gov/rules-policy/regulations/1002/",
                                    target="_blank",
                                ),
                            ]),
                            dbc.ListGroupItem([
                                html.H6("SR 11-7 — Model Risk Management Guidance",
                                        className="mb-1"),
                                html.P(
                                    "Federal Reserve and OCC supervisory guidance on "
                                    "model risk management. Requires model validation, "
                                    "documentation, and ongoing monitoring.",
                                    className="mb-1 text-muted",
                                    style={"fontSize": "0.85rem"},
                                ),
                                html.A(
                                    "federalreserve.gov/supervisionreg/srletters/sr1107.htm",
                                    href="https://www.federalreserve.gov/supervisionreg/srletters/sr1107.htm",
                                    target="_blank",
                                ),
                            ]),
                            dbc.ListGroupItem([
                                html.H6("CFPB — Consumer Financial Protection Bureau",
                                        className="mb-1"),
                                html.P(
                                    "Federal agency overseeing ECOA compliance. "
                                    "Contact information included in all adverse "
                                    "action letters per Regulation B requirements.",
                                    className="mb-1 text-muted",
                                    style={"fontSize": "0.85rem"},
                                ),
                                html.A(
                                    "consumerfinance.gov",
                                    href="https://www.consumerfinance.gov",
                                    target="_blank",
                                ),
                            ]),
                        ], flush=True),
                    ], className="p-4"),
                ]),
            ], width=6),
        ], className="mb-5"),

        # ── Technical Citations ───────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Technical Citations", className="mb-0")),
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                dbc.ListGroup([
                                    dbc.ListGroupItem([
                                        html.H6("LightGBM", className="mb-1"),
                                        html.P(
                                            "Ke, G., et al. (2017). LightGBM: A Highly "
                                            "Efficient Gradient Boosting Decision Tree. "
                                            "Advances in Neural Information Processing "
                                            "Systems (NeurIPS).",
                                            className="mb-1 text-muted",
                                            style={"fontSize": "0.85rem"},
                                        ),
                                        html.A("github.com/microsoft/LightGBM",
                                               href="https://github.com/microsoft/LightGBM",
                                               target="_blank",
                                               style={"fontSize": "0.85rem"}),
                                    ]),
                                    dbc.ListGroupItem([
                                        html.H6("SHAP (SHapley Additive exPlanations)",
                                                className="mb-1"),
                                        html.P(
                                            "Lundberg, S. M., & Lee, S. I. (2017). "
                                            "A unified approach to interpreting model "
                                            "predictions. NeurIPS.",
                                            className="mb-1 text-muted",
                                            style={"fontSize": "0.85rem"},
                                        ),
                                        html.A("github.com/shap/shap",
                                               href="https://github.com/shap/shap",
                                               target="_blank",
                                               style={"fontSize": "0.85rem"}),
                                    ]),
                                    dbc.ListGroupItem([
                                        html.H6("SMOTE", className="mb-1"),
                                        html.P(
                                            "Chawla, N. V., et al. (2002). SMOTE: "
                                            "Synthetic Minority Over-sampling Technique. "
                                            "Journal of Artificial Intelligence Research.",
                                            className="mb-1 text-muted",
                                            style={"fontSize": "0.85rem"},
                                        ),
                                    ]),
                                ], flush=True),
                            ], width=6),
                            dbc.Col([
                                dbc.ListGroup([
                                    dbc.ListGroupItem([
                                        html.H6("Fairlearn", className="mb-1"),
                                        html.P(
                                            "Bird, S., et al. (2020). Fairlearn: A toolkit "
                                            "for assessing and improving fairness in AI. "
                                            "Microsoft Research.",
                                            className="mb-1 text-muted",
                                            style={"fontSize": "0.85rem"},
                                        ),
                                        html.A("fairlearn.org",
                                               href="https://fairlearn.org",
                                               target="_blank",
                                               style={"fontSize": "0.85rem"}),
                                    ]),
                                    dbc.ListGroupItem([
                                        html.H6("Google Gemini API", className="mb-1"),
                                        html.P(
                                            "Used for ECOA-compliant adverse action "
                                            "letter generation. Gemini 3.1 Flash Lite "
                                            "model via Google AI Studio.",
                                            className="mb-1 text-muted",
                                            style={"fontSize": "0.85rem"},
                                        ),
                                        html.A("ai.google.dev",
                                               href="https://ai.google.dev",
                                               target="_blank",
                                               style={"fontSize": "0.85rem"}),
                                    ]),
                                    dbc.ListGroupItem([
                                        html.H6("Plotly Dash", className="mb-1"),
                                        html.P(
                                            "Framework used to build this interactive "
                                            "dashboard. Dash v4.3.0 with "
                                            "dash-bootstrap-components FLATLY theme.",
                                            className="mb-1 text-muted",
                                            style={"fontSize": "0.85rem"},
                                        ),
                                        html.A("dash.plotly.com",
                                               href="https://dash.plotly.com",
                                               target="_blank",
                                               style={"fontSize": "0.85rem"}),
                                    ]),
                                ], flush=True),
                            ], width=6),
                        ]),
                    ], className="p-4"),
                ]),
            ], width=12),
        ], className="mb-5"),

        # ── Model Limitations ─────────────────────────────────────────────
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader(html.H5("Model Limitations & Disclosures",
                                           className="mb-0")),
                    dbc.CardBody([
                        dbc.Alert([
                            html.Strong("SR 11-7 Required Disclosures: "),
                            html.Br(), html.Br(),
                            html.Strong("1. Data Scope: "),
                            "Model trained on SBA 7(a) loans from 2010–2019. "
                            "Performance may differ on post-2019 loan cohorts "
                            "due to macroeconomic shifts including COVID-19. ",
                            html.Br(), html.Br(),
                            html.Strong("2. SMOTE Probability Shift: "),
                            "Synthetic oversampling was applied to the minority "
                            "class (defaults) during training. Predicted "
                            "probabilities may be upward-biased relative to the "
                            "true 7.3% historical default rate. ",
                            html.Br(), html.Br(),
                            html.Strong("3. Tree Model Tail Risk: "),
                            "LightGBM predictions plateau at extreme feature "
                            "values due to leaf node capping. Severe stress "
                            "scenarios may underestimate true tail risk. ",
                            html.Br(), html.Br(),
                            html.Strong("4. Missing Features: "),
                            "Collateral indicator and revolving credit status "
                            "were excluded from the final feature set due to "
                            "data availability constraints. These may improve "
                            "model performance if incorporated. ",
                            html.Br(), html.Br(),
                            html.Strong("5. Geographic Scope: "),
                            "Only 8 borrower states are explicitly modeled as "
                            "features. Borrowers from other states are scored "
                            "using the base model without state-specific adjustment.",
                        ], color="warning", style={"fontSize": "0.85rem"}),
                    ], className="p-4"),
                ]),
            ], width=12),
        ], className="mb-5"),

    ])
