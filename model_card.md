# Model Card — SBA 7(a) Credit Risk Decision Engine

**Author:** Colin McBane
**Date:** 2026-06-11
**Model:** LightGBM
**Version:** 1.0

---

## 1. Model Purpose

This model predicts the probability of default for SBA 7(a) small
business loans. It is designed to assist commercial lending
underwriters in prioritizing loan applications for review and
generating ECOA-compliant Adverse Action notices for denied
applications. The model is not designed to replace human judgment
and should be used as one input among many in the credit decision
process.

**Intended Use:** Credit risk screening for SBA 7(a) loan applications
**Out-of-Scope Use:** Consumer lending, mortgage lending, any use
without human oversight

---

## 2. Training Data

**Source:** SBA 7(a) FOIA Loan Data — raw federal records
**Period:** FY2010 — FY2022 (FY2023+ excluded as right-censored)
**Size:** 382,144 training loans | 95,536 test loans
**Default Rate:** 7.31% (class imbalance addressed via SMOTE)
**Features:** 26 features after Phase 3 feature selection

**Known Data Limitations:**
- SBA 7(a) data does not contain borrower demographic information
  (race, ethnicity, gender, age). Fairness analysis uses economic
  proxy variables only. A complete ECOA demographic analysis requires
  matching with HMDA records or direct demographic collection.
- FY2023-2025 loans excluded due to right-censoring — insufficient
  time to observe default outcomes. These loans are used for
  prediction only, not model training.

---

## 3. Model Performance

| Metric | Value |
|--------|-------|
| AUC-ROC | 0.9667 |
| KS Statistic | 83.63 |
| Gini Coefficient | 93.34% |
| Precision (t=0.35) | 35.66% |
| Recall (t=0.35) | 95.40% |
| F1 Score | 0.5191 |

**Decision Threshold:** 0.35
The threshold was set below 0.50 to maximize recall (catching
defaults) because the cost of a false negative (approving a
defaulting loan) exceeds the cost of a false positive (rejecting
a viable loan) in commercial credit risk.

**Champion Selection:** LightGBM selected over XGBoost (AUC 0.9638)
and Logistic Regression (AUC 0.8338) via challenger-champion
framework evaluated on held-out test set.

---

## 4. Known Technical Limitations

**SMOTE Probability Shift:**
SMOTE (Synthetic Minority Oversampling Technique) was applied during
training to address 7.31% class imbalance. This causes model-output
probabilities to be systematically higher than real-world default
rates. The mean predicted probability on the test set is
approximately 12-13 percentage points above the actual default rate.
Raw probabilities should not be interpreted as calibrated default
probabilities. The 0.35 decision threshold was optimized on the
post-SMOTE probability distribution.

**Tree Model Tail Risk:**
LightGBM handles out-of-distribution feature values by capping
predictions at the furthest leaf node. Under severe macroeconomic
stress scenarios the model may underestimate tail risk. Stress test
results should be interpreted as lower bounds in extreme scenarios.

**Right Censoring:**
Loans approved in FY2023-2025 have not had sufficient time to
default. Model performance metrics reflect the FY2010-2022 vintage
only. Performance on recent vintages should be monitored separately
as outcomes become observable.

---

## 5. Fairness Analysis

**Methodology:** Statistical disparate impact testing via economic
proxy variables. Direct demographic analysis not possible due to
data limitations documented in Section 2.

**Tests Conducted:**
- ECOA 4/5ths disparate impact rule (geographic, business age,
  industry, loan size dimensions)
- Equalized odds ratio analysis (TPR/FPR parity across groups)
- SHAP proxy variable concentration audit
- Independent validation via fairlearn library

**Results Summary:**

| Dimension | DI Flags | EO Flags |
|-----------|----------|----------|
| Dimension          |   DI Flags (4/5ths) |   EO Flags (ratio) |
|:-------------------|--------------------:|-------------------:|
| Geographic (State) |                   2 |                 19 |
| Business Age       |                   0 |                  0 |
| Industry Sector    |                   1 |                 18 |
| Loan Size          |                   0 |                  4 |

**Total Disparate Impact Flags:** 3
**Total Equalized Odds Flags:** 41

**SHAP Proxy Concentration:**
- Geographic (State): 3.9%
- Industry (Sector): 2.73%
- Business Age: 6.46%
- Loan Size: 0.72%

**Omitted Variable Bias Disclosure:**
Because this dataset lacks direct demographic variables, proxy
testing cannot definitively confirm or deny ECOA compliance.
Flagged disparities in proxy variables require investigation of
whether the underlying population of affected borrowers is
demographically concentrated. Human review is required before
any production deployment.

**Business Necessity Documentation:**
Where disparate impact is identified, Phase 4 stress testing
provides quantitative business necessity evidence. New business
loans show 12.66% baseline default rate versus 5.33% for mature
businesses — a 2.4x differential supporting differentiated risk
treatment under ECOA's business necessity defense. Banks deploying
this model must maintain business necessity documentation per
ECOA Regulation B.

---

## 6. Regulatory Compliance Context

**ECOA (Equal Credit Opportunity Act):**
This model was developed with ECOA compliance as a design
constraint. Adverse Action letters generated by the Phase 6
decision engine use SHAP-derived specific reasons as required
by ECOA Regulation B. Statistical disparate impact testing
follows the CFPB's examination procedures for algorithmic
credit models.

**SR 11-7 (Federal Reserve Model Risk Management):**
- Model purpose and limitations documented in this card
- Champion-challenger validation framework implemented
- Independent fairlearn validation cross-checks manual results
- Stress testing conducted across 5 macroeconomic scenarios
- Known failure modes documented in Section 4
- Ongoing monitoring recommendations provided in Section 7

**Fair Housing Act:**
This model covers commercial lending only. Fair Housing Act
mortgage lending provisions do not apply directly, but the
disparate impact standard from the 2015 HUD rule has been
considered in the proxy variable analysis.

---

## 7. Monitoring Recommendations

1. **Performance monitoring:** Recalculate AUC, KS, and Gini
   quarterly as new loan outcomes become observable.
2. **Population stability:** Monitor feature distributions monthly
   for drift from training distribution.
3. **Fairness monitoring:** Rerun disparate impact analysis
   semi-annually or after any model update.
4. **Right-censored vintage:** Begin evaluating FY2023 loan
   performance in late 2025 when sufficient outcomes are observed.
5. **Demographic data:** Recommend collecting voluntary demographic
   data at application to enable full ECOA compliance testing.
6. **Threshold review:** Revisit 0.35 decision threshold annually
   as economic conditions evolve.

---

## 8. Model Governance

**Development:** Colin McBane, NC State University
**Data Source:** SBA Office of Capital Access — FOIA public records
**Validation:** Held-out test set (20%), 5-fold cross-validation,
independent fairlearn library cross-check
**Explainability:** SHAP TreeExplainer — local and global explanations
**Adverse Action Engine:** Anthropic Claude API (claude-sonnet-4-6) —
  Orchestrated via deterministic SHAP feature mappings for standardized,
  ECOA-compliant Adverse Action notice synthesis. The LLM operates as a
  structured text rendering layer only — credit decisions are made
  exclusively by the LightGBM champion model. The API receives ranked
  SHAP values as structured input and is constrained to produce
  standardized regulatory language. No discretionary credit judgment
  is delegated to the language model.

---

*This model card follows the Model Cards for Model Reporting
framework (Mitchell et al., 2019) and Federal Reserve SR 11-7
model risk management guidance.*
