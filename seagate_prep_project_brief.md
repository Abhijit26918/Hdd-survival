# Interview-Prep Project: Hard Drive Failure Prediction on Real Backblaze Data

**Goal:** A single, deeply-explained Jupyter notebook (EDA → survival analysis → predictive model → explainability → dashboard) using **real** hard drive telemetry, built specifically to demonstrate the four concepts most likely to come up in the Seagate Corporate Quality (Data Science) intern interview:
1. Warranty/reliability prediction as a survival problem
2. Time-series/sequential reasoning (with an honest RNN/LSTM vs. tree-model comparison)
3. Feature engineering + evaluation depth under class imbalance
4. Explainability (SHAP) — the "AI agent" / automation angle gets addressed narratively in the README, not in code

Every section should have a markdown cell explaining **why**, not just **what**, so it can be read aloud or referenced live in the interview.

---

## 1. Dataset: Backblaze Drive Stats

- **Source:** https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data (quarterly ZIP archives of daily CSVs)
- **License:** Free to use — Backblaze asks you to (a) cite them as the source, (b) accept responsibility for how you use the data, (c) not resell the raw data itself. Put this in the notebook's first markdown cell.
- **Structure:** One row per drive per day. Columns: `date`, `serial_number`, `model`, `capacity_bytes`, `failure` (0/1), plus ~90 raw + normalized SMART attribute columns (`smart_5_raw`, `smart_187_raw`, `smart_197_raw`, etc.)
- **Scope it down** (this is the "small" part):
  - Pick **one recent quarter** (e.g., Q1 2026 or latest available) rather than the full multi-year archive
  - Filter to **Seagate models only** — directly relevant to the interviewer, and cuts the data drastically
  - From that, keep the drives that either failed during the quarter, or a random sample (e.g., 5,000–10,000) of drives that didn't, to keep the notebook fast to run while preserving the imbalance story (this pre-sampling should itself be a documented, explained EDA step — "here's the true failure rate before I subsampled, and how I preserved it")
  - Keep SMART attributes that Backblaze's own research has flagged as predictive: SMART 5 (Reallocated Sectors), 187 (Reported Uncorrectable Errors), 188 (Command Timeout), 197 (Current Pending Sector), 198 (Uncorrectable Sector Count), plus `smart_9_raw` (power-on hours, i.e. age)

## 2. Notebook structure

### Section 0 — Setup & framing
- License/attribution note
- One paragraph framing: "This mirrors the warranty/reliability prediction problem described in the Seagate Corporate Quality JD — predicting drive failure from field telemetry."

### Section 1 — EDA (with real statistical reasoning, not just plots)
- Overall failure rate in the sample — state it as annualized failure rate (AFR), the way Backblaze itself reports it, since that's the industry-standard framing
- Missing data check on SMART fields (some attributes are model-dependent and legitimately null — explain *why*, don't just impute blindly)
- Class imbalance quantified explicitly (e.g., "X% of drive-days are failures — a naive always-predict-healthy baseline would be Y% accurate and useless")
- Correlation of top SMART attributes with failure (this is literally what Backblaze's own blog does — you can reference their public finding that SMART 5, 187, 188, 197, 198 are the strongest predictors, and verify it independently on your slice)
- Age (power-on hours) vs. failure rate — sets up the "bathtub curve" reliability concept (high early-life failure, low mid-life, rising late-life) — worth a markdown explanation since it's classic reliability-engineering knowledge

### Section 2 — Reframe as a survival problem
- Explain censoring explicitly: a drive still running at quarter-end is *censored*, not "confirmed healthy forever"
- Build a **Kaplan-Meier curve** (use the `lifelines` Python package) of survival probability vs. power-on hours, split by model
- Markdown cell explaining the hazard function conceptually and why it's more actionable than the survival function alone
- Optional: log-rank test comparing two models' survival curves — a real statistical test, good depth signal

### Section 3 — Cox Proportional Hazards model
- Fit Cox PH (`lifelines.CoxPHFitter`) with SMART attributes + age as covariates
- Interpret hazard ratios explicitly in the markdown ("a hazard ratio of 2.1 for SMART 5 > 0 means those drives fail at roughly 2.1x the instantaneous rate, controlling for age") — this is the single most "I actually understand this" signal you can give
- Check (even briefly) the proportional-hazards assumption using `lifelines`' built-in check — mentioning you checked it, even if you don't act extensively on a violation, shows rigor

### Section 4 — Practical classification complement
- Frame explicitly: "Cox PH is the statistically correct approach; here's the practical complement a dashboard consumer would actually want — a calibrated failure-risk score per drive"
- Feature engineering: rolling trend of key SMART attributes (e.g., 7-day change in SMART 5), age, model — document *why* each feature might matter physically, not just "features engineered"
- Handle imbalance explicitly — this is your bridge to your Netsec project: use class weights or SMOTE, and say so in markdown
- Train a gradient-boosted model (XGBoost or LightGBM) — practical default for tabular reliability data
- Evaluate with **PR-AUC and recall/F2**, not accuracy — markdown cell explaining the cost asymmetry (missed failure > false alarm) exactly like we discussed
- Optional stretch: calibration curve — is a predicted 20% risk actually ~20% empirically?

### Section 5 — Sequence-model comparison (your honest RNN/LSTM story)
- Take one SMART attribute (e.g., SMART 5) as a per-drive time series over the quarter
- Build a **small 1-2 layer LSTM** predicting failure from the raw sequence, and compare against the tree-based model on engineered features
- Markdown conclusion should be honest and match what you likely found at IIT Ropar: with limited data/short sequences, the simpler tree-based model on engineered features may perform comparably or better, while the LSTM would need much more data/longer sequences to show its advantage — **this is a genuinely valuable, defensible finding**, not a forced one
- Explicitly tie back: "this mirrors the RNN vs LSTM comparison I did on ECG waveforms — same underlying question of how much sequence memory the model needs to capture the degradation pattern"

### Section 6 — Explainability
- SHAP values on the gradient-boosted model (you already know this stack from the Airbnb project)
- Cross-check: do the top SHAP features match the Cox PH hazard ratios from Section 3? If yes, that's a strong "two independent methods agree" narrative point worth stating explicitly

### Section 7 — Dashboard
- Simple **Streamlit** app (`streamlit run dashboard.py`, separate .py file, not in the notebook) showing:
  - Fleet-level failure rate over time
  - KM survival curve by model (interactive if easy, static otherwise)
  - Top-N highest-risk drives by current predicted risk score, with their top SHAP-driving features
- Keep this genuinely simple — the interview value is "I can go from raw data to something a stakeholder could open," not dashboard polish

---

## 3. What to say about this project in the interview

Lead with: *"I built an end-to-end reliability prediction project on Backblaze's public hard drive telemetry dataset specifically to prepare for this interview, since it's the closest public analog to the warranty/reliability problem in the JD."* Then let them steer — you'll have real, specific numbers (actual AFR, actual hazard ratios, actual model performance) rather than hypotheticals, which is a significant step up from most intern candidates.

## 4. Suggested folder structure for Claude Code

```
seagate-prep/
  data/               # downloaded + filtered Backblaze CSVs (don't commit raw data)
  notebook.ipynb       # main analysis, sections 0-6 above
  dashboard.py         # Streamlit app, section 7
  README.md            # dataset license/attribution + one-paragraph project summary
```
