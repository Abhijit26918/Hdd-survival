# Hard Drive Failure Prediction — Seagate Interview Prep

An end-to-end reliability-prediction project built on **real Backblaze Drive Stats telemetry**
(Q1 2026, Seagate HDDs only), covering the four areas most relevant to a hardware reliability /
warranty-prediction role: survival analysis, sequence modeling, imbalanced classification, and
explainability. Every result below is from an actual run against real data — not illustrative
numbers.

## What's here

| File | What it is |
|---|---|
| `notebook.ipynb` | The full analysis, Sections 0–7 (see below) |
| `dashboard.py` | A small Streamlit app: fleet failure rate, survival curves by model, top-risk drives |
| `notes.txt` | Every concept, formula, real result, and bug hit along the way, section by section |
| `seagate_prep_project_brief.md` | The original project brief this was built from |
| `plots/` | The 14 charts produced by the notebook (also embedded inline in it) |
| `models/`, `data/*.parquet` (small ones) | Exported artifacts `dashboard.py` loads directly, no retraining needed |
| `requirements.txt` | Exact package versions used |

## Dataset

[Backblaze Drive Stats](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data)
— one row per drive per day, ~90 SMART attribute columns. Filtered to Seagate HDDs only
(`model` starting with `"ST"`, which cleanly excludes Seagate's SSD lines too — verified
directly on this data).

Per Backblaze's usage terms: this project cites Backblaze as the data source, the raw data is not
redistributed here (see `.gitignore` — the raw CSVs/ZIP and the full filtered cache are excluded;
only small derived/aggregated exports needed for the dashboard are committed), and all analysis
and its use are the author's own responsibility.

**To reproduce from scratch:** download the latest quarterly ZIP from the link above, extract it
to `data/data_Q1_2026/data_Q1_2026/*.csv` (or update the path in Section 1), then run the notebook
top to bottom — it caches a filtered parquet after the first run.

## What the notebook covers

0. **Setup & framing** — license note, how this maps to a warranty/reliability role
1. **EDA** — annualized failure rate (AFR), missing-data check, class imbalance quantified,
   SMART-attribute correlation with failure, the reliability "bathtub curve" recovered from real
   drive-age data
2. **Survival analysis** — censoring, Kaplan-Meier estimator, the hazard function, log-rank test
   (with the underlying math alongside the code)
3. **Cox Proportional Hazards** — hazard ratios per SMART attribute, plus two real modeling
   issues hit and resolved (a near-duplicate SMART column, and a convergence failure from rare
   binary predictors) — both documented as they happened, not smoothed over
4. **Practical classifier** — XGBoost with class weighting for imbalance, evaluated on
   PR-AUC/F2 (not accuracy), with an honest calibration finding
5. **LSTM vs. tree model** — a controlled, single-attribute sequence-modeling comparison, with
   the LSTM gate equations explained
6. **SHAP explainability** — cross-checked against the Cox PH hazard ratios from Section 3
7. **Dashboard** — `dashboard.py`, built on artifacts the notebook exports

## Some real results

- True annualized failure rate: **1.47%** (on 10M+ drive-days, 114,976 Seagate HDDs)
- A genuine bathtub curve recovered from real drive-age data (infant mortality → useful life →
  wear-out)
- Cox PH hazard ratios: SMART 197 (Current Pending Sector) → **2.14×** instantaneous failure
  risk, controlling for other SMART flags
- Classifier: **35× PR-AUC lift** over random baseline on a 7-day-ahead failure prediction task
- SHAP feature ranking and Cox PH hazard-ratio ranking **agree on 2 of 4** shared attributes —
  reported precisely, not rounded up to "they agree"

See `notes.txt` for the full breakdown, including the real bugs hit along the way (a data-scale
collinearity issue, a Cox PH convergence failure, a SHAP plotting API quirk) and how each was
diagnosed and fixed.

## Setup

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m ipykernel install --user --name seagate-venv --display-name "Python (seagate-venv)"
```

Open `notebook.ipynb` and select the **Python (seagate-venv)** kernel.

## Running the dashboard

```bash
.venv\Scripts\python.exe -m streamlit run dashboard.py
```

(Requires the notebook to have been run at least once, so `models/` and the exported
`data/*.parquet` files exist.)
