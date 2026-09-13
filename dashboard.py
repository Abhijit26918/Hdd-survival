import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import xgboost as xgb
import shap

st.set_page_config(page_title="Seagate HDD Reliability Dashboard", layout="wide")
st.title("Seagate HDD Reliability Dashboard")
st.caption(
    "Backblaze Q1 2026 Seagate HDD telemetry -- built for Seagate Corporate Quality interview prep. "
    "See notebook.ipynb for the full analysis (EDA, survival analysis, Cox PH, classifier, "
    "LSTM comparison, SHAP)."
)

DATA_DIR = Path("data")
MODEL_DIR = Path("models")


@st.cache_data
def load_fleet_daily():
    return pd.read_parquet(DATA_DIR / "fleet_daily_summary.parquet")


@st.cache_data
def load_survival_df():
    return pd.read_parquet(DATA_DIR / "survival_df_export.parquet")


@st.cache_data
def load_snapshot():
    return pd.read_parquet(DATA_DIR / "current_snapshot.parquet")


@st.cache_resource
def load_model_and_meta():
    model = xgb.XGBClassifier()
    model.load_model(str(MODEL_DIR / "xgb_classifier.json"))
    with open(MODEL_DIR / "feature_meta.json") as f:
        meta = json.load(f)
    return model, meta


fleet_daily = load_fleet_daily()
survival_df = load_survival_df()
snapshot = load_snapshot()
clf, meta = load_model_and_meta()
feature_cols = meta["feature_cols"]
model_categories = meta["model_categories"]

# ---------------------------------------------------------------------------
# Fleet-level failure rate over time
# ---------------------------------------------------------------------------
st.header("Fleet-level failure rate over time")

fleet_daily = fleet_daily.sort_values("date")
fleet_daily["daily_failure_rate_pct"] = fleet_daily["failure_count"] / fleet_daily["drive_count"] * 100

fig1, ax1 = plt.subplots(figsize=(9, 3.5))
ax1.plot(fleet_daily["date"], fleet_daily["daily_failure_rate_pct"], color="#4C72B0")
ax1.set_ylabel("Daily failure rate (%)")
ax1.set_xlabel("Date")
ax1.set_title("Seagate HDD fleet -- daily failure rate, Q1 2026")
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
st.pyplot(fig1)
st.caption(
    "Static chart, kept simple per the project's scope. Daily counts are noisy given a "
    "~1.5% true annualized failure rate -- see notebook Section 1 for the AFR calculation."
)

# ---------------------------------------------------------------------------
# Kaplan-Meier survival curve by model
# ---------------------------------------------------------------------------
st.header("Kaplan-Meier survival curve by model")

model_counts = survival_df["model"].value_counts()
default_models = model_counts.head(2).index.tolist()
chosen_models = st.multiselect(
    "Compare models (power-on hours vs. survival probability)",
    options=model_counts.index.tolist(),
    default=default_models,
)

if chosen_models:
    fig2, ax2 = plt.subplots(figsize=(9, 4.5))
    for m in chosen_models:
        subset = survival_df[survival_df["model"] == m]
        kmf = KaplanMeierFitter()
        kmf.fit(subset["duration"], subset["event"], label=f"{m} (n={len(subset):,})")
        kmf.plot_survival_function(ax=ax2)
    ax2.set_xlabel("Power-on hours")
    ax2.set_ylabel("Estimated survival probability")
    plt.tight_layout()
    st.pyplot(fig2)
else:
    st.info("Select at least one model above to see its survival curve.")

# ---------------------------------------------------------------------------
# Top-N highest-risk drives
# ---------------------------------------------------------------------------
st.header("Top-N highest-risk drives (current snapshot)")
st.caption(
    "Scored on the working-set drives used to train/evaluate the Section 4 classifier "
    "(406 failed + 8,000 healthy), not the full ~115K-drive fleet -- see notebook Section 7 for why."
)

top_n = st.slider("How many drives to show", min_value=5, max_value=50, value=15)

X_score = snapshot[feature_cols].copy()
X_score["model"] = pd.Categorical(X_score["model"], categories=model_categories)

risk = clf.predict_proba(X_score)[:, 1]
snapshot_scored = snapshot.copy()
snapshot_scored["predicted_risk_7d"] = risk
top_drives = snapshot_scored.sort_values("predicted_risk_7d", ascending=False).head(top_n)

explainer = shap.TreeExplainer(clf)
top_X = X_score.loc[top_drives.index]
shap_vals = explainer(top_X)
top_drives = top_drives.copy()
top_drives["top_driving_feature"] = [
    feature_cols[np.argmax(np.abs(row))] for row in shap_vals.values
]

display_df = top_drives[
    ["serial_number", "model", "date", "predicted_risk_7d", "top_driving_feature"]
].rename(columns={"predicted_risk_7d": "predicted risk (fails within 7d)"})

st.dataframe(
    display_df.style.format({"predicted risk (fails within 7d)": "{:.1%}"}),
    width="stretch",
)

st.caption(
    "Risk scores are calibrated to the deliberately undersampled working set, not the true "
    "fleet base rate -- use for ranking/triage, not as a literal probability of failure. "
    "See notebook Section 4.5."
)
