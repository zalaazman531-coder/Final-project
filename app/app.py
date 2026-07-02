"""Streamlit app: life-expectancy predictor + findings dashboard.

Loads the pipelines and artifacts exported by app/train_models.py (models/).
Run locally from the repo root:  streamlit run app/app.py
"""

import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_models import engineer_features  # noqa: E402  (shared transformations)

st.set_page_config(page_title="Life Expectancy Explorer", page_icon="🌍", layout="wide")


@st.cache_resource
def load_artifacts():
    return {
        "regressor": joblib.load(MODELS_DIR / "regressor.joblib"),
        "classifier": joblib.load(MODELS_DIR / "classifier.joblib"),
        "meta": joblib.load(MODELS_DIR / "metadata.joblib"),
        "lb_reg": pd.read_csv(MODELS_DIR / "leaderboard_regression.csv"),
        "lb_clf": pd.read_csv(MODELS_DIR / "leaderboard_classification.csv"),
        "test_pred": pd.read_csv(MODELS_DIR / "test_predictions.csv"),
        "imp_reg": pd.read_csv(MODELS_DIR / "importance_regression.csv", index_col=0),
        "imp_clf": pd.read_csv(MODELS_DIR / "importance_classification.csv", index_col=0),
    }


try:
    art = load_artifacts()
except FileNotFoundError:
    st.error("Model artifacts not found. Run `python app/train_models.py` first.")
    st.stop()

meta = art["meta"]
BAND_LABELS = meta["band_labels"]
BAND_COLORS = {"Low": "#c44e52", "Medium": "#dd8452", "High": "#4c72b0"}

st.title("🌍 Life Expectancy Explorer")
st.caption(
    "Trained on the WHO Life Expectancy dataset (193 countries, 2000–2015). "
    "Tuned XGBoost models predict a country-year's life expectancy (regression) "
    "and its Low / Medium / High band (classification)."
)

tab_predict, tab_findings = st.tabs(["🔮 Predictor", "📊 Findings"])

# ---------------------------------------------------------------- Predictor
FORM_FIELDS = {
    "Adult_Mortality": "Adult mortality (deaths ages 15–60 per 1000)",
    "infant_deaths": "Infant deaths (per 1000 population)",
    "under-five_deaths": "Under-five deaths (per 1000 population)",
    "HIV/AIDS": "HIV/AIDS deaths (per 1000 live births, ages 0–4)",
    "Hepatitis_B": "Hepatitis B immunization coverage (%)",
    "Polio": "Polio immunization coverage (%)",
    "Diphtheria": "Diphtheria immunization coverage (%)",
    "Measles": "Measles cases (per 1000 population)",
    "BMI": "Average BMI (whole population)",
    "thinness_1-19_years": "Thinness, ages 10–19 (%)",
    "thinness_5-9_years": "Thinness, ages 5–9 (%)",
    "Alcohol": "Alcohol consumption (litres per capita)",
    "GDP": "GDP per capita (USD)",
    "Population": "Population",
    "percentage_expenditure": "Health expenditure (% of GDP per capita)",
    "Total_expenditure": "Government health expenditure (% of total)",
    "Income_composition_of_resources": "Income composition of resources (HDI, 0–1)",
    "Schooling": "Schooling (expected years)",
}

with tab_predict:
    st.subheader("Enter a country profile")
    st.markdown(
        "Defaults are the dataset medians — adjust the indicators to describe a "
        "country-year, then press **Predict**."
    )
    with st.form("predictor"):
        cols = st.columns(3)
        values = {}
        for i, (col_name, label) in enumerate(FORM_FIELDS.items()):
            stats = meta["raw_input_stats"][col_name]
            values[col_name] = cols[i % 3].number_input(
                label,
                min_value=float(stats["min"]),
                max_value=float(stats["max"]),
                value=float(stats["median"]),
            )
        c1, c2 = st.columns(2)
        status = c1.selectbox("Development status", ["Developing", "Developed"])
        year = c2.selectbox("Year", list(range(2000, 2016)), index=15)
        submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

    if submitted:
        raw = pd.DataFrame([values])
        raw["Status_binary"] = int(status == "Developed")
        features = engineer_features(raw, pd.Series([year]))
        X = features[meta["selected_features"]]

        pred_years = float(art["regressor"].predict(X)[0])
        proba = art["classifier"].predict_proba(X)[0]
        band = BAND_LABELS[int(proba.argmax())]

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted life expectancy", f"{pred_years:.1f} years")
        m2.metric("Predicted band", band, help="Tertile band of the 2000–2015 WHO data")
        m3.metric("Band confidence", f"{proba.max():.0%}")

        prob_df = pd.DataFrame({"band": BAND_LABELS, "probability": proba})
        fig, ax = plt.subplots(figsize=(6, 2.2))
        ax.barh(prob_df.band, prob_df.probability,
                color=[BAND_COLORS[b] for b in prob_df.band])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.invert_yaxis()
        for y_pos, p in enumerate(prob_df.probability):
            ax.text(p + 0.01, y_pos, f"{p:.0%}", va="center")
        st.pyplot(fig, use_container_width=False)

        lo, hi = meta["band_edges"][BAND_LABELS.index(band)]
        st.info(
            f"The **{band}** band covers roughly {lo:.0f}–{hi:.0f} years of life "
            "expectancy in the training data. Predictions near a band boundary "
            "should be treated as low-confidence even when the probability looks "
            "decisive, and the model reflects 2000–2015 associations — it is not "
            "a causal or forecasting tool."
        )

# ---------------------------------------------------------------- Findings
with tab_findings:
    st.subheader("Model leaderboards (held-out test set)")
    st.markdown(
        "Both ladders are evaluated on countries **never seen in training** "
        "(group split by country, stratified by band). The Dummy baseline is kept "
        "in the table so every improvement is measured against doing nothing smart."
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Regression — predicting life expectancy (years)**")
        st.dataframe(art["lb_reg"], hide_index=True, use_container_width=True)
    with c2:
        st.markdown("**Classification — predicting Low / Medium / High band**")
        st.dataframe(art["lb_clf"], hide_index=True, use_container_width=True)

    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Predicted vs. actual (best regressor, test set)**")
        tp = art["test_pred"]
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.scatter(tp.actual, tp.predicted, alpha=0.4, s=18)
        lims = [tp.actual.min() - 2, tp.actual.max() + 2]
        ax.plot(lims, lims, "r--", label="Perfect prediction")
        ax.set_xlabel("Actual life expectancy")
        ax.set_ylabel("Predicted life expectancy")
        ax.legend()
        st.pyplot(fig)
    with c2:
        st.markdown("**Top features (XGBoost importance)**")
        which = st.radio("Task", ["Regression", "Classification"],
                         horizontal=True, label_visibility="collapsed")
        imp = (art["imp_reg"] if which == "Regression" else art["imp_clf"])
        imp = imp.sort_values("importance").tail(10)
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ax.barh(imp.index, imp.importance,
                color="steelblue" if which == "Regression" else "darkorange")
        ax.set_xlabel("Importance")
        st.pyplot(fig)

    st.divider()
    st.markdown(
        """
**How to read these results**

- The tuned XGBoost regressor predicts life expectancy within about **2.2 years
  (MAE)** for countries it has never seen — versus ~7.3 years for the naive baseline.
- HIV/AIDS prevalence and adult mortality dominate the regression — expected, since
  they are near-direct components of life expectancy. Income composition and
  schooling carry most of the remaining signal.
- **What not to trust this model for:** causal claims ("more schooling causes longer
  life"), individual-level predictions (the data is country-aggregated), or
  countries/periods far outside the 2000–2015 WHO reporting coverage.
"""
    )
