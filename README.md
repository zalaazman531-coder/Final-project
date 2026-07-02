# Life Expectancy Explorer 🌍

End-to-end machine-learning project on the **WHO Life Expectancy dataset**
(193 countries, 2000–2015, 2938 rows): what drives life expectancy, and how well
can we predict it for countries the model has never seen?

- **Regression task** — predict a country-year's life expectancy in years.
- **Classification task** — predict its Low / Medium / High life-expectancy band
  (tertiles of the target, derived independently of the input features).

**🔗 Live app:** _add your Streamlit Cloud URL here after deploying_
(share.streamlit.io → Create app → this repo, branch `main`, file `app/app.py`)

## Results at a glance (held-out test set, unseen countries)

| Task | Best model | Key metric | Baseline |
|---|---|---|---|
| Regression | XGBoost (tuned) | MAE 2.19 years, R² 0.88 | MAE 7.33, R² ≈ 0 |
| Classification | XGBoost (tuned) | 77.4% accuracy, 0.92 ROC-AUC | 33.8% accuracy |

Full leaderboards (incl. Dummy, Linear/Logistic, Decision Tree) are in the report
and the app's Findings tab.

## Repository layout

| Path | What it is |
|---|---|
| `notebooks/life_expectancy_analysis.qmd` | **D1** — the full reproducible analysis (Quarto → HTML) |
| `app/app.py` | **D2** — Streamlit app: predictor + findings dashboard |
| `app/train_models.py` | Trains the final models and exports artifacts to `models/` |
| `models/` | Exported pipelines (joblib), leaderboards, feature importances |
| `ai_reflection.md` | **D3** — AI-workflow reflection |
| `slides.qmd` | **D4** — Quarto reveal.js slide deck |
| `summary.md` | **D5** — one-page plain-language summary |
| `data/Life Expectancy Data.csv` | The WHO dataset (public) |

## Reproduce from a clean checkout

```bash
python -m venv venv
venv\Scripts\activate        # Windows; on macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
```

**Render the report** (requires the [Quarto CLI](https://quarto.org/docs/get-started/)):

```bash
quarto render notebooks/life_expectancy_analysis.qmd
```

**Run the app locally:**

```bash
python app/train_models.py     # only if models/ is missing — artifacts are committed
streamlit run app/app.py
```

**Render the slides:**

```bash
quarto render slides.qmd
```

All randomness is seeded (`random_state=42`), so results are reproducible.
Preprocessing (scaling, feature selection) is fitted on the training fold only,
and the train/val/test split is grouped by country and stratified by the
classification band — test scores measure generalization to unseen countries.
