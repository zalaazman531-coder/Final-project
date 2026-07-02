"""Train and export the final models + artifacts consumed by the Streamlit app.

Reproduces the pipeline from notebooks/life_expectancy_analysis.qmd:
country-median imputation, winsorizing, feature engineering, a country-grouped
stratified split, train-fold-only feature selection, and tuned XGBoost models
for both tasks. Artifacts are written to models/ at the repo root.

Run from the repo root:  python app/train_models.py
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, roc_auc_score,
)
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from xgboost import XGBRegressor, XGBClassifier

RANDOM_STATE = 42
BAND_LABELS = ["Low", "Medium", "High"]
REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

RAW_INPUT_COLS = [
    "Adult_Mortality", "infant_deaths", "Alcohol", "percentage_expenditure",
    "Hepatitis_B", "Measles", "BMI", "under-five_deaths", "Polio",
    "Total_expenditure", "Diphtheria", "HIV/AIDS", "GDP", "Population",
    "thinness_1-19_years", "thinness_5-9_years",
    "Income_composition_of_resources", "Schooling",
]

CANDIDATE_FEATURES = [
    "Adult_Mortality", "infant_deaths", "Alcohol", "log_percentage_expenditure",
    "Hepatitis_B", "Measles", "BMI", "under-five_deaths", "Polio",
    "Total_expenditure", "Diphtheria", "HIV/AIDS", "log_GDP", "log_Population",
    "avg_thinness", "Income_composition_of_resources", "Schooling",
    "avg_immunization", "child_to_adult_mortality_ratio", "schooling_x_income",
    "Status_binary", "era_2005-2009", "era_2010-2015",
]


def load_and_clean() -> pd.DataFrame:
    df = pd.read_csv(REPO_ROOT / "data" / "Life Expectancy Data.csv")
    df.columns = df.columns.str.strip().str.replace(r"\s+", "_", regex=True)

    numeric_cols = df.select_dtypes(include=np.number).columns.drop("Year")
    for col in numeric_cols:
        df[col] = df.groupby("Country")[col].transform(lambda s: s.fillna(s.median()))
        df[col] = df[col].fillna(df[col].median())

    for col in ["GDP", "Population", "percentage_expenditure", "Measles"]:
        lo, hi = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(lo, hi)

    df["Status_binary"] = (df.Status == "Developed").astype(int)
    df["life_expectancy_band"] = pd.qcut(df.Life_expectancy, q=3, labels=BAND_LABELS)
    return df


def engineer_features(raw: pd.DataFrame, year: pd.Series) -> pd.DataFrame:
    """Compute the engineered feature set from raw indicator columns.

    Shared with the Streamlit app (which imports this function) so that the
    form inputs go through exactly the same transformations as training data.
    """
    feat = raw.copy()
    feat["log_GDP"] = np.log1p(feat.GDP)
    feat["log_Population"] = np.log1p(feat.Population)
    feat["log_percentage_expenditure"] = np.log1p(feat.percentage_expenditure)
    feat["avg_immunization"] = feat[["Hepatitis_B", "Polio", "Diphtheria"]].mean(axis=1)
    feat["avg_thinness"] = feat[["thinness_1-19_years", "thinness_5-9_years"]].mean(axis=1)
    feat["child_to_adult_mortality_ratio"] = (
        (feat.infant_deaths + feat["under-five_deaths"] + 1) / (feat.Adult_Mortality + 1)
    )
    feat["schooling_x_income"] = feat.Schooling * feat.Income_composition_of_resources
    era = pd.cut(year, bins=[1999, 2004, 2009, 2015],
                 labels=["2000-2004", "2005-2009", "2010-2015"])
    feat["era_2005-2009"] = (era == "2005-2009").astype(bool)
    feat["era_2010-2015"] = (era == "2010-2015").astype(bool)
    return feat


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    clean = load_and_clean()
    feat = engineer_features(clean, clean.Year).reset_index(drop=True)

    X_all = feat[CANDIDATE_FEATURES]
    y_reg = feat["Life_expectancy"]
    y_clf = feat["life_expectancy_band"].cat.codes
    groups = feat["Country"]

    sgkf_outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    trainval_idx, test_idx = next(sgkf_outer.split(X_all, y_clf, groups))
    sgkf_inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    train_rel, _val_rel = next(sgkf_inner.split(
        X_all.iloc[trainval_idx], y_clf.iloc[trainval_idx], groups.iloc[trainval_idx]
    ))
    train_idx = trainval_idx[train_rel]

    X_train, y_reg_train, y_clf_train = (
        X_all.iloc[train_idx], y_reg.iloc[train_idx], y_clf.iloc[train_idx]
    )
    X_test, y_reg_test, y_clf_test = (
        X_all.iloc[test_idx], y_reg.iloc[test_idx], y_clf.iloc[test_idx]
    )

    # Feature selection (train fold only): correlation filter then LassoCV.
    train_corr = X_train.join(y_reg_train).corr()
    target_corr = train_corr["Life_expectancy"].drop("Life_expectancy")
    to_drop = set()
    for i, f1 in enumerate(CANDIDATE_FEATURES):
        for f2 in CANDIDATE_FEATURES[i + 1:]:
            if abs(train_corr.loc[f1, f2]) > 0.85:
                to_drop.add(f1 if abs(target_corr[f1]) < abs(target_corr[f2]) else f2)
    corr_selected = [f for f in CANDIDATE_FEATURES if f not in to_drop]

    lasso_scaler = StandardScaler().fit(X_train[corr_selected])
    lasso_cv = LassoCV(cv=5, random_state=RANDOM_STATE).fit(
        lasso_scaler.transform(X_train[corr_selected]), y_reg_train
    )
    lasso_coef = pd.Series(lasso_cv.coef_, index=corr_selected)
    selected = lasso_coef[lasso_coef.abs() > 1e-3].index.tolist()
    print(f"Selected {len(selected)} features: {selected}")

    X_train, X_test = X_train[selected], X_test[selected]

    param_grid = {
        "model__n_estimators": [200, 400],
        "model__max_depth": [3, 5],
        "model__learning_rate": [0.05, 0.1],
    }

    print("Tuning XGBoost regressor...")
    xgb_reg = GridSearchCV(
        Pipeline([("scaler", StandardScaler()),
                  ("model", XGBRegressor(random_state=RANDOM_STATE, n_jobs=1))]),
        param_grid, cv=5, scoring="neg_root_mean_squared_error", n_jobs=1,
    ).fit(X_train, y_reg_train)

    print("Tuning XGBoost classifier...")
    xgb_clf = GridSearchCV(
        Pipeline([("scaler", StandardScaler()),
                  ("model", XGBClassifier(random_state=RANDOM_STATE, n_jobs=1,
                                          eval_metric="mlogloss"))]),
        param_grid, cv=StratifiedGroupKFold(n_splits=5),
        scoring="roc_auc_ovr_weighted", n_jobs=1,
    ).fit(X_train, y_clf_train, groups=groups.iloc[train_idx])

    # Test-set leaderboards including the simpler ladder rungs, for the dashboard.
    reg_rows, clf_rows = [], []
    reg_models = [
        ("Dummy (mean)", DummyRegressor(strategy="mean")),
        ("Linear Regression", Pipeline([("scaler", StandardScaler()),
                                        ("model", LinearRegression())])),
        ("Decision Tree", DecisionTreeRegressor(max_depth=6, random_state=RANDOM_STATE)),
    ]
    for name, model in reg_models:
        pred = model.fit(X_train, y_reg_train).predict(X_test)
        reg_rows.append({"model": name, "MAE": mean_absolute_error(y_reg_test, pred),
                         "RMSE": mean_squared_error(y_reg_test, pred) ** 0.5,
                         "R2": r2_score(y_reg_test, pred)})
    best_reg_pred = xgb_reg.predict(X_test)
    reg_rows.append({"model": "XGBoost (tuned)",
                     "MAE": mean_absolute_error(y_reg_test, best_reg_pred),
                     "RMSE": mean_squared_error(y_reg_test, best_reg_pred) ** 0.5,
                     "R2": r2_score(y_reg_test, best_reg_pred)})

    clf_models = [
        ("Dummy (most frequent)", DummyClassifier(strategy="most_frequent")),
        ("Logistic Regression", Pipeline([("scaler", StandardScaler()),
                                          ("model", LogisticRegression(max_iter=1000))])),
        ("Decision Tree", DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE)),
    ]
    for name, model in clf_models:
        model.fit(X_train, y_clf_train)
        pred, proba = model.predict(X_test), model.predict_proba(X_test)
        clf_rows.append({"model": name, "Accuracy": accuracy_score(y_clf_test, pred),
                         "F1_weighted": f1_score(y_clf_test, pred, average="weighted"),
                         "ROC_AUC_ovr": roc_auc_score(y_clf_test, proba,
                                                      multi_class="ovr", average="weighted")})
    best_clf_pred, best_clf_proba = xgb_clf.predict(X_test), xgb_clf.predict_proba(X_test)
    clf_rows.append({"model": "XGBoost (tuned)",
                     "Accuracy": accuracy_score(y_clf_test, best_clf_pred),
                     "F1_weighted": f1_score(y_clf_test, best_clf_pred, average="weighted"),
                     "ROC_AUC_ovr": roc_auc_score(y_clf_test, best_clf_proba,
                                                  multi_class="ovr", average="weighted")})

    # Persist models + everything the app's dashboard and input form need.
    joblib.dump(xgb_reg.best_estimator_, MODELS_DIR / "regressor.joblib")
    joblib.dump(xgb_clf.best_estimator_, MODELS_DIR / "classifier.joblib")

    band_edges = pd.qcut(clean.Life_expectancy, q=3).cat.categories
    raw_stats = clean[RAW_INPUT_COLS].agg(["min", "median", "max"]).T
    meta = {
        "selected_features": selected,
        "band_labels": BAND_LABELS,
        "band_edges": [(float(iv.left), float(iv.right)) for iv in band_edges],
        "raw_input_stats": raw_stats.to_dict(orient="index"),
        "best_reg_params": xgb_reg.best_params_,
        "best_clf_params": xgb_clf.best_params_,
    }
    joblib.dump(meta, MODELS_DIR / "metadata.joblib")

    pd.DataFrame(reg_rows).round(3).to_csv(MODELS_DIR / "leaderboard_regression.csv", index=False)
    pd.DataFrame(clf_rows).round(3).to_csv(MODELS_DIR / "leaderboard_classification.csv", index=False)
    pd.DataFrame({"actual": y_reg_test.values, "predicted": best_reg_pred}).to_csv(
        MODELS_DIR / "test_predictions.csv", index=False)
    pd.Series(xgb_reg.best_estimator_.named_steps["model"].feature_importances_,
              index=selected, name="importance").to_csv(MODELS_DIR / "importance_regression.csv")
    pd.Series(xgb_clf.best_estimator_.named_steps["model"].feature_importances_,
              index=selected, name="importance").to_csv(MODELS_DIR / "importance_classification.csv")

    print("\nRegression leaderboard (test):")
    print(pd.DataFrame(reg_rows).round(3).to_string(index=False))
    print("\nClassification leaderboard (test):")
    print(pd.DataFrame(clf_rows).round(3).to_string(index=False))
    print(f"\nArtifacts written to {MODELS_DIR}")


if __name__ == "__main__":
    main()
