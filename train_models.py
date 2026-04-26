"""
Wildfire Prediction - Improved Model Training Script

Improvements over v1:
  1. Lag features — prev-day fire per county (strong temporal signal)
  2. Log-transform FRP target — handles right-skewed fire intensity
  3. Model selection by val PR-AUC (better for imbalanced classes)
  4. Threshold tuned with F-beta (beta=2) to weight recall 2× over precision
"""

import os
import json
import joblib
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix, roc_curve, precision_recall_curve, fbeta_score
)
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
from utils import EnsembleClassifier, IsotonicCalibrator

warnings.filterwarnings("ignore")

DATA_DIR = Path("/Users/dipinjassal/Downloads/Data_Split")
OUT_DIR  = Path(__file__).parent / "models"
OUT_DIR.mkdir(exist_ok=True)

LEAKAGE_COLS = ["max_frp", "max_brightness", "fire_count"]
TARGET_CLF   = "fire_label"
TARGET_REG   = "max_frp"
BETA         = 3   # weight recall 3× over precision (missed fire >> false alarm)


# ── Lag feature engineering ────────────────────────────────────────────────────
def add_lag_features(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
    """Add prev_day_fire lag per county without cross-split leakage."""
    county_cols = [c for c in train.columns if c.startswith("county_")]

    all_splits = pd.concat([
        train.assign(_split="train"),
        val.assign(_split="val"),
        test.assign(_split="test"),
    ], ignore_index=True)

    # Reconstruct county name from one-hot (argmax of county cols)
    all_splits["_county"] = (
        all_splits[county_cols]
        .idxmax(axis=1)
        .str.replace("county_", "", regex=False)
    )
    all_splits["date"] = pd.to_datetime(all_splits["date"])
    all_splits = all_splits.sort_values(["_county", "date"]).reset_index(drop=True)

    # Lag-1 fire per county (NaN on first day of each county → fill 0)
    all_splits["prev_day_fire"] = (
        all_splits.groupby("_county")[TARGET_CLF]
        .shift(1)
        .fillna(0)
    )

    # Lag-2 fire per county
    all_splits["prev2_day_fire"] = (
        all_splits.groupby("_county")[TARGET_CLF]
        .shift(2)
        .fillna(0)
    )

    # 7-day rolling fire count per county (shift to avoid leakage)
    all_splits["fire_7d_rolling"] = (
        all_splits.groupby("_county")[TARGET_CLF]
        .transform(lambda x: x.shift(1).rolling(7, min_periods=1).sum())
        .fillna(0)
    )

    all_splits = all_splits.drop(columns=["_county"])

    train_out = all_splits[all_splits["_split"] == "train"].drop(columns=["_split"])
    val_out   = all_splits[all_splits["_split"] == "val"].drop(columns=["_split"])
    test_out  = all_splits[all_splits["_split"] == "test"].drop(columns=["_split"])

    return train_out, val_out, test_out


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_feature_cols(df: pd.DataFrame, extra_drop: list = []) -> list:
    drop = {"date", TARGET_CLF, "_split", *LEAKAGE_COLS, *extra_drop}
    return [c for c in df.columns if c not in drop]


def load_splits():
    train = pd.read_csv(DATA_DIR / "train.csv")
    val   = pd.read_csv(DATA_DIR / "val.csv")
    test  = pd.read_csv(DATA_DIR / "test.csv")
    smote = pd.read_csv(DATA_DIR / "train_features_smote.csv")
    print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}, SMOTE: {smote.shape}")
    return train, val, test, smote


def best_threshold_fbeta(model, X_val, y_val, beta: float = 2.0) -> float:
    """Find threshold maximising F-beta (beta>1 weights recall more)."""
    proba = model.predict_proba(X_val)[:, 1]
    _, _, thresholds = precision_recall_curve(y_val, proba)
    scores = []
    for t in thresholds:
        preds = (proba >= t).astype(int)
        scores.append(fbeta_score(y_val, preds, beta=beta, zero_division=0))
    return float(thresholds[np.argmax(scores)])


def eval_classifier(model, X, y, name: str, threshold: float) -> dict:
    proba = model.predict_proba(X)[:, 1]
    preds = (proba >= threshold).astype(int)
    roc   = roc_auc_score(y, proba)
    pr    = average_precision_score(y, proba)
    cm    = confusion_matrix(y, preds).tolist()
    rep   = classification_report(y, preds, output_dict=True)

    fire_key = next((k for k in ["1.0", "1"] if k in rep), None)
    fire_rep = rep.get(fire_key, {})

    fpr, tpr, _ = roc_curve(y, proba)
    prec, rec, _ = precision_recall_curve(y, proba)

    return {
        "name": name,
        "roc_auc":        round(roc, 4),
        "pr_auc":         round(pr, 4),
        "f1_fire":        round(fire_rep.get("f1-score", 0), 4),
        "recall_fire":    round(fire_rep.get("recall", 0), 4),
        "precision_fire": round(fire_rep.get("precision", 0), 4),
        "accuracy":       round(rep["accuracy"], 4),
        "confusion_matrix": cm,
        "roc_curve":  {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
        "pr_curve":   {"precision": prec.tolist(), "recall": rec.tolist()},
        "threshold":  threshold,
    }


# ── Optuna XGBoost tuning ──────────────────────────────────────────────────────
def optimize_xgboost(X_train, y_train, X_val, y_val, scale_weight, n_trials=40):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = dict(
            n_estimators=1000,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            max_depth=trial.suggest_int("max_depth", 4, 9),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 30),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            gamma=trial.suggest_float("gamma", 0.0, 2.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            scale_pos_weight=scale_weight,
            eval_metric="aucpr",
            random_state=42,
            n_jobs=-1,
        )
        model = xgb.XGBClassifier(**params, early_stopping_rounds=40)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        return average_precision_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  Best val PR-AUC: {study.best_value:.4f}  params: {study.best_params}")
    return study.best_params, study.best_value


def optimize_lightgbm(X_train, y_train, X_val, y_val, n_trials=40):
    import optuna
    from lightgbm.callback import early_stopping, log_evaluation
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial):
        params = dict(
            n_estimators=1000,
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            num_leaves=trial.suggest_int("num_leaves", 20, 150),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 50),
            subsample=trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.4, 1.0),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            min_split_gain=trial.suggest_float("min_split_gain", 0.0, 2.0),
            class_weight="balanced",
            random_state=42, n_jobs=-1, verbose=-1,
        )
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[early_stopping(40, verbose=False), log_evaluation(-1)],
        )
        return average_precision_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  Best val PR-AUC: {study.best_value:.4f}  params: {study.best_params}")
    return study.best_params, study.best_value


# ── Training ───────────────────────────────────────────────────────────────────
def train_classifiers(train, smote, val, test):
    feat_cols = get_feature_cols(train)
    print(f"Feature count: {len(feat_cols)}")

    X_train, y_train = train[feat_cols].astype(float), train[TARGET_CLF]
    X_val,   y_val   = val[feat_cols].astype(float),   val[TARGET_CLF]
    X_test,  y_test  = test[feat_cols].astype(float),  test[TARGET_CLF]

    # Impute NaNs with training medians (fit on train, apply to val/test)
    col_medians = X_train.median()
    X_train = X_train.fillna(col_medians)
    X_val   = X_val.fillna(col_medians)
    X_test  = X_test.fillna(col_medians)

    smote_feat = [c for c in feat_cols if c in smote.columns]
    X_sm, y_sm = smote[smote_feat].astype(float), smote[TARGET_CLF]
    X_sm = X_sm.fillna(X_sm.median())

    scale_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"Scale weight (neg/pos): {scale_weight:.1f}")

    # ── Optuna: tune XGBoost on val PR-AUC ─────────────────────────────────────
    print("\nRunning Optuna search for XGBoost (100 trials)...")
    best_xgb_params, _ = optimize_xgboost(
        X_train, y_train, X_val, y_val, scale_weight, n_trials=100
    )
    # Build final XGBoost with tuned params + early stopping
    xgb_final = xgb.XGBClassifier(
        **best_xgb_params,
        n_estimators=1000,
        early_stopping_rounds=40,
        scale_pos_weight=scale_weight,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )

    print("\nRunning Optuna search for LightGBM (100 trials)...")
    best_lgb_params, _ = optimize_lightgbm(X_train, y_train, X_val, y_val, n_trials=40)
    from lightgbm.callback import early_stopping as lgb_es, log_evaluation as lgb_log
    lgb_final = lgb.LGBMClassifier(
        **best_lgb_params,
        n_estimators=1000,
        class_weight="balanced",
        random_state=42, n_jobs=-1, verbose=-1,
    )

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight="balanced", max_iter=2000, C=0.1, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced",
            n_jobs=-1, random_state=42
        ),
        "XGBoost": xgb_final,
        "LightGBM": lgb_final,
        "LightGBM + SMOTE": lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=63,
            min_child_samples=20, colsample_bytree=0.8, subsample=0.8,
            random_state=42, n_jobs=-1, verbose=-1
        ),
    }

    results   = {}
    artifacts = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")
        if name == "LightGBM + SMOTE":
            model.fit(X_sm[smote_feat], y_sm)
            Xv, Xt = X_val[smote_feat], X_test[smote_feat]
        elif name == "XGBoost":
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            Xv, Xt = X_val, X_test
        elif name == "LightGBM":
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      callbacks=[lgb_es(40, verbose=False), lgb_log(-1)])
            Xv, Xt = X_val, X_test
        else:
            model.fit(X_train, y_train)
            Xv, Xt = X_val, X_test

        # Isotonic calibration on val set (prefit — model already trained)
        if name != "LightGBM + SMOTE":
            cal = IsotonicCalibrator(model)
            cal.fit(Xv, y_val)
            model = cal

        thresh = best_threshold_fbeta(model, Xv, y_val, beta=BETA)
        val_m  = eval_classifier(model, Xv, y_val,  name, thresh)
        test_m = eval_classifier(model, Xt, y_test, name, thresh)

        results[name]   = {"val": val_m, "test": test_m}
        artifacts[name] = model
        print(f"  Threshold: {thresh:.3f}")
        print(f"  Val  → ROC:{val_m['roc_auc']}  PR:{val_m['pr_auc']}  F1:{val_m['f1_fire']}  Recall:{val_m['recall_fire']}  Prec:{val_m['precision_fire']}")
        print(f"  Test → ROC:{test_m['roc_auc']} PR:{test_m['pr_auc']} F1:{test_m['f1_fire']} Recall:{test_m['recall_fire']} Prec:{test_m['precision_fire']}")

    # ── Ensemble: soft-vote XGBoost + LightGBM ────────────────────────────────
    print("\nBuilding Ensemble (XGBoost + LightGBM)...")
    ensemble = EnsembleClassifier([artifacts["XGBoost"], artifacts["LightGBM"]])
    thresh_ens = best_threshold_fbeta(ensemble, X_val, y_val, beta=BETA)
    val_ens    = eval_classifier(ensemble, X_val,  y_val,  "Ensemble", thresh_ens)
    test_ens   = eval_classifier(ensemble, X_test, y_test, "Ensemble", thresh_ens)
    results["Ensemble"]   = {"val": val_ens, "test": test_ens}
    artifacts["Ensemble"] = ensemble
    print(f"  Threshold: {thresh_ens:.3f}")
    print(f"  Val  → ROC:{val_ens['roc_auc']}  PR:{val_ens['pr_auc']}  F1:{val_ens['f1_fire']}  Recall:{val_ens['recall_fire']}  Prec:{val_ens['precision_fire']}")
    print(f"  Test → ROC:{test_ens['roc_auc']} PR:{test_ens['pr_auc']} F1:{test_ens['f1_fire']} Recall:{test_ens['recall_fire']} Prec:{test_ens['precision_fire']}")

    # Select best by val PR-AUC — exclude SMOTE models (miscalibrated probabilities)
    eligible  = {n: r for n, r in results.items() if "SMOTE" not in n}
    best_name = max(eligible, key=lambda n: eligible[n]["val"]["pr_auc"])
    print(f"\nBest model (by val PR-AUC, excl. SMOTE): {best_name}")

    for name, model in artifacts.items():
        safe = name.replace(" ", "_").replace("+", "plus")
        joblib.dump(model, OUT_DIR / f"{safe}.pkl")
    # Ensemble has no feature_importances_ — skip fi_data silently

    joblib.dump(artifacts[best_name], OUT_DIR / "best_model.pkl")
    joblib.dump(feat_cols, OUT_DIR / "feature_cols.pkl")

    fi_data = {}
    for name, model in artifacts.items():
        base = getattr(model, "estimator", model)  # unwrap CalibratedClassifierCV
        if hasattr(base, "feature_importances_"):
            cols = smote_feat if "SMOTE" in name else feat_cols
            fi = pd.Series(base.feature_importances_, index=cols)
            fi_data[name] = fi.nlargest(20).to_dict()

    return results, best_name, feat_cols, fi_data


# ── Regression ─────────────────────────────────────────────────────────────────
def train_regression(train, val, test):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    feat_cols = get_feature_cols(train)

    reg_feats = [c for c in feat_cols if c != TARGET_REG]

    fire_train = train[train[TARGET_CLF] == 1].copy()
    fire_val   = val[val[TARGET_CLF] == 1].copy()
    fire_test  = test[test[TARGET_CLF] == 1].copy()

    X_tr, y_tr = fire_train[reg_feats].astype(float), fire_train[TARGET_REG]
    X_v,  y_v  = fire_val[reg_feats].astype(float),   fire_val[TARGET_REG]
    X_te, y_te = fire_test[reg_feats].astype(float),  fire_test[TARGET_REG]

    # Impute NaNs with training medians
    tr_medians = X_tr.median()
    X_tr = X_tr.fillna(tr_medians)
    X_v  = X_v.fillna(tr_medians)
    X_te = X_te.fillna(tr_medians)

    print(f"\nRegression — fire-day counts: train={len(X_tr)}, val={len(X_v)}, test={len(X_te)}")

    # Log-transform: log1p compresses the heavy tail and makes squared loss well-behaved
    y_tr_log = np.log1p(y_tr)
    y_v_log  = np.log1p(y_v)

    model = xgb.XGBRegressor(
        n_estimators=400, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, n_jobs=-1
    )
    model.fit(X_tr, y_tr_log, eval_set=[(X_v, y_v_log)], verbose=False)

    log_preds = model.predict(X_te)
    preds     = np.expm1(log_preds)
    y_te_v    = y_te.values

    rmse   = np.sqrt(mean_squared_error(y_te_v, preds))
    mae    = mean_absolute_error(y_te_v, preds)
    r2     = r2_score(y_te_v, preds)
    r2_log = r2_score(np.log1p(y_te_v), log_preds)
    print(f"  RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.3f}  R²(log)={r2_log:.3f}")

    joblib.dump(model, OUT_DIR / "regression_model.pkl")
    joblib.dump(reg_feats, OUT_DIR / "reg_feature_cols.pkl")

    fi = pd.Series(model.feature_importances_, index=reg_feats).nlargest(20).to_dict()

    return {
        "rmse":    round(rmse, 2),
        "mae":     round(mae, 2),
        "r2":      round(r2, 3),
        "r2_log":  round(r2_log, 3),
        "feature_importance": fi,
        "actuals":     y_te_v.tolist()[:500],
        "predictions": preds.tolist()[:500],
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    # Lag features are pre-computed in rebuild_pipeline.py
    train, val, test, smote = load_splits()

    lag_cols = [c for c in train.columns if c in ["prev_day_fire", "prev2_day_fire", "fire_7d_rolling"]]
    if lag_cols:
        print(f"Lag features present: {lag_cols}")
        print(f"  prev_day_fire correlation: {train['prev_day_fire'].corr(train[TARGET_CLF]):.3f}")

    clf_results, best_name, feat_cols, fi_data = train_classifiers(train, smote, val, test)
    reg_results = train_regression(train, val, test)

    # Compute base fire rate from test confusion matrix
    cm_test = clf_results[best_name]["test"]["confusion_matrix"]
    total = sum(sum(r) for r in cm_test)
    fire_days = cm_test[1][0] + cm_test[1][1]

    output = {
        "classification":    clf_results,
        "best_model":        best_name,
        "feature_importance": fi_data,
        "regression":        reg_results,
        "feature_cols":      feat_cols,
        "counties":          [c.replace("county_", "") for c in feat_cols if c.startswith("county_")],
        "base_rate":         round(fire_days / total, 4),
    }

    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nAll artifacts saved to {OUT_DIR}")
    print(f"Best model: {best_name}")


if __name__ == "__main__":
    main()
