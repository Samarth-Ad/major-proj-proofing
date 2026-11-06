
"""
Breast‑cancer Tabular AutoML pipeline
-------------------------------------

This single‑file script implements the complete end‑to‑end workflow described in the
blueprint:

1. Load & validate the raw CSV (Wisconsin Breast Cancer dataset).
2. Stratified split 70 % train / 20 % validation / 10 % test.
3. Deterministic preprocessing pipeline:
   - Median imputation
   - Winsor‑capping (1 % / 99 %)
   - Standard‑scaler
   - Mutual‑information feature selection (k = 20)
4. SMOTETomek oversampling *only* on the training split.
5. TabNet classifier (PyTorch‑TabNet) with early stopping.
6. Evaluation on the untouched test set (ROC‑AUC, accuracy, F1, sensitivity,
   specificity, precision, Brier score).
7. Persist the preprocessing pipeline, the best TabNet weights and a JSON
   report with the test metrics.

All artefacts are written under ``./agent_workspace/trained_models`` so that the
model can be later loaded for deployment (e.g. FastAPI, Gradio, etc.).

Run the script from the repository root after installing the dependencies
listed in ``requirements.txt`` (see the blueprint).

Author: AutoML‑Agent
"""

import os
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    recall_score,
    precision_score,
    brier_score_loss,
)
from imblearn.combine import SMOTETomek
import torch
from pytorch_tabnet.tab_model import TabNetClassifier

# --------------------------------------------------------------------------- #
# Global constants & paths
# --------------------------------------------------------------------------- #
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
RAW_CSV = BASE_DIR / "data" / "raw" / "breast_cancer.csv"
ARTIFACTS_DIR = BASE_DIR / "agent_workspace" / "trained_models"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

PIPELINE_PATH = ARTIFACTS_DIR / "preprocess.pkl"
MODEL_PATH = ARTIFACTS_DIR / "tabnet_best.pt"
METRICS_PATH = ARTIFACTS_DIR / "test_metrics.json"

SELECT_K = 20  # number of features after MI selection

# --------------------------------------------------------------------------- #
# Helper: custom Winsorizer transformer
# --------------------------------------------------------------------------- #
class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip values to lower/upper quantiles computed on the training data."""

    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.lower_ = np.quantile(X, self.lower_quantile, axis=0)
        self.upper_ = np.quantile(X, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X)
        return np.clip(X, self.lower_, self.upper_)


# --------------------------------------------------------------------------- #
# Step 1 – Load & validate raw CSV
# --------------------------------------------------------------------------- #
def load_and_validate(csv_path: Path) -> pd.DataFrame:
    """Load the CSV and run a series of sanity checks."""
    df = pd.read_csv(csv_path)

    issues = []

    # Expected shape for the Wisconsin Breast Cancer dataset
    if df.shape[0] != 569:
        issues.append(f"Unexpected row count: {df.shape[0]} (expected 569)")
    if df.shape[1] != 31:
        issues.append(f"Unexpected column count: {df.shape[1]} (expected 31)")

    # Target column
    if "target" not in df.columns:
        issues.append("Missing required column `target`")
    else:
        if not set(df["target"].unique()).issubset({0, 1}):
            issues.append("`target` must contain only 0 and 1")

    # Numeric feature check (all columns except target)
    numeric_cols = [c for c in df.columns if c != "target"]
    for col in numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(f"Column `{col}` is not numeric")

    # Duplicates & missing values
    if df.duplicated().any():
        issues.append("Duplicate rows detected")
    if df.isna().sum().sum() > 0:
        issues.append("Missing values detected")

    if issues:
        raise ValueError("Data validation failed:\n" + "\n".join(issues))

    return df


# --------------------------------------------------------------------------- #
# Step 2 – Stratified split (70/20/10)
# --------------------------------------------------------------------------- #
def stratified_split(df: pd.DataFrame):
    """Return X_train, X_val, X_test, y_train, y_val, y_test."""
    X = df.drop(columns=["target"]).values
    y = df["target"].values

    # First split 70% train vs 30% temp (val+test)
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=0.30, random_state=SEED
    )
    train_idx, temp_idx = next(sss1.split(X, y))
    X_train, y_train = X[train_idx], y[train_idx]
    X_temp, y_temp = X[temp_idx], y[temp_idx]

    # Then split temp into 20% val / 10% test → proportion 2/3 vs 1/3 of temp
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=1 / 3, random_state=SEED
    )
    val_idx, test_idx = next(sss2.split(X_temp, y_temp))
    X_val, y_val = X_temp[val_idx], y_temp[val_idx]
    X_test, y_test = X_temp[test_idx], y_temp[test_idx]

    return X_train, X_val, X_test, y_train, y_val, y_test


# --------------------------------------------------------------------------- #
# Step 3 – Build preprocessing pipeline
# --------------------------------------------------------------------------- #
def build_preprocess_pipeline():
    pipeline = joblib.pipeline.Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("winsor", Winsorizer(lower_quantile=0.01, upper_quantile=0.99)),
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=mutual_info_classif, k=SELECT_K)),
        ]
    )
    return pipeline


# --------------------------------------------------------------------------- #
# Step 4 – Apply SMOTETomek on the *processed* training data
# --------------------------------------------------------------------------- #
def apply_smotetomek(X, y):
    sampler = SMOTETomek(random_state=SEED)
    X_res, y_res = sampler.fit_resample(X, y)
    return X_res, y_res


# --------------------------------------------------------------------------- #
# Step 5 – Train TabNet
# --------------------------------------------------------------------------- #
def train_tabnet(X_train, y_train, X_val, y_val):
    # Hyper‑parameters (fixed, reproducible)
    tabnet_params = {
        "n_d": 48,
        "n_a": 48,
        "n_steps": 6,
        "gamma": 1.5,
        "lambda_sparse": 1e-3,
        "optimizer_fn": torch.optim.Adam,
        "optimizer_params": {"lr": 8e-4},
        "mask_type": "sparsemax",
        "device_name": "cpu",  # enforce CPU for reproducibility
        "seed": SEED,
        "verbose": 0,
    }

    model = TabNetClassifier(**tabnet_params)

    start = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_name=["val"],
        eval_metric=["auc"],
        max_epochs=200,
        patience=20,
        batch_size=128,
        virtual_batch_size=64,
        num_workers=0,
        drop_last=False,
    )
    train_time = time.time() - start

    # Save best weights
    torch.save(model.state_dict(), MODEL_PATH)

    # Log simple info
    print(f"Training finished in {train_time:.2f}s")
    print(f"Best validation AUC: {model.best_score:.5f}")

    return model, train_time


# --------------------------------------------------------------------------- #
# Step 6 – Evaluation on the untouched test set
# --------------------------------------------------------------------------- #
def evaluate(model, pipeline, X_test, y_test):
    # Transform test data with the *same* pipeline (already fitted)
    X_test_proc = pipeline.transform(X_test)

    # Predict probabilities
    probs = model.predict_proba(X_test_proc)[:, 1]
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "roc_auc": round(roc_auc_score(y_test, probs), 5),
        "accuracy": round(accuracy_score(y_test, preds), 5),
        "f1": round(f1_score(y_test, preds), 5),
        "sensitivity": round(
            recall_score(y_test, preds, pos_label=1), 5
        ),  # TP / (TP+FN)
        "specificity": round(
            recall_score(y_test, preds, pos_label=0), 5
        ),  # TN / (TN+FP)
        "precision": round(precision_score(y_test, preds), 5),
        "brier_score": round(brier_score_loss(y_test, probs), 5),
    }

    # Persist metrics
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nTest‑set metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    return metrics


# --------------------------------------------------------------------------- #
# Main orchestration
# --------------------------------------------------------------------------- #
def main():
    # 1️⃣ Load & validate
    print("Loading and validating raw data...")
    df = load_and_validate(RAW_CSV)

    # 2️⃣ Stratified split
    print("Performing stratified split (70/20/10)...")
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(df)

    # 3️⃣ Build & fit preprocessing pipeline on training data
    print("Building preprocessing pipeline...")
    pipeline = build_preprocess_pipeline()
    pipeline.fit(X_train, y_train)  # selector needs y

    # Persist pipeline
    joblib.dump(pipeline, PIPELINE_PATH)
    print(f"Preprocessing pipeline saved to {PIPELINE_PATH}")

    # 4️⃣ Transform splits
    print("Transforming data...")
    X_train_proc = pipeline.transform(X_train)
    X_val_proc = pipeline.transform(X_val)
    X_test_proc = pipeline.transform(X_test)

    # 5️⃣ SMOTETomek on training data only
    print("Applying SMOTETomek to the training set...")
    X_train_bal, y_train_bal = apply_smotetomek(X_train_proc, y_train)
    print(
        f"Training set size after resampling: {X_train_bal.shape[0]} samples"
    )

    # 6️⃣ Train TabNet
    print("Training TabNet classifier...")
    model, train_time = train_tabnet(
        X_train_bal, y_train_bal, X_val_proc, y_val
    )

    # 7️⃣ Evaluate on test set
    print("Evaluating on the held‑out test set...")
    test_metrics = evaluate(model, pipeline, X_test, y_test)

    # Summary output
    summary = {
        "train_time_sec": round(train_time, 2),
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "test_metrics": test_metrics,
        "pipeline_path": str(PIPELINE_PATH),
        "model_path": str(MODEL_PATH),
    }

    # Save a tiny summary JSON for quick reference
    summary_path = ARTIFACTS_DIR / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll artefacts stored under {ARTIFACTS_DIR}")
    print(f"Run summary written to {summary_path}")


if __name__ == "__main__":
    # Suppress harmless warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
