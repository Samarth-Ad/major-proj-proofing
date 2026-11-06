
"""
Breast‑cancer tabular classification pipeline
============================================

This script is a **stand‑alone** end‑to‑end solution that:

1. Retrieves the CSV (copies it from the original Windows location if it is missing).
2. Validates the raw file (row/column count, target values, numeric types,
   no NaNs, no duplicate rows).
3. Performs a stratified 70 % train / 20 % validation / 10 % test split.
4. Builds a deterministic Scikit‑Learn preprocessing pipeline:
       median imputation → Winsor‑capping (1 % / 99 %) → Standard‑Scaler
       → Mutual‑Information feature selection (top‑20).
5. Applies SMOTETomek **only** on the processed training split.
6. Trains a TabNet classifier (pytorch‑tabnet) with early stopping on the
   validation set.
7. Evaluates the untouched test set (ROC‑AUC, accuracy, F1, sensitivity,
   specificity, precision, Brier score).
8. Persists the preprocessing pipeline, the best TabNet weights and a JSON
   file with test‑set metrics under ``./agent_workspace/trained_models``.

All paths are built relative to the repository root, so the script can be
run from any working directory.  The script also tries to install the
missing ``pytorch‑tabnet`` package automatically (only if it is not already
available) – this makes the script runnable on a fresh environment.
"""

# --------------------------------------------------------------------------- #
# Imports & global configuration
# --------------------------------------------------------------------------- #
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from imblearn.combine import SMOTETomek
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Helper: install missing pip packages at runtime (only when needed)
# --------------------------------------------------------------------------- #
def _install_and_import(package_name: str, import_name: str = None):
    """
    Try to import *import_name* (or *package_name* if import_name is None).
    If the import fails, install the package via pip and retry.
    """
    import importlib
    import_name = import_name or package_name
    try:
        return importlib.import_module(import_name)
    except ModuleNotFoundError:
        print(f"[INFO] Installing missing package: {package_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        return importlib.import_module(import_name)

# Ensure pytorch‑tabnet is available
pytorch_tabnet = _install_and_import("pytorch-tabnet", "pytorch_tabnet")
from pytorch_tabnet.tab_model import TabNetClassifier

# --------------------------------------------------------------------------- #
# 1️⃣ Paths & constants (all built from the repository root)
# --------------------------------------------------------------------------- #
# Repository root = three levels up from this file (adjust if you move the script)
BASE_DIR = Path(__file__).resolve().parents[2]          # <repo_root>
RAW_CSV = BASE_DIR / "data" / "raw" / "breast_cancer.csv"

# Directory where we store artefacts (pipeline, model, metrics)
ARTIFACTS_DIR = BASE_DIR / "agent_workspace" / "trained_models"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# --------------------------------------------------------------------------- #
# 2️⃣ Hyper‑parameters
# --------------------------------------------------------------------------- #
SELECT_K = 20                     # top‑k features after MI‑selection
SMOTETOMEK_RANDOM_STATE = SEED

TABNET_PARAMS = {
    "n_d": 64,
    "n_a": 64,
    "n_steps": 5,
    "gamma": 1.5,
    "lambda_sparse": 1e-3,
    "optimizer_fn": torch.optim.Adam,
    "optimizer_params": dict(lr=2e-2),
    "mask_type": "entmax",
    "verbose": 0,
}
MAX_EPOCHS = 200
PATIENCE = 20
BATCH_SIZE = 128
VIRTUAL_BATCH_SIZE = 64

# --------------------------------------------------------------------------- #
# 3️⃣ Custom transformer – Winsorizer (1 % / 99 %)
# --------------------------------------------------------------------------- #
class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each feature to the 1st and 99th percentiles computed on the fit data."""
    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.lower_ = np.quantile(X, self.lower_quantile, axis=0)
        self.upper_ = np.quantile(X, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.clip(X, self.lower_, self.upper_)

# --------------------------------------------------------------------------- #
# 4️⃣ Helper utilities
# --------------------------------------------------------------------------- #
def _maybe_copy_from_windows():
    """
    If the expected CSV does not exist, try to copy it from the original
    Windows location that the user mentioned.
    """
    if RAW_CSV.is_file():
        return
    win_path = Path(
        r"C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\breast_cancer.csv"
    )
    if win_path.is_file():
        RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
        RAW_CSV.write_bytes(win_path.read_bytes())
        print(f"[INFO] Copied CSV from Windows location to {RAW_CSV}")
    else:
        raise FileNotFoundError(
            f"CSV not found at expected location: {RAW_CSV}\n"
            f"Also could not locate the Windows source file at {win_path}."
        )

def validate_raw_csv(df: pd.DataFrame) -> None:
    """Raise a ValueError if the raw CSV does not meet expectations."""
    issues = []

    # Expected shape
    if df.shape[0] != 569:
        issues.append(f"Row count is {df.shape[0]}, expected 569.")
    if df.shape[1] != 31:
        issues.append(f"Column count is {df.shape[1]}, expected 31 (30 features + target).")

    # Target column
    if "target" not in df.columns:
        issues.append("Missing required column 'target'.")
    else:
        uniq = set(df["target"].unique())
        if uniq != {0, 1}:
            issues.append(f"'target' contains values {uniq}, expected only 0 and 1.")

    # Numeric features
    feature_cols = [c for c in df.columns if c != "target"]
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(f"Feature column '{col}' is not numeric.")

    # Duplicates & missing values
    if df.duplicated().any():
        issues.append("Duplicate rows detected.")
    if df.isna().any().any():
        issues.append("Missing values detected.")

    if issues:
        raise ValueError("Validation errors in raw CSV:\n" + "\n".join(issues))

def stratified_split(X: pd.DataFrame, y: pd.Series,
                    train_size=0.7, val_size=0.2, test_size=0.1):
    """Return train/val/test DataFrames respecting class distribution."""
    assert np.isclose(train_size + val_size + test_size, 1.0), "Splits must sum to 1."

    # First split off the test set
    sss_test = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=SEED
    )
    train_val_idx, test_idx = next(sss_test.split(X, y))

    X_train_val, y_train_val = X.iloc[train_val_idx], y.iloc[train_val_idx]

    # Now split train/val from the remaining part
    val_rel = val_size / (train_size + val_size)   # proportion of the remaining set
    sss_val = StratifiedShuffleSplit(
        n_splits=1, test_size=val_rel, random_state=SEED
    )
    train_idx, val_idx = next(sss_val.split(X_train_val, y_train_val))

    # Map back to original indices
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]

    return (X.iloc[train_idx], X.iloc[val_idx], X.iloc[test_idx],
            y.iloc[train_idx], y.iloc[val_idx], y.iloc[test_idx])

def build_preprocess_pipeline():
    """Construct the deterministic sklearn pipeline."""
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("winsorizer", Winsorizer(lower_quantile=0.01, upper_quantile=0.99)),
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(score_func=mutual_info_classif, k=SELECT_K)),
    ]
    return Pipeline(steps)

# --------------------------------------------------------------------------- #
# 5️⃣ Main pipeline
# --------------------------------------------------------------------------- #
def main():
    # ------------------------------------------------- #
    # 5.1 Ensure CSV is present (copy from Windows if needed)
    # ------------------------------------------------- #
    _maybe_copy_from_windows()

    # ------------------------------------------------- #
    # 5.2 Load & validate raw CSV
    # ------------------------------------------------- #
    raw_df = pd.read_csv(RAW_CSV)
    validate_raw_csv(raw_df)

    # ------------------------------------------------- #
    # 5.3 Separate features / target
    # ------------------------------------------------- #
    feature_cols = [c for c in raw_df.columns if c != "target"]
    X = raw_df[feature_cols]
    y = raw_df["target"]

    # ------------------------------------------------- #
    # 5.4 Stratified train/val/test split (70/20/10)
    # ------------------------------------------------- #
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
        X, y, train_size=0.7, val_size=0.2, test_size=0.1
    )

    # ------------------------------------------------- #
    # 5.5 Build & fit preprocessing pipeline on training data
    # ------------------------------------------------- #
    preprocess_pipe = build_preprocess_pipeline()
    preprocess_pipe.fit(X_train, y_train)          # y needed for SelectKBest

    # Persist the pipeline for later inference
    pipeline_path = ARTIFACTS_DIR / "preprocess.pkl"
    joblib.dump(preprocess_pipe, pipeline_path)

    # ------------------------------------------------- #
    # 5.6 Transform all splits
    # ------------------------------------------------- #
    X_train_proc = preprocess_pipe.transform(X_train)
    X_val_proc   = preprocess_pipe.transform(X_val)
    X_test_proc  = preprocess_pipe.transform(X_test)

    # ------------------------------------------------- #
    # 5.7 SMOTETomek on the *processed* training data only
    # ------------------------------------------------- #
    smt = SMOTETomek(random_state=SMOTETOMEK_RANDOM_STATE)
    X_res, y_res = smt.fit_resample(X_train_proc, y_train)

    # ------------------------------------------------- #
    # 5.8 Initialise TabNet model
    # ------------------------------------------------- #
    tabnet = TabNetClassifier(**TABNET_PARAMS)

    # ------------------------------------------------- #
    # 5.9 Train with early stopping on validation set
    # ------------------------------------------------- #
    start = time.time()
    tabnet.fit(
        X_train=X_res,
        y_train=y_res,
        eval_set=[(X_val_proc, y_val.values)],
        eval_name=["val"],
        eval_metric=["auc"],
        max_epochs=MAX_EPOCHS,
        patience=PATIENCE,
        batch_size=BATCH_SIZE,
        virtual_batch_size=VIRTUAL_BATCH_SIZE,
        num_workers=0,
        drop_last=False,
    )
    train_time = time.time() - start

    # ------------------------------------------------- #
    # 5.10 Save best model weights
    # ------------------------------------------------- #
    model_path = ARTIFACTS_DIR / "tabnet_best.pt"
    torch.save(tabnet.state_dict(), model_path)

    # ------------------------------------------------- #
    # 5.11 Evaluation on the untouched test set
    # ------------------------------------------------- #
    # Load a fresh instance with identical architecture
    eval_model = TabNetClassifier(**TABNET_PARAMS)
    eval_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    eval_model.eval()

    # Predict probabilities for class 1 (malignant)
    with torch.no_grad():
        test_tensor = torch.from_numpy(X_test_proc.astype(np.float32))
        probs = eval_model.predict_proba(test_tensor.numpy())[:, 1]

    y_pred = (probs >= 0.5).astype(int)

    # Compute metrics
    metrics = {
        "roc_auc": round(roc_auc_score(y_test, probs), 5),
        "accuracy": round(accuracy_score(y_test, y_pred), 5),
        "f1": round(f1_score(y_test, y_pred), 5),
        "sensitivity": round(recall_score(y_test, y_pred, pos_label=1), 5),
        "specificity": round(recall_score(y_test, y_pred, pos_label=0), 5),
        "precision": round(precision_score(y_test, y_pred), 5),
        "brier_score": round(brier_score_loss(y_test, probs), 5),
        "train_time_sec": round(train_time, 2),
        "n_parameters": sum(p.numel() for p in tabnet.parameters()),
    }

    # ------------------------------------------------- #
    # 5.12 Persist test‑set metrics
    # ------------------------------------------------- #
    metrics_path = ARTIFACTS_DIR / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # ------------------------------------------------- #
    # 5.13 Print a short summary for the user
    # ------------------------------------------------- #
    print("\n=== Test‑set performance ===")
    for k, v in metrics.items():
        print(f"{k:15}: {v}")

    print(f"\nArtifacts saved in: {ARTIFACTS_DIR}")

    # Return a dictionary for interactive notebooks (optional)
    return {
        "preprocess_pipeline": preprocess_pipe,
        "tabnet_model": tabnet,
        "test_metrics": metrics,
    }

# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # Suppress non‑critical warnings (e.g., from imbalanced‑learn)
    warnings.filterwarnings("ignore", category=UserWarning)
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)
