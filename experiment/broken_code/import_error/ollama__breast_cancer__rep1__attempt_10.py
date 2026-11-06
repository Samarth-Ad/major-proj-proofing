
"""
Breast‑cancer tabular classification pipeline
------------------------------------------------
* Loads the original CSV (Windows path shown – change if needed)
* Validates the raw file
* Stratified 70/20/10 train‑val‑test split
* Deterministic preprocessing pipeline
      median imputation → Winsor‑capping (1 % / 99 %) → Standard‑Scaler
      → optional Mutual‑Information feature selection (top‑20)
* SMOTETomek applied **only** on the training split (after scaling/selection)
* TabNet (pytorch‑tabnet) model with early‑stopping on the validation set
* Evaluation on the untouched test set (ROC‑AUC, accuracy, F1, sensitivity,
  specificity, precision, Brier score)
* Saves:
      - preprocessing pipeline   → ./agent_workspace/trained_models/preprocess.pkl
      - best TabNet weights      → ./agent_workspace/trained_models/tabnet_best.pt
      - test‑set metrics JSON    → ./agent_workspace/trained_models/test_metrics.json
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
from sklearn.pipeline import Pipeline
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
from torch.utils.data import TensorDataset, DataLoader
from pytorch_tabnet.tab_model import TabNetClassifier

# --------------------------------------------------------------------------- #
#                               CONFIGURATION                               #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# 1️⃣ Paths & constants (all built from the repository root)
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent  # folder that contains this script
RAW_CSV = Path(
    r"C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\breast_cancer.csv"
)  # <-- change if the CSV lives elsewhere
ARTIFACTS_DIR = BASE_DIR / "agent_workspace" / "trained_models"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# --------------------------------------------------------------------------- #
# 2️⃣ Hyper‑parameters
# --------------------------------------------------------------------------- #
SELECT_K = 20                     # number of features after MI‑selection
SMOTETOMEK_RANDOM_STATE = SEED
TABNET_PARAMS = {
    "n_d": 64,
    "n_a": 64,
    "n_steps": 5,
    "gamma": 1.5,
    "lambda_sparse": 1e-3,
    "optimizer_fn": torch.optim.Adam,
    "optimizer_params": dict(lr=2e-2),
    "mask_type": "entmax",  # default
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
        X = np.asarray(X)
        self.lower_ = np.quantile(X, self.lower_quantile, axis=0)
        self.upper_ = np.quantile(X, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X)
        return np.clip(X, self.lower_, self.upper_)


# --------------------------------------------------------------------------- #
# 4️⃣ Helper utilities
# --------------------------------------------------------------------------- #
def validate_raw_csv(df: pd.DataFrame) -> None:
    """Raise an informative error if the raw CSV does not meet expectations."""
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
        unique_targets = set(df["target"].unique())
        if unique_targets != {0, 1}:
            issues.append(f"'target' contains values {unique_targets}, expected only 0 and 1.")

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
        raise ValueError("Validation of raw CSV failed:\n" + "\n".join(issues))


def stratified_split(X: pd.DataFrame, y: pd.Series, train_size=0.7, val_size=0.2, test_size=0.1):
    """Return train/val/test indices respecting the class distribution."""
    assert np.isclose(train_size + val_size + test_size, 1.0), "Splits must sum to 1."

    # First split off the test set
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=SEED
    )
    train_val_idx, test_idx = next(sss1.split(X, y))

    X_train_val, y_train_val = X.iloc[train_val_idx], y.iloc[train_val_idx]

    # Now split train/val from the remaining part
    val_relative_size = val_size / (train_size + val_size)  # proportion of the remaining set
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_relative_size, random_state=SEED
    )
    train_idx, val_idx = next(sss2.split(X_train_val, y_train_val))

    # Convert to original indices
    train_idx = train_val_idx[train_idx]
    val_idx = train_val_idx[val_idx]

    return (
        X.iloc[train_idx],
        X.iloc[val_idx],
        X.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[val_idx],
        y.iloc[test_idx],
    )


def build_preprocess_pipeline():
    """Create the deterministic sklearn pipeline."""
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("winsorizer", Winsorizer(lower_quantile=0.01, upper_quantile=0.99)),
        ("scaler", StandardScaler()),
    ]

    # Mutual‑information selector (optional – we always keep it)
    steps.append(
        ("selector", SelectKBest(score_func=mutual_info_classif, k=SELECT_K))
    )

    return Pipeline(steps)


def torch_dataset_from_numpy(X: np.ndarray, y: np.ndarray) -> DataLoader:
    """Wrap NumPy arrays into a PyTorch DataLoader."""
    tensor_x = torch.from_numpy(X.astype(np.float32))
    tensor_y = torch.from_numpy(y.astype(np.int64))
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# --------------------------------------------------------------------------- #
# 5️⃣ Main pipeline
# --------------------------------------------------------------------------- #
def main():
    # ------------------------------------------------- #
    # 5.1 Load & validate raw CSV
    # ------------------------------------------------- #
    raw_df = pd.read_csv(RAW_CSV)
    validate_raw_csv(raw_df)

    # ------------------------------------------------- #
    # 5.2 Separate features / target
    # ------------------------------------------------- #
    feature_cols = [c for c in raw_df.columns if c != "target"]
    X = raw_df[feature_cols]
    y = raw_df["target"]

    # ------------------------------------------------- #
    # 5.3 Stratified train/val/test split (70/20/10)
    # ------------------------------------------------- #
    X_train, X_val, X_test, y_train, y_val, y_test = stratified_split(
        X, y, train_size=0.7, val_size=0.2, test_size=0.1
    )

    # ------------------------------------------------- #
    # 5.4 Build & fit preprocessing pipeline on training data
    # ------------------------------------------------- #
    preprocess_pipe = build_preprocess_pipeline()
    preprocess_pipe.fit(X_train, y_train)  # y needed for SelectKBest

    # Persist the pipeline for later inference
    pipeline_path = ARTIFACTS_DIR / "preprocess.pkl"
    joblib.dump(preprocess_pipe, pipeline_path)

    # ------------------------------------------------- #
    # 5.5 Transform all splits
    # ------------------------------------------------- #
    X_train_proc = preprocess_pipe.transform(X_train)
    X_val_proc = preprocess_pipe.transform(X_val)
    X_test_proc = preprocess_pipe.transform(X_test)

    # ------------------------------------------------- #
    # 5.6 SMOTETomek on the *processed* training data only
    # ------------------------------------------------- #
    smt = SMOTETomek(random_state=SMOTETOMEK_RANDOM_STATE)
    X_res, y_res = smt.fit_resample(X_train_proc, y_train)

    # ------------------------------------------------- #
    # 5.7 Prepare PyTorch DataLoaders
    # ------------------------------------------------- #
    train_loader = torch_dataset_from_numpy(X_res, y_res.values)
    val_loader = torch_dataset_from_numpy(X_val_proc, y_val.values)
    test_loader = torch_dataset_from_numpy(X_test_proc, y_test.values)

    # ------------------------------------------------- #
    # 5.8 Initialise TabNet model
    # ------------------------------------------------- #
    tabnet = TabNetClassifier(**TABNET_PARAMS)

    # ------------------------------------------------- #
    # 5.9 Train with early stopping on validation set
    # ------------------------------------------------- #
    start_time = time.time()
    tabnet.fit(
        X_train=X_res,
        y_train=y_res.values,
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
    train_duration = time.time() - start_time

    # ------------------------------------------------- #
    # 5.10 Save best model weights
    # ------------------------------------------------- #
    model_path = ARTIFACTS_DIR / "tabnet_best.pt"
    torch.save(tabnet.state_dict(), model_path)

    # ------------------------------------------------- #
    # 5.11 Evaluation on the untouched test set
    # ------------------------------------------------- #
    # Load a fresh instance with the same architecture
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
        "sensitivity": round(
            recall_score(y_test, y_pred, pos_label=1), 5
        ),  # TP rate for malignant
        "specificity": round(
            recall_score(y_test, y_pred, pos_label=0), 5
        ),  # TN rate for benign
        "precision": round(precision_score(y_test, y_pred), 5),
        "brier_score": round(brier_score_loss(y_test, probs), 5),
        "train_time_sec": round(train_duration, 2),
        "n_parameters": sum(p.numel() for p in tabnet.parameters()),
    }

    # ------------------------------------------------- #
    # 5.12 Persist test‑set metrics
    # ------------------------------------------------- #
    metrics_path = ARTIFACTS_DIR / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    # ------------------------------------------------- #
    # 5.13 Print a short summary (optional)
    # ------------------------------------------------- #
    print("\n=== Test‑set performance ===")
    for k, v in metrics.items():
        print(f"{k:15}: {v}")

    print(f"\nArtifacts saved in: {ARTIFACTS_DIR}")

    # Return objects for interactive notebooks (optional)
    return {
        "preprocess_pipeline": preprocess_pipe,
        "tabnet_model": tabnet,
        "test_metrics": metrics,
    }


if __name__ == "__main__":
    # Suppress harmless warnings from imbalanced‑learn / sklearn
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
