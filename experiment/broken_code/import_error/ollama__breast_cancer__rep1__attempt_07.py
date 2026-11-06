
"""
End‑to‑end Tabular Classification pipeline for the Breast Cancer dataset.

The script performs:
1. Load & validate the raw CSV.
2. Stratified split (70% train / 20% validation / 10% test).
3. Deterministic preprocessing pipeline:
   - Median imputation
   - Winsorising (1st / 99th percentiles)
   - Standard scaling
   - Optional Mutual‑Information feature selection (top 20)
4. SMOTETomek balancing (training set only).
5. TabNet training with early stopping.
6. Evaluation on the untouched test set.
7. Persist the preprocessing pipeline and the trained TabNet model under
   "./agent_workspace/trained_models".
8. Write a JSON report with test‑set metrics.

All log messages contain only ASCII characters.
"""

# ------------------------------------------------------------------- #
# Imports
# ------------------------------------------------------------------- #
import json
import logging
import time
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------- #
# TabNet import – the package must be installed *before* this script runs.
# If you get a ModuleNotFoundError, install it with:
#   pip install pytorch-tabnet
# ------------------------------------------------------------------- #
from pytorch_tabnet.tab_model import TabNetClassifier

# ------------------------------------------------------------------- #
# Logging configuration (ASCII only)
# ------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format='{"time":"%(asctime)s","level":"%(levelname)s","msg":"%(message)s"}',
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------- #
# Global constants – absolute‑from‑repo paths
# ------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent          # repository root
RAW_CSV = BASE_DIR / "data" / "breast_cancer.csv"
ARTIFACTS_DIR = BASE_DIR / "agent_workspace" / "trained_models"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
SELECT_K = 20          # set to None to skip feature selection

# ------------------------------------------------------------------- #
# Helper – custom Winsorizer (TransformerMixin)
# ------------------------------------------------------------------- #
class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each feature to the lower and upper quantiles computed at fit time."""

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


# ------------------------------------------------------------------- #
# Step 1 – Validate raw CSV
# ------------------------------------------------------------------- #
def validate_raw_csv(csv_path: Path) -> pd.DataFrame:
    logger.info(f"Loading raw CSV from {csv_path}")
    df = pd.read_csv(csv_path)

    issues = []

    # Expected shape for the classic Breast Cancer Wisconsin dataset
    if df.shape[0] != 569:
        issues.append(f"Unexpected row count: {df.shape[0]} (expected 569)")
    if df.shape[1] != 31:
        issues.append(f"Unexpected column count: {df.shape[1]} (expected 31)")

    # Target column
    if "target" not in df.columns:
        issues.append("Missing required column 'target'")
    else:
        uniq = set(df["target"].unique())
        if uniq != {0, 1}:
            issues.append(f"'target' must contain only 0/1, found {uniq}")

    # Numeric feature check (all columns except target)
    feature_cols = [c for c in df.columns if c != "target"]
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(f"Feature column '{col}' is not numeric")

    # Duplicates
    if df.duplicated().any():
        issues.append("Duplicate rows detected")

    # Missing values
    if df.isna().sum().sum() > 0:
        issues.append("Missing values found in the dataset")

    if issues:
        logger.error("Validation failed: " + "; ".join(issues))
        raise ValueError("Raw CSV validation failed")
    logger.info("Raw CSV validation passed")
    return df


# ------------------------------------------------------------------- #
# Step 2 – Stratified split (70/20/10)
# ------------------------------------------------------------------- #
def stratified_split(df: pd.DataFrame, seed: int = SEED):
    X = df.drop(columns=["target"])
    y = df["target"].values

    # 70% train, 30% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    # From temp (30%) -> 2/3 val (20% total) and 1/3 test (10% total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 / 3, stratify=y_temp, random_state=seed
    )

    logger.info(
        f"Split sizes -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}"
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


# ------------------------------------------------------------------- #
# Step 3 – Build preprocessing pipeline
# ------------------------------------------------------------------- #
def build_preprocess_pipeline(select_k: int = SELECT_K) -> Pipeline:
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("winsor", Winsorizer(lower_quantile=0.01, upper_quantile=0.99)),
        ("scaler", StandardScaler()),
    ]

    if select_k is not None and select_k > 0:
        steps.append(
            ("selectk", SelectKBest(score_func=mutual_info_classif, k=select_k))
        )

    pipeline = Pipeline(steps)
    return pipeline


# ------------------------------------------------------------------- #
# Step 4 – Apply SMOTETomek on training data (after preprocessing)
# ------------------------------------------------------------------- #
def balance_training_data(X: np.ndarray, y: np.ndarray):
    logger.info("Applying SMOTETomek to balance the training set")
    sampler = SMOTETomek(random_state=SEED)
    X_res, y_res = sampler.fit_resample(X, y)
    logger.info(
        f"After SMOTETomek -> samples: {X_res.shape[0]}, class distribution: {np.bincount(y_res)}"
    )
    return X_res, y_res


# ------------------------------------------------------------------- #
# Step 5 – Train TabNet
# ------------------------------------------------------------------- #
def train_tabnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_epochs: int = 200,
    patience: int = 20,
):
    logger.info("Initializing TabNet classifier")
    clf = TabNetClassifier(
        n_d=48,
        n_a=48,
        n_steps=6,
        gamma=1.5,
        lambda_sparse=1e-3,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=8e-4),
        mask_type="sparsemax",
        device_name="cpu",
        seed=SEED,
    )

    logger.info("Starting TabNet training")
    start = time.time()
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_name=["val"],
        eval_metric=["auc"],
        max_epochs=max_epochs,
        patience=patience,
        batch_size=128,
        virtual_batch_size=64,
        drop_last=False,
        verbose=0,
    )
    duration = time.time() - start
    logger.info(f"Training completed in {duration:.2f} seconds")
    return clf, duration


# ------------------------------------------------------------------- #
# Step 6 – Evaluation on test set
# ------------------------------------------------------------------- #
def evaluate_model(model: TabNetClassifier, X_test: np.ndarray, y_test: np.ndarray):
    logger.info("Running inference on the test set")
    probs = model.predict_proba(X_test)[:, 1]          # probability of class 1 (malignant)
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "roc_auc": round(roc_auc_score(y_test, probs), 5),
        "accuracy": round(accuracy_score(y_test, preds), 5),
        "f1": round(f1_score(y_test, preds), 5),
        "sensitivity": round(recall_score(y_test, preds, pos_label=1), 5),
        "specificity": round(recall_score(y_test, preds, pos_label=0), 5),
        "precision": round(precision_score(y_test, preds, zero_division=0), 5),
        "brier_score": round(brier_score_loss(y_test, probs), 5),
    }

    logger.info("Test metrics: " + json.dumps(metrics))
    return metrics


# ------------------------------------------------------------------- #
# Step 7 – Persist artefacts
# ------------------------------------------------------------------- #
def persist_artifacts(pipeline: Pipeline, model: TabNetClassifier):
    pipeline_path = ARTIFACTS_DIR / "preprocess.pkl"
    model_path = ARTIFACTS_DIR / "tabnet_best.pt"

    logger.info(f"Saving preprocessing pipeline to {pipeline_path}")
    joblib.dump(pipeline, pipeline_path)

    logger.info(f"Saving TabNet state dict to {model_path}")
    torch.save(model.state_dict(), model_path)

    logger.info("All artefacts saved")
    return pipeline_path, model_path


# ------------------------------------------------------------------- #
# Main orchestration
# ------------------------------------------------------------------- #
def main():
    # 1. Load & validate raw data
    df = validate_raw_csv(RAW_CSV)

    # 2. Train / validation / test split
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test = stratified_split(df)

    # 3. Build and fit preprocessing pipeline on the training data
    preprocess = build_preprocess_pipeline()
    preprocess.fit(X_train_raw, y_train)

    # 4. Transform all splits
    X_train_proc = preprocess.transform(X_train_raw)
    X_val_proc = preprocess.transform(X_val_raw)
    X_test_proc = preprocess.transform(X_test_raw)

    # 5. Balance training data with SMOTETomek
    X_train_bal, y_train_bal = balance_training_data(X_train_proc, y_train)

    # Ensure correct dtypes for TabNet (float32 for X, int64 for y)
    X_train_bal = X_train_bal.astype(np.float32)
    y_train_bal = y_train_bal.astype(np.int64)
    X_val_proc = X_val_proc.astype(np.float32)
    y_val = y_val.astype(np.int64)
    X_test_proc = X_test_proc.astype(np.float32)
    y_test = y_test.astype(np.int64)

    # 6. Train TabNet
    model, train_time = train_tabnet(
        X_train_bal, y_train_bal, X_val_proc, y_val, max_epochs=200, patience=20
    )

    # 7. Evaluate on test set
    test_metrics = evaluate_model(model, X_test_proc, y_test)

    # 8. Persist artefacts
    persist_artifacts(preprocess, model)

    # 9. Write JSON report
    report_path = ARTIFACTS_DIR / "test_metrics.json"
    with open(report_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info(f"Test metrics written to {report_path}")

    # 10. Console summary
    print("\n=== Test set performance ===")
    for k, v in test_metrics.items():
        print(f"{k}: {v}")
    print(f"Training time (seconds): {train_time:.2f}")
    print(f"\nArtifacts saved under: {ARTIFACTS_DIR}")
    print("Use `predict_single(sample_dict)` for ad‑hoc inference.")


# ------------------------------------------------------------------- #
# Tiny inference helper – can be used after the script has run
# ------------------------------------------------------------------- #
def load_artifacts():
    pipeline_path = ARTIFACTS_DIR / "preprocess.pkl"
    model_path = ARTIFACTS_DIR / "tabnet_best.pt"

    if not pipeline_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(
            "Required artefacts not found. Have you run the training pipeline?"
        )

    pipeline = joblib.load(pipeline_path)

    model = TabNetClassifier(
        n_d=48,
        n_a=48,
        n_steps=6,
        gamma=1.5,
        lambda_sparse=1e-3,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=8e-4),
        mask_type="sparsemax",
        device_name="cpu",
        seed=SEED,
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return pipeline, model


def predict_single(sample_dict: dict) -> dict:
    """
    Accepts a dictionary mapping feature name -> value (must contain all 30 features
    exactly as in the original CSV) and returns the malignant probability.
    """
    pipeline, model = load_artifacts()

    # Preserve column order learned during fit
    feature_names = pipeline.named_steps["imputer"].feature_names_in_
    df = pd.DataFrame([sample_dict], columns=feature_names)

    X_processed = pipeline.transform(df).astype(np.float32)

    prob = model.predict_proba(X_processed)[0, 1]   # probability of class 1 (malignant)
    label = "malignant" if prob >= 0.5 else "benign"
    return {"malignant_probability": round(float(prob), 5), "prediction": label}


if __name__ == "__main__":
    main()
