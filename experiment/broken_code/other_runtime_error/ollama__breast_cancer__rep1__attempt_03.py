
"""
End‑to‑end Tabular Classification pipeline for the Breast Cancer dataset.

The script follows the specification you provided:

1️⃣ Load & validate the raw CSV (data/raw/breast_cancer.csv)  
2️⃣ Stratified split → 70 % train, 20 % validation, 10 % test  
3️⃣ Deterministic sklearn Pipeline  
   – Median imputation  
   – Winsor‑capping at the 1st / 99th percentiles  
   – Standard scaling  
   – Mutual‑information feature selection (k=20)  
4️⃣ SMOTETomek **only** on the training set (after scaling/selection)  
5️⃣ Train a TabNet classifier (pytorch‑tabnet) with early stopping  
6️⃣ Evaluate on the held‑out test set (ROC‑AUC, sensitivity, specificity, …)  
7️⃣ Serialize the preprocessing pipeline and the best TabNet weights to
   ./agent_workspace/trained_models/
"""

# --------------------------------------------------------------------------- #
# 0️⃣ Imports
# --------------------------------------------------------------------------- #
import json
import os
import random
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
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# 1️⃣ Ensure required third‑party package (pytorch‑tabnet) is installed
# --------------------------------------------------------------------------- #
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except ImportError:  # pragma: no cover
    # Install the missing package on‑the‑fly (only works if the environment
    # permits pip installs).  This makes the script self‑contained for the
    # evaluation harness.
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pytorch-tabnet==4.3.*"])
    from pytorch_tabnet.tab_model import TabNetClassifier

# --------------------------------------------------------------------------- #
# 2️⃣ Global settings & paths
# --------------------------------------------------------------------------- #
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Project‑relative paths (run from repository root)
BASE_DIR = Path(__file__).resolve().parent
RAW_CSV = BASE_DIR / "data" / "raw" / "breast_cancer.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ARTIFACTS_DIR = BASE_DIR / "src" / "artifacts"
MODEL_SAVE_DIR = Path("./agent_workspace/trained_models")
REPORTS_DIR = BASE_DIR / "reports"

# Ensure output folders exist
for p in (PROCESSED_DIR, ARTIFACTS_DIR, MODEL_SAVE_DIR, REPORTS_DIR):
    p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# 3️⃣ Data validation
# --------------------------------------------------------------------------- #
def validate_raw_csv(csv_path: Path) -> pd.DataFrame:
    """Load CSV, run sanity checks, write a JSON report and return the DataFrame."""
    df = pd.read_csv(csv_path)

    report = {"status": "OK", "issues": []}

    # Row / column count
    if df.shape[0] != 569:
        report["status"] = "FAIL"
        report["issues"].append(f"Expected 569 rows, found {df.shape[0]}")
    if df.shape[1] != 31:
        report["status"] = "FAIL"
        report["issues"].append(f"Expected 31 columns, found {df.shape[1]}")

    # Target column checks
    if "target" not in df.columns:
        report["status"] = "FAIL"
        report["issues"].append("Missing 'target' column")
    else:
        target_vals = set(df["target"].unique())
        if target_vals != {0, 1}:
            report["status"] = "FAIL"
            report["issues"].append(
                f"'target' should contain only 0 and 1, found {target_vals}"
            )

    # Feature dtype checks (all except target must be numeric)
    feature_cols = [c for c in df.columns if c != "target"]
    non_numeric = [
        c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        report["status"] = "FAIL"
        report["issues"].append(f"Non‑numeric feature columns: {non_numeric}")

    # Duplicate rows
    dup_rows = df.duplicated().sum()
    if dup_rows:
        report["status"] = "FAIL"
        report["issues"].append(f"{dup_rows} duplicate rows found")

    # Missing values
    missing_perc = df.isna().mean() * 100
    cols_with_missing = missing_perc[missing_perc > 0].index.tolist()
    if cols_with_missing:
        report["status"] = "FAIL"
        report["issues"].append(
            f"Columns with missing values (>0%): {cols_with_missing}"
        )

    # Write JSON report
    report_path = REPORTS_DIR / "validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if report["status"] != "OK":
        raise ValueError(
            f"Data validation failed – see {report_path} for details."
        )
    print("[INFO] Data validation passed.")
    return df

# --------------------------------------------------------------------------- #
# 4️⃣ Stratified split (70/20/10)
# --------------------------------------------------------------------------- #
def stratified_split(df: pd.DataFrame):
    """Return train/val/test DataFrames (features + target)."""
    X = df.drop(columns=["target"])
    y = df["target"]

    # 1st split: 70 % train vs 30 % temp (val+test)
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=0.30, random_state=SEED
    )
    train_idx, temp_idx = next(sss1.split(X, y))
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_temp, y_temp = X.iloc[temp_idx], y.iloc[temp_idx]

    # 2nd split: temp → 2/3 validation (≈20 % total) and 1/3 test (≈10 % total)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=1 / 3, random_state=SEED
    )
    val_idx, test_idx = next(sss2.split(X_temp, y_temp))
    X_val, y_val = X_temp.iloc[val_idx], y_temp.iloc[val_idx]
    X_test, y_test = X_temp.iloc[test_idx], y_temp.iloc[test_idx]

    # Persist as parquet
    for name, X_part, y_part in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        part = pd.concat([X_part, y_part], axis=1)
        part.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)

    print(
        f"[INFO] Split completed – train:{len(X_train)}, val:{len(X_val)}, test:{len(X_test)}"
    )
    return (
        X_train.reset_index(drop=True),
        y_train.reset_index(drop=True),
        X_val.reset_index(drop=True),
        y_val.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )

# --------------------------------------------------------------------------- #
# 5️⃣ Custom Winsorizer (percentile clipping)
# --------------------------------------------------------------------------- #
class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip numeric columns at the 1st and 99th percentiles (computed on fit)."""

    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        self.lower_ = np.percentile(
            X_arr, self.lower_quantile * 100, axis=0
        )
        self.upper_ = np.percentile(
            X_arr, self.upper_quantile * 100, axis=0
        )
        return self

    def transform(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        return np.clip(X_arr, self.lower_, self.upper_)

# --------------------------------------------------------------------------- #
# 6️⃣ Pre‑processing pipeline (fit on train only)
# --------------------------------------------------------------------------- #
def build_preprocess_pipeline(k_features: int = 20) -> Pipeline:
    """Create a sklearn Pipeline (imputer → winsorizer → scaler → selector)."""
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("winsorizer", Winsorizer()),
        ("scaler", StandardScaler()),
    ]

    # Mutual‑information feature selection (keep k best)
    if k_features and k_features > 0:
        steps.append(
            ("selector", SelectKBest(mutual_info_classif, k=k_features))
        )
    return Pipeline(steps)


def fit_transform_and_resample(
    X_train, y_train, X_val, X_test, pipeline: Pipeline
):
    """
    Fit the preprocessing pipeline on the training data,
    apply SMOTETomek **after** scaling/selection,
    and transform validation & test (no resampling).
    Returns processed arrays ready for TabNet.
    """
    # Fit pipeline on training data
    X_train_proc = pipeline.fit_transform(X_train, y_train)

    # SMOTETomek on the processed training data
    sampler = SMOTETomek(random_state=SEED)
    X_res, y_res = sampler.fit_resample(X_train_proc, y_train)

    # Transform validation & test (no resampling)
    X_val_proc = pipeline.transform(X_val)
    X_test_proc = pipeline.transform(X_test)

    # Persist the whole pipeline for inference
    pipeline_path = ARTIFACTS_DIR / "preprocess.pkl"
    joblib.dump(pipeline, pipeline_path)
    print(f"[INFO] Preprocessing pipeline saved to {pipeline_path}")

    return X_res, y_res, X_val_proc, X_test_proc

# --------------------------------------------------------------------------- #
# 7️⃣ TabNet training
# --------------------------------------------------------------------------- #
def train_tabnet(
    X_train,
    y_train,
    X_val,
    y_val,
    max_epochs: int = 200,
    patience: int = 20,
    batch_size: int = 128,
):
    """Train TabNetClassifier with early stopping on validation AUC."""
    # Convert to numpy with correct dtypes (TabNet expects float32 & int64)
    X_train_np = X_train.astype(np.float32)
    y_train_np = y_train.astype(np.int64)
    X_val_np = X_val.astype(np.float32)
    y_val_np = y_val.astype(np.int64)

    clf = TabNetClassifier(
        n_d=48,
        n_a=48,
        n_steps=6,
        gamma=1.5,
        lambda_sparse=1e-3,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": 8e-4},
        mask_type="sparsemax",
        verbose=0,
        seed=SEED,
        device_name="cpu",
    )

    start_time = time.time()
    clf.fit(
        X_train_np,
        y_train_np,
        eval_set=[(X_val_np, y_val_np)],
        eval_name=["val"],
        eval_metric=["auc"],
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        virtual_batch_size=64,
        num_workers=0,
        drop_last=False,
    )
    train_duration = time.time() - start_time

    # Save the best model (TabNet stores the best weights internally)
    model_path = MODEL_SAVE_DIR / "tabnet_best.pt"
    torch.save(clf.state_dict(), model_path)
    print(f"[INFO] Best TabNet model saved to {model_path}")

    # Model diagnostics
    n_params = sum(p.numel() for p in clf.parameters())
    print(f"[INFO] Model parameters: {n_params:,}")

    # Approximate inference latency (10 000 forward passes, batch‑wise)
    clf.eval()
    dummy_batch = torch.from_numpy(
        X_val_np[:batch_size]
    )
    start_lat = time.perf_counter()
    for _ in range(10_000 // batch_size):
        _ = clf.predict_proba(dummy_batch)
    latency_per_sample = (time.perf_counter() - start_lat) / 10_000
    print(
        f"[INFO] Avg inference latency per sample: {latency_per_sample*1e3:.3f} ms"
    )

    training_info = {
        "train_time_sec": train_duration,
        "n_parameters": n_params,
        "avg_latency_sec": latency_per_sample,
    }
    return clf, training_info

# --------------------------------------------------------------------------- #
# 8️⃣ Evaluation on test set
# --------------------------------------------------------------------------- #
def evaluate_on_test(
    model: TabNetClassifier, X_test: np.ndarray, y_test: pd.Series
):
    """Compute ROC‑AUC, sensitivity, specificity and a few extra metrics."""
    probs = model.predict_proba(X_test)[:, 0]  # class 0 = malignant
    y_pred = (probs >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, probs)
    sensitivity = recall_score(y_test, y_pred, pos_label=0)  # malignant recall
    specificity = recall_score(y_test, y_pred, pos_label=1)  # benign recall
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    brier = brier_score_loss(y_test, probs)

    metrics = {
        "roc_auc": roc_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": accuracy,
        "f1": f1,
        "brier_loss": brier,
    }

    metrics_path = REPORTS_DIR / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Test metrics saved to {metrics_path}")

    return metrics

# --------------------------------------------------------------------------- #
# 9️⃣ Main driver
# --------------------------------------------------------------------------- #
def main():
    # 1️⃣ Load & validate raw data
    df = validate_raw_csv(RAW_CSV)

    # 2️⃣ Train / Validation / Test split (70/20/10)
    X_train, y_train, X_val, y_val, X_test, y_test = stratified_split(df)

    # 3️⃣ Build preprocessing pipeline (k=20 features)
    preprocess_pipe = build_preprocess_pipeline(k_features=20)

    # 4️⃣ Fit pipeline, apply SMOTETomek, transform splits
    X_train_proc, y_train_res, X_val_proc, X_test_proc = fit_transform_and_resample(
        X_train, y_train, X_val, X_test, preprocess_pipe
    )

    # 5️⃣ Train TabNet on the resampled training data
    tabnet_model, train_info = train_tabnet(
        X_train_proc,
        y_train_res,
        X_val_proc,
        y_val,
        max_epochs=200,
        patience=20,
        batch_size=128,
    )

    # 6️⃣ Evaluate on the untouched test set
    test_metrics = evaluate_on_test(tabnet_model, X_test_proc, y_test)

    # 7️⃣ Summary
    print("\n=== PIPELINE SUMMARY ===")
    print("Training info:")
    print(json.dumps(train_info, indent=2))
    print("Test metrics:")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    # Suppress non‑essential warnings for clean CI logs
    warnings.filterwarnings("ignore")
    main()
