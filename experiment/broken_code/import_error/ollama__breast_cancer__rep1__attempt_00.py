
"""
End‑to‑end Tabular Classification pipeline for the Breast Cancer dataset.

Steps
-----
1. Load & validate the raw CSV (data/raw/breast_cancer.csv)
2. Stratified split → 70 % train, 20 % validation, 10 % test
3. Build a deterministic sklearn Pipeline:
   - Median imputation
   - Winsor‑capping at 1st / 99th percentiles
   - Standard scaling
   - (optional) Mutual‑information feature selection (k=20)
4. Apply SMOTETomek **only** on the training set
5. Train a TabNet classifier (pytorch‑tabnet) with early stopping
6. Evaluate on the held‑out test set (ROC‑AUC, sensitivity, specificity, …)
7. Serialize the preprocessing pipeline and the best TabNet weights to
   ./agent_workspace/trained_models/
"""

import os
import json
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    roc_auc_score,
    recall_score,
    accuracy_score,
    f1_score,
    brier_score_loss,
)

from imblearn.combine import SMOTETomek

from pytorch_tabnet.tab_model import TabNetClassifier

# --------------------------------------------------------------------------- #
# GLOBAL SETTINGS
# --------------------------------------------------------------------------- #
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

BASE_DIR = Path(__file__).resolve().parent
RAW_CSV = BASE_DIR / "data" / "raw" / "breast_cancer.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ARTIFACTS_DIR = BASE_DIR / "src" / "artifacts"
MODEL_SAVE_DIR = Path("./agent_workspace/trained_models")
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# 1️⃣  DATA VALIDATION
# --------------------------------------------------------------------------- #
def validate_raw_csv(csv_path: Path) -> pd.DataFrame:
    """Load the CSV, run the sanity checks described in the spec and
    write a JSON validation report."""
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

    # Feature dtype checks
    feature_cols = [c for c in df.columns if c != "target"]
    non_numeric = [
        c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric:
        report["status"] = "FAIL"
        report["issues"].append(f"Non‑numeric feature columns: {non_numeric}")

    # Duplicates
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

    # Write report
    report_path = BASE_DIR / "reports" / "validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    if report["status"] != "OK":
        raise ValueError(
            f"Data validation failed. See {report_path} for details."
        )
    print("✅ Data validation passed.")
    return df


# --------------------------------------------------------------------------- #
# 2️⃣  SPLIT (70/20/10) – stratified
# --------------------------------------------------------------------------- #
def stratified_split(df: pd.DataFrame):
    """Return three DataFrames: train, val, test."""
    X = df.drop(columns=["target"])
    y = df["target"]

    # First split: train (70%) vs temp (30%)
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=0.30, random_state=SEED
    )
    train_idx, temp_idx = next(sss1.split(X, y))
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_temp, y_temp = X.iloc[temp_idx], y.iloc[temp_idx]

    # Second split: val (≈20%) vs test (≈10%) from the temp block
    # 20/30 = 0.6667 → test_size = 1/3 of temp to get ~10% overall
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=1 / 3, random_state=SEED
    )
    val_idx, test_idx = next(sss2.split(X_temp, y_temp))
    X_val, y_val = X_temp.iloc[val_idx], y_temp.iloc[val_idx]
    X_test, y_test = X_temp.iloc[test_idx], y_temp.iloc[test_idx]

    # Persist as parquet
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for name, X_part, y_part in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        part = pd.concat([X_part, y_part], axis=1)
        part.to_parquet(PROCESSED_DIR / f"{name}.parquet", index=False)

    print(
        f"✅ Split completed: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}"
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
# 3️⃣  CUSTOM WIN‑SORIZER
# --------------------------------------------------------------------------- #
class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip numeric columns at the 1st and 99th percentiles (computed on fit)."""

    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        self.lower_ = np.percentile(X, self.lower_quantile * 100, axis=0)
        self.upper_ = np.percentile(X, self.upper_quantile * 100, axis=0)
        return self

    def transform(self, X, y=None):
        X_clipped = np.clip(X, self.lower_, self.upper_)
        return X_clipped


# --------------------------------------------------------------------------- #
# 4️⃣  PRE‑PROCESSING PIPELINE (fit on train only)
# --------------------------------------------------------------------------- #
def build_preprocess_pipeline(k_features: int = 20):
    """Return a sklearn Pipeline that includes imputation, winsorizing,
    scaling, optional feature selection and SMOTETomek (as a separate step)."""
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("winsorizer", Winsorizer()),
        ("scaler", StandardScaler()),
    ]

    # Optional feature selection – keep 20 best features
    if k_features is not None and k_features > 0:
        steps.append(
            ("selector", SelectKBest(mutual_info_classif, k=k_features))
        )
    pipeline = Pipeline(steps)
    return pipeline


def preprocess_and_resample(
    X_train, y_train, X_val, X_test, pipeline: Pipeline
):
    """Fit the pipeline on training data, apply SMOTETomek, and transform
    validation / test data."""
    # Fit pipeline (imputer → winsorizer → scaler → selector)
    X_train_proc = pipeline.fit_transform(X_train, y_train)

    # SMOTETomek on the *processed* training data
    sampler = SMOTETomek(random_state=SEED)
    X_res, y_res = sampler.fit_resample(X_train_proc, y_train)

    # Transform validation & test (no resampling)
    X_val_proc = pipeline.transform(X_val)
    X_test_proc = pipeline.transform(X_test)

    # Persist the whole pipeline
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_path = ARTIFACTS_DIR / "preprocess.pkl"
    joblib.dump(pipeline, pipeline_path)
    print(f"✅ Preprocessing pipeline saved to {pipeline_path}")

    return X_res, y_res, X_val_proc, X_test_proc


# --------------------------------------------------------------------------- #
# 5️⃣  TABNET TRAINING
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
    # Convert to torch tensors
    X_train_t = torch.from_numpy(X_train.astype(np.float32))
    y_train_t = torch.from_numpy(y_train.values.astype(np.int64))
    X_val_t = torch.from_numpy(X_val.astype(np.float32))
    y_val_t = torch.from_numpy(y_val.values.astype(np.int64))

    train_ds = TensorDataset(X_train_t, y_train_t)
    val_ds = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )

    # TabNet hyper‑parameters (as per spec)
    clf = TabNetClassifier(
        n_d=48,
        n_a=48,
        n_steps=6,
        gamma=1.5,
        lambda_sparse=1e-3,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": 8e-4},
        scheduler_params={"step_size": 10, "gamma": 0.9},
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        mask_type="sparsemax",
        verbose=0,
        seed=SEED,
        device_name="cpu",  # force CPU for reproducibility in CI
    )

    start = time.time()
    clf.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_name=["val"],
        eval_metric=["auc"],
        max_epochs=max_epochs,
        patience=patience,
        batch_size=batch_size,
        virtual_batch_size=64,
        num_workers=0,
        drop_last=False,
    )
    duration = time.time() - start

    # Save best model state dict
    model_path = MODEL_SAVE_DIR / "tabnet_best.pt"
    torch.save(clf.state_dict(), model_path)
    print(f"✅ TabNet best model saved to {model_path}")

    # Basic stats
    n_params = sum(p.numel() for p in clf.parameters())
    print(f"Model parameters: {n_params:,}")

    # Inference latency (10 000 forward passes on CPU)
    clf.eval()
    dummy_input = torch.from_numpy(
        X_val[:batch_size].astype(np.float32)
    )  # reuse a batch
    torch.cuda.empty_cache()
    start_lat = time.perf_counter()
    for _ in range(10_000 // batch_size):
        _ = clf.predict_proba(dummy_input)
    latency = (time.perf_counter() - start_lat) / 10_000
    print(f"Avg inference latency per sample: {latency*1e3:.3f} ms")

    # Return model object (still in memory) and some training stats
    training_info = {
        "train_time_sec": duration,
        "n_parameters": n_params,
        "avg_latency_sec": latency,
    }
    return clf, training_info


# --------------------------------------------------------------------------- #
# 6️⃣  EVALUATION ON TEST SET
# --------------------------------------------------------------------------- #
def evaluate_on_test(
    model: TabNetClassifier,
    X_test: np.ndarray,
    y_test: pd.Series,
):
    """Compute ROC‑AUC, sensitivity (recall for class 0), specificity (recall for class 1) and other common metrics."""
    probs = model.predict_proba(X_test)[:, 0]  # probability of class 0 (malignant)
    y_pred = (probs >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, probs)
    sensitivity = recall_score(y_test, y_pred, pos_label=0)
    specificity = recall_score(y_test, y_pred, pos_label=1)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    brier = brier_score_loss(y_test, probs)

    metrics = {
        "roc_auc": roc_auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy": acc,
        "f1": f1,
        "brier_loss": brier,
    }

    # Persist metrics
    reports_dir = BASE_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = reports_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Test metrics saved to {metrics_path}")

    return metrics


# --------------------------------------------------------------------------- #
# 7️⃣  MAIN PIPELINE DRIVER
# --------------------------------------------------------------------------- #
def main():
    # 1️⃣ Load & validate raw data
    df = validate_raw_csv(RAW_CSV)

    # 2️⃣ Split
    X_train, y_train, X_val, y_val, X_test, y_test = stratified_split(df)

    # 3️⃣ Build preprocessing pipeline (k=20)
    preprocess_pipe = build_preprocess_pipeline(k_features=20)

    # 4️⃣ Fit pipeline, SMOTETomek, transform splits
    X_train_proc, y_train_res, X_val_proc, X_test_proc = preprocess_and_resample(
        X_train, y_train, X_val, X_test, preprocess_pipe
    )

    # 5️⃣ Train TabNet
    tabnet_model, train_info = train_tabnet(
        X_train_proc,
        y_train_res,
        X_val_proc,
        y_val,
        max_epochs=200,
        patience=20,
        batch_size=128,
    )

    # 6️⃣ Evaluate on test set
    test_metrics = evaluate_on_test(tabnet_model, X_test_proc, y_test)

    # 7️⃣ Summary printout
    print("\n=== FINAL PIPELINE SUMMARY ===")
    print("Training info:", train_info)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    # Suppress unnecessary warnings for clean CI logs
    warnings.filterwarnings("ignore")
    main()
