
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Breast‑cancer tabular classification pipeline (end‑to‑end)

Steps
-----
1. Load & validate the raw CSV (expected shape 569 × 31, target ∈ {0,1})
2. Stratified split → 70 % train, 20 % validation, 10 % test
3. Deterministic preprocessing pipeline
       median imputation → Winsor‑capping (1 % / 99 %) → Standard‑Scaler
       → Mutual‑Information feature selection (k = 20)
4. SMOTETomek applied **only** on the processed training set
5. Train a TabNet classifier (early‑stopping on validation AUC)
6. Save the fitted pipeline (preprocess.pkl) and the best TabNet weights
   (tabnet_best.pt) to ``./agent_workspace/trained_models``
7. Evaluate on the untouched test set and write ``test_metrics.json``
"""

import json
import os
import random
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------------
# 0️⃣ Global configuration & reproducibility
# -------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Base directory = directory that contains this script
BASE_DIR = Path(__file__).resolve().parent

# -------------------------------------------------------------------------
# 1️⃣ Paths
# -------------------------------------------------------------------------
# Expected location of the raw CSV (relative to the repo root)
RAW_CSV = BASE_DIR / "data" / "raw" / "breast_cancer.csv"

# Directory where artefacts (pipeline, model, metrics) are stored
ARTIFACTS_DIR = BASE_DIR / "agent_workspace" / "trained_models"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# 2️⃣ Validation of the raw CSV
# -------------------------------------------------------------------------
def validate_raw_csv(csv_path: Path) -> pd.DataFrame:
    """Load the CSV and run a series of sanity checks.
    Returns the validated DataFrame or raises an informative exception."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Raw CSV not found at {csv_path}")

    df = pd.read_csv(csv_path)

    # Expected shape for the classic Wisconsin Breast Cancer dataset
    if df.shape[0] != 569:
        raise ValueError(f"Unexpected number of rows: {df.shape[0]} (expected 569)")
    if df.shape[1] != 31:
        raise ValueError(f"Unexpected number of columns: {df.shape[1]} (expected 31)")

    # Target column must exist and be binary 0/1
    if "target" not in df.columns:
        raise ValueError("Missing column `target`")
    if set(df["target"].unique()) != {0, 1}:
        raise ValueError(
            f"`target` contains values {df['target'].unique()} (expected only 0 and 1)"
        )

    # All other columns must be numeric
    feature_cols = [c for c in df.columns if c != "target"]
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Feature column `{col}` is not numeric")

    # No duplicate rows
    dup_cnt = df.duplicated().sum()
    if dup_cnt > 0:
        raise ValueError(f"{dup_cnt} duplicate rows found")

    # No missing values
    miss_cnt = df.isna().sum().sum()
    if miss_cnt > 0:
        raise ValueError(f"{miss_cnt} missing values found")

    print("Validation passed")
    return df


# -------------------------------------------------------------------------
# 3️⃣ Custom Winsorizer transformer
# -------------------------------------------------------------------------
class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each feature to the given lower / upper quantiles (computed on fit)."""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
            raise ValueError("Quantiles must satisfy 0 ≤ lower < upper ≤ 1")
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


# -------------------------------------------------------------------------
# 4️⃣ Build the preprocessing pipeline
# -------------------------------------------------------------------------
def build_preprocess_pipeline(k_features: int = 20) -> Pipeline:
    """Construct the deterministic sklearn pipeline."""
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("winsor", Winsorizer(lower_quantile=0.01, upper_quantile=0.99)),
            ("scaler", StandardScaler()),
            ("selectk", SelectKBest(score_func=mutual_info_classif, k=k_features)),
        ]
    )
    return pipeline


# -------------------------------------------------------------------------
# 5️⃣ Main orchestration
# -------------------------------------------------------------------------
def main():
    # -------------------------------------------------
    # 1️⃣ Locate the CSV (fail fast if not present)
    # -------------------------------------------------
    if not RAW_CSV.is_file():
        raise FileNotFoundError(f"Raw CSV not found at {RAW_CSV}")

    # -------------------------------------------------
    # 2️⃣ Load & validate data
    # -------------------------------------------------
    df = validate_raw_csv(RAW_CSV)

    X = df.drop(columns=["target"])
    y = df["target"].astype(int)

    # -------------------------------------------------
    # 3️⃣ Stratified split → 70 % train, 20 % val, 10 % test
    # -------------------------------------------------
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=SEED
    )
    # Remaining 90 % → split into 70/20 (i.e. 7/2 ratio)
    val_ratio = 0.20 / 0.90
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_ratio,
        stratify=y_temp,
        random_state=SEED,
    )
    print(
        f"Dataset split → train: {X_train.shape[0]}, "
        f"val: {X_val.shape[0]}, test: {X_test.shape[0]}"
    )

    # -------------------------------------------------
    # 4️⃣ Build & fit preprocessing pipeline on training data
    # -------------------------------------------------
    pipeline = build_preprocess_pipeline(k_features=20)
    pipeline.fit(X_train, y_train)  # target needed for SelectKBest

    # Persist the pipeline
    pipeline_path = ARTIFACTS_DIR / "preprocess.pkl"
    joblib.dump(pipeline, pipeline_path)
    print(f"Preprocessing pipeline saved to {pipeline_path}")

    # Transform all splits
    X_train_proc = pipeline.transform(X_train)
    X_val_proc = pipeline.transform(X_val)
    X_test_proc = pipeline.transform(X_test)

    # -------------------------------------------------
    # 5️⃣ SMOTETomek – ONLY on the processed training set
    # -------------------------------------------------
    smote_tomek = SMOTETomek(random_state=SEED)
    X_train_res, y_train_res = smote_tomek.fit_resample(X_train_proc, y_train)

    print(
        f"SMOTETomek applied → original train size {X_train_proc.shape[0]}, "
        f"after resampling {X_train_res.shape[0]}"
    )

    # -------------------------------------------------
    # 6️⃣ Convert to the dtype expected by TabNet (float32 / int64)
    # -------------------------------------------------
    X_train_res = X_train_res.astype(np.float32)
    X_val_proc = X_val_proc.astype(np.float32)
    X_test_proc = X_test_proc.astype(np.float32)

    y_train_res = y_train_res.astype(np.int64)
    y_val = y_val.astype(np.int64)
    y_test = y_test.astype(np.int64)

    # -------------------------------------------------
    # 7️⃣ TabNet model definition & training
    # -------------------------------------------------
    from pytorch_tabnet.tab_model import TabNetClassifier

    tabnet_params = {
        "n_d": 48,
        "n_a": 48,
        "n_steps": 6,
        "gamma": 1.5,
        "lambda_sparse": 1e-3,
        "optimizer_fn": torch.optim.Adam,
        "optimizer_params": {"lr": 8e-4},
        "mask_type": "sparsemax",
        "device_name": "cpu",  # change to "cuda" if a GPU is available
        "seed": SEED,
        "verbose": 0,
    }

    model = TabNetClassifier(**tabnet_params)

    start_time = time.time()
    model.fit(
        X_train_res,
        y_train_res,
        eval_set=[(X_val_proc, y_val)],
        eval_name=["val"],
        eval_metric=["auc"],
        max_epochs=200,
        patience=20,
        batch_size=128,
        virtual_batch_size=64,
        num_workers=0,
        drop_last=False,
    )
    train_duration = time.time() - start_time
    print(f"Training finished in {train_duration:.1f} seconds")

    # -------------------------------------------------
    # 8️⃣ Save the best model weights
    # -------------------------------------------------
    model_path = ARTIFACTS_DIR / "tabnet_best.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Trained TabNet model saved to {model_path}")

    # -------------------------------------------------
    # 9️⃣ Evaluation on the untouched test set
    # -------------------------------------------------
    # Load weights into a fresh instance (good practice for inference)
    inference_model = TabNetClassifier(**tabnet_params)
    inference_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    inference_model.eval()

    # Predict probabilities for the positive class (malignant = 1)
    test_proba = inference_model.predict_proba(X_test_proc)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)

    # Compute metrics
    metrics = {
        "roc_auc": round(roc_auc_score(y_test, test_proba), 5),
        "accuracy": round(accuracy_score(y_test, test_pred), 5),
        "f1": round(f1_score(y_test, test_pred), 5),
        "sensitivity": round(recall_score(y_test, test_pred, pos_label=1), 5),
        "specificity": round(recall_score(y_test, test_pred, pos_label=0), 5),
        "precision": round(precision_score(y_test, test_pred, pos_label=1), 5),
        "brier_score": round(brier_score_loss(y_test, test_proba), 5),
        "train_time_sec": round(train_duration, 2),
        "n_parameters": sum(p.numel() for p in model.parameters()),
    }

    metrics_path = ARTIFACTS_DIR / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Test‑set metrics saved to {metrics_path}")

    # -------------------------------------------------
    # 10️⃣ (Optional) Simple Gradio demo for interactive inference
    # -------------------------------------------------
    try:
        import gradio as gr

        feature_names = X.columns.tolist()  # original order, needed for UI

        # Load pipeline and model once (avoid re‑loading on every request)
        saved_pipeline = joblib.load(pipeline_path)
        saved_model = TabNetClassifier(**tabnet_params)
        saved_model.load_state_dict(torch.load(model_path, map_location="cpu"))
        saved_model.eval()

        def infer_one(sample_dict):
            """Accept a dict of feature_name → value, return probability & label."""
            df_row = pd.DataFrame([sample_dict], columns=feature_names)
            proc = saved_pipeline.transform(df_row).astype(np.float32)
            prob = saved_model.predict_proba(proc)[0, 1]
            label = "malignant" if prob >= 0.5 else "benign"
            return {
                "malignant_probability": round(float(prob), 5),
                "prediction": label,
            }

        demo = gr.Interface(
            fn=infer_one,
            inputs=[gr.Number(label=name, value=0.0) for name in feature_names],
            outputs=[
                gr.Number(label="malignant_probability"),
                gr.Text(label="prediction"),
            ],
            title="Breast‑Cancer TabNet Predictor",
            description=(
                "Enter the 30 feature values (as in the original dataset) "
                "to obtain a malignancy probability."
            ),
            allow_flagging="never",
        )

        if __name__ == "__main__":
            demo.launch()
    except Exception as e:
        warnings.warn(f"Gradio demo could not be started: {e}")


if __name__ == "__main__":
    main()
