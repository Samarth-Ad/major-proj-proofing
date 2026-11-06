
# =============================================================================
# Breast Cancer Tabular Classification – End‑to‑End LightGBM + PyTorch Wrapper
# =============================================================================
# This script implements the complete pipeline described in the play‑book:
#   1️⃣ Load the raw CSV (data/raw/breast_cancer.csv)
#   2️⃣ Pre‑process (encoding, imputation, duplicate removal, split, SMOTE)
#   3️⃣ Train a tuned LightGBM model, calibrate it and serialize it
#   4️⃣ Wrap the calibrated model in a PyTorch‑style nn.Module
#   5️⃣ Save all artefacts under ./agent_workspace/trained_models
#   6️⃣ Provide a minimal Gradio demo for inference
#
# All paths are **relative to the repository root**.  The script can be executed
# with `python pipeline.py` (or `python -m pipeline` if placed inside a
# package).  It requires the conda environment defined in the README
# (environment.yml) to be active.
# =============================================================================

import os
import json
import hashlib
import warnings
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import joblib
import mlflow
import lightgbm as lgb
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss
import torch
import torch.nn as nn
import gradio as gr

# -------------------------------------------------------------------------
# Global configuration & reproducibility
# -------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = Path(__file__).resolve().parent
RAW_CSV = BASE_DIR / "data" / "raw" / "breast_cancer.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "agent_workspace" / "trained_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# Helper utilities
# -------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    """Return SHA‑256 checksum of a file (used for MLflow logging)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def log_mlflow(run_name: str, tags: dict = None):
    """Simple context manager for an MLflow run."""
    return mlflow.start_run(run_name=run_name, tags=tags or {})


# -------------------------------------------------------------------------
# 1️⃣ Data ingestion & preprocessing
# -------------------------------------------------------------------------
def preprocess_and_split(
    csv_path: Path,
    processed_dir: Path,
    test_size: float = 0.10,
    val_size: float = 0.20,
    smote_ratio_threshold: float = 1.5,
) -> Tuple[Path, Path, Path]:
    """
    Load raw CSV, clean it, optionally apply SMOTE on the training split,
    and write train/val/test Parquet files.
    Returns the three Parquet file paths.
    """
    # ---- Load -------------------------------------------------------------
    df = pd.read_csv(csv_path)
    # ---- Target encoding --------------------------------------------------
    if df.iloc[:, -1].dtype == object:
        target_map = {"M": 0, "B": 1}
        df.iloc[:, -1] = df.iloc[:, -1].map(target_map)
    else:
        target_map = None

    # ---- Missing‑value imputation (median) ---------------------------------
    df = df.fillna(df.median(numeric_only=True))

    # ---- Duplicate removal ------------------------------------------------
    dup_cnt = df.duplicated().sum()
    if dup_cnt:
        df = df.drop_duplicates().reset_index(drop=True)

    # ---- Feature order (store for inference) -----------------------------
    feature_cols = [c for c in df.columns if c != df.columns[-1]]
    feature_order_path = MODEL_DIR / "feature_order.json"
    json.dump(feature_cols, open(feature_order_path, "w"))

    # ---- Train / Val / Test split (stratified) ----------------------------
    X = df[feature_cols].values
    y = df.iloc[:, -1].values
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(test_size + val_size),
        stratify=y,
        random_state=SEED,
    )
    # split temp into val and test
    val_frac = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1 - val_frac,
        stratify=y_temp,
        random_state=SEED,
    )

    # ---- SMOTE on training set only (if needed) -------------------------
    ratio = np.bincount(y_train)[0] / np.bincount(y_train)[1]
    smote_applied = False
    if ratio > smote_ratio_threshold:
        sm = SMOTE(random_state=SEED)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        smote_applied = True

    # ---- Persist splits as Parquet (zstd compression) ----------------------
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_path = processed_dir / "train.parquet"
    val_path = processed_dir / "val.parquet"
    test_path = processed_dir / "test.parquet"

    pd.DataFrame(np.column_stack([X_train, y_train]), columns=feature_cols + ["target"]).to_parquet(
        train_path, engine="pyarrow", compression="zstd"
    )
    pd.DataFrame(np.column_stack([X_val, y_val]), columns=feature_cols + ["target"]).to_parquet(
        val_path, engine="pyarrow", compression="zstd"
    )
    pd.DataFrame(np.column_stack([X_test, y_test]), columns=feature_cols + ["target"]).to_parquet(
        test_path, engine="pyarrow", compression="zstd"
    )

    # ---- MLflow logging ---------------------------------------------------
    with log_mlflow("preprocess") as run:
        mlflow.log_artifact(str(csv_path), artifact_path="raw")
        mlflow.log_artifact(str(train_path), artifact_path="processed")
        mlflow.log_artifact(str(val_path), artifact_path="processed")
        mlflow.log_artifact(str(test_path), artifact_path="processed")
        mlflow.log_metric("train_samples", len(y_train))
        mlflow.log_metric("val_samples", len(y_val))
        mlflow.log_metric("test_samples", len(y_test))
        mlflow.log_param("smote_applied", smote_applied)
        mlflow.log_param("target_mapping", json.dumps(target_map) if target_map else "numeric")
        mlflow.log_param("raw_sha256", sha256_file(csv_path))

    return train_path, val_path, test_path


# -------------------------------------------------------------------------
# 2️⃣ LightGBM training + calibration
# -------------------------------------------------------------------------
def train_lightgbm(
    train_path: Path,
    val_path: Path,
    model_dir: Path,
    n_jobs: int = 4,
) -> Tuple[Path, dict]:
    """
    Train LightGBM with the hyper‑parameters from the play‑book,
    calibrate with sigmoid, and serialize both the raw and calibrated models.
    Returns the path to the calibrated model and a dict of metrics.
    """
    # Load splits
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)

    X_train = train_df.drop(columns=["target"]).values
    y_train = train_df["target"].values
    X_val = val_df.drop(columns=["target"]).values
    y_val = val_df["target"].values

    # Compute scale_pos_weight
    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    # Hyper‑parameters (play‑book)
    params = {
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 12,
        "min_data_in_leaf": 25,
        "feature_fraction": 0.92,
        "bagging_fraction": 0.80,
        "bagging_freq": 5,
        "lambda_l1": 0.0,
        "lambda_l2": 0.6,
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "n_jobs": n_jobs,
        "random_state": SEED,
        "scale_pos_weight": scale_pos_weight,
    }

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    with log_mlflow("lightgbm_train", tags={"framework": "lightgbm"}) as run:
        # Train
        booster = lgb.train(
            params,
            lgb_train,
            num_boost_round=420,
            valid_sets=[lgb_train, lgb_val],
            valid_names=["train", "val"],
            early_stopping_rounds=30,
            verbose_eval=False,
        )

        # Metrics on validation set
        val_pred = booster.predict(X_val, num_iteration=booster.best_iteration)
        auc_val = roc_auc_score(y_val, val_pred)

        # Calibration
        calibrator = CalibratedClassifierCV(
            base_estimator=booster,
            method="sigmoid",
            cv="prefit",
        )
        calibrator.fit(X_train, y_train)  # fit on training data

        # Serialize calibrated model
        calibrated_path = model_dir / "lightgbm_best_calibrated.pkl"
        joblib.dump(calibrator, calibrated_path)

        # Log artefacts & metrics
        mlflow.log_params(params)
        mlflow.log_metric("auc_val", auc_val)
        mlflow.log_metric("best_iteration", booster.best_iteration)
        mlflow.log_artifact(str(calibrated_path), artifact_path="model")
        mlflow.log_artifact(str(MODEL_DIR / "feature_order.json"), artifact_path="model")

        # Additional calibration metric
        brier = brier_score_loss(y_val, val_pred)
        mlflow.log_metric("brier_score", brier)

        metrics = {
            "auc_val": auc_val,
            "best_iteration": booster.best_iteration,
            "brier_score": brier,
        }

    return calibrated_path, metrics


# -------------------------------------------------------------------------
# 3️⃣ PyTorch‑style wrapper
# -------------------------------------------------------------------------
class LightGBMClassifier(nn.Module):
    """
    Thin PyTorch wrapper around a calibrated LightGBM model.
    The forward method receives a torch.Tensor of shape (batch, n_features)
    and returns probabilities of the positive class (benign) as a
    torch.Tensor of shape (batch, 1).
    """

    def __init__(self, model_path: Path, feature_order_path: Path):
        super().__init__()
        self.booster = joblib.load(model_path)  # CalibratedClassifierCV
        # Preserve column order for sanity checks
        self.feature_order = json.load(open(feature_order_path, "r"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is on CPU and numpy‑compatible (no copy if possible)
        if x.is_cuda:
            x = x.cpu()
        np_arr = x.detach().numpy()
        # LightGBM expects 2‑D array
        probs = self.booster.predict_proba(np_arr)[:, 1]  # prob of class 1
        return torch.from_numpy(probs.astype(np.float32)).unsqueeze(1)


def save_torch_wrapper(wrapper: LightGBMClassifier, out_path: Path):
    """Save only the state_dict (no heavy Python objects)."""
    torch.save(wrapper.state_dict(), out_path)


# -------------------------------------------------------------------------
# 4️⃣ Gradio demo (single‑sample inference)
# -------------------------------------------------------------------------
def launch_gradio_demo(wrapper_path: Path, feature_order_path: Path):
    """
    Starts a Gradio UI that accepts a JSON dict of feature_name → value.
    The UI converts the dict to a tensor respecting the stored feature order,
    runs the model and displays the probability of the benign class.
    """
    # Load wrapper (weights only) and feature order
    feature_order = json.load(open(feature_order_path, "r"))
    dummy_wrapper = LightGBMClassifier(
        model_path=wrapper_path.parent / "lightgbm_best_calibrated.pkl",
        feature_order_path=feature_order_path,
    )
    # Load the saved state dict
    dummy_wrapper.load_state_dict(torch.load(wrapper_path, map_location="cpu"))
    dummy_wrapper.eval()

    def predict(**kwargs):
        # Build a row respecting the stored order
        row = [float(kwargs.get(col, 0.0)) for col in feature_order]
        tensor = torch.tensor([row], dtype=torch.float32)
        prob = dummy_wrapper(tensor).item()
        return {"Benign (B) probability": prob, "Malignant (M) probability": 1 - prob}

    # Dynamically create Gradio inputs based on feature names
    inputs = [gr.Number(label=col) for col in feature_order]
    outputs = [
        gr.Number(label="Benign (B) probability"),
        gr.Number(label="Malignant (M) probability"),
    ]

    iface = gr.Interface(
        fn=predict,
        inputs=inputs,
        outputs=outputs,
        title="Breast Cancer (Wisconsin) – LightGBM Classifier",
        description="Enter the 30 numeric features of a sample and get the probability of being benign.",
        allow_flagging="never",
    )
    iface.launch(share=False)  # set share=True if you need a public URL


# -------------------------------------------------------------------------
# 5️⃣ Orchestrator (main)
# -------------------------------------------------------------------------
def main():
    # -----------------------------------------------------------------
    # Step 1 – Pre‑process & split
    # -----------------------------------------------------------------
    train_path, val_path, test_path = preprocess_and_split(
        csv_path=RAW_CSV,
        processed_dir=PROCESSED_DIR,
    )

    # -----------------------------------------------------------------
    # Step 2 – Train LightGBM + calibrate
    # -----------------------------------------------------------------
    calibrated_model_path, train_metrics = train_lightgbm(
        train_path=train_path,
        val_path=val_path,
        model_dir=MODEL_DIR,
    )
    print("Training completed. Validation AUC:", train_metrics["auc_val"])

    # -----------------------------------------------------------------
    # Step 3 – Create & save PyTorch wrapper
    # -----------------------------------------------------------------
    feature_order_path = MODEL_DIR / "feature_order.json"
    wrapper = LightGBMClassifier(
        model_path=calibrated_model_path,
        feature_order_path=feature_order_path,
    )
    wrapper_path = MODEL_DIR / "lightgbm_torch_wrapper.pt"
    save_torch_wrapper(wrapper, wrapper_path)
    print(f"PyTorch wrapper saved to {wrapper_path}")

    # -----------------------------------------------------------------
    # Step 4 – (Optional) Launch Gradio demo
    # -----------------------------------------------------------------
    # Comment out the line below if you do not want an interactive UI.
    launch_gradio_demo(wrapper_path, feature_order_path)

    # -----------------------------------------------------------------
    # Step 5 – Save a JSON model‑card/profile (optional but useful)
    # -----------------------------------------------------------------
    profile = {
        "model_name": "lightgbm",
        "framework": "lightgbm==4.5.0",
        "hyperparameters": {
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": 12,
            "min_data_in_leaf": 25,
            "feature_fraction": 0.92,
            "bagging_fraction": 0.80,
            "bagging_freq": 5,
            "lambda_l1": 0.0,
            "lambda_l2": 0.6,
            "n_estimators": 420,
            "scale_pos_weight": wrapper.booster.base_estimator.scale_pos_weight,
            "objective": "binary",
            "metric": "auc",
            "random_state": SEED,
            "n_jobs": 4,
        },
        "metrics": train_metrics,
        "artifact_paths": {
            "calibrated_model": str(calibrated_model_path),
            "torch_wrapper_state_dict": str(wrapper_path),
            "feature_order": str(feature_order_path),
        },
    }
    json.dump(profile, open(MODEL_DIR / "lightgbm_profile.json", "w"), indent=2)
    print("Model profile written to", MODEL_DIR / "lightgbm_profile.json")


if __name__ == "__main__":
    # Suppress LightGBM warnings that are not relevant for CI runs
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
