
"""
diabetes_tabnet_pipeline.py

End‑to‑end pipeline for the tabular regression task described in the
specification. It performs the following steps:

1️⃣ Load the raw CSV (expected at ./data/raw/diabetes.csv).
2️⃣ Split the data into train / validation / test (70 % / 20 % / 10 %).
3️⃣ Build a scikit‑learn preprocessing pipeline:
   – Median imputation
   – Winsorisation (1st‑99th percentile computed on the training set)
   – Robust scaling
   – (optional) log‑transform of the target if its skewness > 1.0
4️⃣ Persist the processed NumPy arrays and the split indices.
5️⃣ Train a TabNetRegressor (PyTorch) with the fixed hyper‑parameters.
6️⃣ Evaluate on validation and test sets (RMSE, MAE, R²).
7️⃣ Save the trained model, the preprocessing pipeline, a meta‑json
   (log‑target flag) and a JSON report with the metrics.

All artefacts are stored under the repository root, exactly as required:

    ./data/raw/diabetes.csv                     – raw data (tracked with DVC)
    ./data/processed/                           – NumPy arrays (X_*.npy, y_*.npy)
    ./data/splits/                              – split_indices.pkl
    ./agent_workspace/trained_models/           – pipeline, meta, model, report
"""

# --------------------------------------------------------------
# 0️⃣ Imports & global reproducibility
# --------------------------------------------------------------
import os
import json
import pickle
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import yaml

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)

# --------------------------------------------------------------
# 1️⃣ Global seeds for reproducibility
# --------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# PyTorch seed will be set later after the import succeeds
# --------------------------------------------------------------
# 2️⃣ Optional import of TabNet (with a clear error message)
# --------------------------------------------------------------
try:
    import torch
    from pytorch_tabnet.tab_model import TabNetRegressor
    torch.manual_seed(SEED)
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "The package 'pytorch-tabnet' (and its PyTorch dependency) is not "
        "installed in the current environment. Please install it via conda:\n"
        "    conda install -c conda-forge pytorch-tabnet\n"
        "or via pip:\n"
        "    pip install pytorch-tabnet\n"
        "and then re‑run this script."
    ) from exc

# --------------------------------------------------------------
# 3️⃣ Custom Winsoriser transformer (1st‑99th percentile)
# --------------------------------------------------------------
class Winsoriser(BaseEstimator, TransformerMixin):
    """
    Clips numeric features to the 1st and 99th percentiles computed on the
    training data. The percentiles are stored in ``self.lower_`` and
    ``self.upper_``.
    """

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        self.lower_ = np.quantile(X_arr, self.lower_quantile, axis=0)
        self.upper_ = np.quantile(X_arr, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        return np.clip(X_arr, self.lower_, self.upper_)


# --------------------------------------------------------------
# 4️⃣ Path configuration (all relative to repository root)
# --------------------------------------------------------------
ROOT = Path.cwd()  # assume script is executed from the repo root

RAW_CSV = ROOT / "data" / "raw" / "diabetes.csv"

PROCESSED_DIR = ROOT / "data" / "processed"
SPLIT_DIR = ROOT / "data" / "splits"
TRAINED_MODELS_DIR = ROOT / "agent_workspace" / "trained_models"

# Ensure output directories exist
for d in (PROCESSED_DIR, SPLIT_DIR, TRAINED_MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------
# 5️⃣ Load raw data
# --------------------------------------------------------------
print("🔎 Loading raw CSV...")
if not RAW_CSV.is_file():
    raise FileNotFoundError(f"Raw CSV not found at {RAW_CSV}")

df = pd.read_csv(RAW_CSV)

TARGET_COL = "target"
if TARGET_COL not in df.columns:
    raise KeyError(f"Target column '{TARGET_COL}' not found in the CSV.")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].astype(float).values  # ensure numeric target

# --------------------------------------------------------------
# 6️⃣ Train / validation / test split (70 % / 20 % / 10 %)
# --------------------------------------------------------------
print("🔀 Creating train / validation / test splits...")
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.10,
    random_state=SEED,
    shuffle=True,
)

# 0.2222 ≈ 20 % / (70 % + 20 %)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.2222,
    random_state=SEED,
    shuffle=True,
)

# Save split indices for reproducibility
split_indices = {
    "train_idx": X_train.index.tolist(),
    "val_idx": X_val.index.tolist(),
    "test_idx": X_test.index.tolist(),
}
split_path = SPLIT_DIR / "split_indices.pkl"
with open(split_path, "wb") as f:
    pickle.dump(split_indices, f)
print(f"📁 Split indices saved to {split_path}")

# --------------------------------------------------------------
# 7️⃣ Build preprocessing pipeline (fit on training data only)
# --------------------------------------------------------------
print("⚙️ Building preprocessing pipeline...")
pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("winsoriser", Winsoriser(lower_quantile=0.01, upper_quantile=0.99)),
        ("scaler", RobustScaler()),
    ]
)

pipeline.fit(X_train.values)  # fit strictly on training data

# --------------------------------------------------------------
# 8️⃣ Optional log‑transform of the target
# --------------------------------------------------------------
log_target = False
skew = pd.Series(y_train).skew()
if abs(skew) > 1.0:
    print(f"📈 Target skew = {skew:.3f} → applying log1p transform.")
    y_train = np.log1p(y_train)
    y_val = np.log1p(y_val)
    y_test = np.log1p(y_test)
    log_target = True
else:
    print(f"📈 Target skew = {skew:.3f} (no log transform).")

# Persist the log‑target flag for later inference
meta_path = TRAINED_MODELS_DIR / "preprocess_meta.json"
with open(meta_path, "w") as f:
    json.dump(
        {"log_target": log_target, "target_column": TARGET_COL},
        f,
        indent=2,
    )
print(f"🗒️ Preprocess meta saved to {meta_path}")

# --------------------------------------------------------------
# 9️⃣ Transform features and persist NumPy arrays
# --------------------------------------------------------------
def _save_numpy(arr: np.ndarray, path: Path) -> None:
    np.save(path, arr)


print("💾 Transforming data and saving NumPy arrays...")
X_train_proc = pipeline.transform(X_train.values)
X_val_proc = pipeline.transform(X_val.values)
X_test_proc = pipeline.transform(X_test.values)

_save_numpy(X_train_proc, PROCESSED_DIR / "X_train.npy")
_save_numpy(y_train, PROCESSED_DIR / "y_train.npy")
_save_numpy(X_val_proc, PROCESSED_DIR / "X_val.npy")
_save_numpy(y_val, PROCESSED_DIR / "y_val.npy")
_save_numpy(X_test_proc, PROCESSED_DIR / "X_test.npy")
_save_numpy(y_test, PROCESSED_DIR / "y_test.npy")
print(f"✅ Processed arrays saved under {PROCESSED_DIR}")

# Persist the pipeline itself
pipeline_path = TRAINED_MODELS_DIR / "preprocess_pipeline.pkl"
joblib.dump(pipeline, pipeline_path)
print(f"🧩 Preprocessing pipeline saved to {pipeline_path}")

# --------------------------------------------------------------
# 🔟 Hyper‑parameters for TabNet (fixed as per spec)
# --------------------------------------------------------------
hyperparams = {
    "n_d": 32,
    "n_a": 32,
    "n_steps": 5,
    "gamma": 1.5,
    "lambda_sparse": 1e-5,
    "optimizer_lr": 2e-2,
    "optimizer_weight_decay": 1e-2,
    "batch_size": 64,
    "max_epochs": 200,
    "early_stopping_patience": 10,
    "seed": SEED,
}
print("🔧 Using TabNet hyper‑parameters:", hyperparams)

# --------------------------------------------------------------
# 1️⃣1️⃣ Train TabNetRegressor
# --------------------------------------------------------------
print("🚂 Starting TabNet training...")
model = TabNetRegressor(
    n_d=hyperparams["n_d"],
    n_a=hyperparams["n_a"],
    n_steps=hyperparams["n_steps"],
    gamma=hyperparams["gamma"],
    lambda_sparse=hyperparams["lambda_sparse"],
    optimizer_params={
        "lr": hyperparams["optimizer_lr"],
        "weight_decay": hyperparams["optimizer_weight_decay"],
    },
    seed=hyperparams["seed"],
    verbose=0,
)

start_time = time.time()
model.fit(
    X_train=X_train_proc,
    y_train=y_train,
    eval_set=[(X_val_proc, y_val)],
    eval_name=["val"],
    eval_metric=["rmse"],
    max_epochs=hyperparams["max_epochs"],
    patience=hyperparams["early_stopping_patience"],
    batch_size=hyperparams["batch_size"],
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False,
)
train_time_sec = time.time() - start_time
print(f"⏱️ Training finished in {train_time_sec:.1f} seconds")

# --------------------------------------------------------------
# 1️⃣2️⃣ Validation metrics
# --------------------------------------------------------------
val_pred = model.predict(X_val_proc)
if log_target:
    val_pred = np.expm1(val_pred)  # invert log1p

val_rmse = mean_squared_error(y_val, val_pred, squared=False)
val_mae = mean_absolute_error(y_val, val_pred)
val_r2 = r2_score(y_val, val_pred)

print("\n📊 Validation metrics")
print(f"RMSE: {val_rmse:.4f}")
print(f"MAE : {val_mae:.4f}")
print(f"R²  : {val_r2:.4f}")

# --------------------------------------------------------------
# 1️⃣3️⃣ Test‑set evaluation
# --------------------------------------------------------------
test_pred = model.predict(X_test_proc)
if log_target:
    test_pred = np.expm1(test_pred)

test_rmse = mean_squared_error(y_test, test_pred, squared=False)
test_mae = mean_absolute_error(y_test, test_pred)
test_r2 = r2_score(y_test, test_pred)

print("\n📊 Test metrics")
print(f"RMSE: {test_rmse:.4f}")
print(f"MAE : {test_mae:.4f}")
print(f"R²  : {test_r2:.4f}")

# --------------------------------------------------------------
# 1️⃣4️⃣ Persist the trained model and a JSON report
# --------------------------------------------------------------
model_path = TRAINED_MODELS_DIR / "tabnet_regressor.pt"
torch.save(model.state_dict(), model_path)
print(f"💾 Model state_dict saved to {model_path}")

metrics_report = {
    "model_name": "TabNetRegressor",
    "validation": {"rmse": val_rmse, "mae": val_mae, "r2": val_r2},
    "test": {"rmse": test_rmse, "mae": test_mae, "r2": test_r2},
    "train_time_sec": train_time_sec,
    "log_target": log_target,
}
metrics_path = TRAINED_MODELS_DIR / "test_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics_report, f, indent=2)
print(f"📄 Test‑set metrics written to {metrics_path}")

print("\n✅ Pipeline completed successfully. All artefacts are stored under:")
print(f"   {TRAINED_MODELS_DIR}")

# --------------------------------------------------------------
# 1️⃣5️⃣ Helper for quick inference (optional)
# --------------------------------------------------------------
def quick_predict(df_input: pd.DataFrame) -> np.ndarray:
    """
    Apply the saved preprocessing pipeline and the trained TabNet model
    to a new pandas DataFrame and return predictions as a NumPy array.
    """
    # Lazy load artefacts
    pipe = joblib.load(pipeline_path)
    state = torch.load(model_path, map_location=torch.device("cpu"))
    tabnet = TabNetRegressor(
        n_d=hyperparams["n_d"],
        n_a=hyperparams["n_a"],
        n_steps=hyperparams["n_steps"],
        gamma=hyperparams["gamma"],
        lambda_sparse=hyperparams["lambda_sparse"],
        optimizer_params={
            "lr": hyperparams["optimizer_lr"],
            "weight_decay": hyperparams["optimizer_weight_decay"],
        },
        seed=hyperparams["seed"],
        verbose=0,
    )
    tabnet.load_state_dict(state)
    tabnet.eval()

    X_new = pipe.transform(df_input.values)
    preds = tabnet.predict(X_new)
    if log_target:
        preds = np.expm1(preds)
    return preds

# Example usage (uncomment to test):
# sample = pd.DataFrame(X_test[:5], columns=X.columns)
# print("Sample predictions:", quick_predict(sample))
