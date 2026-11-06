
"""
diabetes_tabnet_pipeline.py

End‑to‑end script for the **tabular regression** task described in the
project specification. It performs:

1.  Data ingestion
2.  Train/validation/test split (70 % / 20 % / 10 %)
3.  Pre‑processing pipeline
    – Median imputation
    – Winsorisation (1st‑99th percentile on the training set)
    – Robust scaling
    – Optional log‑transform of the target
4.  Model training (TabNetRegressor – PyTorch)
5.  Validation & test evaluation (RMSE, MAE, R²)
6.  Persistence of artefacts:
    * `preprocess_pipeline.pkl`
    * `preprocess_meta.json` (log‑target flag)
    * `tabnet_regressor.pt` (model state_dict)
    * `test_metrics.json`
    All artefacts are stored under ``./agent_workspace/trained_models`` so
    they can be consumed by downstream services (FastAPI, Gradio, etc.).

The script is **self‑contained** – run it from the repository root after the
raw CSV has been placed at ``data/raw/diabetes.csv`` (or copied there with
the helper script provided in the spec).

Dependencies (see ``environment.yml``):
    pandas, numpy, scikit‑learn, joblib, pyyaml,
    torch, pytorch‑tabnet
"""

import os
import json
import yaml
import time
import random
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import torch
from pytorch_tabnet.tab_model import TabNetRegressor

# --------------------------------------------------------------
# 0️⃣  Global settings & reproducibility
# --------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# --------------------------------------------------------------
# 1️⃣  Helper: Winsoriser transformer (1st‑99th percentile)
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
        # X is expected to be a NumPy array or a pandas DataFrame
        X_arr = np.asarray(X)
        self.lower_ = np.quantile(X_arr, self.lower_quantile, axis=0)
        self.upper_ = np.quantile(X_arr, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        X_arr = np.asarray(X)
        X_clipped = np.clip(X_arr, self.lower_, self.upper_)
        return X_clipped

# --------------------------------------------------------------
# 2️⃣  Paths
# --------------------------------------------------------------
ROOT = Path(__file__).parent  # repository root (assuming script lives in repo root)
RAW_CSV = ROOT / "data" / "raw" / "diabetes.csv"

# Output directories (must exist)
ARTIFACT_ROOT = ROOT / "agent_workspace" / "trained_models"
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_DIR = ROOT / "data" / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------
# 3️⃣  Load raw data
# --------------------------------------------------------------
print("🔎 Loading raw CSV...")
df = pd.read_csv(RAW_CSV)

# Identify target and features
TARGET_COL = "target"
if TARGET_COL not in df.columns:
    raise KeyError(f"Target column '{TARGET_COL}' not found in CSV.")
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].astype(float).values  # ensure numeric

# --------------------------------------------------------------
# 4️⃣  Train/validation/test split (70/20/10)
# --------------------------------------------------------------
print("🔀 Creating train/val/test splits...")
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.10, random_state=SEED, shuffle=True
)
# 0.2222 ≈ 20/(70+20)
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

# --------------------------------------------------------------
# 5️⃣  Build preprocessing pipeline (fit on X_train only)
# --------------------------------------------------------------
print("⚙️ Building preprocessing pipeline...")
pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("winsoriser", Winsoriser(lower_quantile=0.01, upper_quantile=0.99)),
        ("scaler", RobustScaler()),
    ]
)

pipeline.fit(X_train.values)  # fit only on training data

# --------------------------------------------------------------
# 6️⃣  Optional log‑transform of the target
# --------------------------------------------------------------
log_target = False
skew = pd.Series(y_train).skew()
if abs(skew) > 1.0:
    print(f"📈 Target skew = {skew:.3f} > 1.0 → applying log1p transform.")
    y_train = np.log1p(y_train)
    y_val = np.log1p(y_val)
    y_test = np.log1p(y_test)
    log_target = True
else:
    print(f"📈 Target skew = {skew:.3f} (no log transform).")

# Store the flag for inference time
meta_path = ARTIFACT_ROOT / "preprocess_meta.json"
with open(meta_path, "w") as f:
    json.dump({"log_target": log_target, "target_column": TARGET_COL}, f, indent=2)

# --------------------------------------------------------------
# 7️⃣  Transform features and persist NumPy arrays
# --------------------------------------------------------------
def _save_numpy(arr, path):
    np.save(path, arr)

print("💾 Transforming and saving processed arrays...")
X_train_proc = pipeline.transform(X_train.values)
X_val_proc = pipeline.transform(X_val.values)
X_test_proc = pipeline.transform(X_test.values)

_save_numpy(X_train_proc, PROCESSED_DIR / "X_train.npy")
_save_numpy(y_train, PROCESSED_DIR / "y_train.npy")
_save_numpy(X_val_proc, PROCESSED_DIR / "X_val.npy")
_save_numpy(y_val, PROCESSED_DIR / "y_val.npy")
_save_numpy(X_test_proc, PROCESSED_DIR / "X_test.npy")
_save_numpy(y_test, PROCESSED_DIR / "y_test.npy")

# Persist the pipeline itself
pipeline_path = ARTIFACT_ROOT / "preprocess_pipeline.pkl"
joblib.dump(pipeline, pipeline_path)

# --------------------------------------------------------------
# 8️⃣  Load hyper‑parameters from YAML
# --------------------------------------------------------------
hyperparams_path = ROOT / "config" / "tabnet_hyperparams.yaml"
if not hyperparams_path.is_file():
    raise FileNotFoundError(f"Hyper‑parameter file not found: {hyperparams_path}")

with open(hyperparams_path, "r") as f:
    cfg = yaml.safe_load(f)

hp = cfg["hyperparameters"]
print("🔧 Hyper‑parameters loaded:", hp)

# --------------------------------------------------------------
# 9️⃣  Train TabNetRegressor
# --------------------------------------------------------------
print("🚂 Starting TabNet training...")
model = TabNetRegressor(
    n_d=hp["n_d"],
    n_a=hp["n_a"],
    n_steps=hp["n_steps"],
    gamma=hp["gamma"],
    lambda_sparse=hp["lambda_sparse"],
    optimizer_params=dict(
        lr=hp["optimizer_lr"], weight_decay=hp["optimizer_weight_decay"]
    ),
    seed=hp["seed"],
    verbose=0,
)

start_time = time.time()
model.fit(
    X_train=X_train_proc,
    y_train=y_train,
    eval_set=[(X_val_proc, y_val)],
    eval_name=["val"],
    eval_metric=["rmse"],
    max_epochs=hp["max_epochs"],
    patience=hp["early_stopping_patience"],
    batch_size=hp["batch_size"],
    virtual_batch_size=128,
    num_workers=0,
    drop_last=False,
)
train_time_sec = time.time() - start_time
print(f"⏱️ Training completed in {train_time_sec:.1f}s")

# --------------------------------------------------------------
# 10️⃣  Validation metrics
# --------------------------------------------------------------
val_pred = model.predict(X_val_proc)
if log_target:
    # Inverse of log1p
    val_pred = np.expm1(val_pred)

val_rmse = mean_squared_error(y_val, val_pred, squared=False)
val_mae = mean_absolute_error(y_val, val_pred)
val_r2 = r2_score(y_val, val_pred)

print("\n📊 Validation metrics")
print(f"RMSE: {val_rmse:.4f}")
print(f"MAE : {val_mae:.4f}")
print(f"R²  : {val_r2:.4f}")

# --------------------------------------------------------------
# 11️⃣  Test‑set evaluation
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
# 12️⃣  Persist the trained model and evaluation report
# --------------------------------------------------------------
model_path = ARTIFACT_ROOT / "tabnet_regressor.pt"
torch.save(model.state_dict(), model_path)
print(f"💾 Model state_dict saved to {model_path}")

# Test‑metrics JSON
metrics_report = {
    "model_name": "TabNetRegressor",
    "validation": {"rmse": val_rmse, "mae": val_mae, "r2": val_r2},
    "test": {"rmse": test_rmse, "mae": test_mae, "r2": test_r2},
    "train_time_sec": train_time_sec,
    "log_target": log_target,
}
metrics_path = ARTIFACT_ROOT / "test_metrics.json"
with open(metrics_path, "w") as f:
    json.dump(metrics_report, f, indent=2)
print(f"📄 Test metrics written to {metrics_path}")

print("\n✅ Pipeline finished successfully. All artefacts are stored under:")
print(f"   {ARTIFACT_ROOT}")

# --------------------------------------------------------------
# 13️⃣  (Optional) Simple inference helper for quick sanity‑check
# --------------------------------------------------------------
def quick_predict(df_input: pd.DataFrame) -> np.ndarray:
    """
    Convenience wrapper that applies the saved preprocessing pipeline
    and the trained TabNet model to a new pandas DataFrame.
    """
    # Load artefacts (lazy – only when the function is called)
    pipe = joblib.load(pipeline_path)
    state = torch.load(model_path, map_location=torch.device("cpu"))
    tabnet = TabNetRegressor(
        n_d=hp["n_d"],
        n_a=hp["n_a"],
        n_steps=hp["n_steps"],
        gamma=hp["gamma"],
        lambda_sparse=hp["lambda_sparse"],
        optimizer_params=dict(
            lr=hp["optimizer_lr"], weight_decay=hp["optimizer_weight_decay"]
        ),
        seed=hp["seed"],
        verbose=0,
    )
    tabnet.load_state_dict(state)
    tabnet.eval()

    # Transform
    X_new = pipe.transform(df_input.values)
    preds = tabnet.predict(X_new)
    if log_target:
        preds = np.expm1(preds)
    return preds

# Example usage (uncomment to test):
# sample = X_test.head(5)  # raw (unprocessed) test rows
# print("Sample predictions:", quick_predict(sample))
