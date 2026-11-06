
"""
End‑to‑end Tabular regression pipeline for the Diabetes progression dataset.

The script performs the following steps:

1. Load the raw CSV from ./data/raw/diabetes.csv (fails fast if missing).
2. Split the data into train / validation / test (70 % / 20 % / 10 %).
3. Build a preprocessing pipeline:
   • Median imputation
   • Winsorisation to the 1st / 99th percentiles
   • Robust scaling
4. (Optional) Log‑transform the target if it is highly skewed.
5. Train a TabNetRegressor (pytorch‑tabnet) with a fixed hyper‑parameter set.
6. Evaluate on validation and test sets (RMSE, MAE, R²).
7. Persist artefacts under ./agent_workspace/trained_models:
   • preprocessing pipeline (joblib)
   • preprocessing metadata (json)
   • trained TabNet state_dict (torch)
   • split indices (pickle)
   • evaluation metrics (json)
"""

# ----------------------------------------------------------------------
# 0️⃣ Imports & reproducibility
# ----------------------------------------------------------------------
import json
import pathlib
import random
import warnings
import pickle
import time
import sys
import subprocess

import numpy as np
import pandas as pd
import torch
from joblib import dump, load
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

# ----------------------------------------------------------------------
# 1️⃣ Ensure pytorch‑tabnet is available
# ----------------------------------------------------------------------
def _install_and_import(package: str, import_name: str):
    """Install a package with pip (if missing) and import it."""
    try:
        return __import__(import_name)
    except ModuleNotFoundError:
        print(f"[INFO] Package '{package}' not found. Installing via pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return __import__(import_name)


# Import TabNetRegressor; install on‑the‑fly if the environment lacks it.
tabnet_mod = _install_and_import("pytorch-tabnet", "pytorch_tabnet")
TabNetRegressor = tabnet_mod.tab_model.TabNetRegressor

# ----------------------------------------------------------------------
# 2️⃣ Reproducibility
# ----------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------------------
# 3️⃣ Repository‑relative paths
# ----------------------------------------------------------------------
# This script is expected to be run from the repository root.
REPO_ROOT = pathlib.Path(__file__).resolve().parent

RAW_CSV = REPO_ROOT / "data" / "raw" / "diabetes.csv"
ARTIFACT_DIR = REPO_ROOT / "agent_workspace" / "trained_models"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "target"   # name of the label column in the CSV

# ----------------------------------------------------------------------
# 4️⃣ Load raw data (fail fast if missing)
# ----------------------------------------------------------------------
if not RAW_CSV.is_file():
    sys.exit(
        f"[ERROR] Raw CSV not found at {RAW_CSV}\n"
        "Make sure the file exists and is placed at the path above."
    )

df = pd.read_csv(RAW_CSV)

if TARGET_COL not in df.columns:
    sys.exit(f"[ERROR] Target column '{TARGET_COL}' not present in the CSV.")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].astype(float).values

# ----------------------------------------------------------------------
# 5️⃣ Train / validation / test split (70 / 20 / 10)
# ----------------------------------------------------------------------
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.10, random_state=SEED, shuffle=True
)

# 20 % of the remaining 90 % → 0.20 / 0.90 ≈ 0.2222
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.2222,
    random_state=SEED,
    shuffle=True,
)

# Persist split indices for reproducibility
split_indices = {
    "train_idx": X_train.index.tolist(),
    "val_idx": X_val.index.tolist(),
    "test_idx": X_test.index.tolist(),
}
with open(ARTIFACT_DIR / "split_indices.pkl", "wb") as f:
    pickle.dump(split_indices, f)

# ----------------------------------------------------------------------
# 6️⃣ Custom Winsoriser (attribute names must match __init__ args)
# ----------------------------------------------------------------------
class Winsoriser(BaseEstimator, TransformerMixin):
    """Clip numeric features to the 1st and 99th percentiles."""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile
        self.lower_ = None
        self.upper_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.lower_ = np.quantile(X, self.lower_quantile, axis=0)
        self.upper_ = np.quantile(X, self.upper_quantile, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.clip(X, self.lower_, self.upper_)

# ----------------------------------------------------------------------
# 7️⃣ Build preprocessing pipeline (fit on training data only)
# ----------------------------------------------------------------------
numeric_features = X_train.columns.tolist()   # all columns are numeric

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("winsor", Winsoriser(lower_quantile=0.01, upper_quantile=0.99)),
        ("scaler", RobustScaler()),
    ]
)

preprocess = ColumnTransformer(
    transformers=[("num", numeric_transformer, numeric_features)],
    remainder="drop",
)

preprocess.fit(X_train)   # IMPORTANT: fit on TRAIN ONLY

# ----------------------------------------------------------------------
# 8️⃣ Optional log‑transform of the target
# ----------------------------------------------------------------------
skewness = pd.Series(y_train).skew()
log_target = abs(skewness) > 1.0

if log_target:
    y_train = np.log1p(y_train)
    y_val   = np.log1p(y_val)
    y_test  = np.log1p(y_test)

# ----------------------------------------------------------------------
# 9️⃣ Persist preprocessing artefacts
# ----------------------------------------------------------------------
meta = {
    "target_column": TARGET_COL,
    "log_target": bool(log_target),
    "feature_names": numeric_features,
}
with open(ARTIFACT_DIR / "preprocess_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

dump(preprocess, ARTIFACT_DIR / "preprocess_pipeline.joblib")

# ----------------------------------------------------------------------
# 10️⃣ Transform splits and store as .npy (fast loading for training)
# ----------------------------------------------------------------------
X_train_proc = preprocess.transform(X_train).astype(np.float32)
X_val_proc   = preprocess.transform(X_val).astype(np.float32)
X_test_proc  = preprocess.transform(X_test).astype(np.float32)

np.save(ARTIFACT_DIR / "X_train.npy", X_train_proc)
np.save(ARTIFACT_DIR / "y_train.npy", y_train.astype(np.float32))
np.save(ARTIFACT_DIR / "X_val.npy",   X_val_proc)
np.save(ARTIFACT_DIR / "y_val.npy",   y_val.astype(np.float32))
np.save(ARTIFACT_DIR / "X_test.npy",  X_test_proc)
np.save(ARTIFACT_DIR / "y_test.npy",  y_test.astype(np.float32))

# ----------------------------------------------------------------------
# 11️⃣ Train TabNetRegressor
# ----------------------------------------------------------------------
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

device = "cuda" if torch.cuda.is_available() else "cpu"

model = TabNetRegressor(
    n_d=hyperparams["n_d"],
    n_a=hyperparams["n_a"],
    n_steps=hyperparams["n_steps"],
    gamma=hyperparams["gamma"],
    lambda_sparse=hyperparams["lambda_sparse"],
    optimizer_lr=hyperparams["optimizer_lr"],
    optimizer_weight_decay=hyperparams["optimizer_weight_decay"],
    device_name=device,
    seed=hyperparams["seed"],
)

print("[INFO] Starting TabNet training ...")
start_time = time.time()
model.fit(
    X_train_proc,
    y_train,
    eval_set=[(X_val_proc, y_val)],
    eval_name=["val"],
    eval_metric=["rmse"],
    max_epochs=hyperparams["max_epochs"],
    patience=hyperparams["early_stopping_patience"],
    batch_size=hyperparams["batch_size"],
    virtual_batch_size=128,
    drop_last=False,
    verbose=0,
)
train_time_sec = time.time() - start_time
print(f"[INFO] Training finished in {train_time_sec:.1f} seconds")

# ----------------------------------------------------------------------
# 12️⃣ Evaluation (validation & test)
# ----------------------------------------------------------------------
def _inverse_log(arr):
    """Undo the optional log‑transform."""
    return np.expm1(arr) if log_target else arr

def _metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "mae": mae, "r2": r2}

# Validation
val_pred = model.predict(X_val_proc)
val_pred = _inverse_log(val_pred)
val_true = _inverse_log(y_val)
val_metrics = _metrics(val_true, val_pred)

# Test
test_pred = model.predict(X_test_proc)
test_pred = _inverse_log(test_pred)
test_true = _inverse_log(y_test)
test_metrics = _metrics(test_true, test_pred)

# ----------------------------------------------------------------------
# 13️⃣ Persist model & metrics
# ----------------------------------------------------------------------
torch.save(model.state_dict(), ARTIFACT_DIR / "tabnet_regressor.pt")

report = {
    "model_name": "TabNetRegressor",
    "validation": val_metrics,
    "test": test_metrics,
    "train_time_sec": train_time_sec,
    "log_target": log_target,
}
with open(ARTIFACT_DIR / "test_metrics.json", "w") as f:
    json.dump(report, f, indent=2)

print("\nValidation metrics:")
print(json.dumps(val_metrics, indent=2))
print("\nTest metrics:")
print(json.dumps(test_metrics, indent=2))
