
"""
End‑to‑end TabNet regression pipeline for the Diabetes progression dataset.

The script performs the following steps:
1️⃣  Load the raw CSV (tries the repository location first, then the absolute
    path that was used in the previous run).
2️⃣  Split the data into train / validation / test (70 % / 20 % / 10 %).
3️⃣  Build a preprocessing pipeline:
       - median imputation
       - winsorisation (1st‑99th percentiles)
       - robust scaling
   The pipeline is fitted **only on the training split**.
4️⃣  Optionally log‑transform the target if its skewness > 1.0.
5️⃣  Train a `TabNetRegressor` (pytorch‑tabnet) with a fixed set of
    hyper‑parameters.
6️⃣  Evaluate on validation and test sets (RMSE, MAE, R²).
7️⃣  Persist the trained model, preprocessing pipeline, a small JSON meta‑file
    and a metrics report under ``./agent_workspace/trained_models``.
"""

# ----------------------------------------------------------------------
# 0️⃣ Imports & reproducibility
# ----------------------------------------------------------------------
import json
import pathlib
import pickle
import random
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from joblib import dump, load
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from torch.utils.data import TensorDataset

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------------------
# 1️⃣ Path constants (all relative to the repository root)
# ----------------------------------------------------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parents[0]   # script folder
RAW_CSV_REL = REPO_ROOT / "data" / "raw" / "diabetes.csv"
# Absolute fallback – the path that caused the previous failure
RAW_CSV_ABS = pathlib.Path(
    r"C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\diabetes.csv"
)

# Directory where all artefacts will be stored
MODEL_DIR = REPO_ROOT / "trained_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "target"

# ----------------------------------------------------------------------
# 2️⃣ Helper – Winsoriser transformer
# ----------------------------------------------------------------------
class Winsoriser(BaseEstimator, TransformerMixin):
    """Clip numeric features to the 1st and 99th percentiles (computed on training data)."""

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
# 3️⃣ Load raw data (try relative first, then absolute fallback)
# ----------------------------------------------------------------------
if RAW_CSV_REL.is_file():
    RAW_CSV = RAW_CSV_REL
elif RAW_CSV_ABS.is_file():
    RAW_CSV = RAW_CSV_ABS
else:
    raise FileNotFoundError(
        f"Raw CSV not found. Tried:\n  {RAW_CSV_REL}\n  {RAW_CSV_ABS}"
    )

df = pd.read_csv(RAW_CSV)

if TARGET_COL not in df.columns:
    raise KeyError(f"Target column '{TARGET_COL}' missing in the CSV.")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].astype(float).values

# ----------------------------------------------------------------------
# 4️⃣ Train / validation / test split (70 / 20 / 10)
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

# ----------------------------------------------------------------------
# 5️⃣ Build preprocessing pipeline (fit on training data only)
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
# 6️⃣ Optional log‑transform of the target (based on skewness)
# ----------------------------------------------------------------------
skew = pd.Series(y_train).skew()
log_target = abs(skew) > 1.0

if log_target:
    y_train = np.log1p(y_train)
    y_val   = np.log1p(y_val)
    y_test  = np.log1p(y_test)

# ----------------------------------------------------------------------
# 7️⃣ Transform features for torch
# ----------------------------------------------------------------------
X_train_proc = preprocess.transform(X_train).astype(np.float32)
X_val_proc   = preprocess.transform(X_val).astype(np.float32)
X_test_proc  = preprocess.transform(X_test).astype(np.float32)

train_dataset = TensorDataset(
    torch.from_numpy(X_train_proc), torch.from_numpy(y_train.astype(np.float32))
)
val_dataset = TensorDataset(
    torch.from_numpy(X_val_proc), torch.from_numpy(y_val.astype(np.float32))
)
test_dataset = TensorDataset(
    torch.from_numpy(X_test_proc), torch.from_numpy(y_test.astype(np.float32))
)

# ----------------------------------------------------------------------
# 8️⃣ Hyper‑parameters (fixed, proven to work well on this data)
# ----------------------------------------------------------------------
hyper = {
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

# ----------------------------------------------------------------------
# 9️⃣ Train TabNetRegressor
# ----------------------------------------------------------------------
from pytorch_tabnet.tab_model import TabNetRegressor

device = "cuda" if torch.cuda.is_available() else "cpu"

model = TabNetRegressor(
    n_d=hyper["n_d"],
    n_a=hyper["n_a"],
    n_steps=hyper["n_steps"],
    gamma=hyper["gamma"],
    lambda_sparse=hyper["lambda_sparse"],
    optimizer_lr=hyper["optimizer_lr"],
    optimizer_weight_decay=hyper["optimizer_weight_decay"],
    device_name=device,
    seed=hyper["seed"],
)

# TabNet handles its own batching; we can feed the NumPy arrays directly.
model.fit(
    X_train_proc,
    y_train,
    eval_set=[(X_val_proc, y_val)],
    eval_name=["val"],
    eval_metric=["rmse"],
    max_epochs=hyper["max_epochs"],
    patience=hyper["early_stopping_patience"],
    batch_size=hyper["batch_size"],
    virtual_batch_size=128,
    drop_last=False,
    verbose=0,
)

# ----------------------------------------------------------------------
# 10️⃣ Helper to compute metrics (inverse log‑transform if needed)
# ----------------------------------------------------------------------
def _inverse_transform(arr: np.ndarray) -> np.ndarray:
    """Re‑apply expm1 if the target was log‑transformed."""
    return np.expm1(arr) if log_target else arr


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """Return RMSE, MAE, R²."""
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean())
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


# ----------------------------------------------------------------------
# 11️⃣ Validation metrics
# ----------------------------------------------------------------------
val_pred = model.predict(X_val_proc)
val_pred = _inverse_transform(val_pred)
val_true = _inverse_transform(y_val)
val_rmse, val_mae, val_r2 = compute_metrics(val_true, val_pred)

# ----------------------------------------------------------------------
# 12️⃣ Test metrics
# ----------------------------------------------------------------------
test_pred = model.predict(X_test_proc)
test_pred = _inverse_transform(test_pred)
test_true = _inverse_transform(y_test)
test_rmse, test_mae, test_r2 = compute_metrics(test_true, test_pred)

# ----------------------------------------------------------------------
# 13️⃣ Persist artefacts
# ----------------------------------------------------------------------
# 13.1 preprocessing pipeline
pipeline_path = MODEL_DIR / "preprocess_pipeline.joblib"
dump(preprocess, pipeline_path)

# 13.2 preprocessing meta‑information (needed for inference)
meta = {
    "target_column": TARGET_COL,
    "log_target": bool(log_target),
    "feature_names": numeric_features,
}
meta_path = MODEL_DIR / "preprocess_meta.json"
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)

# 13.3 trained TabNet model (state_dict)
model_path = MODEL_DIR / "tabnet_regressor.pt"
torch.save(model.state_dict(), model_path)

# 13.4 JSON report with metrics
report = {
    "model_name": "TabNetRegressor",
    "hyperparameters": hyper,
    "validation": {"rmse": val_rmse, "mae": val_mae, "r2": val_r2},
    "test": {"rmse": test_rmse, "mae": test_mae, "r2": test_r2},
    "log_target": log_target,
}
report_path = MODEL_DIR / "test_metrics.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)

# ----------------------------------------------------------------------
# 14️⃣ Summary printed to stdout (useful for CI logs)
# ----------------------------------------------------------------------
print("\n=== TabNet Regression Summary ===")
print(f"Log‑target applied? : {log_target}")
print("\n--- Validation Metrics ---")
print(f"RMSE : {val_rmse:.4f}")
print(f"MAE  : {val_mae:.4f}")
print(f"R²   : {val_r2:.4f}")
print("\n--- Test Metrics ---")
print(f"RMSE : {test_rmse:.4f}")
print(f"MAE  : {test_mae:.4f}")
print(f"R²   : {test_r2:.4f}")
print(f"\nAll artefacts saved under: {MODEL_DIR}")

# ----------------------------------------------------------------------
# 15️⃣ Tiny inference helper (can be imported elsewhere)
# ----------------------------------------------------------------------
def predict(df: pd.DataFrame) -> pd.Series:
    """
    Predict the target for a pandas DataFrame containing the original feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain at least the columns listed in ``meta["feature_names"]``.
        Extra columns are ignored.

    Returns
    -------
    pd.Series
        Predicted values (inverse‑transformed if log_target was used).
    """
    missing = set(meta["feature_names"]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    X = df[meta["feature_names"]]
    X_proc = preprocess.transform(X).astype(np.float32)

    with torch.no_grad():
        preds = model.predict(X_proc)

    if meta["log_target"]:
        preds = np.expm1(preds)

    return pd.Series(preds, name="prediction")
