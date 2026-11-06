
# =============================================================================
# Tabular regression pipeline – Diabetes progression (TabNetRegressor)
# =============================================================================
# This script is self‑contained: it loads the raw CSV, builds a preprocessing
# pipeline, splits the data, trains a TabNetRegressor (PyTorch‑TabNet), evaluates
# the model and finally saves all artefacts under
#   ./agent_workspace/trained_models/
# =============================================================================

import os
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from joblib import dump, load
from pytorch_tabnet.tab_model import TabNetRegressor

# -------------------------------------------------------------------------
# Configuration (you can adapt these values)
# -------------------------------------------------------------------------
SEED = 42
RAW_DATA_PATH = Path("data/raw/diabetes.csv")          # <-- adjust if needed
TARGET_COL = "target"                                 # column name in CSV
MODEL_SAVE_DIR = Path("./agent_workspace/trained_models")
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Hyper‑parameters for TabNet (same as the yaml described earlier)
TABNET_PARAMS = {
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
    "virtual_batch_size": 128,
    "device_name": "cpu",  # will be overwritten if CUDA is available
}

# -------------------------------------------------------------------------
# Utility: reproducibility
# -------------------------------------------------------------------------
def set_global_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_global_seed(SEED)

# -------------------------------------------------------------------------
# Custom transformer – Winsorisation (clip to 1st / 99th percentile)
# -------------------------------------------------------------------------
class Winsoriser(BaseEstimator, TransformerMixin):
    """Clip numeric features to the 1st and 99th percentiles computed on fit data."""
    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_q = lower_quantile
        self.upper_q = upper_quantile
        self.lower_ = None
        self.upper_ = None

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.lower_ = np.quantile(X, self.lower_q, axis=0)
        self.upper_ = np.quantile(X, self.upper_q, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X)
        return np.clip(X, self.lower_, self.upper_)

# -------------------------------------------------------------------------
# Step 1 – Load raw data
# -------------------------------------------------------------------------
if not RAW_DATA_PATH.is_file():
    raise FileNotFoundError(f"Raw CSV not found at {RAW_DATA_PATH.resolve()}")
df = pd.read_csv(RAW_DATA_PATH)

if TARGET_COL not in df.columns:
    raise KeyError(f"Target column '{TARGET_COL}' not present in the dataset.")

# Separate features / target
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].values.astype(np.float32)

# -------------------------------------------------------------------------
# Step 2 – Train / validation / test split (70 / 20 / 10)
# -------------------------------------------------------------------------
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.10, random_state=SEED, shuffle=True
)
# 20% of the remaining 90% → 0.20 / 0.90 ≈ 0.2222
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.2222,
    random_state=SEED,
    shuffle=True,
)

# -------------------------------------------------------------------------
# Step 3 – Build preprocessing pipeline (fit on training data only)
# -------------------------------------------------------------------------
numeric_features = X_train.columns.tolist()  # all columns are numeric in this dataset

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("winsor", Winsoriser(lower_quantile=0.01, upper_quantile=0.99)),
        ("scaler", RobustScaler()),
    ]
)

preprocess = ColumnTransformer(
    transformers=[("num", numeric_transformer, numeric_features)],
    remainder="drop",  # no non‑numeric columns expected
)

# Fit pipeline
preprocess.fit(X_train)

# -------------------------------------------------------------------------
# Step 4 – Optional log‑transform of the target (based on skewness)
# -------------------------------------------------------------------------
log_target = False
skewness = pd.Series(y_train).skew()
if abs(skewness) > 1.0:
    log_target = True
    y_train = np.log1p(y_train)
    y_val = np.log1p(y_val)
    y_test = np.log1p(y_test)

# -------------------------------------------------------------------------
# Step 5 – Transform features
# -------------------------------------------------------------------------
X_train_proc = preprocess.transform(X_train)
X_val_proc = preprocess.transform(X_val)
X_test_proc = preprocess.transform(X_test)

# Convert to float32 for TabNet
X_train_proc = X_train_proc.astype(np.float32)
X_val_proc = X_val_proc.astype(np.float32)
X_test_proc = X_test_proc.astype(np.float32)

# -------------------------------------------------------------------------
# Step 6 – Train TabNetRegressor
# -------------------------------------------------------------------------
# Update device name according to availability
TABNET_PARAMS["device_name"] = "cuda" if torch.cuda.is_available() else "cpu"

tabnet = TabNetRegressor(
    n_d=TABNET_PARAMS["n_d"],
    n_a=TABNET_PARAMS["n_a"],
    n_steps=TABNET_PARAMS["n_steps"],
    gamma=TABNET_PARAMS["gamma"],
    lambda_sparse=TABNET_PARAMS["lambda_sparse"],
    optimizer_lr=TABNET_PARAMS["optimizer_lr"],
    optimizer_weight_decay=TABNET_PARAMS["optimizer_weight_decay"],
    device_name=TABNET_PARAMS["device_name"],
    seed=TABNET_PARAMS["seed"],
)

tabnet.fit(
    X_train_proc,
    y_train,
    eval_set=[(X_val_proc, y_val)],
    eval_name=["val"],
    eval_metric=["rmse"],
    max_epochs=TABNET_PARAMS["max_epochs"],
    patience=TABNET_PARAMS["early_stopping_patience"],
    batch_size=TABNET_PARAMS["batch_size"],
    virtual_batch_size=TABNET_PARAMS["virtual_batch_size"],
    drop_last=False,
    verbose=0,
)

# -------------------------------------------------------------------------
# Step 7 – Evaluation on the held‑out test set
# -------------------------------------------------------------------------
tabnet.eval()
with torch.no_grad():
    y_pred_test = tabnet.predict(X_test_proc)

# Inverse log‑transform if we applied it earlier
if log_target:
    y_pred_test = np.expm1(y_pred_test)
    y_test_original = np.expm1(y_test)  # y_test was already log‑transformed
else:
    y_test_original = y_test

# Metrics
rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_test))
mae = mean_absolute_error(y_test_original, y_pred_test)
r2 = r2_score(y_test_original, y_pred_test)

print("\n=== Test set performance ===")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")

# -------------------------------------------------------------------------
# Step 8 – Save artefacts
# -------------------------------------------------------------------------
# 1) Preprocessing pipeline
pipeline_path = MODEL_SAVE_DIR / "preprocess_pipeline.joblib"
dump(preprocess, pipeline_path)
print(f"Saved preprocessing pipeline to {pipeline_path}")

# 2) Model state dict
model_path = MODEL_SAVE_DIR / "tabnet_regressor.pt"
torch.save(tabnet.state_dict(), model_path)
print(f"Saved TabNet state dict to {model_path}")

# 3) Metadata (log_target flag & feature list)
meta = {
    "target_column": TARGET_COL,
    "log_target": log_target,
    "feature_names": numeric_features,
}
meta_path = MODEL_SAVE_DIR / "preprocess_meta.json"
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"Saved preprocessing metadata to {meta_path}")

# 4) Evaluation report (optional but handy)
report = {
    "rmse": rmse,
    "mae": mae,
    "r2": r2,
    "log_target": log_target,
}
report_path = MODEL_SAVE_DIR / "test_metrics.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
print(f"Saved test metrics to {report_path}")

# -------------------------------------------------------------------------
# Optional: simple inference helper (can be imported elsewhere)
# -------------------------------------------------------------------------
def load_artifacts(base_dir: Path = MODEL_SAVE_DIR):
    """Load preprocessing pipeline, metadata and TabNet model."""
    pipeline = load(base_dir / "preprocess_pipeline.joblib")
    meta = json.loads((base_dir / "preprocess_meta.json").read_text())
    # Re‑create TabNet with same hyper‑parameters
    model = TabNetRegressor(
        n_d=TABNET_PARAMS["n_d"],
        n_a=TABNET_PARAMS["n_a"],
        n_steps=TABNET_PARAMS["n_steps"],
        gamma=TABNET_PARAMS["gamma"],
        lambda_sparse=TABNET_PARAMS["lambda_sparse"],
        optimizer_lr=TABNET_PARAMS["optimizer_lr"],
        optimizer_weight_decay=TABNET_PARAMS["optimizer_weight_decay"],
        device_name=TABNET_PARAMS["device_name"],
        seed=TABNET_PARAMS["seed"],
    )
    model.load_state_dict(torch.load(base_dir / "tabnet_regressor.pt", map_location=model.device))
    model.eval()
    return pipeline, meta, model


def predict(df: pd.DataFrame) -> pd.Series:
    """
    Predict regression target for a pandas DataFrame `df`.
    The DataFrame must contain exactly the same feature columns as used during training.
    Returns a pandas Series named "prediction".
    """
    pipeline, meta, model = load_artifacts()
    missing = set(meta["feature_names"]) - set(df.columns)
    if missing:
        raise ValueError(f"The following required columns are missing: {sorted(missing)}")
    # Preserve column order expected by the pipeline
    X = df[meta["feature_names"]]
    X_proc = pipeline.transform(X).astype(np.float32)
    with torch.no_grad():
        preds = model.predict(X_proc)
    if meta["log_target"]:
        preds = np.expm1(preds)
    return pd.Series(preds, name="prediction")

# -------------------------------------------------------------------------
# End of script
# -------------------------------------------------------------------------
