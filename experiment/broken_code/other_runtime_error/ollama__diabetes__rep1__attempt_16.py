
"""
End‑to‑end Tabular regression pipeline for the Diabetes progression dataset.

The script performs the following steps:

1️⃣  Load the raw CSV from ./data/raw/diabetes.csv (fails fast if missing).
2️⃣  Split the data into train / validation / test (70 % / 20 % / 10 %).
3️⃣  Build a preprocessing pipeline:
    • Median imputation
    • Winsorisation to the 1st / 99th percentiles
    • Robust scaling
4️⃣  (Optional) Log‑transform the target if it is highly skewed.
5️⃣  Train a TabNetRegressor (pytorch‑tabnet) with a fixed hyper‑parameter set.
6️⃣  Evaluate on validation and test sets (RMSE, MAE, R²).
7️⃣  Persist artefacts under ./agent_workspace/trained_models:
    • preprocessing pipeline (joblib)
    • preprocessing metadata (json)
    • trained TabNet state_dict (torch)
    • evaluation metrics (json)
8️⃣  Provide a small Gradio demo for interactive inference.

All paths are **relative to the repository root**, so the script can be
run from any working directory (e.g. `python scripts/run_pipeline.py`).
"""

# ----------------------------------------------------------------------
# 0️⃣  Imports & reproducibility
# ----------------------------------------------------------------------
import json
import pathlib
import random
import warnings
import pickle
import time
import sys

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
import gradio as gr

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Suppress non‑critical warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ----------------------------------------------------------------------
# 1️⃣  Define repository‑relative paths
# ----------------------------------------------------------------------
# Repository root – two levels up from this file (scripts/ folder)
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

RAW_CSV = REPO_ROOT / "data" / "raw" / "diabetes.csv"
ARTIFACT_DIR = REPO_ROOT / "agent_workspace" / "trained_models"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "target"   # column name that holds the label in the CSV

# ----------------------------------------------------------------------
# 2️⃣  Load raw data (fail fast if missing)
# ----------------------------------------------------------------------
if not RAW_CSV.is_file():
    msg = (
        f"Raw CSV not found at {RAW_CSV}\n"
        "Make sure the file exists and is placed at the path above.\n"
        "If the file lives elsewhere, copy it to that location before running the script."
    )
    sys.exit(msg)

df = pd.read_csv(RAW_CSV)

if TARGET_COL not in df.columns:
    sys.exit(f"Target column '{TARGET_COL}' not present in the CSV.")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].astype(float).values

# ----------------------------------------------------------------------
# 3️⃣  Train / validation / test split (70 / 20 / 10)
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
# 4️⃣  Custom Winsoriser (attribute names must match __init__ args)
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
# 5️⃣  Build preprocessing pipeline (fit on training data only)
# ----------------------------------------------------------------------
numeric_features = X_train.columns.tolist()   # all columns are numeric in this dataset

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
# 6️⃣  (Optional) Log‑transform the target if highly skewed
# ----------------------------------------------------------------------
skewness = pd.Series(y_train).skew()
log_target = abs(skewness) > 1.0

if log_target:
    y_train = np.log1p(y_train)
    y_val   = np.log1p(y_val)
    y_test  = np.log1p(y_test)

# ----------------------------------------------------------------------
# 7️⃣  Persist preprocessing artefacts
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
# 8️⃣  Transform splits and store as .npy (fast loading for training)
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
# 9️⃣  Train TabNetRegressor
# ----------------------------------------------------------------------
from pytorch_tabnet.tab_model import TabNetRegressor

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
# 10️⃣  Evaluation (validation & test)
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
# 11️⃣  Persist model & metrics
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

# ----------------------------------------------------------------------
# 12️⃣  Gradio demo for interactive inference
# ----------------------------------------------------------------------
# Load artefacts once (so the demo is fast)
PIPELINE = load(ARTIFACT_DIR / "preprocess_pipeline.joblib")
META = json.loads((ARTIFACT_DIR / "preprocess_meta.json").read_text())
STATE_DICT = torch.load(ARTIFACT_DIR / "tabnet_regressor.pt", map_location="cpu")

# Re‑create model for inference
INFER_MODEL = TabNetRegressor(
    n_d=hyperparams["n_d"],
    n_a=hyperparams["n_a"],
    n_steps=hyperparams["n_steps"],
    gamma=hyperparams["gamma"],
    lambda_sparse=hyperparams["lambda_sparse"],
    optimizer_lr=hyperparams["optimizer_lr"],
    optimizer_weight_decay=hyperparams["optimizer_weight_decay"],
    device_name="cpu",
    seed=hyperparams["seed"],
)
INFER_MODEL.load_state_dict(STATE_DICT)
INFER_MODEL.eval()

def predict(**kwargs):
    """
    Gradio‑compatible prediction function.
    Expects each feature as a separate named argument (matching META["feature_names"]).
    Returns the regression prediction as a plain Python float.
    """
    # Build a DataFrame from the supplied keyword arguments
    input_df = pd.DataFrame([kwargs])

    # Verify that all required columns are present
    missing = set(META["feature_names"]) - set(input_df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Preserve column order expected by the pipeline
    X = input_df[META["feature_names"]]
    X_proc = PIPELINE.transform(X).astype(np.float32)

    with torch.no_grad():
        pred = INFER_MODEL.predict(X_proc)

    if META["log_target"]:
        pred = np.expm1(pred)

    return float(pred[0])

# Dynamically build the Gradio interface from the feature list
iface = gr.Interface(
    fn=predict,
    inputs=[gr.Number(label=col) for col in META["feature_names"]],
    outputs=gr.Number(label="Prediction"),
    title="Diabetes Progression – TabNet Regressor",
    description="Enter the numeric feature values and obtain the predicted disease progression score.",
    allow_flagging="never",
)

if __name__ == "__main__":
    # Launch the demo on a free local port; set `share=True` for a public URL.
    iface.launch(share=False)
