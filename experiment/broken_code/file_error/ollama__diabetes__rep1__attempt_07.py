
"""
End‑to‑end Tabular Regression pipeline for the Diabetes progression dataset.

Steps performed:
1️⃣ Load raw CSV (data/raw/diabetes.csv) – relative to the repository root.
2️⃣ Split into train / validation / test (70 % / 20 % / 10 %) with a fixed seed.
3️⃣ Build a preprocessing pipeline:
   - Median imputation
   - Winsorisation (clip to 1st / 99th percentile)
   - Robust scaling
4️⃣ Optionally log‑transform the target if its skewness > 1.0.
5️⃣ Train a PyTorch‑TabNet regressor.
6️⃣ Evaluate on the held‑out test set (RMSE, MAE, R²).
7️⃣ Persist artefacts under ./agent_workspace/trained_models:
   - preprocess_pipeline.joblib
   - preprocess_meta.json   (log_target flag, target column, feature list)
   - tabnet_regressor.pt      (torch state_dict)
   - test_metrics.json        (evaluation results)

The script assumes that the required packages (including `pytorch-tabnet`)
are already installed in the active environment – no on‑the‑fly installation is
performed.
"""

import json
import random
import warnings
from pathlib import Path

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

# -------------------------------------------------------------------------
# 1️⃣ Configuration & reproducibility
# -------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

RAW_DATA_PATH = Path("data/raw/diabetes.csv")          # relative to repo root
TARGET_COL = "target"                                 # column name in CSV
SAVE_DIR = Path("./agent_workspace/trained_models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Hyper‑parameters for TabNet (mirrors the spec)
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
    "virtual_batch_size": 128,
    "seed": SEED,
    # device will be set automatically below
}

# -------------------------------------------------------------------------
# 2️⃣ Custom transformer – Winsorisation (clip to 1st / 99th percentile)
# -------------------------------------------------------------------------
class Winsoriser(BaseEstimator, TransformerMixin):
    """Clip numeric features to the 1st and 99th percentiles computed on the training data."""
    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_q = lower_quantile
        self.upper_q = upper_quantile
        self.lower_ = None
        self.upper_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.lower_ = np.quantile(X, self.lower_q, axis=0)
        self.upper_ = np.quantile(X, self.upper_q, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.clip(X, self.lower_, self.upper_)

# -------------------------------------------------------------------------
# 3️⃣ Load raw data
# -------------------------------------------------------------------------
if not RAW_DATA_PATH.is_file():
    raise FileNotFoundError(f"Raw CSV not found at {RAW_DATA_PATH.resolve()}")

df = pd.read_csv(RAW_DATA_PATH)

if TARGET_COL not in df.columns:
    raise KeyError(f"Target column '{TARGET_COL}' not present in the dataset.")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].values.astype(np.float32)

# -------------------------------------------------------------------------
# 4️⃣ Train / validation / test split (70 / 20 / 10)
# -------------------------------------------------------------------------
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

# -------------------------------------------------------------------------
# 5️⃣ Build preprocessing pipeline (fit on training data only)
# -------------------------------------------------------------------------
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

preprocess.fit(X_train)   # fit on training data only

# -------------------------------------------------------------------------
# 6️⃣ Optional log‑transform of the target (based on skewness)
# -------------------------------------------------------------------------
log_target = False
skewness = pd.Series(y_train).skew()
if abs(skewness) > 1.0:
    log_target = True
    y_train = np.log1p(y_train)
    y_val   = np.log1p(y_val)
    y_test  = np.log1p(y_test)

# -------------------------------------------------------------------------
# 7️⃣ Transform features
# -------------------------------------------------------------------------
X_train_proc = preprocess.transform(X_train).astype(np.float32)
X_val_proc   = preprocess.transform(X_val).astype(np.float32)
X_test_proc  = preprocess.transform(X_test).astype(np.float32)

# -------------------------------------------------------------------------
# 8️⃣ Train TabNetRegressor
# -------------------------------------------------------------------------
from pytorch_tabnet.tab_model import TabNetRegressor  # noqa: E402

device_name = "cuda" if torch.cuda.is_available() else "cpu"
TABNET_PARAMS["device_name"] = device_name

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
# 9️⃣ Evaluation on the held‑out test set
# -------------------------------------------------------------------------
tabnet.eval()
with torch.no_grad():
    y_pred_test = tabnet.predict(X_test_proc)

# Inverse log‑transform if it was applied
if log_target:
    y_pred_test = np.expm1(y_pred_test)
    y_test_original = np.expm1(y_test)   # y_test was already log‑transformed
else:
    y_test_original = y_test

rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_test))
mae  = mean_absolute_error(y_test_original, y_pred_test)
r2   = r2_score(y_test_original, y_pred_test)

print("\n=== Test set performance ===")
print(f"RMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")

# -------------------------------------------------------------------------
# 🔟 Persist artefacts
# -------------------------------------------------------------------------
# 1) Preprocessing pipeline
pipeline_path = SAVE_DIR / "preprocess_pipeline.joblib"
dump(preprocess, pipeline_path)
print(f"Saved preprocessing pipeline → {pipeline_path}")

# 2) Metadata (log_target flag, target column, feature list)
meta = {
    "target_column": TARGET_COL,
    "log_target": log_target,
    "feature_names": numeric_features,
}
meta_path = SAVE_DIR / "preprocess_meta.json"
meta_path.write_text(json.dumps(meta, indent=2))
print(f"Saved preprocessing metadata → {meta_path}")

# 3) TabNet state dict
model_path = SAVE_DIR / "tabnet_regressor.pt"
torch.save(tabnet.state_dict(), model_path)
print(f"Saved TabNet state dict → {model_path}")

# 4) Test‑set metrics
metrics = {
    "rmse": rmse,
    "mae": mae,
    "r2": r2,
    "log_target": log_target,
}
metrics_path = SAVE_DIR / "test_metrics.json"
metrics_path.write_text(json.dumps(metrics, indent=2))
print(f"Saved test metrics → {metrics_path}")

# -------------------------------------------------------------------------
# 📦 Optional inference helper (can be imported elsewhere)
# -------------------------------------------------------------------------
def _load_artifacts(base_dir: Path = SAVE_DIR):
    """Load pipeline, metadata and a ready‑to‑use TabNet model."""
    pipeline = load(base_dir / "preprocess_pipeline.joblib")
    meta = json.loads((base_dir / "preprocess_meta.json").read_text())

    # Re‑create TabNet with the same hyper‑parameters
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
    state_dict = torch.load(base_dir / "tabnet_regressor.pt", map_location=model.device)
    model.load_state_dict(state_dict)
    model.eval()
    return pipeline, meta, model


def predict(df: pd.DataFrame) -> pd.Series:
    """
    Predict the regression target for a pandas DataFrame `df`.

    The DataFrame must contain exactly the same feature columns as used during training.
    Returns a pandas Series named "prediction".
    """
    pipeline, meta, model = _load_artifacts()
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
