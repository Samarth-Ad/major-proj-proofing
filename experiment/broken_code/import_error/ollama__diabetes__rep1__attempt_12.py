
"""
End‑to‑end Tabular Regression Pipeline for the Diabetes progression dataset.

All artefacts are saved under the repository root (relative paths).  
The script assumes that the required packages (including `pytorch-tabnet`) are
already installed in the active environment – it **does not** try to install
them at runtime.

Run from the repository root:

    python scripts/run_diabetes_tabnet.py
"""

import json
import pathlib
import random
import sys
import time
import warnings
import pickle

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

# ----------------------------------------------------------------------
# 0️⃣  Global reproducibility & constants
# ----------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
warnings.filterwarnings("ignore", category=UserWarning)

# Repository‑relative paths
ROOT = pathlib.Path(__file__).resolve().parents[0]          # folder containing this script
RAW_CSV = ROOT.parent / "data" / "raw" / "diabetes.csv"
PROCESSED_DIR = ROOT.parent / "data" / "processed"
MODEL_DIR = ROOT.parent / "agent_workspace" / "trained_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "target"          # column name used in the CSV
LOG_TARGET_THRESHOLD = 1.0    # absolute skewness above which we log‑transform

# ----------------------------------------------------------------------
# 1️⃣  Verify that the required library is available
# ----------------------------------------------------------------------
try:
    from pytorch_tabnet.tab_model import TabNetRegressor
except ModuleNotFoundError as exc:  # pragma: no cover
    sys.stderr.write(
        "ERROR: The package `pytorch-tabnet` is not installed in the current "
        "environment. Please install it via the provided `environment.yml` "
        "before running this script.\n"
    )
    raise exc

# ----------------------------------------------------------------------
# 2️⃣  Custom Winsoriser – attribute names must match __init__ args
# ----------------------------------------------------------------------
class Winsoriser(BaseEstimator, TransformerMixin):
    """Clip numeric features to the lower/upper quantiles (computed on fit data)."""

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
# 3️⃣  Load raw data (fail fast if missing)
# ----------------------------------------------------------------------
if not RAW_CSV.is_file():
    raise FileNotFoundError(f"Raw CSV not found at {RAW_CSV.resolve()}")

df = pd.read_csv(RAW_CSV)

if TARGET_COL not in df.columns:
    raise KeyError(f"Target column '{TARGET_COL}' not found in CSV")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].astype(float).values

# ----------------------------------------------------------------------
# 4️⃣  Train / validation / test split (70 % / 20 % / 10 %)
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
# 5️⃣  Build preprocessing pipeline (fit on training data only)
# ----------------------------------------------------------------------
numeric_features = X_train.columns.tolist()

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

preprocess.fit(X_train)  # IMPORTANT: fit on TRAIN ONLY

# ----------------------------------------------------------------------
# 6️⃣  Optional log‑transform of the target
# ----------------------------------------------------------------------
skew = pd.Series(y_train).skew()
log_target = abs(skew) > LOG_TARGET_THRESHOLD

if log_target:
    y_train = np.log1p(y_train)
    y_val   = np.log1p(y_val)
    y_test  = np.log1p(y_test)

# ----------------------------------------------------------------------
# 7️⃣  Persist preprocessing artefacts (pipeline + meta)
# ----------------------------------------------------------------------
meta = {
    "target_column": TARGET_COL,
    "log_target": bool(log_target),
    "feature_names": numeric_features,
}
meta_path = MODEL_DIR / "preprocess_meta.json"
meta_path.write_text(json.dumps(meta, indent=2))

pipeline_path = MODEL_DIR / "preprocess_pipeline.joblib"
dump(preprocess, pipeline_path)

# ----------------------------------------------------------------------
# 8️⃣  Transform splits and store as .npy (convenient for later scripts)
# ----------------------------------------------------------------------
X_train_proc = preprocess.transform(X_train).astype(np.float32)
X_val_proc   = preprocess.transform(X_val).astype(np.float32)
X_test_proc  = preprocess.transform(X_test).astype(np.float32)

np.save(PROCESSED_DIR / "X_train.npy", X_train_proc)
np.save(PROCESSED_DIR / "y_train.npy", y_train.astype(np.float32))
np.save(PROCESSED_DIR / "X_val.npy",   X_val_proc)
np.save(PROCESSED_DIR / "y_val.npy",   y_val.astype(np.float32))
np.save(PROCESSED_DIR / "X_test.npy",  X_test_proc)
np.save(PROCESSED_DIR / "y_test.npy",  y_test.astype(np.float32))

# ----------------------------------------------------------------------
# 9️⃣  TabNet hyper‑parameters (could be moved to a YAML file)
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
# 1️⃣0️⃣  Instantiate and train TabNetRegressor
# ----------------------------------------------------------------------
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

start = time.time()
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
train_time_sec = time.time() - start

# ----------------------------------------------------------------------
# 1️⃣1️⃣  Evaluate on the test set
# ----------------------------------------------------------------------
y_pred = model.predict(X_test_proc)

if log_target:
    # Inverse the earlier log1p transformation
    y_pred = np.expm1(y_pred)
    y_test_original = np.expm1(y_test)
else:
    y_test_original = y_test

mae = mean_absolute_error(y_test_original, y_pred)
rmse = mean_squared_error(y_test_original, y_pred, squared=False)
r2 = r2_score(y_test_original, y_pred)

metrics = {
    "MAE": mae,
    "RMSE": rmse,
    "R2": r2,
    "train_time_sec": train_time_sec,
    "log_target": bool(log_target),
}

print("\n=== Test set metrics ===")
for k, v in metrics.items():
    print(f"{k}: {v:.5f}")

# ----------------------------------------------------------------------
# 1️⃣2️⃣  Persist the trained model (state_dict) and a tiny inference helper
# ----------------------------------------------------------------------
model_path = MODEL_DIR / "tabnet_regressor.pt"
torch.save(model.state_dict(), model_path)

# ---- Inference helper (single function `predict(df)`) -----------------
inference_helper_path = MODEL_DIR / "inference_helper.py"
inference_code = f'''"""
Tiny inference helper for the trained TabNet regression model.

Usage:
    from inference_helper import predict
    preds = predict(pd.DataFrame(...))
"""

import json
import pathlib
import numpy as np
import pandas as pd
import torch
from joblib import load
from pytorch_tabnet.tab_model import TabNetRegressor

BASE = pathlib.Path(__file__).resolve().parent

# Load artefacts once at import time
PIPELINE = load(BASE / "preprocess_pipeline.joblib")
META = json.loads((BASE / "preprocess_meta.json").read_text())
HYPER = {json.dumps(hyper, indent=4)}

STATE = torch.load(BASE / "tabnet_regressor.pt", map_location="cpu")

# Re‑create TabNet model with the same hyper‑parameters
model = TabNetRegressor(
    n_d=HYPER["n_d"],
    n_a=HYPER["n_a"],
    n_steps=HYPER["n_steps"],
    gamma=HYPER["gamma"],
    lambda_sparse=HYPER["lambda_sparse"],
    optimizer_lr=HYPER["optimizer_lr"],
    optimizer_weight_decay=HYPER["optimizer_weight_decay"],
    device_name="cpu",
    seed=HYPER["seed"],
)
model.load_state_dict(STATE)
model.eval()


def predict(df: pd.DataFrame) -> pd.Series:
    """
    Return predictions for a DataFrame that contains the original feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain **all** columns listed in ``META["feature_names"]``.
        Extra columns are ignored.

    Returns
    -------
    pd.Series
        Regression predictions named ``"prediction"``.
    """
    missing = set(META["feature_names"]) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {{sorted(missing)}}")

    # Preserve column order expected by the pipeline
    X = df[META["feature_names"]]
    X_proc = PIPELINE.transform(X).astype(np.float32)

    with torch.no_grad():
        preds = model.predict(X_proc)

    if META["log_target"]:
        preds = np.expm1(preds)

    return pd.Series(preds, name="prediction")
'''
inference_helper_path.write_text(inference_code)

# ----------------------------------------------------------------------
# 1️⃣3️⃣  Save a tiny JSON report for downstream consumption
# ----------------------------------------------------------------------
report_path = MODEL_DIR / "test_metrics.json"
report_path.write_text(json.dumps(metrics, indent=2))

print("\n=== Artefacts saved under:", MODEL_DIR)
print(f" • Preprocess pipeline : {pipeline_path.name}")
print(f" • Preprocess meta    : {meta_path.name}")
print(f" • Model state‑dict   : {model_path.name}")
print(f" • Inference helper  : {inference_helper_path.name}")
print(f" • Test metrics JSON : {report_path.name}")
