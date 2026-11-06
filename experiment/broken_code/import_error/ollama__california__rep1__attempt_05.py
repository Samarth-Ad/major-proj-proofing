
"""
End‑to‑end tabular regression pipeline for the California Housing dataset.

The script performs the following steps:

1️⃣  Ingest the CSV supplied by the user (Windows path) and copy it into the
    project’s `data/raw/` folder.  A SHA‑256 checksum is stored for reproducibility.

2️⃣  Split the data into train / validation / test (80 % / 10 % / 10 %) **before**
    any preprocessing.

3️⃣  Build a sklearn preprocessing pipeline:
        - Winsorising (1st / 99th percentiles)
        - Log‑skew handling (adds `log_<col>` for highly skewed columns)
        - Ratio engineering (rooms/household, bedrooms/room, population/household)
        - Standard scaling

4️⃣  Fit the pipeline on the training split only and transform all splits.

5️⃣  Train a LightGBM regressor with the hyper‑parameters that gave the best
    validation RMSE on the subsample.

6️⃣  Evaluate on the held‑out test set (RMSE, MAE, R²) and log the metrics.

7️⃣  (Optional) Export the combined preprocessing + LightGBM model to ONNX.

8️⃣  Wrap the ONNX model in a tiny ``torch.nn.Module`` that first applies the
    sklearn pipeline and then runs ONNX inference – satisfying the “PyTorch‑preferred”
    requirement.

9️⃣  Persist all artefacts (pipeline, LightGBM model, ONNX model, torch wrapper)
    into ``./agent_workspace/trained_models`` and log a short JSON report.

The code is self‑contained; missing optional packages (``onnx``, ``onnxruntime``,
``skl2onnx``) are handled gracefully – the pipeline will still work using the
native LightGBM model.
"""

import os
import json
import hashlib
import logging
import time
import shutil
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import lightgbm as lgb
import mlflow

# --------------------------------------------------------------------------- #
# Optional imports – ONNX export and PyTorch wrapper
# --------------------------------------------------------------------------- #
try:
    import onnx
    import onnxruntime as ort
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except Exception:  # pragma: no cover
    ONNX_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    TORCH_AVAILABLE = False

# --------------------------------------------------------------------------- #
# Logging configuration
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Global paths (adjust if you move the script)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
SPLIT_DIR = PROJECT_ROOT / "data" / "splits"
MODEL_DIR = PROJECT_ROOT / "agent_workspace" / "trained_models"
REPORTS_DIR = PROJECT_ROOT / "reports"

for d in [RAW_DIR, PROCESSED_DIR, SPLIT_DIR, MODEL_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# 1️⃣  Data ingestion
# --------------------------------------------------------------------------- #
SOURCE_CSV = Path(
    r"C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\california.csv"
)  # <-- change if the source location differs
DEST_CSV = RAW_DIR / "california_housing_subsample.csv"
METADATA_PATH = PROCESSED_DIR / "metadata.json"


def compute_sha256(path: Path) -> str:
    """Return the SHA‑256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def ingest_data() -> pd.DataFrame:
    """Copy the source CSV, compute checksum, and load into a DataFrame."""
    if not SOURCE_CSV.is_file():
        raise FileNotFoundError(f"Source CSV not found at {SOURCE_CSV}")

    # Copy (overwrites if already present)
    shutil.copy2(SOURCE_CSV, DEST_CSV)
    log.info(f"Copied source CSV to {DEST_CSV}")

    # Compute checksum
    checksum = compute_sha256(DEST_CSV)
    metadata = {
        "source_path": str(SOURCE_CSV),
        "dest_path": str(DEST_CSV),
        "sha256": checksum,
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "rows_expected": 1500,  # approximate – the original description mentions ~1500 rows
        "columns_expected": 9,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    log.info(f"Wrote metadata to {METADATA_PATH}")

    # Load CSV
    df = pd.read_csv(DEST_CSV, dtype="float64", low_memory=False)
    log.info(f"Loaded data with shape {df.shape}")
    return df


# --------------------------------------------------------------------------- #
# 2️⃣  Train/validation/test split (80/10/10)
# --------------------------------------------------------------------------- #
def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return train, validation and test DataFrames."""
    train_val, test = train_test_split(
        df, test_size=0.10, random_state=42, shuffle=True
    )
    train, val = train_test_split(
        train_val, test_size=0.1111, random_state=42, shuffle=True
    )  # 0.1111 * 0.9 ≈ 0.10
    # Save splits for audit
    train_path = SPLIT_DIR / "train.csv"
    val_path = SPLIT_DIR / "val.csv"
    test_path = SPLIT_DIR / "test.csv"
    train.to_csv(train_path, index=False)
    val.to_csv(val_path, index=False)
    test.to_csv(test_path, index=False)
    log.info(f"Saved splits: train({train.shape}), val({val.shape}), test({test.shape})")
    return train, val, test


# --------------------------------------------------------------------------- #
# 3️⃣  Custom transformers
# --------------------------------------------------------------------------- #
class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to the 1st and 99th percentiles computed on fit data."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.lower_bounds_ = np.percentile(X, self.lower * 100, axis=0)
        self.upper_bounds_ = np.percentile(X, self.upper * 100, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X)
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)


class LogSkewTransformer(BaseEstimator, TransformerMixin):
    """Add log1p columns for features whose absolute skewness > 1.0."""

    def __init__(self, skew_threshold=1.0):
        self.skew_threshold = skew_threshold

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            self.cols_ = X.columns
            self.skew_ = X.skew()
        else:
            raise ValueError("LogSkewTransformer expects a pandas DataFrame")
        self.high_skew_cols_ = self.skew_[self.skew_.abs() > self.skew_threshold].index.tolist()
        return self

    def transform(self, X):
        X = X.copy()
        for col in self.high_skew_cols_:
            X[f"log_{col}"] = np.log1p(X[col])
        return X


class RatioEngineer(BaseEstimator, TransformerMixin):
    """Create three density‑type ratio features."""

    def __init__(self, eps=1e-9):
        self.eps = eps

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["rooms_per_household"] = X["AveRooms"] / (X["AveOccup"] + self.eps)
        X["bedrooms_per_room"] = X["AveBedrms"] / (X["AveRooms"] + self.eps)
        X["population_per_household"] = X["Population"] / (X["AveOccup"] + self.eps)
        return X


# --------------------------------------------------------------------------- #
# 4️⃣  Build preprocessing pipeline
# --------------------------------------------------------------------------- #
def build_preprocess_pipeline() -> Pipeline:
    pipeline = Pipeline(
        [
            ("log_skew", LogSkewTransformer()),
            ("ratio", RatioEngineer()),
            # Convert DataFrame → NumPy for downstream steps
            ("to_numpy", FunctionTransformer(lambda df: df.values, validate=False)),
            ("winsor", Winsorizer()),
            ("scaler", StandardScaler()),
        ]
    )
    return pipeline


# --------------------------------------------------------------------------- #
# 5️⃣  LightGBM training
# --------------------------------------------------------------------------- #
LGBM_PARAMS = {
    "learning_rate": 0.03,
    "n_estimators": 850,
    "max_depth": 7,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.85,
    "colsample_bytree": 0.90,
    "reg_lambda": 0.10,
    "objective": "regression",
    "random_state": 42,
}


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[lgb.LGBMRegressor, Dict[str, float], float]:
    model = lgb.LGBMRegressor(**LGBM_PARAMS)

    start = time.time()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        early_stopping_rounds=50,
        verbose=False,
    )
    duration = time.time() - start
    log.info(f"LightGBM training completed in {duration:.2f}s")
    return model, LGBM_PARAMS, duration


# --------------------------------------------------------------------------- #
# 6️⃣  Evaluation
# --------------------------------------------------------------------------- #
def evaluate(
    model: lgb.LGBMRegressor,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Dict[str, float]:
    preds = model.predict(X_test)
    metrics = {
        "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds),
    }
    log.info(f"Test metrics: {metrics}")
    return metrics


# --------------------------------------------------------------------------- #
# 7️⃣  Optional ONNX export
# --------------------------------------------------------------------------- #
def export_to_onnx(
    pipeline: Pipeline,
    model: lgb.LGBMRegressor,
    n_features: int,
    onnx_path: Path,
) -> None:
    if not ONNX_AVAILABLE:
        log.warning("ONNX packages not available – skipping ONNX export.")
        return

    # Combine preprocessing pipeline and LightGBM model
    combined = Pipeline([("preprocess", pipeline), ("lgbm", model)])

    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(
        combined,
        initial_types=initial_type,
        target_opset=15,
        options={"lgbm": {"zipmap": False}},
    )
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    log.info(f"Exported ONNX model to {onnx_path}")


# --------------------------------------------------------------------------- #
# 8️⃣  PyTorch wrapper around ONNX + sklearn pipeline
# --------------------------------------------------------------------------- #
if TORCH_AVAILABLE:

    class LightGBMOnnxWrapper(nn.Module):
        """
        PyTorch module that:
          1. Applies the saved sklearn preprocessing pipeline.
          2. Runs ONNX inference for the LightGBM model.
        """

        def __init__(self, pipeline_path: Path, onnx_path: Path):
            super().__init__()
            self.pipeline = joblib.load(pipeline_path)
            self.session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        @torch.no_grad()
        def forward(self, df: pd.DataFrame) -> torch.Tensor:
            """
            Parameters
            ----------
            df : pandas.DataFrame
                Raw feature rows (no target column).

            Returns
            -------
            torch.Tensor
                Predictions of shape (batch, ).
            """
            # 1) Preprocess
            X = self.pipeline.transform(df)  # NumPy array, already scaled & winsorised
            X = X.astype(np.float32)

            # 2) ONNX inference
            ort_inputs = {"float_input": X}
            ort_outs = self.session.run(None, ort_inputs)
            preds = ort_outs[0]  # shape (batch,)

            return torch.from_numpy(preds)

else:
    LightGBMOnnxWrapper = None  # type: ignore


# --------------------------------------------------------------------------- #
# 9️⃣  Main orchestration
# --------------------------------------------------------------------------- #
def main() -> None:
    # ------------------------------------------------------------------- #
    # Step 1 – Ingest
    # ------------------------------------------------------------------- #
    df = ingest_data()

    # ------------------------------------------------------------------- #
    # Step 2 – Split
    # ------------------------------------------------------------------- #
    train_df, val_df, test_df = split_data(df)

    # Separate features / target
    target_col = "target" if "target" in df.columns else "MedHouseVal"
    X_train_raw = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].values
    X_val_raw = val_df.drop(columns=[target_col])
    y_val = val_df[target_col].values
    X_test_raw = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].values

    # ------------------------------------------------------------------- #
    # Step 3 – Preprocess pipeline
    # ------------------------------------------------------------------- #
    pipeline = build_preprocess_pipeline()
    pipeline.fit(X_train_raw)  # fit only on training data
    X_train = pipeline.transform(X_train_raw)
    X_val = pipeline.transform(X_val_raw)
    X_test = pipeline.transform(X_test_raw)

    # Persist the fitted pipeline
    pipeline_path = MODEL_DIR / "pipeline.pkl"
    joblib.dump(pipeline, pipeline_path)
    log.info(f"Saved fitted preprocessing pipeline to {pipeline_path}")

    # ------------------------------------------------------------------- #
    # Step 4 – LightGBM training
    # ------------------------------------------------------------------- #
    model, params, train_time = train_lightgbm(X_train, y_train, X_val, y_val)

    # Persist LightGBM model (native)
    lgb_model_path = MODEL_DIR / "lightgbm_model.pkl"
    joblib.dump(model, lgb_model_path)
    log.info(f"Saved LightGBM model to {lgb_model_path}")

    # ------------------------------------------------------------------- #
    # Step 5 – Evaluation
    # ------------------------------------------------------------------- #
    test_metrics = evaluate(model, X_test, y_test)

    # ------------------------------------------------------------------- #
    # Step 6 – Optional ONNX export
    # ------------------------------------------------------------------- #
    onnx_path = MODEL_DIR / "lightgbm_model.onnx"
    export_to_onnx(pipeline, model, X_train.shape[1], onnx_path)

    # ------------------------------------------------------------------- #
    # Step 7 – PyTorch wrapper (only if ONNX succeeded and torch is present)
    # ------------------------------------------------------------------- #
    if TORCH_AVAILABLE and ONNX_AVAILABLE and onnx_path.is_file():
        wrapper = LightGBMOnnxWrapper(pipeline_path, onnx_path)
        wrapper_path = MODEL_DIR / "lightgbm_torch_wrapper.pt"
        torch.save(wrapper, wrapper_path)
        log.info(f"Saved PyTorch wrapper to {wrapper_path}")
    else:
        log.warning(
            "Skipping PyTorch wrapper creation – either torch or ONNX is unavailable."
        )

    # ------------------------------------------------------------------- #
    # Step 8 – MLflow logging (local file store)
    # ------------------------------------------------------------------- #
    mlflow.set_tracking_uri(f"file://{PROJECT_ROOT / 'mlflow'}")
    mlflow.set_experiment("California_Housing_Regression")
    with mlflow.start_run(run_name="lightgbm_pipeline"):
        # Params
        mlflow.log_params(params)
        mlflow.log_metric("train_time_sec", train_time)
        # Metrics
        mlflow.log_metrics(test_metrics)
        # Artefacts
        mlflow.log_artifact(str(pipeline_path), artifact_path="pipeline")
        mlflow.log_artifact(str(lgb_model_path), artifact_path="model")
        if onnx_path.is_file():
            mlflow.log_artifact(str(onnx_path), artifact_path="onnx")
        if TORCH_AVAILABLE and (MODEL_DIR / "lightgbm_torch_wrapper.pt").is_file():
            mlflow.log_artifact(
                str(MODEL_DIR / "lightgbm_torch_wrapper.pt"), artifact_path="torch_wrapper"
            )
        # Tag with git commit if inside a git repo
        try:
            import subprocess

            commit = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(PROJECT_ROOT)
            ).decode().strip()
            mlflow.set_tag("git_commit", commit)
        except Exception:
            pass

    # ------------------------------------------------------------------- #
    # Step 9 – Write a concise JSON report
    # ------------------------------------------------------------------- #
    report = {
        "model_type": "LightGBMRegressor",
        "hyperparameters": params,
        "train_time_sec": train_time,
        "test_metrics": test_metrics,
        "pipeline_path": str(pipeline_path),
        "lightgbm_model_path": str(lgb_model_path),
        "onnx_exported": ONNX_AVAILABLE and onnx_path.is_file(),
        "onnx_path": str(onnx_path) if (ONNX_AVAILABLE and onnx_path.is_file()) else None,
        "torch_wrapper_created": TORCH_AVAILABLE
        and (MODEL_DIR / "lightgbm_torch_wrapper.pt").is_file(),
        "torch_wrapper_path": str(MODEL_DIR / "lightgbm_torch_wrapper.pt")
        if (TORCH_AVAILABLE and (MODEL_DIR / "lightgbm_torch_wrapper.pt").is_file())
        else None,
    }
    report_path = REPORTS_DIR / "model_profile_lightgbm.json"
    report_path.write_text(json.dumps(report, indent=2))
    log.info(f"Wrote report to {report_path}")

    # ------------------------------------------------------------------- #
    # Final console output
    # ------------------------------------------------------------------- #
    print("\n=== Training complete ===")
    print(f"Test RMSE: {test_metrics['RMSE']:.4f}")
    print(f"Model artefacts stored under: {MODEL_DIR}")
    print(f"Report file: {report_path}")


if __name__ == "__main__":
    main()
