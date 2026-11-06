
"""
California Housing – End‑to‑End Tabular Regression Pipeline
============================================================

This script implements the complete workflow described in the specification:

1️⃣  Data ingestion (copy, SHA‑256 checksum, lightweight validation)
2️⃣  Train / validation / test split
3️⃣  Sklearn preprocessing pipeline
    – Winsorising (1st / 99th percentile)
    – Log‑skew handling (log1p for |skew| > 1)
    – Engineered ratios
    – StandardScaler
4️⃣  LightGBM training with the fixed hyper‑parameters
5️⃣  Persistence of the fitted pipeline and the LightGBM model
6️⃣  (Optional) ONNX export – performed only if the `onnx`/`skl2onnx`
    packages are available.

All artefacts are stored under ``./agent_workspace/trained_models``.
Run the script directly:

    python run_pipeline.py
"""

import hashlib
import json
import logging
import os
import shutil
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import pandera as pa
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------- #
# Configuration & Paths
# --------------------------------------------------------------------------- #

# Adjust this source path if the CSV lives elsewhere.
SOURCE_CSV = Path(
    r"C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\california.csv"
)

PROJECT_ROOT = Path("./agent_workspace")
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "trained_models"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure directory hierarchy exists
for d in [RAW_DIR, PROCESSED_DIR, MODEL_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(PROJECT_ROOT / "pipeline.log"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# 1️⃣ Data ingestion & checksum
# --------------------------------------------------------------------------- #
def copy_and_checksum(src: Path, dst: Path) -> str:
    """Copy src → dst and return SHA‑256 checksum of the copied file."""
    shutil.copy(src, dst)
    sha256 = hashlib.sha256()
    with open(dst, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    checksum = sha256.hexdigest()
    log.info(f"Copied CSV to {dst} – SHA256: {checksum}")
    return checksum


def write_metadata(metadata_path: Path, **kwargs):
    """Write a JSON file with arbitrary keyword arguments."""
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(kwargs, f, indent=2)
    log.info(f"Wrote metadata to {metadata_path}")

# --------------------------------------------------------------------------- #
# 2️⃣ Pandera schema validation
# --------------------------------------------------------------------------- #
class HousingSchema(pa.SchemaModel):
    MedInc = pa.Field(nullable=False, dtype=pa.Float64)
    HouseAge = pa.Field(nullable=False, dtype=pa.Float64)
    AveRooms = pa.Field(nullable=False, dtype=pa.Float64)
    AveBedrms = pa.Field(nullable=False, dtype=pa.Float64)
    Population = pa.Field(nullable=False, dtype=pa.Float64)
    AveOccup = pa.Field(nullable=False, dtype=pa.Float64)
    Latitude = pa.Field(nullable=False, dtype=pa.Float64)
    Longitude = pa.Field(nullable=False, dtype=pa.Float64)
    target = pa.Field(nullable=False, dtype=pa.Float64, checks=pa.Check.gt(0))

    class Config:
        strict = True
        coerce = True


def validate_with_pandera(df: pd.DataFrame) -> pd.DataFrame:
    """Validate dataframe against the HousingSchema; raise on failure."""
    schema = HousingSchema.to_schema()
    validated = schema.validate(df, lazy=True)
    log.info("Pandera validation passed.")
    return validated


# --------------------------------------------------------------------------- #
# 3️⃣ Custom transformers for the preprocessing pipeline
# --------------------------------------------------------------------------- #
class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip each column to its 1st and 99th percentile (computed on fit)."""

    def __init__(self, lower_quantile=0.01, upper_quantile=0.99):
        self.lower_q = lower_quantile
        self.upper_q = upper_quantile

    def fit(self, X, y=None):
        # X is a NumPy array
        self.lower_ = np.percentile(X, self.lower_q * 100, axis=0)
        self.upper_ = np.percentile(X, self.upper_q * 100, axis=0)
        return self

    def transform(self, X):
        return np.clip(X, self.lower_, self.upper_)


class LogSkewTransformer(BaseEstimator, TransformerMixin):
    """
    For columns whose absolute skewness > threshold, add a new column
    ``log_<col> = log1p(col)``. Original column is retained.
    """

    def __init__(self, skew_threshold=1.0):
        self.threshold = skew_threshold

    def fit(self, X, y=None):
        # X is a pandas DataFrame
        self.cols_to_log_ = [
            col for col in X.columns if abs(X[col].skew()) > self.threshold
        ]
        log.info(f"Columns selected for log‑skew handling: {self.cols_to_log_}")
        return self

    def transform(self, X):
        X_out = X.copy()
        for col in self.cols_to_log_:
            X_out[f"log_{col}"] = np.log1p(X_out[col])
        return X_out


class RatioEngineer(BaseEstimator, TransformerMixin):
    """Add three engineered ratio features."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_out = X.copy()
        eps = 1e-9
        X_out["rooms_per_household"] = X_out["AveRooms"] / (X_out["AveOccup"] + eps)
        X_out["bedrooms_per_room"] = X_out["AveBedrms"] / (X_out["AveRooms"] + eps)
        X_out["population_per_household"] = X_out["Population"] / (X_out["AveOccup"] + eps)
        return X_out


# --------------------------------------------------------------------------- #
# 4️⃣ Build the full sklearn pipeline
# --------------------------------------------------------------------------- #
def build_preprocessing_pipeline():
    """
    Returns a sklearn Pipeline that works on a *pandas DataFrame* and finally
    outputs a NumPy array ready for LightGBM.
    """
    pipeline = Pipeline(
        steps=[
            ("log_skew", LogSkewTransformer()),
            ("ratio", RatioEngineer()),
            # Convert DataFrame → NumPy array (required for Winsorizer & Scaler)
            ("to_numpy", FunctionTransformer(lambda df: df.values, validate=False)),
            ("winsor", Winsorizer()),
            ("scaler", StandardScaler()),
        ]
    )
    return pipeline


# --------------------------------------------------------------------------- #
# 5️⃣ LightGBM training utilities
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


def train_lightgbm(X_train, y_train, X_val, y_val):
    model = lgb.LGBMRegressor(**LGBM_PARAMS)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        early_stopping_rounds=50,
        verbose=False,
    )
    return model


def evaluate_model(model, X, y, split_name="test"):
    preds = model.predict(X)
    rmse = mean_squared_error(y, preds, squared=False)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    log.info(
        f"{split_name.upper()} – RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}"
    )
    return {"rmse": rmse, "mae": mae, "r2": r2}


# --------------------------------------------------------------------------- #
# 6️⃣ (Optional) ONNX export – performed only if `onnx`/`skl2onnx` are present
# --------------------------------------------------------------------------- #
def export_to_onnx(preprocess_pipe, model, onnx_path, feature_columns):
    """
    Build a combined sklearn Pipeline (preprocess + LightGBM) and convert it
    to ONNX. If the required packages are missing, the function logs a warning
    and returns without error.
    """
    try:
        import onnx
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
    except Exception as e:
        log.warning(
            "ONNX export skipped – required packages not installed: %s", e
        )
        return

    combined = Pipeline(
        steps=[
            ("preprocess", preprocess_pipe),
            ("lgbm", model),
        ]
    )
    # Dummy DataFrame to infer input shape
    dummy_df = pd.DataFrame(
        np.zeros((1, len(feature_columns))), columns=feature_columns
    )
    dummy_input = preprocess_pipe.transform(dummy_df)
    n_features = dummy_input.shape[1]

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
# Main execution
# --------------------------------------------------------------------------- #
def main():
    # ------------------------------------------------------------------- #
    # 1️⃣ Copy CSV & compute checksum
    # ------------------------------------------------------------------- #
    raw_csv_path = RAW_DIR / "california_housing_subsample.csv"
    checksum = copy_and_checksum(SOURCE_CSV, raw_csv_path)

    metadata_path = PROCESSED_DIR / "metadata.json"
    write_metadata(
        metadata_path,
        source_path=str(SOURCE_CSV),
        dest_path=str(raw_csv_path),
        sha256=checksum,
        timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        rows_expected=1500,
        columns_expected=9,
    )

    # ------------------------------------------------------------------- #
    # 2️⃣ Load & validate
    # ------------------------------------------------------------------- #
    df = pd.read_csv(raw_csv_path, dtype="float64")
    df = validate_with_pandera(df)

    # Persist validated data – try Parquet first, fall back to CSV if engine missing
    processed_path = PROCESSED_DIR / "california_housing.parquet"
    try:
        df.to_parquet(processed_path, index=False)
        log.info(f"Saved validated data to Parquet at {processed_path}")
    except Exception as e:
        log.warning(
            f"Parquet write failed ({e}); falling back to CSV storage."
        )
        processed_path = PROCESSED_DIR / "california_housing.csv"
        df.to_csv(processed_path, index=False)
        log.info(f"Saved validated data to CSV at {processed_path}")

    # ------------------------------------------------------------------- #
    # 3️⃣ Train / val / test split
    # ------------------------------------------------------------------- #
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=42, shuffle=True
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, shuffle=True
    )
    log.info(
        f"Split sizes – train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}"
    )

    # ------------------------------------------------------------------- #
    # 4️⃣ Build & fit preprocessing pipeline
    # ------------------------------------------------------------------- #
    feature_columns = [c for c in df.columns if c != "target"]
    preprocess_pipe = build_preprocessing_pipeline()
    preprocess_pipe.fit(train_df[feature_columns])

    pipeline_path = MODEL_DIR / "pipeline.pkl"
    joblib.dump(preprocess_pipe, pipeline_path)
    log.info(f"Saved preprocessing pipeline to {pipeline_path}")

    # ------------------------------------------------------------------- #
    # 5️⃣ Transform data for LightGBM
    # ------------------------------------------------------------------- #
    X_train = preprocess_pipe.transform(train_df[feature_columns])
    X_val = preprocess_pipe.transform(val_df[feature_columns])
    X_test = preprocess_pipe.transform(test_df[feature_columns])

    y_train = train_df["target"].values
    y_val = val_df["target"].values
    y_test = test_df["target"].values

    # ------------------------------------------------------------------- #
    # 6️⃣ Train LightGBM
    # ------------------------------------------------------------------- #
    start = time.time()
    lgbm_model = train_lightgbm(X_train, y_train, X_val, y_val)
    training_time = time.time() - start
    log.info(f"LightGBM training completed in {training_time:.2f} seconds")

    # ------------------------------------------------------------------- #
    # 7️⃣ Evaluate on test set
    # ------------------------------------------------------------------- #
    test_metrics = evaluate_model(lgbm_model, X_test, y_test, split_name="test")

    # ------------------------------------------------------------------- #
    # 8️⃣ Persist LightGBM model
    # ------------------------------------------------------------------- #
    lgbm_path = MODEL_DIR / "lightgbm_model.pkl"
    joblib.dump(lgbm_model, lgbm_path)
    log.info(f"Saved LightGBM model to {lgbm_path}")

    # ------------------------------------------------------------------- #
    # 9️⃣ Save a JSON report with metrics & hyper‑parameters
    # ------------------------------------------------------------------- #
    report = {
        "hyperparameters": LGBM_PARAMS,
        "training_time_seconds": training_time,
        "test_metrics": test_metrics,
        "model_file_size_bytes": lgbm_path.stat().st_size,
    }
    report_path = REPORTS_DIR / "model_profile_lightgbm.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log.info(f"Wrote model report to {report_path}")

    # ------------------------------------------------------------------- #
    # 10️⃣ (Optional) Export to ONNX
    # ------------------------------------------------------------------- #
    onnx_path = MODEL_DIR / "lightgbm_model.onnx"
    export_to_onnx(preprocess_pipe, lgbm_model, onnx_path, feature_columns)

    log.info("Pipeline execution finished successfully.")


if __name__ == "__main__":
    main()
