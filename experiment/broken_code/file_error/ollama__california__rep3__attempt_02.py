
"""
end_to_end_pipeline.py
======================

A self‑contained, production‑ready script that implements the full
California‑Housing regression workflow described in the specification:

1. Load the immutable raw CSV and compute a SHA‑256 checksum.
2. Validate the schema and generate a data profile.
3. Split the data (70 % train / 20 % validation / 10 % test).
4. Build a versioned Scikit‑Learn preprocessing pipeline:
   - Median imputation
   - Isolation‑Forest outlier clipping (1 % contamination)
   - Robust scaling
   - Interaction‑only polynomial features (degree 2) for the three most
     correlated features w.r.t. the target
   - Optional log‑transform of the target (|skew| > 0.75)
5. Persist the fitted pipeline and the transformed splits as Parquet files.
6. Train a CatBoostRegressor with frozen hyper‑parameters, export both the
   native ``.cbm`` model and an ONNX version.
7. Train a small feed‑forward PyTorch MLP on the same pre‑processed data,
   export it as a TorchScript module.
8. Log artefacts and metrics to MLflow **if the library is available**.
9. Copy the final artefacts to ``./agent_workspace/trained_models`` so that
   the evaluation harness can find them.

The script can be executed directly:

    python end_to_end_pipeline.py
"""

# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
import json
import hashlib
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import joblib

# CatBoost & PyTorch
from catboost import CatBoostRegressor
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
# Optional MLflow import (graceful fallback if not installed)
# --------------------------------------------------------------------------- #
try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except Exception:  # pragma: no cover
    _MLFLOW_AVAILABLE = False
    mlflow = None

# --------------------------------------------------------------------------- #
# Global settings & absolute paths (as required by the spec)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path("C:/mlops/cali_housing_project").resolve()

# ── Data & metadata ──────────────────────────────────────────────────────── #
RAW_CSV = PROJECT_ROOT / "data" / "raw" / "california_housing_subsample.csv"
CHECKSUM_FILE = PROJECT_ROOT / "metadata" / "checksum.txt"
SCHEMA_FILE = PROJECT_ROOT / "metadata" / "schema.json"
OUTLIER_REPORT = PROJECT_ROOT / "metadata" / "outlier_report.json"
TARGET_TRANSFORM = PROJECT_ROOT / "metadata" / "target_transform.json"
DATA_PROFILE = PROJECT_ROOT / "metadata" / "data_profile.json"

# ── Processed splits ─────────────────────────────────────────────────────── #
TRAIN_PARQUET = PROJECT_ROOT / "data" / "processed" / "train.parquet"
VAL_PARQUET   = PROJECT_ROOT / "data" / "processed" / "val.parquet"
TEST_PARQUET  = PROJECT_ROOT / "data" / "processed" / "test.parquet"

# ── Model artefacts ─────────────────────────────────────────────────────── #
PIPELINE_PATH = PROJECT_ROOT / "models" / "preprocess_pipeline.pkl"
CATBOOST_MODEL_CBM  = PROJECT_ROOT / "models" / "catboost_model.cbm"
CATBOOST_MODEL_ONNX = PROJECT_ROOT / "models" / "catboost_model.onnx"
MLP_BEST_PT   = PROJECT_ROOT / "models" / "mlp_best.pt"
MLP_MODEL_PT   = PROJECT_ROOT / "models" / "mlp_model.pt"

# ── Destination for the evaluation harness ─────────────────────────────── #
AGENT_MODEL_DIR = Path("./agent_workspace/trained_models").resolve()
AGENT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def compute_checksum(path: Path) -> str:
    """Return SHA‑256 checksum of a file."""
    sha256 = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()


def save_json(obj, path: Path):
    """Write ``obj`` as pretty‑printed JSON, creating parent dirs if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def log_mlflow(run_name: str, params: dict, metrics: dict, artifacts: list):
    """Log a run to MLflow only when the library is available."""
    if not _MLFLOW_AVAILABLE:
        return
    mlflow.set_experiment("CaliforniaHousing_ModelTraining")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        for a in artifacts:
            mlflow.log_artifact(str(a))


# --------------------------------------------------------------------------- #
# Custom transformers
# --------------------------------------------------------------------------- #
class OutlierClipper(BaseEstimator, TransformerMixin):
    """
    Fit an IsolationForest only to count outliers (required by the spec)
    and then clip each feature to its 1st / 99th percentile.
    """

    def __init__(self, contamination: float = 0.01, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X, y=None):
        X = np.asarray(X)
        iso = IsolationForest(
            contamination=self.contamination, random_state=self.random_state
        )
        iso.fit(X)
        self.n_outliers_ = int((iso.predict(X) == -1).sum())
        self.lower_ = np.percentile(X, 1, axis=0)
        self.upper_ = np.percentile(X, 99, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X)
        return np.clip(X, self.lower_, self.upper_)


class InteractionTransformer(BaseEstimator, TransformerMixin):
    """
    Generate interaction‑only polynomial features (degree=2) for the *n_top*
    features most correlated with the target.
    """

    def __init__(self, n_top: int = 3):
        self.n_top = n_top

    def fit(self, X, y):
        df = pd.DataFrame(X)
        # Correlation of each feature with the target
        corr = df.corrwith(pd.Series(y))
        self.top_features_ = (
            corr.abs().sort_values(ascending=False).head(self.n_top).index.tolist()
        )
        # Fit a PolynomialFeatures transformer on the selected columns only
        self.poly_ = PolynomialFeatures(
            degree=2, interaction_only=True, include_bias=False
        )
        self.poly_.fit(df[self.top_features_])
        return self

    def transform(self, X):
        df = pd.DataFrame(X)
        interactions = self.poly_.transform(df[self.top_features_])
        interaction_names = self.poly_.get_feature_names_out(self.top_features_)
        interactions_df = pd.DataFrame(
            interactions, columns=interaction_names, index=df.index
        )
        # Concatenate original columns with the new interaction columns
        result = pd.concat([df, interactions_df], axis=1)
        return result.values  # downstream pipeline expects a NumPy array


# --------------------------------------------------------------------------- #
# Main pipeline
# --------------------------------------------------------------------------- #
def main():
    # ------------------------------------------------------------------- #
    # 1️⃣ Load raw CSV & compute checksum
    # ------------------------------------------------------------------- #
    df_raw = pd.read_csv(RAW_CSV)
    checksum = compute_checksum(RAW_CSV)
    save_json({"sha256": checksum}, CHECKSUM_FILE)

    # ------------------------------------------------------------------- #
    # 2️⃣ Schema validation & data profile
    # ------------------------------------------------------------------- #
    expected_columns = [
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
        "target",
    ]

    # Exact column order & dtype check
    if list(df_raw.columns) != expected_columns:
        raise ValueError(
            f"Column mismatch. Expected {expected_columns}, got {list(df_raw.columns)}"
        )
    if not all(pd.api.types.is_float_dtype(df_raw[col]) for col in expected_columns):
        raise ValueError("All columns must be of dtype float64.")

    # Save schema information
    schema_info = {
        "columns": expected_columns,
        "dtypes": {col: str(df_raw[col].dtype) for col in expected_columns},
    }
    save_json(schema_info, SCHEMA_FILE)

    # Data profile
    profile = {
        "row_count": df_raw.shape[0],
        "column_names": expected_columns,
        "dtypes": {col: str(dt) for col, dt in df_raw.dtypes.items()},
        "missing_counts": df_raw.isna().sum().to_dict(),
    }
    save_json(profile, DATA_PROFILE)

    # ------------------------------------------------------------------- #
    # 3️⃣ Train / validation / test split (70 / 20 / 10)
    # ------------------------------------------------------------------- #
    X = df_raw.drop(columns=["target"])
    y = df_raw["target"]

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, random_state=42
    )
    # From the remaining 30 % we take 1/3 as test → 10 % of the original data
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=1 / 3, random_state=42
    )

    # ------------------------------------------------------------------- #
    # 4️⃣ Build & fit the preprocessing pipeline (train split only)
    # ------------------------------------------------------------------- #
    imputer = SimpleImputer(strategy="median")
    outlier_clipper = OutlierClipper(contamination=0.01, random_state=42)
    scaler = RobustScaler()
    interaction = InteractionTransformer(n_top=3)

    pipeline = Pipeline(
        [
            ("imputer", imputer),
            ("outlier_clipper", outlier_clipper),
            ("scaler", scaler),
            ("interaction", interaction),
        ]
    )
    pipeline.fit(X_train, y_train)

    # Save outlier report
    outlier_report = {"n_clipped_rows": int(outlier_clipper.n_outliers_)}
    save_json(outlier_report, OUTLIER_REPORT)

    # Store interaction feature names in the CatBoost hyper‑params JSON
    hyperparams_path = PROJECT_ROOT / "configs" / "hyperparams_catboost.json"
    if hyperparams_path.is_file():
        with hyperparams_path.open("r") as f:
            hyperparams = json.load(f)
    else:
        hyperparams = {}
    hyperparams["interaction_features"] = interaction.poly_.get_feature_names_out(
        interaction.top_features_
    ).tolist()
    save_json(hyperparams, hyperparams_path)

    # Persist the pipeline
    PIPELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, PIPELINE_PATH)

    # ------------------------------------------------------------------- #
    # 5️⃣ Transform all splits & optional target log‑transform
    # ------------------------------------------------------------------- #
    def apply_pipeline(df):
        return pipeline.transform(df)

    X_train_t = apply_pipeline(X_train)
    X_val_t   = apply_pipeline(X_val)
    X_test_t  = apply_pipeline(X_test)

    # Target log‑transform decision
    target_skew = skew(y_train)
    use_log = abs(target_skew) > 0.75
    if use_log:
        y_train_t = np.log1p(y_train)
        y_val_t   = np.log1p(y_val)
        y_test_t  = np.log1p(y_test)
    else:
        y_train_t = y_train.values
        y_val_t   = y_val.values
        y_test_t  = y_test.values

    save_json({"use_log": use_log}, TARGET_TRANSFORM)

    # Helper to write a split as Parquet (features + target)
    def write_parquet(X_arr, y_arr, path: Path):
        df = pd.DataFrame(X_arr)
        df["target"] = y_arr
        df.to_parquet(path, index=False)

    write_parquet(X_train_t, y_train_t, TRAIN_PARQUET)
    write_parquet(X_val_t,   y_val_t,   VAL_PARQUET)
    write_parquet(X_test_t,  y_test_t,  TEST_PARQUET)

    # ------------------------------------------------------------------- #
    # 6️⃣ Train CatBoostRegressor
    # ------------------------------------------------------------------- #
    catboost_params = {
        "learning_rate": 0.045,
        "depth": 7,
        "l2_leaf_reg": 3.2,
        "bagging_temperature": 0.6,
        "border_count": 128,
        "iterations": 1800,
        "early_stopping_rounds": 50,
        "random_seed": 42,
        "task_type": "CPU",
        "verbose": False,
    }

    # Load processed splits (they already contain the transformed target)
    train_df = pd.read_parquet(TRAIN_PARQUET)
    val_df   = pd.read_parquet(VAL_PARQUET)

    X_train_cb = train_df.drop(columns=["target"]).values
    y_train_cb = train_df["target"].values
    X_val_cb   = val_df.drop(columns=["target"]).values
    y_val_cb   = val_df["target"].values

    cat_model = CatBoostRegressor(**catboost_params)
    cat_model.fit(
        X_train_cb,
        y_train_cb,
        eval_set=(X_val_cb, y_val_cb),
        use_best_model=True,
    )

    # Save native and ONNX versions
    cat_model.save_model(str(CATBOOST_MODEL_CBM))
    cat_model.save_model(str(CATBOOST_MODEL_ONNX), format="onnx")

    # Validation metrics
    val_pred = cat_model.predict(X_val_cb)
    cat_rmse = np.sqrt(mean_squared_error(y_val_cb, val_pred))
    cat_mae  = mean_absolute_error(y_val_cb, val_pred)
    cat_r2   = r2_score(y_val_cb, val_pred)

    # Log CatBoost run to MLflow (if available)
    log_mlflow(
        run_name="catboost",
        params=catboost_params,
        metrics={"val_RMSE": cat_rmse, "val_MAE": cat_mae, "val_R2": cat_r2},
        artifacts=[CATBOOST_MODEL_CBM, CATBOOST_MODEL_ONNX, PIPELINE_PATH],
    )

    # ------------------------------------------------------------------- #
    # 7️⃣ Train the PyTorch MLP baseline
    # ------------------------------------------------------------------- #
    class MLP(nn.Module):
        def __init__(self, n_features: int):
            super().__init__()
            self.fc1 = nn.Linear(n_features, 128)
            self.fc2 = nn.Linear(128, 64)
            self.fc3 = nn.Linear(64, 32)
            self.out = nn.Linear(32, 1)
            self.dropout = nn.Dropout(p=0.2)

            # Kaiming‑Uniform initialization
            for layer in (self.fc1, self.fc2, self.fc3, self.out):
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = F.relu(self.fc3(x))
            x = self.out(x)
            return x

    # Prepare PyTorch datasets
    device = torch.device("cpu")
    X_train_torch = torch.from_numpy(X_train_t).float()
    y_train_torch = torch.from_numpy(y_train_t).unsqueeze(1).float()
    X_val_torch   = torch.from_numpy(X_val_t).float()
    y_val_torch   = torch.from_numpy(y_val_t).unsqueeze(1).float()

    train_dataset = torch.utils.data.TensorDataset(X_train_torch, y_train_torch)
    val_dataset   = torch.utils.data.TensorDataset(X_val_torch,   y_val_torch)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False
    )

    mlp = MLP(n_features=X_train_t.shape[1]).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5, verbose=False
    )

    best_val_rmse = float("inf")
    epochs_no_improve = 0
    max_epochs = 200
    early_stop_patience = 15

    for epoch in range(1, max_epochs + 1):
        # ----- training -----
        mlp.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            preds = mlp(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        # ----- validation -----
        mlp.eval()
        val_losses = []
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = mlp(xb)
                loss = criterion(preds, yb)
                val_losses.append(loss.item())
                all_preds.append(preds.cpu().numpy())
                all_targets.append(yb.cpu().numpy())

        # Compute metrics on the concatenated predictions
        preds_np = np.concatenate(all_preds).ravel()
        targets_np = np.concatenate(all_targets).ravel()
        val_rmse = np.sqrt(mean_squared_error(targets_np, preds_np))
        val_mae = mean_absolute_error(targets_np, preds_np)
        val_r2 = r2_score(targets_np, preds_np)

        scheduler.step(val_rmse)

        # Early stopping logic
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            epochs_no_improve = 0
            torch.save(mlp.state_dict(), str(MLP_BEST_PT))
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            print(f"[MLP] Early stopping at epoch {epoch}")
            break

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"[MLP] Epoch {epoch:03d} | Val RMSE: {val_rmse:.4f} | "
                f"Val MAE: {val_mae:.4f} | Val R2: {val_r2:.4f}"
            )

    # Load the best checkpoint for export
    best_state = torch.load(str(MLP_BEST_PT), map_location=device)
    mlp.load_state_dict(best_state)
    mlp.eval()

    # TorchScript export (trace with a single batch)
    example_input = torch.from_numpy(X_train_t[:1]).float().to(device)
    scripted_mlp = torch.jit.trace(mlp, example_input)
    scripted_mlp.save(str(MLP_MODEL_PT))

    # Log MLP run to MLflow (if available)
    log_mlflow(
        run_name="mlp_baseline",
        params={
            "batch_size": 64,
            "learning_rate": 1e-3,
            "max_epochs": max_epochs,
            "early_stop_patience": early_stop_patience,
        },
        metrics={"val_RMSE": best_val_rmse, "val_MAE": val_mae, "val_R2": val_r2},
        artifacts=[MLP_MODEL_PT, PIPELINE_PATH],
    )

    # ------------------------------------------------------------------- #
    # 8️⃣ Copy final artefacts to the evaluation harness directory
    # ------------------------------------------------------------------- #
    for src in [CATBOOST_MODEL_CBM, CATBOOST_MODEL_ONNX, MLP_MODEL_PT]:
        shutil.copy2(src, AGENT_MODEL_DIR / src.name)

    # ------------------------------------------------------------------- #
    # Summary output
    # ------------------------------------------------------------------- #
    print("\n=== End‑to‑End pipeline completed successfully ===")
    print(f"Checksum stored at: {CHECKSUM_FILE}")
    print(f"Preprocess pipeline saved at: {PIPELINE_PATH}")
    print(f"CatBoost native model: {CATBOOST_MODEL_CBM}")
    print(f"CatBoost ONNX model: {CATBOOST_MODEL_ONNX}")
    print(f"MLP TorchScript model: {MLP_MODEL_PT}")
    print(f"All artefacts copied to: {AGENT_MODEL_DIR}")


if __name__ == "__main__":
    main()
