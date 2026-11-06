
# -*- coding: utf-8 -*-
"""
MLOps‑ready end‑to‑end pipeline for the California Housing regression task.

The script performs the following steps:

1️⃣  Copies the user‑provided CSV into the project’s ``data/raw`` folder.  
2️⃣  Computes a SHA‑256 checksum and a tiny JSON data‑profile (rows, columns, dtypes, missing).  
3️⃣  Builds a reproducible scikit‑learn preprocessing pipeline:
    • Median imputation
    • IsolationForest‑based outlier clipping
    • Robust scaling
    • Interaction‑only polynomial features on the three most correlated columns
    • Optional log‑transform of the target (recorded in JSON)
4️⃣  Splits the data (70 % train, 20 % validation, 10 % test).  
5️⃣  Trains a CatBoostRegressor with fixed hyper‑parameters, saves both the native ``.cbm`` and ONNX ``.onnx`` models.  
6️⃣  Trains a small PyTorch MLP baseline, saves the best checkpoint and a TorchScript version.  
7️⃣  Copies the final artefacts to ``./agent_workspace/trained_models`` so that the evaluation harness can locate them.

All paths are built from the environment variable ``CALIHOUSE_ROOT``; if the variable is not set the script falls back
to the directory that contains the script (useful for quick local testing).

Optional MLflow logging is guarded by a safe import – the script works even when MLflow is not installed.
"""

# --------------------------------------------------------------------------- #
# 1️⃣ Imports (optional MLflow is imported lazily)
# --------------------------------------------------------------------------- #
import os
import json
import hashlib
import shutil
import time
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import joblib

# CatBoost
from catboost import CatBoostRegressor, Pool

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Optional MLflow (guarded)
try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except Exception:  # pragma: no cover
    _MLFLOW_AVAILABLE = False

# --------------------------------------------------------------------------- #
# 2️⃣ Helper utilities (paths, checksum, data‑profile)
# --------------------------------------------------------------------------- #
def _project_root() -> Path:
    """Return the absolute project root as a Path object."""
    env_root = os.getenv("CALIHOUSE_ROOT")
    if env_root:
        return Path(env_root).resolve()
    # fallback – assume the script lives inside the project folder
    return Path(__file__).resolve().parents[0]


ROOT = _project_root()

# --------------------------------------------------------------------------- #
#   Absolute paths (hard‑coded source CSV, derived project sub‑folders)
# --------------------------------------------------------------------------- #
RAW_CSV_SRC = Path(
    r"C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\california.csv"
)
RAW_CSV_DST = ROOT / "data" / "raw" / "california_housing_subsample.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
AGENT_MODEL_DIR = ROOT / "agent_workspace" / "trained_models"
METADATA_DIR = ROOT / "metadata"

# Ensure that all required directories exist
for p in (RAW_CSV_DST.parent, PROCESSED_DIR, MODEL_DIR, AGENT_MODEL_DIR, METADATA_DIR):
    p.mkdir(parents=True, exist_ok=True)


def compute_sha256(path: Path) -> str:
    """Return SHA‑256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(data: dict, path: Path) -> None:
    """Write a JSON file with native Python types only."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


# --------------------------------------------------------------------------- #
# 3️⃣ Custom transformers
# --------------------------------------------------------------------------- #
class OutlierClipper(BaseEstimator, TransformerMixin):
    """Fit IsolationForest, then clip each column to its 1‑ and 99‑percentile."""

    def __init__(self, contamination: float = 0.01, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state

    def fit(self, X: np.ndarray, y=None):
        from sklearn.ensemble import IsolationForest

        self.iforest_ = IsolationForest(
            contamination=self.contamination, random_state=self.random_state
        )
        self.iforest_.fit(X)
        # Percentiles are computed on the *clean* data (after imputation)
        self.lower_ = np.percentile(X, 1, axis=0)
        self.upper_ = np.percentile(X, 99, axis=0)
        return self

    def transform(self, X: np.ndarray):
        return np.clip(X, self.lower_, self.upper_)


class InteractionTransformer(BaseEstimator, TransformerMixin):
    """
    Append interaction‑only polynomial features (degree=2) for the *top_k*
    features that have the highest absolute Pearson correlation with the target.
    """

    def __init__(self, top_idx: List[int], degree: int = 2):
        """
        Parameters
        ----------
        top_idx : List[int]
            Column indices (relative to the input matrix) that should be used.
        degree : int, default 2
            Degree for PolynomialFeatures.
        """
        self.top_idx = top_idx
        self.degree = degree

    def fit(self, X: np.ndarray, y=None):
        self.poly_ = PolynomialFeatures(
            degree=self.degree, interaction_only=True, include_bias=False
        )
        self.poly_.fit(X[:, self.top_idx])
        # Store readable feature names for later reference (optional)
        self.input_feature_names_ = [f"x{i}" for i in self.top_idx]
        self.interaction_feature_names_ = self.poly_.get_feature_names_out(
            input_features=self.input_feature_names_
        )
        return self

    def transform(self, X: np.ndarray):
        interactions = self.poly_.transform(X[:, self.top_idx])
        return np.hstack([X, interactions])


# --------------------------------------------------------------------------- #
# 4️⃣ Data ingestion & profile generation
# --------------------------------------------------------------------------- #
def ingest_data() -> pd.DataFrame:
    """Copy the source CSV to the project folder, compute checksum & profile, and return the raw DataFrame."""
    # 1️⃣ Copy
    shutil.copy2(RAW_CSV_SRC, RAW_CSV_DST)

    # 2️⃣ Checksum
    checksum = compute_sha256(RAW_CSV_DST)
    write_json({"sha256": checksum}, METADATA_DIR / "checksum.json")

    # 3️⃣ Load & profile
    df = pd.read_csv(RAW_CSV_DST)
    profile = {
        "row_count": int(df.shape[0]),
        "column_names": list(df.columns),
        "dtypes": {c: str(dt) for c, dt in df.dtypes.items()},
        "missing_counts": df.isna().sum().to_dict(),
    }
    write_json(profile, METADATA_DIR / "data_profile.json")
    return df


# --------------------------------------------------------------------------- #
# 5️⃣ Helper to decide target column & optional log‑transform
# --------------------------------------------------------------------------- #
def infer_target_column(df: pd.DataFrame) -> str:
    """Return the name of the target column (tries a few common names, otherwise last column)."""
    candidates = [
        "median_house_value",
        "MedHouseVal",
        "target",
        "price",
        "value",
    ]
    lowered = [c.lower() for c in df.columns]
    for cand in candidates:
        if cand.lower() in lowered:
            return df.columns[lowered.index(cand.lower())]
    # Fallback – assume the last column is the target
    return df.columns[-1]


def decide_target_transform(y_train: np.ndarray) -> Tuple[np.ndarray, bool]:
    """If target skew > 0.75, apply log1p; return transformed y and flag (plain bool)."""
    from scipy.stats import skew

    sk = skew(y_train)
    use_log = abs(sk) > 0.75
    if use_log:
        y_tr = np.log1p(y_train)
    else:
        y_tr = y_train.astype(np.float32)

    # Store flag as plain Python bool for JSON compatibility
    write_json(
        {"use_log": bool(use_log), "skew": float(sk)},
        METADATA_DIR / "target_transform.json",
    )
    return y_tr, use_log


# --------------------------------------------------------------------------- #
# 6️⃣ Pre‑processing pipeline construction
# --------------------------------------------------------------------------- #
def build_preprocess_pipeline(
    X_train: np.ndarray, y_train: np.ndarray, top_k: int = 3
) -> Tuple[Pipeline, List[int]]:
    """
    Create and fit the preprocessing pipeline on training data.
    Returns the fitted pipeline and the list of column indices used for interaction features.
    """
    # 1️⃣ Median imputation
    imputer = SimpleImputer(strategy="median")

    # 2️⃣ Outlier clipping
    outlier = OutlierClipper(contamination=0.01, random_state=42)

    # 3️⃣ Robust scaling
    scaler = RobustScaler()

    # 4️⃣ Determine top‑k correlated columns (indices)
    # Compute correlation matrix between each feature and the target
    corrs = np.abs(np.corrcoef(X_train, y_train, rowvar=False)[-1, :-1])
    top_idx = list(np.argsort(corrs)[-top_k:])  # indices of the three most correlated features

    # 5️⃣ Interaction transformer (uses the indices from above)
    interaction = InteractionTransformer(top_idx=top_idx, degree=2)

    pipeline = Pipeline(
        [
            ("imputer", imputer),
            ("outlier", outlier),
            ("scaler", scaler),
            ("interaction", interaction),
        ]
    )
    pipeline.fit(X_train, y_train)

    # Store interaction feature names for later reference (optional)
    interaction_names = interaction.interaction_feature_names_.tolist()
    write_json(
        {"interaction_features": interaction_names},
        METADATA_DIR / "interaction_features.json",
    )
    return pipeline, top_idx


# --------------------------------------------------------------------------- #
# 7️⃣ Model training utilities
# --------------------------------------------------------------------------- #
def train_catboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    use_log: bool,
) -> Tuple[CatBoostRegressor, Dict]:
    """Train CatBoost with fixed hyper‑parameters, return model and validation metrics."""
    cat_params = {
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

    train_pool = Pool(X_train, y_train)
    val_pool = Pool(X_val, y_val)

    model = CatBoostRegressor(**cat_params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    # Validation predictions (inverse transform if needed)
    pred_val = model.predict(X_val)
    if use_log:
        pred_val = np.expm1(pred_val)
        y_val_orig = np.expm1(y_val)
    else:
        y_val_orig = y_val

    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_val_orig, pred_val))),
        "MAE": float(mean_absolute_error(y_val_orig, pred_val)),
        "R2": float(r2_score(y_val_orig, pred_val)),
    }

    # Save models
    cat_path = MODEL_DIR / "catboost_model.cbm"
    onnx_path = MODEL_DIR / "catboost_model.onnx"
    model.save_model(str(cat_path))
    model.save_model(str(onnx_path), format="onnx")

    # Optional MLflow logging
    if _MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name="CatBoost_Training"):
            mlflow.log_params(cat_params)
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(cat_path))
            mlflow.log_artifact(str(onnx_path))
            mlflow.log_artifact(str(MODEL_DIR / "preprocess_pipeline.pkl"))

    return model, metrics


class MLPRegressor(nn.Module):
    """Simple feed‑forward MLP for regression."""

    def __init__(self, in_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        # Kaiming‑Uniform init for all Linear layers
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    use_log: bool,
) -> Tuple[MLPRegressor, Dict]:
    """Train the PyTorch MLP, return best model and validation metrics."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_feat = X_train.shape[1]

    model = MLPRegressor(in_feat).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )  # note: no verbose argument for compatibility

    # Datasets / loaders
    train_ds = TensorDataset(
        torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float()
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val).float(), torch.from_numpy(y_val).float()
    )
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    best_rmse = float("inf")
    best_state = None
    epochs_no_improve = 0
    max_epochs = 200
    early_stop_patience = 15

    start_time = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        # ---- Validation ----
        model.eval()
        with torch.no_grad():
            preds_list, targets_list = [], []
            for xb, yb in val_loader:
                xb = xb.to(device)
                out = model(xb).cpu().numpy().ravel()
                preds_list.append(out)
                targets_list.append(yb.numpy())
            preds = np.concatenate(preds_list)
            targets = np.concatenate(targets_list)

        if use_log:
            preds_inv = np.expm1(preds)
            targets_inv = np.expm1(targets)
        else:
            preds_inv = preds
            targets_inv = targets

        rmse = np.sqrt(mean_squared_error(targets_inv, preds_inv))
        mae = mean_absolute_error(targets_inv, preds_inv)
        r2 = r2_score(targets_inv, preds_inv)

        scheduler.step(rmse)

        # Early‑stopping logic
        if rmse < best_rmse - 1e-6:
            best_rmse = rmse
            best_state = model.state_dict()
            best_mae = mae
            best_r2 = r2
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            break

    training_time = time.time() - start_time

    # Load best weights
    model.load_state_dict(best_state)

    # Save checkpoint
    ckpt_path = MODEL_DIR / "mlp_best.pt"
    torch.save(
        {"model_state_dict": best_state, "in_features": in_feat}, ckpt_path
    )

    # Export TorchScript
    model.eval()
    dummy = torch.randn(1, in_feat).to(device)
    scripted = torch.jit.trace(model, dummy)
    scripted_path = MODEL_DIR / "mlp_model.pt"
    scripted.save(str(scripted_path))

    metrics = {
        "RMSE": float(best_rmse),
        "MAE": float(best_mae),
        "R2": float(best_r2),
        "train_time_s": float(training_time),
    }

    # Optional MLflow logging
    if _MLFLOW_AVAILABLE:
        with mlflow.start_run(run_name="MLP_Baseline"):
            mlflow.log_params(
                {
                    "batch_size": 64,
                    "lr": 1e-3,
                    "max_epochs": max_epochs,
                    "early_stop_patience": early_stop_patience,
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(ckpt_path))
            mlflow.log_artifact(str(scripted_path))

    return model, metrics


# --------------------------------------------------------------------------- #
# 8️⃣ Main orchestration
# --------------------------------------------------------------------------- #
def main() -> None:
    # --------------------------------------------------- #
    # 1️⃣ Data ingestion
    # --------------------------------------------------- #
    raw_df = ingest_data()

    # --------------------------------------------------- #
    # 2️⃣ Identify target & feature columns
    # --------------------------------------------------- #
    TARGET_COL = infer_target_column(raw_df)
    FEATURE_COLS = [c for c in raw_df.columns if c != TARGET_COL]

    X_raw = raw_df[FEATURE_COLS].values.astype(np.float32)
    y_raw = raw_df[TARGET_COL].values.astype(np.float32)

    # --------------------------------------------------- #
    # 3️⃣ Train / validation / test split (raw, before any preprocessing)
    # --------------------------------------------------- #
    X_train_raw, X_tmp, y_train_raw, y_tmp = train_test_split(
        X_raw, y_raw, test_size=0.30, random_state=42
    )
    X_val_raw, X_test_raw, y_val_raw, y_test_raw = train_test_split(
        X_tmp, y_tmp, test_size=0.3333, random_state=42
    )  # ≈10 % test

    # --------------------------------------------------- #
    # 4️⃣ Target log‑transform decision
    # --------------------------------------------------- #
    y_train, use_log = decide_target_transform(y_train_raw)

    if use_log:
        y_val = np.log1p(y_val_raw)
        y_test = np.log1p(y_test_raw)
    else:
        y_val = y_val_raw.astype(np.float32)
        y_test = y_test_raw.astype(np.float32)

    # --------------------------------------------------- #
    # 5️⃣ Build & fit preprocessing pipeline on training data
    # --------------------------------------------------- #
    pipeline, _ = build_preprocess_pipeline(X_train_raw, y_train, top_k=3)

    # Save the pipeline for inference
    pipeline_path = MODEL_DIR / "preprocess_pipeline.pkl"
    joblib.dump(pipeline, pipeline_path)

    # Transform all splits
    X_train = pipeline.transform(X_train_raw)
    X_val = pipeline.transform(X_val_raw)
    X_test = pipeline.transform(X_test_raw)

    # --------------------------------------------------- #
    # 6️⃣ Train CatBoost
    # --------------------------------------------------- #
    cat_model, cat_val_metrics = train_catboost(
        X_train, y_train, X_val, y_val, use_log=use_log
    )

    # --------------------------------------------------- #
    # 7️⃣ Train PyTorch MLP baseline
    # --------------------------------------------------- #
    mlp_model, mlp_val_metrics = train_mlp(
        X_train, y_train, X_val, y_val, use_log=use_log
    )

    # --------------------------------------------------- #
    # 8️⃣ Evaluate both models on the held‑out test set
    # --------------------------------------------------- #
    def _evaluate(model, X, y_true, is_torch: bool = False) -> Dict:
        if is_torch:
            model.eval()
            with torch.no_grad():
                preds = (
                    model(torch.from_numpy(X).float())
                    .cpu()
                    .numpy()
                    .ravel()
                )
        else:
            preds = model.predict(X)

        if use_log:
            preds = np.expm1(preds)
            y_true = np.expm1(y_true)

        return {
            "RMSE": float(np.sqrt(mean_squared_error(y_true, preds))),
            "MAE": float(mean_absolute_error(y_true, preds)),
            "R2": float(r2_score(y_true, preds)),
        }

    cat_test_metrics = _evaluate(cat_model, X_test, y_test, is_torch=False)
    mlp_test_metrics = _evaluate(mlp_model, X_test, y_test, is_torch=True)

    # Store test metrics for audit
    write_json(
        {"catboost_test": cat_test_metrics, "mlp_test": mlp_test_metrics},
        METADATA_DIR / "test_metrics.json",
    )

    # --------------------------------------------------- #
    # 9️⃣ Copy final artefacts to the harness directory
    # --------------------------------------------------- #
    artefacts = [
        MODEL_DIR / "catboost_model.cbm",
        MODEL_DIR / "catboost_model.onnx",
        MODEL_DIR / "mlp_model.pt",
        pipeline_path,
    ]
    for src in artefacts:
        dst = AGENT_MODEL_DIR / src.name
        shutil.copy2(src, dst)

    # --------------------------------------------------- #
    # 🔟 Print a concise summary (captured by the evaluation harness)
    # --------------------------------------------------- #
    summary = {
        "train_validation_split": {"train": 0.7, "valid": 0.2, "test": 0.1},
        "target_column": TARGET_COL,
        "target_log_transform": use_log,
        "catboost_validation": cat_val_metrics,
        "catboost_test": cat_test_metrics,
        "mlp_validation": mlp_val_metrics,
        "mlp_test": mlp_test_metrics,
        "saved_models": [p.name for p in artefacts],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
