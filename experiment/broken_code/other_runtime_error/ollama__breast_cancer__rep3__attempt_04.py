
"""
Breast Cancer Tabular Classification – LightGBM + PyTorch Wrapper

This script implements the full end‑to‑end pipeline described in the
play‑book:

1️⃣ Load the raw CSV (data/raw/breast_cancer.csv)  
2️⃣ Pre‑process (encoding, imputation, duplicate removal, split, optional SMOTE)  
3️⃣ Train a tuned LightGBM model, calibrate it and serialize it  
4️⃣ Wrap the calibrated model in a PyTorch‑style ``nn.Module``  
5️⃣ Save all artefacts under ``./agent_workspace/trained_models``  

The code is defensive: if optional packages such as ``pyarrow`` or ``mlflow``
are missing it will automatically fall back to CSV storage and skip MLflow
logging, so the script can run on a minimal environment.

Run with:
    python pipeline.py
"""

# -------------------------------------------------------------------------
# Imports & optional‑dependency handling
# -------------------------------------------------------------------------
import os
import json
import hashlib
import warnings
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss

# ---- Optional pyarrow ----------------------------------------------------
try:
    import pyarrow  # noqa: F401
    HAVE_PYARROW = True
except Exception:  # pragma: no cover
    HAVE_PYARROW = False

# ---- Optional MLflow ------------------------------------------------------
try:
    import mlflow

    class mlflow_run:
        """Thin wrapper around mlflow.start_run that can be used as a context."""
        def __init__(self, name: str, tags: dict | None = None):
            self.name = name
            self.tags = tags or {}

        def __enter__(self):
            self.run = mlflow.start_run(run_name=self.name, tags=self.tags)
            return self.run

        def __exit__(self, exc_type, exc_val, exc_tb):
            mlflow.end_run()
except Exception:  # pragma: no cover
    # Dummy objects when mlflow is not installed
    class _DummyRun:
        def __enter__(self): return self
        def __exit__(self, *exc): pass

    class mlflow_run:
        def __init__(self, *_, **__): pass
        def __enter__(self): return _DummyRun()
        def __exit__(self, *exc): pass

    class _DummyMlflow:
        @staticmethod
        def log_param(*_, **__): pass
        @staticmethod
        def log_metric(*_, **__): pass
        @staticmethod
        def log_artifact(*_, **__): pass
        @staticmethod
        def start_run(*_, **__): return _DummyRun()
    mlflow = _DummyMlflow()  # type: ignore

# -------------------------------------------------------------------------
# Global configuration & reproducibility
# -------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Repository‑relative paths
ROOT_DIR = Path.cwd()
RAW_CSV = ROOT_DIR / "data" / "raw" / "breast_cancer.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "agent_workspace" / "trained_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------------------------------
# Helper utilities
# -------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    """Return SHA‑256 checksum of a file (used for optional MLflow logging)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _determine_file_extension() -> str:
    """Return '.parquet' if pyarrow is available, otherwise '.csv'."""
    return ".parquet" if HAVE_PYARROW else ".csv"


def _save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """
    Save ``df`` to ``path`` using Parquet (zstd) when pyarrow is present,
    otherwise fall back to CSV.
    The function respects the suffix of ``path`` – it will write to that exact
    suffix.
    """
    if HAVE_PYARROW and path.suffix == ".parquet":
        df.to_parquet(path, engine="pyarrow", compression="zstd")
    else:
        # Ensure we write CSV regardless of the suffix (use .csv)
        csv_path = path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)


def _load_dataframe(path: Path) -> pd.DataFrame:
    """
    Load a DataFrame saved by ``_save_dataframe``.
    Detects Parquet vs CSV automatically (based on file existence).
    """
    if HAVE_PYARROW and path.suffix == ".parquet" and path.exists():
        return pd.read_parquet(path, engine="pyarrow")
    # CSV fallback – the actual file will have a .csv suffix
    csv_path = path.with_suffix(".csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Neither {path} nor {csv_path} exist.")
    return pd.read_csv(csv_path)


# -------------------------------------------------------------------------
# 1️⃣ Data ingestion & preprocessing
# -------------------------------------------------------------------------
def preprocess_and_split(
    csv_path: Path,
    processed_dir: Path,
    test_size: float = 0.10,
    val_size: float = 0.20,
    smote_ratio_threshold: float = 1.5,
) -> Tuple[Path, Path, Path]:
    """
    Load raw CSV, clean it, optionally apply SMOTE on the training split,
    and write train/val/test files.
    Returns the three file *paths* (with the correct suffix).
    """
    # ---- Load -------------------------------------------------------------
    df = pd.read_csv(csv_path)

    # ---- Target encoding --------------------------------------------------
    target_col = df.columns[-1]          # classic Breast Cancer dataset: last column is diagnosis
    if df[target_col].dtype == object:
        target_map = {"M": 0, "B": 1}
        df[target_col] = df[target_col].map(target_map)
    else:
        target_map = None

    # ---- Missing‑value imputation (median) ---------------------------------
    df = df.fillna(df.median(numeric_only=True))

    # ---- Duplicate removal ------------------------------------------------
    dup_cnt = df.duplicated().sum()
    if dup_cnt:
        df = df.drop_duplicates().reset_index(drop=True)

    # ---- Feature order (store for inference) -----------------------------
    feature_cols = [c for c in df.columns if c != target_col]
    feature_order_path = MODEL_DIR / "feature_order.json"
    json.dump(feature_cols, open(feature_order_path, "w"))

    # ---- Train / Val / Test split (stratified) ----------------------------
    X = df[feature_cols].values
    y = df[target_col].values
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), stratify=y, random_state=SEED
    )
    val_frac = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_frac, stratify=y_temp, random_state=SEED
    )

    # ---- SMOTE on training set only (if needed) -------------------------
    ratio = np.bincount(y_train)[0] / np.bincount(y_train)[1]
    smote_applied = False
    if ratio > smote_ratio_threshold:
        sm = SMOTE(random_state=SEED)
        X_train, y_train = sm.fit_resample(X_train, y_train)
        smote_applied = True

    # ---- Persist splits ---------------------------------------------------
    processed_dir.mkdir(parents=True, exist_ok=True)
    ext = _determine_file_extension()
    train_path = processed_dir / f"train{ext}"
    val_path = processed_dir / f"val{ext}"
    test_path = processed_dir / f"test{ext}"

    _save_dataframe(
        pd.DataFrame(np.column_stack([X_train, y_train]),
                     columns=feature_cols + ["target"]),
        train_path,
    )
    _save_dataframe(
        pd.DataFrame(np.column_stack([X_val, y_val]),
                     columns=feature_cols + ["target"]),
        val_path,
    )
    _save_dataframe(
        pd.DataFrame(np.column_stack([X_test, y_test]),
                     columns=feature_cols + ["target"]),
        test_path,
    )

    # ---- Optional MLflow logging -----------------------------------------
    with mlflow_run("preprocess") as _:
        mlflow.log_artifact(str(csv_path), artifact_path="raw")
        mlflow.log_artifact(str(train_path), artifact_path="processed")
        mlflow.log_artifact(str(val_path), artifact_path="processed")
        mlflow.log_artifact(str(test_path), artifact_path="processed")
        mlflow.log_metric("train_samples", len(y_train))
        mlflow.log_metric("val_samples", len(y_val))
        mlflow.log_metric("test_samples", len(y_test))
        mlflow.log_param("smote_applied", smote_applied)
        mlflow.log_param(
            "target_mapping", json.dumps(target_map) if target_map else "numeric"
        )
        mlflow.log_param("raw_sha256", sha256_file(csv_path))

    return train_path, val_path, test_path


# -------------------------------------------------------------------------
# 2️⃣ LightGBM training + calibration
# -------------------------------------------------------------------------
def train_lightgbm(
    train_path: Path,
    val_path: Path,
    model_dir: Path,
    n_jobs: int = 4,
) -> Tuple[Path, Dict[str, float]]:
    """
    Train LightGBM with the hyper‑parameters from the play‑book,
    calibrate with sigmoid, and serialize both the raw and calibrated models.
    Returns the path to the calibrated model and a dict of metrics.
    """
    # Load splits (parquet or CSV – handled internally)
    train_df = _load_dataframe(train_path)
    val_df = _load_dataframe(val_path)

    X_train = train_df.drop(columns=["target"]).values
    y_train = train_df["target"].values
    X_val = val_df.drop(columns=["target"]).values
    y_val = val_df["target"].values

    # Compute scale_pos_weight (helps with class imbalance)
    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    # Hyper‑parameters (play‑book)
    params = {
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 12,
        "min_data_in_leaf": 25,
        "feature_fraction": 0.92,
        "bagging_fraction": 0.80,
        "bagging_freq": 5,
        "lambda_l1": 0.0,
        "lambda_l2": 0.6,
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "n_jobs": n_jobs,
        "random_state": SEED,
        "scale_pos_weight": scale_pos_weight,
    }

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    with mlflow_run("lightgbm_train", tags={"framework": "lightgbm"}) as _:
        booster = lgb.train(
            params,
            lgb_train,
            num_boost_round=420,
            valid_sets=[lgb_train, lgb_val],
            valid_names=["train", "val"],
            early_stopping_rounds=30,
            verbose_eval=False,
        )

        # Validation metrics
        val_pred = booster.predict(X_val, num_iteration=booster.best_iteration)
        auc_val = roc_auc_score(y_val, val_pred)
        brier = brier_score_loss(y_val, val_pred)

        # Calibration (sigmoid)
        calibrator = CalibratedClassifierCV(
            base_estimator=booster, method="sigmoid", cv="prefit"
        )
        calibrator.fit(X_train, y_train)

        # Serialize calibrated model
        calibrated_path = model_dir / "lightgbm_best_calibrated.pkl"
        joblib.dump(calibrator, calibrated_path)

        # Log artefacts & metrics (if mlflow is active)
        mlflow.log_params(params)
        mlflow.log_metric("auc_val", auc_val)
        mlflow.log_metric("best_iteration", booster.best_iteration)
        mlflow.log_metric("brier_score", brier)
        mlflow.log_artifact(str(calibrated_path), artifact_path="model")
        mlflow.log_artifact(
            str(model_dir / "feature_order.json"), artifact_path="model"
        )

        metrics = {
            "auc_val": auc_val,
            "best_iteration": booster.best_iteration,
            "brier_score": brier,
        }

    return calibrated_path, metrics


# -------------------------------------------------------------------------
# 3️⃣ PyTorch‑style wrapper
# -------------------------------------------------------------------------
class LightGBMClassifier(nn.Module):
    """
    Thin PyTorch wrapper around a calibrated LightGBM model.
    ``forward`` receives a ``torch.Tensor`` of shape (batch, n_features)
    and returns probabilities of the positive class as a ``torch.Tensor``
    of shape (batch, 1).
    """

    def __init__(self, model_path: Path, feature_order_path: Path):
        super().__init__()
        self.booster = joblib.load(model_path)          # CalibratedClassifierCV
        self.feature_order: List[str] = json.load(open(feature_order_path, "r"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.is_cuda:
            x = x.cpu()
        # Ensure contiguous NumPy array of type float32
        np_arr = x.detach().numpy().astype(np.float32)
        # Predict probabilities for class 1 (benign)
        probs = self.booster.predict_proba(np_arr)[:, 1]
        return torch.from_numpy(probs).unsqueeze(1).float()


def save_torch_wrapper(wrapper: LightGBMClassifier, out_path: Path) -> None:
    """Save only the ``state_dict`` – the heavy LightGBM object stays on disk."""
    torch.save(wrapper.state_dict(), out_path)


# -------------------------------------------------------------------------
# 4️⃣ Orchestrator (main)
# -------------------------------------------------------------------------
def main():
    # -----------------------------------------------------------------
    # Step 1 – Pre‑process & split
    # -----------------------------------------------------------------
    train_path, val_path, test_path = preprocess_and_split(
        csv_path=RAW_CSV,
        processed_dir=PROCESSED_DIR,
    )
    print(f"Splits saved → train:{train_path}, val:{val_path}, test:{test_path}")

    # -----------------------------------------------------------------
    # Step 2 – Train LightGBM + calibrate
    # -----------------------------------------------------------------
    calibrated_model_path, train_metrics = train_lightgbm(
        train_path=train_path,
        val_path=val_path,
        model_dir=MODEL_DIR,
    )
    print(f"Training completed – Validation AUC: {train_metrics['auc_val']:.4f}")

    # -----------------------------------------------------------------
    # Step 3 – Create & save PyTorch wrapper
    # -----------------------------------------------------------------
    feature_order_path = MODEL_DIR / "feature_order.json"
    wrapper = LightGBMClassifier(
        model_path=calibrated_model_path,
        feature_order_path=feature_order_path,
    )
    wrapper_path = MODEL_DIR / "lightgbm_torch_wrapper.pt"
    save_torch_wrapper(wrapper, wrapper_path)
    print(f"PyTorch wrapper state dict saved to {wrapper_path}")

    # -----------------------------------------------------------------
    # Step 4 – Save a JSON model‑card / profile
    # -----------------------------------------------------------------
    profile = {
        "model_name": "lightgbm",
        "framework": "lightgbm==4.5.0",
        "hyperparameters": {
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": 12,
            "min_data_in_leaf": 25,
            "feature_fraction": 0.92,
            "bagging_fraction": 0.80,
            "bagging_freq": 5,
            "lambda_l1": 0.0,
            "lambda_l2": 0.6,
            "n_estimators": 420,
            "scale_pos_weight": wrapper.booster.base_estimator.scale_pos_weight,
            "objective": "binary",
            "metric": "auc",
            "random_state": SEED,
            "n_jobs": 4,
        },
        "metrics": train_metrics,
        "artifact_paths": {
            "calibrated_model": str(calibrated_model_path),
            "torch_wrapper_state_dict": str(wrapper_path),
            "feature_order": str(feature_order_path),
        },
    }
    profile_path = MODEL_DIR / "lightgbm_profile.json"
    json.dump(profile, open(profile_path, "w"), indent=2)
    print(f"Model profile written to {profile_path}")


if __name__ == "__main__":
    # Suppress non‑critical LightGBM warnings for cleaner logs / CI output
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
