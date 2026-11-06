
"""
Breast‑cancer binary classification pipeline (LightGBM + PyTorch wrapper)

Run as:
    python -m src.pipeline               # default run
    python -m src.pipeline --use-smote   # optional SMOTE
    python -m src.pipeline --no-mlflow   # disable MLflow logging

All artefacts are written under the repository layout described in the
specification, e.g.:
    data/processed/
    agent_workspace/trained_models/
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

# ----------------------------------------------------------------------
# Optional dependencies – they must never crash the pipeline if missing.
# ----------------------------------------------------------------------
try:
    import pyarrow  # noqa: F401
    _HAS_PYARROW = True
except Exception:  # pragma: no cover
    _HAS_PYARROW = False

try:
    import mlflow
    _HAS_MLFLOW = True
except Exception:  # pragma: no cover
    _HAS_MLFLOW = False

# ----------------------------------------------------------------------
# Core dependencies (always required)
# ----------------------------------------------------------------------
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from imblearn.over_sampling import SMOTE
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------------
# Helper utilities for I/O
# ----------------------------------------------------------------------
def _determine_file_extension() -> str:
    """Return '.parquet' if pyarrow is available, otherwise '.csv'."""
    return ".parquet" if _HAS_PYARROW else ".csv"


def _save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` to ``path`` using Parquet when possible, else CSV."""
    if path.suffix == ".parquet" and _HAS_PYARROW:
        df.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    else:
        df.to_csv(path, index=False)


def _load_dataframe(path: Path) -> pd.DataFrame:
    """Read a DataFrame from ``path`` (Parquet if possible, else CSV)."""
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")
    if path.suffix == ".parquet" and _HAS_PYARROW:
        return pd.read_parquet(path, engine="pyarrow")
    return pd.read_csv(path)


# ----------------------------------------------------------------------
# 1️⃣  Pre‑processing
# ----------------------------------------------------------------------
def run_preprocess(use_smote: bool = False, no_mlflow: bool = False) -> Tuple[Path, Optional[Path], Path]:
    """
    Perform cleaning, optional SMOTE, and deterministic stratified splits.

    Returns
    -------
    train_path : Path
    val_path   : Path | None
    test_path  : Path
    """
    raw_path = Path("data/raw/breast_cancer.csv")
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw CSV not found at expected location: {raw_path}")

    df = pd.read_csv(raw_path)

    # --------------------------------------------------------------
    # Target encoding (last column)
    # --------------------------------------------------------------
    target_col = df.columns[-1]
    if df[target_col].dtype == object:
        mapping = {"M": 0, "B": 1}
        df[target_col] = df[target_col].map(mapping)
        # store mapping
        target_map_path = Path("agent_workspace/trained_models/target_mapping.json")
        target_map_path.parent.mkdir(parents=True, exist_ok=True)
        target_map_path.write_text(json.dumps(mapping, indent=2), encoding="utf-8")
    else:
        mapping = None  # already numeric

    # --------------------------------------------------------------
    # Median imputation (all features are numeric)
    # --------------------------------------------------------------
    df = df.fillna(df.median())

    # --------------------------------------------------------------
    # Duplicate removal
    # --------------------------------------------------------------
    before_dup = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = before_dup - len(df)

    # --------------------------------------------------------------
    # Feature order persistence
    # --------------------------------------------------------------
    feature_cols = [c for c in df.columns if c != target_col]
    feature_order_path = Path("agent_workspace/trained_models/feature_order.json")
    feature_order_path.parent.mkdir(parents=True, exist_ok=True)
    feature_order_path.write_text(json.dumps(feature_cols, indent=2), encoding="utf-8")

    # --------------------------------------------------------------
    # Stratified split
    # --------------------------------------------------------------
    X = df[feature_cols]
    y = df[target_col]

    # first split: train_val / test (10% test)
    train_val_X, test_X, train_val_y, test_y = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=42
    )
    # second split: train / val (20% of original = 0.20/0.90 of train_val)
    train_X, val_X, train_y, val_y = train_test_split(
        train_val_X,
        train_val_y,
        test_size=0.20 / 0.90,
        stratify=train_val_y,
        random_state=42,
    )

    # sanity check – each split must have at least 5 rows
    if any(len(s) < 5 for s in (train_X, val_X, test_X)):
        # fallback: 70/30 train‑test, no validation
        train_X, test_X, train_y, test_y = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=42
        )
        val_X = None
        val_y = None
        val_path = None
    else:
        val_path = Path("data/processed/val" + _determine_file_extension())

    # --------------------------------------------------------------
    # Optional SMOTE on training set only
    # --------------------------------------------------------------
    if use_smote:
        class_counts = train_y.value_counts()
        if class_counts.max() / class_counts.min() > 1.5:
            sm = SMOTE(random_state=42)
            train_X, train_y = sm.fit_resample(train_X, train_y)

    # --------------------------------------------------------------
    # Persist splits
    # --------------------------------------------------------------
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    ext = _determine_file_extension()
    train_path = processed_dir / f"train{ext}"
    test_path = processed_dir / f"test{ext}"
    _save_dataframe(pd.concat([train_X, train_y.rename("target")], axis=1), train_path)
    _save_dataframe(pd.concat([test_X, test_y.rename("target")], axis=1), test_path)
    if val_X is not None:
        _save_dataframe(pd.concat([val_X, val_y.rename("target")], axis=1), val_path)

    # --------------------------------------------------------------
    # Optional MLflow logging
    # --------------------------------------------------------------
    if _HAS_MLFLOW and not no_mlflow:
        mlflow.set_experiment("breast_cancer_autml")
        with mlflow.start_run(run_name="preprocess"):
            mlflow.log_param("use_smote", use_smote)
            mlflow.log_param("duplicates_removed", duplicates_removed)
            mlflow.log_metric("train_samples", len(train_X))
            mlflow.log_metric("val_samples", len(val_X) if val_X is not None else 0)
            mlflow.log_metric("test_samples", len(test_X))
            mlflow.log_artifact(str(train_path))
            mlflow.log_artifact(str(test_path))
            if val_X is not None:
                mlflow.log_artifact(str(val_path))
            mlflow.log_artifact(str(feature_order_path))
            if mapping is not None:
                mlflow.log_artifact(str(target_map_path))

    # --------------------------------------------------------------
    # ASCII console output
    # --------------------------------------------------------------
    print(f"--> Pre-processing completed. Files written with extension '{ext}'")
    print(f"    Train samples: {len(train_X)}")
    print(f"    Validation samples: {len(val_X) if val_X is not None else 0}")
    print(f"    Test samples: {len(test_X)}")
    return train_path, val_path, test_path


# ----------------------------------------------------------------------
# 2️⃣  LightGBM training + calibration
# ----------------------------------------------------------------------
def train_lightgbm(
    train_path: Path,
    val_path: Optional[Path],
    model_dir: Path = Path("agent_workspace/trained_models"),
    no_mlflow: bool = False,
) -> Tuple[Path, dict]:
    """
    Train LightGBM on the provided splits, calibrate with sigmoid,
    and persist the calibrated model.

    Returns
    -------
    calibrated_model_path : Path
    profile               : dict
    """
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    train_df = _load_dataframe(train_path)
    if val_path is None:
        # fallback split – create a validation set from training data
        X_full = train_df.drop(columns=["target"])
        y_full = train_df["target"]
        train_df, val_df = train_test_split(
            train_df, test_size=0.10, stratify=y_full, random_state=42
        )
    else:
        val_df = _load_dataframe(val_path)

    # Separate features / target
    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]
    X_val = val_df.drop(columns=["target"])
    y_val = val_df["target"]

    # scale_pos_weight
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = (neg / pos) if pos > 0 else 1.0

    # Hyper‑parameters (exact)
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
        "n_jobs": 4,
        "random_state": 42,
        "scale_pos_weight": scale_pos_weight,
    }

    # LightGBM datasets
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    # Callbacks for early stopping (LightGBM 4.x)
    callbacks = [
        lgb.early_stopping(stopping_rounds=30, verbose=False),
        lgb.log_evaluation(period=0),  # silence default printing
    ]

    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=420,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "val"],
        callbacks=callbacks,
    )

    # Validation metrics
    val_pred = booster.predict(X_val, num_iteration=booster.best_iteration)
    auc_val = roc_auc_score(y_val, val_pred)
    brier_val = brier_score_loss(y_val, val_pred)

    # Calibration (sigmoid)
    calibrator = CalibratedClassifierCV(
        base_estimator=booster, method="sigmoid", cv="prefit"
    )
    calibrator.fit(X_train, y_train)

    # Persist calibrated model
    calibrated_path = model_dir / "lightgbm_best_calibrated.pkl"
    joblib.dump(calibrator, calibrated_path)

    # Profile JSON
    profile = {
        "hyperparameters": params,
        "best_iteration": int(booster.best_iteration),
        "validation_auc": float(auc_val),
        "validation_brier": float(brier_val),
        "model_size_bytes": calibrated_path.stat().st_size,
        "num_trees": int(booster.num_trees()),
        "approx_leaf_nodes": int(booster.num_trees() * params["max_depth"]),
        "estimated_flops": int(booster.num_trees() * params["max_depth"] * 2),
    }
    profile_path = model_dir / "lightgbm_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    # Optional MLflow logging
    if _HAS_MLFLOW and not no_mlflow:
        mlflow.set_experiment("breast_cancer_autml")
        with mlflow.start_run(run_name="lightgbm_train"):
            for k, v in params.items():
                mlflow.log_param(k, v)
            mlflow.log_metric("validation_auc", auc_val)
            mlflow.log_metric("validation_brier", brier_val)
            mlflow.log_metric("best_iteration", booster.best_iteration)
            mlflow.log_artifact(str(calibrated_path))
            mlflow.log_artifact(str(profile_path))

    print("--> LightGBM training & calibration completed.")
    print(f"    Validation AUC : {auc_val:.4f}")
    print(f"    Model size    : {profile['model_size_bytes'] / 1e6:.2f} MB")
    return calibrated_path, profile


# ----------------------------------------------------------------------
# 3️⃣  PyTorch wrapper
# ----------------------------------------------------------------------
class LightGBMClassifier(nn.Module):
    """
    Thin PyTorch wrapper around a calibrated LightGBM model.
    The wrapper is stateless except for a dummy Parameter so that
    ``state_dict()`` is non‑empty (required for torch.save).
    """

    def __init__(self, model_path: Path, feature_order_path: Path):
        super().__init__()
        if not model_path.is_file():
            raise FileNotFoundError(f"Calibrated model not found: {model_path}")
        if not feature_order_path.is_file():
            raise FileNotFoundError(f"Feature order file not found: {feature_order_path}")

        self.calibrator = joblib.load(model_path)
        self.feature_order = json.loads(feature_order_path.read_text(encoding="utf-8"))
        # dummy parameter – no gradients needed
        self._dummy = nn.Parameter(torch.empty(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 2:
            raise ValueError("Input tensor must be 2‑D (batch, features)")
        if x.size(1) != len(self.feature_order):
            raise ValueError(
                f"Expected {len(self.feature_order)} features, got {x.size(1)}"
            )
        # Ensure CPU numpy array
        np_arr = x.detach().cpu().numpy().astype(np.float32)
        probs = self.calibrator.predict_proba(np_arr)[:, 1]  # prob of class 1 (benign)
        probs_tensor = torch.from_numpy(probs.astype(np.float32)).unsqueeze(1)
        return probs_tensor

    def state_dict(self, *args, **kwargs):
        # Return a minimal dict – the heavy model lives on disk.
        return {"_dummy": self._dummy}

    def load_state_dict(self, state_dict, strict: bool = True):
        # No‑op – nothing to load.
        pass


def save_torch_wrapper(wrapper: LightGBMClassifier, out_path: Path) -> None:
    """Save only the (tiny) state dict of the wrapper."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(wrapper.state_dict(), out_path)


# ----------------------------------------------------------------------
# 4️⃣  Orchestrator (single entry point)
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Breast‑cancer LightGBM + PyTorch wrapper pipeline"
    )
    parser.add_argument(
        "--use-smote",
        action="store_true",
        help="Apply SMOTE on the training set (default: off)",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging even if the package is installed",
    )
    args = parser.parse_args()

    # --------------------------------------------------------------
    # Step 1 – preprocessing
    # --------------------------------------------------------------
    train_path, val_path, test_path = run_preprocess(
        use_smote=args.use_smote, no_mlflow=args.no_mlflow
    )

    # --------------------------------------------------------------
    # Step 2 – LightGBM training & calibration
    # --------------------------------------------------------------
    calibrated_path, profile = train_lightgbm(
        train_path, val_path, no_mlflow=args.no_mlflow
    )

    # --------------------------------------------------------------
    # Step 3 – instantiate PyTorch wrapper and save its state dict
    # --------------------------------------------------------------
    feature_order_path = Path("agent_workspace/trained_models/feature_order.json")
    wrapper = LightGBMClassifier(calibrated_path, feature_order_path)
    wrapper_path = Path("agent_workspace/trained_models/lightgbm_torch_wrapper.pt")
    save_torch_wrapper(wrapper, wrapper_path)

    # --------------------------------------------------------------
    # Final ASCII summary
    # --------------------------------------------------------------
    abs_model_dir = Path("agent_workspace/trained_models").resolve()
    print("\n--> Pipeline completed successfully.")
    print(f"    Model artefacts stored in: {abs_model_dir}")
    print(f"    Validation AUC : {profile['validation_auc']:.4f}")
    print(f"    Model size    : {profile['model_size_bytes'] / 1e6:.2f} MB")
    print(f"    Wrapper state dict saved at: {wrapper_path.resolve()}")


if __name__ == "__main__":
    # Allows execution as a script: `python src/pipeline.py`
    main()
