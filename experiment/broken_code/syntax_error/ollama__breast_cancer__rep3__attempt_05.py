
"""
Tabular classification pipeline for the Breast Cancer Wisconsin dataset.

Features
--------
* Optional support for `pyarrow` (Parquet) – falls back to CSV if unavailable.
* Optional support for `mlflow` – a no‑op dummy is used when the package is missing.
* Reproducible train/val/test split (70/20/10) with a sanity check for tiny splits.
* Optional SMOTE (disabled by default for this dataset).
* LightGBM model trained with a calibrated sigmoid and saved with `joblib`.
* PyTorch wrapper (`LightGBMClassifier`) exposing a `forward` method that returns
  class‑1 probabilities.
* Simple Gradio demo that accepts the 30 feature values and returns the benign
  probability.
* All artefacts are written under `agent_workspace/trained_models/`.

Run the whole pipeline with:
    python -m src.pipeline [--use-smote] [--no-mlflow]

The script is self‑contained – no external configuration files are required.
"""

import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Tuple, List, Optional

# --------------------------------------------------------------------------- #
# 1️⃣  Optional imports (pyarrow, mlflow) – provide safe fall‑backs
# --------------------------------------------------------------------------- #
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

    class _DummyMLflowRun:
        """Very small subset of the mlflow API used in this repo."""

        def __init__(self, *_, **__):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            return False

        def log_param(self, *_, **__):
            pass

        def log_metric(self, *_, **__):
            pass

        def log_artifact(self, *_, **__):
            pass

        def set_tag(self, *_, **__):
            pass

    class _DummyMLflow:
        @staticmethod
        def start_run(*_, **__):
            return _DummyMLflowRun()

        @staticmethod
        def log_param(*_, **__):
            pass

        @staticmethod
        def log_metric(*_, **__):
            pass

        @staticmethod
        def log_artifact(*_, **__):
            pass

    mlflow = _DummyMLflow()  # type: ignore

# --------------------------------------------------------------------------- #
# 2️⃣  Global constants & helper utilities
# --------------------------------------------------------------------------- #
SEED = 42
random.seed(SEED)
import numpy as np

np.random.seed(SEED)

ROOT_DIR = Path(__file__).resolve().parents[2]  # repo root (src/../..)
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "agent_workspace" / "trained_models"

# Ensure directories exist
for p in (RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR):
    p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# 3️⃣  I/O helpers (CSV ↔ Parquet fallback)
# --------------------------------------------------------------------------- #
def _determine_file_extension() -> str:
    """Return the file suffix we can safely write with the current env."""
    return ".parquet" if _HAS_PYARROW else ".csv"


def _save_dataframe(df: "pd.DataFrame", path: Path) -> None:
    """Save df using Parquet if possible, otherwise CSV."""
    if _HAS_PYARROW:
        df.to_parquet(path, engine="pyarrow", compression="zstd", index=False)
    else:
        df.to_csv(path, index=False)


def _load_dataframe(path: Path) -> "pd.DataFrame":
    """Load a DataFrame regardless of its extension."""
    if path.suffix == ".parquet":
        import pandas as pd

        return pd.read_parquet(path, engine="pyarrow")
    else:
        import pandas as pd

        return pd.read_csv(path)


# --------------------------------------------------------------------------- #
# 4️⃣  Pre‑processing
# --------------------------------------------------------------------------- #
def run_preprocess(use_smote: bool = False) -> Tuple[Path, Path, Path]:
    """
    Load the raw CSV, clean it, split it, optionally apply SMOTE and write
    the three splits to disk.

    Returns
    -------
    train_path, val_path, test_path : pathlib.Path
        Absolute paths to the three split files.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE

    # ------------------------------------------------------------------- #
    # 4.1 Load raw CSV (the user must have placed the file here)
    # ------------------------------------------------------------------- #
    raw_csv_path = RAW_DATA_DIR / "breast_cancer.csv"
    if not raw_csv_path.is_file():
        raise FileNotFoundError(
            f"Raw dataset not found at {raw_csv_path}. "
            "Copy the original CSV to this location."
        )
    df = pd.read_csv(raw_csv_path)

    # ------------------------------------------------------------------- #
    # 4.2 Target encoding (assume last column is the label)
    # ------------------------------------------------------------------- #
    target_col = df.columns[-1]
    if df[target_col].dtype == object:
        mapping = {"M": 0, "B": 1}
        df[target_col] = df[target_col].map(mapping)
        # store mapping for reproducibility
        (MODEL_DIR / "target_mapping.json").write_text(json.dumps(mapping))
    else:
        mapping = None

    # ------------------------------------------------------------------- #
    # 4.3 Median imputation + duplicate removal
    # ------------------------------------------------------------------- #
    df = df.fillna(df.median())
    before_dup = len(df)
    df = df.drop_duplicates()
    dup_removed = before_dup - len(df)

    # ------------------------------------------------------------------- #
    # 4.4 Feature order JSON (required by the PyTorch wrapper)
    # ------------------------------------------------------------------- #
    feature_cols = [c for c in df.columns if c != target_col]
    (MODEL_DIR / "feature_order.json").write_text(json.dumps(feature_cols, indent=2))

    # ------------------------------------------------------------------- #
    # 4.5 Stratified split with sanity checks
    # ------------------------------------------------------------------- #
    X = df[feature_cols]
    y = df[target_col]

    # initial 70/30 split (train+val vs test)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.10,
        stratify=y,
        random_state=SEED,
    )
    # now split temp into train/val (70/20)
    val_ratio = 0.20 / 0.90  # proportion of the remaining 90%
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_ratio,
        stratify=y_temp,
        random_state=SEED,
    )

    # sanity check – each split must have at least 5 samples
    for name, split in zip(
        ["train", "val", "test"], [(X_train, y_train), (X_val, y_val), (X_test, y_test)]
    ):
        if len(split[0]) < 5:
            # fallback to 70/30 (train/test) and drop validation
            X_train = X_temp
            y_train = y_temp
            X_val = pd.DataFrame(columns=feature_cols)
            y_val = pd.Series(dtype=int)
            break

    # ------------------------------------------------------------------- #
    # 4.6 Optional SMOTE (default off)
    # ------------------------------------------------------------------- #
    if use_smote and len(y_train) > 0:
        # compute imbalance ratio
        class_counts = y_train.value_counts()
        ratio = class_counts.max() / class_counts.min()
        if ratio > 1.5:
            sm = SMOTE(random_state=SEED)
            X_res, y_res = sm.fit_resample(X_train, y_train)
            X_train, y_train = X_res, y_res

    # ------------------------------------------------------------------- #
    # 4.7 Persist splits
    # ------------------------------------------------------------------- #
    ext = _determine_file_extension()
    train_path = PROCESSED_DATA_DIR / f"train{ext}"
    val_path = PROCESSED_DATA_DIR / f"val{ext}"
    test_path = PROCESSED_DATA_DIR / f"test{ext}"

    _save_dataframe(pd.concat([X_train, y_train], axis=1), train_path)
    if not X_val.empty:
        _save_dataframe(pd.concat([X_val, y_val], axis=1), val_path)
    _save_dataframe(pd.concat([X_test, y_test], axis=1), test_path)

    # ------------------------------------------------------------------- #
    # 4.8 MLflow logging (if available)
    # ------------------------------------------------------------------- #
    with mlflow.start_run() if _HAS_MLFLOW else mlflow.start_run():
        mlflow.log_param("use_smote", use_smote)
        mlflow.log_metric("train_samples", len(X_train))
        mlflow.log_metric("val_samples", len(X_val))
        mlflow.log_metric("test_samples", len(X_test))
        mlflow.log_metric("duplicates_removed", dup_removed)
        if mapping:
            mlflow.log_artifact(str(MODEL_DIR / "target_mapping.json"))
        mlflow.log_artifact(str(MODEL_DIR / "feature_order.json"))

    print(f"--> Pre‑processing completed. Files written with extension '{ext}'.")
    return train_path, val_path, test_path


# --------------------------------------------------------------------------- #
# 5️⃣  LightGBM training + calibration
# --------------------------------------------------------------------------- #
def train_lightgbm(
    train_path: Path,
    val_path: Optional[Path],
    model_dir: Path = MODEL_DIR,
) -> Tuple[Path, dict]:
    """
    Train LightGBM on the training split, calibrate with sigmoid, and persist.

    Returns
    -------
    model_path : pathlib.Path
        Path to the calibrated model pickle.
    metrics : dict
        Validation metrics (AUC, Brier) and training info.
    """
    import pandas as pd
    import lightgbm as lgb
    import joblib
    from sklearn.metrics import roc_auc_score, brier_score_loss

    # ------------------------------------------------------------------- #
    # Load data
    # ------------------------------------------------------------------- #
    train_df = _load_dataframe(train_path)
    target_col = train_df.columns[-1]
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col]

    if val_path and val_path.is_file():
        val_df = _load_dataframe(val_path)
        X_val = val_df.drop(columns=[target_col])
        y_val = val_df[target_col]
    else:
        # No validation set – use a small part of training data for early stopping
        from sklearn.model_selection import train_test_split

        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            test_size=0.10,
            stratify=y_train,
            random_state=SEED,
        )

    # ------------------------------------------------------------------- #
    # Compute scale_pos_weight
    # ------------------------------------------------------------------- #
    pos = np.sum(y_train == 1)
    neg = np.sum(y_train == 0)
    scale_pos_weight = neg / pos if pos > 0 else 1.0

    # ------------------------------------------------------------------- #
    # LightGBM parameters (exact as requested)
    # ------------------------------------------------------------------- #
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
        "random_state": SEED,
        "scale_pos_weight": scale_pos_weight,
    }

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)

    # ------------------------------------------------------------------- #
    # Train
    # ------------------------------------------------------------------- #
    evals_result = {}
    booster = lgb.train(
        params,
        lgb_train,
        num_boost_round=420,
        valid_sets=[lgb_train, lgb_val],
        valid_names=["train", "val"],
        early_stopping_rounds=30,
        evals_result=evals_result,
        verbose_eval=False,
    )

    # ------------------------------------------------------------------- #
    # Validation metrics
    # ------------------------------------------------------------------- #
    val_pred = booster.predict(X_val, num_iteration=booster.best_iteration)
    auc_val = roc_auc_score(y_val, val_pred)
    brier_val = brier_score_loss(y_val, val_pred)

    # ------------------------------------------------------------------- #
    # Calibration (sigmoid)
    # ------------------------------------------------------------------- #
    from sklearn.calibration import CalibratedClassifierCV

    calibrator = CalibratedClassifierCV(
        base_estimator=booster, method="sigmoid", cv="prefit"
    )
    calibrator.fit(X_train, y_train)

    # ------------------------------------------------------------------- #
    # Persist calibrated model
    # ------------------------------------------------------------------- #
    model_path = model_dir / "lightgbm_best_calibrated.pkl"
    joblib.dump(calibrator, model_path)

    # ------------------------------------------------------------------- #
    # Profile JSON
    # ------------------------------------------------------------------- #
    profile = {
        "hyperparameters": params,
        "best_iteration": booster.best_iteration,
        "validation_auc": auc_val,
        "validation_brier": brier_val,
        "model_size_bytes": model_path.stat().st_size,
        "num_trees": booster.num_trees(),
        "approx_leaf_nodes": booster.num_trees() * params["max_depth"],
        "estimated_flops": booster.num_trees() * params["max_depth"] * 2,
    }
    profile_path = model_dir / "lightgbm_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2))

    # ------------------------------------------------------------------- #
    # MLflow logging (if available)
    # ------------------------------------------------------------------- #
    with mlflow.start_run() if _HAS_MLFLOW else mlflow.start_run():
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("val_auc", auc_val)
        mlflow.log_metric("val_brier", brier_val)
        mlflow.log_metric("best_iteration", booster.best_iteration)
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(profile_path))

    print(f"--> LightGBM training finished. Validation AUC: {auc_val:.4f}")
    return model_path, profile


# --------------------------------------------------------------------------- #
# 6️⃣  PyTorch wrapper exposing the calibrated model
# --------------------------------------------------------------------------- #
import torch
import torch.nn as nn


class LightGBMClassifier(nn.Module):
    """
    Thin PyTorch wrapper around a calibrated LightGBM model.
    The forward method returns the probability of the positive class (label 1).
    """

    def __init__(self, model_path: Path, feature_order_path: Path):
        super().__init__()
        if not model_path.is_file():
            raise FileNotFoundError(f"Calibrated model not found at {model_path}")
        if not feature_order_path.is_file():
            raise FileNotFoundError(f"Feature order file not found at {feature_order_path}")

        import joblib

        self.calibrator = joblib.load(model_path)  # CalibratedClassifierCV instance
        self.feature_order: List[str] = json.loads(feature_order_path.read_text())
        # No learnable parameters – state_dict will be empty
        self._dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, n_features). Must be ordered exactly as `feature_order`.

        Returns
        -------
        torch.Tensor
            Shape (batch, 1) – probability of class 1.
        """
        if x.dim() != 2:
            raise ValueError(f"Input tensor must be 2‑D, got shape {x.shape}")
        if x.size(1) != len(self.feature_order):
            raise ValueError(
                f"Expected {len(self.feature_order)} features, got {x.size(1)}"
            )
        # LightGBM works on CPU NumPy arrays
        np_arr = x.detach().cpu().numpy().astype(np.float32)
        probs = self.calibrator.predict_proba(np_arr)[:, 1]  # prob of class 1
        probs_tensor = torch.from_numpy(probs).unsqueeze(1).to(x.device)
        return probs_tensor

    def state_dict(self, *args, **kwargs):
        """Return an empty state dict – the heavy model lives on disk."""
        return {"_dummy": self._dummy_param}

    def load_state_dict(self, state_dict, strict: bool = True):
        """No‑op – kept for API compatibility."""
        pass


def save_torch_wrapper(wrapper: LightGBMClassifier, out_path: Path) -> None:
    """Save the (empty) state dict of the wrapper."""
    torch.save(wrapper.state_dict(), out_path)


# --------------------------------------------------------------------------- #
# 7️⃣  Gradio demo (simple feature‑wise input)
# --------------------------------------------------------------------------- #
def _load_wrapper() -> LightGBMClassifier:
    model_path = MODEL_DIR / "lightgbm_best_calibrated.pkl"
    feature_order_path = MODEL_DIR / "feature_order.json"
    return LightGBMClassifier(model_path, feature_order_path)


def _gradio_predict(**kwargs):
    """
    Gradio passes each feature as a separate keyword argument.
    The function converts them to a tensor, runs the wrapper and returns
    the probability of the benign class (label 1).
    """
    wrapper = _load_wrapper()
    # Preserve column order
    ordered_vals = [float(kwargs[col]) for col in wrapper.feature_order]
    tensor = torch.tensor([ordered_vals], dtype=torch.float32)
    prob = wrapper(tensor).item()
    return f"{prob:.4f}"


def launch_gradio_demo():
    """Launch a minimal Gradio UI for interactive inference."""
    wrapper = _load_wrapper()
    iface = gr.Interface(
        fn=_gradio_predict,
        inputs=[
            gr.Number(label=col, value=0.0) for col in wrapper.feature_order
        ],
        outputs="text",
        title="Breast Cancer (Wisconsin) – LightGBM‑PyTorch Wrapper",
        description=(
            "Enter the 30 numeric features and obtain the probability "
            "that the tumor is *benign* (class 1)."
        ),
        allow_flagging="never",
    )
    iface.launch(share=False)  # change to True for a public link


# --------------------------------------------------------------------------- #
# 8️⃣  Orchestrator – one‑command entry point
# --------------------------------------------------------------------------- #
def main(use_smote: bool = False, no_mlflow: bool = False) -> None:
    """
    Execute the full pipeline:
        1. Pre‑process & split the data
        2. Train & calibrate LightGBM
        3. Create the PyTorch wrapper and save its (empty) state dict
        4. Launch a Gradio demo (optional – comment out if not needed)
    """
    # ------------------------------------------------------------------- #
    # 8.1 Pre‑process
    # ------------------------------------------------------------------- #
    train_path, val_path, test_path = run_preprocess(use_smote=use_smote)

    # ------------------------------------------------------------------- #
    # 8.2 Train LightGBM
    # ------------------------------------------------------------------- #
    model_path, profile = train_lightgbm(train_path, val_path)

    # ------------------------------------------------------------------- #
    # 8.3 PyTorch wrapper – save empty state dict
    # ------------------------------------------------------------------- #
    wrapper = LightGBMClassifier(
        model_path=model_path,
        feature_order_path=MODEL_DIR / "feature_order.json",
    )
    wrapper_state_path = MODEL_DIR / "lightgbm_torch_wrapper.pt"
    save_torch_wrapper(wrapper, wrapper_state_path)

    # ------------------------------------------------------------------- #
    # 8.4 Summary output (ASCII only)
    # ------------------------------------------------------------------- #
    print("--> Pipeline completed successfully.")
    print(f"    Model artefacts stored in: {MODEL_DIR}")
    print(f"    Validation AUC : {profile['validation_auc']:.4f}")
    print(f"    Model size    : {profile['model_size_bytes'] / 1e6:.2f} MB")
    print(f"    Wrapper state dict saved at: {wrapper_state_path}")

    # ------------------------------------------------------------------- #
    # 8.5 Optional: launch Gradio UI
    # ------------------------------------------------------------------- #
    # Uncomment the line below if you want the demo to start automatically.
    # launch_gradio_demo()


if __name__ == "__main__":
    # Simple CLI parsing – no external dependencies
    import argparse

    parser = argparse.ArgumentParser(
        description="End‑to‑end LightGBM‑PyTorch pipeline for Breast Cancer classification."
    )
    parser.add_argument(
        "--use-smote",
        action="store_true",
        help="Apply SMOTE to the training split (disabled by default).",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Suppress MLflow logging even if the package is installed.",
    )
    args = parser.parse_args()

    # If the user explicitly disables MLflow, replace the dummy with a no‑op
    if args.no_mlflow:
        global _HAS_MLFLOW
        _HAS_MLFLOW = False

    main(use_smote=args.use_smote, no_mlflow=args.no_mlflow)
