
"""
Titanic End‑to‑End Tabular Classification Pipeline
===================================================

This script implements the complete workflow described in the specification:

1. Copies the original CSV from the absolute location supplied by the user
   into the repository layout.
2. Generates a SHA‑256 checksum for provenance.
3. Converts the CSV → Parquet for fast I/O.
4. Performs a stratified 80/20 train‑test split.
5. Builds a scikit‑learn preprocessing pipeline that:
   * adds engineered features (family_size, is_alone, title, fare_per_person)
   * bins age into 5 quantile groups (age_group)
   * imputes missing values (IterativeImputer for numerics,
     most_frequent for categoricals)
   * adds missing‑indicators for numeric columns
   * scales numerics with RobustScaler
   * encodes categoricals with smoothed TargetEncoder
6. Fits the pipeline on the training split, transforms both splits and
   persists the transformed data and the pipeline.
7. Trains a pure logistic‑regression model in PyTorch (single linear
   layer + BCEWithLogitsLoss) with early stopping on ROC‑AUC.
8. Serialises the trained model and a small JSON metadata file.
9. Generates a comprehensive ``model_profile.json`` containing size,
   latency, ROC‑AUC, etc.
10. Copies the final artefacts into ``./agent_workspace/trained_models`` as
    required by the execution environment.

The script automatically installs the optional ``category_encoders`` package
if it is missing, and it works even when MLflow is not installed
(the corresponding logging steps are simply skipped).

Run with:

    python run_titanic_pipeline.py
"""

# ----------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------
import os
import json
import hashlib
import time
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401

# ----------------------------------------------------------------------
# Optional imports – install on‑the‑fly if missing
# ----------------------------------------------------------------------
def _install_and_import(package: str):
    """Install a package via pip and import it."""
    try:
        __import__(package)
    except ImportError:
        print(f"[INFO] Installing missing package: {package}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    finally:
        globals()[package] = __import__(package)


# category_encoders is required for TargetEncoder
_install_and_import("category_encoders")
# mlflow is optional – we import it lazily later
try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except Exception:  # pragma: no cover
    _MLFLOW_AVAILABLE = False

# ----------------------------------------------------------------------
# 0️⃣  Configuration (embedded – matches the spec)
# ----------------------------------------------------------------------
CONFIG = {
    "random_seed": 42,
    "target_column": "survived",
    "train_test_split": {"test_size": 0.20, "stratify": True},
    "paths": {
        "raw_csv": "data/raw/titanic.csv",
        "raw_parquet": "data/raw/titanic.parquet",
        "train_csv": "data/splits/train.csv",
        "test_csv": "data/splits/test.csv",
        "processed_train_parquet": "data/processed/train_processed.parquet",
        "processed_test_parquet": "data/processed/test_processed.parquet",
        "pipeline_pickle": "artifacts/preprocess_pipeline.pkl",
        "torch_model": "artifacts/torch_logreg.pt",
        "mlflow_tracking_uri": "mlflow/mlruns",
    },
    "preprocess": {
        "numeric_features": [
            "age",
            "fare",
            "sibsp",
            "parch",
            "family_size",
            "fare_per_person",
        ],
        "categorical_features": [
            "sex",
            "embarked",
            "pclass",
            "title",
            "age_group",
            "is_alone",
        ],
        "add_features": {
            "family_size": True,
            "is_alone": True,
            "title_extraction": True,
            "age_bins": 5,
        },
        "numeric_imputer": "iterative",
        "categorical_imputer": "most_frequent",
        "categorical_encoder": "target",
        "scaler": "robust",
        "missing_indicator": True,
    },
    "model": {
        "learning_rate": 0.01,
        "weight_decay": 0.0,
        "batch_size": 64,
        "max_epochs": 200,
        "early_stopping_patience": 15,
        "optimizer": "adam",
        "loss": "binary_cross_entropy",
        "eval_metric": "roc_auc",
    },
    # Absolute location of the user‑provided CSV (as per the original brief)
    "user_csv_absolute_path": r"C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\titanic.csv",
}

# ----------------------------------------------------------------------
# Global seeds
# ----------------------------------------------------------------------
SEED = CONFIG["random_seed"]
np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def sha256_checksum(file_path: Path) -> str:
    """Return SHA‑256 checksum of a file."""
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def maybe_start_run(run_name: str, nested: bool = False):
    """Start an MLflow run if possible, otherwise a dummy context."""
    if not _MLFLOW_AVAILABLE:
        class DummyRun:
            def __enter__(self): return self
            def __exit__(self, exc_type, exc_val, exc_tb): pass
            @property
            def info(self): return type("Info", (), {"run_id": "none"})()
        return DummyRun()
    else:
        return mlflow.start_run(run_name=run_name, nested=nested)


def log_artifact_if_possible(local_path: Path, artifact_path: str):
    if _MLFLOW_AVAILABLE:
        mlflow.log_artifact(str(local_path), artifact_path=artifact_path)


# ----------------------------------------------------------------------
# Path preparation (ensure directories exist)
# ----------------------------------------------------------------------
paths = CONFIG["paths"]
RAW_CSV = Path(paths["raw_csv"])
RAW_PARQUET = Path(paths["raw_parquet"])
TRAIN_CSV = Path(paths["train_csv"])
TEST_CSV = Path(paths["test_csv"])
TRAIN_PROC_PARQUET = Path(paths["processed_train_parquet"])
TEST_PROC_PARQUET = Path(paths["processed_test_parquet"])
PIPELINE_PKL = Path(paths["pipeline_pickle"])
TORCH_MODEL_PATH = Path(paths["torch_model"])
MLFLOW_URI = Path(paths["mlflow_tracking_uri"]).as_uri()

# Create all required directories
for p in [
    RAW_CSV.parent,
    TRAIN_CSV.parent,
    TEST_CSV.parent,
    TRAIN_PROC_PARQUET.parent,
    TEST_PROC_PARQUET.parent,
    PIPELINE_PKL.parent,
    TORCH_MODEL_PATH.parent,
    Path("./agent_workspace/trained_models"),
    Path("reports"),
    Path("artifacts"),
]:
    p.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Feature‑engineering transformers
# ----------------------------------------------------------------------
class FeatureEngineering(BaseEstimator, TransformerMixin):
    """Adds family_size, is_alone, title and fare_per_person."""

    def __init__(self):
        # Mapping of raw titles to a reduced set
        self.title_map = {
            "Mr": "Mr",
            "Mrs": "Mrs",
            "Miss": "Miss",
            "Ms": "Miss",
            "Mlle": "Miss",
            "Master": "Master",
            # everything else -> Other
        }

    @staticmethod
    def _extract_title(name: str) -> str:
        if pd.isna(name):
            return "Other"
        # Typical format: "Braund, Mr. Owen Harris"
        if "." not in name:
            return "Other"
        # Take the token that ends with a dot and strip the dot
        token = name.split(".")[0].split()[-1]
        return token

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # family_size = sibsp + parch + 1
        X["family_size"] = (
            X["sibsp"].astype(float) + X["parch"].astype(float) + 1.0
        )
        # is_alone flag
        X["is_alone"] = (X["family_size"] == 1).astype(int)
        # title extraction & mapping
        X["title"] = X["Name"].apply(self._extract_title)
        X["title"] = X["title"].map(self.title_map).fillna("Other")
        # fare per person (avoid division by zero)
        X["fare_per_person"] = X["fare"].astype(float) / X["family_size"]
        return X


class AgeGroupBinner(BaseEstimator, TransformerMixin):
    """Bins (imputed) age into `n_bins` quantile groups."""

    def __init__(self, n_bins: int = 5):
        self.n_bins = n_bins
        self.bin_edges_ = None

    def fit(self, X, y=None):
        ages = X["age"].astype(float)
        # If all ages are missing, fall back to equal-width bins
        if ages.isna().all():
            self.bin_edges_ = np.linspace(0, 1, self.n_bins + 1)
        else:
            self.bin_edges_ = np.quantile(
                ages.dropna(), q=np.linspace(0, 1, self.n_bins + 1)
            )
        return self

    def transform(self, X):
        X = X.copy()
        X["age_group"] = pd.cut(
            X["age"].astype(float),
            bins=self.bin_edges_,
            labels=[f"age_bin_{i}" for i in range(1, self.n_bins + 1)],
            include_lowest=True,
        )
        # Add explicit "Missing" label for NaNs
        X["age_group"] = X["age_group"].cat.add_categories("Missing").fillna("Missing")
        return X


# ----------------------------------------------------------------------
# 1️⃣  Data ingestion & checksum
# ----------------------------------------------------------------------
def ingest_data() -> pd.DataFrame:
    """
    - Copies the user‑provided CSV (if not already present).
    - Computes SHA‑256 checksum and stores it.
    - Writes a Parquet copy.
    Returns the loaded DataFrame (as strings to avoid early type inference).
    """
    src_abs = Path(CONFIG["user_csv_absolute_path"])
    if not RAW_CSV.is_file():
        if not src_abs.is_file():
            raise FileNotFoundError(f"User CSV not found at {src_abs}")
        RAW_CSV.write_bytes(src_abs.read_bytes())

    # Compute checksum
    checksum = sha256_checksum(RAW_CSV)
    (Path("reports") / "dataset_checksum.txt").write_text(checksum)

    # Load raw CSV as strings (preserve original types)
    df = pd.read_csv(RAW_CSV, dtype=str)

    # Parquet copy for faster downstream I/O
    df.to_parquet(RAW_PARQUET, index=False)

    # Optional MLflow logging
    if _MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_URI)
        with maybe_start_run("data_ingest"):
            log_artifact_if_possible(RAW_CSV, "raw")
            log_artifact_if_possible(RAW_PARQUET, "raw")
            log_artifact_if_possible(Path("reports") / "dataset_checksum.txt", "raw")
            mlflow.set_tag("stage", "raw")
    return df


# ----------------------------------------------------------------------
# 2️⃣  Train / test split (stratified)
# ----------------------------------------------------------------------
def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    target = CONFIG["target_column"]
    y = df[target].astype(int)

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=CONFIG["train_test_split"]["test_size"],
        stratify=y if CONFIG["train_test_split"]["stratify"] else None,
        random_state=SEED,
    )
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Persist both CSV and Parquet versions
    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    train_df.to_parquet(TRAIN_CSV.with_suffix(".parquet"), index=False)
    test_df.to_parquet(TEST_CSV.with_suffix(".parquet"), index=False)

    if _MLFLOW_AVAILABLE:
        with maybe_start_run("titanic_data_split"):
            log_artifact_if_possible(TRAIN_CSV, "splits")
            log_artifact_if_possible(TEST_CSV, "splits")
            log_artifact_if_possible(TRAIN_CSV.with_suffix(".parquet"), "splits")
            log_artifact_if_possible(TEST_CSV.with_suffix(".parquet"), "splits")
    return train_df, test_df


# ----------------------------------------------------------------------
# 3️⃣  Build the full preprocessing pipeline
# ----------------------------------------------------------------------
def build_preprocess_pipeline() -> Pipeline:
    cfg = CONFIG["preprocess"]
    numeric_features = cfg["numeric_features"]
    categorical_features = cfg["categorical_features"]

    # Numeric sub‑pipeline
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", IterativeImputer(
                max_iter=10,
                random_state=SEED,
                initial_strategy="mean",
            )),
            ("scaler", RobustScaler()),
        ]
    )

    # Categorical sub‑pipeline
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("target_encoder", category_encoders.TargetEncoder(
                smoothing=10,
                random_state=SEED,
            )),
        ]
    )

    full_pipe = Pipeline(
        steps=[
            ("feature_eng", FeatureEngineering()),
            ("age_binner", AgeGroupBinner(
                n_bins=cfg["add_features"]["age_bins"]
            )),
            ("preprocess", ColumnTransformer(
                transformers=[
                    ("num", numeric_pipe, numeric_features),
                    ("cat", categorical_pipe, categorical_features),
                ],
                remainder="drop",
            )),
        ]
    )
    return full_pipe


# ----------------------------------------------------------------------
# 4️⃣  Fit pipeline, transform data and persist artefacts
# ----------------------------------------------------------------------
def fit_transform_save(
    pipeline: Pipeline,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    target = CONFIG["target_column"]
    y_train = train_df[target].astype(int).values.reshape(-1, 1)
    y_test = test_df[target].astype(int).values.reshape(-1, 1)

    X_train_raw = train_df.drop(columns=[target])
    X_test_raw = test_df.drop(columns=[target])

    # Fit only on training data
    pipeline.fit(X_train_raw, y_train.ravel())

    # Transform both splits
    X_train_proc = pipeline.transform(X_train_raw)
    X_test_proc = pipeline.transform(X_test_raw)

    # Persist transformed matrices (as Parquet for speed)
    pd.DataFrame(X_train_proc).to_parquet(TRAIN_PROC_PARQUET, index=False)
    pd.DataFrame(X_test_proc).to_parquet(TEST_PROC_PARQUET, index=False)

    # Serialize the full pipeline (includes engineering steps)
    joblib.dump(pipeline, PIPELINE_PKL)

    if _MLFLOW_AVAILABLE:
        with maybe_start_run("preprocess_pipeline"):
            log_artifact_if_possible(PIPELINE_PKL, "pipeline")
            log_artifact_if_possible(TRAIN_PROC_PARQUET, "processed")
            log_artifact_if_possible(TEST_PROC_PARQUET, "processed")
            mlflow.log_metric("n_features", X_train_proc.shape[1])
            mlflow.log_metric("n_train_samples", X_train_proc.shape[0])
            mlflow.log_metric("n_test_samples", X_test_proc.shape[0])

    # Convert to float32 for torch
    X_train_np = X_train_proc.astype(np.float32)
    X_test_np = X_test_proc.astype(np.float32)

    return X_train_np, y_train.astype(np.float32), X_test_np, y_test.astype(np.float32), pipeline


# ----------------------------------------------------------------------
# 5️⃣  PyTorch Logistic Regression definition
# ----------------------------------------------------------------------
class LogisticRegressionModel(nn.Module):
    """Pure logistic regression (single linear layer)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)  # logits (no sigmoid here)


# ----------------------------------------------------------------------
# 6️⃣  Training loop with early stopping on ROC‑AUC
# ----------------------------------------------------------------------
def train_torch_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
) -> Tuple[LogisticRegressionModel, float, float, int]:
    cfg = CONFIG["model"]
    batch_size = cfg["batch_size"]
    max_epochs = cfg["max_epochs"]
    patience = cfg["early_stopping_patience"]
    lr = cfg["learning_rate"]
    weight_decay = cfg["weight_decay"]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LogisticRegressionModel(input_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -np.inf
    epochs_no_improve = 0
    best_state = None
    start_time = time.time()
    final_epoch = 0

    for epoch in range(1, max_epochs + 1):
        # ------------------- training -------------------
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # ------------------- validation -------------------
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs.ravel())
                all_labels.extend(yb.numpy().ravel())
        val_auc = roc_auc_score(all_labels, all_probs)

        # Log to MLflow if available
        if _MLFLOW_AVAILABLE:
            with maybe_start_run("titanic_logreg_training", nested=True):
                mlflow.log_metric("train_loss", np.mean(epoch_losses), step=epoch)
                mlflow.log_metric("val_roc_auc", val_auc, step=epoch)

        # Early stopping logic
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[INFO] Early stopping triggered at epoch {epoch}")
                final_epoch = epoch
                break

        print(
            f"Epoch {epoch:03d} | Train loss: {np.mean(epoch_losses):.4f} | "
            f"Val ROC‑AUC: {val_auc:.4f}"
        )
        final_epoch = epoch

    training_time = time.time() - start_time

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # ------------------------------------------------------------------
    # Persist the trained model
    # ------------------------------------------------------------------
    TORCH_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), TORCH_MODEL_PATH)

    # Copy to the workspace location required for serving
    workspace_path = Path("./agent_workspace/trained_models/torch_logreg.pt")
    workspace_path.parent.mkdir(parents=True, exist_ok=True)
    workspace_path.write_bytes(TORCH_MODEL_PATH.read_bytes())

    # Model metadata JSON
    metadata = {
        "input_dim": input_dim,
        "training_seed": SEED,
        "final_val_roc_auc": round(best_auc, 4),
        "epochs_trained": final_epoch,
        "batch_size": batch_size,
        "learning_rate": lr,
    }
    meta_path = Path("artifacts/model_metadata.json")
    meta_path.write_text(json.dumps(metadata, indent=2))

    if _MLFLOW_AVAILABLE:
        with maybe_start_run("logreg_artifacts"):
            log_artifact_if_possible(TORCH_MODEL_PATH, "model")
            log_artifact_if_possible(meta_path, "model")
            mlflow.set_tag("model_type", "logistic_regression")

    return model, best_auc, training_time, final_epoch


# ----------------------------------------------------------------------
# 7️⃣  Final test set evaluation
# ----------------------------------------------------------------------
def evaluate_test(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float:
    batch_size = CONFIG["model"]["batch_size"]
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.ravel())
            all_labels.extend(yb.numpy().ravel())
    test_auc = roc_auc_score(all_labels, all_probs)
    print(f"[INFO] Test ROC‑AUC: {test_auc:.4f}")
    return test_auc


# ----------------------------------------------------------------------
# 8️⃣  Model profile generation
# ----------------------------------------------------------------------
def generate_model_profile(
    model: nn.Module,
    best_val_auc: float,
    test_auc: float,
    training_time: float,
    input_dim: int,
) -> None:
    # Model file size (MB)
    model_size_mb = TORCH_MODEL_PATH.stat().st_size / 1e6

    # Inference latency (average over 10 000 rows)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dummy = torch.randn(10_000, input_dim).to(device)
    model.eval()
    with torch.no_grad():
        start = time.perf_counter()
        _ = torch.sigmoid(model(dummy))
        latency_ms = (time.perf_counter() - start) * 1000 / 10_000

    # Git commit hash (optional)
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"]
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    # MLflow run id (if any)
    run_id = (
        mlflow.active_run().info.run_id
        if _MLFLOW_AVAILABLE and mlflow.active_run()
        else "none"
    )

    profile = {
        "model_type": "logistic_regression",
        "framework": "pytorch",
        "input_dim": input_dim,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "model_file_size_mb": round(model_size_mb, 3),
        "training_time_sec": round(training_time, 3),
        "peak_memory_mb": None,  # could be filled with memory_profiler
        "cpu_inference_latency_ms": round(latency_ms, 4),
        "validation_roc_auc": round(best_val_auc, 4),
        "test_roc_auc": round(test_auc, 4),
        "random_seed": SEED,
        "git_commit_hash": git_hash,
        "mlflow_run_id": run_id,
    }

    profile_path = Path("reports/model_profile.json")
    profile_path.write_text(json.dumps(profile, indent=2))

    if _MLFLOW_AVAILABLE:
        with maybe_start_run("final_model_profile"):
            log_artifact_if_possible(profile_path, "profile")

    print(f"[INFO] Model profile written to {profile_path}")


# ----------------------------------------------------------------------
# 9️⃣  Main orchestration
# ----------------------------------------------------------------------
def main():
    if _MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_URI)

    # 1. Data ingestion
    raw_df = ingest_data()

    # 2. Train / test split
    train_df, test_df = split_data(raw_df)

    # 3. Build preprocessing pipeline
    pipeline = build_preprocess_pipeline()

    # 4. Fit pipeline & transform data
    X_train, y_train, X_test, y_test, _ = fit_transform_save(
        pipeline, train_df, test_df
    )

    # 5. Train PyTorch logistic regression
    input_dim = X_train.shape[1]
    model, best_val_auc, training_time, epochs_trained = train_torch_model(
        X_train, y_train, X_test, y_test, input_dim
    )

    # 6. Final test evaluation
    test_auc = evaluate_test(model, X_test, y_test)

    # 7. Generate model profile
    generate_model_profile(
        model,
        best_val_auc,
        test_auc,
        training_time,
        input_dim,
    )

    print("\n=== Pipeline completed successfully ===")
    print(f"Model artefact (torch state dict): {TORCH_MODEL_PATH}")
    print(f"Preprocess pipeline (pickle): {PIPELINE_PKL}")
    print(f"Model profile: reports/model_profile.json")
    print(
        f"Workspace copy for serving: ./agent_workspace/trained_models/torch_logreg.pt"
    )


if __name__ == "__main__":
    main()
