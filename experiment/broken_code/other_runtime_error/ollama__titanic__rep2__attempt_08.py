
"""
Titanic tabular classification – end‑to‑end pipeline
===================================================

This script implements the complete MLOps play‑book described in the
instruction set.  It:

1️⃣  Loads a ``config.yaml`` (if present) or falls back to a built‑in
    default configuration that matches the required repository layout.
2️⃣  Copies the user‑provided CSV into ``data/raw/titanic.csv``.
3️⃣  Writes a SHA‑256 checksum (auditability).
4️⃣  Converts the CSV → Parquet.
5️⃣  Performs a stratified 80/20 train‑test split.
6️⃣  Builds a scikit‑learn preprocessing pipeline that
    * adds engineered features,
    * bins age,
    * imputes numerics with ``IterativeImputer`` (adds missing‑indicators),
    * scales numerics with ``RobustScaler``,
    * imputes categoricals with ``SimpleImputer`` and encodes them with
      ``OrdinalEncoder`` (a lightweight replacement for ``TargetEncoder``).
7️⃣  Fits the pipeline on the training split and persists it.
8️⃣  Transforms train / test, stores the matrices as Parquet (float32) and
    also as ``.npy`` files for the PyTorch DataLoader.
9️⃣  Trains a pure‑PyTorch logistic‑regression model with early‑stopping
    on ROC‑AUC.
🔟  Saves the model state‑dict, copies it to ``./agent_workspace/trained_models``,
    and writes a small JSON metadata file.
1️⃣1️⃣ Generates a model‑profile JSON (size, latency, parameters, final AUC).

All paths and hyper‑parameters are taken from ``config.yaml`` (or the
fallback).  The script can be executed directly from the repository root
after installing the packages listed in ``requirements.txt``.
"""

# ----------------------------------------------------------------------
# 0️⃣  Imports & configuration loading
# ----------------------------------------------------------------------
import json
import os
import shutil
import hashlib
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import joblib
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# scikit‑learn imports (IterativeImputer is experimental)
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ----------------------------------------------------------------------
# 1️⃣  Configuration handling (load or create a default)
# ----------------------------------------------------------------------
def load_config() -> dict:
    """
    Look for ``config.yaml`` in the current working directory or any parent
    (up to 5 levels).  If it cannot be found, a default configuration that
    matches the required repository layout is returned.
    """
    start = Path.cwd()
    for _ in range(5):
        candidate = start / "config.yaml"
        if candidate.is_file():
            with candidate.open("r") as f:
                return yaml.safe_load(f)
        if start.parent == start:
            break
        start = start.parent

    # ------------------------------------------------------------------
    # Default configuration (exactly the one from the play‑book)
    # ------------------------------------------------------------------
    default_cfg = {
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
            "categorical_encoder": "ordinal",   # we use OrdinalEncoder
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
    }
    print("[CONFIG] No config.yaml found – using built‑in default configuration.")
    return default_cfg


CFG = load_config()

# ----------------------------------------------------------------------
# 2️⃣  Helper utilities
# ----------------------------------------------------------------------
def set_random_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sha256_checksum(file_path: Path) -> str:
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------
# 3️⃣  Data ingestion & split
# ----------------------------------------------------------------------
def ingest_user_csv(source_abs_path: str) -> Path:
    """Copy the user CSV into the location defined in the config."""
    src = Path(source_abs_path)
    if not src.is_file():
        raise FileNotFoundError(f"Source CSV not found at {src}")

    dst = Path(CFG["paths"]["raw_csv"])
    ensure_dir(dst.parent)
    shutil.copy(src, dst)
    print(f"[INGEST] Copied {src} → {dst}")
    return dst


def write_checksum(csv_path: Path) -> None:
    checksum = sha256_checksum(csv_path)
    out_file = Path("reports/dataset_checksum.txt")
    ensure_dir(out_file.parent)
    out_file.write_text(checksum)
    print(f"[CHECKSUM] SHA‑256 written to {out_file}")


def csv_to_parquet(csv_path: Path) -> Path:
    df = pd.read_csv(csv_path, dtype=str)  # keep strings for later processing
    parquet_path = Path(CFG["paths"]["raw_parquet"])
    ensure_dir(parquet_path.parent)
    df.to_parquet(parquet_path, index=False)
    print(f"[PARQUET] Saved {parquet_path}")
    return parquet_path


def stratified_split(parquet_path: Path) -> tuple[Path, Path]:
    df = pd.read_parquet(parquet_path)

    target = CFG["target_column"]
    y = df[target].astype(int)

    train_idx, test_idx = train_test_split(
        df.index,
        test_size=CFG["train_test_split"]["test_size"],
        stratify=y if CFG["train_test_split"]["stratify"] else None,
        random_state=CFG["random_seed"],
    )
    train_df = df.loc[train_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)

    train_path = Path(CFG["paths"]["train_csv"])
    test_path = Path(CFG["paths"]["test_csv"])
    ensure_dir(train_path.parent)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # also parquet versions for speed
    train_df.to_parquet(train_path.with_suffix(".parquet"), index=False)
    test_df.to_parquet(test_path.with_suffix(".parquet"), index=False)

    print(f"[SPLIT] train → {train_path}, test → {test_path}")
    return train_path, test_path


# ----------------------------------------------------------------------
# 4️⃣  Feature‑engineering transformers
# ----------------------------------------------------------------------
class FeatureEngineering(BaseEstimator, TransformerMixin):
    """Adds family_size, is_alone, title, fare_per_person."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def fit(self, X, y=None):
        return self

    @staticmethod
    def _extract_title(name: str) -> str:
        if pd.isna(name):
            return "Unknown"
        parts = name.split(",")
        if len(parts) > 1:
            sub = parts[1].split(".")
            if len(sub) > 0:
                return sub[0].strip()
        return "Unknown"

    def transform(self, X):
        X = X.copy()
        if self.cfg["add_features"]["family_size"]:
            X["family_size"] = (
                pd.to_numeric(X["sibsp"], errors="coerce").fillna(0)
                + pd.to_numeric(X["parch"], errors="coerce").fillna(0)
                + 1
            )
        if self.cfg["add_features"]["is_alone"]:
            X["is_alone"] = (X["family_size"] == 1).astype(int)
        if self.cfg["add_features"]["title_extraction"]:
            X["title"] = X["name"].apply(self._extract_title)
        if "fare" in X.columns:
            X["fare_per_person"] = pd.to_numeric(X["fare"], errors="coerce") / X["family_size"]
        return X


class AgeGroupBinner(BaseEstimator, TransformerMixin):
    """Bins age into quantile groups."""

    def __init__(self, n_bins: int = 5):
        self.n_bins = n_bins
        self.bin_edges_ = None

    def fit(self, X, y=None):
        ages = pd.to_numeric(X["age"], errors="coerce")
        self.bin_edges_ = np.quantile(
            ages.dropna(), q=np.linspace(0, 1, self.n_bins + 1)
        )
        return self

    def transform(self, X):
        X = X.copy()
        ages = pd.to_numeric(X["age"], errors="coerce")
        X["age_group"] = pd.cut(
            ages,
            bins=self.bin_edges_,
            labels=[f"g{i}" for i in range(1, self.n_bins + 1)],
            include_lowest=True,
        )
        return X


# ----------------------------------------------------------------------
# 5️⃣  Build the full preprocessing pipeline
# ----------------------------------------------------------------------
def build_preprocess_pipeline() -> Pipeline:
    # Feature‑engineering steps
    fe = FeatureEngineering(cfg=CFG["preprocess"])
    age_binner = AgeGroupBinner(
        n_bins=CFG["preprocess"]["add_features"]["age_bins"]
    )

    numeric_features = CFG["preprocess"]["numeric_features"]
    categorical_features = CFG["preprocess"]["categorical_features"]

    # Numeric sub‑pipeline
    numeric_steps = [
        (
            "imputer",
            IterativeImputer(
                add_indicator=CFG["preprocess"]["missing_indicator"],
                random_state=CFG["random_seed"],
            ),
        ),
        ("scaler", RobustScaler()),
    ]
    numeric_pipe = Pipeline(numeric_steps)

    # Categorical sub‑pipeline (SimpleImputer + OrdinalEncoder)
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0,
    )

    full_pipe = Pipeline(
        steps=[
            ("feature_engineering", fe),
            ("age_binner", age_binner),
            ("preprocess", preprocessor),
        ]
    )
    return full_pipe


def fit_and_save_pipeline(train_path: Path) -> Pipeline:
    pipeline = build_preprocess_pipeline()
    train_df = pd.read_csv(train_path)
    y = train_df[CFG["target_column"]].astype(int).values
    X = train_df.drop(columns=[CFG["target_column"]])

    pipeline.fit(X, y)

    pipeline_path = Path(CFG["paths"]["pipeline_pickle"])
    ensure_dir(pipeline_path.parent)
    joblib.dump(pipeline, pipeline_path)
    print(f"[PIPELINE] Saved to {pipeline_path}")
    return pipeline


def transform_and_save(pipeline: Pipeline, train_path: Path, test_path: Path) -> None:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y_train = train_df[CFG["target_column"]].astype(int).values.reshape(-1, 1)
    y_test = test_df[CFG["target_column"]].astype(int).values.reshape(-1, 1)

    X_train_raw = train_df.drop(columns=[CFG["target_column"]])
    X_test_raw = test_df.drop(columns=[CFG["target_column"]])

    # Fit only on training data (pipeline already fitted, but calling fit_transform
    # is safe and ensures any stateful steps are consistent)
    X_train = pipeline.transform(X_train_raw).astype(np.float32)
    X_test = pipeline.transform(X_test_raw).astype(np.float32)

    # Persist as Parquet (fast I/O for later)
    train_out = Path(CFG["paths"]["processed_train_parquet"])
    test_out = Path(CFG["paths"]["processed_test_parquet"])
    ensure_dir(train_out.parent)

    pd.DataFrame(X_train).to_parquet(train_out, index=False)
    pd.DataFrame(X_test).to_parquet(test_out, index=False)

    # Also persist raw numpy arrays for the PyTorch DataLoader
    np.save("data/processed/X_train.npy", X_train)
    np.save("data/processed/X_test.npy", X_test)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/y_test.npy", y_test)

    print(f"[TRANSFORM] Train shape {X_train.shape}, Test shape {X_test.shape}")


# ----------------------------------------------------------------------
# 6️⃣  PyTorch logistic‑regression model definition
# ----------------------------------------------------------------------
class LogisticRegressionModel(nn.Module):
    """Single linear layer – outputs logits (no sigmoid)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


# ----------------------------------------------------------------------
# 7️⃣  Training loop with early stopping on ROC‑AUC
# ----------------------------------------------------------------------
def train_torch_model() -> dict:
    # Load transformed data
    X_train = np.load("data/processed/X_train.npy")
    X_test = np.load("data/processed/X_test.npy")
    y_train = np.load("data/processed/y_train.npy").ravel()
    y_test = np.load("data/processed/y_test.npy").ravel()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = CFG["model"]["batch_size"]
    max_epochs = CFG["model"]["max_epochs"]
    patience = CFG["model"]["early_stopping_patience"]
    lr = CFG["model"]["learning_rate"]
    weight_decay = CFG["model"]["weight_decay"]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = LogisticRegressionModel(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -np.inf
    epochs_no_improve = 0
    best_state = None
    total_train_time = 0.0
    final_epoch = 0

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        total_train_time += time.time() - epoch_start

        # ---------- validation ----------
        model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy().ravel()
                all_probs.extend(probs)
                all_labels.extend(yb.numpy())
        val_auc = roc_auc_score(all_labels, all_probs)
        print(f"[EPOCH {epoch:03d}] val_roc_auc = {val_auc:.4f}")

        # ---------- early stopping ----------
        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
            best_state = model.state_dict()
            final_epoch = epoch
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"[EARLY STOP] No improvement for {patience} epochs.")
                break

    # Load best weights
    model.load_state_dict(best_state)

    # ---------- save model ----------
    model_path = Path(CFG["paths"]["torch_model"])
    ensure_dir(model_path.parent)
    torch.save(model.state_dict(), model_path)
    print(f"[MODEL] State dict saved to {model_path}")

    # Copy to the location expected by downstream inference
    deploy_dir = Path("./agent_workspace/trained_models")
    ensure_dir(deploy_dir)
    shutil.copy(model_path, deploy_dir / model_path.name)
    print(f"[DEPLOY] Model copied to {deploy_dir}")

    # ---------- metadata ----------
    metadata = {
        "input_dim": X_train.shape[1],
        "final_val_roc_auc": float(best_auc),
        "epochs_trained": final_epoch,
        "batch_size": batch_size,
        "learning_rate": lr,
        "training_seed": CFG["random_seed"],
        "training_time_seconds": total_train_time,
    }
    meta_path = Path("artifacts/model_metadata.json")
    ensure_dir(meta_path.parent)
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"[METADATA] Written to {meta_path}")

    return metadata


# ----------------------------------------------------------------------
# 8️⃣  Model profiling (size, latency, parameters)
# ----------------------------------------------------------------------
def profile_model(metadata: dict) -> None:
    model_path = Path(CFG["paths"]["torch_model"])
    size_bytes = os.path.getsize(model_path)

    input_dim = metadata["input_dim"]
    model = LogisticRegressionModel(input_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Dummy data for latency measurement (10 000 rows)
    dummy_X = np.random.randn(10_000, input_dim).astype(np.float32)
    dummy_tensor = torch.from_numpy(dummy_X)

    start = time.perf_counter()
    with torch.no_grad():
        _ = model(dummy_tensor)
    latency_ms_per_row = ((time.perf_counter() - start) * 1000) / dummy_X.shape[0]

    profile = {
        "model_file_size_bytes": size_bytes,
        "num_trainable_parameters": sum(p.numel() for p in model.parameters()),
        "average_inference_latency_ms_per_row": latency_ms_per_row,
        "final_val_roc_auc": metadata["final_val_roc_auc"],
    }

    profile_path = Path("reports/model_profile.json")
    ensure_dir(profile_path.parent)
    profile_path.write_text(json.dumps(profile, indent=2))
    print(f"[PROFILE] Saved to {profile_path}")


# ----------------------------------------------------------------------
# 9️⃣  Main orchestration
# ----------------------------------------------------------------------
def main():
    # --------------------------------------------------------------
    # Global seed
    # --------------------------------------------------------------
    set_random_seed(CFG["random_seed"])

    # --------------------------------------------------------------
    # 1️⃣  Ingest CSV – replace the path below with the absolute path
    #     to the user‑provided Titanic CSV on the host machine.
    # --------------------------------------------------------------
    USER_CSV_ABS_PATH = (
        r"C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\titanic.csv"
    )
    raw_csv = ingest_user_csv(USER_CSV_ABS_PATH)

    # --------------------------------------------------------------
    # 2️⃣  Write checksum
    # --------------------------------------------------------------
    write_checksum(raw_csv)

    # --------------------------------------------------------------
    # 3️⃣  CSV → Parquet
    # --------------------------------------------------------------
    raw_parquet = csv_to_parquet(raw_csv)

    # --------------------------------------------------------------
    # 4️⃣  Train‑test split
    # --------------------------------------------------------------
    train_csv, test_csv = stratified_split(raw_parquet)

    # --------------------------------------------------------------
    # 5️⃣  Build & fit preprocessing pipeline
    # --------------------------------------------------------------
    pipeline = fit_and_save_pipeline(train_csv)

    # --------------------------------------------------------------
    # 6️⃣  Transform data & persist
    # --------------------------------------------------------------
    transform_and_save(pipeline, train_csv, test_csv)

    # --------------------------------------------------------------
    # 7️⃣  Train PyTorch logistic regression
    # --------------------------------------------------------------
    metadata = train_torch_model()

    # --------------------------------------------------------------
    # 8️⃣  Profile the trained model
    # --------------------------------------------------------------
    profile_model(metadata)


if __name__ == "__main__":
    main()
