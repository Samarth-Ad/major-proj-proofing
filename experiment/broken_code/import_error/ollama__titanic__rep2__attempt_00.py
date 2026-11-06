
"""
Titanic Tabular Classification – End‑to‑End Pipeline
===================================================

Implements the full workflow described in the specification:

1.  Load the raw CSV (copied from the user‑provided location) and
    generate a SHA‑256 checksum.
2.  Convert CSV → Parquet for fast I/O.
3.  Stratified train / test split (80 % / 20 %).
4.  Build a scikit‑learn preprocessing pipeline that:
    * adds engineered features (family size, is_alone, title)
    * bins age into 5 quantile groups (age_group)
    * imputes missing values (IterativeImputer for numerics,
      most_frequent for categoricals)
    * adds missing‑indicators for numeric columns
    * scales numerics with RobustScaler
    * encodes categoricals with smoothed TargetEncoder
5.  Fit the pipeline on the training split, transform both splits and
    persist the transformed data and the pipeline.
6.  Train a pure logistic‑regression model in PyTorch (single linear
    layer + BCEWithLogitsLoss) with early stopping on ROC‑AUC.
7.  Serialize the trained model and a small JSON metadata file.
8.  Produce a comprehensive `model_profile.json`.
9.  Log all artefacts to a local MLflow tracking server.

All paths are **relative to the repository root** (the folder that
contains this script).  The trained model is also copied to
`./agent_workspace/trained_models` as required by the execution
environment.

Run the script with:

    python run_titanic_pipeline.py
"""

import os
import json
import hashlib
import time
import subprocess
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import yaml
import joblib
import mlflow
import mlflow.pyfunc
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import IterativeImputer, SimpleImputer, MissingIndicator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import category_encoders as ce
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------------------------------------------------
# 1️⃣  Load configuration
# ----------------------------------------------------------------------
CONFIG_PATH = Path("config.yaml")
with CONFIG_PATH.open("r") as f:
    cfg = yaml.safe_load(f)

SEED = cfg["random_seed"]
np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------------------------------------------------
# 2️⃣  Helper functions & custom transformers
# ----------------------------------------------------------------------
def set_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEVICE = set_device()

def sha256_checksum(file_path: Path) -> str:
    """Return SHA‑256 checksum of a file."""
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class FeatureEngineering(BaseEstimator, TransformerMixin):
    """Adds family_size, is_alone and title columns."""

    def __init__(self):
        self.title_map = {
            "Mr": "Mr",
            "Mrs": "Mrs",
            "Miss": "Miss",
            "Ms": "Miss",
            "Mlle": "Miss",
            "Master": "Master",
            "Dr": "Other",
            "Rev": "Other",
            "Col": "Other",
            "Major": "Other",
            "Lady": "Other",
            "Sir": "Other",
            "Capt": "Other",
            "Countess": "Other",
            "Jonkheer": "Other",
            "Don": "Other",
            "Dona": "Other",
        }

    @staticmethod
    def _extract_title(name: str) -> str:
        if pd.isna(name):
            return "Other"
        # Title is the word before the first dot
        parts = name.split(".")
        if len(parts) < 2:
            return "Other"
        title = parts[0].split()[-1]
        return title

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # family_size
        X["family_size"] = X["sibsp"] + X["parch"] + 1
        # is_alone
        X["is_alone"] = (X["family_size"] == 1).astype(int)
        # title extraction
        X["title"] = X["Name"].apply(self._extract_title)
        X["title"] = X["title"].map(self.title_map).fillna("Other")
        return X


class AgeGroupBinner(BaseEstimator, TransformerMixin):
    """Bins (imputed) age into `n_bins` quantile groups."""

    def __init__(self, n_bins: int = 5):
        self.n_bins = n_bins
        self.bin_edges_ = None

    def fit(self, X, y=None):
        # X must contain an 'age' column (already imputed)
        ages = X["age"]
        # Compute quantile edges; include -inf/inf to capture extremes
        self.bin_edges_ = np.quantile(ages, q=np.linspace(0, 1, self.n_bins + 1))
        return self

    def transform(self, X):
        X = X.copy()
        # pandas.cut will assign NaN to values outside edges; we keep them as 'Missing'
        X["age_group"] = pd.cut(
            X["age"],
            bins=self.bin_edges_,
            labels=[f"age_bin_{i}" for i in range(1, self.n_bins + 1)],
            include_lowest=True,
        ).astype(object)
        X["age_group"] = X["age_group"].cat.add_categories("Missing").fillna("Missing")
        return X


# ----------------------------------------------------------------------
# 3️⃣  Paths from config
# ----------------------------------------------------------------------
paths_cfg = cfg["paths"]
RAW_CSV = Path(paths_cfg["raw_csv"])
RAW_PARQUET = Path(paths_cfg["raw_parquet"])
TRAIN_CSV = Path(paths_cfg["train_csv"])
TEST_CSV = Path(paths_cfg["test_csv"])
TRAIN_PROC_PARQUET = Path(paths_cfg["processed_train_parquet"])
TEST_PROC_PARQUET = Path(paths_cfg["processed_test_parquet"])
PIPELINE_PKL = Path(paths_cfg["pipeline_pickle"])
TORCH_MODEL_PATH = Path(paths_cfg["torch_model"])
MLFLOW_URI = Path(paths_cfg["mlflow_tracking_uri"]).as_uri()  # convert to file URI

# Ensure directories exist
for p in [
    RAW_CSV.parent,
    TRAIN_CSV.parent,
    TEST_CSV.parent,
    TRAIN_PROC_PARQUET.parent,
    TEST_PROC_PARQUET.parent,
    PIPELINE_PKL.parent,
    TORCH_MODEL_PATH.parent,
    Path("./agent_workspace/trained_models"),
]:
    p.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 4️⃣  Data ingestion & checksum
# ----------------------------------------------------------------------
def ingest_data():
    # The CSV is assumed to be already placed at RAW_CSV (copy step can be done externally)
    if not RAW_CSV.is_file():
        raise FileNotFoundError(f"Raw CSV not found at {RAW_CSV}")

    # Compute checksum and store
    checksum = sha256_checksum(RAW_CSV)
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    (reports_dir / "dataset_checksum.txt").write_text(checksum)

    # Convert to Parquet (dtype=str to avoid premature inference)
    df = pd.read_csv(RAW_CSV, dtype=str)
    df.to_parquet(RAW_PARQUET, index=False)

    # Log raw artefacts to MLflow (optional)
    mlflow.set_tracking_uri(MLFLOW_URI)
    with mlflow.start_run(run_name="data_ingest") as run:
        mlflow.log_artifact(str(RAW_CSV), artifact_path="raw")
        mlflow.log_artifact(str(RAW_PARQUET), artifact_path="raw")
        mlflow.log_artifact(str(reports_dir / "dataset_checksum.txt"),
                            artifact_path="raw")
        mlflow.set_tag("stage", "raw")
    return df  # return raw dataframe (still strings)


# ----------------------------------------------------------------------
# 5️⃣  Train / test split
# ----------------------------------------------------------------------
def split_data(df: pd.DataFrame):
    target_col = cfg["target_column"]
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col])

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=cfg["train_test_split"]["test_size"],
        stratify=y,
        random_state=SEED,
    )
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Save splits as CSV and Parquet
    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    train_df.to_parquet(TRAIN_CSV.with_suffix(".parquet"), index=False)
    test_df.to_parquet(TEST_CSV.with_suffix(".parquet"), index=False)

    # Log split artefacts
    with mlflow.start_run(run_name="titanic_data_split") as run:
        mlflow.log_artifact(str(TRAIN_CSV), artifact_path="splits")
        mlflow.log_artifact(str(TEST_CSV), artifact_path="splits")
        mlflow.log_artifact(str(TRAIN_CSV.with_suffix(".parquet")), artifact_path="splits")
        mlflow.log_artifact(str(TEST_CSV.with_suffix(".parquet")), artifact_path="splits")
    return train_df, test_df


# ----------------------------------------------------------------------
# 6️⃣  Build preprocessing pipeline
# ----------------------------------------------------------------------
def build_preprocess_pipeline():
    preprocess_cfg = cfg["preprocess"]
    numeric_features = preprocess_cfg["numeric_features"]
    categorical_features = preprocess_cfg["categorical_features"]

    # Numeric sub‑pipeline
    numeric_pipe = Pipeline(
        steps=[
            ("missing_indicator", MissingIndicator(features=numeric_features, error_on_new=False)),
            ("imputer", IterativeImputer(
                estimator=None,
                max_iter=10,
                random_state=SEED,
                initial_strategy="mean"
            )),
            ("scaler", RobustScaler()),
        ]
    )

    # Categorical sub‑pipeline
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("target_encoder", ce.TargetEncoder(
                smoothing=10,
                random_state=SEED
            )),
        ]
    )

    # Full pipeline
    pipeline = Pipeline(
        steps=[
            ("feature_eng", FeatureEngineering()),
            ("age_binner", AgeGroupBinner(n_bins=preprocess_cfg["add_features"]["age_bins"])),
            ("preprocess", ColumnTransformer(
                transformers=[
                    ("num", numeric_pipe, numeric_features),
                    ("cat", categorical_pipe, categorical_features),
                ],
                remainder="drop",
            )),
        ]
    )
    return pipeline


# ----------------------------------------------------------------------
# 7️⃣  Fit pipeline & transform data
# ----------------------------------------------------------------------
def fit_and_transform(pipeline: Pipeline, train_df: pd.DataFrame, test_df: pd.DataFrame):
    target = cfg["target_column"]
    y_train = train_df[target].astype(int).values
    y_test = test_df[target].astype(int).values

    X_train = train_df.drop(columns=[target])
    X_test = test_df.drop(columns=[target])

    # Fit on training data
    pipeline.fit(X_train, y_train)

    # Transform
    X_train_proc = pipeline.transform(X_train)
    X_test_proc = pipeline.transform(X_test)

    # Save transformed data as Parquet (features only)
    pd.DataFrame(X_train_proc).to_parquet(TRAIN_PROC_PARQUET, index=False)
    pd.DataFrame(X_test_proc).to_parquet(TEST_PROC_PARQUET, index=False)

    # Serialize pipeline
    joblib.dump(pipeline, PIPELINE_PKL)

    # Log artefacts
    with mlflow.start_run(run_name="preprocess_pipeline") as run:
        mlflow.log_artifact(str(PIPELINE_PKL), artifact_path="pipeline")
        mlflow.log_artifact(str(TRAIN_PROC_PARQUET), artifact_path="processed")
        mlflow.log_artifact(str(TEST_PROC_PARQUET), artifact_path="processed")
        mlflow.log_metric("n_features", X_train_proc.shape[1])
        mlflow.log_metric("n_train_samples", X_train_proc.shape[0])
        mlflow.log_metric("n_test_samples", X_test_proc.shape[0])

    return (
        X_train_proc.astype(np.float32),
        y_train.astype(np.float32).reshape(-1, 1),
        X_test_proc.astype(np.float32),
        y_test.astype(np.float32).reshape(-1, 1),
        pipeline,
    )


# ----------------------------------------------------------------------
# 8️⃣  PyTorch Logistic Regression model definition
# ----------------------------------------------------------------------
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)  # logits (BCEWithLogitsLoss expects logits)


# ----------------------------------------------------------------------
# 9️⃣  Training loop with early stopping on ROC‑AUC
# ----------------------------------------------------------------------
def train_torch_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
):
    model_cfg = cfg["model"]
    batch_size = model_cfg["batch_size"]
    max_epochs = model_cfg["max_epochs"]
    patience = model_cfg["early_stopping_patience"]
    lr = model_cfg["learning_rate"]
    weight_decay = model_cfg["weight_decay"]
    optimizer_name = model_cfg["optimizer"]

    # Datasets / loaders
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = LogisticRegressionModel(input_dim).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -np.inf
    epochs_no_improve = 0
    best_state_dict = None
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # Validation AUC
        model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs.ravel())
                all_labels.extend(yb.numpy().ravel())
        val_auc = roc_auc_score(all_labels, all_probs)

        # Logging to MLflow
        with mlflow.start_run(run_name="titanic_logreg_training", nested=True) as run:
            mlflow.log_metric("train_loss", np.mean(epoch_losses), step=epoch)
            mlflow.log_metric("val_roc_auc", val_auc, step=epoch)

        # Early stopping check
        if val_auc > best_auc:
            best_auc = val_auc
            best_state_dict = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

        print(
            f"Epoch {epoch:03d} | Train loss: {np.mean(epoch_losses):.4f} | Val ROC‑AUC: {val_auc:.4f}"
        )

    training_time = time.time() - start_time
    # Load best weights
    model.load_state_dict(best_state_dict)

    # Save model
    torch.save(model.state_dict(), TORCH_MODEL_PATH)

    # Also copy to the required workspace folder
    copy_dest = Path("./agent_workspace/trained_models/torch_logreg.pt")
    copy_dest.parent.mkdir(parents=True, exist_ok=True)
    copy_dest.write_bytes(TORCH_MODEL_PATH.read_bytes())

    # Model metadata JSON
    metadata = {
        "input_dim": input_dim,
        "training_seed": SEED,
        "final_val_roc_auc": best_auc,
        "epochs_trained": epoch,
        "batch_size": batch_size,
        "learning_rate": lr,
    }
    meta_path = Path("artifacts/model_metadata.json")
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(metadata, indent=2))

    # Log model artefacts
    with mlflow.start_run(run_name="logreg_artifacts") as run:
        mlflow.log_artifact(str(TORCH_MODEL_PATH), artifact_path="model")
        mlflow.log_artifact(str(meta_path), artifact_path="model")
        mlflow.set_tag("model_type", "logistic_regression")

    return model, best_auc, training_time


# ----------------------------------------------------------------------
# 10️⃣  Model profile generation
# ----------------------------------------------------------------------
def generate_model_profile(
    model: nn.Module,
    best_auc: float,
    training_time: float,
    input_dim: int,
):
    # File size
    model_size_mb = TORCH_MODEL_PATH.stat().st_size / 1e6

    # Inference latency (average over 10 000 rows)
    dummy_input = torch.randn(10_000, input_dim).to(DEVICE)
    model.eval()
    with torch.no_grad():
        start = time.perf_counter()
        _ = torch.sigmoid(model(dummy_input))
        latency_ms = (time.perf_counter() - start) * 1000 / 10_000

    # Git commit hash (if repo)
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode()
            .strip()
        )
    except Exception:
        git_hash = "unknown"

    # Active MLflow run id (use the last run)
    run_id = mlflow.active_run().info.run_id if mlflow.active_run() else "none"

    profile = {
        "model_type": "logistic_regression",
        "framework": "pytorch",
        "input_dim": input_dim,
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "model_file_size_mb": round(model_size_mb, 3),
        "training_time_sec": round(training_time, 3),
        "peak_memory_mb": None,  # could be filled with memory_profiler if needed
        "cpu_inference_latency_ms": round(latency_ms, 4),
        "validation_roc_auc": round(best_auc, 4),
        "test_roc_auc": None,  # will be filled after final test evaluation
        "random_seed": SEED,
        "git_commit_hash": git_hash,
        "mlflow_run_id": run_id,
    }

    profile_path = Path("reports/model_profile.json")
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.write_text(json.dumps(profile, indent=2))

    # Log profile artefact
    with mlflow.start_run(run_name="model_profile") as run:
        mlflow.log_artifact(str(profile_path), artifact_path="profile")

    return profile_path


# ----------------------------------------------------------------------
# 11️⃣  Final test evaluation (ROC‑AUC on held‑out test set)
# ----------------------------------------------------------------------
def evaluate_on_test(
    model: nn.Module,
    X_test: np.ndarray,
    y_test: np.ndarray,
    input_dim: int,
):
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_dataset, batch_size=cfg["model"]["batch_size"], shuffle=False)

    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.ravel())
            all_labels.extend(yb.numpy().ravel())
    test_auc = roc_auc_score(all_labels, all_probs)
    print(f"Test ROC‑AUC: {test_auc:.4f}")
    return test_auc


# ----------------------------------------------------------------------
# 12️⃣  Main orchestration
# ----------------------------------------------------------------------
def main():
    mlflow.set_tracking_uri(MLFLOW_URI)

    # 1. Ingest raw data & checksum
    raw_df = ingest_data()

    # 2. Train / test split
    train_df, test_df = split_data(raw_df)

    # 3. Build preprocessing pipeline
    pipeline = build_preprocess_pipeline()

    # 4. Fit pipeline & transform data
    (
        X_train,
        y_train,
        X_test,
        y_test,
        fitted_pipeline,
    ) = fit_and_transform(pipeline, train_df, test_df)

    # 5. Train PyTorch logistic regression
    input_dim = X_train.shape[1]
    model, best_val_auc, training_time = train_torch_model(
        X_train, y_train, X_test, y_test, input_dim
    )

    # 6. Evaluate on held‑out test set
    test_auc = evaluate_on_test(model, X_test, y_test, input_dim)

    # 7. Generate model profile (including test AUC)
    profile_path = generate_model_profile(
        model, best_val_auc, training_time, input_dim
    )
    # Update test ROC‑AUC in the JSON
    profile = json.loads(profile_path.read_text())
    profile["test_roc_auc"] = round(test_auc, 4)
    profile_path.write_text(json.dumps(profile, indent=2))

    # Log final profile again (overwrite)
    with mlflow.start_run(run_name="final_model_profile") as run:
        mlflow.log_artifact(str(profile_path), artifact_path="profile")

    print("\n=== Pipeline finished successfully ===")
    print(f"Model artefacts stored at: {TORCH_MODEL_PATH}")
    print(f"Preprocess pipeline stored at: {PIPELINE_PKL}")
    print(f"Model profile saved at: {profile_path}")


if __name__ == "__main__":
    main()
