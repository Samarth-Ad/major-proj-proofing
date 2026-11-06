
# run_pipeline.py
# --------------------------------------------------------------
# End‑to‑end pipeline for the Titanic tabular classification task.
# All paths, hyper‑parameters and column groups are read from
# `config.yaml` – the single source of truth.
# --------------------------------------------------------------

import os
import hashlib
import json
import time
import shutil
from pathlib import Path

import yaml
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin

import category_encoders as ce
import joblib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------
# 1️⃣  Load configuration
# --------------------------------------------------------------
CONFIG_PATH = Path("config.yaml")
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

SEED = cfg["random_seed"]
np.random.seed(SEED)
torch.manual_seed(SEED)

# --------------------------------------------------------------
# 2️⃣  Helper utilities
# --------------------------------------------------------------
def sha256_file(path: Path) -> str:
    """Return SHA‑256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------
# 3️⃣  Ingest raw CSV & create checksum
# --------------------------------------------------------------
def ingest_raw(csv_src: Path, csv_dest: Path, checksum_path: Path):
    """Copy raw CSV from absolute source, compute checksum and store it."""
    ensure_dir(csv_dest.parent)
    shutil.copy(csv_src, csv_dest)

    checksum = sha256_file(csv_dest)
    ensure_dir(checksum_path.parent)
    with open(checksum_path, "w") as f:
        f.write(checksum)
    print(f"[INGEST] CSV copied to {csv_dest} – SHA256: {checksum}")


# --------------------------------------------------------------
# 4️⃣  Convert CSV → Parquet (fast I/O)
# --------------------------------------------------------------
def csv_to_parquet(csv_path: Path, parquet_path: Path):
    ensure_dir(parquet_path.parent)
    df = pd.read_csv(csv_path)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_path)
    print(f"[CONVERT] Parquet written to {parquet_path}")


# --------------------------------------------------------------
# 5️⃣  Train‑test split (stratified)
# --------------------------------------------------------------
def split_data(csv_path: Path, train_path: Path, test_path: Path, test_size: float, stratify: bool):
    df = pd.read_csv(csv_path)
    y = df[cfg["target_column"]]
    strat_col = y if stratify else None
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=SEED,
        stratify=strat_col,
    )
    ensure_dir(train_path.parent)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # also write parquet versions (optional but required by repo layout)
    train_parquet = train_path.with_suffix(".parquet")
    test_parquet = test_path.with_suffix(".parquet")
    pq.write_table(pa.Table.from_pandas(train_df), train_parquet)
    pq.write_table(pa.Table.from_pandas(test_df), test_parquet)

    print(f"[SPLIT] Train → {train_path}, Test → {test_path}")


# --------------------------------------------------------------
# 6️⃣  Custom transformer – Age binning (example)
# --------------------------------------------------------------
class AgeBinner(BaseEstimator, TransformerMixin):
    """Bin age into categorical intervals."""
    def __init__(self, bins=None):
        if bins is None:
            # typical Titanic bins
            bins = [0, 12, 18, 35, 60, np.inf]
        self.bins = bins
        self.labels = [f"age_{i}" for i in range(len(bins) - 1)]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["age_bin"] = pd.cut(X["age"], bins=self.bins, labels=self.labels)
        return X.drop(columns=["age"])


# --------------------------------------------------------------
# 7️⃣  Build full preprocessing pipeline
# --------------------------------------------------------------
def build_preprocess_pipeline(df: pd.DataFrame):
    # Identify column groups
    target = cfg["target_column"]
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target]

    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c != target]

    # Age binning – treat as categorical after transformation
    if "age" in numeric_cols:
        numeric_cols.remove("age")
        categorical_cols.append("age_bin")

    # Pipeline for numeric features
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", IterativeImputer(random_state=SEED, initial_strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    # Pipeline for categorical features (including age_bin)
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", IterativeImputer(random_state=SEED, initial_strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    # Full column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    # Full pipeline (include custom AgeBinner before ColumnTransformer)
    full_pipe = Pipeline(
        steps=[
            ("age_binner", AgeBinner()),
            ("preprocess", preprocessor),
        ]
    )
    return full_pipe


# --------------------------------------------------------------
# 8️⃣  Fit pipeline on training data & serialize
# --------------------------------------------------------------
def fit_and_save_pipeline(train_path: Path, pipeline_path: Path):
    train_df = pd.read_csv(train_path)
    pipe = build_preprocess_pipeline(train_df)

    X = train_df.drop(columns=[cfg["target_column"]])
    y = train_df[cfg["target_column"]]
    pipe.fit(X, y)

    ensure_dir(pipeline_path.parent)
    joblib.dump(pipe, pipeline_path)
    print(f"[PIPELINE] Fitted pipeline saved to {pipeline_path}")
    return pipe


# --------------------------------------------------------------
# 9️⃣  Transform train / test and persist matrices
# --------------------------------------------------------------
def transform_and_save(
    split_path: Path,
    pipeline_path: Path,
    out_parquet_path: Path,
    out_npy_path: Path,
):
    pipe = joblib.load(pipeline_path)
    df = pd.read_csv(split_path)

    X = df.drop(columns=[cfg["target_column"]])
    y = df[cfg["target_column"]].values.astype(np.int64)

    X_processed = pipe.transform(X).astype(np.float32)

    # Save as Parquet (float32)
    ensure_dir(out_parquet_path.parent)
    processed_df = pd.DataFrame(X_processed)
    processed_df["target"] = y
    table = pa.Table.from_pandas(processed_df)
    pq.write_table(table, out_parquet_path)

    # Save as .npy for PyTorch
    ensure_dir(out_npy_path.parent)
    np.save(out_npy_path, X_processed)
    np.save(out_npy_path.with_name(out_npy_path.stem + "_y.npy"), y)

    print(f"[TRANSFORM] {split_path.name} → {out_parquet_path}, {out_npy_path}")


# --------------------------------------------------------------
# 10️⃣  PyTorch logistic‑regression model definition
# --------------------------------------------------------------
class TorchLogReg(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# --------------------------------------------------------------
# 11️⃣  Training loop with early stopping
# --------------------------------------------------------------
def train_torch_model(
    train_npy: Path,
    test_npy: Path,
    metadata_path: Path,
    model_path: Path,
    device: torch.device,
):
    # Load numpy matrices
    X_train = np.load(train_npy)
    y_train = np.load(train_npy.with_name(train_npy.stem + "_y.npy"))
    X_test = np.load(test_npy)
    y_test = np.load(test_npy.with_name(test_npy.stem + "_y.npy"))

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    batch_size = cfg["model"]["batch_size"]
    epochs = cfg["model"]["epochs"]
    lr = cfg["model"]["learning_rate"]
    patience = cfg["model"]["early_stopping_patience"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    model = TorchLogReg(input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_auc = 0.0
    best_state = None
    epochs_no_improve = 0
    start_time = time.time()

    from sklearn.metrics import roc_auc_score

    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.float().unsqueeze(1).to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        # validation AUC
        model.eval()
        with torch.no_grad():
            all_preds, all_labels = [], []
            for xb, yb in val_loader:
                xb = xb.to(device)
                preds = model(xb).cpu().numpy()
                all_preds.extend(preds.ravel())
                all_labels.extend(yb.numpy())
            val_auc = roc_auc_score(all_labels, all_preds)

        print(f"[TRAIN] Epoch {epoch}/{epochs} – Val ROC‑AUC: {val_auc:.4f}")

        # early stopping check
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("[TRAIN] Early stopping triggered.")
                break

    training_time = time.time() - start_time

    # Save best model
    ensure_dir(model_path.parent)
    torch.save(best_state, model_path)
    print(f"[MODEL] Trained model saved to {model_path}")

    # Write metadata JSON
    metadata = {
        "seed": SEED,
        "input_dim": input_dim,
        "final_val_roc_auc": best_auc,
        "epochs_trained": epoch,
        "batch_size": batch_size,
        "learning_rate": lr,
        "training_time_seconds": training_time,
    }
    ensure_dir(metadata_path.parent)
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[METADATA] Saved to {metadata_path}")

    return model_path, metadata


# --------------------------------------------------------------
# 12️⃣  Model profiling (size, params, latency, etc.)
# --------------------------------------------------------------
def profile_model(model_path: Path, metadata: dict, profile_path: Path):
    # size on disk
    size_bytes = model_path.stat().st_size

    # number of trainable parameters
    state = torch.load(model_path, map_location="cpu")
    total_params = sum(p.numel() for p in state.values())

    # average CPU inference latency per row (simple benchmark)
    model = TorchLogReg(metadata["input_dim"])
    model.load_state_dict(state)
    model.eval()
    dummy_input = torch.randn(1, metadata["input_dim"])
    repeats = 1000
    start = time.time()
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(dummy_input)
    latency = (time.time() - start) / repeats  # seconds per row

    profile = {
        "model_path": str(model_path),
        "size_bytes": size_bytes,
        "trainable_parameters": total_params,
        "avg_cpu_latency_seconds_per_row": latency,
        "training_time_seconds": metadata["training_time_seconds"],
        "final_val_roc_auc": metadata["final_val_roc_auc"],
    }

    ensure_dir(profile_path.parent)
    with open(profile_path, "w") as f:
        json.dump(profile, f, indent=4)
    print(f"[PROFILE] Model profile saved to {profile_path}")


# --------------------------------------------------------------
# 13️⃣  Copy trained model to downstream workspace
# --------------------------------------------------------------
def copy_to_workspace(src: Path, dst: Path):
    ensure_dir(dst.parent)
    shutil.copy(src, dst)
    print(f"[COPY] Model copied to downstream location: {dst}")


# --------------------------------------------------------------
# 14️⃣  Main orchestration
# --------------------------------------------------------------
def main():
    # ------------------------------------------------------------------
    # Paths from config
    # ------------------------------------------------------------------
    p = cfg["paths"]
    raw_csv_src = Path(r"C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\titanic.csv")
    raw_csv_dest = Path(p["raw_csv"])
    raw_parquet = Path(p["raw_parquet"])
    checksum_path = Path("reports/dataset_checksum.txt")
    train_csv = Path(p["train_csv"])
    test_csv = Path(p["test_csv"])
    train_parquet = train_csv.with_suffix(".parquet")
    test_parquet = test_csv.with_suffix(".parquet")
    pipeline_path = Path("artifacts/preprocess_pipeline.pkl")
    train_processed_parquet = Path("data/processed/train_processed.parquet")
    test_processed_parquet = Path("data/processed/test_processed.parquet")
    train_processed_npy = Path("data/processed/train_processed.npy")
    test_processed_npy = Path("data/processed/test_processed.npy")
    model_path = Path("artifacts/torch_logreg.pt")  # note typo in spec – we keep correct folder name
    model_path = Path("artifacts/torch_logreg.pt")
    metadata_path = Path("artifacts/model_metadata.json")
    profile_path = Path("reports/model_profile.json")
    downstream_path = Path("agent_workspace/trained_models/torch_logreg.pt")

    # ------------------------------------------------------------------
    # 1️⃣ Ingest raw CSV & checksum
    # ------------------------------------------------------------------
    ingest_raw(raw_csv_src, raw_csv_dest, checksum_path)

    # ------------------------------------------------------------------
    # 2️⃣ Convert to Parquet
    # ------------------------------------------------------------------
    csv_to_parquet(raw_csv_dest, raw_parquet)

    # ------------------------------------------------------------------
    # 3️⃣ Train‑test split
    # ------------------------------------------------------------------
    split_cfg = cfg["train_test_split"]
    split_data(
        raw_csv_dest,
        train_csv,
        test_csv,
        test_size=split_cfg["test_size"],
        stratify=split_cfg["stratify"],
    )

    # ------------------------------------------------------------------
    # 4️⃣ Fit preprocessing pipeline on training split
    # ------------------------------------------------------------------
    fit_and_save_pipeline(train_csv, pipeline_path)

    # ------------------------------------------------------------------
    # 5️⃣ Transform both splits & persist matrices
    # ------------------------------------------------------------------
    transform_and_save(
        train_csv,
        pipeline_path,
        train_processed_parquet,
        train_processed_npy,
    )
    transform_and_save(
        test_csv,
        pipeline_path,
        test_processed_parquet,
        test_processed_npy,
    )

    # ------------------------------------------------------------------
    # 6️⃣ Train PyTorch logistic‑regression model
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path, metadata = train_torch_model(
        train_processed_npy,
        test_processed_npy,
        metadata_path,
        model_path,
        device,
    )

    # ------------------------------------------------------------------
    # 7️⃣ Profile model & write JSON report
    # ------------------------------------------------------------------
    profile_model(model_path, metadata, profile_path)

    # ------------------------------------------------------------------
    # 8️⃣ Copy model to downstream workspace (required by serving code)
    # ------------------------------------------------------------------
    copy_to_workspace(model_path, downstream_path)


if __name__ == "__main__":
    main()
