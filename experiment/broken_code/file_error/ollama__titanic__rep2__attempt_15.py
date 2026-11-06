
"""
run_pipeline.py

End‑to‑end MLOps pipeline for the Titanic tabular‑classification task.
All paths, hyper‑parameters and column groups are read from
`config.yaml` (the single source of truth).  The script creates the
required folder layout, copies the raw CSV, computes a SHA‑256 checksum,
converts to Parquet, performs a stratified train‑test split, builds and
fits a scikit‑learn preprocessing pipeline, transforms the data, trains a
PyTorch logistic‑regression model with early stopping, saves artefacts and
generates a small model‑profile report.

The final trained model state‑dict is stored in
`artifacts/torch_logreg.pt` **and** copied to
`./agent_workspace/trained_models/torch_logreg.pt` – the location
expected by downstream serving code.

Run from the repository root:

    python run_pipeline.py
"""

# ----------------------------------------------------------------------
# 0️⃣  Common imports & configuration loading
# ----------------------------------------------------------------------
import json
import hashlib
import os
import shutil
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler

# ----------------------------------------------------------------------
# 1️⃣  Load configuration (must exist)
# ----------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[0]  # repository root
CONFIG_PATH = REPO_ROOT / "config.yaml"

if not CONFIG_PATH.is_file():
    raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

# ----------------------------------------------------------------------
# 2️⃣  Helper utilities
# ----------------------------------------------------------------------
def set_random_seed(seed: int = 42) -> None:
    """Make experiments deterministic (as far as possible)."""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_checksum(file_path: Path) -> str:
    """Return hex SHA‑256 checksum of a file."""
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(path: Path) -> None:
    """Create parent directories for a given path if they do not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# 3️⃣  Step 0 – Create folder tree (idempotent)
# ----------------------------------------------------------------------
def create_folder_structure() -> None:
    """Create the exact folder layout required by the specification."""
    folders = [
        REPO_ROOT / "data" / "raw",
        REPO_ROOT / "data" / "splits",
        REPO_ROOT / "data" / "processed",
        REPO_ROOT / "artifacts",
        REPO_ROOT / "reports",
        REPO_ROOT / "src" / "preprocessing",
        REPO_ROOT / "src" / "model",
        REPO_ROOT / "mlflow",  # optional
        REPO_ROOT / "agent_workspace" / "trained_models",
    ]
    for f in folders:
        f.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# 4️⃣  Step 1 – Copy raw CSV from the absolute user path
# ----------------------------------------------------------------------
def copy_raw_csv() -> Path:
    """
    Copies the user‑provided CSV into the repository at the location
    defined in config.yaml (data/raw/titanic.csv) and returns the Path.
    """
    src = Path(
        r"C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\titanic.csv"
    )
    dst = REPO_ROOT / CFG["paths"]["raw_csv"]
    if not src.is_file():
        raise FileNotFoundError(f"Source CSV not found at {src}")
    shutil.copyfile(src, dst)
    print(f"[INGEST] Copied raw CSV to {dst}")
    return dst


# ----------------------------------------------------------------------
# 5️⃣  Step 2 – Compute checksum & write audit file
# ----------------------------------------------------------------------
def write_checksum(csv_path: Path) -> None:
    checksum = sha256_checksum(csv_path)
    report_path = REPO_ROOT / "reports" / "dataset_checksum.txt"
    ensure_dir(report_path)
    report_path.write_text(checksum, encoding="utf-8")
    print(f"[CHECKSUM] SHA‑256 written to {report_path}")


# ----------------------------------------------------------------------
# 6️⃣  Step 3 – Convert CSV → Parquet (fast I/O)
# ----------------------------------------------------------------------
def csv_to_parquet(csv_path: Path) -> Path:
    parquet_path = REPO_ROOT / CFG["paths"]["raw_parquet"]
    df = pd.read_csv(csv_path, dtype=str)  # keep strings initially
    df.to_parquet(parquet_path, index=False)
    print(f"[PARQUET] Saved to {parquet_path}")
    return parquet_path


# ----------------------------------------------------------------------
# 7️⃣  Step 4 – Stratified train / test split (80/20)
# ----------------------------------------------------------------------
def stratified_split(parquet_path: Path) -> None:
    df = pd.read_parquet(parquet_path)

    y = df[CFG["target_column"]].astype(int)
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=CFG["train_test_split"]["test_size"],
        stratify=y if CFG["train_test_split"]["stratify"] else None,
        random_state=CFG["random_seed"],
    )

    train_df = df.loc[train_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)

    train_csv = REPO_ROOT / CFG["paths"]["train_csv"]
    test_csv = REPO_ROOT / CFG["paths"]["test_csv"]

    ensure_dir(train_csv)
    ensure_dir(test_csv)

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # also write parquet versions for convenience
    train_df.to_parquet(train_csv.with_suffix(".parquet"), index=False)
    test_df.to_parquet(test_csv.with_suffix(".parquet"), index=False)

    print(f"[SPLIT] train → {train_csv}, test → {test_csv}")


# ----------------------------------------------------------------------
# 8️⃣  Step 5 – Custom feature‑engineering transformers
# ----------------------------------------------------------------------
class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Adds:
        - family_size = sibsp + parch + 1
        - is_alone   = (family_size == 1).astype(int)
        - title      = extracted from name (Mr, Mrs, Miss, etc.)
        - fare_per_person = fare / family_size
    """

    def __init__(self, add_features: dict):
        self.add_features = add_features

    @staticmethod
    def _extract_title(name: str) -> str:
        # Very simple heuristic – split by comma and period
        if pd.isna(name):
            return "Unknown"
        parts = name.split(",")
        if len(parts) < 2:
            return "Unknown"
        title_part = parts[1].strip().split(".")[0].strip()
        return title_part

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.add_features.get("family_size", False):
            X["family_size"] = (
                X["sibsp"].astype(float) + X["parch"].astype(float) + 1.0
            )
        if self.add_features.get("is_alone", False):
            if "family_size" not in X.columns:
                # ensure family_size exists
                X["family_size"] = (
                    X["sibsp"].astype(float) + X["parch"].astype(float) + 1.0
                )
            X["is_alone"] = (X["family_size"] == 1).astype(int)
        if self.add_features.get("title_extraction", False):
            X["title"] = X["name"].apply(self._extract_title)
        if self.add_features.get("fare_per_person", False):
            if "family_size" not in X.columns:
                X["family_size"] = (
                    X["sibsp"].astype(float) + X["parch"].astype(float) + 1.0
                )
            X["fare_per_person"] = X["fare"].astype(float) / X["family_size"]
        return X


class AgeGroupBinner(BaseEstimator, TransformerMixin):
    """
    Bins the continuous `age` column into `n_bins` quantile groups.
    The new column name is `age_group`.
    """

    def __init__(self, n_bins: int = 5):
        self.n_bins = n_bins
        self.bin_edges_ = None

    def fit(self, X, y=None):
        ages = X["age"].astype(float)
        # pandas qcut returns bin edges; we store them for transform
        _, bin_edges = pd.qcut(ages, q=self.n_bins, duplicates="drop", retbins=True)
        self.bin_edges_ = bin_edges
        return self

    def transform(self, X):
        X = X.copy()
        ages = X["age"].astype(float)
        X["age_group"] = pd.cut(
            ages,
            bins=self.bin_edges_,
            labels=False,
            include_lowest=True,
        ).astype(int)
        return X


# ----------------------------------------------------------------------
# 9️⃣  Step 6 – Build full preprocessing pipeline
# ----------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    # ------------------------------------------------------------------
    # 9.1  Feature engineering stage
    # ------------------------------------------------------------------
    fe = FeatureEngineering(add_features=CFG["preprocess"]["add_features"])

    # ------------------------------------------------------------------
    # 9.2  Age binning stage (if configured)
    # ------------------------------------------------------------------
    age_binner = AgeGroupBinner(
        n_bins=CFG["preprocess"]["add_features"]["age_bins"]
    )

    # ------------------------------------------------------------------
    # 9.3  Column groups from config
    # ------------------------------------------------------------------
    numeric_feats = CFG["preprocess"]["numeric_features"]
    categorical_feats = CFG["preprocess"]["categorical_features"]

    # ------------------------------------------------------------------
    # 9.4  Numeric sub‑pipeline
    # ------------------------------------------------------------------
    if CFG["preprocess"]["numeric_imputer"] == "iterative":
        numeric_imputer = IterativeImputer(
            random_state=CFG["random_seed"],
            add_indicator=CFG["preprocess"]["missing_indicator"],
        )
    else:
        numeric_imputer = SimpleImputer(
            strategy="median",
            add_indicator=CFG["preprocess"]["missing_indicator"],
        )
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", numeric_imputer),
            ("scaler", RobustScaler()),
        ]
    )

    # ------------------------------------------------------------------
    # 9.5  Categorical sub‑pipeline
    # ------------------------------------------------------------------
    if CFG["preprocess"]["categorical_imputer"] == "most_frequent":
        cat_imputer = SimpleImputer(strategy="most_frequent")
    else:
        cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")

    cat_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", cat_imputer),
            ("encoder", cat_encoder),
        ]
    )

    # ------------------------------------------------------------------
    # 9.6  Assemble ColumnTransformer
    # ------------------------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_feats),
            ("cat", categorical_pipeline, categorical_feats),
        ],
        remainder="drop",  # any unexpected column is dropped
    )

    # ------------------------------------------------------------------
    # 9.7  Full pipeline (feature eng → age binning → preprocessing)
    # ------------------------------------------------------------------
    full_pipe = Pipeline(
        steps=[
            ("feature_eng", fe),
            ("age_binner", age_binner),
            ("preprocess", preprocessor),
        ]
    )
    return full_pipe


# ----------------------------------------------------------------------
# 10️⃣  Step 7 – Fit pipeline on training split & persist
# ----------------------------------------------------------------------
def fit_and_save_pipeline() -> Pipeline:
    train_path = REPO_ROOT / CFG["paths"]["train_csv"]
    train_df = pd.read_csv(train_path)

    y = train_df[CFG["target_column"]].astype(int).values
    X = train_df.drop(columns=[CFG["target_column"]])

    pipe = build_pipeline()
    pipe.fit(X, y)

    pipeline_path = REPO_ROOT / CFG["paths"]["pipeline_pickle"]
    ensure_dir(pipeline_path)
    joblib.dump(pipe, pipeline_path)
    print(f"[PIPELINE] Saved to {pipeline_path}")
    return pipe


# ----------------------------------------------------------------------
# 11️⃣  Step 8 – Transform both splits & store matrices
# ----------------------------------------------------------------------
def transform_and_save(pipe: Pipeline) -> None:
    """
    Transforms train & test splits, stores float32 Parquet matrices
    and also NumPy .npy files for convenient torch DataLoader usage.
    """
    for split in ("train", "test"):
        csv_path = REPO_ROOT / CFG["paths"][f"{split}_csv"]
        df = pd.read_csv(csv_path)

        y = df[CFG["target_column"]].astype(int).values
        X = df.drop(columns=[CFG["target_column"]])

        X_processed = pipe.transform(X).astype(np.float32)

        # 1️⃣  Parquet (float32)
        parquet_path = REPO_ROOT / CFG[
            "processed_" + f"{split}_parquet"
        ]  # e.g. data/processed/train_processed.parquet
        ensure_dir(parquet_path)
        # Convert to DataFrame to keep column names (use generic names)
        proc_df = pd.DataFrame(
            X_processed,
            columns=[
                f"col_{i}" for i in range(X_processed.shape[1])
            ],  # column names are not needed downstream
        )
        proc_df.to_parquet(parquet_path, index=False)

        # 2️⃣  NumPy files (X & y)
        np_dir = REPO_ROOT / "data" / "processed"
        np_dir.mkdir(parents=True, exist_ok=True)
        np.save(np_dir / f"X_{split}.npy", X_processed)
        np.save(np_dir / f"y_{split}.npy", y.astype(np.int64))

        print(f"[TRANSFORM] {split} → {parquet_path} + X_{split}.npy, y_{split}.npy")


# ----------------------------------------------------------------------
# 12️⃣  Step 9 – PyTorch logistic‑regression model definition
# ----------------------------------------------------------------------
class LogisticRegressionModel(torch.nn.Module):
    """Single linear layer (logits output)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)  # logits


# ----------------------------------------------------------------------
# 13️⃣  Step 10 – Training loop with early stopping on ROC‑AUC
# ----------------------------------------------------------------------
def train_torch_model() -> dict:
    """
    Trains the logistic‑regression model, saves the best checkpoint,
    copies it to the downstream folder and writes `model_metadata.json`.

    Returns
    -------
    metadata : dict
        Dictionary with training statistics (saved also to JSON).
    """
    # ------------------------------------------------------------------
    # Load processed NumPy matrices
    # ------------------------------------------------------------------
    proc_dir = REPO_ROOT / "data" / "processed"
    X_train = np.load(proc_dir / "X_train.npy")
    y_train = np.load(proc_dir / "y_train.npy")
    X_val = np.load(proc_dir / "X_test.npy")  # using the *test* split as validation per spec
    y_val = np.load(proc_dir / "y_test.npy")

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    batch_size = CFG["model"]["batch_size"]
    train_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train).float().unsqueeze(1)
    )
    val_dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val).float().unsqueeze(1)
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    # ------------------------------------------------------------------
    # Model, optimizer, loss
    # ------------------------------------------------------------------
    input_dim = X_train.shape[1]
    model = LogisticRegressionModel(input_dim).to("cpu")  # CPU‑only as per requirements
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CFG["model"]["learning_rate"],
        weight_decay=CFG["model"]["weight_decay"],
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    best_auc = -np.inf
    epochs_without_improve = 0
    patience = CFG["model"]["early_stopping_patience"]
    max_epochs = CFG["model"]["max_epochs"]
    start_time = time.time()

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        # ------------------------------------------------------------------
        # Validation ROC‑AUC
        # ------------------------------------------------------------------
        model.eval()
        with torch.no_grad():
            all_logits = []
            all_targets = []
            for xb, yb in val_loader:
                logits = model(xb)
                all_logits.append(logits.squeeze().cpu().numpy())
                all_targets.append(yb.squeeze().cpu().numpy())
            val_probs = np.concatenate(all_logits)
            val_true = np.concatenate(all_targets)
            val_auc = roc_auc_score(val_true, val_probs)

        # ------------------------------------------------------------------
        # Early‑stopping bookkeeping
        # ------------------------------------------------------------------
        if val_auc > best_auc:
            best_auc = val_auc
            epochs_without_improve = 0
            # Save best checkpoint
            best_state = model.state_dict()
        else:
            epochs_without_improve += 1

        print(
            f"[EPOCH {epoch:03d}] loss={np.mean(epoch_losses):.4f}  val_roc_auc={val_auc:.4f}"
        )

        if epochs_without_improve >= patience:
            print(
                f"[EARLY STOP] No improvement for {patience} epochs. Stopping at epoch {epoch}."
            )
            break

    training_time = time.time() - start_time

    # ------------------------------------------------------------------
    # Load best checkpoint back into model
    # ------------------------------------------------------------------
    model.load_state_dict(best_state)

    # ------------------------------------------------------------------
    # Save artefacts
    # ------------------------------------------------------------------
    torch_path = REPO_ROOT / CFG["paths"]["torch_model"]
    ensure_dir(torch_path)
    torch.save(best_state, torch_path)
    print(f"[MODEL] Torch state‑dict saved to {torch_path}")

    # Copy to downstream expected location
    downstream_path = REPO_ROOT / "agent_workspace" / "trained_models" / "torch_logreg.pt"
    ensure_dir(downstream_path)
    shutil.copyfile(torch_path, downstream_path)
    print(f"[MODEL] Copied to downstream location {downstream_path}")

    # ------------------------------------------------------------------
    # Write model_metadata.json
    # ------------------------------------------------------------------
    metadata = {
        "seed": CFG["random_seed"],
        "input_dim": input_dim,
        "final_val_roc_auc": float(best_auc),
        "epochs_trained": epoch,
        "batch_size": batch_size,
        "learning_rate": CFG["model"]["learning_rate"],
        "weight_decay": CFG["model"]["weight_decay"],
        "training_time_seconds": float(training_time),
    }
    meta_path = REPO_ROOT / "artifacts" / "model_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[METADATA] Written to {meta_path}")

    return metadata


# ----------------------------------------------------------------------
# 14️⃣  Step 11 – Model profiling (size, params, latency, git hash)
# ----------------------------------------------------------------------
def generate_model_profile(metadata: dict) -> None:
    torch_path = REPO_ROOT / CFG["paths"]["torch_model"]
    model = LogisticRegressionModel(metadata["input_dim"])
    model.load_state_dict(torch.load(torch_path, map_location="cpu"))
    model.eval()

    # Number of trainable parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # File size in bytes
    file_size = os.path.getsize(torch_path)

    # CPU inference latency (average per row) on dummy batch of 10 000 rows
    dummy = torch.randn(10_000, metadata["input_dim"], dtype=torch.float32)
    start = time.perf_counter()
    with torch.no_grad():
        _ = model(dummy)
    latency_ms_per_row = ((time.perf_counter() - start) * 1000) / 10_000

    # Git commit hash (optional – ignore failures)
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except Exception:
        git_hash = None

    profile = {
        "model_type": "torch_logistic_regression",
        "framework": "pytorch",
        "input_dim": metadata["input_dim"],
        "num_parameters": num_params,
        "model_file_size_bytes": file_size,
        "average_cpu_latency_ms_per_row": latency_ms_per_row,
        "final_val_roc_auc": metadata["final_val_roc_auc"],
        "training_time_seconds": metadata["training_time_seconds"],
        "git_commit_hash": git_hash,
    }

    profile_path = REPO_ROOT / "reports" / "model_profile.json"
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"[PROFILE] Model profile written to {profile_path}")


# ----------------------------------------------------------------------
# 15️⃣  Orchestrator – run everything in order
# ----------------------------------------------------------------------
def main() -> None:
    set_random_seed(CFG["random_seed"])

    # 0️⃣  Folder layout
    create_folder_structure()

    # 1️⃣  Copy raw CSV
    csv_path = copy_raw_csv()

    # 2️⃣  Checksum
    write_checksum(csv_path)

    # 3️⃣  CSV → Parquet
    parquet_path = csv_to_parquet(csv_path)

    # 4️⃣  Train‑test split
    stratified_split(parquet_path)

    # 5️⃣  Fit preprocessing pipeline
    pipeline = fit_and_save_pipeline()

    # 6️⃣  Transform splits & persist matrices
    transform_and_save(pipeline)

    # 7️⃣  Train PyTorch logistic regression
    metadata = train_torch_model()

    # 8️⃣  Generate profiling report
    generate_model_profile(metadata)

    print("\n=== PIPELINE FINISHED SUCCESSFULLY ===")
    print(f"Model artefacts are in {REPO_ROOT / 'artifacts'}")
    print(f"Down‑stream model for serving: {REPO_ROOT / 'agent_workspace' / 'trained_models' / 'torch_logreg.pt'}")
    print(f"Reports are in {REPO_ROOT / 'reports'}")


if __name__ == "__main__":
    main()
