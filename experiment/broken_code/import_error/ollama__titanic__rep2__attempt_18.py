
"""
run_pipeline.py

End‑to‑end MLOps pipeline for the Titanic tabular‑classification task.
All artefacts are created exactly as described in the specification and
all paths are taken from a single ``config.yaml`` file located at the
repository root.

The script can be launched from any working directory – it discovers the
repo root by walking upwards until ``config.yaml`` is found (or creates a
default one if it does not exist).

Steps performed
----------------
0.  Load (or create) ``config.yaml``.
1.  Create the required folder tree (idempotent).
2.  Copy the raw CSV from the absolute user‑provided location into
    ``data/raw/titanic.csv``.
3.  Compute a SHA‑256 checksum of the raw CSV and write it to
    ``reports/dataset_checksum.txt``.
4.  Convert the CSV → Parquet (``data/raw/titanic.parquet``).
5.  Stratified 80 %/20 % train‑test split (CSV + Parquet).
6.  Build a scikit‑learn preprocessing pipeline (feature engineering,
    imputation, scaling, ordinal encoding).
7.  Fit the pipeline on the training split and persist it
    (``artifacts/preprocess_pipeline.pkl``).
8.  Transform both splits, store the float‑32 feature matrices as Parquet
    and as NumPy ``.npy`` files (used later by PyTorch).
9.  Train a PyTorch logistic‑regression model with early stopping on ROC‑AUC.
10. Save the best state‑dict to ``artifacts/torch_logreg.pt`` **and** copy it
    to ``./agent_workspace/trained_models/torch_logreg.pt`` (the location
    expected by downstream serving code).
11. Write ``artifacts/model_metadata.json`` and a profiling report
    ``reports/model_profile.json``.
"""

# ----------------------------------------------------------------------
# 0️⃣  Imports & configuration handling
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
# 1️⃣  Locate repository root & load/create config.yaml
# ----------------------------------------------------------------------
def locate_repo_root(start_path: Path) -> Path:
    """
    Walk upwards from ``start_path`` until a directory containing ``config.yaml``
    is found.  If none is found, assume the repository root is three levels up
    from ``start_path`` (the layout used by the spec) and create a default
    ``config.yaml`` there.
    """
    for parent in [start_path] + list(start_path.parents):
        if (parent / "config.yaml").is_file():
            return parent

    # ---- No config found → create a default one at an assumed repo root ----
    assumed_root = start_path.parents[2]  # e.g. .../agent_workspace/exp -> repo root
    default_cfg = """\
random_seed: 42
target_column: survived

train_test_split:
  test_size: 0.20
  stratify: true

paths:
  raw_csv: data/raw/titanic.csv
  raw_parquet: data/raw/titanic.parquet
  train_csv: data/splits/train.csv
  test_csv: data/splits/test.csv
  processed_train_parquet: data/processed/train_processed.parquet
  processed_test_parquet: data/processed/test_processed.parquet
  pipeline_pickle: artifacts/preprocess_pipeline.pkl
  torch_model: artifacts/torch_logreg.pt
  mlflow_tracking_uri: mlflow/mlruns

preprocess:
  numeric_features:
    - age
    - fare
    - sibsp
    - parch
    - family_size
    - fare_per_person
  categorical_features:
    - sex
    - embarked
    - pclass
    - title
    - age_group
    - is_alone
  add_features:
    family_size: true
    is_alone: true
    title_extraction: true
    age_bins: 5
  numeric_imputer: iterative
  categorical_imputer: most_frequent
  categorical_encoder: ordinal
  scaler: robust
  missing_indicator: true

model:
  learning_rate: 0.01
  weight_decay: 0.0
  batch_size: 64
  max_epochs: 200
  early_stopping_patience: 15
  optimizer: adam
  loss: binary_cross_entropy
  eval_metric: roc_auc
"""
    cfg_path = assumed_root / "config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(default_cfg, encoding="utf-8")
    print(f"[CONFIG] Default config.yaml created at {cfg_path}")
    return assumed_root


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = locate_repo_root(SCRIPT_PATH.parent)

CONFIG_PATH = REPO_ROOT / "config.yaml"
with CONFIG_PATH.open("r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

# ----------------------------------------------------------------------
# 2️⃣  Small helper utilities
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
    """Create parent directories for ``path`` if they do not exist."""
    path.parent.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# 3️⃣  Create the exact folder layout (idempotent)
# ----------------------------------------------------------------------
def create_folder_structure() -> None:
    folders = [
        REPO_ROOT / "data" / "raw",
        REPO_ROOT / "data" / "splits",
        REPO_ROOT / "data" / "processed",
        REPO_ROOT / "artifacts",
        REPO_ROOT / "reports",
        REPO_ROOT / "src" / "preprocessing",
        REPO_ROOT / "src" / "model",
        REPO_ROOT / "mlflow",
        REPO_ROOT / "agent_workspace" / "trained_models",
    ]
    for f in folders:
        f.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# 4️⃣  Copy raw CSV from the absolute user path
# ----------------------------------------------------------------------
def copy_raw_csv() -> Path:
    """
    Copies the user‑provided CSV into the repository at the location defined
    in ``config.yaml`` (``data/raw/titanic.csv``) and returns the destination
    Path.
    """
    src = Path(
        r"C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\titanic.csv"
    )
    dst = REPO_ROOT / CFG["paths"]["raw_csv"]
    if not src.is_file():
        raise FileNotFoundError(f"Source CSV not found at {src}")
    ensure_dir(dst)
    shutil.copyfile(src, dst)
    print(f"[INGEST] Copied raw CSV to {dst}")
    return dst


# ----------------------------------------------------------------------
# 5️⃣  Compute checksum & write audit file
# ----------------------------------------------------------------------
def write_checksum(csv_path: Path) -> None:
    checksum = sha256_checksum(csv_path)
    report_path = REPO_ROOT / "reports" / "dataset_checksum.txt"
    ensure_dir(report_path)
    report_path.write_text(checksum, encoding="utf-8")
    print(f"[CHECKSUM] SHA-256 written to {report_path}")


# ----------------------------------------------------------------------
# 6️⃣  Convert CSV → Parquet (fast I/O)
# ----------------------------------------------------------------------
def csv_to_parquet(csv_path: Path) -> Path:
    parquet_path = REPO_ROOT / CFG["paths"]["raw_parquet"]
    df = pd.read_csv(csv_path, dtype=str)  # keep everything as string initially
    ensure_dir(parquet_path)
    df.to_parquet(parquet_path, index=False)
    print(f"[PARQUET] Saved to {parquet_path}")
    return parquet_path


# ----------------------------------------------------------------------
# 7️⃣  Stratified train / test split (80/20)
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

    # optional parquet versions for debugging
    train_df.to_parquet(train_csv.with_suffix(".parquet"), index=False)
    test_df.to_parquet(test_csv.with_suffix(".parquet"), index=False)

    print(f"[SPLIT] train → {train_csv}, test → {test_csv}")


# ----------------------------------------------------------------------
# 8️⃣  Custom feature‑engineering transformers
# ----------------------------------------------------------------------
class FeatureEngineering(BaseEstimator, TransformerMixin):
    """
    Adds engineered columns according to ``add_features`` in the config:
        - family_size = sibsp + parch + 1
        - is_alone   = (family_size == 1).astype(int)
        - title      = extracted from name (Mr, Mrs, Miss, etc.)
        - fare_per_person = fare / family_size
    """

    def __init__(self, add_features: dict):
        self.add_features = add_features

    @staticmethod
    def _extract_title(name: str) -> str:
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
        af = self.add_features

        if af.get("family_size", False):
            X["family_size"] = (
                X["sibsp"].astype(float) + X["parch"].astype(float) + 1.0
            )
        if af.get("is_alone", False):
            if "family_size" not in X.columns:
                X["family_size"] = (
                    X["sibsp"].astype(float) + X["parch"].astype(float) + 1.0
                )
            X["is_alone"] = (X["family_size"] == 1).astype(int)
        if af.get("title_extraction", False):
            X["title"] = X["name"].apply(self._extract_title)
        if af.get("fare_per_person", False):
            if "family_size" not in X.columns:
                X["family_size"] = (
                    X["sibsp"].astype(float) + X["parch"].astype(float) + 1.0
                )
            X["fare_per_person"] = X["fare"].astype(float) / X["family_size"]
        return X


class AgeGroupBinner(BaseEstimator, TransformerMixin):
    """
    Bins the continuous ``age`` column into ``n_bins`` quantile groups.
    The new column name is ``age_group`` (integer labels 0 … n_bins‑1).
    """

    def __init__(self, n_bins: int = 5):
        self.n_bins = n_bins
        self.bin_edges_ = None

    def fit(self, X, y=None):
        ages = X["age"].astype(float)
        _, bin_edges = pd.qcut(
            ages, q=self.n_bins, duplicates="drop", retbins=True
        )
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
# 9️⃣  Build full preprocessing pipeline
# ----------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    # Feature engineering step
    fe = FeatureEngineering(add_features=CFG["preprocess"]["add_features"])

    # Age binning step
    age_binner = AgeGroupBinner(
        n_bins=CFG["preprocess"]["add_features"]["age_bins"]
    )

    # Column groups
    numeric_feats = CFG["preprocess"]["numeric_features"]
    categorical_feats = CFG["preprocess"]["categorical_features"]

    # Numeric sub‑pipeline
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
    numeric_pipe = Pipeline(
        steps=[("imputer", numeric_imputer), ("scaler", RobustScaler())]
    )

    # Categorical sub‑pipeline
    if CFG["preprocess"]["categorical_imputer"] == "most_frequent":
        cat_imputer = SimpleImputer(strategy="most_frequent")
    else:
        cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
    cat_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    )
    categorical_pipe = Pipeline(
        steps=[("imputer", cat_imputer), ("encoder", cat_encoder)]
    )

    # ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_feats),
            ("cat", categorical_pipe, categorical_feats),
        ],
        remainder="drop",
    )

    # Full pipeline
    full_pipe = Pipeline(
        steps=[
            ("feature_eng", fe),
            ("age_binner", age_binner),
            ("preprocess", preprocessor),
        ]
    )
    return full_pipe


# ----------------------------------------------------------------------
# 1️⃣0️⃣  Fit pipeline on training split & persist
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
# 1️⃣1️⃣  Transform splits & store matrices (Parquet + .npy)
# ----------------------------------------------------------------------
def transform_and_save(pipe: Pipeline) -> None:
    for split in ("train", "test"):
        csv_path = REPO_ROOT / CFG["paths"][f"{split}_csv"]
        df = pd.read_csv(csv_path)

        y = df[CFG["target_column"]].astype(int).values
        X = df.drop(columns=[CFG["target_column"]])

        X_processed = pipe.transform(X).astype(np.float32)

        # Parquet (float32)
        parquet_path = REPO_ROOT / CFG["paths"][f"processed_{split}_parquet"]
        ensure_dir(parquet_path)
        proc_df = pd.DataFrame(
            X_processed,
            columns=[f"col_{i}" for i in range(X_processed.shape[1])],
        )
        proc_df.to_parquet(parquet_path, index=False)

        # NumPy files (used by PyTorch)
        np_dir = REPO_ROOT / "data" / "processed"
        np_dir.mkdir(parents=True, exist_ok=True)
        np.save(np_dir / f"X_{split}.npy", X_processed)
        np.save(np_dir / f"y_{split}.npy", y.astype(np.int64))

        print(
            f"[TRANSFORM] {split} → {parquet_path} + X_{split}.npy / y_{split}.npy"
        )


# ----------------------------------------------------------------------
# 1️⃣2️⃣  PyTorch logistic‑regression model definition
# ----------------------------------------------------------------------
class LogisticRegressionModel(torch.nn.Module):
    """Single linear layer – outputs logits (no sigmoid)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)  # logits


# ----------------------------------------------------------------------
# 1️⃣3️⃣  Training loop with early stopping on ROC‑AUC
# ----------------------------------------------------------------------
def train_torch_model() -> dict:
    # Load processed NumPy matrices
    proc_dir = REPO_ROOT / "data" / "processed"
    X_train = np.load(proc_dir / "X_train.npy")
    y_train = np.load(proc_dir / "y_train.npy")
    X_val = np.load(proc_dir / "X_test.npy")   # using test split as validation
    y_val = np.load(proc_dir / "y_test.npy")

    # DataLoaders
    batch_size = CFG["model"]["batch_size"]
    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train).float().unsqueeze(1),
    )
    val_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val).float().unsqueeze(1),
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )

    # Model, optimizer, loss
    input_dim = X_train.shape[1]
    model = LogisticRegressionModel(input_dim).to("cpu")
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CFG["model"]["learning_rate"],
        weight_decay=CFG["model"]["weight_decay"],
    )
    criterion = torch.nn.BCEWithLogitsLoss()

    # Training loop
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

        # Validation ROC‑AUC
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

        # Early‑stopping bookkeeping
        if val_auc > best_auc:
            best_auc = val_auc
            epochs_without_improve = 0
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

    # Load best checkpoint
    model.load_state_dict(best_state)

    # Save artefacts
    torch_path = REPO_ROOT / CFG["paths"]["torch_model"]
    ensure_dir(torch_path)
    torch.save(best_state, torch_path)
    print(f"[MODEL] Torch state‑dict saved to {torch_path}")

    # Copy to downstream location
    downstream_path = REPO_ROOT / "agent_workspace" / "trained_models" / "torch_logreg.pt"
    ensure_dir(downstream_path)
    shutil.copyfile(torch_path, downstream_path)
    print(f"[MODEL] Copied to downstream location {downstream_path}")

    # Write model_metadata.json
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
# 1️⃣4️⃣  Model profiling (size, params, latency, git hash)
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

    # Git commit hash (optional)
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
# 1️⃣5️⃣  Orchestrator – run everything in order
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
    print(f"Model artefacts   : {REPO_ROOT / 'artifacts'}")
    print(
        f"Down‑stream model : {REPO_ROOT / 'agent_workspace' / 'trained_models' / 'torch_logreg.pt'}"
    )
    print(f"Reports            : {REPO_ROOT / 'reports'}")


if __name__ == "__main__":
    main()
