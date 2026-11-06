
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Titanic tabular‑classification end‑to‑end pipeline.

All paths, hyper‑parameters and column groups are defined in ``config.yaml`` – the
only file that non‑technical users may edit.  The script creates the exact
repository layout you specified, installs missing dependencies, and produces
the following artefacts:

* data/raw/titanic.csv & .parquet
* reports/dataset_checksum.txt
* data/splits/train.{csv,parquet} & test.{csv,parquet}
* artifacts/preprocess_pipeline.pkl
* data/processed/*_processed.{parquet,npy}
* artifacts/torch_logreg.pt
* artifacts/model_metadata.json
* reports/model_profile.json
* ./agent_workspace/trained_models/torch_logreg.pt   (down‑stream copy)
"""

# ----------------------------------------------------------------------
# 0️⃣  Imports & on‑the‑fly dependency installation
# ----------------------------------------------------------------------
import sys
import subprocess
import importlib

def _install_and_import(pkg_name: str, import_name: str = None):
    """Install a package with pip if it cannot be imported, then import it."""
    import_name = import_name or pkg_name
    try:
        return importlib.import_module(import_name)
    except ImportError:
        print(f"[INSTALL] Installing missing package: {pkg_name}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
        return importlib.import_module(import_name)

# Core libraries (always present)
import os
import hashlib
import json
import time
import shutil
from pathlib import Path

# Third‑party libraries – installed if absent
yaml = _install_and_import("PyYAML", "yaml")
pd = _install_and_import("pandas")
np = _install_and_import("numpy")
pa = _install_and_import("pyarrow")
pq = _install_and_import("pyarrow.parquet")
joblib = _install_and_import("joblib")
torch = _install_and_import("torch")
sklearn = _install_and_import("scikit-learn")
# sub‑modules from sklearn
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
# optional but requested in requirements
mlflow = _install_and_import("mlflow")
# category_encoders for more sophisticated encoding (not used directly here)
ce = _install_and_import("category_encoders")

# ----------------------------------------------------------------------
# 1️⃣  Load configuration (single source of truth)
# ----------------------------------------------------------------------
CONFIG_PATH = Path("config.yaml")
if not CONFIG_PATH.is_file():
    # If the file does not exist we create it with the exact content you gave.
    CONFIG_PATH.write_text(
        """# ==============================================================
# 0️⃣  Global settings
# ==============================================================
random_seed: 42
target_column: survived

# --------------------------------------------------------------
# 1️⃣  Train‑test split
# --------------------------------------------------------------
train_test_split:
  test_size: 0.20          # 80 % train, 20 % test
  stratify: true

# --------------------------------------------------------------
# 2️⃣  File system locations (relative to repo root)
# --------------------------------------------------------------
paths:
  raw_csv: data/raw/titanic.csv
  raw_parquet: data/raw/titanic.parquet
  train_csv: data/splits/train.csv
  test_csv: data/splits/test.csv
  train_parquet: data/splits/train.parquet
  test_parquet: data/splits/test.parquet
  processed_train_parquet: data/processed/train_processed.parquet
  processed_test_parquet: data/processed/test_processed.parquet
  processed_train_npy: data/processed/train_processed.npy
  processed_test_npy: data/processed/test_processed.npy
  pipeline: artifacts/preprocess_pipeline.pkl
  model_pt: artifacts/torch_logreg.pt
  model_metadata: artifacts/model_metadata.json
  model_profile: reports/model_profile.json
  downstream_model: agent_workspace/trained_models/torch_logreg.pt
  dataset_checksum: reports/dataset_checksum.txt
""",
        encoding="utf-8",
    )
    print("[INFO] Created default config.yaml – edit it if you need different paths/hyper‑params.")

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

SEED = cfg["random_seed"]
np.random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------------------------------------------------------
# 2️⃣  Helper utilities
# ----------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    """Return SHA‑256 checksum of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(p: Path):
    """Create a directory (including parents) if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# 3️⃣  Ingest raw CSV & create checksum
# ----------------------------------------------------------------------
def ingest_raw(csv_src: Path, csv_dest: Path, checksum_path: Path):
    """Copy raw CSV from absolute source, compute checksum and store it."""
    ensure_dir(csv_dest.parent)
    shutil.copy(csv_src, csv_dest)

    checksum = sha256_file(csv_dest)
    ensure_dir(checksum_path.parent)
    checksum_path.write_text(checksum, encoding="utf-8")
    print(f"[INGEST] CSV copied → {csv_dest} | SHA256: {checksum}")


# ----------------------------------------------------------------------
# 4️⃣  Convert CSV → Parquet (fast I/O)
# ----------------------------------------------------------------------
def csv_to_parquet(csv_path: Path, parquet_path: Path):
    ensure_dir(parquet_path.parent)
    df = pd.read_csv(csv_path)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_path)
    print(f"[CONVERT] Parquet written → {parquet_path}")


# ----------------------------------------------------------------------
# 5️⃣  Train‑test split (stratified)
# ----------------------------------------------------------------------
def split_data(csv_path: Path, train_path: Path, test_path: Path,
               train_parquet: Path, test_parquet: Path,
               test_size: float, stratify: bool):
    df = pd.read_csv(csv_path)
    y = df[cfg["target_column"]]
    strat_col = y if stratify else None

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=SEED,
        stratify=strat_col,
    )

    # CSV versions
    ensure_dir(train_path.parent)
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # Parquet versions (optional but required by repo layout)
    ensure_dir(train_parquet.parent)
    pq.write_table(pa.Table.from_pandas(train_df), train_parquet)
    pq.write_table(pa.Table.from_pandas(test_df), test_parquet)

    print(f"[SPLIT] Train → {train_path} | Test → {test_path}")


# ----------------------------------------------------------------------
# 6️⃣  Custom transformer – Age binning (example)
# ----------------------------------------------------------------------
class AgeBinner(BaseEstimator, TransformerMixin):
    """Bin age into categorical intervals and drop the original column."""
    def __init__(self, bins=None):
        if bins is None:
            bins = [0, 12, 18, 35, 60, np.inf]  # typical Titanic bins
        self.bins = bins
        self.labels = [f"age_{i}" for i in range(len(bins) - 1)]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X["age_bin"] = pd.cut(X["age"], bins=self.bins, labels=self.labels)
        return X.drop(columns=["age"])


# ----------------------------------------------------------------------
# 7️⃣  Build full preprocessing pipeline
# ----------------------------------------------------------------------
def build_preprocess_pipeline(df: pd.DataFrame) -> Pipeline:
    target = cfg["target_column"]

    # Numeric columns (exclude target)
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != target]

    # Categorical columns (exclude target)
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c != target]

    # Age binning – treat the generated bin as categorical
    if "age" in numeric_cols:
        numeric_cols.remove("age")
        categorical_cols.append("age_bin")

    # Numeric pipeline
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", IterativeImputer(random_state=SEED, initial_strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    # Categorical pipeline (including age_bin)
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", IterativeImputer(random_state=SEED, initial_strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    # ColumnTransformer that applies the above pipelines
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )

    # Full pipeline – AgeBinner first, then ColumnTransformer
    full_pipe = Pipeline(
        steps=[
            ("age_binner", AgeBinner()),
            ("preprocess", preprocessor),
        ]
    )
    return full_pipe


# ----------------------------------------------------------------------
# 8️⃣  Fit pipeline on training split & serialize
# ----------------------------------------------------------------------
def fit_and_save_pipeline(train_path: Path, pipeline_path: Path) -> Pipeline:
    train_df = pd.read_csv(train_path)
    pipe = build_preprocess_pipeline(train_df)

    X = train_df.drop(columns=[cfg["target_column"]])
    y = train_df[cfg["target_column"]]
    pipe.fit(X, y)

    ensure_dir(pipeline_path.parent)
    joblib.dump(pipe, pipeline_path)
    print(f"[PIPELINE] Fitted pipeline saved → {pipeline_path}")
    return pipe


# ----------------------------------------------------------------------
# 9️⃣  Transform splits & persist matrices (Parquet + .npy)
# ----------------------------------------------------------------------
def transform_and_save(split_path: Path, pipeline_path: Path,
                       out_parquet_path: Path, out_npy_path: Path):
    pipe = joblib.load(pipeline_path)
    df = pd.read_csv(split_path)

    X = df.drop(columns=[cfg["target_column"]])
    y = df[cfg["target_column"]].values.astype(np.int64)

    X_processed = pipe.transform(X).astype(np.float32)

    # Parquet (float32) – we also store the target column for easy inspection
    ensure_dir(out_parquet_path.parent)
    proc_df = pd.DataFrame(X_processed)
    proc_df["target"] = y
    table = pa.Table.from_pandas(proc_df)
    pq.write_table(table, out_parquet_path)

    # .npy files for PyTorch
    ensure_dir(out_npy_path.parent)
    np.save(out_npy_path, X_processed)
    np.save(out_npy_path.with_name(out_npy_path.stem + "_y.npy"), y)

    print(f"[TRANSFORM] {split_path.name} → {out_parquet_path} & {out_npy_path}")


# ----------------------------------------------------------------------
# 🔟  PyTorch logistic‑regression model definition
# ----------------------------------------------------------------------
class TorchLogReg(torch.nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# ----------------------------------------------------------------------
# 1️⃣1️⃣  Training loop with early stopping
# ----------------------------------------------------------------------
def train_torch_model(train_npy: Path, test_npy: Path,
                      metadata_path: Path, model_path: Path,
                      device: torch.device):
    # Load numpy matrices (features + labels)
    X_train = np.load(train_npy)
    y_train = np.load(train_npy.with_name(train_npy.stem + "_y.npy"))
    X_test = np.load(test_npy)
    y_test = np.load(test_npy.with_name(test_npy.stem + "_y.npy"))

    # PyTorch datasets / loaders
    train_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    val_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )

    # Hyper‑parameters – read from config (fallback defaults if missing)
    model_cfg = cfg.get("model", {})
    batch_size = model_cfg.get("batch_size", 64)
    epochs = model_cfg.get("epochs", 100)
    lr = model_cfg.get("learning_rate", 1e-3)
    patience = model_cfg.get("early_stopping_patience", 7)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]
    model = TorchLogReg(input_dim).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_auc = 0.0
    best_state = None
    epochs_no_improve = 0
    start_time = time.time()

    from sklearn.metrics import roc_auc_score

    for epoch in range(1, epochs + 1):
        # ---------- training ----------
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.float().unsqueeze(1).to(device)   # shape (B,1)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

        # ---------- validation ----------
        model.eval()
        with torch.no_grad():
            all_preds, all_labels = [], []
            for xb, yb in val_loader:
                xb = xb.to(device)
                preds = model(xb).cpu().numpy()
                all_preds.extend(preds.ravel())
                all_labels.extend(yb.numpy())
            val_auc = roc_auc_score(all_labels, all_preds)

        print(f"[TRAIN] Epoch {epoch:03d}/{epochs} – Val ROC‑AUC: {val_auc:.4f}")

        # ---------- early stopping ----------
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

    # Save the best model state‑dict
    ensure_dir(model_path.parent)
    torch.save(best_state, model_path)
    print(f"[MODEL] Best model saved → {model_path}")

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
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"[METADATA] Saved → {metadata_path}")

    return model_path, metadata


# ----------------------------------------------------------------------
# 1️⃣2️⃣  Model profiling (size, params, latency, etc.)
# ----------------------------------------------------------------------
def profile_model(model_path: Path, metadata: dict, profile_path: Path):
    # Size on disk (bytes)
    size_bytes = model_path.stat().st_size

    # Number of trainable parameters (from saved state‑dict)
    state = torch.load(model_path, map_location="cpu")
    total_params = sum(p.numel() for p in state.values())

    # Simple CPU latency benchmark (average time per single row)
    model = TorchLogReg(metadata["input_dim"])
    model.load_state_dict(state)
    model.eval()
    dummy = torch.randn(1, metadata["input_dim"])
    repeats = 2000
    start = time.time()
    with torch.no_grad():
        for _ in range(repeats):
            _ = model(dummy)
    avg_latency = (time.time() - start) / repeats  # seconds per row

    profile = {
        "model_path": str(model_path),
        "size_bytes": size_bytes,
        "trainable_parameters": total_params,
        "avg_cpu_latency_seconds_per_row": avg_latency,
        "training_time_seconds": metadata["training_time_seconds"],
        "final_val_roc_auc": metadata["final_val_roc_auc"],
        # optional fields (filled if git/MLflow are available)
        "git_commit_hash": None,
        "mlflow_run_id": None,
    }

    # Try to fetch git commit hash (non‑fatal if not a git repo)
    try:
        import subprocess
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        profile["git_commit_hash"] = commit
    except Exception:
        pass

    # If an MLflow run was started elsewhere we could capture its ID;
    # here we simply leave it as None.

    ensure_dir(profile_path.parent)
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=4)
    print(f"[PROFILE] Model profile saved → {profile_path}")


# ----------------------------------------------------------------------
# 1️⃣3️⃣  Copy model to downstream workspace (required by serving code)
# ----------------------------------------------------------------------
def copy_to_workspace(src: Path, dst: Path):
    ensure_dir(dst.parent)
    shutil.copy(src, dst)
    print(f"[COPY] Model copied to downstream location → {dst}")


# ----------------------------------------------------------------------
# 1️⃣4️⃣  Main orchestration
# ----------------------------------------------------------------------
def main():
    # ------------------------------------------------------------------
    # Resolve all paths from config (relative to repository root)
    # ------------------------------------------------------------------
    p = cfg["paths"]
    raw_csv_src = Path(
        r"C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\titanic.csv"
    )
    raw_csv_dest = Path(p["raw_csv"])
    raw_parquet = Path(p["raw_parquet"])
    checksum_path = Path(p["dataset_checksum"])

    train_csv = Path(p["train_csv"])
    test_csv = Path(p["test_csv"])
    train_parquet = Path(p["train_parquet"])
    test_parquet = Path(p["test_parquet"])

    pipeline_path = Path(p["pipeline"])

    processed_train_parquet = Path(p["processed_train_parquet"])
    processed_test_parquet = Path(p["processed_test_parquet"])
    processed_train_npy = Path(p["processed_train_npy"])
    processed_test_npy = Path(p["processed_test_npy"])

    model_path = Path(p["model_pt"])
    metadata_path = Path(p["model_metadata"])
    profile_path = Path(p["model_profile"])
    downstream_path = Path(p["downstream_model"])

    # ------------------------------------------------------------------
    # 1️⃣ Ingest raw CSV & checksum
    # ------------------------------------------------------------------
    ingest_raw(raw_csv_src, raw_csv_dest, checksum_path)

    # ------------------------------------------------------------------
    # 2️⃣ Convert CSV → Parquet
    # ------------------------------------------------------------------
    csv_to_parquet(raw_csv_dest, raw_parquet)

    # ------------------------------------------------------------------
    # 3️⃣ Train‑test split (stratified)
    # ------------------------------------------------------------------
    split_cfg = cfg["train_test_split"]
    split_data(
        raw_csv_dest,
        train_csv,
        test_csv,
        train_parquet,
        test_parquet,
        test_size=split_cfg["test_size"],
        stratify=split_cfg["stratify"],
    )

    # ------------------------------------------------------------------
    # 4️⃣ Fit preprocessing pipeline on training split
    # ------------------------------------------------------------------
    fit_and_save_pipeline(train_csv, pipeline_path)

    # ------------------------------------------------------------------
    # 5️⃣ Transform both splits & persist matrices (Parquet + .npy)
    # ------------------------------------------------------------------
    transform_and_save(
        train_csv,
        pipeline_path,
        processed_train_parquet,
        processed_train_npy,
    )
    transform_and_save(
        test_csv,
        pipeline_path,
        processed_test_parquet,
        processed_test_npy,
    )

    # ------------------------------------------------------------------
    # 6️⃣ Train PyTorch logistic‑regression model
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path, metadata = train_torch_model(
        processed_train_npy,
        processed_test_npy,
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
