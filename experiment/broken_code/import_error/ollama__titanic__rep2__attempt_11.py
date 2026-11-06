
# --------------------------------------------------------------
# run_pipeline.py
# --------------------------------------------------------------
# End‑to‑end Titanic tabular‑classification pipeline (pure‑PyTorch
# logistic‑regression) that follows the repository layout and
# config.yaml described in the specification.
# --------------------------------------------------------------

import os
import sys
import json
import time
import shutil
import hashlib
import warnings
import yaml
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import RobustScaler, OrdinalEncoder
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.base import BaseEstimator, TransformerMixin

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ------------------------------------------------------------------
# 0️⃣  Load configuration (single source of truth)
# ------------------------------------------------------------------
CONFIG_PATH = Path(__file__).parent / "config.yaml"
if not CONFIG_PATH.is_file():
    raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")

with CONFIG_PATH.open("r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

# ------------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------------
def sha256(file_path: Path) -> str:
    """Return SHA‑256 checksum of a file."""
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(p: Path) -> None:
    """Create parent directories if they do not exist."""
    p.parent.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------
# 1️⃣  Copy raw CSV into repository & write checksum
# ------------------------------------------------------------------
def ingest_and_checksum() -> None:
    src = Path(
        r"C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\titanic.csv"
    )
    dst = Path(CFG["paths"]["raw_csv"])
    ensure_dir(dst)

    if not src.is_file():
        raise FileNotFoundError(f"Source CSV not found at {src}")

    shutil.copy(src, dst)
    print(f"[INGEST] Copied raw CSV to {dst}")

    # checksum
    checksum = sha256(dst)
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "dataset_checksum.txt").write_text(
        checksum, encoding="utf-8"
    )
    print(f"[CHECKSUM] SHA‑256 = {checksum}")


# ------------------------------------------------------------------
# 2️⃣  Convert CSV → Parquet (fast I/O)
# ------------------------------------------------------------------
def csv_to_parquet() -> None:
    raw_csv = Path(CFG["paths"]["raw_csv"])
    raw_parquet = Path(CFG["paths"]["raw_parquet"])
    ensure_dir(raw_parquet)

    df = pd.read_csv(raw_csv, dtype=str)  # keep strings for safe engineering
    df.to_parquet(raw_parquet, index=False)
    print(f"[PARQUET] Saved to {raw_parquet}")


# ------------------------------------------------------------------
# 3️⃣  Stratified train‑test split (80 % / 20 %)
# ------------------------------------------------------------------
def stratified_split() -> None:
    raw_parquet = Path(CFG["paths"]["raw_parquet"])
    train_csv = Path(CFG["paths"]["train_csv"])
    test_csv = Path(CFG["paths"]["test_csv"])

    df = pd.read_parquet(raw_parquet)

    y = df[CFG["target_column"]].astype(int)
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=CFG["train_test_split"]["test_size"],
        stratify=y if CFG["train_test_split"]["stratify"] else None,
        random_state=CFG["random_seed"],
    )

    train_df = df.loc[train_idx].reset_index(drop=True)
    test_df = df.loc[test_idx].reset_index(drop=True)

    ensure_dir(train_csv)
    ensure_dir(test_csv)

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # also write parquet versions (useful for debugging)
    train_df.to_parquet(train_csv.with_suffix(".parquet"), index=False)
    test_df.to_parquet(test_csv.with_suffix(".parquet"), index=False)

    print(f"[SPLIT] Train → {train_csv}, Test → {test_csv}")


# ------------------------------------------------------------------
# 4️⃣  Feature‑engineering transformers
# ------------------------------------------------------------------
class FeatureEngineering(BaseEstimator, TransformerMixin):
    """Adds family_size, is_alone, title, fare_per_person."""

    def __init__(self, cfg):
        self.cfg = cfg

    @staticmethod
    def _extract_title(name):
        if pd.isna(name):
            return "Unknown"
        parts = name.split(",")
        if len(parts) > 1:
            sub = parts[1].split(".")
            if len(sub) > 0:
                return sub[0].strip()
        return "Unknown"

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # family size
        if self.cfg["add_features"]["family_size"]:
            X["family_size"] = (
                pd.to_numeric(X["sibsp"], errors="coerce").fillna(0)
                + pd.to_numeric(X["parch"], errors="coerce").fillna(0)
                + 1
            )
        # is_alone
        if self.cfg["add_features"]["is_alone"]:
            X["is_alone"] = (X["family_size"] == 1).astype(int)
        # title extraction
        if self.cfg["add_features"]["title_extraction"]:
            X["title"] = X["name"].apply(self._extract_title)
        # fare per person
        if "fare" in X.columns:
            X["fare_per_person"] = pd.to_numeric(
                X["fare"], errors="coerce"
            ) / X["family_size"]
        return X


class AgeGroupBinner(BaseEstimator, TransformerMixin):
    """Bins age into `n_bins` quantile groups."""

    def __init__(self, n_bins=5):
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
        # explicit “Missing” label for NaNs
        X["age_group"] = X["age_group"].cat.add_categories("Missing").fillna(
            "Missing"
        )
        return X


# ------------------------------------------------------------------
# 5️⃣  Build full preprocessing pipeline
# ------------------------------------------------------------------
def build_pipeline() -> Pipeline:
    cfg = CFG["preprocess"]

    # 1️⃣ Feature engineering
    fe = FeatureEngineering(cfg)

    # 2️⃣ Age binning
    age_binner = AgeGroupBinner(n_bins=cfg["add_features"]["age_bins"])

    # 3️⃣ Numeric sub‑pipeline
    numeric_pipe = Pipeline(
        [
            (
                "imputer",
                IterativeImputer(
                    add_indicator=cfg["missing_indicator"],
                    random_state=CFG["random_seed"],
                ),
            ),
            ("scaler", RobustScaler()),
        ]
    )

    # 4️⃣ Categorical sub‑pipeline
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                ),
            ),
        ]
    )

    # 5️⃣ ColumnTransformer
    preproc = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, cfg["numeric_features"]),
            ("cat", cat_pipe, cfg["categorical_features"]),
        ],
        remainder="drop",
        sparse_threshold=0,
    )

    # 6️⃣ Full pipeline
    full_pipe = Pipeline(
        steps=[
            ("feature_engineering", fe),
            ("age_binner", age_binner),
            ("preprocess", preproc),
        ]
    )
    return full_pipe


# ------------------------------------------------------------------
# 6️⃣  Fit pipeline on training split & persist it
# ------------------------------------------------------------------
def fit_and_save_pipeline() -> None:
    train_path = Path(CFG["paths"]["train_csv"])
    pipeline_path = Path(CFG["paths"]["pipeline_pickle"])

    train_df = pd.read_csv(train_path)
    y = train_df[CFG["target_column"]].astype(int).values
    X = train_df.drop(columns=[CFG["target_column"]])

    pipe = build_pipeline()
    pipe.fit(X, y)

    ensure_dir(pipeline_path)
    joblib.dump(pipe, pipeline_path)
    print(f"[PIPELINE] Saved to {pipeline_path}")


# ------------------------------------------------------------------
# 7️⃣  Transform train / test splits and store matrices
# ------------------------------------------------------------------
def transform_and_save() -> None:
    pipe = joblib.load(Path(CFG["paths"]["pipeline_pickle"]))

    # ---------- train ----------
    train_df = pd.read_csv(Path(CFG["paths"]["train_csv"]))
    y_train = (
        train_df[CFG["target_column"]].astype(int).values.reshape(-1, 1)
    )
    X_train_raw = train_df.drop(columns=[CFG["target_column"]])
    X_train = pipe.transform(X_train_raw).astype(np.float32)

    # ---------- test ----------
    test_df = pd.read_csv(Path(CFG["paths"]["test_csv"]))
    y_test = test_df[CFG["target_column"]].astype(int).values.reshape(-1, 1)
    X_test_raw = test_df.drop(columns=[CFG["target_column"]])
    X_test = pipe.transform(X_test_raw).astype(np.float32)

    # Persist as Parquet (fast downstream I/O)
    ensure_dir(Path(CFG["paths"]["processed_train_parquet"]))
    ensure_dir(Path(CFG["paths"]["processed_test_parquet"]))

    pd.DataFrame(X_train).to_parquet(
        CFG["paths"]["processed_train_parquet"], index=False
    )
    pd.DataFrame(X_test).to_parquet(
        CFG["paths"]["processed_test_parquet"], index=False
    )

    # Also persist as .npy for the PyTorch DataLoader
    np.save("data/processed/X_train.npy", X_train)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/X_test.npy", X_test)
    np.save("data/processed/y_test.npy", y_test)

    print(f"[TRANSFORM] Train shape {X_train.shape}, Test shape {X_test.shape}")


# ------------------------------------------------------------------
# 8️⃣  PyTorch logistic‑regression model definition
# ------------------------------------------------------------------
class LogisticRegressionModel(nn.Module):
    """Single linear layer – outputs logits (no sigmoid)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # bias included automatically

    def forward(self, x):
        return self.linear(x)  # logits


# ------------------------------------------------------------------
# 9️⃣  Training loop with early stopping on ROC‑AUC
# ------------------------------------------------------------------
def train() -> dict:
    # Load transformed data (numpy)
    X_train = np.load("data/processed/X_train.npy")
    y_train = np.load("data/processed/y_train.npy").ravel()
    X_val = np.load("data/processed/X_test.npy")
    y_val = np.load("data/processed/y_test.npy").ravel()

    # Hyper‑parameters from config
    cfg = CFG["model"]
    batch_size = cfg["batch_size"]
    max_epochs = cfg["max_epochs"]
    patience = cfg["early_stopping_patience"]
    lr = cfg["learning_rate"]
    wd = cfg["weight_decay"]

    # DataLoaders
    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train)
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LogisticRegressionModel(input_dim=X_train.shape[1]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -np.inf
    epochs_without_improve = 0
    best_state = None
    total_train_time = 0.0
    final_epoch = 0

    for epoch in range(1, max_epochs + 1):
        # ------------------- training -------------------
        model.train()
        epoch_start = time.time()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        total_train_time += time.time() - epoch_start

        # ------------------- validation -------------------
        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                probs = torch.sigmoid(logits).cpu().numpy().ravel()
                all_probs.extend(probs)
                all_labels.extend(yb.numpy())
        val_auc = roc_auc_score(all_labels, all_probs)
        print(f"[EPOCH {epoch:03d}] val ROC‑AUC = {val_auc:.4f}")

        # ------------------- early stopping -------------------
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = model.state_dict()
            epochs_without_improve = 0
            final_epoch = epoch
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= patience:
                print(f"[EARLY STOP] No improvement for {patience} epochs – stopping.")
                break

    # Load best weights
    model.load_state_dict(best_state)

    # ------------------------------------------------------------------
    # Persist the model (state‑dict)
    # ------------------------------------------------------------------
    model_path = Path(CFG["paths"]["torch_model"])
    ensure_dir(model_path)
    torch.save(model.state_dict(), model_path)
    print(f"[MODEL] State dict saved to {model_path}")

    # Copy to downstream expected location
    deploy_dir = Path("./agent_workspace/trained_models")
    deploy_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(model_path, deploy_dir / model_path.name)

    # ------------------------------------------------------------------
    # Write metadata JSON
    # ------------------------------------------------------------------
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
    ensure_dir(meta_path)
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[METADATA] Written to {meta_path}")

    return metadata


# ------------------------------------------------------------------
# 10️⃣  Optional: final test‑set evaluation (can be called separately)
# ------------------------------------------------------------------
def evaluate_test_set() -> float:
    X_test = np.load("data/processed/X_test.npy")
    y_test = np.load("data/processed/y_test.npy").ravel()

    model = LogisticRegressionModel(input_dim=X_test.shape[1])
    model.load_state_dict(
        torch.load(Path(CFG["paths"]["torch_model"]), map_location="cpu")
    )
    model.eval()

    test_ds = TensorDataset(
        torch.from_numpy(X_test), torch.from_numpy(y_test)
    )
    test_loader = DataLoader(
        test_ds, batch_size=CFG["model"]["batch_size"], shuffle=False
    )

    all_probs, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            logits = model(xb)
            probs = torch.sigmoid(logits).numpy().ravel()
            all_probs.extend(probs)
            all_labels.extend(yb.numpy())

    test_auc = roc_auc_score(all_labels, all_probs)
    print(f"[TEST] ROC‑AUC = {test_auc:.4f}")
    return float(test_auc)


# ------------------------------------------------------------------
# 11️⃣  Orchestrator – run everything in the correct order
# ------------------------------------------------------------------
def main() -> None:
    # 1️⃣  Ingest raw CSV & write checksum
    ingest_and_checksum()

    # 2️⃣  Convert to Parquet
    csv_to_parquet()

    # 3️⃣  Train‑test split
    stratified_split()

    # 4️⃣  Fit preprocessing pipeline
    fit_and_save_pipeline()

    # 5️⃣  Transform data & persist matrices
    transform_and_save()

    # 6️⃣  Train PyTorch logistic regression
    metadata = train()

    # 7️⃣  (Optional) evaluate on the held‑out test set and add to metadata
    test_auc = evaluate_test_set()
    metadata["test_roc_auc"] = test_auc
    meta_path = Path("artifacts/model_metadata.json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("[METADATA] Updated with test ROC‑AUC")

    print("\n=== PIPELINE FINISHED SUCCESSFULLY ===")
    print(f"Model artefacts are in ./artifacts and ./agent_workspace/trained_models")
    print(f"Metadata: {metadata}")


if __name__ == "__main__":
    # Suppress non‑essential warnings for a clean run
    warnings.filterwarnings("ignore")
    main()
