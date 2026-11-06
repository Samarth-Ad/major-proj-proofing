
"""
Titanic tabular classification – end‑to‑end pipeline
===================================================

This single‑file script implements the complete workflow described in the
play‑book:

1.  Locate and read ``config.yaml`` (searched upward from the script location).
2.  Copy the user‑provided CSV into ``data/raw/titanic.csv``.
3.  Write a SHA‑256 checksum (auditability).
4.  Convert the CSV → Parquet.
5.  Perform a stratified 80/20 train‑test split.
6.  Build a scikit‑learn preprocessing pipeline that
    * adds engineered features,
    * bins age,
    * imputes numerics with ``IterativeImputer`` (adds missing‑indicators),
    * scales numerics with ``RobustScaler``,
    * imputes categoricals with ``SimpleImputer`` and target‑encodes them.
7.  Fit the pipeline on the training split and persist it.
8.  Transform train / test, store the matrices as Parquet (float32).
9.  Train a pure‑PyTorch logistic‑regression model with early‑stopping on
    ROC‑AUC.
10. Save the model state‑dict, copy it to ``./agent_workspace/trained_models/``,
    and write a small JSON metadata file.
11. Generate a model‑profile JSON (size, latency, number of parameters, final AUC).
12. (Optional) launch a tiny Gradio demo – it will be skipped if ``gradio`` is
    not installed.

All hyper‑parameters, file locations and feature‑engineering flags are read
from ``config.yaml`` (must exist at the repository root).  The script can be
run directly from the repository root after installing the packages listed in
``requirements.txt``.
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

# ----------------------------------------------------------------------
# Helper: locate ``config.yaml`` by walking up the directory tree
# ----------------------------------------------------------------------
def find_config(start_path: Path) -> Path:
    """Search upwards from ``start_path`` for a file named ``config.yaml``."""
    cur = start_path.resolve()
    for _ in range(10):  # safety guard – 10 levels is enough
        candidate = cur / "config.yaml"
        if candidate.is_file():
            return candidate
        if cur.parent == cur:  # reached filesystem root
            break
        cur = cur.parent
    raise FileNotFoundError("Could not locate config.yaml in any parent directory.")

CONFIG_PATH = find_config(Path(__file__).parent)
with open(CONFIG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

# ----------------------------------------------------------------------
# 1️⃣  Utility functions
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
# 2️⃣  Data ingestion
# ----------------------------------------------------------------------
def ingest_user_csv(source_abs_path: str) -> Path:
    """Copy the user CSV into the repository (location defined in config)."""
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
    print(f"[CHECKSUM] SHA‑256 → {out_file}")


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

    from sklearn.model_selection import train_test_split

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
# 3️⃣  Feature‑engineering transformers
# ----------------------------------------------------------------------
from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineering(BaseEstimator, TransformerMixin):
    """Adds family_size, is_alone, title, fare_per_person."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def fit(self, X, y=None):
        return self

    def _extract_title(self, name: str) -> str:
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
# 4️⃣  Build the full preprocessing pipeline
# ----------------------------------------------------------------------
def build_preprocess_pipeline() -> "sklearn.pipeline.Pipeline":
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler
    from sklearn.impute import SimpleImputer
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer

    # Feature‑engineering steps
    fe = FeatureEngineering(cfg=CFG["preprocess"])
    age_binner = AgeGroupBinner(n_bins=CFG["preprocess"]["add_features"]["age_bins"])

    # Column groups from config
    numeric_features = CFG["preprocess"]["numeric_features"]
    categorical_features = CFG["preprocess"]["categorical_features"]

    # Numeric sub‑pipeline
    numeric_steps = []
    if CFG["preprocess"]["missing_indicator"]:
        numeric_steps.append(
            ("imputer", IterativeImputer(add_indicator=True, random_state=CFG["random_seed"]))
        )
    else:
        numeric_steps.append(
            ("imputer", IterativeImputer(random_state=CFG["random_seed"]))
        )
    numeric_steps.append(("scaler", RobustScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    # Categorical sub‑pipeline (target encoder from category_encoders)
    try:
        from category_encoders import TargetEncoder
    except ImportError as e:
        raise ImportError(
            "Package 'category_encoders' is required for target encoding. "
            "Install it via `pip install category_encoders`."
        ) from e

    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", TargetEncoder()),
        ]
    )

    # Assemble ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0,
    )

    # Full pipeline
    full_pipe = Pipeline(
        steps=[
            ("feature_engineering", fe),
            ("age_binner", age_binner),
            ("preprocess", preprocessor),
        ]
    )
    return full_pipe


def fit_and_save_pipeline(train_path: Path) -> "sklearn.pipeline.Pipeline":
    import pandas as pd

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


def transform_and_save(pipeline, train_path: Path, test_path: Path) -> tuple[Path, Path]:
    """Transform train / test, store as Parquet (float32) and also as .npy for torch."""
    import pandas as pd

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y_train = train_df[CFG["target_column"]].astype(int).values.reshape(-1, 1)
    y_test = test_df[CFG["target_column"]].astype(int).values.reshape(-1, 1)

    X_train_raw = train_df.drop(columns=[CFG["target_column"]])
    X_test_raw = test_df.drop(columns=[CFG["target_column"]])

    X_train = pipeline.transform(X_train_raw).astype(np.float32)
    X_test = pipeline.transform(X_test_raw).astype(np.float32)

    # Persist Parquet (fast I/O for later)
    train_out = Path(CFG["paths"]["processed_train_parquet"])
    test_out = Path(CFG["paths"]["processed_test_parquet"])
    ensure_dir(train_out.parent)

    pd.DataFrame(X_train).to_parquet(train_out, index=False)
    pd.DataFrame(X_test).to_parquet(test_out, index=False)

    # Also persist raw numpy arrays for the torch DataLoader
    np.save("data/processed/X_train.npy", X_train)
    np.save("data/processed/X_test.npy", X_test)
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/y_test.npy", y_test)

    print(f"[TRANSFORM] Train shape {X_train.shape}, Test shape {X_test.shape}")
    return train_out, test_out


# ----------------------------------------------------------------------
# 5️⃣  PyTorch logistic‑regression model definition
# ----------------------------------------------------------------------
class LogisticRegressionModel(nn.Module):
    """Single linear layer – outputs logits (no sigmoid)."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)


# ----------------------------------------------------------------------
# 6️⃣  Training loop with early stopping on ROC‑AUC
# ----------------------------------------------------------------------
def train_torch_model() -> dict:
    """Loads transformed data, trains the model, saves artefacts and returns metadata."""
    # Load transformed numpy arrays
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

    from sklearn.metrics import roc_auc_score

    for epoch in range(1, max_epochs + 1):
        start_epoch = time.time()
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        total_train_time += time.time() - start_epoch

        # ---- validation ROC‑AUC ----
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

        print(f"[EPOCH {epoch:03d}] val_roc_auc={val_auc:.4f}")

        # Early‑stopping check
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

    # ------------------------------------------------------------------
    # Save model artefacts
    # ------------------------------------------------------------------
    model_path = Path(CFG["paths"]["torch_model"])
    ensure_dir(model_path.parent)
    torch.save(model.state_dict(), model_path)
    print(f"[MODEL] State dict saved to {model_path}")

    # Copy to the location expected by the downstream inference service
    deploy_dir = Path("./agent_workspace/trained_models")
    ensure_dir(deploy_dir)
    shutil.copy(model_path, deploy_dir / model_path.name)
    print(f"[DEPLOY] Model copied to {deploy_dir}")

    # ------------------------------------------------------------------
    # Metadata JSON
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
    ensure_dir(meta_path.parent)
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"[METADATA] Written to {meta_path}")

    return metadata


# ----------------------------------------------------------------------
# 7️⃣  Model profiling (size, latency, parameters)
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
# 8️⃣  Optional Gradio demo
# ----------------------------------------------------------------------
def launch_gradio_demo() -> None:
    try:
        import gradio as gr
        import joblib
        import torch
    except Exception:
        print("[GRADIO] Skipping demo – required packages not installed.")
        return

    pipeline = joblib.load(Path(CFG["paths"]["pipeline_pickle"]))
    model = LogisticRegressionModel(pipeline.transform(pd.DataFrame()).shape[1])
    model.load_state_dict(torch.load(Path(CFG["paths"]["torch_model"]), map_location="cpu"))
    model.eval()

    # Build UI based on raw columns (excluding target)
    raw_df = pd.read_csv(Path(CFG["paths"]["raw_csv"]), nrows=1)
    input_features = [c for c in raw_df.columns if c != CFG["target_column"]]

    def predict(**kwargs):
        df = pd.DataFrame([kwargs])
        X = pipeline.transform(df).astype(np.float32)
        with torch.no_grad():
            logits = model(torch.from_numpy(X))
            prob = torch.sigmoid(logits).item()
        return {"survival_probability": prob}

    iface = gr.Interface(
        fn=predict,
        inputs=[gr.Textbox(label=col) for col in input_features],
        outputs=gr.Number(label="Survival probability"),
        title="Titanic Survival – Logistic Regression (PyTorch)",
        description="Enter passenger details to obtain a survival probability.",
    )
    iface.launch(share=False)


# ----------------------------------------------------------------------
# 9️⃣  Main orchestration
# ----------------------------------------------------------------------
def main():
    # ------------------------------------------------------------------
    # Global seed
    # ------------------------------------------------------------------
    set_random_seed(CFG["random_seed"])

    # ------------------------------------------------------------------
    # 1️⃣  Ingest CSV – replace the path below with the absolute path to the
    #     user‑provided Titanic CSV on the host machine.
    # ------------------------------------------------------------------
    USER_CSV_ABS_PATH = (
        r"C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\titanic.csv"
    )
    raw_csv = ingest_user_csv(USER_CSV_ABS_PATH)

    # ------------------------------------------------------------------
    # 2️⃣  Write checksum
    # ------------------------------------------------------------------
    write_checksum(raw_csv)

    # ------------------------------------------------------------------
    # 3️⃣  CSV → Parquet
    # ------------------------------------------------------------------
    raw_parquet = csv_to_parquet(raw_csv)

    # ------------------------------------------------------------------
    # 4️⃣  Train‑test split
    # ------------------------------------------------------------------
    train_csv, test_csv = stratified_split(raw_parquet)

    # ------------------------------------------------------------------
    # 5️⃣  Build & fit preprocessing pipeline
    # ------------------------------------------------------------------
    pipeline = fit_and_save_pipeline(train_csv)

    # ------------------------------------------------------------------
    # 6️⃣  Transform data & persist
    # ------------------------------------------------------------------
    transform_and_save(pipeline, train_csv, test_csv)

    # ------------------------------------------------------------------
    # 7️⃣  Train PyTorch logistic regression
    # ------------------------------------------------------------------
    metadata = train_torch_model()

    # ------------------------------------------------------------------
    # 8️⃣  Profile the trained model
    # ------------------------------------------------------------------
    profile_model(metadata)

    # ------------------------------------------------------------------
    # 9️⃣  (Optional) Gradio demo
    # ------------------------------------------------------------------
    launch_gradio_demo()


if __name__ == "__main__":
    main()
