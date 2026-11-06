
"""
Titanic tabular classification pipeline
========================================

A **single‑file** end‑to‑end implementation that follows the play‑book
described in the previous answer.  It

1. copies the user CSV into the repository,
2. creates a SHA‑256 checksum,
3. converts the CSV → Parquet,
4. performs a stratified train‑test split,
5. builds a scikit‑learn preprocessing pipeline (feature engineering,
   iterative imputation, robust scaling, target encoding),
6. trains a pure‑PyTorch logistic‑regression model with early‑stopping on
   ROC‑AUC,
7. saves the pipeline, the trained model and a **model‑profile** JSON,
8. copies the model to ``./agent_workspace/trained_models`` – the location
   expected by the downstream inference service,
9. (optionally) launches a tiny Gradio demo for manual testing.

All paths, hyper‑parameters and feature‑engineering flags are read from
``config.yaml`` (must exist at the repository root).  The script is
self‑contained – just run it from the repo root after installing the
dependencies listed in ``requirements.txt``.
"""

import json
import os
import shutil
import hashlib
import time
from pathlib import Path

# ----------------------------------------------------------------------
# 1️⃣  Load configuration
# ----------------------------------------------------------------------
import yaml

CONFIG_PATH = Path(__file__).parents[0] / "config.yaml"
if not CONFIG_PATH.is_file():
    raise FileNotFoundError(f"Configuration file not found at {CONFIG_PATH}")

with open(CONFIG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

# ----------------------------------------------------------------------
# 2️⃣  Helper utilities
# ----------------------------------------------------------------------
def set_random_seed(seed: int = 42) -> None:
    import random, numpy as np, torch

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
# 3️⃣  Step‑by‑step pipeline
# ----------------------------------------------------------------------
def ingest_user_csv(source_abs_path: str) -> Path:
    """
    Copies the user‑provided CSV into ``data/raw/titanic.csv``.
    """
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
    import pandas as pd

    df = pd.read_csv(csv_path, dtype=str)  # keep strings for now
    parquet_path = Path(CFG["paths"]["raw_parquet"])
    ensure_dir(parquet_path.parent)
    df.to_parquet(parquet_path, index=False)
    print(f"[PARQUET] Saved to {parquet_path}")
    return parquet_path


def stratified_split(parquet_path: Path) -> tuple[Path, Path]:
    import pandas as pd
    from sklearn.model_selection import train_test_split

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

    # also parquet versions (useful for fast I/O later)
    train_df.to_parquet(train_path.with_suffix(".parquet"), index=False)
    test_df.to_parquet(test_path.with_suffix(".parquet"), index=False)

    print(f"[SPLIT] Train → {train_path}, Test → {test_path}")
    return train_path, test_path


# ----------------------------------------------------------------------
# 4️⃣  Feature‑engineering transformers
# ----------------------------------------------------------------------
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import pandas as pd


class FeatureEngineering(BaseEstimator, TransformerMixin):
    """Adds family_size, is_alone, title, fare_per_person."""

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def fit(self, X, y=None):
        return self

    def _extract_title(self, name: str) -> str:
        # Very simple heuristic – split by comma and period
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
        # family_size
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
    """Bins age into quantile groups (default 5)."""

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
# 5️⃣  Build full preprocessing pipeline
# ----------------------------------------------------------------------
def build_preprocess_pipeline() -> "sklearn.pipeline.Pipeline":
    """
    Returns a fitted scikit‑learn pipeline (not yet fitted on data).
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import RobustScaler
    from sklearn.impute import SimpleImputer
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.impute import IterativeImputer
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import OneHotEncoder

    # Enable experimental import before using IterativeImputer
    # (already done above)

    # ------------------------------------------------------------------
    # 5.1 Feature engineering steps
    # ------------------------------------------------------------------
    fe = FeatureEngineering(cfg=CFG["preprocess"])
    age_binner = AgeGroupBinner(n_bins=CFG["preprocess"]["add_features"]["age_bins"])

    # ------------------------------------------------------------------
    # 5.2 Column groups
    # ------------------------------------------------------------------
    numeric_features = CFG["preprocess"]["numeric_features"]
    categorical_features = CFG["preprocess"]["categorical_features"]

    # ------------------------------------------------------------------
    # 5.3 Numeric pipeline
    # ------------------------------------------------------------------
    numeric_pipe_steps = []

    # Missing‑indicator (IterativeImputer can add indicator)
    if CFG["preprocess"]["missing_indicator"]:
        numeric_pipe_steps.append(
            ("imputer", IterativeImputer(add_indicator=True, random_state=CFG["random_seed"]))
        )
    else:
        numeric_pipe_steps.append(
            ("imputer", IterativeImputer(random_state=CFG["random_seed"]))
        )
    numeric_pipe_steps.append(("scaler", RobustScaler()))
    numeric_pipe = Pipeline(numeric_pipe_steps)

    # ------------------------------------------------------------------
    # 5.4 Categorical pipeline
    # ------------------------------------------------------------------
    cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # TargetEncoder is in the optional package `category_encoders`
            ("encoder", __import__("category_encoders").encoding.TargetEncoder()),
        ]
    )

    # ------------------------------------------------------------------
    # 5.5 Assemble ColumnTransformer
    # ------------------------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0,
    )

    # ------------------------------------------------------------------
    # 5.6 Full pipeline
    # ------------------------------------------------------------------
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
    import joblib

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


def transform_and_save(
    pipeline,
    train_path: Path,
    test_path: Path,
) -> tuple[Path, Path]:
    """Transform train / test and persist as parquet (float32)."""
    import pandas as pd
    import numpy as np

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y_train = train_df[CFG["target_column"]].astype(int).values.reshape(-1, 1)
    y_test = test_df[CFG["target_column"]].astype(int).values.reshape(-1, 1)

    X_train_raw = train_df.drop(columns=[CFG["target_column"]])
    X_test_raw = test_df.drop(columns=[CFG["target_column"]])

    X_train = pipeline.transform(X_train_raw).astype(np.float32)
    X_test = pipeline.transform(X_test_raw).astype(np.float32)

    # Save as parquet for fast torch loading
    train_out = Path(CFG["paths"]["processed_train_parquet"])
    test_out = Path(CFG["paths"]["processed_test_parquet"])
    ensure_dir(train_out.parent)

    pd.DataFrame(X_train).to_parquet(train_out, index=False)
    pd.DataFrame(X_test).to_parquet(test_out, index=False)

    # Also keep the labels (numpy) – useful for training loop
    np.save("data/processed/y_train.npy", y_train)
    np.save("data/processed/y_test.npy", y_test)

    print(f"[TRANSFORM] Train shape {X_train.shape}, Test shape {X_test.shape}")
    return train_out, test_out


# ----------------------------------------------------------------------
# 6️⃣  PyTorch logistic‑regression model
# ----------------------------------------------------------------------
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)  # bias included

    def forward(self, x):
        return self.linear(x)  # logits (no sigmoid)


def train_torch_model(
    train_parquet: Path,
    test_parquet: Path,
    pipeline_path: Path,
) -> dict:
    """Trains the model with early stopping on ROC‑AUC."""
    import numpy as np
    from sklearn.metrics import roc_auc_score

    # Load data
    X_train = np.load(train_parquet.with_suffix(".npy"))
    X_test = np.load(test_parquet.with_suffix(".npy"))
    y_train = np.load("data/processed/y_train.npy").ravel()
    y_test = np.load("data/processed/y_test.npy").ravel()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = CFG["model"]["batch_size"]
    max_epochs = CFG["model"]["max_epochs"]
    patience = CFG["model"]["early_stopping_patience"]
    lr = CFG["model"]["learning_rate"]
    weight_decay = CFG["model"]["weight_decay"]

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    model = LogisticRegressionModel(input_dim=X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -np.inf
    epochs_no_improve = 0
    best_state = None
    training_time = 0.0

    for epoch in range(1, max_epochs + 1):
        start = time.time()
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        training_time += time.time() - start

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

        print(
            f"[EPOCH {epoch:03d}] val_roc_auc={val_auc:.4f}  loss={loss.item():.4f}"
        )

        # early stopping check
        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
            best_state = model.state_dict()
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
    print(f"[MODEL] Saved state dict to {model_path}")

    # copy to the location expected by the serving layer
    deploy_dir = Path("./agent_workspace/trained_models")
    ensure_dir(deploy_dir)
    shutil.copy(model_path, deploy_dir / model_path.name)
    print(f"[DEPLOY] Copied model to {deploy_dir}")

    # ------------------------------------------------------------------
    # Save metadata JSON
    # ------------------------------------------------------------------
    metadata = {
        "input_dim": X_train.shape[1],
        "final_val_roc_auc": float(best_auc),
        "epochs_trained": epoch,
        "batch_size": batch_size,
        "learning_rate": lr,
        "training_seed": CFG["random_seed"],
        "training_time_seconds": training_time,
    }
    meta_path = Path("artifacts/model_metadata.json")
    ensure_dir(meta_path.parent)
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"[METADATA] Written to {meta_path}")

    return metadata


# ----------------------------------------------------------------------
# 7️⃣  Model profiling (size, latency, etc.)
# ----------------------------------------------------------------------
def profile_model(metadata: dict) -> None:
    import os
    import time
    import torch
    import numpy as np
    from sklearn.metrics import roc_auc_score

    model_path = Path(CFG["paths"]["torch_model"])
    size_bytes = os.path.getsize(model_path)

    # Load model
    input_dim = metadata["input_dim"]
    model = LogisticRegressionModel(input_dim)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Dummy data for latency measurement (10k rows)
    dummy_X = np.random.randn(10_000, input_dim).astype(np.float32)
    dummy_tensor = torch.from_numpy(dummy_X)

    start = time.perf_counter()
    with torch.no_grad():
        _ = model(dummy_tensor)
    latency_ms = (time.perf_counter() - start) * 1000 / dummy_X.shape[0]

    profile = {
        "model_file_size_bytes": size_bytes,
        "num_trainable_parameters": sum(p.numel() for p in model.parameters()),
        "average_inference_latency_ms_per_row": latency_ms,
        "final_val_roc_auc": metadata["final_val_roc_auc"],
    }

    profile_path = Path("reports/model_profile.json")
    ensure_dir(profile_path.parent)
    profile_path.write_text(json.dumps(profile, indent=2))
    print(f"[PROFILE] Saved to {profile_path}")


# ----------------------------------------------------------------------
# 8️⃣  (Optional) Gradio demo – quick sanity check
# ----------------------------------------------------------------------
def launch_gradio_demo(pipeline_path: Path, model_path: Path) -> None:
    try:
        import gradio as gr
        import pandas as pd
        import joblib
        import torch
    except Exception as e:
        print("[GRADIO] Skipping demo – required packages not installed.")
        return

    pipeline = joblib.load(pipeline_path)
    model = LogisticRegressionModel(pipeline.transform(pd.DataFrame()).shape[1])
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()

    # Build a simple UI based on the original raw columns (excluding target)
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
        description="Enter passenger details and get the predicted survival probability.",
    )
    iface.launch(share=False)


# ----------------------------------------------------------------------
# 9️⃣  Main orchestration
# ----------------------------------------------------------------------
def main():
    # ------------------------------------------------------------------
    # 0️⃣  Global seed
    # ------------------------------------------------------------------
    set_random_seed(CFG["random_seed"])

    # ------------------------------------------------------------------
    # 1️⃣  Ingest the user CSV (absolute path must be supplied by the
    #     operator – adjust `USER_CSV_ABS_PATH` accordingly)
    # ------------------------------------------------------------------
    USER_CSV_ABS_PATH = (
        r"C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\titanic.csv"
    )
    raw_csv = ingest_user_csv(USER_CSV_ABS_PATH)

    # ------------------------------------------------------------------
    # 2️⃣  Checksum
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
    train_parquet, test_parquet = transform_and_save(pipeline, train_csv, test_csv)

    # ------------------------------------------------------------------
    # 7️⃣  Train PyTorch logistic regression
    # ------------------------------------------------------------------
    metadata = train_torch_model(
        train_parquet=train_parquet,
        test_parquet=test_parquet,
        pipeline_path=Path(CFG["paths"]["pipeline_pickle"]),
    )

    # ------------------------------------------------------------------
    # 8️⃣  Profile model
    # ------------------------------------------------------------------
    profile_model(metadata)

    # ------------------------------------------------------------------
    # 9️⃣  Optional Gradio demo
    # ------------------------------------------------------------------
    launch_gradio_demo(
        pipeline_path=Path(CFG["paths"]["pipeline_pickle"]),
        model_path=Path(CFG["paths"]["torch_model"]),
    )


if __name__ == "__main__":
    main()
