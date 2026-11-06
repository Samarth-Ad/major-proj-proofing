
# -*- coding: utf-8 -*-
"""
End‑to‑end tabular regression pipeline (Diabetes dataset example).

The script follows the robust MLOps checklist described in the prompt:
1️⃣  Load & validate raw CSV.
2️⃣  Train/validation/test split (70/20/10) with stratified bins.
3️⃣  Pre‑process: median imputation → Yeo‑Johnson power transform → Robust scaling.
4️⃣  PyTorch shallow MLP (10‑64‑1) with Kaiming‑Uniform init.
5️⃣  Training loop with Adam, ReduceLROnPlateau (no `verbose`), early‑stopping.
6️⃣  Evaluation (RMSE, MAE, R²) and JSON report.
7️⃣  Export artefacts (model checkpoint, pipeline, metadata).
8️⃣  Simple Gradio demo for inference.

All artefacts are stored under the repository root:
    ./data/raw/
    ./data/processed/
    ./models/
    ./agent_workspace/trained_models/
    ./reports/
"""

import os
import json
import hashlib
import logging
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gradio as gr

# --------------------------------------------------------------------------- #
# 0️⃣  Global configuration & reproducibility
# --------------------------------------------------------------------------- #
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

DEVICE = torch.device("cpu")  # CPU‑only for this tiny dataset

# --------------------------------------------------------------------------- #
# 1️⃣  Paths
# --------------------------------------------------------------------------- #
ROOT = Path.cwd()  # assume script is run from repository root
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
TRAINED_MODEL_DIR = ROOT / "agent_workspace" / "trained_models"
REPORT_DIR = ROOT / "reports"

for d in (RAW_DIR, PROC_DIR, MODEL_DIR, TRAINED_MODEL_DIR, REPORT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# 2️⃣  Logging (standard library)
# --------------------------------------------------------------------------- #
log_file = ROOT / "logs" / "pipeline.log"
log_file.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("diabetes_regression")

# --------------------------------------------------------------------------- #
# 3️⃣  Helper utilities
# --------------------------------------------------------------------------- #
def md5_checksum(path: Path) -> str:
    """Return the MD5 checksum of a file."""
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# 4️⃣  Ingestion & validation
# --------------------------------------------------------------------------- #
def ingest(csv_path: Path) -> pd.DataFrame:
    """
    Copy the user CSV to the raw folder, validate its shape and missingness,
    and generate a schema.json report.
    """
    logger.info("Starting ingestion")
    if not csv_path.is_file():
        raise FileNotFoundError(f"Provided CSV not found: {csv_path}")

    # copy to raw folder
    dest = RAW_DIR / "Diabetes_raw.csv"
    dest.write_bytes(csv_path.read_bytes())
    logger.info(f"Copied raw CSV to {dest}")

    # load & validate
    df = pd.read_csv(dest)
    expected_cols = 11  # 10 features + target
    if df.shape[1] != expected_cols:
        raise ValueError(f"Expected {expected_cols} columns, got {df.shape[1]}")
    missing_pct = df.isna().mean()
    if (missing_pct > 0.05).any():
        raise ValueError("One or more columns have >5% missing values")
    logger.info(f"Dataset shape: {df.shape}")

    # schema.json
    schema = {
        col: {"dtype": str(df[col].dtype), "missing_pct": float(missing_pct[col])}
        for col in df.columns
    }
    schema_path = REPORT_DIR / "schema.json"
    schema_path.write_text(json.dumps(schema, indent=2))
    logger.info(f"Wrote schema to {schema_path}")

    # log checksum
    checksum = md5_checksum(dest)
    logger.info(f"MD5 checksum of raw file: {checksum}")

    return df


# --------------------------------------------------------------------------- #
# 5️⃣  Train/validation/test split (stratified on target bins)
# --------------------------------------------------------------------------- #
def split_data(df: pd.DataFrame):
    logger.info("Splitting data")
    X = df.drop(columns=["target"])
    y = df["target"]

    # create stratification bins (5‑quantile)
    bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
    X_train, X_temp, y_train, y_temp, bins_train, bins_temp = train_test_split(
        X,
        y,
        bins,
        test_size=0.30,
        stratify=bins,
        random_state=SEED,
    )
    # second split: 2/3 of temp -> val (≈20%), 1/3 -> test (≈10%)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1 / 3,
        stratify=bins_temp,
        random_state=SEED,
    )

    # persist as .npy (keep column names for later reconstruction)
    for split, X_split, y_split in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        X_path = PROC_DIR / f"X_{split}.npy"
        y_path = PROC_DIR / f"y_{split}.npy"
        X_split.to_numpy().astype(np.float32).tofile(X_path)  # raw binary
        y_split.to_numpy().astype(np.float32).tofile(y_path)
        # also store shape metadata for loading
        meta = {
            "shape": X_split.shape,
            "columns": list(X_split.columns),
        }
        (PROC_DIR / f"{split}_meta.json").write_text(json.dumps(meta))
        logger.info(f"Saved {split} split: X {X_path}, y {y_path}")

    logger.info(
        f"Split counts -> train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}"
    )
    return (
        (X_train, y_train),
        (X_val, y_val),
        (X_test, y_test),
    )  # returning DataFrames for preprocessing step


# --------------------------------------------------------------------------- #
# 6️⃣  Pre‑processing pipeline (single linear pipeline)
# --------------------------------------------------------------------------- #
def fit_preprocess_pipeline(X_train: pd.DataFrame) -> Pipeline:
    logger.info("Fitting preprocessing pipeline")
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("power", PowerTransformer(method="yeo-johnson")),
            ("scaler", RobustScaler()),
        ]
    )
    pipeline.fit(X_train)
    pipeline_path = MODEL_DIR / "preprocess_pipe.joblib"
    joblib.dump(pipeline, pipeline_path)
    logger.info(f"Saved preprocessing pipeline to {pipeline_path}")

    # Transform and overwrite the .npy files for all splits
    for split in ("train", "val", "test"):
        meta_path = PROC_DIR / f"{split}_meta.json"
        meta = json.loads(meta_path.read_text())
        cols = meta["columns"]
        X_path = PROC_DIR / f"X_{split}.npy"
        # load raw binary with known shape
        X_arr = np.fromfile(X_path, dtype=np.float32).reshape(meta["shape"])
        X_df = pd.DataFrame(X_arr, columns=cols)
        X_trans = pipeline.transform(X_df).astype(np.float32)
        # overwrite
        X_path.unlink()
        X_trans.tofile(X_path)
        logger.info(f"Transformed {split} data and overwrote {X_path}")

    return pipeline


# --------------------------------------------------------------------------- #
# 7️⃣  PyTorch Dataset wrapper
# --------------------------------------------------------------------------- #
class DiabetesDataset(Dataset):
    def __init__(self, X_path: Path, y_path: Path):
        # Load raw binary files (saved earlier)
        X_meta = json.loads((PROC_DIR / f"{X_path.stem.split('_')[1]}_meta.json").read_text())
        X = np.fromfile(X_path, dtype=np.float32).reshape(X_meta["shape"])
        y = np.fromfile(y_path, dtype=np.float32).reshape(-1, 1)
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(batch_size: int = 32):
    train_dataset = DiabetesDataset(
        PROC_DIR / "X_train.npy", PROC_DIR / "y_train.npy"
    )
    val_dataset = DiabetesDataset(
        PROC_DIR / "X_val.npy", PROC_DIR / "y_val.npy"
    )
    test_dataset = DiabetesDataset(
        PROC_DIR / "X_test.npy", PROC_DIR / "y_test.npy"
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader, test_loader


# --------------------------------------------------------------------------- #
# 8️⃣  Model definition
# --------------------------------------------------------------------------- #
class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Kaiming‑Uniform init
        nn.init.kaiming_uniform_(self.net[0].weight, nonlinearity="relu")
        nn.init.zeros_(self.net[0].bias)
        nn.init.kaiming_uniform_(self.net[2].weight, nonlinearity="linear")
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x):
        return self.net(x)


def build_model():
    model = MLPRegressor().to(DEVICE)
    return model


# --------------------------------------------------------------------------- #
# 9️⃣  Training loop with early stopping & LR scheduler
# --------------------------------------------------------------------------- #
def train(
    model,
    train_loader,
    val_loader,
    max_epochs: int = 200,
    patience: int = 15,
):
    logger.info("Starting training")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )  # **no `verbose` argument**

    best_val_rmse = float("inf")
    epochs_no_improve = 0
    best_ckpt_path = TRAINED_MODEL_DIR / "torch_mlp_best.pt"

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = np.mean(train_losses)

        # ---- validation ----
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                preds = model(xb).cpu().numpy()
                val_preds.append(preds)
                val_targets.append(yb.numpy())
        val_preds = np.concatenate(val_preds).ravel()
        val_targets = np.concatenate(val_targets).ravel()

        val_rmse = mean_squared_error(val_targets, val_preds, squared=False)
        val_mae = mean_absolute_error(val_targets, val_preds)
        val_r2 = r2_score(val_targets, val_preds)

        scheduler.step(val_rmse)

        logger.info(
            f"Epoch {epoch:03d} | TrainLoss {avg_train_loss:.4f} | "
            f"Val RMSE {val_rmse:.4f} | MAE {val_mae:.4f} | R2 {val_r2:.4f}"
        )

        # ---- checkpoint ----
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), best_ckpt_path)
            logger.info(f"New best model saved (RMSE={best_val_rmse:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logger.info(
                f"Early stopping triggered after {epoch} epochs (no improvement for {patience} epochs)."
            )
            break

    # load best checkpoint before returning
    model.load_state_dict(torch.load(best_ckpt_path, map_location=DEVICE))
    logger.info(f"Training finished. Best model at {best_ckpt_path}")
    return model, best_ckpt_path


# --------------------------------------------------------------------------- #
# 10️⃣  Evaluation on test set
# --------------------------------------------------------------------------- #
def evaluate(model, test_loader):
    logger.info("Evaluating on test set")
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            out = model(xb).cpu().numpy()
            preds.append(out)
            targets.append(yb.numpy())
    preds = np.concatenate(preds).ravel()
    targets = np.concatenate(targets).ravel()

    rmse = mean_squared_error(targets, preds, squared=False)
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)

    metrics = {"RMSE": rmse, "MAE": mae, "R2": r2}
    metrics_path = REPORT_DIR / "test_metrics.json"
    report = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "git_commit": get_git_sha(),
        "metrics": metrics,
    }
    metrics_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Test metrics saved to {metrics_path}")
    logger.info(f"Test results: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
    return metrics


def get_git_sha() -> str:
    """Return current git commit SHA if repository is a git repo, else empty."""
    try:
        import subprocess

        sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        return sha
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# 11️⃣  Export deployment artefacts (metadata)
# --------------------------------------------------------------------------- #
def export_artifacts(pipeline_path: Path, model_ckpt_path: Path):
    deployment_dir = MODEL_DIR / "deployment"
    deployment_dir.mkdir(parents=True, exist_ok=True)

    # copy artefacts
    (deployment_dir / pipeline_path.name).write_bytes(pipeline_path.read_bytes())
    (deployment_dir / model_ckpt_path.name).write_bytes(model_ckpt_path.read_bytes())

    metadata = {
        "model_name": "torch_mlp_diabetes_v1",
        "framework": "pytorch",
        "framework_version": torch.__version__,
        "preprocess_version": "1.0",
        "training_date_utc": datetime.utcnow().isoformat() + "Z",
        "git_commit": get_git_sha(),
    }
    (deployment_dir / "metadata.yaml").write_text(
        "\n".join(f"{k}: {v}" for k, v in metadata.items())
    )
    logger.info(f"Exported deployment artefacts to {deployment_dir}")
    return deployment_dir


# --------------------------------------------------------------------------- #
# 12️⃣  Gradio demo (single‑record inference)
# --------------------------------------------------------------------------- #
def build_gradio_interface(pipeline: Pipeline, model: nn.Module):
    # Ensure model is in eval mode
    model.eval()

    # Feature order must match training order
    feature_names = [
        "age",
        "sex",
        "bmi",
        "bp",
        "s1",
        "s2",
        "s3",
        "s4",
        "s5",
        "s6",
    ]

    def predict(**kwargs):
        # 1) build DataFrame
        df = pd.DataFrame([kwargs], columns=feature_names)
        # 2) preprocess
        X = pipeline.transform(df).astype(np.float32)
        # 3) torch inference
        with torch.no_grad():
            tensor = torch.from_numpy(X).to(DEVICE)
            out = model(tensor).cpu().numpy().ravel()[0]
        return float(out)

    iface = gr.Interface(
        fn=predict,
        inputs=[gr.Number(label=name) for name in feature_names],
        outputs=gr.Number(label="Predicted target"),
        title="Diabetes Progression Regression",
        description="Enter the 10 clinical features to obtain the predicted disease progression value.",
        allow_flagging="never",
    )
    return iface


# --------------------------------------------------------------------------- #
# 13️⃣  Main orchestration
# --------------------------------------------------------------------------- #
def main():
    # ------------------------------------------------------------------- #
    # Step 0 – locate the raw CSV (adjust the path to your environment)
    # ------------------------------------------------------------------- #
    user_csv = Path(
        r"C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\diabetes.csv"
    )
    df = ingest(user_csv)

    # ------------------------------------------------------------------- #
    # Step 1 – split
    # ------------------------------------------------------------------- #
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(df)

    # ------------------------------------------------------------------- #
    # Step 2 – preprocessing
    # ------------------------------------------------------------------- #
    pipeline = fit_preprocess_pipeline(X_train)

    # ------------------------------------------------------------------- #
    # Step 3 – dataloaders
    # ------------------------------------------------------------------- #
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)

    # ------------------------------------------------------------------- #
    # Step 4 – model definition
    # ------------------------------------------------------------------- #
    model = build_model()

    # ------------------------------------------------------------------- #
    # Step 5 – training
    # ------------------------------------------------------------------- #
    model, best_ckpt_path = train(model, train_loader, val_loader)

    # ------------------------------------------------------------------- #
    # Step 6 – evaluation
    # ------------------------------------------------------------------- #
    test_metrics = evaluate(model, test_loader)

    # ------------------------------------------------------------------- #
    # Step 7 – export artefacts
    # ------------------------------------------------------------------- #
    export_artifacts(Path(MODEL_DIR / "preprocess_pipe.joblib"), best_ckpt_path)

    # ------------------------------------------------------------------- #
    # Step 8 – Gradio demo (optional – will launch a local UI)
    # ------------------------------------------------------------------- #
    iface = build_gradio_interface(pipeline, model)
    iface.launch(share=False)  # set share=True if you want a public URL

    return {
        "test_metrics": test_metrics,
        "model_checkpoint": str(best_ckpt_path),
        "preprocess_pipeline": str(MODEL_DIR / "preprocess_pipe.joblib"),
    }


if __name__ == "__main__":
    results = main()
    print("\n=== Final Results ===")
    print(json.dumps(results, indent=2))
