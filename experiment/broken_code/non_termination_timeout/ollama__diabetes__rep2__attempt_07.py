
# -*- coding: utf-8 -*-
"""
End‑to‑end tabular regression pipeline for the Diabetes dataset.

Features
--------
1. Ingest & validate the raw CSV.
2. Stratified 70/20/10 train‑val‑test split.
3. Pre‑processing pipeline (median imputer → Yeo‑Johnson power transform → Robust scaler).
4. Shallow PyTorch MLP (10‑64‑1) with Kaiming‑Uniform initialization.
5. Training loop with Adam, ReduceLROnPlateau (no verbose), early‑stopping.
6. Evaluation (RMSE, MAE, R²) and JSON report.
7. Export artefacts (model checkpoint, preprocessing pipeline, metadata) to
   ``./agent_workspace/trained_models``.
8. Optional Gradio UI for single‑record inference.
"""

# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
import os
import json
import hashlib
import logging
import random
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Gradio is optional – import only if available
try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except Exception:  # pragma: no cover
    GRADIO_AVAILABLE = False

# --------------------------------------------------------------------------- #
# Global configuration & reproducibility
# --------------------------------------------------------------------------- #
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

DEVICE = torch.device("cpu")  # CPU‑only (tiny dataset)

# --------------------------------------------------------------------------- #
# Folder layout (all paths are absolute, based on the current working dir)
# --------------------------------------------------------------------------- #
ROOT = Path.cwd()

RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
TRAINED_MODEL_DIR = ROOT / "agent_workspace" / "trained_models"
REPORT_DIR = ROOT / "reports"
LOG_DIR = ROOT / "logs"

for d in (RAW_DIR, PROC_DIR, MODEL_DIR, TRAINED_MODEL_DIR, REPORT_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Logging (standard library – no extra dependency)
# --------------------------------------------------------------------------- #
log_file = LOG_DIR / "pipeline.log"
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
# Helper utilities
# --------------------------------------------------------------------------- #
def md5_checksum(path: Path) -> str:
    """Return the MD5 checksum of a file."""
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def get_git_sha() -> str:
    """Return current git commit SHA if inside a git repo, else empty string."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        return sha
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
# 1️⃣  Ingestion & validation
# --------------------------------------------------------------------------- #
def ingest(csv_path: Path) -> pd.DataFrame:
    logger.info("=== Ingestion start ===")
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    dest = RAW_DIR / "Diabetes_raw.csv"
    shutil.copyfile(csv_path, dest)
    logger.info(f"Copied raw CSV to {dest}")

    df = pd.read_csv(dest)
    expected_cols = 11  # 10 features + target
    if df.shape[1] != expected_cols:
        raise ValueError(f"Expected {expected_cols} columns, got {df.shape[1]}")
    if (df.isna().mean() > 0.05).any():
        raise ValueError("One or more columns have >5% missing values")
    logger.info(f"Dataset shape after load: {df.shape}")

    # schema.json
    schema = {
        col: {"dtype": str(df[col].dtype), "missing_pct": float(df[col].isna().mean())}
        for col in df.columns
    }
    (REPORT_DIR / "schema.json").write_text(json.dumps(schema, indent=2))
    logger.info("Wrote schema.json")

    checksum = md5_checksum(dest)
    logger.info(f"MD5 checksum of raw file: {checksum}")
    logger.info("=== Ingestion finished ===")
    return df


# --------------------------------------------------------------------------- #
# 2️⃣  Train/validation/test split (stratified on target bins)
# --------------------------------------------------------------------------- #
def split_data(df: pd.DataFrame):
    logger.info("=== Splitting data ===")
    X = df.drop(columns=["target"])
    y = df["target"]

    # 5‑quantile stratification bins
    bins = pd.qcut(y, q=5, labels=False, duplicates="drop")

    X_train, X_temp, y_train, y_temp, bins_train, bins_temp = train_test_split(
        X,
        y,
        bins,
        test_size=0.30,
        stratify=bins,
        random_state=SEED,
    )
    # temp → val (2/3) & test (1/3)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=1 / 3,
        stratify=bins_temp,
        random_state=SEED,
    )

    # Save splits as .npy (preserve column order via a side‑car JSON)
    for split_name, X_split, y_split in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        X_path = PROC_DIR / f"X_{split_name}.npy"
        y_path = PROC_DIR / f"y_{split_name}.npy"
        np.save(X_path, X_split.to_numpy().astype(np.float32))
        np.save(y_path, y_split.to_numpy().astype(np.float32))

        meta = {"columns": list(X_split.columns)}
        (PROC_DIR / f"{split_name}_meta.json").write_text(json.dumps(meta))
        logger.info(f"Saved {split_name} split: {X_path}, {y_path}")

    logger.info(
        f"Split counts → train: {len(X_train)}, val: {len(X_val)}, test: {len(X_test)}"
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# --------------------------------------------------------------------------- #
# 3️⃣  Pre‑processing pipeline (single linear pipeline)
# --------------------------------------------------------------------------- #
def fit_preprocess_pipeline(X_train: pd.DataFrame) -> Pipeline:
    logger.info("=== Fitting preprocessing pipeline ===")
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("power", PowerTransformer(method="yeo-johnson")),
            ("scaler", RobustScaler()),
        ]
    )
    pipeline.fit(X_train)

    pipe_path = MODEL_DIR / "preprocess_pipe.joblib"
    joblib.dump(pipeline, pipe_path)
    logger.info(f"Saved pipeline to {pipe_path}")

    # Transform all splits and overwrite the .npy files
    for split in ("train", "val", "test"):
        meta_path = PROC_DIR / f"{split}_meta.json"
        cols = json.loads(meta_path.read_text())["columns"]
        X_path = PROC_DIR / f"X_{split}.npy"
        X_arr = np.load(X_path)  # (n_samples, n_features)
        X_df = pd.DataFrame(X_arr, columns=cols)
        X_trans = pipeline.transform(X_df).astype(np.float32)
        np.save(X_path, X_trans)
        logger.info(f"Transformed and overwrote {X_path}")

    return pipeline


# --------------------------------------------------------------------------- #
# 4️⃣  PyTorch Dataset wrapper
# --------------------------------------------------------------------------- #
class DiabetesDataset(Dataset):
    def __init__(self, split: str):
        """
        split: one of 'train', 'val', 'test'
        """
        X_path = PROC_DIR / f"X_{split}.npy"
        y_path = PROC_DIR / f"y_{split}.npy"

        self.X = torch.from_numpy(np.load(X_path)).float()
        self.y = torch.from_numpy(np.load(y_path)).unsqueeze(1).float()  # (N,1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(batch_size: int = 32):
    train_ds = DiabetesDataset("train")
    val_ds = DiabetesDataset("val")
    test_ds = DiabetesDataset("test")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


# --------------------------------------------------------------------------- #
# 5️⃣  Model definition
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


def build_model() -> nn.Module:
    model = MLPRegressor().to(DEVICE)
    return model


# --------------------------------------------------------------------------- #
# 6️⃣  Training loop with early stopping & LR scheduler
# --------------------------------------------------------------------------- #
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int = 200,
    patience: int = 15,
):
    logger.info("=== Training start ===")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )  # no verbose argument

    best_rmse = float("inf")
    epochs_no_improve = 0
    best_ckpt_path = TRAINED_MODEL_DIR / "torch_mlp_best.pt"

    for epoch in range(1, max_epochs + 1):
        # ---------- training ----------
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

        # ---------- validation ----------
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                out = model(xb).cpu().numpy()
                val_preds.append(out)
                val_targets.append(yb.numpy())
        val_preds = np.concatenate(val_preds).ravel()
        val_targets = np.concatenate(val_targets).ravel()

        mse = mean_squared_error(val_targets, val_preds)
        val_rmse = np.sqrt(mse)
        val_mae = mean_absolute_error(val_targets, val_preds)
        val_r2 = r2_score(val_targets, val_preds)

        scheduler.step(val_rmse)

        logger.info(
            f"Epoch {epoch:03d} | TrainLoss {avg_train_loss:.5f} | "
            f"Val RMSE {val_rmse:.5f} | MAE {val_mae:.5f} | R2 {val_r2:.5f}"
        )

        # ---------- checkpoint ----------
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model.state_dict(), best_ckpt_path)
            logger.info(f"New best model saved (RMSE={best_rmse:.5f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # ---------- early stopping ----------
        if epochs_no_improve >= patience:
            logger.info(
                f"Early stopping triggered after {epoch} epochs "
                f"(no improvement for {patience} epochs)."
            )
            break

    # Load best checkpoint before returning
    model.load_state_dict(torch.load(best_ckpt_path, map_location=DEVICE))
    logger.info(f"Training finished. Best checkpoint → {best_ckpt_path}")
    return model, best_ckpt_path


# --------------------------------------------------------------------------- #
# 7️⃣  Evaluation on test set
# --------------------------------------------------------------------------- #
def evaluate(model: nn.Module, test_loader: DataLoader):
    logger.info("=== Evaluation on test set ===")
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

    mse = mean_squared_error(targets, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)

    metrics = {"RMSE": rmse, "MAE": mae, "R2": r2}
    report = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "git_commit": get_git_sha(),
        "metrics": metrics,
    }
    (REPORT_DIR / "test_metrics.json").write_text(json.dumps(report, indent=2))
    logger.info(f"Test metrics saved to {REPORT_DIR / 'test_metrics.json'}")
    logger.info(f"Test results → RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    return metrics


# --------------------------------------------------------------------------- #
# 8️⃣  Export deployment artefacts (pipeline, model checkpoint, metadata)
# --------------------------------------------------------------------------- #
def export_artifacts(pipeline_path: Path, model_ckpt_path: Path):
    logger.info("=== Exporting deployment artefacts ===")
    deployment_dir = TRAINED_MODEL_DIR / "deployment"
    deployment_dir.mkdir(parents=True, exist_ok=True)

    # copy artefacts
    shutil.copy2(pipeline_path, deployment_dir / pipeline_path.name)
    shutil.copy2(model_ckpt_path, deployment_dir / model_ckpt_path.name)

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
    logger.info(f"Deployment artefacts written to {deployment_dir}")
    return deployment_dir


# --------------------------------------------------------------------------- #
# 9️⃣  Gradio demo (optional)
# --------------------------------------------------------------------------- #
def build_gradio_interface(pipeline: Pipeline, model: nn.Module):
    """Create a Gradio Interface for single‑record inference."""
    if not GRADIO_AVAILABLE:
        raise RuntimeError("Gradio is not installed; cannot build UI.")

    model.eval()
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
        df = pd.DataFrame([kwargs], columns=feature_names)
        X = pipeline.transform(df).astype(np.float32)
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
    )
    return iface


# --------------------------------------------------------------------------- #
# Main orchestration
# --------------------------------------------------------------------------- #
def main():
    # ------------------------------------------------------------------- #
    # 0️⃣  Locate the raw CSV (adjust if your CSV lives elsewhere)
    # ------------------------------------------------------------------- #
    user_csv = Path(
        r"C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\diabetes.csv"
    )
    df = ingest(user_csv)

    # ------------------------------------------------------------------- #
    # 1️⃣  Train/val/test split
    # ------------------------------------------------------------------- #
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(df)

    # ------------------------------------------------------------------- #
    # 2️⃣  Pre‑processing pipeline
    # ------------------------------------------------------------------- #
    pipeline = fit_preprocess_pipeline(X_train)

    # ------------------------------------------------------------------- #
    # 3️⃣  DataLoaders
    # ------------------------------------------------------------------- #
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)

    # ------------------------------------------------------------------- #
    # 4️⃣  Model definition
    # ------------------------------------------------------------------- #
    model = build_model()

    # ------------------------------------------------------------------- #
    # 5️⃣  Training
    # ------------------------------------------------------------------- #
    model, best_ckpt_path = train(model, train_loader, val_loader)

    # ------------------------------------------------------------------- #
    # 6️⃣  Evaluation
    # ------------------------------------------------------------------- #
    test_metrics = evaluate(model, test_loader)

    # ------------------------------------------------------------------- #
    # 7️⃣  Export artefacts
    # ------------------------------------------------------------------- #
    export_artifacts(
        pipeline_path=MODEL_DIR / "preprocess_pipe.joblib",
        model_ckpt_path=best_ckpt_path,
    )

    # ------------------------------------------------------------------- #
    # 8️⃣  Gradio demo (optional – comment out if you do not need a UI)
    # ------------------------------------------------------------------- #
    if GRADIO_AVAILABLE:
        iface = build_gradio_interface(pipeline, model)
        iface.launch(share=False)  # set share=True for a public URL if desired

    # ------------------------------------------------------------------- #
    # Return a concise summary
    # ------------------------------------------------------------------- #
    return {
        "test_metrics": test_metrics,
        "model_checkpoint": str(best_ckpt_path),
        "preprocess_pipeline": str(MODEL_DIR / "preprocess_pipe.joblib"),
    }


if __name__ == "__main__":
    results = main()
    print("\n=== Final Results ===")
    print(json.dumps(results, indent=2))
