
"""
End‑to‑end tabular regression pipeline for the Diabetes dataset
----------------------------------------------------------------
- Data ingestion & validation
- Train / validation / test split (70/20/10) with stratified bins
- Scikit‑learn preprocessing pipeline:
    median imputation → Yeo‑Johnson power transform (skewed cols) → Robust scaling
- PyTorch shallow MLP (10‑64‑1) trained with early stopping
- Evaluation (RMSE, MAE, R²) on the held‑out test set
- Export of:
    * fitted preprocessing pipeline (joblib)
    * best model checkpoint (torch state_dict)
    * test metrics (json)
All artefacts are stored under ``./agent_workspace/trained_models`` so that they
can be picked up by downstream deployment scripts (e.g. FastAPI, Gradio, etc.).
"""

import os
import json
import random
import hashlib
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from loguru import logger

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import skew

import joblib
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------------------------------------------------
# Global configuration & reproducibility
# ----------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")  # dataset is tiny; CPU is sufficient

# ----------------------------------------------------------------------
# Paths (all relative to the repository root)
# ----------------------------------------------------------------------
ROOT = Path.cwd()
RAW_CSV_SRC = (
    Path(r"C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\diabetes.csv")
)  # original location supplied by the user
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "agent_workspace" / "trained_models"
REPORTS_DIR = ROOT / "reports"

# Ensure directories exist
for d in [RAW_DIR, PROCESSED_DIR, MODEL_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ----------------------------------------------------------------------
# 1️⃣ Data ingestion & validation
# ----------------------------------------------------------------------
def ingest_and_validate(csv_src: Path, dest_dir: Path) -> pd.DataFrame:
    """Copy raw CSV, validate shape & missingness, write schema.json."""
    dest_path = dest_dir / "Diabetes_raw.csv"
    # copy (overwrites if exists)
    dest_path.write_bytes(csv_src.read_bytes())
    logger.info(f"Copied raw CSV to {dest_path}")

    df = pd.read_csv(dest_path)
    expected_shape = (442, 11)  # 10 features + target
    if df.shape != expected_shape:
        logger.warning(
            f"Unexpected shape {df.shape}; expected {expected_shape}. Continuing anyway."
        )
    # missing‑value percentages
    miss_pct = df.isna().mean() * 100
    for col, pct in miss_pct.items():
        if pct > 5.0:
            logger.warning(f"Column '{col}' has {pct:.2f}% missing values (>5%).")

    # schema.json
    schema = {
        col: {"dtype": str(df[col].dtype), "missing_pct": float(pct)}
        for col, pct in miss_pct.items()
    }
    schema_path = REPORTS_DIR / "schema.json"
    schema_path.write_text(json.dumps(schema, indent=2))
    logger.info(f"Wrote schema to {schema_path}")

    return df


# ----------------------------------------------------------------------
# 2️⃣ Train / validation / test split (stratified on target bins)
# ----------------------------------------------------------------------
def split_data(df: pd.DataFrame):
    X = df.drop(columns=["target"]).values.astype(np.float32)
    y = df["target"].values.astype(np.float32)

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
    # split temp into val (2/3 of temp) and test (1/3 of temp)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.3333,
        stratify=bins_temp,
        random_state=SEED,
    )

    # persist as .npy
    np.save(PROCESSED_DIR / "X_train.npy", X_train)
    np.save(PROCESSED_DIR / "y_train.npy", y_train)
    np.save(PROCESSED_DIR / "X_val.npy", X_val)
    np.save(PROCESSED_DIR / "y_val.npy", y_val)
    np.save(PROCESSED_DIR / "X_test.npy", X_test)
    np.save(PROCESSED_DIR / "y_test.npy", y_test)

    logger.info(
        f"Split data -> train:{X_train.shape[0]}, val:{X_val.shape[0]}, test:{X_test.shape[0]}"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ----------------------------------------------------------------------
# 3️⃣ Pre‑processing pipeline (fit on training data only)
# ----------------------------------------------------------------------
def build_and_apply_preprocess():
    """Fit pipeline on X_train, transform all splits, and persist the pipeline."""
    X_train = np.load(PROCESSED_DIR / "X_train.npy")
    X_val = np.load(PROCESSED_DIR / "X_val.npy")
    X_test = np.load(PROCESSED_DIR / "X_test.npy")

    # Identify skewed columns on training data
    skewness = pd.DataFrame(X_train).apply(lambda col: skew(col))
    skewed_idx = np.where(np.abs(skewness) > 1.0)[0]
    logger.info(f"Skewed column indices (|skew|>1.0): {skewed_idx.tolist()}")

    # ColumnTransformer
    preprocess = ColumnTransformer(
        transformers=[
            ("imputer", SimpleImputer(strategy="median"), slice(0, X_train.shape[1])),
            ("power", PowerTransformer(method="yeo-johnson"), skewed_idx),
            ("scaler", RobustScaler(), slice(0, X_train.shape[1])),
        ],
        remainder="passthrough",  # just in case
    )

    pipeline = Pipeline(steps=[("preprocess", preprocess)])

    # Fit on training data
    pipeline.fit(X_train)

    # Transform all splits
    X_train_t = pipeline.transform(X_train)
    X_val_t = pipeline.transform(X_val)
    X_test_t = pipeline.transform(X_test)

    # Overwrite the .npy files with transformed data
    np.save(PROCESSED_DIR / "X_train.npy", X_train_t)
    np.save(PROCESSED_DIR / "X_val.npy", X_val_t)
    np.save(PROCESSED_DIR / "X_test.npy", X_test_t)

    # Persist pipeline
    pipeline_path = MODEL_DIR / "preprocess_pipe.joblib"
    joblib.dump(pipeline, pipeline_path)
    logger.info(f"Saved preprocessing pipeline to {pipeline_path}")

    return pipeline_path


# ----------------------------------------------------------------------
# 4️⃣ PyTorch Dataset & DataLoader helpers
# ----------------------------------------------------------------------
class DiabetesDataset(torch.utils.data.Dataset):
    def __init__(self, X_path: Path, y_path: Path):
        self.X = torch.from_numpy(np.load(X_path)).float()
        self.y = torch.from_numpy(np.load(y_path)).float().unsqueeze(1)  # (N,1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_loaders(batch_size: int = 32):
    train_ds = DiabetesDataset(PROCESSED_DIR / "X_train.npy", PROCESSED_DIR / "y_train.npy")
    val_ds = DiabetesDataset(PROCESSED_DIR / "X_val.npy", PROCESSED_DIR / "y_val.npy")
    test_ds = DiabetesDataset(PROCESSED_DIR / "X_test.npy", PROCESSED_DIR / "y_test.npy")

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader, test_loader


# ----------------------------------------------------------------------
# 5️⃣ Model definition (single hidden‑layer MLP)
# ----------------------------------------------------------------------
class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int = 10, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Kaiming init for hidden layer
        nn.init.kaiming_uniform_(self.net[0].weight, nonlinearity="relu")
        nn.init.zeros_(self.net[0].bias)
        nn.init.kaiming_uniform_(self.net[2].weight, nonlinearity="linear")
        nn.init.zeros_(self.net[2].bias)

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------
# 6️⃣ Training loop with early stopping & LR scheduler
# ----------------------------------------------------------------------
def train(
    model: nn.Module,
    train_loader,
    val_loader,
    max_epochs: int = 200,
    patience: int = 15,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    best_val_rmse = float("inf")
    epochs_no_improve = 0
    best_state_path = MODEL_DIR / "torch_mlp_best.pt"

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

        # Validation
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                preds = model(xb).cpu()
                val_preds.append(preds.numpy())
                val_targets.append(yb.numpy())
        val_preds = np.concatenate(val_preds).ravel()
        val_targets = np.concatenate(val_targets).ravel()
        val_rmse = mean_squared_error(val_targets, val_preds, squared=False)
        val_mae = mean_absolute_error(val_targets, val_preds)
        val_r2 = r2_score(val_targets, val_preds)

        scheduler.step(val_rmse)

        logger.info(
            f"Epoch {epoch:03d} | train_loss={np.mean(train_losses):.4f} | "
            f"val_RMSE={val_rmse:.3f} | val_MAE={val_mae:.3f} | val_R2={val_r2:.3f}"
        )

        # Early‑stopping check
        if val_rmse < best_val_rmse - 1e-4:
            best_val_rmse = val_rmse
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_state_path)
            logger.info(f"  --> New best model saved (RMSE={best_val_rmse:.3f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(
                    f"No improvement for {patience} epochs. Stopping early at epoch {epoch}."
                )
                break

    # Load best weights before returning
    model.load_state_dict(torch.load(best_state_path, map_location=DEVICE))
    return model, best_state_path


# ----------------------------------------------------------------------
# 7️⃣ Evaluation on test set
# ----------------------------------------------------------------------
def evaluate(model: nn.Module, test_loader):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            out = model(xb).cpu()
            preds.append(out.numpy())
            targets.append(yb.numpy())
    preds = np.concatenate(preds).ravel()
    targets = np.concatenate(targets).ravel()

    rmse = mean_squared_error(targets, preds, squared=False)
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)

    metrics = {"RMSE": rmse, "MAE": mae, "R2": r2}
    logger.info(f"Test set metrics: {metrics}")

    # Save metrics JSON
    metrics_path = REPORTS_DIR / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "metrics": metrics,
                "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
            },
            f,
            indent=2,
        )
    logger.info(f"Wrote test metrics to {metrics_path}")

    return metrics


# ----------------------------------------------------------------------
# 8️⃣ Model & pipeline export (already saved during training)
# ----------------------------------------------------------------------
def export_artifacts(model_path: Path, pipeline_path: Path):
    """Create a deployment folder with model, pipeline and metadata."""
    deployment_dir = MODEL_DIR / "deployment"
    deployment_dir.mkdir(parents=True, exist_ok=True)

    # copy artefacts
    (deployment_dir / "model.pt").write_bytes(model_path.read_bytes())
    (deployment_dir / "preprocess_pipe.joblib").write_bytes(pipeline_path.read_bytes())

    # metadata.yaml
    metadata = {
        "model_name": "torch_mlp_diabetes_v1",
        "framework": "pytorch",
        "framework_version": torch.__version__,
        "preprocess_version": "1.0",
        "training_date": datetime.datetime.utcnow().isoformat() + "Z",
        "git_commit": os.popen("git rev-parse HEAD").read().strip(),
    }
    metadata_path = deployment_dir / "metadata.yaml"
    metadata_path.write_text(
        "\n".join([f"{k}: {v}" for k, v in metadata.items()]), encoding="utf-8"
    )
    logger.info(f"Exported deployment artefacts to {deployment_dir}")


# ----------------------------------------------------------------------
# 9️⃣ Orchestration (main)
# ----------------------------------------------------------------------
def main():
    # 1️⃣ Ingest & validate
    df = ingest_and_validate(RAW_CSV_SRC, RAW_DIR)

    # 2️⃣ Split
    split_data(df)  # creates .npy files

    # 3️⃣ Preprocess
    pipeline_path = build_and_apply_preprocess()

    # 4️⃣ DataLoaders
    train_loader, val_loader, test_loader = get_loaders(batch_size=32)

    # 5️⃣ Model
    model = MLPRegressor(input_dim=10, hidden_dim=64).to(DEVICE)

    # 6️⃣ Train
    model, best_model_path = train(
        model,
        train_loader,
        val_loader,
        max_epochs=200,
        patience=15,
        lr=1e-3,
        weight_decay=1e-4,
    )

    # 7️⃣ Evaluate
    test_metrics = evaluate(model, test_loader)

    # 8️⃣ Export artefacts for deployment
    export_artifacts(best_model_path, pipeline_path)

    logger.success("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
