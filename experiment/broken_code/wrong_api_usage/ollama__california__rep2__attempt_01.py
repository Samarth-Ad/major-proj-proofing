
"""
California Housing – Tabular Regression with FT‑Transformer
================================================================
An end‑to‑end, self‑contained script that:

1️⃣ Loads the raw CSV (user‑provided path or default Windows location).  
2️⃣ Performs deterministic preprocessing: median imputation, log1p target,
   stratified split (70 % train / 20 % val / 10 % test), StandardScaler
   fitted on the training features only.  
3️⃣ Wraps the splits in a PyTorch ``Dataset`` / ``DataLoader``.  
4️⃣ Builds the FT‑Transformer model with the exact hyper‑parameters required.  
5️⃣ Trains with AdamW + CosineAnnealingLR, early‑stopping (patience = 10).  
6️⃣ Evaluates on the held‑out test set (RMSE, MAE, R²).  
7️⃣ Saves the best model, scaler and evaluation artefacts under
   ``./agent_workspace`` (the required ``trained_models`` sub‑folder is used).  
8️⃣ (Optional) launches a tiny Gradio demo for interactive inference.

The script can be executed directly:

    python ft_transformer_regression.py

All randomness is fixed (seed = 42) for reproducibility.
"""

import os
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# --------------------------------------------------------------
# 0️⃣ Global settings & reproducibility
# --------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")

# --------------------------------------------------------------
# 1️⃣ Paths
# --------------------------------------------------------------
# Default CSV location (can be overridden with env var CALIFORNIA_CSV)
RAW_CSV_PATH = os.getenv(
    "CALIFORNIA_CSV",
    r"C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\california.csv",
)

BASE_DIR = Path("./agent_workspace")
PROCESSED_DIR = BASE_DIR / "processed"
TRAINED_MODEL_DIR = BASE_DIR / "trained_models"
EVAL_DIR = BASE_DIR / "evaluation"

for d in (PROCESSED_DIR, TRAINED_MODEL_DIR, EVAL_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------
# 2️⃣ Data loading & sanity checks
# --------------------------------------------------------------
def load_and_check(csv_path: str) -> pd.DataFrame:
    """Load CSV and run basic sanity checks."""
    df = pd.read_csv(csv_path)
    assert df.shape[1] == 9, f"Expected 9 columns (8 features + target), got {df.shape[1]}"
    assert df.shape[0] >= 1_000, "Dataset should contain at least 1 000 rows"
    assert np.all(
        df.dtypes.apply(lambda dt: np.issubdtype(dt, np.number))
    ), "All columns must be numeric"
    return df


# --------------------------------------------------------------
# 3️⃣ Pre‑processing (deterministic)
# --------------------------------------------------------------
def preprocess(df: pd.DataFrame):
    """Impute, log‑transform target, stratified split, and scale."""
    # 3.1 Median imputation (including target – harmless)
    imputer = SimpleImputer(strategy="median")
    df_imputed = pd.DataFrame(
        imputer.fit_transform(df), columns=df.columns, index=df.index
    )

    # 3.2 Log‑transform the target (named "target")
    if "target" not in df_imputed.columns:
        raise KeyError("The dataset must contain a column named 'target'")
    df_imputed["target_log"] = np.log1p(df_imputed["target"])

    # 3.3 Stratified split on the *original* (non‑log) target
    df_imputed["target_bin"] = pd.qcut(
        df_imputed["target"], q=5, labels=False, duplicates="drop"
    )
    train_val, test = train_test_split(
        df_imputed,
        test_size=0.10,
        stratify=df_imputed["target_bin"],
        random_state=SEED,
    )
    train, val = train_test_split(
        train_val,
        test_size=0.20 / 0.90,  # 20 % of total → 20/90 of remaining
        stratify=train_val["target_bin"],
        random_state=SEED,
    )
    # Drop helper column (use non‑inplace to avoid deprecation warning)
    train = train.drop(columns=["target_bin"])
    val = val.drop(columns=["target_bin"])
    test = test.drop(columns=["target_bin"])

    # 3.4 Fit StandardScaler on training *features* only
    feature_cols = [c for c in df.columns if c != "target"]
    scaler = StandardScaler()
    scaler.fit(train[feature_cols])

    # Apply scaling to all splits (in‑place is fine here)
    for split in (train, val, test):
        split[feature_cols] = scaler.transform(split[feature_cols])

    # 3.5 Persist artefacts
    scaler_path = PROCESSED_DIR / "scaler.joblib"
    joblib.dump(scaler, scaler_path)

    for name, split in zip(["train", "val", "test"], [train, val, test]):
        split.to_csv(PROCESSED_DIR / f"{name}.csv", index=False)

    # 3.6 Write a tiny JSON summary
    summary = {
        "seed": SEED,
        "imputer": "median",
        "target_log_transform": True,
        "scaler": "StandardScaler",
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
    }
    with open(PROCESSED_DIR / "preprocessing_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return scaler_path, feature_cols


# --------------------------------------------------------------
# 4️⃣ PyTorch Dataset & DataLoaders
# --------------------------------------------------------------
class TabularDataset(Dataset):
    """Dataset returning (features, target_log) tensors."""

    def __init__(self, csv_path: Path, feature_cols):
        df = pd.read_csv(csv_path)
        self.X = torch.tensor(df[feature_cols].values, dtype=torch.float32)
        self.y = (
            torch.tensor(df["target_log"].values, dtype=torch.float32)
            .unsqueeze(1)
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(feature_cols, batch_size=64):
    """Create deterministic DataLoaders (num_workers=0 to avoid pickling issues)."""
    train_ds = TabularDataset(PROCESSED_DIR / "train.csv", feature_cols)
    val_ds = TabularDataset(PROCESSED_DIR / "val.csv", feature_cols)
    test_ds = TabularDataset(PROCESSED_DIR / "test.csv", feature_cols)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # deterministic on Windows
        pin_memory=DEVICE.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=DEVICE.type == "cuda",
    )
    return train_loader, val_loader, test_loader


# --------------------------------------------------------------
# 5️⃣ FT‑Transformer model (exact hyper‑parameters)
# --------------------------------------------------------------
class FTTransformer(nn.Module):
    """
    Minimal FT‑Transformer implementation matching the specification:
    - d_model = 32
    - n_heads = 4
    - n_layers = 4
    - ffn_dim = 128
    - dropout = 0.1
    - use_residual & batch‑norm enabled
    - No CLS token (mean‑pool over token embeddings)
    """

    def __init__(
        self,
        num_features: int,
        d_model: int = 32,
        n_heads: int = 4,
        n_layers: int = 4,
        ffn_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model

        # Linear “embedding” for each scalar feature (shared across tokens)
        self.feature_proj = nn.Linear(1, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.bn = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)

        # Regression head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x):
        """
        x : Tensor of shape (batch, num_features)
        """
        batch_size = x.size(0)

        # Expand each feature to (batch, num_features, 1) then project
        x = x.unsqueeze(-1)                     # (B, F, 1)
        x = self.feature_proj(x)                # (B, F, d_model)

        # Transformer encoder
        x = self.transformer_encoder(x)         # (B, F, d_model)

        # Residual + batch‑norm (as in the paper)
        x = self.bn(x.reshape(-1, self.d_model)).reshape(batch_size, -1, self.d_model)
        x = self.dropout(x)

        # Mean‑pool over token dimension
        x = x.mean(dim=1)                       # (B, d_model)

        out = self.head(x)                      # (B, 1)
        return out


def build_model(num_features: int):
    cfg = {
        "d_model": 32,
        "n_heads": 4,
        "n_layers": 4,
        "ffn_dim": 128,
        "dropout": 0.1,
    }
    model = FTTransformer(num_features=num_features, **cfg)
    return model, cfg


# --------------------------------------------------------------
# 6️⃣ Training utilities
# --------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    return epoch_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)
            loss_sum += loss.item() * xb.size(0)
            all_preds.append(preds.cpu())
            all_targets.append(yb.cpu())
    avg_loss = loss_sum / len(loader.dataset)
    preds = torch.cat(all_preds).squeeze().numpy()
    targets = torch.cat(all_targets).squeeze().numpy()
    rmse = mean_squared_error(targets, preds, squared=False)
    mae = mean_absolute_error(targets, preds)
    r2 = r2_score(targets, preds)
    return avg_loss, {"rmse": rmse, "mae": mae, "r2": r2}, preds, targets


# --------------------------------------------------------------
# 7️⃣ Main orchestration
# --------------------------------------------------------------
def main():
    # ------------------------------------------------------------------
    # 7.1 Load raw data & run preprocessing
    # ------------------------------------------------------------------
    raw_df = load_and_check(RAW_CSV_PATH)
    scaler_path, feature_cols = preprocess(raw_df)

    # ------------------------------------------------------------------
    # 7.2 DataLoaders
    # ------------------------------------------------------------------
    train_loader, val_loader, test_loader = get_dataloaders(feature_cols, batch_size=64)

    # ------------------------------------------------------------------
    # 7.3 Model, optimizer, scheduler, loss
    # ------------------------------------------------------------------
    model, model_cfg = build_model(num_features=len(feature_cols))
    model = model.to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    criterion = nn.MSELoss()

    # ------------------------------------------------------------------
    # 7.4 Training loop with early stopping
    # ------------------------------------------------------------------
    best_val_rmse = float("inf")
    patience = 10
    epochs_no_improve = 0
    max_epochs = 200
    start_time = time.time()
    best_checkpoint_path = TRAINED_MODEL_DIR / "ft_transformer_tmp.pt"

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_metrics, _, _ = evaluate(model, val_loader, criterion, DEVICE)
        scheduler.step()

        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.5f} | "
            f"val_loss={val_loss:.5f} | RMSE={val_metrics['rmse']:.4f}"
        )

        # Early‑stopping based on validation RMSE
        if val_metrics["rmse"] < best_val_rmse - 1e-6:
            best_val_rmse = val_metrics["rmse"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_checkpoint_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("[INFO] Early stopping triggered.")
                break

    train_time = time.time() - start_time

    # Load the best checkpoint
    model.load_state_dict(torch.load(best_checkpoint_path, map_location=DEVICE))
    final_model_path = TRAINED_MODEL_DIR / "ft_transformer_best.pt"
    torch.save(model.state_dict(), final_model_path)

    # ------------------------------------------------------------------
    # 7.5 Test‑set evaluation
    # ------------------------------------------------------------------
    _, test_metrics_log, test_preds_log, test_targets_log = evaluate(
        model, test_loader, criterion, DEVICE
    )
    # Inverse log1p to obtain original scale
    test_preds = np.expm1(test_preds_log)
    test_targets = np.expm1(test_targets_log)

    test_rmse = mean_squared_error(test_targets, test_preds, squared=False)
    test_mae = mean_absolute_error(test_targets, test_preds)
    test_r2 = r2_score(test_targets, test_preds)

    test_metrics = {"rmse": test_rmse, "mae": test_mae, "r2": test_r2}
    print("\n=== Test set performance ===")
    print(f"RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f} | R²: {test_r2:.4f}")

    # ------------------------------------------------------------------
    # 7.6 Persist evaluation artefacts
    # ------------------------------------------------------------------
    # Metrics JSON
    with open(EVAL_DIR / "metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=2)

    # Raw predictions CSV
    pred_df = pd.DataFrame(
        {
            "id": np.arange(len(test_preds)),
            "predicted_value": test_preds,
            "true_value": test_targets,
        }
    )
    pred_df.to_csv(EVAL_DIR / "test_predictions.csv", index=False)

    # Model profile JSON (fill‑in numbers that are easy to compute)
    num_params = sum(p.numel() for p in model.parameters())
    model_profile = {
        "model_name": "FT-Transformer",
        "framework": "pytorch",
        "hyperparameters": {
            **model_cfg,
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "batch_size": 64,
            "epochs_trained": epoch,
            "seed": SEED,
        },
        "train_time_seconds": round(train_time, 2),
        "num_parameters": num_params,
        "model_file_size_mb": round(final_model_path.stat().st_size / 1e6, 2),
        "validation_metrics": {"rmse": best_val_rmse},
        "test_metrics": test_metrics,
    }
    with open(EVAL_DIR / "model_profile.json", "w") as f:
        json.dump(model_profile, f, indent=2)

    # ------------------------------------------------------------------
    # 7.7 Optional Gradio demo (runs only if Gradio is installed)
    # ------------------------------------------------------------------
    try:
        import gradio as gr

        # Load scaler and model once for the demo
        scaler = joblib.load(scaler_path)
        demo_model = FTTransformer(num_features=len(feature_cols), **model_cfg).to(DEVICE)
        demo_model.load_state_dict(torch.load(final_model_path, map_location=DEVICE))
        demo_model.eval()

        def predict(*features):
            """Accept eight numeric inputs, return the predicted house value."""
            arr = np.array(features, dtype=np.float32).reshape(1, -1)
            arr = scaler.transform(arr)
            tensor = torch.from_numpy(arr).float().to(DEVICE)
            with torch.no_grad():
                log_pred = demo_model(tensor).cpu().numpy().squeeze()
            return np.expm1(log_pred).item()

        iface = gr.Interface(
            fn=predict,
            inputs=[gr.Number(label=col) for col in feature_cols],
            outputs=gr.Number(label="Predicted median house value"),
            title="California Housing – FT‑Transformer",
            description="Enter the eight numeric features to obtain a house‑price prediction.",
        )
        iface.launch(share=False, inline=False, debug=False)
    except Exception as e:
        print(f"[INFO] Gradio demo could not be started: {e}")

    # ------------------------------------------------------------------
    # 7.8 Return paths for downstream automation (optional)
    # ------------------------------------------------------------------
    return {
        "model_path": str(final_model_path),
        "scaler_path": str(scaler_path),
        "metrics_path": str(EVAL_DIR / "metrics.json"),
        "predictions_path": str(EVAL_DIR / "test_predictions.csv"),
    }


if __name__ == "__main__":
    artefacts = main()
    print("\nSaved artefacts:")
    for k, v in artefacts.items():
        print(f"{k}: {v}")
