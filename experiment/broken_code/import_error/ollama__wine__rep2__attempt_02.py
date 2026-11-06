
"""
Wine DEC clustering pipeline (self‑contained)

This script follows the executive‑summary design and implements the full
end‑to‑end flow:

1️⃣  Copy the user‑provided CSV to the project data folder and validate it.  
2️⃣  Create a stratified train/val/test split (70/20/10).  
3️⃣  Build a scikit‑learn preprocessing Pipeline (SimpleImputer + StandardScaler)  
    and persist it with joblib.  
4️⃣  Train a Deep‑Embedded Clustering (DEC) model using PyTorch‑Lightning:  
    – pre‑training auto‑encoder (reconstruction loss)  
    – clustering phase (KL‑divergence loss) with K‑Means centroid init.  
5️⃣  Evaluate the final model on the whole data set (ARI, AMI, Silhouette).  
6️⃣  Save the best checkpoint and the preprocessing pipeline under
    ``./agent_workspace/trained_models``.  
7️⃣  (Optional) expose a tiny Gradio demo for inference.

The script only relies on the packages listed in the provided
``environment.yml``.  It does **not** require ``pyarrow`` – CSV files are used
instead of Parquet to avoid the missing‑dependency error.

Run from the project root:

    python wine_dec_pipeline.py
"""

# --------------------------------------------------------------------------- #
# 1️⃣ Imports & global configuration
# --------------------------------------------------------------------------- #
import hashlib
import json
import os
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    silhouette_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
# 2️⃣ Configuration (mirrors the YAML from the spec)
# --------------------------------------------------------------------------- #
CFG = {
    "paths": {
        # absolute source supplied by the user
        "raw_src": r"C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\wine.csv",
        # internal project layout
        "raw_dst": "data/raw/Wine.csv",
        "splits_dir": "data/splits",
        "manifest": "outputs/metadata/data_manifest.json",
        "pipeline_out": "outputs/models/preprocess_pipe.pkl",
        "ckpt_out": "outputs/models/dec_best.ckpt",
        "metrics_out": "outputs/metrics/dec_metrics.json",
        # final artefacts required by the problem statement
        "trained_models_dir": "./agent_workspace/trained_models",
    },
    "preprocess": {
        "impute_strategy": "median",
        "scale_method": "standard",
        "random_state": 42,
    },
    "model": {
        "latent_dim": 10,
        "hidden_dims": [64, 32],
        "pretrain_epochs": 30,
        "cluster_epochs": 80,
        "batch_size": 32,
        "learning_rate": 5e-4,
        "optimizer": "adam",
        "weight_decay": 0.0,
        "update_interval": 10,
        "tol": 1e-4,
        "random_state": 42,
        "n_clusters": 3,
    },
    "split": {
        "train_frac": 0.70,
        "val_frac": 0.20,
        "test_frac": 0.10,
        "stratify": True,
    },
}


# --------------------------------------------------------------------------- #
# 3️⃣ Utility functions
# --------------------------------------------------------------------------- #
def ensure_dirs() -> None:
    """Create all required directories (idempotent)."""
    for p in [
        "data/raw",
        "data/splits",
        "outputs/models",
        "outputs/metrics",
        "outputs/metadata",
        "outputs/logs",
        CFG["paths"]["trained_models_dir"],
    ]:
        Path(p).mkdir(parents=True, exist_ok=True)


def copy_raw_csv() -> None:
    src = Path(CFG["paths"]["raw_src"])
    dst = Path(CFG["paths"]["raw_dst"])
    if not src.is_file():
        raise FileNotFoundError(f"Source CSV not found at {src}")
    shutil.copy2(src, dst)
    print(f"[INFO] Copied raw CSV to {dst}")


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def validate_dataset(df: pd.DataFrame) -> None:
    """Raise if any of the required checks fail."""
    expected_shape = (178, 14)  # 13 features + target
    if df.shape != expected_shape:
        raise ValueError(f"Unexpected shape {df.shape}, expected {expected_shape}")

    if "target" not in df.columns:
        raise ValueError("Column 'target' missing")

    if not set(df["target"].unique()).issubset({0, 1, 2}):
        raise ValueError("Target column must contain only {0,1,2}")

    if df.duplicated().any():
        raise ValueError("Duplicate rows detected")

    if df.isnull().sum().sum() != 0:
        raise ValueError("Missing values detected")

    print("[INFO] Dataset validation passed.")


def save_manifest(df: pd.DataFrame) -> None:
    manifest = {
        "sha256": compute_sha256(Path(CFG["paths"]["raw_dst"])),
        "columns": {c: str(dt) for c, dt in df.dtypes.items()},
        "row_count": df.shape[0],
    }
    Path(CFG["paths"]["manifest"]).parent.mkdir(parents=True, exist_ok=True)
    with open(CFG["paths"]["manifest"], "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"[INFO] Data manifest saved to {CFG['paths']['manifest']}")


def stratified_split(df: pd.DataFrame) -> dict:
    """Create train/val/test splits and store them as CSV files."""
    X = df.drop(columns=["target"]).values
    y = df["target"].values
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=CFG["split"]["val_frac"] + CFG["split"]["test_frac"],
        random_state=CFG["preprocess"]["random_state"],
    )
    train_idx, temp_idx = next(sss.split(X, y))

    # Relative proportion of validation within the temporary set
    val_frac_rel = CFG["split"]["val_frac"] / (
        CFG["split"]["val_frac"] + CFG["split"]["test_frac"]
    )
    sss_val = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1 - val_frac_rel,
        random_state=CFG["preprocess"]["random_state"],
    )
    val_idx, test_idx = next(sss_val.split(X[temp_idx], y[temp_idx]))
    val_idx = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]

    splits = {
        "train": df.iloc[train_idx].reset_index(drop=True),
        "val": df.iloc[val_idx].reset_index(drop=True),
        "test": df.iloc[test_idx].reset_index(drop=True),
    }

    for name, split_df in splits.items():
        out_path = Path(CFG["paths"]["splits_dir"]) / f"{name}.csv"
        split_df.to_csv(out_path, index=False)
        print(f"[INFO] Saved {name} split to {out_path}")

    return splits


def build_preprocess_pipeline() -> Pipeline:
    steps = [
        ("imputer", SimpleImputer(strategy=CFG["preprocess"]["impute_strategy"])),
    ]
    if CFG["preprocess"]["scale_method"] == "standard":
        steps.append(("scaler", StandardScaler()))
    else:
        raise ValueError("Only 'standard' scaling is supported in this script")
    return Pipeline(steps)


# --------------------------------------------------------------------------- #
# 4️⃣ DEC model (PyTorch‑Lightning)
# --------------------------------------------------------------------------- #
class DEC(pl.LightningModule):
    """Deep Embedded Clustering (DEC) implementation."""

    def __init__(self, n_features: int, cfg: dict):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg
        hidden_dims = cfg["hidden_dims"]
        latent_dim = cfg["latent_dim"]

        # ---------- Encoder ----------
        encoder_layers = []
        in_dim = n_features
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # ---------- Decoder (mirror) ----------
        decoder_layers = []
        rev_hidden = list(reversed(hidden_dims))
        in_dim = latent_dim
        for h in rev_hidden:
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, n_features))
        self.decoder = nn.Sequential(*decoder_layers)

        # ---------- Clustering layer ----------
        self.n_clusters = cfg["n_clusters"]
        self.cluster_layer = nn.Parameter(
            torch.randn(self.n_clusters, latent_dim)
        )  # will be re‑initialized after pre‑training

        # ---------- Misc ----------
        self.update_interval = cfg["update_interval"]
        self.tol = cfg["tol"]
        self.p = None  # target distribution (updated periodically)

    # ------------------------------------------------------------------- #
    # Helper methods
    # ------------------------------------------------------------------- #
    def forward(self, x):
        """Return reconstruction and latent embedding."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x):
        return self.encoder(x)

    def compute_q(self, z):
        """Student‑t distribution (α=1) soft assignments."""
        # z: (batch, latent_dim); centroids: (n_clusters, latent_dim)
        expanded_z = z.unsqueeze(1)          # (batch, 1, latent)
        expanded_c = self.cluster_layer.unsqueeze(0)  # (1, n_clusters, latent)
        dist = torch.sum((expanded_z - expanded_c) ** 2, dim=2)  # (batch, n_clusters)
        q = 1.0 / (1.0 + dist)
        q = q ** ((1 + 1) / 2)  # α=1
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

    @staticmethod
    def target_distribution(q):
        """Compute the target distribution p from soft assignments q."""
        weight = q ** 2 / torch.sum(q, dim=0, keepdim=True)
        p = weight / torch.sum(weight, dim=1, keepdim=True)
        return p

    # ------------------------------------------------------------------- #
    # Training step
    # ------------------------------------------------------------------- #
    def training_step(self, batch, batch_idx):
        x = batch  # shape: (batch, n_features)

        # ---- Phase 1: pre‑training (reconstruction) ----
        if self.current_epoch < self.cfg["pretrain_epochs"]:
            x_hat, _ = self.forward(x)
            loss = F.mse_loss(x_hat, x)
            self.log("pretrain_loss", loss, prog_bar=True)
            return loss

        # ---- Phase 2: clustering (KL divergence) ----
        _, z = self.forward(x)
        q = self.compute_q(z)

        if self.p is None:
            # First call – initialise p with current q
            self.p = q.detach()

        loss = F.kl_div(q.log(), self.p, reduction="batchmean")
        self.log("cluster_loss", loss, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        """Update target distribution every `update_interval` epochs."""
        if self.current_epoch < self.cfg["pretrain_epochs"]:
            return

        if (self.current_epoch - self.cfg["pretrain_epochs"]) % self.update_interval == 0:
            self.eval()
            all_z = []
            with torch.no_grad():
                for batch in self.trainer.datamodule.train_dataloader():
                    batch = batch.to(self.device)
                    _, z = self.forward(batch)
                    all_z.append(z)
            z_all = torch.cat(all_z, dim=0)
            q_all = self.compute_q(z_all)
            self.p = self.target_distribution(q_all).detach()
            self.train()

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.cfg["learning_rate"],
            weight_decay=self.cfg["weight_decay"],
        )

    # ------------------------------------------------------------------- #
    # Post‑training utilities
    # ------------------------------------------------------------------- #
    def init_cluster_centroids(self, dataloader):
        """Run K‑Means on latent embeddings after pre‑training."""
        self.eval()
        all_z = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(self.device)
                z = self.encode(batch)
                all_z.append(z.cpu().numpy())
        z_all = np.concatenate(all_z, axis=0)
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=20,
            random_state=self.cfg["random_state"],
        )
        kmeans.fit(z_all)
        centroids = torch.tensor(
            kmeans.cluster_centers_, dtype=torch.float32, device=self.device
        )
        self.cluster_layer.data = centroids
        print("[INFO] Cluster centroids initialized with K‑Means.")


# --------------------------------------------------------------------------- #
# 5️⃣ LightningDataModule (handles train/val splits)
# --------------------------------------------------------------------------- #
class WineDataModule(pl.LightningDataModule):
    def __init__(self, X_train: np.ndarray, X_val: np.ndarray, batch_size: int):
        super().__init__()
        self.X_train = X_train.astype(np.float32)
        self.X_val = X_val.astype(np.float32)
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_ds = TensorDataset(torch.from_numpy(self.X_train))
        self.val_ds = TensorDataset(torch.from_numpy(self.X_val))

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )


# --------------------------------------------------------------------------- #
# 6️⃣ Main pipeline execution
# --------------------------------------------------------------------------- #
def main():
    # ------------------------------------------------- #
    # 0️⃣ Global setup
    # ------------------------------------------------- #
    ensure_dirs()
    np.random.seed(CFG["preprocess"]["random_state"])
    torch.manual_seed(CFG["model"]["random_state"])

    # ------------------------------------------------- #
    # 1️⃣ Data ingestion & validation
    # ------------------------------------------------- #
    copy_raw_csv()
    df = pd.read_csv(CFG["paths"]["raw_dst"])
    validate_dataset(df)
    save_manifest(df)
    splits = stratified_split(df)

    # ------------------------------------------------- #
    # 2️⃣ Pre‑processing
    # ------------------------------------------------- #
    feature_cols = [c for c in df.columns if c != "target"]
    pipeline = build_preprocess_pipeline()

    X_train = pipeline.fit_transform(splits["train"][feature_cols])
    X_val = pipeline.transform(splits["val"][feature_cols])
    X_test = pipeline.transform(splits["test"][feature_cols])  # not used for training

    # Persist the fitted pipeline
    joblib.dump(pipeline, CFG["paths"]["pipeline_out"])
    print(f"[INFO] Preprocessing pipeline saved to {CFG['paths']['pipeline_out']}")

    # (Optional) Save the scaled training matrix for debugging – CSV is enough
    train_scaled_df = pd.DataFrame(
        X_train, columns=feature_cols, index=splits["train"].index
    )
    train_scaled_path = Path("data/processed/train_scaled.csv")
    train_scaled_path.parent.mkdir(parents=True, exist_ok=True)
    train_scaled_df.to_csv(train_scaled_path, index=False)
    print(f"[INFO] Scaled training data saved to {train_scaled_path}")

    # ------------------------------------------------- #
    # 3️⃣ Model training (pre‑train + clustering)
    # ------------------------------------------------- #
    n_features = X_train.shape[1]
    model_cfg = CFG["model"]
    dec = DEC(n_features=n_features, cfg=model_cfg)

    data_module = WineDataModule(
        X_train=X_train, X_val=X_val, batch_size=model_cfg["batch_size"]
    )
    data_module.setup()

    # Callbacks: checkpoint (best by pre‑train loss initially) + early stopping
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=Path(CFG["paths"]["trained_models_dir"]),
        filename="dec-{epoch:02d}-{pretrain_loss:.4f}",
        monitor="pretrain_loss",
        mode="min",
        save_top_k=1,
    )
    early_stop_cb = pl.callbacks.EarlyStopping(
        monitor="pretrain_loss",
        patience=5,
        mode="min",
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=model_cfg["pretrain_epochs"] + model_cfg["cluster_epochs"],
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=False,
        enable_progress_bar=True,
        deterministic=True,
    )

    # ------------------- Pre‑training ------------------- #
    print("\n=== Pre‑training auto‑encoder ===")
    trainer.fit(dec, datamodule=data_module)

    # ---------------- Initialise centroids --------------- #
    dec.init_cluster_centroids(data_module.train_dataloader())

    # ------------------- Clustering ------------------- #
    # Replace early‑stopping monitor to clustering loss
    early_stop_cb = pl.callbacks.EarlyStopping(
        monitor="cluster_loss",
        patience=5,
        mode="min",
        verbose=True,
    )
    # Update checkpoint monitor to clustering loss as well
    checkpoint_cb = pl.callbacks.ModelCheckpoint(
        dirpath=Path(CFG["paths"]["trained_models_dir"]),
        filename="dec-{epoch:02d}-{cluster_loss:.4f}",
        monitor="cluster_loss",
        mode="min",
        save_top_k=1,
    )
    trainer = pl.Trainer(
        max_epochs=model_cfg["pretrain_epochs"] + model_cfg["cluster_epochs"],
        callbacks=[checkpoint_cb, early_stop_cb],
        logger=False,
        enable_progress_bar=True,
        deterministic=True,
    )
    print("\n=== Clustering phase ===")
    trainer.fit(dec, datamodule=data_module)

    # ------------------- Save best checkpoint ------------------- #
    best_ckpt_path = checkpoint_cb.best_model_path
    final_ckpt_path = Path(CFG["paths"]["ckpt_out"])
    shutil.copy2(best_ckpt_path, final_ckpt_path)
    print(f"[INFO] Best checkpoint copied to {final_ckpt_path}")

    # Also copy pipeline & checkpoint into the mandated top‑level folder
    shutil.copy2(CFG["paths"]["pipeline_out"], CFG["paths"]["trained_models_dir"])
    shutil.copy2(final_ckpt_path, CFG["paths"]["trained_models_dir"])
    print(f"[INFO] Artefacts also placed in {CFG['paths']['trained_models_dir']}")

    # ------------------------------------------------- #
    # 4️⃣ Evaluation on the whole dataset
    # ------------------------------------------------- #
    full_X = pipeline.transform(df[feature_cols])
    full_tensor = torch.from_numpy(full_X.astype(np.float32)).to(dec.device)

    dec.eval()
    with torch.no_grad():
        latent = dec.encode(full_tensor).cpu().numpy()
        q = dec.compute_q(dec.encode(full_tensor)).cpu().numpy()
    pred_labels = np.argmax(q, axis=1)

    ari = adjusted_rand_score(df["target"], pred_labels)
    ami = adjusted_mutual_info_score(df["target"], pred_labels)
    sil = silhouette_score(latent, pred_labels)

    metrics = {"ARI": ari, "AMI": ami, "Silhouette": sil}
    print("\n=== Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # Persist metrics
    Path(CFG["paths"]["metrics_out"]).parent.mkdir(parents=True, exist_ok=True)
    with open(CFG["paths"]["metrics_out"], "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[INFO] Metrics saved to {CFG['paths']['metrics_out']}")

    # ------------------------------------------------- #
    # 5️⃣ Optional Gradio demo for inference
    # ------------------------------------------------- #
    def predict(csv_file):
        """Accept a CSV (no target column) and return cluster IDs."""
        df_user = pd.read_csv(csv_file.name)
        missing = set(feature_cols) - set(df_user.columns)
        if missing:
            return {"error": f"Missing columns: {sorted(missing)}"}
        X_user = pipeline.transform(df_user[feature_cols])
        X_user_tensor = torch.from_numpy(X_user.astype(np.float32)).to(dec.device)
        dec.eval()
        with torch.no_grad():
            q_user = dec.compute_q(dec.encode(X_user_tensor)).cpu().numpy()
        labels = np.argmax(q_user, axis=1)
        return {"cluster_id": labels.tolist()}

    demo = gr.Interface(
        fn=predict,
        inputs=gr.File(label="Upload CSV (13 wine features, no target)"),
        outputs=gr.JSON(label="Predicted cluster IDs"),
        title="Wine DEC Clustering Demo",
        description="Upload a CSV containing the same 13 feature columns as the training data "
        "to obtain cluster assignments produced by the DEC model.",
    )
    demo.launch(share=False, debug=False)


if __name__ == "__main__":
    main()
