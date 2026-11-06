
# -*- coding: utf-8 -*-
"""
Wine dataset Deep‑Embedded Clustering (DEC) pipeline.
The script is fully self‑contained, reproducible and logs everything to a local
MLflow experiment.  All artefacts (pre‑processing pipeline, model checkpoint,
metrics and data manifest) are saved under the repository root or under
"./agent_workspace/trained_models" as required.
"""

import os
import json
import shutil
import hashlib
import random
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.pytorch
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    silhouette_score,
)
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

# ----------------------------------------------------------------------
# 1️⃣  Global configuration (ASCII only – no fancy dashes)
# ----------------------------------------------------------------------
CFG: Dict[str, Any] = {
    "paths": {
        "raw_src": "data/Wine.csv",                     # source CSV (must exist)
        "raw_dst": "data/raw/Wine.csv",                 # copy inside repo
        "processed_path": "data/processed/X_scaled.parquet",
        "splits_dir": "data/splits",
        "manifest_path": "outputs/metadata/data_manifest.json",
        "pipeline_path": "outputs/models/preprocess_pipe.pkl",
        "checkpoint_path": "agent_workspace/trained_models/dec_best.ckpt",
        "metrics_path": "outputs/metrics/dec_metrics.json",
    },
    "preprocess": {
        "impute_strategy": "median",
        "scale_method": "standard",
        "random_state": 42,
    },
    "model": {
        "latent_dim": 10,
        "hidden_dims": [64, 32],
        "n_clusters": 3,
        "pretrain_epochs": 30,
        "cluster_epochs": 80,
        "batch_size": 32,
        "learning_rate": 5e-4,
        "weight_decay": 0.0,
        "optimizer": "adam",
        "update_interval": 10,
        "tol": 1e-4,
        "early_stop_patience": 5,
        "random_state": 42,
    },
    "split": {
        "train_frac": 0.70,
        "val_frac": 0.20,
        "test_frac": 0.10,
        "stratify": True,
    },
    "mlflow": {
        "experiment_name": "wine_dec_clustering",
        "tracking_uri": "file://" + str(Path.cwd() / "mlruns"),
        "run_name": "DEC_run",
        "tags": {
            "project": "wine_clustering",
            "model_type": "DEC",
            "framework": "pytorch-lightning",
        },
    },
}


# ----------------------------------------------------------------------
# 2️⃣  Helper utilities
# ----------------------------------------------------------------------
def ensure_dirs() -> None:
    """Create all required directories (idempotent)."""
    for p in [
        "data/raw",
        "data/processed",
        "data/splits",
        "outputs/models",
        "outputs/metrics",
        "outputs/metadata",
        "agent_workspace/trained_models",
    ]:
        Path(p).mkdir(parents=True, exist_ok=True)


def compute_sha256(file_path: str) -> str:
    """Return the SHA‑256 hash of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def flatten_cfg(cfg: Dict[str, Any], parent_key: str = "") -> Dict[str, Any]:
    """Flatten nested dict for MLflow logging."""
    items = {}
    for k, v in cfg.items():
        new_key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_cfg(v, new_key))
        else:
            items[new_key] = v
    return items


# ----------------------------------------------------------------------
# 3️⃣  Data ingestion & validation
# ----------------------------------------------------------------------
def ingest_and_split() -> None:
    """Copy raw CSV, validate, create stratified splits and a data manifest."""
    src = CFG["paths"]["raw_src"]
    dst = CFG["paths"]["raw_dst"]
    shutil.copy2(src, dst)

    df = pd.read_csv(dst)

    # ---- validation ----------------------------------------------------
    expected_shape = (178, 14)  # Wine dataset shape
    if df.shape != expected_shape:
        raise ValueError(f"Unexpected shape {df.shape}, expected {expected_shape}")

    if "target" not in df.columns:
        raise ValueError("Column 'target' missing from dataset")

    if not set(df["target"].unique()).issubset({0, 1, 2}):
        raise ValueError("Target column must contain only {0,1,2}")

    if df.duplicated().any():
        raise ValueError("Duplicate rows detected")

    if df.isnull().any().any():
        raise ValueError("Missing values detected")

    # ---- manifest ------------------------------------------------------
    manifest = {
        "sha256": compute_sha256(dst),
        "columns": {c: str(dt) for c, dt in zip(df.columns, df.dtypes)},
        "row_count": len(df),
    }
    with open(CFG["paths"]["manifest_path"], "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # ---- stratified split -----------------------------------------------
    ss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=CFG["split"]["val_frac"] + CFG["split"]["test_frac"],
        random_state=CFG["preprocess"]["random_state"],
    )
    X = df.drop(columns=["target"])
    y = df["target"]
    for train_idx, temp_idx in ss.split(X, y):
        train_df = df.iloc[train_idx]
        temp_df = df.iloc[temp_idx]

    # split temp into val and test
    val_frac_rel = CFG["split"]["val_frac"] / (CFG["split"]["val_frac"] + CFG["split"]["test_frac"])
    ss_val = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1 - val_frac_rel,
        random_state=CFG["preprocess"]["random_state"],
    )
    X_temp = temp_df.drop(columns=["target"])
    y_temp = temp_df["target"]
    for val_idx, test_idx in ss_val.split(X_temp, y_temp):
        val_df = temp_df.iloc[val_idx]
        test_df = temp_df.iloc[test_idx]

    # write splits atomically
    for name, frame in zip(
        ["train.csv", "val.csv", "test.csv"], [train_df, val_df, test_df]
    ):
        tmp_path = Path(CFG["paths"]["splits_dir"]) / f".{name}.tmp"
        final_path = Path(CFG["paths"]["splits_dir"]) / name
        frame.to_csv(tmp_path, index=False)
        tmp_path.replace(final_path)


# ----------------------------------------------------------------------
# 4️⃣  Pre‑processing pipeline
# ----------------------------------------------------------------------
def build_and_save_preprocess() -> None:
    """Fit a SimpleImputer + StandardScaler pipeline on the training split."""
    splits_dir = Path(CFG["paths"]["splits_dir"])
    train_df = pd.read_csv(splits_dir / "train.csv")
    val_df = pd.read_csv(splits_dir / "val.csv")
    test_df = pd.read_csv(splits_dir / "test.csv")

    X_train = train_df.drop(columns=["target"])
    X_val = val_df.drop(columns=["target"])
    X_test = test_df.drop(columns=["target"])

    # deterministic imputer (median) – seed already set globally
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=CFG["preprocess"]["impute_strategy"])),
            ("scaler", StandardScaler()),
        ]
    )
    pipeline.fit(X_train)

    # transform all splits
    X_train_scaled = pipeline.transform(X_train)
    X_val_scaled = pipeline.transform(X_val)
    X_test_scaled = pipeline.transform(X_test)

    # persist pipeline
    joblib.dump(pipeline, CFG["paths"]["pipeline_path"])

    # save scaled training matrix (used later for pre‑training)
    pd.DataFrame(X_train_scaled).to_parquet(
        CFG["paths"]["processed_path"], engine="pyarrow"
    )

    # also store the scaled validation & test sets for quick access later
    pd.DataFrame(X_val_scaled).to_parquet(
        Path(CFG["paths"]["processed_path"]).with_name("X_val_scaled.parquet"),
        engine="pyarrow",
    )
    pd.DataFrame(X_test_scaled).to_parquet(
        Path(CFG["paths"]["processed_path"]).with_name("X_test_scaled.parquet"),
        engine="pyarrow",
    )


# ----------------------------------------------------------------------
# 5️⃣  DEC LightningModule
# ----------------------------------------------------------------------
class DEC(pl.LightningModule):
    """Deep‑Embedded Clustering implementation."""

    def __init__(self, n_features: int, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.n_clusters = cfg["model"]["n_clusters"]
        self.latent_dim = cfg["model"]["latent_dim"]
        self.update_interval = cfg["model"]["update_interval"]
        self.pretrain_epochs = cfg["model"]["pretrain_epochs"]
        self.tol = cfg["model"]["tol"]

        # ------------------- encoder -------------------
        encoder_layers = []
        in_dim = n_features
        for h in cfg["model"]["hidden_dims"]:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, self.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # ------------------- decoder (mirror) -------------------
        decoder_layers = []
        in_dim = self.latent_dim
        for h in reversed(cfg["model"]["hidden_dims"]):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, n_features))
        self.decoder = nn.Sequential(*decoder_layers)

        # ------------------- cluster centroids -------------------
        self.cluster_layer = nn.Parameter(torch.Tensor(self.n_clusters, self.latent_dim))
        torch.nn.init.xavier_uniform_(self.cluster_layer)

        # placeholder for target distribution (updated every `update_interval`)
        self.p_target = None

        # reproducibility
        torch.manual_seed(cfg["model"]["random_state"])

    # ------------------------------------------------------------------
    def forward(self, x):
        """Return reconstruction and latent embedding."""
        z = self.encode(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x):
        return self.encoder(x)

    @staticmethod
    def compute_q(z, centroids):
        """Student‑t distribution (α=1) soft assignment."""
        # z: (batch, latent_dim) ; centroids: (n_clusters, latent_dim)
        expanded_z = z.unsqueeze(1)  # (batch,1,latent)
        expanded_c = centroids.unsqueeze(0)  # (1,nc,latent)
        dist = torch.sum((expanded_z - expanded_c) ** 2, dim=2)  # (batch,nc)
        q = 1.0 / (1.0 + dist)
        q = q ** ((1 + 1) / 2)  # α=1 → exponent = 1
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

    @staticmethod
    def target_distribution(q):
        """Compute the target distribution p."""
        weight = q ** 2 / torch.sum(q, dim=0, keepdim=True)
        p = weight / torch.sum(weight, dim=1, keepdim=True)
        return p

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.cfg["model"]["learning_rate"],
            weight_decay=self.cfg["model"]["weight_decay"],
        )
        return optimizer

    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        x, _ = batch  # labels are not used in unsupervised training
        x = x.float()
        x_hat, z = self.forward(x)

        if self.current_epoch < self.pretrain_epochs:
            # reconstruction loss only
            loss = F.mse_loss(x_hat, x)
            self.log("train_loss", loss, prog_bar=True)
            return loss

        # clustering phase
        q = self.compute_q(z, self.cluster_layer)
        if self.p_target is None or (self.global_step % self.update_interval == 0):
            # recompute p on the whole training set (handled in on_train_epoch_end)
            pass

        p = self.p_target  # set in on_train_epoch_end
        loss = F.kl_div(q.log(), p, reduction="batchmean")
        self.log("train_loss", loss, prog_bar=True)
        return loss

    # ------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x = x.float()
        x_hat, z = self.forward(x)

        if self.current_epoch < self.pretrain_epochs:
            loss = F.mse_loss(x_hat, x)
        else:
            q = self.compute_q(z, self.cluster_layer)
            loss = F.kl_div(q.log(), self.p_target, reduction="batchmean")
        self.log("val_loss", loss, prog_bar=True)

    # ------------------------------------------------------------------
    def on_train_epoch_end(self):
        """Update target distribution p every `update_interval` epochs (clustering phase)."""
        if self.current_epoch < self.pretrain_epochs:
            return  # still pre‑training

        if (self.current_epoch - self.pretrain_epochs) % self.update_interval != 0:
            return

        # compute latent embeddings for the whole training set
        self.eval()
        device = self.device
        train_loader = self.trainer.datamodule.train_dataloader()
        all_z = []
        with torch.no_grad():
            for xb, _ in train_loader:
                xb = xb.float().to(device)
                z = self.encode(xb)
                all_z.append(z)
        all_z = torch.cat(all_z, dim=0)

        # compute q and then p
        q = self.compute_q(all_z, self.cluster_layer)
        self.p_target = self.target_distribution(q).detach()
        self.train()


# ----------------------------------------------------------------------
# 6️⃣  LightningDataModule (handles train/val loaders)
# ----------------------------------------------------------------------
class WineDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg["model"]["batch_size"]
        self.processed_path = cfg["paths"]["processed_path"]
        self.val_path = Path(self.processed_path).with_name("X_val_scaled.parquet")
        self.test_path = Path(self.processed_path).with_name("X_test_scaled.parquet")

    def setup(self, stage: str = None):
        # Load scaled matrices
        X_train = pd.read_parquet(self.processed_path).values.astype(np.float32)
        X_val = pd.read_parquet(self.val_path).values.astype(np.float32)
        X_test = pd.read_parquet(self.test_path).values.astype(np.float32)

        # Dummy labels (required by DataLoader but not used)
        y_dummy = np.zeros(len(X_train), dtype=np.int64)

        self.train_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_dummy)
        )
        self.val_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_val), torch.from_numpy(np.zeros(len(X_val), dtype=np.int64))
        )
        self.test_dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_test), torch.from_numpy(np.zeros(len(X_test), dtype=np.int64))
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )


# ----------------------------------------------------------------------
# 7️⃣  Training orchestration
# ----------------------------------------------------------------------
def train_and_log():
    # --------------------------------------------------------------
    # Global seeds
    # --------------------------------------------------------------
    SEED = CFG["preprocess"]["random_state"]
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # --------------------------------------------------------------
    # MLflow setup
    # --------------------------------------------------------------
    mlflow.set_tracking_uri(CFG["mlflow"]["tracking_uri"])
    mlflow.set_experiment(CFG["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=CFG["mlflow"]["run_name"]) as run:
        # Log all hyper‑parameters (flattened)
        mlflow.log_params(flatten_cfg(CFG))

        # ----------------------------------------------------------
        # DataModule & model
        # ----------------------------------------------------------
        dm = WineDataModule(CFG)
        dm.setup()

        n_features = dm.train_dataset.tensors[0].shape[1]
        model = DEC(n_features=n_features, cfg=CFG)

        # ----------------------------------------------------------
        # Callbacks
        # ----------------------------------------------------------
        checkpoint_cb = ModelCheckpoint(
            dirpath="agent_workspace/trained_models",
            filename="dec_best",
            save_top_k=1,
            monitor="val_loss",
            mode="min",
        )
        early_stop_cb = EarlyStopping(
            monitor="val_loss",
            patience=CFG["model"]["early_stop_patience"],
            mode="min",
        )

        # ----------------------------------------------------------
        # Trainer
        # ----------------------------------------------------------
        trainer = pl.Trainer(
            max_epochs=CFG["model"]["pretrain_epochs"] + CFG["model"]["cluster_epochs"],
            callbacks=[early_stop_cb, checkpoint_cb],
            logger=pl.loggers.MLFlowLogger(
                experiment_name=CFG["mlflow"]["experiment_name"],
                tracking_uri=CFG["mlflow"]["tracking_uri"],
                run_id=run.info.run_id,
            ),
            accelerator="cpu",  # change to "gpu" if CUDA is available
            enable_checkpointing=True,
            log_every_n_steps=5,
        )

        # ----------------------------------------------------------
        # Fit
        # ----------------------------------------------------------
        trainer.fit(model, dm)

        # ----------------------------------------------------------
        # Log artefacts
        # ----------------------------------------------------------
        mlflow.log_artifact(CFG["paths"]["pipeline_path"], artifact_path="preprocess")
        mlflow.log_artifact(CFG["paths"]["manifest_path"], artifact_path="metadata")
        mlflow.log_artifact(CFG["paths"]["checkpoint_path"], artifact_path="model")

        # ----------------------------------------------------------
        # Evaluation on the *full* dataset (train+val+test)
        # ----------------------------------------------------------
        evaluate_and_log(run.info.run_id)


# ----------------------------------------------------------------------
# 8️⃣  Evaluation & metric logging
# ----------------------------------------------------------------------
def evaluate_and_log(run_id: str) -> None:
    # Load raw data
    raw_df = pd.read_csv(CFG["paths"]["raw_dst"])
    feature_cols = [c for c in raw_df.columns if c != "target"]
    X_full = raw_df[feature_cols].values.astype(np.float32)

    # Load preprocessing pipeline and transform full data
    pipeline = joblib.load(CFG["paths"]["pipeline_path"])
    X_full_scaled = pipeline.transform(X_full)

    # Load trained DEC checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X_full_scaled.shape[1]
    dec = DEC(n_features=n_features, cfg=CFG)
    checkpoint = torch.load(CFG["paths"]["checkpoint_path"], map_location=device)
    dec.load_state_dict(checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint)
    dec.to(device)
    dec.eval()

    # Compute latent embeddings
    with torch.no_grad():
        tensor = torch.from_numpy(X_full_scaled).to(device)
        latent = dec.encode(tensor).cpu().numpy()

    # Soft assignments & hard labels
    q = DEC.compute_q(
        torch.from_numpy(latent).to(device), dec.cluster_layer.detach().cpu()
    )
    y_pred = torch.argmax(q, dim=1).cpu().numpy()
    y_true = raw_df["target"].values

    # Metrics
    ari = adjusted_rand_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    sil = silhouette_score(latent, y_pred)

    metrics = {
        "ARI": round(float(ari), 5),
        "AMI": round(float(ami), 5),
        "Silhouette": round(float(sil), 5),
    }

    # Save metrics JSON
    with open(CFG["paths"]["metrics_path"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Log metrics & artefact to MLflow
    mlflow.set_tracking_uri(CFG["mlflow"]["tracking_uri"])
    mlflow.set_experiment(CFG["mlflow"]["experiment_name"])
    mlflow.start_run(run_id=run_id)
    mlflow.log_metrics(metrics)
    mlflow.log_artifact(CFG["paths"]["metrics_path"], artifact_path="metrics")
    mlflow.end_run()


# ----------------------------------------------------------------------
# 9️⃣  Main entry point
# ----------------------------------------------------------------------
def main() -> None:
    ensure_dirs()
    ingest_and_split()
    build_and_save_preprocess()
    train_and_log()
    print("✅ Pipeline finished. Check ./outputs and ./agent_workspace for artefacts.")


if __name__ == "__main__":
    main()
