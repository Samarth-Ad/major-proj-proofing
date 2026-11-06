
# -*- coding: ascii -*-
"""
Wine dataset Deep‑Embedded Clustering (DEC) pipeline.
The script follows the reproducible blueprint:
1. ingest & validate raw CSV
2. stratified train/val/test split
3. build & save sklearn preprocessing pipeline
4. train DEC with PyTorch‑Lightning
5. evaluate (ARI, AMI, Silhouette) on the full data
6. log everything to a local MLflow run
7. save artefacts (pipeline, checkpoint, metrics, data manifest)

All artefacts are stored under the repository root.
The trained model checkpoint is saved to "./agent_workspace/trained_models".
"""

# ----------------------------------------------------------------------
# 0️⃣  Helper: install missing third‑party packages on the fly
# ----------------------------------------------------------------------
import sys
import subprocess

def _install(pkg: str):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

# try‑except imports – install if not present
try:
    import mlflow
except Exception:  # pragma: no cover
    _install("mlflow")
    import mlflow

try:
    import pytorch_lightning as pl
except Exception:  # pragma: no cover
    _install("pytorch-lightning")
    import pytorch_lightning as pl

# ----------------------------------------------------------------------
# 1️⃣  Standard library imports
# ----------------------------------------------------------------------
import os
import json
import shutil
import hashlib
import random
from pathlib import Path
from typing import Dict, Any

# ----------------------------------------------------------------------
# 2️⃣  Third‑party imports
# ----------------------------------------------------------------------
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    silhouette_score,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

# ----------------------------------------------------------------------
# 3️⃣  Configuration (ASCII only)
# ----------------------------------------------------------------------
CFG: Dict[str, Any] = {
    "paths": {
        "raw_src": "data/Wine.csv",                     # <-- adjust if needed
        "raw_dst": "data/raw/Wine.csv",
        "processed_path": "data/processed/X_train_scaled.parquet",
        "splits_dir": "data/splits",
        "manifest_path": "outputs/metadata/data_manifest.json",
        "pipeline_path": "outputs/models/preprocess_pipe.pkl",
        "checkpoint_path": "agent_workspace/trained_models/dec_best.ckpt",
        "metrics_path": "outputs/metrics/dec_metrics.json",
    },
    "preprocess": {
        "impute_strategy": "median",
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
        "tracking_uri": "file://"+str(Path.cwd() / "mlruns"),
        "run_name": "DEC_run",
        "tags": {
            "project": "wine_clustering",
            "model_type": "DEC",
            "framework": "pytorch-lightning",
        },
    },
}

# ----------------------------------------------------------------------
# 4️⃣  Utility functions
# ----------------------------------------------------------------------
def ensure_dirs() -> None:
    """Create all required directories (idempotent)."""
    dirs = [
        "data/raw",
        "data/processed",
        "data/splits",
        "outputs/models",
        "outputs/metrics",
        "outputs/metadata",
        "agent_workspace/trained_models",
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with file_path.open("rb") as f:
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
# 5️⃣  Data ingestion & validation
# ----------------------------------------------------------------------
def ingest_and_split() -> None:
    src = Path(CFG["paths"]["raw_src"])
    dst = Path(CFG["paths"]["raw_dst"])
    shutil.copy2(src, dst)

    df = pd.read_csv(dst)

    # ---- validation ----------------------------------------------------
    expected_shape = (178, 14)  # known shape of the Wine dataset
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
    Path(CFG["paths"]["manifest_path"]).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # ---- stratified split -----------------------------------------------
    ss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=CFG["split"]["val_frac"] + CFG["split"]["test_frac"],
        random_state=CFG["preprocess"]["random_state"],
    )
    X = df.drop(columns=["target"])
    y = df["target"]
    train_idx, temp_idx = next(ss.split(X, y))
    train_df = df.iloc[train_idx]
    temp_df = df.iloc[temp_idx]

    # split temp into val and test
    val_frac_rel = CFG["split"]["val_frac"] / (
        CFG["split"]["val_frac"] + CFG["split"]["test_frac"]
    )
    ss_val = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1 - val_frac_rel,
        random_state=CFG["preprocess"]["random_state"],
    )
    X_temp = temp_df.drop(columns=["target"])
    y_temp = temp_df["target"]
    val_idx, test_idx = next(ss_val.split(X_temp, y_temp))
    val_df = temp_df.iloc[val_idx]
    test_df = temp_df.iloc[test_idx]

    # write splits atomically
    split_dir = Path(CFG["paths"]["splits_dir"])
    for name, frame in zip(
        ["train.csv", "val.csv", "test.csv"], [train_df, val_df, test_df]
    ):
        tmp = split_dir / f".{name}.tmp"
        final = split_dir / name
        frame.to_csv(tmp, index=False)
        tmp.replace(final)

# ----------------------------------------------------------------------
# 6️⃣  Pre‑processing pipeline
# ----------------------------------------------------------------------
def build_and_save_preprocess() -> None:
    split_dir = Path(CFG["paths"]["splits_dir"])
    train_df = pd.read_csv(split_dir / "train.csv")
    val_df = pd.read_csv(split_dir / "val.csv")
    test_df = pd.read_csv(split_dir / "test.csv")

    X_train = train_df.drop(columns=["target"])
    X_val = val_df.drop(columns=["target"])
    X_test = test_df.drop(columns=["target"])

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

    # save scaled matrices as parquet
    pd.DataFrame(X_train_scaled).to_parquet(
        CFG["paths"]["processed_path"], engine="pyarrow"
    )
    pd.DataFrame(X_val_scaled).to_parquet(
        Path(CFG["paths"]["processed_path"]).with_name("X_val_scaled.parquet"),
        engine="pyarrow",
    )
    pd.DataFrame(X_test_scaled).to_parquet(
        Path(CFG["paths"]["processed_path"]).with_name("X_test_scaled.parquet"),
        engine="pyarrow",
    )

# ----------------------------------------------------------------------
# 7️⃣  DEC LightningModule
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

        # encoder
        encoder_layers = []
        in_dim = n_features
        for h in cfg["model"]["hidden_dims"]:
            encoder_layers.append(nn.Linear(in_dim, h))
            encoder_layers.append(nn.ReLU())
            in_dim = h
        encoder_layers.append(nn.Linear(in_dim, self.latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # decoder (mirror)
        decoder_layers = []
        in_dim = self.latent_dim
        for h in reversed(cfg["model"]["hidden_dims"]):
            decoder_layers.append(nn.Linear(in_dim, h))
            decoder_layers.append(nn.ReLU())
            in_dim = h
        decoder_layers.append(nn.Linear(in_dim, n_features))
        self.decoder = nn.Sequential(*decoder_layers)

        # cluster centroids (trainable)
        self.cluster_layer = nn.Parameter(
            torch.Tensor(self.n_clusters, self.latent_dim)
        )
        torch.nn.init.xavier_uniform_(self.cluster_layer)

        self.p_target = None  # will hold the target distribution

        # reproducibility
        torch.manual_seed(cfg["model"]["random_state"])

    # ------------------------------------------------------------------
    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x):
        return self.encoder(x)

    @staticmethod
    def compute_q(z, centroids):
        """Student‑t distribution (alpha=1) soft assignments."""
        # z: (N, d) ; centroids: (K, d)
        expanded_z = z.unsqueeze(1)          # (N,1,d)
        expanded_c = centroids.unsqueeze(0)  # (1,K,d)
        dist = torch.sum((expanded_z - expanded_c) ** 2, dim=2)  # (N,K)
        q = 1.0 / (1.0 + dist)
        q = q ** ((1 + 1) / 2)  # alpha=1 => exponent = 1
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

    @staticmethod
    def target_distribution(q):
        """Compute the target distribution p from soft assignments q."""
        weight = q ** 2 / torch.sum(q, dim=0, keepdim=True)
        p = weight / torch.sum(weight, dim=1, keepdim=True)
        return p

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(),
            lr=self.cfg["model"]["learning_rate"],
            weight_decay=self.cfg["model"]["weight_decay"],
        )

    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x = x.float()
        x_hat, z = self.forward(x)

        if self.current_epoch < self.pretrain_epochs:
            loss = F.mse_loss(x_hat, x)
            self.log("train_loss", loss, prog_bar=True)
            return loss

        # clustering phase
        q = self.compute_q(z, self.cluster_layer)
        if self.p_target is None:
            # first time – will be set in on_train_epoch_end
            loss = F.kl_div(q.log(), q, reduction="batchmean")
        else:
            loss = F.kl_div(q.log(), self.p_target, reduction="batchmean")
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
        """Update target distribution every `update_interval` epochs (clustering phase)."""
        if self.current_epoch < self.pretrain_epochs:
            return

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

        q = self.compute_q(all_z, self.cluster_layer)
        self.p_target = self.target_distribution(q).detach()
        self.train()

# ----------------------------------------------------------------------
# 8️⃣  LightningDataModule
# ----------------------------------------------------------------------
class WineDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg["model"]["batch_size"]
        self.train_path = Path(cfg["paths"]["processed_path"])
        self.val_path = self.train_path.parent / "X_val_scaled.parquet"
        self.test_path = self.train_path.parent / "X_test_scaled.parquet"

    def setup(self, stage: str = None):
        X_train = pd.read_parquet(self.train_path).values.astype(np.float32)
        X_val = pd.read_parquet(self.val_path).values.astype(np.float32)
        X_test = pd.read_parquet(self.test_path).values.astype(np.float32)

        # dummy labels (required by DataLoader)
        y_dummy = np.zeros(len(X_train), dtype=np.int64)

        self.train_dataset = TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_dummy)
        )
        self.val_dataset = TensorDataset(
            torch.from_numpy(X_val), torch.from_numpy(np.zeros(len(X_val), dtype=np.int64))
        )
        self.test_dataset = TensorDataset(
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
# 9️⃣  Training orchestration
# ----------------------------------------------------------------------
def train_and_log():
    # ------------------------------------------------------------------
    # Global seeds
    # ------------------------------------------------------------------
    SEED = CFG["preprocess"]["random_state"]
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # ------------------------------------------------------------------
    # MLflow setup
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(CFG["mlflow"]["tracking_uri"])
    mlflow.set_experiment(CFG["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=CFG["mlflow"]["run_name"]) as run:
        # log all hyper‑parameters
        mlflow.log_params(flatten_cfg(CFG))

        # ------------------------------------------------------------------
        # DataModule & model
        # ------------------------------------------------------------------
        dm = WineDataModule(CFG)
        dm.setup()
        n_features = dm.train_dataset.tensors[0].shape[1]
        model = DEC(n_features=n_features, cfg=CFG)

        # ------------------------------------------------------------------
        # Callbacks
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # Trainer
        # ------------------------------------------------------------------
        logger = MLFlowLogger(
            experiment_name=CFG["mlflow"]["experiment_name"],
            tracking_uri=CFG["mlflow"]["tracking_uri"],
            run_id=run.info.run_id,
        )
        trainer = pl.Trainer(
            max_epochs=CFG["model"]["pretrain_epochs"] + CFG["model"]["cluster_epochs"],
            callbacks=[early_stop_cb, checkpoint_cb],
            logger=logger,
            accelerator="cpu",  # change to "gpu" if CUDA is available
            enable_checkpointing=True,
            log_every_n_steps=5,
        )

        # ------------------------------------------------------------------
        # Fit
        # ------------------------------------------------------------------
        trainer.fit(model, dm)

        # ------------------------------------------------------------------
        # Log artefacts
        # ------------------------------------------------------------------
        mlflow.log_artifact(CFG["paths"]["pipeline_path"], artifact_path="preprocess")
        mlflow.log_artifact(CFG["paths"]["manifest_path"], artifact_path="metadata")
        mlflow.log_artifact(CFG["paths"]["checkpoint_path"], artifact_path="model")

        # ------------------------------------------------------------------
        # Evaluation on the full dataset
        # ------------------------------------------------------------------
        evaluate_and_log(run.info.run_id)

# ----------------------------------------------------------------------
# 🔟  Evaluation & metric logging
# ----------------------------------------------------------------------
def evaluate_and_log(run_id: str) -> None:
    # Load raw data
    raw_df = pd.read_csv(CFG["paths"]["raw_dst"])
    feature_cols = [c for c in raw_df.columns if c != "target"]
    X_full = raw_df[feature_cols].values.astype(np.float32)

    # Load preprocessing pipeline
    pipeline = joblib.load(CFG["paths"]["pipeline_path"])
    X_full_scaled = pipeline.transform(X_full)

    # Load trained DEC checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X_full_scaled.shape[1]
    dec = DEC(n_features=n_features, cfg=CFG)
    checkpoint = torch.load(CFG["paths"]["checkpoint_path"], map_location=device)
    # Lightning saves only state_dict in the checkpoint file
    if "state_dict" in checkpoint:
        dec.load_state_dict(checkpoint["state_dict"])
    else:
        dec.load_state_dict(checkpoint)
    dec.to(device)
    dec.eval()

    # Compute latent embeddings for the whole data
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
    Path(CFG["paths"]["metrics_path"]).write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    # Log metrics & artefact to MLflow
    mlflow.start_run(run_id=run_id)
    mlflow.log_metrics(metrics)
    mlflow.log_artifact(CFG["paths"]["metrics_path"], artifact_path="metrics")
    mlflow.end_run()

# ----------------------------------------------------------------------
# 🏁  Main entry point
# ----------------------------------------------------------------------
def main() -> None:
    ensure_dirs()
    ingest_and_split()
    build_and_save_preprocess()
    train_and_log()
    print("✅ Pipeline completed. Check the ./outputs and ./agent_workspace directories.")

if __name__ == "__main__":
    main()
