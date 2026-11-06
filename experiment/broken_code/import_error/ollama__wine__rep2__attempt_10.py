
# -*- coding: utf-8 -*-
"""
End‑to‑end DEC clustering pipeline for the Wine dataset.

The script follows the reproducible blueprint described in the
project specification.  All artefacts are written under the
repository root, and the final model + preprocessing pipeline are
copied to ``./agent_workspace/trained_models`` for downstream use.

Run with:
    python -m dec_wine_pipeline
"""

import json
import hashlib
import shutil
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import mlflow
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, silhouette_score
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------------------------------------------------
# 1️⃣  Configuration (ASCII only)
# ----------------------------------------------------------------------
CONFIG = {
    "paths": {
        "raw_src": "C:/Users/Samarth_Ad/Desktop/College/SEM-7/AutoMLAgent/experiment/datasets/wine.csv",
        "raw_dst": "data/raw/Wine.csv",
        "processed_path": "data/processed/X_train_scaled.parquet",
        "splits_dir": "data/splits",
        "manifest_path": "outputs/metadata/data_manifest.json",
        "pipeline_path": "outputs/models/preprocess_pipe.pkl",
        "checkpoint_path": "outputs/models/dec_best.ckpt",
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
        "tracking_uri": "file://./mlruns",
        "run_name": "DEC_run",
        "tags": {
            "project": "wine_clustering",
            "model_type": "DEC",
            "framework": "pytorch-lightning",
        },
    },
}

# ----------------------------------------------------------------------
# 2️⃣  Utility helpers
# ----------------------------------------------------------------------
def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------
# 3️⃣  Data ingestion, validation and split
# ----------------------------------------------------------------------
def ingest_and_split(cfg: dict) -> None:
    paths = cfg["paths"]
    split_cfg = cfg["split"]
    seed = cfg["preprocess"]["random_state"]

    # Create directories
    for p in [
        Path(paths["raw_dst"]).parent,
        Path(paths["splits_dir"]),
        Path(paths["manifest_path"]).parent,
    ]:
        ensure_dir(p)

    # Copy CSV
    src = Path(cfg["paths"]["raw_src"])
    dst = Path(cfg["paths"]["raw_dst"])
    if not src.is_file():
        raise FileNotFoundError(f"Source CSV not found at {src}")
    shutil.copy2(src, dst)

    # Load and validate
    df = pd.read_csv(dst)
    expected_shape = (178, 14)  # 13 features + target
    if df.shape != expected_shape:
        raise ValueError(f"Unexpected shape {df.shape}, expected {expected_shape}")
    if "target" not in df.columns:
        raise ValueError("Column 'target' missing")
    if not set(df["target"].unique()).issubset({0, 1, 2}):
        raise ValueError("Target column must contain only 0,1,2")
    if df.duplicated().any():
        raise ValueError("Duplicate rows found")
    if df.isnull().any().any():
        raise ValueError("Missing values detected")

    # Manifest
    manifest = {
        "sha256": compute_sha256(dst),
        "columns": {c: str(dt) for c, dt in zip(df.columns, df.dtypes)},
        "row_count": df.shape[0],
    }
    with open(paths["manifest_path"], "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Stratified split
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1 - split_cfg["train_frac"],
        random_state=seed,
    )
    X = df.drop(columns=["target"])
    y = df["target"]
    train_idx, temp_idx = next(sss.split(X, y))
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_temp, y_temp = X.iloc[temp_idx], y.iloc[temp_idx]

    # Remaining split into val / test
    val_frac_rel = split_cfg["val_frac"] / (split_cfg["val_frac"] + split_cfg["test_frac"])
    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1 - val_frac_rel,
        random_state=seed,
    )
    val_idx, test_idx = next(sss2.split(X_temp, y_temp))
    X_val, y_val = X_temp.iloc[val_idx], y_temp.iloc[val_idx]
    X_test, y_test = X_temp.iloc[test_idx], y_temp.iloc[test_idx]

    # Save splits safely
    for name, X_part, y_part in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        out_path = Path(paths["splits_dir"]) / f"{name}.csv"
        tmp_path = out_path.with_suffix(".tmp")
        pd.concat([X_part, y_part], axis=1).to_csv(tmp_path, index=False)
        tmp_path.replace(out_path)


# ----------------------------------------------------------------------
# 4️⃣  Pre‑processing pipeline
# ----------------------------------------------------------------------
def build_and_save_preprocess(cfg: dict) -> None:
    paths = cfg["paths"]
    seed = cfg["preprocess"]["random_state"]
    impute_strategy = cfg["preprocess"]["impute_strategy"]

    # Ensure output dirs
    ensure_dir(Path(paths["processed_path"]).parent)
    ensure_dir(Path(paths["pipeline_path"]).parent)

    # Load splits
    train_df = pd.read_csv(Path(paths["splits_dir"]) / "train.csv")
    val_df = pd.read_csv(Path(paths["splits_dir"]) / "val.csv")
    test_df = pd.read_csv(Path(paths["splits_dir"]) / "test.csv")

    X_train = train_df.drop(columns=["target"])
    X_val = val_df.drop(columns=["target"])
    X_test = test_df.drop(columns=["target"])

    # Build pipeline
    np.random.seed(seed)  # deterministic seed for any internal RNG
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=impute_strategy)),
            ("scaler", StandardScaler()),
        ]
    )
    pipeline.fit(X_train)

    # Transform and save
    X_train_scaled = pd.DataFrame(
        pipeline.transform(X_train), columns=X_train.columns
    )
    X_val_scaled = pd.DataFrame(pipeline.transform(X_val), columns=X_val.columns)
    X_test_scaled = pd.DataFrame(pipeline.transform(X_test), columns=X_test.columns)

    X_train_scaled.to_parquet(paths["processed_path"], engine="pyarrow")
    X_val_scaled.to_parquet(
        Path(paths["processed_path"]).with_name("X_val_scaled.parquet"),
        engine="pyarrow",
    )
    X_test_scaled.to_parquet(
        Path(paths["processed_path"]).with_name("X_test_scaled.parquet"),
        engine="pyarrow",
    )

    # Persist pipeline
    joblib.dump(pipeline, paths["pipeline_path"])


# ----------------------------------------------------------------------
# 5️⃣  Lightning DataModule
# ----------------------------------------------------------------------
class WineDataModule(pl.LightningDataModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg["model"]["batch_size"]
        self.paths = cfg["paths"]

    def setup(self, stage=None):
        # Load scaled parquet files
        train_path = Path(self.paths["processed_path"])
        val_path = train_path.with_name("X_val_scaled.parquet")
        test_path = train_path.with_name("X_test_scaled.parquet")

        X_train = pd.read_parquet(train_path).astype(np.float32).values
        X_val = pd.read_parquet(val_path).astype(np.float32).values
        X_test = pd.read_parquet(test_path).astype(np.float32).values

        # Dummy targets (not used)
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
# 6️⃣  DEC LightningModule
# ----------------------------------------------------------------------
class DEC(pl.LightningModule):
    def __init__(self, n_features: int, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.n_clusters = cfg["model"]["n_clusters"]
        self.latent_dim = cfg["model"]["latent_dim"]
        self.pretrain_epochs = cfg["model"]["pretrain_epochs"]
        self.update_interval = cfg["model"]["update_interval"]
        self.lr = cfg["model"]["learning_rate"]
        self.wd = cfg["model"]["weight_decay"]
        self.tol = cfg["model"]["tol"]
        self.device_type = None  # will be set by Lightning

        # Build encoder
        encoder_layers = []
        in_dim = n_features
        for h in cfg["model"]["hidden_dims"]:
            encoder_layers.append(torch.nn.Linear(in_dim, h))
            encoder_layers.append(torch.nn.ReLU())
            in_dim = h
        encoder_layers.append(torch.nn.Linear(in_dim, self.latent_dim))
        self.encoder = torch.nn.Sequential(*encoder_layers)

        # Build decoder (mirror)
        decoder_layers = []
        in_dim = self.latent_dim
        for h in reversed(cfg["model"]["hidden_dims"]):
            decoder_layers.append(torch.nn.Linear(in_dim, h))
            decoder_layers.append(torch.nn.ReLU())
            in_dim = h
        decoder_layers.append(torch.nn.Linear(in_dim, n_features))
        self.decoder = torch.nn.Sequential(*decoder_layers)

        # Cluster centroids (trainable)
        self.cluster_layer = torch.nn.Parameter(
            torch.randn(self.n_clusters, self.latent_dim)
        )
        torch.nn.init.xavier_uniform_(self.cluster_layer)

        self.p_target = None  # target distribution for KL loss

        # Reproducibility
        torch.manual_seed(cfg["model"]["random_state"])

    @staticmethod
    def compute_q(z: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        # Student‑t distribution (alpha = 1)
        # z: (N, d), centroids: (K, d)
        diff = z.unsqueeze(1) - centroids.unsqueeze(0)  # (N, K, d)
        dist_sq = torch.sum(diff ** 2, dim=2)  # (N, K)
        q = (1.0 + dist_sq).pow(-1)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

    @staticmethod
    def target_distribution(q: torch.Tensor) -> torch.Tensor:
        weight = q ** 2 / torch.sum(q, dim=0, keepdim=True)
        p = weight / torch.sum(weight, dim=1, keepdim=True)
        return p

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def configure_optimizers(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.wd
        )

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, z = self.forward(x)

        if self.current_epoch < self.pretrain_epochs:
            loss = F.mse_loss(x_hat, x)
        else:
            q = self.compute_q(z, self.cluster_layer)
            if self.p_target is None:
                # first clustering epoch – use q as a placeholder
                loss = F.kl_div(q.log(), q, reduction="batchmean")
            else:
                loss = F.kl_div(q.log(), self.p_target, reduction="batchmean")
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat, z = self.forward(x)

        if self.current_epoch < self.pretrain_epochs:
            loss = F.mse_loss(x_hat, x)
        else:
            q = self.compute_q(z, self.cluster_layer)
            if self.p_target is None:
                loss = F.kl_div(q.log(), q, reduction="batchmean")
            else:
                loss = F.kl_div(q.log(), self.p_target, reduction="batchmean")
        self.log("val_loss", loss, prog_bar=True)

    def on_train_epoch_end(self):
        # Update target distribution after pre‑training and every update_interval
        if self.current_epoch >= self.pretrain_epochs:
            if (self.current_epoch - self.pretrain_epochs) % self.update_interval == 0:
                self._update_target_distribution()

    def _update_target_distribution(self):
        # Run encoder on the whole training set
        self.eval()
        device = self.device
        dm = self.trainer.datamodule
        loader = dm.train_dataloader()
        all_z = []
        with torch.no_grad():
            for batch in loader:
                x, _ = batch
                x = x.to(device)
                _, z = self.forward(x)
                all_z.append(z.cpu())
        Z = torch.cat(all_z, dim=0).to(device)
        q = self.compute_q(Z, self.cluster_layer)
        self.p_target = self.target_distribution(q).detach()
        self.train()  # back to training mode

    def init_cluster_centroids(self, dataloader):
        # Called once after pre‑training
        self.eval()
        device = self.device
        all_z = []
        with torch.no_grad():
            for batch in dataloader:
                x, _ = batch
                x = x.to(device)
                _, z = self.forward(x)
                all_z.append(z.cpu())
        Z = torch.cat(all_z, dim=0).numpy()
        kmeans = KMeans(
            n_clusters=self.n_clusters,
            n_init=20,
            random_state=self.cfg["model"]["random_state"],
        )
        kmeans.fit(Z)
        centroids = torch.from_numpy(kmeans.cluster_centers_).to(device)
        self.cluster_layer.data = centroids
        self.train()


# ----------------------------------------------------------------------
# 7️⃣  Callback to initialise centroids after pre‑training
# ----------------------------------------------------------------------
class InitCentroidsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.called = False

    def on_train_epoch_end(self, trainer, pl_module):
        if not self.called and trainer.current_epoch == pl_module.pretrain_epochs - 1:
            pl_module.init_cluster_centroids(trainer.datamodule.train_dataloader())
            self.called = True


# ----------------------------------------------------------------------
# 8️⃣  Training orchestration
# ----------------------------------------------------------------------
def train_and_evaluate(cfg: dict) -> None:
    # ------------------------------------------------------------------
    # 8.1  Global reproducibility
    # ------------------------------------------------------------------
    seed = cfg["preprocess"]["random_state"]
    seed_everything(seed, workers=True)

    # ------------------------------------------------------------------
    # 8.2  DataModule
    # ------------------------------------------------------------------
    dm = WineDataModule(cfg)
    dm.setup()

    # ------------------------------------------------------------------
    # 8.3  Model
    # ------------------------------------------------------------------
    n_features = dm.train_dataset.tensors[0].shape[1]
    model = DEC(n_features=n_features, cfg=cfg)

    # ------------------------------------------------------------------
    # 8.4  Callbacks
    # ------------------------------------------------------------------
    early_stop_cb = EarlyStopping(
        monitor="val_loss",
        patience=cfg["model"]["early_stop_patience"],
        mode="min",
        verbose=False,
    )
    checkpoint_cb = ModelCheckpoint(
        dirpath="outputs/models",
        filename="dec_best",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    init_centroids_cb = InitCentroidsCallback()

    # ------------------------------------------------------------------
    # 8.5  MLflow logger
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    mlflow.start_run(run_name=cfg["mlflow"]["run_name"])
    run = mlflow.active_run()
    # log tags
    for k, v in cfg["mlflow"]["tags"].items():
        mlflow.set_tag(k, v)

    # ------------------------------------------------------------------
    # 8.6  Trainer
    # ------------------------------------------------------------------
    max_epochs = cfg["model"]["pretrain_epochs"] + cfg["model"]["cluster_epochs"]
    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=[early_stop_cb, checkpoint_cb, init_centroids_cb],
        logger=False,  # we use raw mlflow calls
        enable_checkpointing=True,
        log_every_n_steps=5,
        accelerator="cpu",  # change to "gpu" if CUDA is available
    )

    # ------------------------------------------------------------------
    # 8.7  Fit
    # ------------------------------------------------------------------
    trainer.fit(model, dm)

    # ------------------------------------------------------------------
    # 8.8  Save final artefacts (already saved by checkpoint callback)
    # ------------------------------------------------------------------
    best_ckpt = checkpoint_cb.best_model_path
    shutil.copy2(best_ckpt, cfg["paths"]["checkpoint_path"])

    # ------------------------------------------------------------------
    # 8.9  Evaluation on the whole dataset
    # ------------------------------------------------------------------
    evaluate_and_log(model, cfg, run.info.run_id)

    # ------------------------------------------------------------------
    # 8.10  Copy artefacts to downstream folder
    # ------------------------------------------------------------------
    dst_dir = Path("agent_workspace/trained_models")
    ensure_dir(dst_dir)
    shutil.copy2(cfg["paths"]["pipeline_path"], dst_dir / Path(cfg["paths"]["pipeline_path"]).name)
    shutil.copy2(cfg["paths"]["checkpoint_path"], dst_dir / Path(cfg["paths"]["checkpoint_path"]).name)

    mlflow.end_run()


# ----------------------------------------------------------------------
# 9️⃣  Evaluation & metric logging
# ----------------------------------------------------------------------
def evaluate_and_log(model: DEC, cfg: dict, run_id: str) -> None:
    # Load raw data (with true labels)
    raw_df = pd.read_csv(Path(cfg["paths"]["raw_dst"]))
    X = raw_df.drop(columns=["target"]).values.astype(np.float32)
    y_true = raw_df["target"].values

    # Load preprocessing pipeline and transform
    pipeline = joblib.load(cfg["paths"]["pipeline_path"])
    X_scaled = pipeline.transform(X)

    device = torch.device("cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        latent = model.encode(torch.from_numpy(X_scaled).to(device)).cpu().numpy()

    # Soft assignments and hard labels
    q = DEC.compute_q(
        torch.from_numpy(latent).to(device),
        model.cluster_layer.detach().cpu(),
    )
    y_pred = torch.argmax(q, dim=1).cpu().numpy()

    # Metrics
    ari = adjusted_rand_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    sil = silhouette_score(latent, y_pred)

    metrics = {
        "ARI": round(float(ari), 5),
        "AMI": round(float(ami), 5),
        "Silhouette": round(float(sil), 5),
    }

    # Save JSON
    ensure_dir(Path(cfg["paths"]["metrics_path"]).parent)
    with open(cfg["paths"]["metrics_path"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Log to MLflow
    mlflow.start_run(run_id=run_id)
    mlflow.log_metrics(metrics)
    mlflow.log_artifact(cfg["paths"]["metrics_path"], artifact_path="metrics")
    mlflow.end_run()


# ----------------------------------------------------------------------
# 10️⃣  Main entry point
# ----------------------------------------------------------------------
def main():
    cfg = CONFIG
    ingest_and_split(cfg)
    build_and_save_preprocess(cfg)
    train_and_evaluate(cfg)
    print("[INFO] Pipeline completed successfully.")
    print("[INFO] Artefacts stored under ./outputs and ./agent_workspace/trained_models")


if __name__ == "__main__":
    # Suppress unnecessary warnings for a clean console output
    warnings.filterwarnings("ignore")
    main()
