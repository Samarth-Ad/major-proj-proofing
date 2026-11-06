
# -*- coding: utf-8 -*-
"""
Wine DEC clustering pipeline (single‑file implementation).

The script follows the reproducible blueprint described in the
project specification.  All artefacts are written under the repository
root and the final model + preprocessing pipeline are copied to
./agent_workspace/trained_models for downstream use.

Run with:
    python wine_dec_pipeline.py
"""

import json
import hashlib
import shutil
import warnings
import sys
from pathlib import Path

# ----------------------------------------------------------------------
# 0️⃣  Imports (all required packages must be installed beforehand)
# ----------------------------------------------------------------------
import yaml
import mlflow
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    adjusted_mutual_info_score,
    silhouette_score,
)

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule, Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback
from pytorch_lightning.loggers import MLFlowLogger
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------------------------------------------------
# 1️⃣  Helper utilities
# ----------------------------------------------------------------------
def ensure_dir(p: Path) -> None:
    """Create a directory (including parents) if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)


def compute_sha256(file_path: Path) -> str:
    """Return the SHA‑256 hash of a file."""
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config() -> dict:
    """Load configuration from configs/config.yaml (ASCII only)."""
    cfg_path = Path("configs") / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def flatten_cfg(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Flatten a nested dict for easy logging."""
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_cfg(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items


# ----------------------------------------------------------------------
# 2️⃣  Data ingestion, validation and stratified split
# ----------------------------------------------------------------------
def ingest_and_split(cfg: dict) -> None:
    p = cfg["paths"]
    split_cfg = cfg["split"]
    seed = cfg["preprocess"]["random_state"]

    # ------------------------------------------------------------------
    # Create required directories
    # ------------------------------------------------------------------
    for d in [
        Path(p["raw_dst"]).parent,
        Path(p["splits_dir"]),
        Path(p["manifest_path"]).parent,
    ]:
        ensure_dir(d)

    # ------------------------------------------------------------------
    # Copy CSV from absolute source to project folder
    # ------------------------------------------------------------------
    src = Path(p["raw_src"])
    dst = Path(p["raw_dst"])
    if not src.is_file():
        raise FileNotFoundError(f"Source CSV not found at {src}")
    shutil.copy2(src, dst)

    # ------------------------------------------------------------------
    # Load and validate
    # ------------------------------------------------------------------
    df = pd.read_csv(dst)
    expected_shape = (178, 14)  # 13 features + target (original Wine dataset)
    if df.shape != expected_shape:
        raise ValueError(f"Unexpected shape {df.shape}, expected {expected_shape}")
    if "target" not in df.columns:
        raise ValueError("Column 'target' missing from the dataset")
    if not set(df["target"].unique()).issubset({0, 1, 2}):
        raise ValueError("Target column must contain only 0, 1, 2")
    if df.duplicated().any():
        raise ValueError("Duplicate rows detected")
    if df.isnull().any().any():
        raise ValueError("Missing values detected")

    # ------------------------------------------------------------------
    # Data manifest (SHA‑256, column dtypes, row count)
    # ------------------------------------------------------------------
    manifest = {
        "sha256": compute_sha256(dst),
        "columns": {c: str(dt) for c, dt in zip(df.columns, df.dtypes)},
        "row_count": df.shape[0],
    }
    with open(p["manifest_path"], "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # ------------------------------------------------------------------
    # Stratified split (train / val / test)
    # ------------------------------------------------------------------
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

    # Validation / test split (relative fractions)
    val_frac_rel = split_cfg["val_frac"] / (split_cfg["val_frac"] + split_cfg["test_frac"])
    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1 - val_frac_rel,
        random_state=seed,
    )
    val_idx, test_idx = next(sss2.split(X_temp, y_temp))
    X_val, y_val = X_temp.iloc[val_idx], y_temp.iloc[val_idx]
    X_test, y_test = X_temp.iloc[test_idx], y_temp.iloc[test_idx]

    # Write splits safely (write to .tmp then rename)
    for name, X_part, y_part in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        out_path = Path(p["splits_dir"]) / f"{name}.csv"
        tmp_path = out_path.with_suffix(".tmp")
        pd.concat([X_part, y_part], axis=1).to_csv(tmp_path, index=False)
        tmp_path.replace(out_path)


# ----------------------------------------------------------------------
# 3️⃣  Pre‑processing pipeline (Imputer + StandardScaler)
# ----------------------------------------------------------------------
def build_and_save_preprocess(cfg: dict) -> None:
    p = cfg["paths"]
    seed = cfg["preprocess"]["random_state"]
    impute_strategy = cfg["preprocess"]["impute_strategy"]

    # Ensure output directories exist
    ensure_dir(Path(p["processed_path"]).parent)
    ensure_dir(Path(p["pipeline_path"]).parent)

    # Load split CSVs
    train_df = pd.read_csv(Path(p["splits_dir"]) / "train.csv")
    val_df = pd.read_csv(Path(p["splits_dir"]) / "val.csv")
    test_df = pd.read_csv(Path(p["splits_dir"]) / "test.csv")

    X_train = train_df.drop(columns=["target"])
    X_val = val_df.drop(columns=["target"])
    X_test = test_df.drop(columns=["target"])

    # Deterministic pipeline
    np.random.seed(seed)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=impute_strategy)),
            ("scaler", StandardScaler()),
        ]
    )
    pipeline.fit(X_train)

    # Transform and persist
    X_train_scaled = pd.DataFrame(pipeline.transform(X_train), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(pipeline.transform(X_val), columns=X_val.columns)
    X_test_scaled = pd.DataFrame(pipeline.transform(X_test), columns=X_test.columns)

    X_train_scaled.to_parquet(p["processed_path"], engine="pyarrow")
    X_val_scaled.to_parquet(
        Path(p["processed_path"]).with_name("X_val_scaled.parquet"), engine="pyarrow"
    )
    X_test_scaled.to_parquet(
        Path(p["processed_path"]).with_name("X_test_scaled.parquet"), engine="pyarrow"
    )

    # Save the fitted pipeline
    joblib.dump(pipeline, p["pipeline_path"])


# ----------------------------------------------------------------------
# 4️⃣  Lightning DataModule (loads scaled Parquet files)
# ----------------------------------------------------------------------
class WineDataModule(LightningDataModule):
    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.batch_size = cfg["model"]["batch_size"]
        self.paths = cfg["paths"]

    def setup(self, stage=None):
        train_path = Path(self.paths["processed_path"])
        val_path = train_path.with_name("X_val_scaled.parquet")
        test_path = train_path.with_name("X_test_scaled.parquet")

        X_train = pd.read_parquet(train_path).astype(np.float32).values
        X_val = pd.read_parquet(val_path).astype(np.float32).values
        X_test = pd.read_parquet(test_path).astype(np.float32).values

        # Dummy targets (not used for clustering)
        y_dummy_train = np.zeros(len(X_train), dtype=np.int64)
        y_dummy_val = np.zeros(len(X_val), dtype=np.int64)
        y_dummy_test = np.zeros(len(X_test), dtype=np.int64)

        self.train_dataset = TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_dummy_train)
        )
        self.val_dataset = TensorDataset(
            torch.from_numpy(X_val), torch.from_numpy(y_dummy_val)
        )
        self.test_dataset = TensorDataset(
            torch.from_numpy(X_test), torch.from_numpy(y_dummy_test)
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
# 5️⃣  DEC LightningModule (pure Lightning implementation)
# ----------------------------------------------------------------------
class DEC(LightningModule):
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

        # -------------------- Encoder --------------------
        encoder_layers = []
        in_dim = n_features
        for h in cfg["model"]["hidden_dims"]:
            encoder_layers.append(torch.nn.Linear(in_dim, h))
            encoder_layers.append(torch.nn.ReLU())
            in_dim = h
        encoder_layers.append(torch.nn.Linear(in_dim, self.latent_dim))
        self.encoder = torch.nn.Sequential(*encoder_layers)

        # -------------------- Decoder (mirror) --------------------
        decoder_layers = []
        in_dim = self.latent_dim
        for h in reversed(cfg["model"]["hidden_dims"]):
            decoder_layers.append(torch.nn.Linear(in_dim, h))
            decoder_layers.append(torch.nn.ReLU())
            in_dim = h
        decoder_layers.append(torch.nn.Linear(in_dim, n_features))
        self.decoder = torch.nn.Sequential(*decoder_layers)

        # -------------------- Trainable centroids --------------------
        self.cluster_layer = torch.nn.Parameter(
            torch.randn(self.n_clusters, self.latent_dim)
        )
        torch.nn.init.xavier_uniform_(self.cluster_layer)

        # Placeholder for target distribution
        self.p_target = None

        # Reproducibility
        torch.manual_seed(cfg["model"]["random_state"])

    @staticmethod
    def compute_q(z: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
        """Student‑t soft assignment (α = 1)."""
        diff = z.unsqueeze(1) - centroids.unsqueeze(0)  # (N, K, d)
        dist_sq = torch.sum(diff ** 2, dim=2)            # (N, K)
        q = (1.0 + dist_sq).pow(-1)
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

    @staticmethod
    def target_distribution(q: torch.Tensor) -> torch.Tensor:
        """Compute the target distribution p from q."""
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
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.wd)

    def training_step(self, batch, batch_idx):
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
        """Update target distribution during clustering phase."""
        if self.current_epoch >= self.pretrain_epochs:
            if (self.current_epoch - self.pretrain_epochs) % self.update_interval == 0:
                self._update_target_distribution()

    def _update_target_distribution(self):
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
        self.train()

    def init_cluster_centroids(self, dataloader):
        """Run K‑Means on the latent space after pre‑training."""
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
# 6️⃣  Callback that triggers centroid initialisation after pre‑training
# ----------------------------------------------------------------------
class InitCentroidsCallback(Callback):
    def __init__(self):
        super().__init__()
        self.called = False

    def on_train_epoch_end(self, trainer, pl_module):
        if (not self.called) and (trainer.current_epoch == pl_module.pretrain_epochs - 1):
            pl_module.init_cluster_centroids(trainer.datamodule.train_dataloader())
            self.called = True


# ----------------------------------------------------------------------
# 7️⃣  MLflow helper utilities
# ----------------------------------------------------------------------
def start_mlflow_run(cfg: dict):
    # Resolve possible placeholder $(pwd) in tracking_uri
    tracking_uri = cfg["mlflow"]["tracking_uri"]
    if "$(pwd)" in tracking_uri:
        tracking_uri = tracking_uri.replace("$(pwd)", str(Path.cwd()))
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    mlflow.start_run(run_name=cfg["mlflow"]["run_name"])
    run = mlflow.active_run()
    for k, v in cfg["mlflow"]["tags"].items():
        mlflow.set_tag(k, v)
    return run


def log_params_from_cfg(cfg: dict):
    flat = flatten_cfg(cfg)
    mlflow.log_params(flat)


# ----------------------------------------------------------------------
# 8️⃣  Evaluation & metric logging
# ----------------------------------------------------------------------
def evaluate_and_log(model: DEC, cfg: dict, run_id: str):
    # Load raw data (contains true labels)
    raw_df = pd.read_csv(Path(cfg["paths"]["raw_dst"]))
    X = raw_df.drop(columns=["target"]).values.astype(np.float32)
    y_true = raw_df["target"].values

    # Load preprocessing pipeline and transform the whole dataset
    pipeline = joblib.load(cfg["paths"]["pipeline_path"])
    X_scaled = pipeline.transform(X)

    device = torch.device("cpu")
    model.to(device)
    model.eval()

    with torch.no_grad():
        latent = model.encode(torch.from_numpy(X_scaled).to(device)).cpu().numpy()

    q = DEC.compute_q(
        torch.from_numpy(latent).to(device),
        model.cluster_layer.detach().cpu(),
    )
    y_pred = torch.argmax(q, dim=1).cpu().numpy()

    ari = adjusted_rand_score(y_true, y_pred)
    ami = adjusted_mutual_info_score(y_true, y_pred)
    sil = silhouette_score(latent, y_pred)

    metrics = {
        "ARI": round(float(ari), 5),
        "AMI": round(float(ami), 5),
        "Silhouette": round(float(sil), 5),
    }

    # Persist metrics JSON
    ensure_dir(Path(cfg["paths"]["metrics_path"]).parent)
    with open(cfg["paths"]["metrics_path"], "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Log metrics and artefact to the same MLflow run
    mlflow.start_run(run_id=run_id)
    mlflow.log_metrics(metrics)
    mlflow.log_artifact(cfg["paths"]["metrics_path"], artifact_path="metrics")
    mlflow.end_run()


# ----------------------------------------------------------------------
# 9️⃣  Training orchestration (sets seeds, logger, trainer, etc.)
# ----------------------------------------------------------------------
def train_and_evaluate(cfg: dict) -> None:
    # Global reproducibility
    seed = cfg["preprocess"]["random_state"]
    seed_everything(seed, workers=True)

    # DataModule
    dm = WineDataModule(cfg)
    dm.setup()

    # Model
    n_features = dm.train_dataset.tensors[0].shape[1]
    model = DEC(n_features=n_features, cfg=cfg)

    # Callbacks
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

    # MLflow run
    run = start_mlflow_run(cfg)
    log_params_from_cfg(cfg)

    mlflow_logger = MLFlowLogger(
        experiment_name=cfg["mlflow"]["experiment_name"],
        tracking_uri=mlflow.get_tracking_uri(),
        run_id=run.info.run_id,
    )

    # Trainer
    max_epochs = cfg["model"]["pretrain_epochs"] + cfg["model"]["cluster_epochs"]
    trainer = Trainer(
        max_epochs=max_epochs,
        callbacks=[early_stop_cb, checkpoint_cb, init_centroids_cb],
        logger=mlflow_logger,
        enable_checkpointing=True,
        log_every_n_steps=5,
        accelerator="cpu",  # change to "gpu" if CUDA is available
    )

    # Fit
    trainer.fit(model, dm)

    # Copy best checkpoint to canonical location
    best_ckpt_path = checkpoint_cb.best_model_path
    shutil.copy2(best_ckpt_path, cfg["paths"]["checkpoint_path"])

    # Evaluation & MLflow logging
    evaluate_and_log(model, cfg, run.info.run_id)

    # Copy artefacts to downstream folder
    downstream_dir = Path("agent_workspace") / "trained_models"
    ensure_dir(downstream_dir)
    shutil.copy2(cfg["paths"]["pipeline_path"], downstream_dir / Path(cfg["paths"]["pipeline_path"]).name)
    shutil.copy2(cfg["paths"]["checkpoint_path"], downstream_dir / Path(cfg["paths"]["checkpoint_path"]).name)

    mlflow.end_run()


# ----------------------------------------------------------------------
# 🔟  Main entry point
# ----------------------------------------------------------------------
def main():
    warnings.filterwarnings("ignore")
    cfg = load_config()
    ingest_and_split(cfg)
    build_and_save_preprocess(cfg)
    train_and_evaluate(cfg)
    print("[INFO] Pipeline completed successfully.")
    print("[INFO] Artefacts stored under ./outputs and ./agent_workspace/trained_models")


if __name__ == "__main__":
    main()
