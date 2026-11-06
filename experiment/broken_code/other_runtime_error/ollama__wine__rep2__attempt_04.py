
"""
Wine DEC clustering pipeline (stand‑alone, no Lightning dependency)

This script follows the executive‑summary design and implements the full
end‑to‑end flow:

1️⃣  Copy the user‑provided CSV to the project data folder and validate it.  
2️⃣  Create a stratified train/val/test split (70/20/10).  
3️⃣  Build a scikit‑learn preprocessing Pipeline (SimpleImputer + StandardScaler)  
    and persist it with joblib.  
4️⃣  Train a Deep‑Embedded Clustering (DEC) model using plain PyTorch:  
   – pre‑training auto‑encoder (reconstruction loss)  
   – clustering phase (KL‑divergence loss) with K‑Means centroid init.  
5️⃣  Evaluate the final model on the whole data set (ARI, AMI, Silhouette).  
6️⃣  Save the best checkpoint and the preprocessing pipeline under
    ``./agent_workspace/trained_models``.  
7️⃣  (Optional) expose a tiny Gradio demo for inference.

The script is self‑contained – it creates all required folders, reads the
configuration from an in‑script dictionary (mirroring the YAML spec) and
does not rely on any external files.  Run it with:

    python wine_dec_pipeline.py
"""

# --------------------------------------------------------------------------- #
# 0️⃣ Imports
# --------------------------------------------------------------------------- #
import hashlib
import json
import os
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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
# 1️⃣ Configuration (mirrors the YAML from the spec)
# --------------------------------------------------------------------------- #
CFG = {
    "paths": {
        # absolute source supplied by the user (must exist on the host machine)
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
        "early_stop_patience": 5,
    },
    "split": {
        "train_frac": 0.70,
        "val_frac": 0.20,
        "test_frac": 0.10,
        "stratify": True,
    },
}


# --------------------------------------------------------------------------- #
# 2️⃣ Utility helpers
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
        for chunk in iter(lambda: f.read(8192), b""):
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
    rs = CFG["preprocess"]["random_state"]

    # First split off the test+val part
    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=CFG["split"]["val_frac"] + CFG["split"]["test_frac"],
        random_state=rs,
    )
    train_idx, temp_idx = next(sss.split(X, y))

    # Now split the remaining part into val / test
    val_frac_rel = CFG["split"]["val_frac"] / (
        CFG["split"]["val_frac"] + CFG["split"]["test_frac"]
    )
    sss_val = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1 - val_frac_rel,
        random_state=rs,
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
# 3️⃣ DEC model (plain PyTorch)
# --------------------------------------------------------------------------- #
class DEC(nn.Module):
    """Deep Embedded Clustering (auto‑encoder + clustering layer)."""

    def __init__(self, n_features: int, cfg: dict):
        super().__init__()
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
        # will be overwritten after pre‑training with K‑Means centroids
        self.cluster_layer = nn.Parameter(
            torch.randn(self.n_clusters, latent_dim)
        )

    # ------------------------------------------------------------------- #
    # Forward passes
    # ------------------------------------------------------------------- #
    def forward(self, x):
        """Reconstruction + latent embedding."""
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z

    def encode(self, x):
        return self.encoder(x)

    def compute_q(self, z):
        """Student‑t distribution (α=1) soft assignments."""
        # z: (batch, latent_dim); centroids: (n_clusters, latent_dim)
        expanded_z = z.unsqueeze(1)                     # (batch, 1, latent)
        expanded_c = self.cluster_layer.unsqueeze(0)   # (1, n_clusters, latent)
        dist = torch.sum((expanded_z - expanded_c) ** 2, dim=2)  # (batch, n_clusters)
        q = 1.0 / (1.0 + dist)
        q = q ** ((1 + 1) / 2)  # α = 1
        q = q / torch.sum(q, dim=1, keepdim=True)
        return q

    @staticmethod
    def target_distribution(q):
        """Compute the target distribution p from soft assignments q."""
        weight = q ** 2 / torch.sum(q, dim=0, keepdim=True)
        p = weight / torch.sum(weight, dim=1, keepdim=True)
        return p

    # ------------------------------------------------------------------- #
    # Helper for centroid initialization after pre‑training
    # ------------------------------------------------------------------- #
    def init_cluster_centroids(self, dataloader, device):
        self.eval()
        all_z = []
        with torch.no_grad():
            for batch in dataloader:
                batch = batch.to(device)
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
            kmeans.cluster_centers_, dtype=torch.float32, device=device
        )
        self.cluster_layer.data = centroids
        print("[INFO] Cluster centroids initialized with K‑Means.")


# --------------------------------------------------------------------------- #
# 4️⃣ Training utilities
# --------------------------------------------------------------------------- #
def train_autoencoder(
    model: DEC,
    dataloader: DataLoader,
    cfg: dict,
    device: torch.device,
) -> dict:
    """Pre‑training phase – minimize reconstruction MSE."""
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    best_loss = float("inf")
    patience = cfg["early_stop_patience"]
    wait = 0
    history = {"pretrain_loss": []}

    for epoch in range(cfg["pretrain_epochs"]):
        model.train()
        epoch_losses = []
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            x_hat, _ = model(batch)
            loss = F.mse_loss(x_hat, batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        epoch_loss = np.mean(epoch_losses)
        history["pretrain_loss"].append(epoch_loss)
        print(f"[PRETRAIN] Epoch {epoch+1:02d}/{cfg['pretrain_epochs']}  loss={epoch_loss:.6f}")

        # early stopping check
        if epoch_loss < best_loss - cfg["tol"]:
            best_loss = epoch_loss
            wait = 0
            # save temporary best weights
            torch.save(model.state_dict(), "tmp_pretrain.ckpt")
        else:
            wait += 1
            if wait >= patience:
                print("[PRETRAIN] Early stopping triggered.")
                break

    # load best weights
    model.load_state_dict(torch.load("tmp_pretrain.ckpt"))
    os.remove("tmp_pretrain.ckpt")
    return history


def train_clustering(
    model: DEC,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: dict,
    device: torch.device,
) -> dict:
    """Clustering phase – minimize KL divergence between q and target p."""
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    best_loss = float("inf")
    patience = cfg["early_stop_patience"]
    wait = 0
    history = {"cluster_loss": []}
    update_interval = cfg["update_interval"]
    p_distribution = None  # will be set after first update

    for epoch in range(cfg["cluster_epochs"]):
        model.train()
        epoch_losses = []

        # ---- Update target distribution every `update_interval` epochs ----
        if epoch % update_interval == 0:
            model.eval()
            all_z = []
            with torch.no_grad():
                for batch in train_loader:
                    batch = batch.to(device)
                    _, z = model(batch)
                    all_z.append(z)
            z_all = torch.cat(all_z, dim=0)
            q_all = model.compute_q(z_all)
            p_distribution = DEC.target_distribution(q_all).detach()
            model.train()

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            _, z = model(batch)
            q = model.compute_q(z)
            # use the most recent target distribution (same size as batch)
            # we compute p for the whole dataset once per interval, then slice
            # to match the batch size via indexing (simpler: recompute per batch)
            # For simplicity we recompute p on‑the‑fly:
            p = DEC.target_distribution(q).detach()
            loss = F.kl_div(q.log(), p, reduction="batchmean")
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        epoch_loss = np.mean(epoch_losses)
        history["cluster_loss"].append(epoch_loss)
        print(f"[CLUSTER] Epoch {epoch+1:02d}/{cfg['cluster_epochs']}  loss={epoch_loss:.6f}")

        # early stopping based on validation loss
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                _, z = model(batch)
                q = model.compute_q(z)
                p_val = DEC.target_distribution(q).detach()
                loss = F.kl_div(q.log(), p_val, reduction="batchmean")
                val_losses.append(loss.item())
        val_loss = np.mean(val_losses)

        if val_loss < best_loss - cfg["tol"]:
            best_loss = val_loss
            wait = 0
            torch.save(model.state_dict(), "tmp_cluster.ckpt")
        else:
            wait += 1
            if wait >= patience:
                print("[CLUSTER] Early stopping triggered.")
                break

    # load best weights
    model.load_state_dict(torch.load("tmp_cluster.ckpt"))
    os.remove("tmp_cluster.ckpt")
    return history


# --------------------------------------------------------------------------- #
# 5️⃣ Main pipeline execution
# --------------------------------------------------------------------------- #
def main():
    # ------------------------------------------------- #
    # Global reproducibility
    # ------------------------------------------------- #
    ensure_dirs()
    np.random.seed(CFG["preprocess"]["random_state"])
    torch.manual_seed(CFG["model"]["random_state"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

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

    # ------------------------------------------------- #
    # 3️⃣ Prepare DataLoaders
    # ------------------------------------------------- #
    train_ds = TensorDataset(torch.from_numpy(X_train.astype(np.float32)))
    val_ds = TensorDataset(torch.from_numpy(X_val.astype(np.float32)))
    train_loader = DataLoader(
        train_ds, batch_size=CFG["model"]["batch_size"], shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=CFG["model"]["batch_size"], shuffle=False, drop_last=False
    )

    # ------------------------------------------------- #
    # 4️⃣ Model definition
    # ------------------------------------------------- #
    n_features = X_train.shape[1]
    dec = DEC(n_features=n_features, cfg=CFG["model"]).to(device)

    # ------------------------------------------------- #
    # 5️⃣ Pre‑training (auto‑encoder)
    # ------------------------------------------------- #
    print("\n=== Pre‑training auto‑encoder ===")
    train_autoencoder(dec, train_loader, CFG["model"], device)

    # ------------------------------------------------- #
    # 6️⃣ Initialise cluster centroids with K‑Means
    # ------------------------------------------------- #
    dec.init_cluster_centroids(train_loader, device)

    # ------------------------------------------------- #
    # 7️⃣ Clustering phase
    # ------------------------------------------------- #
    print("\n=== Clustering phase ===")
    train_clustering(dec, train_loader, val_loader, CFG["model"], device)

    # ------------------------------------------------- #
    # 8️⃣ Save best checkpoint to the mandated folder
    # ------------------------------------------------- #
    ckpt_path = Path(CFG["paths"]["ckpt_out"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dec.state_dict(), ckpt_path)
    print(f"[INFO] Best model checkpoint saved to {ckpt_path}")

    # Also copy artefacts to the top‑level trained_models directory
    shutil.copy2(CFG["paths"]["pipeline_out"], CFG["paths"]["trained_models_dir"])
    shutil.copy2(ckpt_path, CFG["paths"]["trained_models_dir"])
    print(f"[INFO] Artefacts also placed in {CFG['paths']['trained_models_dir']}")

    # ------------------------------------------------- #
    # 9️⃣ Evaluation on the whole dataset
    # ------------------------------------------------- #
    full_X = pipeline.transform(df[feature_cols])
    full_tensor = torch.from_numpy(full_X.astype(np.float32)).to(device)

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
    # 10️⃣ Optional Gradio demo for inference
    # ------------------------------------------------- #
    def predict(csv_file):
        """Accept a CSV (no target column) and return cluster IDs."""
        user_df = pd.read_csv(csv_file.name)
        missing = set(feature_cols) - set(user_df.columns)
        if missing:
            return {"error": f"Missing columns: {sorted(missing)}"}
        X_user = pipeline.transform(user_df[feature_cols])
        X_user_tensor = torch.from_numpy(X_user.astype(np.float32)).to(device)
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
        description=(
            "Upload a CSV containing the same 13 feature columns as the training data "
            "to obtain cluster assignments produced by the DEC model."
        ),
    )
    # Launch only when the script is executed directly (not during unit tests)
    demo.launch(share=False, debug=False)


if __name__ == "__main__":
    main()
