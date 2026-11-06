
"""
Wine‑Clustering pipeline
------------------------

A self‑contained, “run‑once” script that:

1. Loads the raw CSV (expected at ``data/raw/wine.csv``).
2. Computes a SHA‑256 checksum for provenance.
3. Preprocesses the data (median imputation + RobustScaler).
4. Trains a shallow PyTorch auto‑encoder (latent dim = 10) with early‑stop.
5. Encodes the whole dataset and stores the latent vectors.
6. Fits a K‑Means model (k = 3) on the latent space.
7. Evaluates the clustering (ARI, NMI, Silhouette) and writes a markdown report.
8. Persists the encoder checkpoint, the K‑Means model and the metrics.

All artefacts are written under the ``agent_workspace`` folder, which is
created automatically if it does not exist.

Run with:

    python wine_clustering_pipeline.py
"""

import json
import random
import hashlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

# --------------------------------------------------------------------------- #
# Configuration (all paths are relative to the repository root)
# --------------------------------------------------------------------------- #
CFG = {
    "seed": 42,
    "data": {
        "raw_path": "data/raw/wine.csv",
        "target_column": "target",          # column name in the original CSV
    },
    "preprocess": {
        "imputer_strategy": "median",
        "scaler_type": "robust",
    },
    "autoencoder": {
        "latent_dim": 10,
        "hidden_dims": [64, 32],
        "activation": "ReLU",               # exact class name in torch.nn
        "dropout": 0.0,
        "batch_size": 32,
        "max_epochs": 150,
        "early_stop_patience": 10,
        "learning_rate": 1e-3,
        "optimizer": "Adam",
        "loss": "MSELoss",
        "device": "cpu",                    # overridden to "cuda" if available
        "checkpoint_path": "agent_workspace/trained_models/encoder.ckpt",
    },
    "clustering": {
        "n_clusters": 3,
        "init": "k-means++",
        "n_init": 20,
        "max_iter": 500,
        "model_path": "agent_workspace/trained_models/kmeans.pkl",
    },
    "paths": {
        "preprocess_pipeline": "agent_workspace/trained_models/preprocess_pipeline.pkl",
        "latent_vectors": "agent_workspace/artifacts/latent_vectors.npy",
        "metrics": "agent_workspace/artifacts/metrics.json",
        "checksum": "agent_workspace/artifacts/checksum.json",
        "report": "agent_workspace/reports/final_report.md",
    },
    "evaluation": {
        "metrics": ["adjusted_rand_score", "normalized_mutual_info_score", "silhouette_score"],
    },
}


# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def set_global_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: Path) -> None:
    """Make sure the parent directory of *path* exists."""
    path.parent.mkdir(parents=True, exist_ok=True)


def compute_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# 1. Data loading
# --------------------------------------------------------------------------- #
def load_raw_data(cfg: dict) -> pd.DataFrame:
    raw_path = Path(cfg["data"]["raw_path"])
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw CSV not found at {raw_path.resolve()}")
    df = pd.read_csv(raw_path)
    target = cfg["data"]["target_column"]
    if target not in df.columns:
        raise KeyError(f"Target column '{target}' not found in CSV")
    return df


# --------------------------------------------------------------------------- #
# 2. Pre‑processing pipeline
# --------------------------------------------------------------------------- #
class PreprocessPipeline:
    def __init__(self, cfg: dict):
        imputer = SimpleImputer(strategy=cfg["preprocess"]["imputer_strategy"])
        scaler = RobustScaler()
        self.pipeline = Pipeline([("imputer", imputer), ("scaler", scaler)])

    def fit(self, X: np.ndarray) -> None:
        self.pipeline.fit(X)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.transform(X).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.fit_transform(X).astype(np.float32)

    def save(self, path: str) -> None:
        ensure_dir(Path(path))
        joblib.dump(self.pipeline, path)


# --------------------------------------------------------------------------- #
# 3. Auto‑encoder definition
# --------------------------------------------------------------------------- #
class WineAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        latent_dim: int,
        activation: str = "ReLU",
        dropout: float = 0.0,
    ):
        super().__init__()
        act_fn = getattr(nn, activation)  # case‑sensitive lookup

        # Encoder
        enc_layers = []
        prev = input_dim
        for h in hidden_dims:
            enc_layers.append(nn.Linear(prev, h))
            enc_layers.append(act_fn())
            if dropout > 0:
                enc_layers.append(nn.Dropout(dropout))
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (mirror of encoder)
        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers.append(nn.Linear(prev, h))
            dec_layers.append(act_fn())
            if dropout > 0:
                dec_layers.append(nn.Dropout(dropout))
            prev = h
        dec_layers.append(nn.Linear(prev, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon


# --------------------------------------------------------------------------- #
# 4. Training utilities
# --------------------------------------------------------------------------- #
def train_autoencoder(cfg: dict, X: np.ndarray) -> None:
    """Trains the auto‑encoder and saves the *encoder* checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else cfg["autoencoder"]["device"])
    input_dim = X.shape[1]

    # Train/validation split (90/10)
    n = X.shape[0]
    idx = np.random.permutation(n)
    split = int(0.9 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    X_train = torch.from_numpy(X[train_idx])
    X_val = torch.from_numpy(X[val_idx])

    train_loader = DataLoader(TensorDataset(X_train), batch_size=cfg["autoencoder"]["batch_size"], shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val), batch_size=cfg["autoencoder"]["batch_size"], shuffle=False)

    # Model, loss, optimizer
    model = WineAutoEncoder(
        input_dim=input_dim,
        hidden_dims=cfg["autoencoder"]["hidden_dims"],
        latent_dim=cfg["autoencoder"]["latent_dim"],
        activation=cfg["autoencoder"]["activation"],
        dropout=cfg["autoencoder"]["dropout"],
    ).to(device)

    loss_fn = getattr(nn, cfg["autoencoder"]["loss"])()
    optim_cls = getattr(torch.optim, cfg["autoencoder"]["optimizer"])
    optimizer = optim_cls(model.parameters(), lr=cfg["autoencoder"]["learning_rate"])

    best_val = float("inf")
    patience = 0

    for epoch in range(1, cfg["autoencoder"]["max_epochs"] + 1):
        # ----- training -----
        model.train()
        train_losses = []
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            _, recon = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # ----- validation -----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                _, recon = model(batch)
                loss = loss_fn(recon, batch)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train loss: {avg_train:.6f} | Val loss: {avg_val:.6f}")

        # Early stopping
        if avg_val < best_val - 1e-6:
            best_val = avg_val
            patience = 0
            # Save best encoder state
            ensure_dir(Path(cfg["autoencoder"]["checkpoint_path"]))
            torch.save(model.encoder.state_dict(), cfg["autoencoder"]["checkpoint_path"])
        else:
            patience += 1
            if patience >= cfg["autoencoder"]["early_stop_patience"]:
                print(f"Early stopping triggered at epoch {epoch}")
                break


def encode_dataset(cfg: dict, X: np.ndarray) -> np.ndarray:
    """Loads the saved encoder and returns latent vectors for *X*."""
    device = torch.device("cuda" if torch.cuda.is_available() else cfg["autoencoder"]["device"])
    input_dim = X.shape[1]

    encoder = WineAutoEncoder(
        input_dim=input_dim,
        hidden_dims=cfg["autoencoder"]["hidden_dims"],
        latent_dim=cfg["autoencoder"]["latent_dim"],
        activation=cfg["autoencoder"]["activation"],
        dropout=cfg["autoencoder"]["dropout"],
    ).to(device)

    ckpt_path = Path(cfg["autoencoder"]["checkpoint_path"])
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Encoder checkpoint not found at {ckpt_path}")
    encoder.load_state_dict(torch.load(ckpt_path, map_location=device))
    encoder.eval()

    loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=cfg["autoencoder"]["batch_size"], shuffle=False)
    latent = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            z, _ = encoder(batch)
            latent.append(z.cpu().numpy())
    Z = np.concatenate(latent, axis=0)

    # Save latent vectors
    latent_path = Path(cfg["paths"]["latent_vectors"])
    ensure_dir(latent_path)
    np.save(latent_path, Z.astype(np.float32))

    return Z


# --------------------------------------------------------------------------- #
# 5. Clustering wrapper
# --------------------------------------------------------------------------- #
class KMeansWrapper:
    def __init__(self, cfg: dict, seed: int):
        self.km = KMeans(
            n_clusters=cfg["clustering"]["n_clusters"],
            init=cfg["clustering"]["init"],
            n_init=cfg["clustering"]["n_init"],
            max_iter=cfg["clustering"]["max_iter"],
            random_state=seed,
        )

    def fit(self, Z: np.ndarray) -> None:
        self.km.fit(Z)

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return self.km.predict(Z)

    def save(self, path: str) -> None:
        ensure_dir(Path(path))
        joblib.dump(self.km, path)


# --------------------------------------------------------------------------- #
# 6. Evaluation & reporting
# --------------------------------------------------------------------------- #
def evaluate_clustering(y_true: np.ndarray, y_pred: np.ndarray, Z: np.ndarray, cfg: dict) -> dict:
    metrics = {}
    if "adjusted_rand_score" in cfg["evaluation"]["metrics"]:
        metrics["ARI"] = adjusted_rand_score(y_true, y_pred)
    if "normalized_mutual_info_score" in cfg["evaluation"]["metrics"]:
        metrics["NMI"] = normalized_mutual_info_score(y_true, y_pred)
    if "silhouette_score" in cfg["evaluation"]["metrics"]:
        metrics["Silhouette"] = silhouette_score(Z, y_pred)

    # Persist metrics
    metrics_path = Path(cfg["paths"]["metrics"])
    ensure_dir(metrics_path)
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def write_report(checksum: str, metrics: dict, cfg: dict) -> None:
    report_path = Path(cfg["paths"]["report"])
    ensure_dir(report_path)

    def fsize(p):
        try:
            return round(Path(p).stat().st_size / 1024, 2)
        except Exception:
            return "N/A"

    report = f"""# Wine Clustering – Final Report

**Dataset checksum (SHA‑256)**: `{checksum}`  

## Pre‑processing
- Median imputation (`SimpleImputer(strategy='median')`)
- Robust scaling (`RobustScaler`)

## Auto‑Encoder
- Input dimension: 13
- Hidden layers: {cfg["autoencoder"]["hidden_dims"]}
- Latent dimension: {cfg["autoencoder"]["latent_dim"]}
- Activation: {cfg["autoencoder"]["activation"]}
- Optimizer: {cfg["autoencoder"]["optimizer"]} (lr={cfg["autoencoder"]["learning_rate"]})
- Loss: {cfg["autoencoder"]["loss"]}
- Early‑stop patience: {cfg["autoencoder"]["early_stop_patience"]} epochs
- Encoder checkpoint size: {fsize(cfg["autoencoder"]["checkpoint_path"])} KB

## Clustering (K‑Means)
- n_clusters = {cfg["clustering"]["n_clusters"]}
- init = `{cfg["clustering"]["init"]}`
- n_init = {cfg["clustering"]["n_init"]}
- max_iter = {cfg["clustering"]["max_iter"]}
- Model file size: {fsize(cfg["clustering"]["model_path"])} KB

## Evaluation Metrics
| Metric | Value |
|--------|-------|
| Adjusted Rand Index (ARI) | {metrics.get('ARI', 'N/A'):.4f} |
| Normalized Mutual Information (NMI) | {metrics.get('NMI', 'N/A'):.4f} |
| Silhouette Score | {metrics.get('Silhouette', 'N/A'):.4f} |

*All artefacts are stored under the `agent_workspace` directory.*"""

    with report_path.open("w", encoding="utf-8") as f:
        f.write(report)


# --------------------------------------------------------------------------- #
# 7. Orchestrator (main)
# --------------------------------------------------------------------------- #
def main() -> None:
    # ------------------------------------------------------------------- #
    # 0. Global seeds & device
    # ------------------------------------------------------------------- #
    set_global_seeds(CFG["seed"])
    if torch.cuda.is_available():
        CFG["autoencoder"]["device"] = "cuda"

    # ------------------------------------------------------------------- #
    # 1. Load raw data
    # ------------------------------------------------------------------- #
    df = load_raw_data(CFG)

    # ------------------------------------------------------------------- #
    # 2. Compute and store checksum
    # ------------------------------------------------------------------- #
    checksum = compute_sha256(Path(CFG["data"]["raw_path"]))
    checksum_path = Path(CFG["paths"]["checksum"])
    ensure_dir(checksum_path)
    with checksum_path.open("w") as f:
        json.dump({"sha256": checksum}, f, indent=2)

    # ------------------------------------------------------------------- #
    # 3. Separate features / target
    # ------------------------------------------------------------------- #
    target_col = CFG["data"]["target_column"]
    y = df[target_col].values
    X = df.drop(columns=[target_col]).values

    # ------------------------------------------------------------------- #
    # 4. Pre‑process
    # ------------------------------------------------------------------- #
    preproc = PreprocessPipeline(CFG)
    X_pre = preproc.fit_transform(X)
    preproc.save(CFG["paths"]["preprocess_pipeline"])

    # ------------------------------------------------------------------- #
    # 5. Train auto‑encoder
    # ------------------------------------------------------------------- #
    print("\n=== Training Auto-Encoder ===")
    train_autoencoder(CFG, X_pre)   # checkpoint saved inside the function

    # ------------------------------------------------------------------- #
    # 6. Encode full dataset
    # ------------------------------------------------------------------- #
    print("\n=== Encoding dataset ===")
    Z = encode_dataset(CFG, X_pre)

    # ------------------------------------------------------------------- #
    # 7. K‑Means clustering
    # ------------------------------------------------------------------- #
    print("\n=== K-Means clustering ===")
    km = KMeansWrapper(CFG, seed=CFG["seed"])
    km.fit(Z)
    y_pred = km.predict(Z)
    km.save(CFG["clustering"]["model_path"])

    # ------------------------------------------------------------------- #
    # 8. Evaluation
    # ------------------------------------------------------------------- #
    print("\n=== Evaluation ===")
    metrics = evaluate_clustering(y_true=y, y_pred=y_pred, Z=Z, cfg=CFG)

    # ------------------------------------------------------------------- #
    # 9. Report
    # ------------------------------------------------------------------- #
    write_report(checksum, metrics, CFG)

    print("\nPipeline finished successfully.")
    print(f"Report written to: {CFG['paths']['report']}")
    print(f"Encoder checkpoint: {CFG['autoencoder']['checkpoint_path']}")
    print(f"K-Means model: {CFG['clustering']['model_path']}")
    print(f"Metrics JSON: {CFG['paths']['metrics']}")
    print(f"Latent vectors: {CFG['paths']['latent_vectors']}")


if __name__ == "__main__":
    # Suppress non‑critical warnings (e.g., future warnings from sklearn)
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
