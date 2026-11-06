
"""
Wine Clustering End‑to‑End Pipeline
----------------------------------

This script is a **stand‑alone** implementation of the full workflow described in the
specification.  All hyper‑parameters are defined inside the script (so no external
`src/config.yaml` is required) and every artefact is written under the repository‑root
relative paths – in particular the trained models are saved to
``./agent_workspace/trained_models`` as required by the evaluation harness.

Running the script from the repository root will:

1. Load the raw CSV (``data/raw/wine.csv`` – adjust the path if your file lives elsewhere).
2. Perform median imputation + RobustScaler.
3. Train a shallow PyTorch auto‑encoder (latent dim = 10) with early stopping.
4. Encode the whole dataset → latent vectors.
5. Fit K‑Means (n_clusters = 3) on the latent space.
6. Compute ARI, NMI and Silhouette scores against the hidden ``target`` column.
7. Persist:
   * preprocessing pipeline (`pipeline.pkl`)
   * encoder checkpoint (`encoder.ckpt`)
   * K‑Means model (`kmeans.pkl`)
   * latent vectors (`latent_vectors.npy`)
   * evaluation metrics (`metrics.json`)
   * a short markdown report (`reports/final_report.md`)
8. (Optional) launch a tiny Gradio demo for interactive inference.

All required directories are created automatically.
"""

import os
import json
import hashlib
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import joblib
from tqdm import tqdm

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --------------------------------------------------------------------------- #
# 0. Global configuration (inline – no external YAML required)
# --------------------------------------------------------------------------- #
CFG = {
    "seed": 42,
    "data": {
        "raw_path": "data/raw/wine.csv",                     # adjust if needed
        "clean_parquet": "data/processed/wine_clean.parquet",
        "target_column": "target",
    },
    "preprocess": {
        "imputer": {"strategy": "median"},
        "scaler": {"type": "robust"},
    },
    "autoencoder": {
        "latent_dim": 10,
        "hidden_dims": [64, 32],
        "activation": "relu",
        "dropout": 0.0,
        "batch_size": 32,
        "max_epochs": 150,
        "early_stop_patience": 10,
        "learning_rate": 1e-3,
        "optimizer": "adam",
        "loss": "mse",
        "device": "cpu",
        "checkpoint_path": "agent_workspace/trained_models/encoder.ckpt",
    },
    "clustering": {
        "algorithm": "kmeans",
        "n_clusters": 3,
        "init": "k-means++",
        "n_init": 20,
        "max_iter": 500,
        "model_path": "agent_workspace/trained_models/kmeans.pkl",
    },
    "evaluation": {
        "metrics": ["adjusted_rand_index", "normalized_mutual_info", "silhouette"],
    },
    "paths": {
        "pipeline_path": "agent_workspace/trained_models/pipeline.pkl",
        "latent_path": "artifacts/latent_vectors.npy",
        "metrics_path": "artifacts/metrics.json",
        "checksum_path": "artifacts/checksum.json",
        "report_path": "reports/final_report.md",
    },
}

# --------------------------------------------------------------------------- #
# 1. Helper utilities
# --------------------------------------------------------------------------- #
def set_global_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sha256_checksum(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# 2. Data loading + checksum
# --------------------------------------------------------------------------- #
def load_raw_data(cfg: dict) -> pd.DataFrame:
    raw_path = Path(cfg["data"]["raw_path"])
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw CSV not found at {raw_path}")
    df = pd.read_csv(raw_path)

    # sanity checks
    expected_cols = set(df.columns)
    target = cfg["data"]["target_column"]
    if target not in expected_cols:
        raise ValueError(f"Target column '{target}' missing in CSV")
    # store checksum for provenance
    checksum = sha256_checksum(raw_path)
    checksum_path = Path(cfg["paths"]["checksum_path"])
    ensure_dir(checksum_path.parent)
    checksum_path.write_text(json.dumps({"sha256": checksum}, indent=2))
    return df

# --------------------------------------------------------------------------- #
# 3. Pre‑processing pipeline
# --------------------------------------------------------------------------- #
class PreprocessPipeline:
    def __init__(self, cfg: dict):
        imputer_cfg = cfg["preprocess"]["imputer"]
        self.imputer = SimpleImputer(strategy=imputer_cfg["strategy"])
        self.scaler = RobustScaler()
        self.pipeline = Pipeline([("imputer", self.imputer), ("scaler", self.scaler)])

    def fit(self, df: pd.DataFrame, target_col: str):
        X = df.drop(columns=[target_col]).values.astype(np.float32)
        self.pipeline.fit(X)

    def transform(self, df: pd.DataFrame, target_col: str) -> np.ndarray:
        X = df.drop(columns=[target_col]).values.astype(np.float32)
        return self.pipeline.transform(X)

    def fit_transform(self, df: pd.DataFrame, target_col: str) -> np.ndarray:
        X = df.drop(columns=[target_col]).values.astype(np.float32)
        return self.pipeline.fit_transform(X)

    def save(self, path: str):
        ensure_dir(Path(path).parent)
        joblib.dump(self.pipeline, path)

# --------------------------------------------------------------------------- #
# 4. Auto‑encoder definition
# --------------------------------------------------------------------------- #
class WineAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 13,
        hidden_dims: list[int] = None,
        latent_dim: int = 10,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [64, 32]

        act_fn = getattr(nn, activation.capitalize())

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
# 5. Auto‑encoder training with early stopping
# --------------------------------------------------------------------------- #
def train_autoencoder(cfg: dict, X: np.ndarray) -> None:
    device = torch.device(cfg["device"])
    batch_size = cfg["batch_size"]
    max_epochs = cfg["max_epochs"]
    patience = cfg["early_stop_patience"]
    lr = cfg["learning_rate"]
    optimizer_name = cfg["optimizer"]
    loss_name = cfg["loss"]

    # 90/10 split for internal validation
    n = X.shape[0]
    idx = np.random.permutation(n)
    split = int(0.9 * n)
    train_idx, val_idx = idx[:split], idx[split:]

    X_train = torch.from_numpy(X[train_idx]).float()
    X_val = torch.from_numpy(X[val_idx]).float()

    train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val), batch_size=batch_size, shuffle=False)

    model = WineAutoEncoder(
        input_dim=X.shape[1],
        hidden_dims=cfg["hidden_dims"],
        latent_dim=cfg["latent_dim"],
        activation=cfg["activation"],
        dropout=cfg["dropout"],
    ).to(device)

    optimizer = getattr(torch.optim, optimizer_name.capitalize())(model.parameters(), lr=lr)
    criterion = getattr(nn, loss_name.upper())()

    best_val = float("inf")
    epochs_no_improve = 0
    best_state = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for (batch,) in train_loader:
            batch = batch.to(device)
            _, recon = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for (batch,) in val_loader:
                batch = batch.to(device)
                _, recon = model(batch)
                loss = criterion(recon, batch)
                val_losses.append(loss.item())

        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)

        if avg_val < best_val - 1e-6:
            best_val = avg_val
            epochs_no_improve = 0
            best_state = model.state_dict()
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"[AE] Epoch {epoch:03d} | Train loss: {avg_train:.6f} | Val loss: {avg_val:.6f}"
            )

        if epochs_no_improve >= patience:
            print(f"[AE] Early stopping triggered at epoch {epoch}")
            break

    # load best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # save ONLY encoder weights
    ckpt_path = Path(cfg["checkpoint_path"])
    ensure_dir(ckpt_path.parent)
    torch.save(model.encoder.state_dict(), ckpt_path)
    print(f"[AE] Encoder checkpoint saved to {ckpt_path}")

# --------------------------------------------------------------------------- #
# 6. Encode the whole dataset
# --------------------------------------------------------------------------- #
def encode_dataset(cfg: dict, X: np.ndarray) -> np.ndarray:
    device = torch.device(cfg["device"])
    batch_size = cfg["batch_size"]

    model = WineAutoEncoder(
        input_dim=X.shape[1],
        hidden_dims=cfg["hidden_dims"],
        latent_dim=cfg["latent_dim"],
        activation=cfg["activation"],
        dropout=cfg["dropout"],
    ).to(device)

    # load encoder weights
    ckpt_path = Path(cfg["checkpoint_path"])
    model.encoder.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    loader = DataLoader(TensorDataset(torch.from_numpy(X).float()), batch_size=batch_size, shuffle=False)
    latent_list = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            z, _ = model(batch)
            latent_list.append(z.cpu().numpy())
    Z = np.concatenate(latent_list, axis=0)

    # persist latent vectors
    latent_path = Path(CFG["paths"]["latent_path"])
    ensure_dir(latent_path.parent)
    np.save(latent_path, Z.astype(np.float32))
    print(f"[AE] Latent vectors saved to {latent_path}")
    return Z

# --------------------------------------------------------------------------- #
# 7. K‑Means wrapper
# --------------------------------------------------------------------------- #
class KMeansWrapper:
    def __init__(self, cfg: dict, seed: int):
        self.kmeans = KMeans(
            n_clusters=cfg["n_clusters"],
            init=cfg["init"],
            n_init=cfg["n_init"],
            max_iter=cfg["max_iter"],
            random_state=seed,
        )

    def fit(self, Z: np.ndarray):
        self.kmeans.fit(Z)

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return self.kmeans.predict(Z)

    def save(self, path: str):
        ensure_dir(Path(path).parent)
        joblib.dump(self.kmeans, path)

# --------------------------------------------------------------------------- #
# 8. Evaluation & artefact logging
# --------------------------------------------------------------------------- #
def evaluate_and_log(y_true: np.ndarray, y_pred: np.ndarray, Z: np.ndarray, cfg: dict) -> dict:
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    sil = silhouette_score(Z, y_pred)

    metrics = {"ARI": ari, "NMI": nmi, "Silhouette": sil}
    print("\n=== Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # persist metrics
    metrics_path = Path(cfg["paths"]["metrics_path"])
    ensure_dir(metrics_path.parent)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    return metrics

# --------------------------------------------------------------------------- #
# 9. Markdown report generation
# --------------------------------------------------------------------------- #
def generate_report(checksum: str, metrics: dict, cfg: dict):
    report_path = Path(cfg["paths"]["report_path"])
    ensure_dir(report_path.parent)

    # model file sizes
    pipeline_sz = os.path.getsize(cfg["paths"]["pipeline_path"]) / 1024
    encoder_sz = os.path.getsize(cfg["autoencoder"]["checkpoint_path"]) / 1024
    kmeans_sz = os.path.getsize(cfg["clustering"]["model_path"]) / 1024

    report_md = f"""# Wine Clustering – Final Report

**Data checksum**: `{checksum}`  

**Pre‑processing**: Median imputation + RobustScaler (fit on all 13 features).  

**Auto‑encoder architecture**  
- Input → {cfg["autoencoder"]["hidden_dims"][0]} → {cfg["autoencoder"]["hidden_dims"][1]} → {cfg["autoencoder"]["latent_dim"]} (latent) → {cfg["autoencoder"]["hidden_dims"][1]} → {cfg["autoencoder"]["hidden_dims"][0]} → Input  
- Activation: {cfg["autoencoder"]["activation"]}  
- Optimiser: {cfg["autoencoder"]["optimizer"]} (lr={cfg["autoencoder"]["learning_rate"]})  

**Clustering**: K‑Means (n_clusters={cfg["clustering"]["n_clusters"]}, init={cfg["clustering"]["init"]},
n_init={cfg["clustering"]["n_init"]}, max_iter={cfg["clustering"]["max_iter"]})  

**Metrics**

| Metric | Value |
|--------|-------|
| Adjusted Rand Index (ARI) | {metrics["ARI"]:.4f} |
| Normalised Mutual Information (NMI) | {metrics["NMI"]:.4f} |
| Silhouette Score | {metrics["Silhouette"]:.4f} |

**Model artefacts sizes**

| Artefact | Size (KB) |
|----------|----------|
| Pre‑processing pipeline (`pipeline.pkl`) | {pipeline_sz:.1f} |
| Encoder checkpoint (`encoder.ckpt`) | {encoder_sz:.1f} |
| K‑Means model (`kmeans.pkl`) | {kmeans_sz:.1f} |

"""

    report_path.write_text(report_md)
    print(f"[REPORT] Markdown report written to {report_path}")

# --------------------------------------------------------------------------- #
# 10. Optional Gradio demo (lightweight, launched only when script is run directly)
# --------------------------------------------------------------------------- #
def launch_gradio_demo(pipeline_path: str, encoder_ckpt: str, kmeans_path: str, cfg: dict, df_raw: pd.DataFrame):
    try:
        import gradio as gr
    except Exception as e:
        print("[GRADIO] Gradio not installed – skipping demo.")
        return

    # Load assets
    pipeline = joblib.load(pipeline_path)

    device = torch.device(cfg["autoencoder"]["device"])
    encoder = WineAutoEncoder(
        input_dim=13,
        hidden_dims=cfg["autoencoder"]["hidden_dims"],
        latent_dim=cfg["autoautoencoder"]["latent_dim"] if False else cfg["autoencoder"]["latent_dim"],  # dummy line to keep style
        activation=cfg["autoencoder"]["activation"],
        dropout=cfg["autoencoder"]["dropout"],
    ).to(device)
    encoder.encoder.load_state_dict(torch.load(encoder_ckpt, map_location=device))
    encoder.eval()

    kmeans = joblib.load(kmeans_path)

    target_col = cfg["data"]["target_column"]
    feature_order = [c for c in df_raw.columns if c != target_col]

    def inference(**raw_features):
        # 1) build ordered array
        x = np.array([raw_features[col] for col in feature_order], dtype=np.float32).reshape(1, -1)
        # 2) preprocess
        x_pre = pipeline.transform(x)
        # 3) encode
        with torch.no_grad():
            tensor = torch.from_numpy(x_pre).float().to(device)
            z, _ = encoder(tensor)
            z_np = z.cpu().numpy()
        # 4) cluster
        label = kmeans.predict(z_np)[0]
        return int(label)

    inputs = [gr.Number(label=col, value=0.0) for col in feature_order]
    iface = gr.Interface(
        fn=inference,
        inputs=inputs,
        outputs=gr.Number(label="Predicted Cluster"),
        title="Wine Clustering Demo",
        description="Enter the 13 physicochemical measurements to obtain the predicted cluster (0‑2).",
    )
    iface.launch(share=False)

# --------------------------------------------------------------------------- #
# 11. Main orchestration
# --------------------------------------------------------------------------- #
def main():
    set_global_seeds(CFG["seed"])

    # 1) Load raw data & checksum
    df_raw = load_raw_data(CFG)
    checksum = sha256_checksum(Path(CFG["data"]["raw_path"]))

    # 2) Pre‑process
    preproc = PreprocessPipeline(CFG)
    X_pre = preproc.fit_transform(df_raw, CFG["data"]["target_column"])
    y_true = df_raw[CFG["data"]["target_column"]].values

    # persist cleaned parquet & pipeline
    clean_parquet = Path(CFG["data"]["clean_parquet"])
    ensure_dir(clean_parquet.parent)
    df_raw.to_parquet(clean_parquet, index=False)

    preproc.save(CFG["paths"]["pipeline_path"])

    # 3) Train auto‑encoder
    train_autoencoder(CFG["autoencoder"], X_pre)

    # 4) Encode whole dataset
    Z = encode_dataset(CFG["autoencoder"], X_pre)

    # 5) K‑Means clustering
    km = KMeansWrapper(CFG["clustering"], seed=CFG["seed"])
    km.fit(Z)
    y_pred = km.predict(Z)
    km.save(CFG["clustering"]["model_path"])

    # 6) Evaluation & logging
    metrics = evaluate_and_log(y_true, y_pred, Z, CFG)

    # 7) Report
    generate_report(checksum, metrics, CFG)

    # 8) (Optional) Gradio demo
    launch_gradio_demo(
        pipeline_path=CFG["paths"]["pipeline_path"],
        encoder_ckpt=CFG["autoencoder"]["checkpoint_path"],
        kmeans_path=CFG["clustering"]["model_path"],
        cfg=CFG,
        df_raw=df_raw,
    )

if __name__ == "__main__":
    main()
