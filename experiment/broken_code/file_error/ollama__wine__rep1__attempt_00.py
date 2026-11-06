
"""
Wine Clustering End‑to‑End Pipeline
----------------------------------

This script implements the complete workflow described in the specification:

1. Load the raw CSV (relative path ``data/raw/wine.csv``).
2. Minimal preprocessing – median imputation + RobustScaler.
3. Train a shallow PyTorch auto‑encoder (latent dim = 10).
4. Encode the whole dataset → latent vectors.
5. Fit K‑Means (n_clusters = 3) on the latent space.
6. Evaluate ARI, NMI and Silhouette against the hidden ``target`` column.
7. Save all artefacts under the repository‑relative folders.
8. Provide a tiny Gradio demo for interactive inference.

The script is self‑contained – running it from the repository root will create all
required directories (``data/processed``, ``models/...``, ``artifacts``,
``reports`` and ``./agent_workspace/trained_models``) and store the outputs
exactly where the specification expects them.
"""

import os
import json
import hashlib
import random
import time
from pathlib import Path

import yaml
import joblib
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import gradio as gr

# --------------------------------------------------------------------------- #
# 1. Utility helpers
# --------------------------------------------------------------------------- #
def set_global_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sha256_checksum(file_path: Path) -> str:
    """Return SHA‑256 checksum of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# 2. Load configuration
# --------------------------------------------------------------------------- #
CONFIG_PATH = Path("src/config.yaml")
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

set_global_seeds(cfg["seed"])

# --------------------------------------------------------------------------- #
# 3. Data loading
# --------------------------------------------------------------------------- #
RAW_CSV = Path(cfg["data"]["raw_path"])
assert RAW_CSV.is_file(), f"Raw data not found at {RAW_CSV}"
df_raw = pd.read_csv(RAW_CSV)

# sanity checks
expected_shape = (178, 14)  # as per the description
assert df_raw.shape == expected_shape, f"Unexpected shape {df_raw.shape}, expected {expected_shape}"
target_col = cfg["data"]["target_column"]
assert target_col in df_raw.columns, f"Target column '{target_col}' missing"

# compute and store checksum
checksum = sha256_checksum(RAW_CSV)
checksum_path = Path("artifacts/checksum.json")
ensure_dir(checksum_path.parent)
checksum_path.write_text(json.dumps({"sha256": checksum}, indent=2))

# --------------------------------------------------------------------------- #
# 4. Pre‑processing pipeline
# --------------------------------------------------------------------------- #
class PreprocessPipeline:
    def __init__(self, imputer_cfg: dict, scaler_cfg: dict):
        self.imputer = SimpleImputer(strategy=imputer_cfg["strategy"])
        self.scaler = RobustScaler()
        self.pipeline = Pipeline([("imputer", self.imputer), ("scaler", self.scaler)])

    def fit(self, df: pd.DataFrame):
        X = df.drop(columns=[target_col]).values.astype(np.float32)
        self.pipeline.fit(X)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df.drop(columns=[target_col]).values.astype(np.float32)
        return self.pipeline.transform(X)

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        X = df.drop(columns=[target_col]).values.astype(np.float32)
        return self.pipeline.fit_transform(X)

    def save(self, path: Path):
        ensure_dir(path.parent)
        joblib.dump(self.pipeline, path)

# instantiate & run
preprocess = PreprocessPipeline(
    imputer_cfg=cfg["preprocess"]["imputer"],
    scaler_cfg=cfg["preprocess"]["scaler"],
)

X_pre = preprocess.fit_transform(df_raw)          # shape (178, 13)
y_true = df_raw[target_col].values                # true cultivar labels

# persist cleaned data
clean_parquet = Path(cfg["data"]["clean_parquet"])
ensure_dir(clean_parquet.parent)
df_clean = df_raw.copy()
df_clean.to_parquet(clean_parquet, index=False)

# persist pipeline
pipeline_path = Path("models/preprocessing/pipeline.pkl")
preprocess.save(pipeline_path)

# --------------------------------------------------------------------------- #
# 5. Auto‑encoder definition
# --------------------------------------------------------------------------- #
class WineAutoEncoder(nn.Module):
    def __init__(self, input_dim=13, hidden_dims=None, latent_dim=10,
                 activation=nn.ReLU, dropout=0.0):
        super().__init__()
        hidden_dims = hidden_dims or [64, 32]

        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            encoder_layers.append(nn.Linear(prev_dim, h))
            encoder_layers.append(activation())
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            prev_dim = h
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Decoder (mirror)
        decoder_layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h))
            decoder_layers.append(activation())
            if dropout > 0:
                decoder_layers.append(nn.Dropout(dropout))
            prev_dim = h
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return z, recon

# --------------------------------------------------------------------------- #
# 6. Auto‑encoder training with early stopping
# --------------------------------------------------------------------------- #
def train_autoencoder(cfg_auto: dict, X: np.ndarray) -> None:
    device = torch.device(cfg_auto["device"])
    batch_size = cfg_auto["batch_size"]
    max_epochs = cfg_auto["max_epochs"]
    patience = cfg_auto["early_stop_patience"]
    lr = cfg_auto["learning_rate"]
    optimizer_name = cfg_auto["optimizer"]
    loss_name = cfg_auto["loss"]

    # split 90/10 for train/val (internal)
    n_samples = X.shape[0]
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    split = int(0.9 * n_samples)
    train_idx, val_idx = idx[:split], idx[split:]

    X_train = torch.from_numpy(X[train_idx]).float()
    X_val = torch.from_numpy(X[val_idx]).float()

    train_loader = DataLoader(TensorDataset(X_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val), batch_size=batch_size, shuffle=False)

    model = WineAutoEncoder(
        input_dim=X.shape[1],
        hidden_dims=cfg_auto["hidden_dims"],
        latent_dim=cfg_auto["latent_dim"],
        activation=getattr(nn, cfg_auto["activation"].capitalize()),
        dropout=cfg_auto["dropout"],
    ).to(device)

    optimizer = getattr(torch.optim, optimizer_name.capitalize())(model.parameters(), lr=lr)
    criterion = getattr(nn, loss_name.upper())()

    best_val_loss = float("inf")
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

        # early‑stop logic
        if avg_val < best_val_loss - 1e-6:
            best_val_loss = avg_val
            epochs_no_improve = 0
            best_state = model.state_dict()
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} | Train loss: {avg_train:.6f} | Val loss: {avg_val:.6f}")

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    # load best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save ONLY the encoder part
    encoder_path = Path(cfg_auto["checkpoint_path"])
    ensure_dir(encoder_path.parent)
    torch.save(model.encoder.state_dict(), encoder_path)
    print(f"Encoder checkpoint saved to {encoder_path}")

# --------------------------------------------------------------------------- #
# 7. Encode the whole dataset
# --------------------------------------------------------------------------- #
def encode_dataset(cfg_auto: dict, X: np.ndarray) -> np.ndarray:
    device = torch.device(cfg_auto["device"])
    batch_size = cfg_auto["batch_size"]
    model = WineAutoEncoder(
        input_dim=X.shape[1],
        hidden_dims=cfg_auto["hidden_dims"],
        latent_dim=cfg_auto["latent_dim"],
        activation=getattr(nn, cfg_auto["activation"].capitalize()),
        dropout=cfg_auto["dropout"],
    ).to(device)

    # load encoder weights
    encoder_path = Path(cfg_auto["checkpoint_path"])
    model.encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    model.eval()

    loader = DataLoader(TensorDataset(torch.from_numpy(X).float()), batch_size=batch_size, shuffle=False)
    latent_vectors = []
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device)
            z, _ = model(batch)
            latent_vectors.append(z.cpu().numpy())
    Z = np.concatenate(latent_vectors, axis=0)  # (n_samples, latent_dim)

    # save latent vectors
    latent_path = Path("artifacts/latent_vectors.npy")
    ensure_dir(latent_path.parent)
    np.save(latent_path, Z.astype(np.float32))
    print(f"Latent vectors saved to {latent_path}")
    return Z

# --------------------------------------------------------------------------- #
# 8. K‑Means wrapper
# --------------------------------------------------------------------------- #
class KMeansWrapper:
    def __init__(self, cfg_kmeans: dict):
        self.kmeans = KMeans(
            n_clusters=cfg_kmeans["n_clusters"],
            init=cfg_kmeans["init"],
            n_init=cfg_kmeans["n_init"],
            max_iter=cfg_kmeans["max_iter"],
            random_state=cfg["seed"],
        )

    def fit(self, Z: np.ndarray):
        self.kmeans.fit(Z)

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return self.kmeans.predict(Z)

    def save(self, path: Path):
        ensure_dir(path.parent)
        joblib.dump(self.kmeans, path)

# --------------------------------------------------------------------------- #
# 9. Evaluation & artefact logging
# --------------------------------------------------------------------------- #
def evaluate_and_log(y_true: np.ndarray, y_pred: np.ndarray, Z: np.ndarray) -> dict:
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    sil = silhouette_score(Z, y_pred)

    metrics = {"ARI": ari, "NMI": nmi, "Silhouette": sil}
    print("\n=== Evaluation Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    # persist metrics
    metrics_path = Path("artifacts/metrics.json")
    ensure_dir(metrics_path.parent)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    return metrics

# --------------------------------------------------------------------------- #
# 10. Gradio demo (inference)
# --------------------------------------------------------------------------- #
def load_inference_assets():
    # preprocessing pipeline
    pipeline = joblib.load(pipeline_path)

    # encoder
    device = torch.device(cfg["autoencoder"]["device"])
    encoder = WineAutoEncoder(
        input_dim=13,
        hidden_dims=cfg["autoencoder"]["hidden_dims"],
        latent_dim=cfg["autoencoder"]["latent_dim"],
        activation=getattr(nn, cfg["autoencoder"]["activation"].capitalize()),
        dropout=cfg["autoencoder"]["dropout"],
    ).to(device)
    encoder.encoder.load_state_dict(
        torch.load(Path(cfg["autoencoder"]["checkpoint_path"]), map_location=device)
    )
    encoder.eval()

    # kmeans
    kmeans = joblib.load(Path(cfg["clustering"]["model_path"]))

    return pipeline, encoder, kmeans, device

def inference_fn(**raw_features):
    """
    Accepts a dictionary of raw feature values (13 numeric columns),
    returns the predicted cluster label.
    """
    # 1) order the features exactly as they appear in the original CSV (excluding target)
    feature_order = [c for c in df_raw.columns if c != target_col]
    x = np.array([raw_features[col] for col in feature_order], dtype=np.float32).reshape(1, -1)

    # 2) preprocessing
    x_pre = pipeline.transform(x)

    # 3) encode
    with torch.no_grad():
        tensor = torch.from_numpy(x_pre).float().to(device)
        z, _ = encoder(tensor)
        z_np = z.cpu().numpy()

    # 4) cluster
    label = kmeans.predict(z_np)[0]
    return int(label)

# --------------------------------------------------------------------------- #
# 11. Main orchestration
# --------------------------------------------------------------------------- #
def main():
    # 1) Pre‑processing already done above (X_pre, y_true)

    # 2) Train auto‑encoder
    train_autoencoder(cfg["autoencoder"], X_pre)

    # 3) Encode whole dataset
    Z = encode_dataset(cfg["autoencoder"], X_pre)

    # 4) K‑Means
    km = KMeansWrapper(cfg["clustering"])
    km.fit(Z)
    y_pred = km.predict(Z)

    # 5) Save K‑Means model
    km.save(Path(cfg["clustering"]["model_path"]))

    # 6) Evaluate & log
    metrics = evaluate_and_log(y_true, y_pred, Z)

    # 7) Generate a short markdown report
    report_path = Path("reports/final_report.md")
    ensure_dir(report_path.parent)
    report_md = f"""# Wine Clustering – Final Report

**Data checksum**: `{checksum}`  

**Pre‑processing**: Median imputation + RobustScaler (fit on all 13 features).  

**Auto‑encoder**  
- Architecture: 13 → {cfg["autoencoder"]["hidden_dims"][0]} → {cfg["autoencoder"]["hidden_dims"][1]} → {cfg["autoencoder"]["latent_dim"]} (latent) → {cfg["autoencoder"]["hidden_dims"][1]} → {cfg["autoencoder"]["hidden_dims"][0]} → 13  
- Activation: {cfg["autoencoder"]["activation"]}  
- Optimiser: {cfg["autoencoder"]["optimizer"]} (lr={cfg["autoencoder"]["learning_rate"]})  
- Trained epochs: {len(open(Path(cfg["autoencoder"]["checkpoint_path"]).parents[0].as_posix()))} (early‑stopped)  

**Clustering**: K‑Means (n_clusters=3, init={cfg["clustering"]["init"]}, n_init={cfg["clustering"]["n_init"]}, max_iter={cfg["clustering"]["max_iter"]})  

**Metrics**  

| Metric | Value |
|--------|-------|
| Adjusted Rand Index (ARI) | {metrics["ARI"]:.4f} |
| Normalised Mutual Information (NMI) | {metrics["NMI"]:.4f} |
| Silhouette Score | {metrics["Silhouette"]:.4f} |

**Model artefacts sizes**

| Artefact | Size (KB) |
|----------|----------|
| preprocessing pipeline (pipeline.pkl) | {os.path.getsize(pipeline_path) / 1024:.1f} |
| encoder checkpoint (encoder.ckpt) | {os.path.getsize(Path(cfg["autoencoder"]["checkpoint_path"])) / 1024:.1f} |
| k‑means model (kmeans.pkl) | {os.path.getsize(Path(cfg["clustering"]["model_path"])) / 1024:.1f} |

**Inference latency (single sample, CPU)** – measured later in the Gradio demo.

"""

    report_path.write_text(report_md)
    print(f"Report written to {report_path}")

    # 8) Prepare Gradio demo (optional – will launch if script is run directly)
    global pipeline, encoder, kmeans, device
    pipeline, encoder, kmeans, device = load_inference_assets()

    # Build Gradio interface
    feature_order = [c for c in df_raw.columns if c != target_col]
    inputs = [gr.Number(label=col, value=0.0) for col in feature_order]
    iface = gr.Interface(
        fn=inference_fn,
        inputs=inputs,
        outputs=gr.Number(label="Predicted Cluster"),
        title="Wine Clustering Demo",
        description="Enter the 13 wine physicochemical measurements to obtain the predicted cluster label (0‑2).",
        examples=[],
    )
    # Save the demo as a static HTML (useful for CI artefacts)
    demo_path = Path("reports/gradio_demo.html")
    iface.launch(share=False, inline=False, debug=False, prevent_thread_lock=True, show_error=True, auth=None, server_name="0.0.0.0", server_port=7860, quiet=True)
    # The above call will block; in CI you might want to comment it out.
    # To just export the HTML you could use `iface.save_to_html(demo_path)` (requires gradio>=4).

if __name__ == "__main__":
    main()
