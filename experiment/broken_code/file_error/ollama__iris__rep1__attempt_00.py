
"""
Iris clustering pipeline
------------------------

* Loads the original CSV from the absolute path supplied by the user.
* Copies it to the project‑relative folder ``data/raw/iris.csv``.
* Validates the schema (5 columns, numeric measurements, target ∈ {0,1,2}).
* Preprocesses the data:
    - fills missing values with column‑wise median,
    - standard‑scales the four measurement columns,
    - stores the fitted ``StandardScaler``.
* Trains a tiny PyTorch auto‑encoder (2‑layer encoder → 2‑D latent space).
* Extracts the latent vectors and stores them.
* Fits a scikit‑learn ``KMeans`` (n_clusters=3) on the latent vectors.
* Computes external (ARI, NMI) and internal (Silhouette, CH, DB) metrics.
* Persists all artefacts (scaler, auto‑encoder state_dict, KMeans model,
  centroids, metrics) under ``./agent_workspace/trained_models``.
* (Optional) logs the run to MLflow if a tracking server is reachable.

The script is self‑contained – running it from the repository root will
reproduce the full end‑to‑end solution described in the specification.
"""

import os
import random
import shutil
import json
import logging
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)

# --------------------------------------------------------------------------- #
# 1. Global configuration & utilities
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Helper to load the yaml configuration (relative to the repo root)
# --------------------------------------------------------------------------- #
def load_config() -> dict:
    cfg_path = Path(__file__).parent / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Configuration file not found at {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


CONFIG = load_config()

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = CONFIG.get("seed", 42)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info(f"Using device: {DEVICE}")

# --------------------------------------------------------------------------- #
# Directory layout (all relative to repository root)
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).parent  # repository root (where this script lives)

DATA_RAW_DIR = ROOT / "data" / "raw"
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

ARTIFACTS_DIR = ROOT / "artifacts"
SCALER_DIR = ARTIFACTS_DIR / "scaler"
ENCODER_DIR = ARTIFACTS_DIR / "encoder"
CLUSTER_DIR = ARTIFACTS_DIR / "clustering"
REPORTS_DIR = ARTIFACTS_DIR / "reports"

for d in [SCALER_DIR, ENCODER_DIR, CLUSTER_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Final deployment folder required by the prompt
DEPLOY_DIR = Path("./agent_workspace/trained_models")
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# 2. Data ingestion & validation
# --------------------------------------------------------------------------- #
def ingest_and_validate() -> pd.DataFrame:
    src_path = Path(CONFIG["paths"]["raw_csv"])
    if not src_path.is_file():
        raise FileNotFoundError(f"Source CSV not found at {src_path}")

    dst_path = DATA_RAW_DIR / "iris.csv"
    shutil.copy2(src_path, dst_path)
    log.info(f"Copied raw CSV to {dst_path}")

    df = pd.read_csv(dst_path)

    # Expected schema
    expected_columns = set(
        CONFIG["preprocess"]["columns"] + [CONFIG["preprocess"]["target"]]
    )
    if set(df.columns) != expected_columns:
        raise ValueError(
            f"CSV columns mismatch. Expected {expected_columns}, got {set(df.columns)}"
        )

    # Numeric checks
    numeric_cols = CONFIG["preprocess"]["columns"]
    for col in numeric_cols:
        if not np.issubdtype(df[col].dtype, np.number):
            raise TypeError(f"Column '{col}' must be numeric.")

    # Target check
    target_vals = set(df[CONFIG["preprocess"]["target"]].unique())
    if not target_vals.issubset({0, 1, 2}):
        raise ValueError(
            f"Target column must contain only 0,1,2. Found values: {target_vals}"
        )

    # Persist a parquet copy for downstream steps
    parquet_path = ROOT / "data" / "processed" / "iris_clean.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    log.info(f"Saved cleaned parquet to {parquet_path}")

    return df


# --------------------------------------------------------------------------- #
# 3. Pre‑processing
# --------------------------------------------------------------------------- #
def preprocess(df: pd.DataFrame) -> np.ndarray:
    feature_cols = CONFIG["preprocess"]["columns"]
    target_col = CONFIG["preprocess"]["target"]

    # Fill missing values with column median (robust to future NaNs)
    medians = df[feature_cols].median()
    df[feature_cols] = df[feature_cols].fillna(medians)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(df[feature_cols])

    # Save scaler
    scaler_path = SCALER_DIR / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    log.info(f"Saved StandardScaler to {scaler_path}")

    # Also copy to deployment folder
    shutil.copy2(scaler_path, DEPLOY_DIR / "scaler.pkl")

    # Store the standardized matrix for the auto‑encoder
    X_path = ENCODER_DIR / "X_std.npy"
    np.save(X_path, X_std)
    log.info(f"Saved standardized features to {X_path}")

    return X_std, df[target_col].values


# --------------------------------------------------------------------------- #
# 4. Auto‑encoder definition & training
# --------------------------------------------------------------------------- #
class TinyAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Decoder (mirror)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon


def train_autoencoder(X_std: np.ndarray) -> TinyAutoEncoder:
    cfg = CONFIG["autoencoder"]
    model = TinyAutoEncoder(
        input_dim=4, hidden_dim=cfg["hidden_dim"], latent_dim=cfg["latent_dim"]
    ).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    dataset = TensorDataset(torch.from_numpy(X_std).float())
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model.train()
    for epoch in range(1, cfg["epochs"] + 1):
        epoch_losses = []
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        avg_loss = np.mean(epoch_losses)
        if epoch % 20 == 0 or epoch == cfg["epochs"]:
            log.info(f"Epoch {epoch:03d} – AE loss: {avg_loss:.6f}")

    # Save full state_dict (encoder + decoder) – needed for reproducibility
    ae_path = ENCODER_DIR / "ae_state_dict.pt"
    torch.save(model.state_dict(), ae_path)
    log.info(f"Saved auto‑encoder state_dict to {ae_path}")

    # Also copy to deployment folder
    shutil.copy2(ae_path, DEPLOY_DIR / "ae_state_dict.pt")

    return model


def extract_latent(model: TinyAutoEncoder, X_std: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        tensor = torch.from_numpy(X_std).float().to(DEVICE)
        latent = model.encoder(tensor).cpu().numpy()
    latent_path = ENCODER_DIR / "Z_latent.npy"
    np.save(latent_path, latent)
    log.info(f"Saved latent vectors to {latent_path}")

    # Also copy to deployment folder
    shutil.copy2(latent_path, DEPLOY_DIR / "Z_latent.npy")
    return latent


# --------------------------------------------------------------------------- #
# 5. K‑Means clustering
# --------------------------------------------------------------------------- #
def train_kmeans(Z_latent: np.ndarray):
    cfg = CONFIG["clustering"]
    kmeans = KMeans(
        n_clusters=cfg["n_clusters"],
        init=cfg["init"],
        n_init=10,
        max_iter=300,
        random_state=SEED,
    )
    kmeans.fit(Z_latent)
    log.info("K‑Means clustering completed.")

    # Persist model
    kmeans_path = CLUSTER_DIR / "kmeans.pkl"
    joblib.dump(kmeans, kmeans_path)
    log.info(f"Saved K‑Means model to {kmeans_path}")

    # Persist centroids as CSV for human inspection
    centroids_path = CLUSTER_DIR / "centroids.csv"
    pd.DataFrame(kmeans.cluster_centers_, columns=["dim1", "dim2"]).to_csv(
        centroids_path, index=False
    )
    log.info(f"Saved centroids to {centroids_path}")

    # Copy to deployment folder
    shutil.copy2(kmeans_path, DEPLOY_DIR / "kmeans.pkl")
    shutil.copy2(centroids_path, DEPLOY_DIR / "centroids.csv")

    return kmeans


# --------------------------------------------------------------------------- #
# 6. Evaluation
# --------------------------------------------------------------------------- #
def evaluate(
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    Z_latent: np.ndarray,
) -> dict:
    metrics = {
        "Adjusted Rand Index": round(adjusted_rand_score(true_labels, pred_labels), 3),
        "Normalized Mutual Info": round(
            normalized_mutual_info_score(true_labels, pred_labels), 3
        ),
        "Silhouette Score": round(silhouette_score(Z_latent, pred_labels), 3),
        "Calinski-Harabasz": round(
            calinski_harabasz_score(Z_latent, pred_labels), 3
        ),
        "Davies-Bouldin": round(davies_bouldin_score(Z_latent, pred_labels), 3),
    }

    metrics_path = REPORTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log.info(f"Saved evaluation metrics to {metrics_path}")

    # Also copy to deployment folder
    shutil.copy2(metrics_path, DEPLOY_DIR / "metrics.json")
    return metrics


# --------------------------------------------------------------------------- #
# 7. (Optional) MLflow logging – will be a no‑op if the server is unreachable
# --------------------------------------------------------------------------- #
def maybe_log_to_mlflow(metrics: dict):
    try:
        import mlflow

        mlflow.set_tracking_uri(CONFIG["mlflow"]["tracking_uri"])
        mlflow.set_experiment(CONFIG["mlflow"]["experiment_name"])

        with mlflow.start_run(run_name=f"iris_clustering_{datetime.now():%Y%m%d_%H%M%S}"):
            # Log parameters
            mlflow.log_params(
                {
                    "seed": SEED,
                    "autoencoder.latent_dim": CONFIG["autoencoder"]["latent_dim"],
                    "autoencoder.hidden_dim": CONFIG["autoencoder"]["hidden_dim"],
                    "autoencoder.epochs": CONFIG["autoencoder"]["epochs"],
                    "autoencoder.lr": CONFIG["autoencoder"]["lr"],
                    "clustering.n_clusters": CONFIG["clustering"]["n_clusters"],
                }
            )
            # Log metrics
            for k, v in metrics.items():
                mlflow.log_metric(k, v)

            # Log artefacts
            mlflow.log_artifact(str(SCALER_DIR / "scaler.pkl"))
            mlflow.log_artifact(str(ENCODER_DIR / "ae_state_dict.pt"))
            mlflow.log_artifact(str(CLUSTER_DIR / "kmeans.pkl"))
            mlflow.log_artifact(str(REPORTS_DIR / "metrics.json"))
    except Exception as e:
        log.warning(f"MLflow logging skipped – {e}")


# --------------------------------------------------------------------------- #
# 8. Orchestrator (main)
# --------------------------------------------------------------------------- #
def main():
    # 1️⃣ Ingest & validate
    df = ingest_and_validate()

    # 2️⃣ Preprocess
    X_std, y_true = preprocess(df)

    # 3️⃣ Auto‑encoder training
    ae_model = train_autoencoder(X_std)

    # 4️⃣ Latent extraction
    Z_latent = extract_latent(ae_model, X_std)

    # 5️⃣ K‑Means
    kmeans = train_kmeans(Z_latent)
    y_pred = kmeans.labels_

    # 6️⃣ Evaluation
    metrics = evaluate(y_true, y_pred, Z_latent)

    # 7️⃣ (optional) MLflow logging
    maybe_log_to_mlflow(metrics)

    log.info("Pipeline finished successfully.")
    log.info(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
