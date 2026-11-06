
"""
Iris clustering pipeline – K‑Means (k=3) implemented in pure PyTorch.

The script follows the exact play‑book described in the specification:
1. Load the Iris CSV (user‑provided path).
2. Validate the schema.
3. Persist raw feature matrix and true labels.
4. Stratified split (70 % train / 20 % val / 10 % test) – reproducible with seed 42.
5. Fit a sklearn StandardScaler on the training split and transform the whole set.
6. Train a deterministic K‑Means model in PyTorch (max_iter=500, tol=1e‑4).
7. Save the scaler and the learned centroids.
8. Evaluate on validation and test splits (Adjusted Rand Index & Silhouette Score).
9. Persist the evaluation report as JSON.
10. All artefacts are stored under ``./agent_workspace/trained_models`` so that the
   downstream deployment service can simply load them.

The script is self‑contained – run it from the repository root (the folder that
contains the ``data`` directory). No external configuration files are required;
all paths and hyper‑parameters are defined as constants inside the script.
"""

import os
import json
import random
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import yaml

import torch
from torch import Tensor

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score

# --------------------------------------------------------------------------- #
# GLOBAL SETTINGS & CONSTANTS
# --------------------------------------------------------------------------- #
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# --------------------------------------------------------------------------- #
# Paths (all relative to the repository root)
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent  # folder that contains this script
DATA_SRC = Path(
    r"C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\iris.csv"
)  # original location supplied by the user
DATA_DST = PROJECT_ROOT / "data" / "Iris.csv"

ARTIFACT_ROOT = PROJECT_ROOT / "agent_workspace" / "trained_models"
RAW_ARTIFACT_DIR = ARTIFACT_ROOT / "raw"
PREP_ARTIFACT_DIR = ARTIFACT_ROOT / "prep"
MODEL_ARTIFACT_DIR = ARTIFACT_ROOT / "model"
REPORT_ARTIFACT_DIR = ARTIFACT_ROOT / "reports"

# Ensure directory hierarchy exists
for d in (RAW_ARTIFACT_DIR, PREP_ARTIFACT_DIR, MODEL_ARTIFACT_DIR, REPORT_ARTIFACT_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Filenames
X_RAW_PATH = RAW_ARTIFACT_DIR / "X_raw.parquet"
Y_TRUE_PATH = RAW_ARTIFACT_DIR / "y_true.parquet"

TRAIN_IDX_PATH = ARTIFACT_ROOT / "train_idx.npy"
VAL_IDX_PATH = ARTIFACT_ROOT / "val_idx.npy"
TEST_IDX_PATH = ARTIFACT_ROOT / "test_idx.npy"

SCALER_PATH = PREP_ARTIFACT_DIR / "scaler.joblib"
X_PREP_PATH = PREP_ARTIFACT_DIR / "X_prepped.parquet"

CENTROIDS_PATH = MODEL_ARTIFACT_DIR / "kmeans_centroids.pt"
KMEANS_CFG_PATH = MODEL_ARTIFACT_DIR / "kmeans_config.yaml"

EVAL_JSON_PATH = REPORT_ARTIFACT_DIR / "evaluation.json"

# K‑Means hyper‑parameters
N_CLUSTERS = 3
MAX_ITER = 500
TOL = 1e-4

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def sha256_of_file(path: Path) -> str:
    """Return SHA‑256 checksum of a file – useful for reproducibility checks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# 1️⃣  Copy the user‑provided CSV to the canonical location
# --------------------------------------------------------------------------- #
def copy_raw_csv():
    DATA_DST.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_SRC.is_file():
        raise FileNotFoundError(f"Source CSV not found at {DATA_SRC}")
    # Overwrite if already present (the pipeline is idempotent)
    DATA_DST.write_bytes(DATA_SRC.read_bytes())
    print(f"Copied raw CSV to {DATA_DST}")


# --------------------------------------------------------------------------- #
# 2️⃣  Load CSV, validate schema, and persist raw artefacts
# --------------------------------------------------------------------------- #
def load_and_save_raw():
    df = pd.read_csv(DATA_DST)

    # Expected columns for the classic Iris dataset
    expected_cols = {"sepal_length", "sepal_width", "petal_length", "petal_width", "target"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # No NaNs allowed
    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values (NaNs).")

    # Separate features and true labels
    X_raw = df.drop(columns=["target"]).astype(np.float32)
    y_true = df["target"].astype(np.int64)

    # Persist as Parquet (fast I/O, columnar)
    X_raw.to_parquet(X_RAW_PATH, index=False)
    y_true.to_frame(name="target").to_parquet(Y_TRUE_PATH, index=False)

    print(f"Saved raw feature matrix to {X_RAW_PATH}")
    print(f"Saved true labels to {Y_TRUE_PATH}")


# --------------------------------------------------------------------------- #
# 3️⃣  Stratified split (70/20/10)
# --------------------------------------------------------------------------- #
def stratified_split():
    y = pd.read_parquet(Y_TRUE_PATH)["target"].values
    # First split: train (70%) vs temp (30%)
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=0.30, random_state=SEED
    )
    train_idx, temp_idx = next(sss1.split(np.zeros_like(y), y))

    # Second split: val (20%) vs test (10%) → from the 30% temp we take 1/3 for test
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=1 / 3, random_state=SEED
    )
    val_idx, test_idx = next(sss2.split(np.zeros_like(temp_idx), y[temp_idx]))

    # Convert to original indices
    val_idx = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]

    # Save indices
    np.save(TRAIN_IDX_PATH, train_idx)
    np.save(VAL_IDX_PATH, val_idx)
    np.save(TEST_IDX_PATH, test_idx)

    print(f"Saved split indices: train ({len(train_idx)}), val ({len(val_idx)}), test ({len(test_idx)})")


# --------------------------------------------------------------------------- #
# 4️⃣  Standard‑scale (fit on train only)
# --------------------------------------------------------------------------- #
def scale_features():
    X = pd.read_parquet(X_RAW_PATH).values  # shape (150, 4)
    train_idx = np.load(TRAIN_IDX_PATH)

    scaler = StandardScaler()
    scaler.fit(X[train_idx])

    # Persist scaler
    joblib.dump(scaler, SCALER_PATH)

    # Transform whole dataset
    X_scaled = scaler.transform(X)

    # Save scaled matrix
    pd.DataFrame(X_scaled, columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]).to_parquet(
        X_PREP_PATH, index=False
    )

    print(f"Fitted StandardScaler on train split and saved to {SCALER_PATH}")
    print(f"Saved scaled feature matrix to {X_PREP_PATH}")


# --------------------------------------------------------------------------- #
# 5️⃣  Pure‑PyTorch K‑Means
# --------------------------------------------------------------------------- #
def train_kmeans():
    # Load scaled data and split indices
    X_scaled = pd.read_parquet(X_PREP_PATH).values.astype(np.float32)
    train_idx = np.load(TRAIN_IDX_PATH)

    X_train = torch.from_numpy(X_scaled[train_idx]).to(device)  # shape (N_train, 4)

    # Initialise centroids by sampling without replacement from the training data
    torch.manual_seed(SEED)
    perm = torch.randperm(X_train.shape[0])
    centroids = X_train[perm[:N_CLUSTERS]].clone()  # (3, 4)

    for iteration in range(1, MAX_ITER + 1):
        # Compute squared Euclidean distance between each point and each centroid
        # X_train: (N, D), centroids: (K, D) → distances: (N, K)
        dists = torch.cdist(X_train, centroids, p=2) ** 2  # shape (N, K)

        # Assign each point to the nearest centroid
        assignments = torch.argmin(dists, dim=1)  # (N,)

        # Compute new centroids
        new_centroids = []
        for k in range(N_CLUSTERS):
            mask = assignments == k
            if mask.any():
                new_centroids.append(X_train[mask].mean(dim=0))
            else:
                # Empty cluster → re‑initialise from a random training point
                rand_idx = torch.randint(0, X_train.shape[0], (1,))
                new_centroids.append(X_train[rand_idx])
        new_centroids = torch.stack(new_centroids)

        # Check convergence (max shift < tolerance)
        shift = torch.norm(centroids - new_centroids, dim=1).max()
        centroids = new_centroids
        if shift.item() < TOL:
            print(f"K‑Means converged after {iteration} iterations (shift={shift:.6f})")
            final_iter = iteration
            break
    else:
        final_iter = MAX_ITER
        print(f"K‑Means reached max_iter={MAX_ITER} without full convergence (last shift={shift:.6f})")

    # Persist centroids
    torch.save(centroids.cpu(), CENTROIDS_PATH)

    # Persist config (hyper‑params + final iteration)
    cfg = {
        "n_clusters": N_CLUSTERS,
        "max_iter": MAX_ITER,
        "tolerance": TOL,
        "seed": SEED,
        "final_iteration": int(final_iter),
    }
    with open(KMEANS_CFG_PATH, "w") as f:
        yaml.safe_dump(cfg, f)

    print(f"Saved centroids to {CENTROIDS_PATH}")
    print(f"Saved K‑Means config to {KMEANS_CFG_PATH}")

    return centroids.cpu()


# --------------------------------------------------------------------------- #
# 6️⃣  Inference helper (assign points to nearest centroid)
# --------------------------------------------------------------------------- #
def assign_clusters(X: np.ndarray, centroids: Tensor) -> np.ndarray:
    """
    X : (N, D) numpy array (already scaled)
    centroids : (K, D) torch tensor on CPU
    Returns: (N,) numpy array of cluster indices
    """
    X_tensor = torch.from_numpy(X.astype(np.float32))
    dists = torch.cdist(X_tensor, centroids, p=2) ** 2
    assignments = torch.argmin(dists, dim=1).numpy()
    return assignments


# --------------------------------------------------------------------------- #
# 7️⃣  Evaluation (ARI & Silhouette) on validation and test splits
# --------------------------------------------------------------------------- #
def evaluate():
    X_scaled = pd.read_parquet(X_PREP_PATH).values.astype(np.float32)
    y_true = pd.read_parquet(Y_TRUE_PATH)["target"].values.astype(np.int64)

    centroids = torch.load(CENTROIDS_PATH)  # (3, 4)

    val_idx = np.load(VAL_IDX_PATH)
    test_idx = np.load(TEST_IDX_PATH)

    val_pred = assign_clusters(X_scaled[val_idx], centroids)
    test_pred = assign_clusters(X_scaled[test_idx], centroids)

    ari_val = adjusted_rand_score(y_true[val_idx], val_pred)
    ari_test = adjusted_rand_score(y_true[test_idx], test_pred)

    sil_val = silhouette_score(X_scaled[val_idx], val_pred)
    sil_test = silhouette_score(X_scaled[test_idx], test_pred)

    evaluation = {
        "timestamp": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "model": "kmeans_pytorch",
        "n_clusters": N_CLUSTERS,
        "final_iteration": yaml.safe_load(open(KMEANS_CFG_PATH))["final_iteration"],
        "ari_val": float(ari_val),
        "ari_test": float(ari_test),
        "silhouette_val": float(sil_val),
        "silhouette_test": float(sil_test),
        # optional meta‑data
        "scaler_checksum": sha256_of_file(SCALER_PATH),
        "centroids_checksum": sha256_of_file(CENTROIDS_PATH),
    }

    # Persist JSON report
    with open(EVAL_JSON_PATH, "w") as f:
        json.dump(evaluation, f, indent=2)

    print(f"Evaluation saved to {EVAL_JSON_PATH}")
    print(json.dumps(evaluation, indent=2))


# --------------------------------------------------------------------------- #
# 8️⃣  Main orchestration
# --------------------------------------------------------------------------- #
def main():
    # 1. Copy raw CSV
    copy_raw_csv()

    # 2. Load CSV, validate, and persist raw artefacts
    load_and_save_raw()

    # 3. Stratified split
    stratified_split()

    # 4. Standard‑scale
    scale_features()

    # 5. Train K‑Means in PyTorch
    train_kmeans()

    # 6. Evaluate on validation & test
    evaluate()


if __name__ == "__main__":
    main()
