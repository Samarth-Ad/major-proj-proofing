
"""
Iris clustering pipeline – end‑to‑end implementation.

The script follows the exact steps described in the specification:

1️⃣  Load configuration (constants.yaml)
2️⃣  Load the raw CSV (already copied to /project/data/Iris.csv)
3️⃣  Validate schema with jsonschema
4️⃣  Persist raw feature matrix and true labels as Parquet
5️⃣  Stratified split → save train/val/test index arrays
6️⃣  Standard‑scale (fit on train only) → persist scaler & scaled matrix
7️⃣  K‑Means clustering implemented **pure PyTorch**
8️⃣  Assign clusters to validation & test sets
9️⃣  Compute Adjusted Rand Index (ARI) & Silhouette Score
🔟  Persist centroids, k‑means config, evaluation report
1️⃣1️⃣ Log everything to a local MLflow run
"""

import os
import json
import yaml
import random
import datetime
import numpy as np
import pandas as pd
import joblib
import torch
import mlflow
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import adjusted_rand_score, silhouette_score
from jsonschema import Draft7Validator, ValidationError

# --------------------------------------------------------------------------- #
# 1. Load configuration
# --------------------------------------------------------------------------- #
CONFIG_PATH = "/project/config/constants.yaml"
with open(CONFIG_PATH, "r") as f:
    cfg = yaml.safe_load(f)

SEED = cfg["seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cpu")  # Iris is tiny; CPU is sufficient

# --------------------------------------------------------------------------- #
# 2. Helper utilities
# --------------------------------------------------------------------------- #
def ensure_dir(path: str):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def log_artifact_dir(src_dir: str, artifact_path: str, run):
    """Recursively log a directory as an MLflow artifact."""
    for root, _, files in os.walk(src_dir):
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, src_dir)
            run.log_artifact(full_path, artifact_path=os.path.join(artifact_path, os.path.dirname(rel_path)))


# --------------------------------------------------------------------------- #
# 3. Load raw data & schema validation
# --------------------------------------------------------------------------- #
RAW_CSV_PATH = cfg["data"]["raw_path"]
df = pd.read_csv(RAW_CSV_PATH)

# Expected schema (json‑schema)
schema = {
    "type": "object",
    "properties": {
        "sepal_length": {"type": "number"},
        "sepal_width": {"type": "number"},
        "petal_length": {"type": "number"},
        "petal_width": {"type": "number"},
        "target": {"type": "integer", "enum": [0, 1, 2]},
    },
    "required": ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"],
    "additionalProperties": False,
}
validator = Draft7Validator(schema)

# Validate each row
try:
    for _, row in df.iterrows():
        validator.validate(row.to_dict())
except ValidationError as e:
    raise ValueError(f"Schema validation error: {e.message}")

# Missing values check
if df.isnull().any().any():
    raise ValueError("Missing values detected in the dataset.")

# --------------------------------------------------------------------------- #
# 4. Persist raw artefacts (X_raw, y_true) as Parquet
# --------------------------------------------------------------------------- #
RAW_ARTIFACT_DIR = cfg["data"]["parquet_dir"]
ensure_dir(RAW_ARTIFACT_DIR)

X_raw = df.drop(columns=["target"]).astype(np.float32)
y_true = df["target"].astype(np.int64)

X_raw_path = os.path.join(RAW_ARTIFACT_DIR, "X_raw.parquet")
y_true_path = os.path.join(RAW_ARTIFACT_DIR, "y_true.parquet")
X_raw.to_parquet(X_raw_path, index=False)
y_true.to_frame(name="target").to_parquet(y_true_path, index=False)

# --------------------------------------------------------------------------- #
# 5. Stratified split (70/20/10) → save index arrays
# --------------------------------------------------------------------------- #
splits_dir = "/project/splits"
ensure_dir(splits_dir)

# First split: 70% train, 30% temp (val+test)
sss_outer = StratifiedShuffleSplit(
    n_splits=1, test_size=0.30, random_state=SEED
)
train_idx, temp_idx = next(sss_outer.split(X_raw, y_true))

# Second split: split temp into val (2/3 of temp = 20% overall) and test (1/3 = 10%)
sss_inner = StratifiedShuffleSplit(
    n_splits=1, test_size=1 / 3, random_state=SEED
)
val_idx, test_idx = next(
    sss_inner.split(X_raw.iloc[temp_idx], y_true.iloc[temp_idx])
)
val_idx = temp_idx[val_idx]
test_idx = temp_idx[test_idx]

# Save indices
np.save(cfg["splits"]["train_idx"], train_idx)
np.save(cfg["splits"]["val_idx"], val_idx)
np.save(cfg["splits"]["test_idx"], test_idx)

# --------------------------------------------------------------------------- #
# 6. Standard scaling (fit on train only) → persist scaler & scaled matrix
# --------------------------------------------------------------------------- #
prep_dir = "/project/artifacts/prep"
ensure_dir(prep_dir)

scaler_path = cfg["preprocess"]["scaler_path"]
scaled_parquet_path = cfg["preprocess"]["scaled_parquet"]

# Fit scaler on train data
X_train = X_raw.iloc[train_idx].values
scaler = StandardScaler()
scaler.fit(X_train)

# Persist scaler
joblib.dump(scaler, scaler_path)

# Transform all data
X_scaled = scaler.transform(X_raw.values)
X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=X_raw.columns,
    dtype=np.float32,
)
X_scaled_df.to_parquet(scaled_parquet_path, index=False)

# --------------------------------------------------------------------------- #
# 7. PyTorch K‑Means implementation
# --------------------------------------------------------------------------- #
model_dir = "/project/artifacts/model"
ensure_dir(model_dir)

centroid_path = cfg["model"]["kmeans"]["centroid_path"]
kmeans_config_path = cfg["model"]["kmeans"]["config_path"]

n_clusters = cfg["model"]["kmeans"]["n_clusters"]
max_iter = cfg["model"]["kmeans"]["max_iter"]
tolerance = cfg["model"]["kmeans"]["tolerance"]

# Convert train data to torch tensor
X_train_tensor = torch.from_numpy(X_scaled[train_idx]).to(DEVICE)  # shape (N_train, 4)

# Initialise centroids by sampling without replacement from training points
torch.manual_seed(SEED)
init_indices = torch.randperm(X_train_tensor.shape[0])[:n_clusters]
centroids = X_train_tensor[init_indices].clone()  # shape (k, 4)

for iteration in range(max_iter):
    # Compute squared Euclidean distances (broadcast)
    # X: (N, D) -> (N, 1, D)
    # centroids: (k, D) -> (1, k, D)
    diff = X_train_tensor[:, None, :] - centroids[None, :, :]  # (N, k, D)
    dist = (diff ** 2).sum(dim=2)  # (N, k)

    # Assignment step
    labels = torch.argmin(dist, dim=1)  # (N,)

    # Update centroids
    new_centroids = []
    for c in range(n_clusters):
        mask = labels == c
        if mask.any():
            new_c = X_train_tensor[mask].mean(dim=0)
        else:
            # Empty cluster → re‑initialize from a random point
            rand_idx = torch.randint(0, X_train_tensor.shape[0], (1,))
            new_c = X_train_tensor[rand_idx]
        new_centroids.append(new_c)
    new_centroids = torch.stack(new_centroids)

    # Convergence check
    centroid_shift = torch.norm(new_centroids - centroids, dim=1).max()
    centroids = new_centroids
    if centroid_shift.item() < tolerance:
        print(f"K‑Means converged after {iteration+1} iterations (shift={centroid_shift:.6f})")
        break
else:
    print(f"K‑Means reached max_iter={max_iter} without full convergence.")

# Persist centroids
torch.save(centroids, centroid_path)

# Persist k‑means config (hyper‑parameters + seed)
kmeans_cfg = {
    "n_clusters": n_clusters,
    "max_iter": max_iter,
    "tolerance": tolerance,
    "seed": SEED,
    "final_iteration": iteration + 1,
}
with open(kmeans_config_path, "w") as f:
    yaml.safe_dump(kmeans_cfg, f)

# --------------------------------------------------------------------------- #
# 8. Inference on validation & test splits
# --------------------------------------------------------------------------- #
def assign_clusters(data_np: np.ndarray, centroids_t: torch.Tensor) -> np.ndarray:
    """Assign each row in data_np to the nearest centroid."""
    data_t = torch.from_numpy(data_np).to(DEVICE)
    diff = data_t[:, None, :] - centroids_t[None, :, :]
    dist = (diff ** 2).sum(dim=2)
    labels = torch.argmin(dist, dim=1).cpu().numpy()
    return labels


# Load scaled matrix (already in memory as X_scaled)
val_pred = assign_clusters(X_scaled[val_idx], centroids)
test_pred = assign_clusters(X_scaled[test_idx], centroids)

# --------------------------------------------------------------------------- #
# 9. Evaluation (ARI & Silhouette)
# --------------------------------------------------------------------------- #
ari_val = adjusted_rand_score(y_true.iloc[val_idx], val_pred)
ari_test = adjusted_rand_score(y_true.iloc[test_idx], test_pred)

sil_val = silhouette_score(X_scaled[val_idx], val_pred)
sil_test = silhouette_score(X_scaled[test_idx], test_pred)

evaluation = {
    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    "model": "kmeans_pytorch",
    "n_clusters": n_clusters,
    "ari_val": float(ari_val),
    "ari_test": float(ari_test),
    "silhouette_val": float(sil_val),
    "silhouette_test": float(sil_test),
}
report_path = cfg["evaluation"]["report_path"]
ensure_dir(os.path.dirname(report_path))
with open(report_path, "w") as f:
    json.dump(evaluation, f, indent=4)

# --------------------------------------------------------------------------- #
# 10. MLflow logging
# --------------------------------------------------------------------------- #
mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

with mlflow.start_run(run_name="iris_kmeans_simple") as run:
    # Log parameters
    mlflow.log_params(
        {
            "seed": SEED,
            "n_clusters": n_clusters,
            "max_iter": max_iter,
            "tolerance": tolerance,
        }
    )
    # Log metrics
    mlflow.log_metric("ARI_val", ari_val)
    mlflow.log_metric("ARI_test", ari_test)
    mlflow.log_metric("Silhouette_val", sil_val)
    mlflow.log_metric("Silhouette_test", sil_test)

    # Log artefacts
    # Raw data folder
    mlflow.log_artifact(RAW_ARTIFACT_DIR, artifact_path="raw")
    # Split index files
    mlflow.log_artifact(cfg["splits"]["train_idx"], artifact_path="splits")
    mlflow.log_artifact(cfg["splits"]["val_idx"], artifact_path="splits")
    mlflow.log_artifact(cfg["splits"]["test_idx"], artifact_path="splits")
    # Scaler & scaled matrix
    mlflow.log_artifact(scaler_path, artifact_path="preprocess")
    mlflow.log_artifact(scaled_parquet_path, artifact_path="preprocess")
    # Model artefacts
    mlflow.log_artifact(centroid_path, artifact_path="model")
    mlflow.log_artifact(kmeans_config_path, artifact_path="model")
    # Evaluation report
    mlflow.log_artifact(report_path, artifact_path="reports")

    # Tag the run for easier tracking
    mlflow.set_tag("pipeline", "iris_simple_kmeans")
    mlflow.set_tag("stage", "production")

print("\n=== Pipeline finished ===")
print(f"Evaluation report saved to: {report_path}")
print(f"MLflow run URL (if UI is available): {mlflow.get_artifact_uri()}")

# --------------------------------------------------------------------------- #
# 11. Optional: Save predictions for debugging (not required by spec)
# --------------------------------------------------------------------------- #
debug_dir = "/project/artifacts/reports"
ensure_dir(debug_dir)
np.save(os.path.join(debug_dir, "val_pred.npy"), val_pred)
np.save(os.path.join(debug_dir, "test_pred.npy"), test_pred)
