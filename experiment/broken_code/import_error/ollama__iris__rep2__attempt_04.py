
"""
Iris clustering pipeline – end‑to‑end implementation.

This script follows the specification:

1️⃣  Load the raw CSV (expected at ./agent_workspace/data/Iris.csv).  
    If the file is missing, the classic Iris dataset is loaded from
    ``sklearn.datasets`` and saved to the expected location automatically.
2️⃣  Validate schema (optional – skips if ``jsonschema`` is not installed).
3️⃣  Persist raw feature matrix and true labels as Parquet.
4️⃣  Stratified split → save train/val/test index arrays.
5️⃣  Standard‑scale (fit on train only) → persist scaler & scaled matrix.
6️⃣  K‑Means clustering implemented **pure PyTorch**.
7️⃣  Assign clusters to validation & test sets.
8️⃣  Compute Adjusted Rand Index (ARI) & Silhouette Score.
9️⃣  Persist centroids, scaler, and evaluation report.
🔟  (Optional) Log everything to a local MLflow run.
"""

# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
import os
import json
import datetime
import pathlib
import numpy as np
import pandas as pd
import joblib
import torch
import yaml  # pyyaml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.datasets import load_iris

# Optional jsonschema – if unavailable we simply skip schema validation
try:
    from jsonschema import Draft7Validator, ValidationError
    JSONSCHEMA_AVAILABLE = True
except Exception:  # pragma: no cover
    JSONSCHEMA_AVAILABLE = False

# Optional MLflow – if unavailable we skip logging
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except Exception:  # pragma: no cover
    MLFLOW_AVAILABLE = False

# --------------------------------------------------------------------------- #
# Configuration (hard‑coded for a self‑contained script)
# --------------------------------------------------------------------------- #
BASE_DIR = os.path.abspath("./agent_workspace")

# --------------------------- Paths ---------------------------------------- #
RAW_CSV_PATH = os.path.join(BASE_DIR, "data", "Iris.csv")

ARTIFACTS_RAW_DIR = os.path.join(BASE_DIR, "artifacts", "raw")
ARTIFACTS_PREP_DIR = os.path.join(BASE_DIR, "artifacts", "prep")
ARTIFACTS_MODEL_DIR = os.path.join(BASE_DIR, "artifacts", "model")
ARTIFACTS_REPORT_DIR = os.path.join(BASE_DIR, "artifacts", "reports")
SPLITS_DIR = os.path.join(BASE_DIR, "splits")
TRAINED_MODELS_DIR = os.path.join(BASE_DIR, "trained_models")   # required location

# --------------------------- Files ---------------------------------------- #
X_RAW_PARQUET = os.path.join(ARTIFACTS_RAW_DIR, "X_raw.parquet")
Y_TRUE_PARQUET = os.path.join(ARTIFACTS_RAW_DIR, "y_true.parquet")
TRAIN_IDX_NPY = os.path.join(SPLITS_DIR, "train_idx.npy")
VAL_IDX_NPY = os.path.join(SPLITS_DIR, "val_idx.npy")
TEST_IDX_NPY = os.path.join(SPLITS_DIR, "test_idx.npy")
SCALER_PATH = os.path.join(ARTIFACTS_PREP_DIR, "scaler.joblib")
X_SCALED_PARQUET = os.path.join(ARTIFACTS_PREP_DIR, "X_prepped.parquet")
CENTROID_PATH = os.path.join(TRAINED_MODELS_DIR, "kmeans_centroids.pt")
KMEANS_CONFIG_PATH = os.path.join(TRAINED_MODELS_DIR, "kmeans_config.yaml")
EVAL_REPORT_PATH = os.path.join(ARTIFACTS_REPORT_DIR, "evaluation.json")

# --------------------------- Hyper‑parameters ----------------------------- #
SEED = 42
N_CLUSTERS = 3
MAX_ITER = 500
TOLERANCE = 1e-4

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def ensure_dir(path: str) -> None:
    """Create a directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def log_mlflow_artifact(path: str, artifact_path: str = None):
    """Wrap mlflow.log_artifact so the script does not crash when mlflow is missing."""
    if MLFLOW_AVAILABLE:
        mlflow.log_artifact(path, artifact_path=artifact_path)


# --------------------------------------------------------------------------- #
# 1. Load raw data & (optional) schema validation
# --------------------------------------------------------------------------- #
ensure_dir(ARTIFACTS_RAW_DIR)

if not os.path.isfile(RAW_CSV_PATH):
    # CSV missing → fall back to sklearn's built‑in Iris dataset
    iris = load_iris()
    df = pd.DataFrame(
        data=np.c_[iris["data"], iris["target"]],
        columns=iris["feature_names"] + ["target"],
    )
    # Rename columns to match the schema used elsewhere
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]
    # Ensure the target is integer 0/1/2
    df["target"] = df["target"].astype(int)
    # Save for future runs
    os.makedirs(os.path.dirname(RAW_CSV_PATH), exist_ok=True)
    df.to_csv(RAW_CSV_PATH, index=False)
    print(f"Raw CSV not found – generated Iris data and saved to {RAW_CSV_PATH}")
else:
    df = pd.read_csv(RAW_CSV_PATH)

# ------------------- Schema validation (optional) ------------------------ #
if JSONSCHEMA_AVAILABLE:
    schema = {
        "type": "object",
        "properties": {
            "sepal_length": {"type": "number"},
            "sepal_width": {"type": "number"},
            "petal_length": {"type": "number"},
            "petal_width": {"type": "number"},
            "target": {"type": "integer", "enum": [0, 1, 2]},
        },
        "required": [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
            "target",
        ],
        "additionalProperties": False,
    }
    validator = Draft7Validator(schema)
    for idx, row in df.iterrows():
        try:
            validator.validate(row.to_dict())
        except ValidationError as exc:  # pragma: no cover
            raise ValueError(f"Schema validation error on row {idx}: {exc.message}")

# ------------------- Missing‑value guard --------------------------------- #
if df.isnull().any().any():
    raise ValueError("Missing values detected in the dataset.")

# --------------------------------------------------------------------------- #
# 2. Persist raw artefacts (features & true labels) as Parquet
# --------------------------------------------------------------------------- #
X_raw = df.drop(columns=["target"]).astype(np.float32)
y_true = df["target"].astype(np.int64)

X_raw.to_parquet(X_RAW_PARQUET, index=False)
y_true.to_frame(name="target").to_parquet(Y_TRUE_PARQUET, index=False)

# --------------------------------------------------------------------------- #
# 3. Stratified split (70 % train, 20 % val, 10 % test)
# --------------------------------------------------------------------------- #
ensure_dir(SPLITS_DIR)

sss_outer = StratifiedShuffleSplit(
    n_splits=1, test_size=0.30, random_state=SEED
)
train_idx, temp_idx = next(sss_outer.split(X_raw, y_true))

# Split the temporary set into validation (2/3 of temp) and test (1/3 of temp)
sss_inner = StratifiedShuffleSplit(
    n_splits=1, test_size=1 / 3, random_state=SEED
)
val_idx_rel, test_idx_rel = next(
    sss_inner.split(X_raw.iloc[temp_idx], y_true.iloc[temp_idx])
)
val_idx = temp_idx[val_idx_rel]
test_idx = temp_idx[test_idx_rel]

# Save index arrays
np.save(TRAIN_IDX_NPY, train_idx)
np.save(VAL_IDX_NPY, val_idx)
np.save(TEST_IDX_NPY, test_idx)

# --------------------------------------------------------------------------- #
# 4. Standard scaling (fit on train only) → persist scaler & scaled matrix
# --------------------------------------------------------------------------- #
ensure_dir(ARTIFACTS_PREP_DIR)

X_train_np = X_raw.iloc[train_idx].values
scaler = StandardScaler()
scaler.fit(X_train_np)

# Persist the scaler
joblib.dump(scaler, SCALER_PATH)

# Transform the whole dataset
X_scaled_np = scaler.transform(X_raw.values)
X_scaled_df = pd.DataFrame(
    X_scaled_np, columns=X_raw.columns, dtype=np.float32
)
X_scaled_df.to_parquet(X_SCALED_PARQUET, index=False)

# --------------------------------------------------------------------------- #
# 5. K‑Means clustering (pure PyTorch)
# --------------------------------------------------------------------------- #
ensure_dir(TRAINED_MODELS_DIR)

device = torch.device("cpu")
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Training data as torch tensor
X_train_tensor = torch.from_numpy(X_scaled_np[train_idx]).to(device)  # (N_train, 4)

# Initialise centroids by sampling without replacement from the training points
init_perm = torch.randperm(X_train_tensor.shape[0])[:N_CLUSTERS]
centroids = X_train_tensor[init_perm].clone()  # (k, 4)

final_iter = 0
for it in range(MAX_ITER):
    # ---- Assignment step -------------------------------------------------
    diff = X_train_tensor[:, None, :] - centroids[None, :, :]          # (N, k, D)
    dist = (diff ** 2).sum(dim=2)                                      # (N, k)
    labels = torch.argmin(dist, dim=1)                                 # (N,)

    # ---- Update step -----------------------------------------------------
    new_centroids = []
    for c in range(N_CLUSTERS):
        mask = labels == c
        if mask.any():
            new_c = X_train_tensor[mask].mean(dim=0)
        else:
            # Empty cluster – re‑initialise from a random training point
            rand_idx = torch.randint(0, X_train_tensor.shape[0], (1,))
            new_c = X_train_tensor[rand_idx]
        new_centroids.append(new_c)
    new_centroids = torch.stack(new_centroids)                         # (k, D)

    # ---- Convergence check ------------------------------------------------
    shift = torch.norm(new_centroids - centroids, dim=1).max()
    centroids = new_centroids
    final_iter = it + 1
    if shift.item() < TOLERANCE:
        print(f"K‑Means converged after {final_iter} iterations (shift={shift:.6f})")
        break
else:
    print(f"K‑Means reached max_iter={MAX_ITER} without full convergence.")

# Persist centroids
torch.save(centroids, CENTROID_PATH)

# Persist hyper‑parameters as YAML
kmeans_cfg = {
    "n_clusters": N_CLUSTERS,
    "max_iter": MAX_ITER,
    "tolerance": TOLERANCE,
    "seed": SEED,
    "final_iteration": final_iter,
}
with open(KMEANS_CONFIG_PATH, "w") as f:
    yaml.safe_dump(kmeans_cfg, f)

# --------------------------------------------------------------------------- #
# 6. Inference on validation & test splits
# --------------------------------------------------------------------------- #
def assign_clusters(data_np: np.ndarray, centroids_t: torch.Tensor) -> np.ndarray:
    """Assign each row in data_np to the nearest centroid."""
    data_t = torch.from_numpy(data_np).to(device)
    diff = data_t[:, None, :] - centroids_t[None, :, :]
    dist = (diff ** 2).sum(dim=2)
    return torch.argmin(dist, dim=1).cpu().numpy()


val_pred = assign_clusters(X_scaled_np[val_idx], centroids)
test_pred = assign_clusters(X_scaled_np[test_idx], centroids)

# --------------------------------------------------------------------------- #
# 7. Evaluation (Adjusted Rand Index & Silhouette Score)
# --------------------------------------------------------------------------- #
ari_val = adjusted_rand_score(y_true.iloc[val_idx], val_pred)
ari_test = adjusted_rand_score(y_true.iloc[test_idx], test_pred)

sil_val = silhouette_score(X_scaled_np[val_idx], val_pred)
sil_test = silhouette_score(X_scaled_np[test_idx], test_pred)

evaluation = {
    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    "model": "kmeans_pytorch",
    "n_clusters": N_CLUSTERS,
    "final_iteration": final_iter,
    "ari_val": float(ari_val),
    "ari_test": float(ari_test),
    "silhouette_val": float(sil_val),
    "silhouette_test": float(sil_test),
}
ensure_dir(ARTIFACTS_REPORT_DIR)
with open(EVAL_REPORT_PATH, "w") as f:
    json.dump(evaluation, f, indent=4)

# --------------------------------------------------------------------------- #
# 8. (Optional) MLflow logging – skipped if mlflow is not installed
# --------------------------------------------------------------------------- #
if MLFLOW_AVAILABLE:
    mlflow.set_tracking_uri("http://localhost:5000")          # adjust if needed
    mlflow.set_experiment("iris_clustering_simple")

    with mlflow.start_run(run_name="iris_kmeans_simple") as run:
        # Parameters
        mlflow.log_params(
            {
                "seed": SEED,
                "n_clusters": N_CLUSTERS,
                "max_iter": MAX_ITER,
                "tolerance": TOLERANCE,
            }
        )
        # Metrics
        mlflow.log_metric("ARI_val", ari_val)
        mlflow.log_metric("ARI_test", ari_test)
        mlflow.log_metric("Silhouette_val", sil_val)
        mlflow.log_metric("Silhouette_test", sil_test)

        # Artifacts
        log_mlflow_artifact(X_RAW_PARQUET, artifact_path="raw")
        log_mlflow_artifact(Y_TRUE_PARQUET, artifact_path="raw")
        log_mlflow_artifact(TRAIN_IDX_NPY, artifact_path="splits")
        log_mlflow_artifact(VAL_IDX_NPY, artifact_path="splits")
        log_mlflow_artifact(TEST_IDX_NPY, artifact_path="splits")
        log_mlflow_artifact(SCALER_PATH, artifact_path="preprocess")
        log_mlflow_artifact(X_SCALED_PARQUET, artifact_path="preprocess")
        log_mlflow_artifact(CENTROID_PATH, artifact_path="model")
        log_mlflow_artifact(KMEANS_CONFIG_PATH, artifact_path="model")
        log_mlflow_artifact(EVAL_REPORT_PATH, artifact_path="reports")

        mlflow.set_tag("pipeline", "iris_simple_kmeans")
        mlflow.set_tag("stage", "production")

# --------------------------------------------------------------------------- #
# 9. Summary output
# --------------------------------------------------------------------------- #
print("\n=== Pipeline finished ===")
print(f"Evaluation report saved to: {EVAL_REPORT_PATH}")
print(f"Centroids saved to: {CENTROID_PATH}")
print(f"Scaler saved to: {SCALER_PATH}")
if MLFLOW_AVAILABLE:
    print(f"MLflow run URL (if UI is running): {mlflow.get_artifact_uri()}")
else:
    print("MLflow not installed – logging was skipped.")
