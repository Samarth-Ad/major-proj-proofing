
"""
Iris clustering pipeline – K‑Means (k=3) implemented in pure PyTorch.

The script is fully self‑contained and follows the required layout:

project/
│
├─ data/
│   └─ Iris.csv                     ← copied from the user‑provided location
│
├─ splits/
│   ├─ train_idx.npy
│   ├─ val_idx.npy
│   └─ test_idx.npy
│
├─ artifacts/
│   ├─ raw/
│   │   ├─ X_raw.parquet
│   │   └─ y_true.parquet
│   ├─ prep/
│   │   ├─ scaler.joblib
│   │   └─ X_prepped.parquet
│   ├─ model/
│   │   ├─ kmeans_centroids.pt
│   │   └─ kmeans_config.yaml
│   └─ reports/
│       └─ evaluation.json
│
├─ config/
│   ├─ constants.yaml   ← not used directly here (all constants are defined below)
│   └─ top1_candidate.yaml
│
├─ requirements.txt
└─ pipeline.py          ← **this file**
"""

# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
from pathlib import Path
import json
import random
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import yaml
import torch

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import adjusted_rand_score, silhouette_score

# --------------------------------------------------------------------------- #
# Global deterministic seed
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

# --------------------------------------------------------------------------- #
# 1️⃣  Data locations
# --------------------------------------------------------------------------- #
# User‑provided source – change only if the file lives somewhere else
DATA_SRC = Path(
    r"C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\iris.csv"
)
DATA_DST = PROJECT_ROOT / "data" / "Iris.csv"

# --------------------------------------------------------------------------- #
# 2️⃣  Artefact hierarchy (must match the required output folder)
# --------------------------------------------------------------------------- #
ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"

RAW_ARTIFACT_DIR = ARTIFACT_ROOT / "raw"
PREP_ARTIFACT_DIR = ARTIFACT_ROOT / "prep"
MODEL_ARTIFACT_DIR = ARTIFACT_ROOT / "model"
REPORT_ARTIFACT_DIR = ARTIFACT_ROOT / "reports"
SPLITS_DIR = PROJECT_ROOT / "splits"

# Ensure directory hierarchy exists
for d in (RAW_ARTIFACT_DIR, PREP_ARTIFACT_DIR, MODEL_ARTIFACT_DIR,
          REPORT_ARTIFACT_DIR, SPLITS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# 3️⃣  Filenames inside the artefact hierarchy
# --------------------------------------------------------------------------- #
X_RAW_PATH = RAW_ARTIFACT_DIR / "X_raw.parquet"
Y_TRUE_PATH = RAW_ARTIFACT_DIR / "y_true.parquet"

TRAIN_IDX_PATH = SPLITS_DIR / "train_idx.npy"
VAL_IDX_PATH = SPLITS_DIR / "val_idx.npy"
TEST_IDX_PATH = SPLITS_DIR / "test_idx.npy"

SCALER_PATH = PREP_ARTIFACT_DIR / "scaler.joblib"
X_PREP_PATH = PREP_ARTIFACT_DIR / "X_prepped.parquet"

CENTROIDS_PATH = MODEL_ARTIFACT_DIR / "kmeans_centroids.pt"
KMEANS_CFG_PATH = MODEL_ARTIFACT_DIR / "kmeans_config.yaml"

EVAL_JSON_PATH = REPORT_ARTIFACT_DIR / "evaluation.json"

# --------------------------------------------------------------------------- #
# 4️⃣  K‑Means hyper‑parameters
# --------------------------------------------------------------------------- #
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
# 5️⃣  Copy the user‑provided CSV to the canonical location
# --------------------------------------------------------------------------- #
def copy_raw_csv() -> None:
    """Copy the source CSV to ``data/Iris.csv`` (overwrites if present)."""
    DATA_DST.parent.mkdir(parents=True, exist_ok=True)
    if not DATA_SRC.is_file():
        raise FileNotFoundError(f"Source CSV not found at {DATA_SRC}")
    # Simple binary copy – idempotent
    DATA_DST.write_bytes(DATA_SRC.read_bytes())
    print(f"[INFO] Copied raw CSV to {DATA_DST}")


# --------------------------------------------------------------------------- #
# 6️⃣  Load CSV, normalise column names, and persist raw artefacts
# --------------------------------------------------------------------------- #
def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust normalisation:
    * lower‑case
    * replace spaces, hyphens and parentheses with underscores
    * strip trailing underscores
    * map known variants to the canonical feature names
    * ensure a numeric ``target`` column exists
    """
    # 1️⃣  Generic clean‑up
    df = df.rename(columns=lambda c: (
        c.strip()
         .lower()
         .replace(" ", "_")
         .replace("-", "_")
         .replace("(", "")
         .replace(")", "")
         .replace(".", "")
    ))

    # 2️⃣  Remove common suffixes like "cm"
    df = df.rename(columns=lambda c: c.replace("_cm", "").replace("cm", ""))

    # 3️⃣  Expected feature set
    expected = {
        "sepal_length": "sepal_length",
        "sepallength": "sepal_length",
        "sepal_length_cm": "sepal_length",
        "sepalwidth": "sepal_width",
        "sepal_width": "sepal_width",
        "sepal_width_cm": "sepal_width",
        "petallength": "petal_length",
        "petal_length": "petal_length",
        "petal_length_cm": "petal_length",
        "petalwidth": "petal_width",
        "petal_width": "petal_width",
        "petal_width_cm": "petal_width",
    }

    # Build a mapping from current columns to canonical ones
    col_mapping = {}
    for col in df.columns:
        if col in expected:
            col_mapping[col] = expected[col]

    # If we could not map all four features, fall back to positional assignment
    if len(col_mapping) < 4:
        # Assume the first four columns are the features
        feature_cols = list(df.columns)[:4]
        col_mapping = {c: name for c, name in zip(feature_cols,
                                                   ["sepal_length", "sepal_width",
                                                    "petal_length", "petal_width"])}
    df = df.rename(columns=col_mapping)

    # ------------------------------------------------------------------- #
    # 4️⃣  Handle the label column
    # ------------------------------------------------------------------- #
    label_col = None
    for candidate in ("target", "species", "class", "label"):
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        # Pick the last column as a last resort
        label_col = df.columns[-1]

    # Convert string labels to integers if needed
    if not np.issubdtype(df[label_col].dtype, np.integer):
        le = LabelEncoder()
        df["target"] = le.fit_transform(df[label_col].astype(str))
    else:
        df["target"] = df[label_col].astype(int)

    # Keep only the canonical feature columns + target, in the right order
    df = df[["sepal_length", "sepal_width",
             "petal_length", "petal_width", "target"]]
    return df


def load_and_save_raw() -> None:
    """Read the CSV, normalise column names, and persist raw X & y as Parquet."""
    df = pd.read_csv(DATA_DST)
    df = _normalise_columns(df)

    # Separate features and true labels
    X_raw = df.drop(columns=["target"]).astype(np.float32)
    y_true = df["target"].astype(np.int64)

    # Persist as Parquet (fast columnar I/O)
    X_raw.to_parquet(X_RAW_PATH, index=False)
    y_true.to_frame(name="target").to_parquet(Y_TRUE_PATH, index=False)

    print(f"[INFO] Saved raw feature matrix to {X_RAW_PATH}")
    print(f"[INFO] Saved true labels to {Y_TRUE_PATH}")


# --------------------------------------------------------------------------- #
# 7️⃣  Stratified split (70 % train / 20 % val / 10 % test)
# --------------------------------------------------------------------------- #
def stratified_split() -> None:
    """Create reproducible stratified train/val/test index files."""
    y = pd.read_parquet(Y_TRUE_PATH)["target"].values

    # First split: train (70 %) vs temporary (30 %)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=SEED)
    train_idx, temp_idx = next(sss1.split(np.zeros_like(y), y))

    # Second split: val (20 %) vs test (10 %) – performed on the temporary set
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1 / 3, random_state=SEED)
    val_idx, test_idx = next(sss2.split(np.zeros_like(temp_idx), y[temp_idx]))

    # Convert back to original indices
    val_idx = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]

    # Save the three index arrays
    np.save(TRAIN_IDX_PATH, train_idx)
    np.save(VAL_IDX_PATH, val_idx)
    np.save(TEST_IDX_PATH, test_idx)

    print(f"[INFO] Saved split indices – train: {len(train_idx)}, "
          f"val: {len(val_idx)}, test: {len(test_idx)}")


# --------------------------------------------------------------------------- #
# 8️⃣  Standard‑scale (fit on train only)
# --------------------------------------------------------------------------- #
def scale_features() -> None:
    """Fit StandardScaler on the training split and transform the whole dataset."""
    X = pd.read_parquet(X_RAW_PATH).values.astype(np.float32)   # shape (150, 4)
    train_idx = np.load(TRAIN_IDX_PATH)

    scaler = StandardScaler()
    scaler.fit(X[train_idx])          # fit only on training data

    # Persist the scaler for later inference
    joblib.dump(scaler, SCALER_PATH)

    # Transform the entire dataset
    X_scaled = scaler.transform(X)

    # Save the scaled matrix (same column order as before)
    pd.DataFrame(
        X_scaled,
        columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]
    ).to_parquet(X_PREP_PATH, index=False)

    print(f"[INFO] Fitted StandardScaler (saved to {SCALER_PATH})")
    print(f"[INFO] Saved scaled feature matrix to {X_PREP_PATH}")


# --------------------------------------------------------------------------- #
# 9️⃣  Pure‑PyTorch K‑Means
# --------------------------------------------------------------------------- #
def train_kmeans() -> None:
    """Train deterministic K‑Means on the scaled training data."""
    X_scaled = pd.read_parquet(X_PREP_PATH).values.astype(np.float32)
    train_idx = np.load(TRAIN_IDX_PATH)

    X_train = torch.from_numpy(X_scaled[train_idx])  # (N_train, 4)

    # Initialise centroids by sampling without replacement from the training data
    torch.manual_seed(SEED)
    perm = torch.randperm(X_train.shape[0])
    centroids = X_train[perm[:N_CLUSTERS]].clone()  # (K, D)

    final_iter = MAX_ITER
    for iteration in range(1, MAX_ITER + 1):
        # Compute squared Euclidean distances (using torch.cdist)
        dists = torch.cdist(X_train, centroids, p=2) ** 2  # (N, K)

        # Assign each point to the nearest centroid
        assignments = torch.argmin(dists, dim=1)           # (N,)

        # Re‑compute centroids
        new_centroids = []
        for k in range(N_CLUSTERS):
            mask = assignments == k
            if mask.any():
                new_centroids.append(X_train[mask].mean(dim=0))
            else:
                # Empty cluster – re‑initialise from a random training point
                rand_idx = torch.randint(0, X_train.shape[0], (1,))
                new_centroids.append(X_train[rand_idx])
        new_centroids = torch.stack(new_centroids)        # (K, D)

        # Convergence check
        shift = torch.norm(centroids - new_centroids, dim=1).max()
        centroids = new_centroids
        if shift.item() < TOL:
            final_iter = iteration
            print(f"[INFO] K‑Means converged after {iteration} iterations "
                  f"(shift={shift:.6f})")
            break
    else:
        print(f"[WARN] K‑Means reached max_iter={MAX_ITER} without full convergence "
              f"(last shift={shift:.6f})")

    # Persist centroids (CPU tensor) and config
    torch.save(centroids.cpu(), CENTROIDS_PATH)

    cfg = {
        "n_clusters": N_CLUSTERS,
        "max_iter": MAX_ITER,
        "tolerance": TOL,
        "seed": SEED,
        "final_iteration": int(final_iter),
    }
    with open(KMEANS_CFG_PATH, "w") as f:
        yaml.safe_dump(cfg, f)

    print(f"[INFO] Saved centroids to {CENTROIDS_PATH}")
    print(f"[INFO] Saved K‑Means config to {KMEANS_CFG_PATH}")


# --------------------------------------------------------------------------- #
# 🔧  Helper: assign points to nearest centroid
# --------------------------------------------------------------------------- #
def assign_clusters(X: np.ndarray, centroids: torch.Tensor) -> np.ndarray:
    """
    Parameters
    ----------
    X : np.ndarray, shape (N, D)
        Scaled feature matrix (already transformed by the StandardScaler).
    centroids : torch.Tensor, shape (K, D)
        Centroids learned by K‑Means (CPU tensor).

    Returns
    -------
    np.ndarray, shape (N,)
        Cluster index for each row in ``X``.
    """
    X_tensor = torch.from_numpy(X.astype(np.float32))
    dists = torch.cdist(X_tensor, centroids, p=2) ** 2
    assignments = torch.argmin(dists, dim=1).numpy()
    return assignments


# --------------------------------------------------------------------------- #
# 📊  Evaluation (ARI & Silhouette) on validation and test splits
# --------------------------------------------------------------------------- #
def evaluate() -> None:
    X_scaled = pd.read_parquet(X_PREP_PATH).values.astype(np.float32)
    y_true = pd.read_parquet(Y_TRUE_PATH)["target"].values.astype(np.int64)

    centroids = torch.load(CENTROIDS_PATH)                     # (K, D)

    val_idx = np.load(VAL_IDX_PATH)
    test_idx = np.load(TEST_IDX_PATH)

    val_pred = assign_clusters(X_scaled[val_idx], centroids)
    test_pred = assign_clusters(X_scaled[test_idx], centroids)

    ari_val = adjusted_rand_score(y_true[val_idx], val_pred)
    ari_test = adjusted_rand_score(y_true[test_idx], test_pred)

    sil_val = silhouette_score(X_scaled[val_idx], val_pred)
    sil_test = silhouette_score(X_scaled[test_idx], test_pred)

    # Load final iteration from the K‑Means config for completeness
    with open(KMEANS_CFG_PATH, "r") as f:
        kmeans_cfg = yaml.safe_load(f)

    evaluation = {
        "timestamp": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "model": "kmeans_pytorch",
        "n_clusters": N_CLUSTERS,
        "final_iteration": kmeans_cfg.get("final_iteration"),
        "ari_val": float(ari_val),
        "ari_test": float(ari_test),
        "silhouette_val": float(sil_val),
        "silhouette_test": float(sil_test),
        # Optional meta‑data for reproducibility
        "scaler_checksum": sha256_of_file(SCALER_PATH),
        "centroids_checksum": sha256_of_file(CENTROIDS_PATH),
    }

    # Persist JSON report
    with open(EVAL_JSON_PATH, "w") as f:
        json.dump(evaluation, f, indent=2)

    print(f"[INFO] Evaluation saved to {EVAL_JSON_PATH}")
    print(json.dumps(evaluation, indent=2))


# --------------------------------------------------------------------------- #
# 🚀  Main orchestration
# --------------------------------------------------------------------------- #
def main() -> None:
    # 1️⃣  Copy raw CSV to the canonical location
    copy_raw_csv()

    # 2️⃣  Load CSV, normalise column names, and persist raw artefacts
    load_and_save_raw()

    # 3️⃣  Stratified train/val/test split
    stratified_split()

    # 4️⃣  Standard‑scale (fit on train only, transform all)
    scale_features()

    # 5️⃣  Train deterministic K‑Means in PyTorch
    train_kmeans()

    # 6️⃣  Evaluate on validation & test splits
    evaluate()


if __name__ == "__main__":
    main()
