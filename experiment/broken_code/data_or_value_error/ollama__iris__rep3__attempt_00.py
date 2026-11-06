
"""
Iris K‑Means clustering pipeline (pure‑torch implementation)

- Ingests the original CSV from the absolute user location.
- Validates schema, types, missing values, duplicates and basic ranges.
- Standard‑scales the four numeric features using PyTorch tensors.
- Trains a K‑Means model (k‑means++ init, n_init=20, max_iter=500, tol=1e‑5).
- Evaluates the clustering against the hidden target column (ARI, NMI, Silhouette).
- Persists the scaler, the centroids and a small JSON report in
  "./agent_workspace/trained_models".

All artefacts are written relative to the repository root, so the script can be
run from any working directory (e.g. `python pipeline.py`).

Required packages (see the specification):
pandas, numpy, torch, scikit‑learn, tqdm
"""

import os
import shutil
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

# --------------------------------------------------------------------------- #
# Global configuration (mirrors the yaml described in the spec)
# --------------------------------------------------------------------------- #
CONFIG = {
    "source_csv": r"C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\iris.csv",
    "repo_root": Path(__file__).parent,  # directory where this script lives
    "data_dir": Path("data/raw"),
    "artifacts_dir": Path("artifacts"),
    "model_dir": Path("./agent_workspace/trained_models"),
    "model_params": {
        "n_clusters": 3,
        "init": "kmeans_pp",
        "n_init": 20,
        "max_iter": 500,
        "tol": 1e-5,
        "random_state": 42,
    },
}
# Resolve all paths relative to the repo root
for key in ["data_dir", "artifacts_dir", "model_dir"]:
    CONFIG[key] = CONFIG["repo_root"] / CONFIG[key]

# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# 1️⃣ Ingestion
# --------------------------------------------------------------------------- #
def ingest() -> pd.DataFrame:
    """Copy the user CSV into the repo, read it and return a DataFrame."""
    ensure_dir(CONFIG["data_dir"])
    dest_csv = CONFIG["data_dir"] / "iris.csv"
    # Copy – overwrite if already present (useful for repeated runs)
    shutil.copy2(CONFIG["source_csv"], dest_csv)

    df = pd.read_csv(dest_csv)
    print(f"[ingest] Loaded {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


# --------------------------------------------------------------------------- #
# 2️⃣ Validation
# --------------------------------------------------------------------------- #
def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Perform schema, type, missing, duplicate and range checks."""
    expected_cols = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "target",
    ]
    # 2.1 schema
    if list(df.columns) != expected_cols:
        raise ValueError(
            f"Column mismatch. Expected {expected_cols}, got {list(df.columns)}"
        )
    # 2.2 dtypes
    for col in expected_cols[:-1]:
        if not pd.api.types.is_float_dtype(df[col]):
            raise TypeError(f"Column {col} must be float.")
    if not pd.api.types.is_integer_dtype(df["target"]):
        raise TypeError("Column target must be integer.")

    # 2.3 missing
    if df.isnull().any().any():
        raise ValueError("Missing values detected in the dataset.")

    # 2.4 duplicates
    if df.duplicated().any():
        dup_cnt = df.duplicated().sum()
        print(f"[validate] Found {dup_cnt} duplicate rows – dropping them.")
        df = df.drop_duplicates().reset_index(drop=True)

    # 2.5 range sanity (0‑100 cm)
    numeric = df[expected_cols[:-1]]
    out_of_range = ((numeric < 0) | (numeric > 100)).any().any()
    if out_of_range:
        print("[validate] Warning: some measurement values are out of the 0‑100 cm range.")

    # Save validated parquet
    ensure_dir(CONFIG["artifacts_dir"] / "validated")
    valid_path = CONFIG["artifacts_dir"] / "validated" / "iris_valid.parquet"
    df.to_parquet(valid_path, compression="snappy")
    print(f"[validate] Saved validated data to {valid_path}")
    return df


# --------------------------------------------------------------------------- #
# 3️⃣ Pre‑processing (standard scaling with torch)
# --------------------------------------------------------------------------- #
def preprocess(df: pd.DataFrame) -> torch.Tensor:
    """Compute mean/std with torch, scale the features and persist artefacts."""
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X_np = df[feature_cols].to_numpy(dtype=np.float32)  # (N,4)
    X = torch.from_numpy(X_np)  # float32 tensor

    # Compute mean & std (unbiased=False to match sklearn's StandardScaler)
    mean = X.mean(dim=0)
    std = X.std(dim=0, unbiased=False)

    # Save scaler
    ensure_dir(CONFIG["artifacts_dir"] / "scaler")
    scaler_path = CONFIG["artifacts_dir"] / "scaler" / "scaler.pt"
    torch.save({"mean": mean, "std": std}, scaler_path)
    print(f"[preprocess] Saved scaler to {scaler_path}")

    # Scale
    X_scaled = (X - mean) / std

    # Save scaled matrix
    ensure_dir(CONFIG["artifacts_dir"] / "features")
    X_scaled_path = CONFIG["artifacts_dir"] / "features" / "X_scaled.pt"
    torch.save(X_scaled, X_scaled_path)
    print(f"[preprocess] Saved scaled features to {X_scaled_path}")

    return X_scaled


# --------------------------------------------------------------------------- #
# 4️⃣ K‑Means implementation (pure torch)
# --------------------------------------------------------------------------- #
def _kmeans_pp_init(
    X: torch.Tensor, n_clusters: int, random_state: int
) -> torch.Tensor:
    """k‑means++ seeding using torch operations."""
    torch.manual_seed(random_state)
    N = X.shape[0]

    # Choose first centroid uniformly at random
    indices = torch.randperm(N)[:1]
    centroids = X[indices]

    # Choose remaining centroids
    for _ in range(1, n_clusters):
        # Compute squared distances to nearest existing centroid
        dist_sq = torch.min(
            torch.cdist(X, centroids, p=2).pow(2), dim=1
        ).values  # (N,)

        # Probability proportional to distance squared
        probs = dist_sq / dist_sq.sum()
        cumulative = torch.cumsum(probs, dim=0)
        r = torch.rand(1).item()
        next_idx = torch.searchsorted(cumulative, torch.tensor(r))
        centroids = torch.cat([centroids, X[next_idx].unsqueeze(0)], dim=0)

    return centroids  # (n_clusters, D)


def _run_loyd(
    X: torch.Tensor,
    init_centroids: torch.Tensor,
    max_iter: int,
    tol: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Run Lloyd's algorithm starting from `init_centroids`.

    Returns:
        centroids (tensor), assignments (tensor of shape N), final inertia (float)
    """
    centroids = init_centroids.clone()
    N = X.shape[0]

    for it in range(max_iter):
        # Distance matrix (N, K)
        dists = torch.cdist(X, centroids, p=2)  # Euclidean
        assignments = torch.argmin(dists, dim=1)  # (N,)

        # Compute new centroids
        new_centroids = []
        for k in range(centroids.shape[0]):
            mask = assignments == k
            if mask.any():
                new_centroids.append(X[mask].mean(dim=0))
            else:
                # Empty cluster – re‑initialize to a random point
                new_centroids.append(X[torch.randint(0, N, (1,))].squeeze(0))
        new_centroids = torch.stack(new_centroids)

        # Check convergence
        shift = torch.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift.item() < tol:
            break

    # Final inertia
    final_dists = torch.cdist(X, centroids, p=2)
    min_dists = final_dists[torch.arange(N), assignments]
    inertia = (min_dists.pow(2).sum()).item()
    return centroids, assignments, inertia


def train_kmeans(
    X_scaled: torch.Tensor,
    params: dict,
) -> tuple[torch.Tensor, float]:
    """Train K‑Means with the hyper‑parameters defined in `params`.

    Returns:
        best_centroids (tensor of shape (n_clusters, D))
        best_inertia (float)
    """
    n_clusters = params["n_clusters"]
    n_init = params["n_init"]
    max_iter = params["max_iter"]
    tol = params["tol"]
    random_state = params["random_state"]
    init_method = params["init"]

    best_inertia = float("inf")
    best_centroids = None

    for seed in range(n_init):
        # seeding
        if init_method == "kmeans_pp":
            init_centroids = _kmeans_pp_init(
                X_scaled, n_clusters=n_clusters, random_state=random_state + seed
            )
        else:
            # simple random init
            indices = torch.randperm(X_scaled.shape[0])[:n_clusters]
            init_centroids = X_scaled[indices]

        centroids, _, inertia = _run_loyd(
            X_scaled,
            init_centroids,
            max_iter=max_iter,
            tol=tol,
        )

        if inertia < best_inertia:
            best_inertia = inertia
            best_centroids = centroids

    print(f"[train_kmeans] Best inertia after {n_init} inits: {best_inertia:.4f}")
    return best_centroids, best_inertia


# --------------------------------------------------------------------------- #
# 5️⃣ Evaluation
# --------------------------------------------------------------------------- #
def evaluate(
    X_scaled: torch.Tensor,
    centroids: torch.Tensor,
    y_true: np.ndarray,
) -> dict:
    """Compute ARI, NMI and Silhouette for the clustering."""
    # Predict cluster IDs
    dists = torch.cdist(X_scaled, centroids, p=2)
    y_pred = torch.argmin(dists, dim=1).cpu().numpy()

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    sil = silhouette_score(X_scaled.numpy(), y_pred)

    metrics = {"ARI": ari, "NMI": nmi, "Silhouette": sil}
    print("[evaluate] Metrics:", metrics)
    return metrics


# --------------------------------------------------------------------------- #
# 6️⃣ Persistence (model + report)
# --------------------------------------------------------------------------- #
def persist_model(
    scaler: dict,
    centroids: torch.Tensor,
    metrics: dict,
) -> None:
    """Save scaler, centroids and a JSON report to the model directory."""
    ensure_dir(CONFIG["model_dir"])

    # scaler already saved in artifacts; copy it for convenience
    shutil.copy2(
        CONFIG["artifacts_dir"] / "scaler" / "scaler.pt",
        CONFIG["model_dir"] / "scaler.pt",
    )
    # centroids
    cent_path = CONFIG["model_dir"] / "kmeans_centroids.pt"
    torch.save(centroids, cent_path)
    print(f"[persist] Saved centroids to {cent_path}")

    # JSON report
    report_path = CONFIG["model_dir"] / "report.json"
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[persist] Saved evaluation report to {report_path}")


# --------------------------------------------------------------------------- #
# 7️⃣ Main orchestration
# --------------------------------------------------------------------------- #
def main() -> None:
    set_seed(42)

    # 1️⃣ Ingest
    df_raw = ingest()

    # 2️⃣ Validate
    df_valid = validate(df_raw)

    # 3️⃣ Pre‑process (scaling)
    X_scaled = preprocess(df_valid)

    # 4️⃣ Train K‑Means
    centroids, inertia = train_kmeans(
        X_scaled, CONFIG["model_params"]
    )

    # Save centroids (also done later in persist_model)
    ensure_dir(CONFIG["artifacts_dir"] / "models")
    torch.save(centroids, CONFIG["artifacts_dir"] / "models" / "kmeans.pt")

    # 5️⃣ Evaluation – need true labels
    y_true = df_valid["target"].to_numpy()
    metrics = evaluate(X_scaled, centroids, y_true)
    metrics["inertia"] = inertia

    # 6️⃣ Persist everything for downstream deployment
    # Load scaler dict from file (used only for copying)
    scaler_dict = torch.load(CONFIG["artifacts_dir"] / "scaler" / "scaler.pt")
    persist_model(scaler_dict, centroids, metrics)

    print("\nPipeline finished. All artefacts are stored under:")
    print(f"  - Scaler & scaled data: {CONFIG['artifacts_dir']}")
    print(f"  - Deployable model & report: {CONFIG['model_dir']}")


if __name__ == "__main__":
    main()
