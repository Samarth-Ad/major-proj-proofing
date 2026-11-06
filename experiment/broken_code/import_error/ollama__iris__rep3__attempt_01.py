
"""
Iris K‑Means clustering pipeline (pure‑torch implementation)

- Copies the original CSV from the absolute user location into the repo.
- Validates the schema (renames columns if they contain spaces/units).
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

import json
import random
import shutil
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
    # ----------------------------------------------------------------------- #
    # 1️⃣  Absolute source location – change only here if the CSV moves.
    # ----------------------------------------------------------------------- #
    "source_csv": r"C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\iris.csv",
    # ----------------------------------------------------------------------- #
    # 2️⃣  Relative repository layout (all resolved from the script location)
    # ----------------------------------------------------------------------- #
    "repo_root": Path(__file__).parent,                     # repo root
    "data_dir": Path("data/raw"),                           # raw CSV copy
    "artifacts_dir": Path("artifacts"),                     # all artefacts
    "model_dir": Path("./agent_workspace/trained_models"), # final deployable model
    # ----------------------------------------------------------------------- #
    # 3️⃣  Model hyper‑parameters (the “optimal” values from the spec)
    # ----------------------------------------------------------------------- #
    "model_params": {
        "n_clusters": 3,
        "init": "kmeans_pp",   # k‑means++ seeding
        "n_init": 20,
        "max_iter": 500,
        "tol": 1e-5,
        "random_state": 42,
    },
}
# Resolve relative paths once, relative to the repo root
for key in ["data_dir", "artifacts_dir", "model_dir"]:
    CONFIG[key] = CONFIG["repo_root"] / CONFIG[key]

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42) -> None:
    """Make NumPy, random and torch deterministic."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(p: Path) -> None:
    """Create a directory (including parents) if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# 1️⃣ Ingestion
# --------------------------------------------------------------------------- #
def ingest() -> pd.DataFrame:
    """
    Copy the user CSV into the repository, read it with pandas and return the
    DataFrame.
    """
    ensure_dir(CONFIG["data_dir"])
    dest_csv = CONFIG["data_dir"] / "iris.csv"
    shutil.copy2(CONFIG["source_csv"], dest_csv)

    df = pd.read_csv(dest_csv)
    print(f"[ingest] Loaded {df.shape[0]} rows, {df.shape[1]} columns from {dest_csv}")

    # also keep a raw parquet backup (immutable snapshot)
    raw_parquet_dir = CONFIG["artifacts_dir"] / "raw"
    ensure_dir(raw_parquet_dir)
    raw_parquet_path = raw_parquet_dir / "iris_raw.parquet"
    df.to_parquet(raw_parquet_path, compression="snappy")
    print(f"[ingest] Saved raw parquet to {raw_parquet_path}")

    return df


# --------------------------------------------------------------------------- #
# 2️⃣ Validation (including column‑name normalisation)
# --------------------------------------------------------------------------- #
def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    The classic Iris CSV sometimes ships with column names that contain spaces
    and units, e.g. "sepal length (cm)".  This helper maps those to the expected
    snake_case names.
    """
    rename_map = {
        "sepal length (cm)": "sepal_length",
        "sepal width (cm)": "sepal_width",
        "petal length (cm)": "petal_length",
        "petal width (cm)": "petal_width",
        # keep the target column as‑is if it already matches
        "target": "target",
    }
    # Apply rename only for columns that exist in the map
    df = df.rename(columns={c: rename_map.get(c, c) for c in df.columns})
    return df


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform schema, type, missing‑value, duplicate and range checks.
    The cleaned DataFrame is saved as a parquet file.
    """
    df = _normalise_columns(df)

    expected_cols = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "target",
    ]

    # 2.1 Schema check
    if list(df.columns) != expected_cols:
        raise ValueError(
            f"Column mismatch. Expected {expected_cols}, got {list(df.columns)}"
        )

    # 2.2 Type check
    for col in expected_cols[:-1]:  # first four must be floats
        if not pd.api.types.is_float_dtype(df[col]):
            raise TypeError(f"Column '{col}' must be of float dtype.")
    if not pd.api.types.is_integer_dtype(df["target"]):
        raise TypeError("Column 'target' must be of integer dtype.")

    # 2.3 Missing values
    if df.isnull().any().any():
        raise ValueError("Missing values detected in the dataset.")

    # 2.4 Duplicates
    if df.duplicated().any():
        dup_cnt = df.duplicated().sum()
        print(f"[validate] Found {dup_cnt} duplicate rows – dropping them.")
        df = df.drop_duplicates().reset_index(drop=True)

    # 2.5 Range sanity (0‑100 cm) – only a warning, never abort
    numeric = df[expected_cols[:-1]]
    out_of_range = ((numeric < 0) | (numeric > 100)).any().any()
    if out_of_range:
        print("[validate] Warning: some measurement values are out of the 0‑100 cm range.")

    # Save validated parquet
    validated_dir = CONFIG["artifacts_dir"] / "validated"
    ensure_dir(validated_dir)
    valid_path = validated_dir / "iris_valid.parquet"
    df.to_parquet(valid_path, compression="snappy")
    print(f"[validate] Saved validated data to {valid_path}")

    return df


# --------------------------------------------------------------------------- #
# 3️⃣ Pre‑processing (standard scaling with torch)
# --------------------------------------------------------------------------- #
def preprocess(df: pd.DataFrame) -> torch.Tensor:
    """
    Compute column‑wise mean/std with torch, scale the features and persist the
    scaler and the scaled matrix.
    Returns the scaled tensor (N, 4).
    """
    feature_cols = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    X_np = df[feature_cols].to_numpy(dtype=np.float32)  # shape (N,4)
    X = torch.from_numpy(X_np)                           # torch.float32

    # Compute mean & std (unbiased=False → matches sklearn's StandardScaler)
    mean = X.mean(dim=0)
    std = X.std(dim=0, unbiased=False)

    # Save scaler (dictionary with two tensors)
    scaler_dir = CONFIG["artifacts_dir"] / "scaler"
    ensure_dir(scaler_dir)
    scaler_path = scaler_dir / "scaler.pt"
    torch.save({"mean": mean, "std": std}, scaler_path)
    print(f"[preprocess] Saved scaler to {scaler_path}")

    # Scale
    X_scaled = (X - mean) / std

    # Save scaled matrix
    feats_dir = CONFIG["artifacts_dir"] / "features"
    ensure_dir(feats_dir)
    X_scaled_path = feats_dir / "X_scaled.pt"
    torch.save(X_scaled, X_scaled_path)
    print(f"[preprocess] Saved scaled features to {X_scaled_path}")

    return X_scaled


# --------------------------------------------------------------------------- #
# 4️⃣ K‑Means implementation (pure torch)
# --------------------------------------------------------------------------- #
def _kmeans_pp_init(
    X: torch.Tensor, n_clusters: int, random_state: int
) -> torch.Tensor:
    """
    k‑means++ seeding using only torch operations.
    Returns a tensor of shape (n_clusters, D) containing the initial centroids.
    """
    torch.manual_seed(random_state)
    N = X.shape[0]

    # Choose the first centroid uniformly at random
    first_idx = torch.randint(0, N, (1,))
    centroids = X[first_idx]  # (1, D)

    # Choose remaining centroids
    for _ in range(1, n_clusters):
        # Compute squared distance to the nearest already‑chosen centroid
        dists = torch.cdist(X, centroids, p=2).pow(2)          # (N, k)
        min_dists = dists.min(dim=1).values                     # (N,)

        # Probability proportional to distance squared
        probs = min_dists / min_dists.sum()
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
    """
    Perform Lloyd's algorithm (the classic K‑Means iteration) starting from
    `init_centroids`.

    Returns
    -------
    centroids : torch.Tensor (K, D)
        Final centroid positions.
    assignments : torch.Tensor (N,)
        Cluster index for each sample.
    inertia : float
        Sum of squared distances of samples to their closest centroid.
    """
    centroids = init_centroids.clone()
    N = X.shape[0]

    for it in range(max_iter):
        # Distance matrix (N, K)
        dists = torch.cdist(X, centroids, p=2)          # Euclidean
        assignments = torch.argmin(dists, dim=1)        # (N,)

        # Re‑compute centroids
        new_centroids = []
        for k in range(centroids.shape[0]):
            mask = assignments == k
            if mask.any():
                new_centroids.append(X[mask].mean(dim=0))
            else:
                # Empty cluster → re‑initialize to a random point
                new_centroids.append(X[torch.randint(0, N, (1,))].squeeze(0))
        new_centroids = torch.stack(new_centroids)

        # Convergence check
        shift = torch.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift.item() < tol:
            break

    # Final inertia
    final_dists = torch.cdist(X, centroids, p=2)
    min_final = final_dists[torch.arange(N), assignments]
    inertia = min_final.pow(2).sum().item()

    return centroids, assignments, inertia


def train_kmeans(
    X_scaled: torch.Tensor,
    params: dict,
) -> tuple[torch.Tensor, float]:
    """
    Train K‑Means using the hyper‑parameters defined in `params`.

    Returns
    -------
    best_centroids : torch.Tensor (K, D)
    best_inertia   : float
    """
    n_clusters = params["n_clusters"]
    n_init = params["n_init"]
    max_iter = params["max_iter"]
    tol = params["tol"]
    random_state = params["random_state"]
    init_method = params["init"]

    best_inertia = float("inf")
    best_centroids = None

    for seed_offset in range(n_init):
        # --------------------------------------------------------------- #
        # Seeding
        # --------------------------------------------------------------- #
        if init_method == "kmeans_pp":
            init_centroids = _kmeans_pp_init(
                X_scaled,
                n_clusters=n_clusters,
                random_state=random_state + seed_offset,
            )
        else:  # fallback: random selection
            idx = torch.randperm(X_scaled.shape[0])[:n_clusters]
            init_centroids = X_scaled[idx]

        # --------------------------------------------------------------- #
        # Lloyd iteration
        # --------------------------------------------------------------- #
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
    """
    Compute ARI, NMI and Silhouette for the clustering.
    Returns a dict with the three metrics.
    """
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
    centroids: torch.Tensor,
    metrics: dict,
) -> None:
    """
    Save the centroids tensor and a JSON report to the final model directory.
    The scaler is also copied there for convenience.
    """
    ensure_dir(CONFIG["model_dir"])

    # Copy scaler for easy downstream loading
    src_scaler = CONFIG["artifacts_dir"] / "scaler" / "scaler.pt"
    dst_scaler = CONFIG["model_dir"] / "scaler.pt"
    shutil.copy2(src_scaler, dst_scaler)
    print(f"[persist] Copied scaler to {dst_scaler}")

    # Save centroids (the actual model)
    cent_path = CONFIG["model_dir"] / "kmeans_centroids.pt"
    torch.save(centroids, cent_path)
    print(f"[persist] Saved centroids to {cent_path}")

    # JSON report with metrics (including inertia)
    report_path = CONFIG["model_dir"] / "report.json"
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[persist] Saved evaluation report to {report_path}")


# --------------------------------------------------------------------------- #
# 7️⃣ Main orchestration
# --------------------------------------------------------------------------- #
def main() -> None:
    set_seed(42)

    # --------------------------------------------------------------- #
    # 1️⃣ Ingest
    # --------------------------------------------------------------- #
    df_raw = ingest()

    # --------------------------------------------------------------- #
    # 2️⃣ Validate (includes column normalisation)
    # --------------------------------------------------------------- #
    df_valid = validate(df_raw)

    # --------------------------------------------------------------- #
    # 3️⃣ Pre‑process (standard scaling)
    # --------------------------------------------------------------- #
    X_scaled = preprocess(df_valid)

    # --------------------------------------------------------------- #
    # 4️⃣ Train K‑Means
    # --------------------------------------------------------------- #
    centroids, inertia = train_kmeans(
        X_scaled, CONFIG["model_params"]
    )

    # Save centroids also under the generic artefacts folder (optional but handy)
    models_dir = CONFIG["artifacts_dir"] / "models"
    ensure_dir(models_dir)
    torch.save(centroids, models_dir / "kmeans.pt")
    print(f"[main] Saved centroids to {models_dir / 'kmeans.pt'}")

    # --------------------------------------------------------------- #
    # 5️⃣ Evaluation – need true labels
    # --------------------------------------------------------------- #
    y_true = df_valid["target"].to_numpy()
    metrics = evaluate(X_scaled, centroids, y_true)
    metrics["inertia"] = inertia  # add inertia for completeness

    # --------------------------------------------------------------- #
    # 6️⃣ Persist model & report for downstream deployment
    # --------------------------------------------------------------- #
    persist_model(centroids, metrics)

    print("\n=== Pipeline finished ===")
    print(f"Artifacts are stored under: {CONFIG['artifacts_dir']}")
    print(f"Deployable model & report are stored under: {CONFIG['model_dir']}")


if __name__ == "__main__":
    main()
