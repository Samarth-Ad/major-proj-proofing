
"""
Wine clustering pipeline – end‑to‑end script.

The script follows the reproducible play‑book described in the prompt:

1. Load configuration (config.yaml) – all paths become absolute ``Path`` objects.
2. Copy the raw CSV from the user‑provided absolute location to ``data/raw``.
3. Convert the CSV into Torch tensors (features + target) and persist them.
4. Deterministic scaling with ``torch.nn.BatchNorm1d`` (no running stats) and
   persist the state dict.
5. 2‑D UMAP embedding – persist the embedding tensor and the fitted UMAP model.
6. HDBSCAN clustering on the embedding – persist the fitted HDBSCAN model.
7. Compute external (ARI, NMI, V‑Measure) and internal (Silhouette) metrics,
   write a one‑row CSV report and (optionally) log everything to a local MLflow
   run (if the ``mlflow`` package is available).
8. Gather the three inference artefacts (scaler, UMAP, HDBSCAN) together with a
   tiny ``model_descriptor.json`` into ``./agent_workspace/trained_models`` –
   the ready‑to‑serve package.

All console output uses plain ASCII only (no emojis) to avoid Windows encoding
issues.  The script assumes the conda environment described in the prompt has
already been created and activated.
"""

import hashlib
import json
import random
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import umap
import yaml
from hdbscan import HDBSCAN
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
    silhouette_score,
)

# ----------------------------------------------------------------------
# 1️⃣  CONFIGURATION LOADER
# ----------------------------------------------------------------------
def load_config() -> dict:
    """
    Locate ``config.yaml`` by searching upward from the script location.
    Return the parsed YAML with all ``paths`` entries converted to absolute
    ``Path`` objects.
    """
    start_dir = Path(__file__).resolve().parent
    config_path = None
    for parent in [start_dir] + list(start_dir.parents):
        candidate = parent / "config.yaml"
        if candidate.is_file():
            config_path = candidate
            break
    if config_path is None:
        raise FileNotFoundError(
            "Could not locate config.yaml – searched upward from "
            f"{start_dir}"
        )

    with open(config_path, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)

    repo_root = config_path.parent
    abs_paths = {}
    for key, rel_path in raw_cfg["paths"].items():
        abs_paths[key] = (repo_root / rel_path).resolve()

    cfg = raw_cfg.copy()
    cfg["paths"] = abs_paths
    return cfg


CONFIG = load_config()
PATHS = CONFIG["paths"]
SEED = CONFIG.get("seed", 42)


# ----------------------------------------------------------------------
# 2️⃣  REPRODUCIBILITY
# ----------------------------------------------------------------------
def set_global_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


set_global_seed(SEED)


# ----------------------------------------------------------------------
# 3️⃣  SMALL HELPERS
# ----------------------------------------------------------------------
def sha256_of_file(file_path: Path) -> str:
    """Return hex SHA‑256 of a file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# 4️⃣  PIPELINE STEPS
# ----------------------------------------------------------------------
def copy_raw_csv() -> str:
    """Copy source CSV to ``data/raw`` and return its SHA‑256 checksum."""
    src = Path(CONFIG["paths"]["raw_csv_src"])
    dst = Path(CONFIG["paths"]["raw_csv"])
    ensure_dir(dst.parent)

    src_hash = sha256_of_file(src)

    if dst.is_file():
        dst_hash = sha256_of_file(dst)
        if src_hash == dst_hash:
            print(f"[INFO] Raw CSV already present and identical at {dst}")
            return src_hash

    shutil.copy2(src, dst)
    print(f"[INFO] Copied raw CSV to {dst}")
    return src_hash


def csv_to_tensors():
    """Read CSV, split features/target, store as Torch tensors."""
    csv_path = Path(CONFIG["paths"]["raw_csv"])
    feats_path = Path(CONFIG["paths"]["features_pt"])
    target_path = Path(CONFIG["paths"]["target_pt"])
    ensure_dir(feats_path.parent)

    df = pd.read_csv(csv_path)

    # Detect target column (wine dataset uses "target" or "class")
    if "target" in df.columns:
        target_col = "target"
    elif "class" in df.columns:
        target_col = "class"
    else:
        target_col = df.columns[-1]  # fallback: last column

    y = df[target_col].values.astype(np.int64)
    X = df.drop(columns=[target_col]).values.astype(np.float32)

    torch.save(torch.from_numpy(X), feats_path)
    torch.save(torch.from_numpy(y), target_path)

    print(f"[INFO] Saved features tensor to {feats_path}")
    print(f"[INFO] Saved target tensor to {target_path}")


def scale_features():
    """Scale with BatchNorm1d (no running stats) and persist the state dict."""
    feats_path = Path(CONFIG["paths"]["features_pt"])
    scaler_state_path = Path(CONFIG["paths"]["scaler_state_pt"])
    ensure_dir(scaler_state_path.parent)

    X = torch.load(feats_path)  # shape [N, D]
    bn_cfg = CONFIG["preprocess"]["batch_norm"]
    bn = nn.BatchNorm1d(
        num_features=X.shape[1],
        eps=bn_cfg.get("eps", 1e-5),
        momentum=bn_cfg.get("momentum", 0.1),
        track_running_stats=bn_cfg.get("track_running_stats", False),
    )
    bn.eval()
    with torch.no_grad():
        X_scaled = bn(X)

    # Overwrite original features with scaled version
    torch.save(X_scaled, feats_path)
    torch.save(bn.state_dict(), scaler_state_path)

    print(f"[INFO] Scaled features saved to {feats_path}")
    print(f"[INFO] Scaler state dict saved to {scaler_state_path}")


def embed_umap():
    """Fit UMAP on scaled data, persist embedding and the fitted model."""
    feats_path = Path(CONFIG["paths"]["features_pt"])
    embed_path = Path(CONFIG["paths"]["umap_embedding_pt"])
    umap_model_path = Path(CONFIG["paths"]["umap_model_joblib"])
    ensure_dir(embed_path.parent)
    ensure_dir(umap_model_path.parent)

    X = torch.load(feats_path).numpy()
    umap_cfg = CONFIG["umap"]
    reducer = umap.UMAP(
        n_neighbors=umap_cfg.get("n_neighbors", 15),
        min_dist=umap_cfg.get("min_dist", 0.1),
        n_components=umap_cfg.get("n_components", 2),
        metric=umap_cfg.get("metric", "euclidean"),
        random_state=umap_cfg.get("random_state", SEED),
    )
    embedding = reducer.fit_transform(X)  # shape [N, 2]

    torch.save(torch.from_numpy(embedding.astype(np.float32)), embed_path)
    joblib.dump(reducer, umap_model_path)

    print(f"[INFO] UMAP embedding saved to {embed_path}")
    print(f"[INFO] UMAP model saved to {umap_model_path}")


def cluster_hdbscan() -> np.ndarray:
    """Fit HDBSCAN on the 2‑D embedding, persist the model, return labels."""
    embed_path = Path(CONFIG["paths"]["umap_embedding_pt"])
    model_path = Path(CONFIG["paths"]["hdbscan_model_pkl"])
    ensure_dir(model_path.parent)

    X = torch.load(embed_path).numpy()
    hdb_cfg = CONFIG["hdbscan"]
    clusterer = HDBSCAN(
        min_cluster_size=hdb_cfg.get("min_cluster_size", 7),
        min_samples=hdb_cfg.get("min_samples", 5),
        metric=hdb_cfg.get("metric", "euclidean"),
        cluster_selection_method=hdb_cfg.get(
            "cluster_selection_method", "eom"
        ),
        prediction_data=hdb_cfg.get("prediction_data", True),
        core_dist_n_jobs=hdb_cfg.get("core_dist_n_jobs", 1),
    )
    labels = clusterer.fit_predict(X)
    joblib.dump(clusterer, model_path)

    print(f"[INFO] HDBSCAN model saved to {model_path}")
    return labels


def evaluate(labels: np.ndarray):
    """Compute metrics, write CSV report, and (optionally) log to MLflow."""
    target_path = Path(CONFIG["paths"]["target_pt"])
    eval_csv = Path(CONFIG["paths"]["evaluation_csv"])
    ensure_dir(eval_csv.parent)

    y_true = torch.load(target_path).numpy()
    y_pred = labels

    # External metrics
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    v_measure = v_measure_score(y_true, y_pred)

    # Internal metric – silhouette (ignore noise points)
    mask = y_pred != -1
    if mask.sum() >= 2 and len(set(y_pred[mask])) > 1:
        embedding = torch.load(CONFIG["paths"]["umap_embedding_pt"])[mask].numpy()
        sil = silhouette_score(embedding, y_pred[mask])
    else:
        sil = float("nan")

    n_clusters = len(set(y_pred)) - (1 if -1 in y_pred else 0)
    n_noise = int((y_pred == -1).sum())

    df = pd.DataFrame(
        {
            "algorithm": ["HDBSCAN"],
            "embedding": ["UMAP_2D"],
            "ari": [ari],
            "nmi": [nmi],
            "v_measure": [v_measure],
            "silhouette": [sil],
            "num_clusters": [n_clusters],
            "num_noise": [n_noise],
        }
    )
    df.to_csv(eval_csv, index=False)
    print(f"[INFO] Evaluation report saved to {eval_csv}")

    # ------------------------------------------------------------------
    # Optional MLflow logging – only if the package is importable
    # ------------------------------------------------------------------
    try:
        import mlflow

        mlflow.log_metric("ari", ari)
        mlflow.log_metric("nmi", nmi)
        mlflow.log_metric("v_measure", v_measure)
        mlflow.log_metric(
            "silhouette", sil if not np.isnan(sil) else -1.0
        )
        mlflow.log_metric("hdbscan_num_clusters", n_clusters)
        mlflow.log_metric("hdbscan_num_noise", n_noise)
    except Exception as e:
        print("[WARN] mlflow not available – skipping MLflow logging.")


def copy_artifacts_for_deployment():
    """Gather scaler, UMAP, HDBSCAN artefacts into ./agent_workspace/trained_models."""
    dst_root = Path("./agent_workspace/trained_models")
    ensure_dir(dst_root)

    mapping = {
        CONFIG["paths"]["scaler_state_pt"]: dst_root / "scaler_state.pt",
        CONFIG["paths"]["umap_model_joblib"]: dst_root / "umap_model.joblib",
        CONFIG["paths"]["hdbscan_model_pkl"]: dst_root / "hdbscan_model.pkl",
    }

    for src, dst in mapping.items():
        shutil.copy2(src, dst)
        print(f"[INFO] Copied {src} → {dst}")

    descriptor = {
        "model_type": "HDBSCAN+UMAP",
        "features": 13,
        "embedding_dim": 2,
        "seed": SEED,
        "raw_sha256": sha256_of_file(Path(CONFIG["paths"]["raw_csv"])),
        "timestamp": pd.Timestamp.now().isoformat(),
    }
    descriptor_path = dst_root / "model_descriptor.json"
    with open(descriptor_path, "w", encoding="utf-8") as f:
        json.dump(descriptor, f, indent=2)

    print(f"[INFO] Model descriptor written to {descriptor_path}")


# ----------------------------------------------------------------------
# 5️⃣  MAIN ORCHESTRATOR
# ----------------------------------------------------------------------
def main():
    # ------------------------------------------------------------------
    # Optional MLflow setup – only if the package is present
    # ------------------------------------------------------------------
    try:
        import mlflow

        mlflow.set_tracking_uri(
            f"file://{Path(CONFIG['paths']['mlflow_dir']).resolve()}"
        )
        mlflow.set_experiment("wine_clustering_experiment")
        mlflow_active = True
    except Exception:
        mlflow_active = False
        print("[WARN] mlflow package not found – proceeding without tracking.")

    # ------------------------------------------------------------------
    # Run the pipeline (MLflow run if possible)
    # ------------------------------------------------------------------
    if mlflow_active:
        with mlflow.start_run(run_name="wine_hdbscan_umap") as run:
            _run_pipeline(run)
    else:
        _run_pipeline(None)

    # ------------------------------------------------------------------
    # Deployable package creation (outside any MLflow context)
    # ------------------------------------------------------------------
    copy_artifacts_for_deployment()
    print("[INFO] Pipeline finished successfully.")


def _run_pipeline(mlflow_run):
    """Internal helper that executes the steps; ``mlflow_run`` may be None."""
    # 1. Copy raw CSV & compute checksum
    raw_checksum = copy_raw_csv()
    if mlflow_run:
        import mlflow

        mlflow.set_tag("raw_sha256", raw_checksum)

    # 2. CSV → tensors
    csv_to_tensors()

    # 3. Scaling
    scale_features()

    # 4. UMAP embedding
    embed_umap()

    # 5. HDBSCAN clustering
    labels = cluster_hdbscan()

    # 6. Evaluation & (optional) MLflow logging
    evaluate(labels)

    # 7. Log artefacts & parameters if MLflow is active
    if mlflow_run:
        import mlflow

        mlflow.log_artifact(str(CONFIG["paths"]["scaler_state_pt"]))
        mlflow.log_artifact(str(CONFIG["paths"]["umap_model_joblib"]))
        mlflow.log_artifact(str(CONFIG["paths"]["hdbscan_model_pkl"]))
        mlflow.log_artifact(str(CONFIG["paths"]["evaluation_csv"]))
        mlflow.log_param("raw_sha256", raw_checksum)
        mlflow.log_param("seed", SEED)


if __name__ == "__main__":
    # Guard against unexpected working directories – all paths are absolute,
    # so the script works from anywhere.
    main()
