
"""
Wine clustering pipeline – end‑to‑end script.

This script follows the reproducible play‑book described in the prompt:

1. Copy the raw CSV from the user‑provided absolute location to ``data/raw``.
2. Convert the CSV into Torch tensors (features + target) and persist them.
3. Deterministic scaling with ``torch.nn.BatchNorm1d`` (no running stats) and
   persist the state dict.
4. 2‑D UMAP embedding – persist the embedding tensor and the fitted UMAP model.
5. HDBSCAN clustering on the embedding – persist the fitted HDBSCAN model.
6. Compute external (ARI, NMI, V‑Measure) and internal (Silhouette) metrics,
   write a one‑row CSV report.
7. Gather the three inference artefacts (scaler, UMAP, HDBSCAN) together with a
   tiny ``model_descriptor.json`` into ``./agent_workspace/trained_models`` –
   the ready‑to‑serve package.

All console output uses plain ASCII only (no emojis) to avoid Windows encoding
issues.  The script assumes the conda environment described in the prompt has
already been created and activated.
"""

import hashlib
import json
import shutil
import random
import sys
from pathlib import Path
from datetime import datetime

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
# 1️⃣  CONFIGURATION (hard‑coded for a self‑contained script)
# ----------------------------------------------------------------------
CONFIG = {
    "paths": {
        # Absolute source CSV – change only if the file moves
        "raw_csv_src": r"C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\wine.csv",
        # Relative paths – will be resolved against the repository root
        "raw_csv": "data/raw/wine.csv",
        "features_pt": "data/processed/features.pt",
        "target_pt": "data/processed/target.pt",
        "scaler_state_pt": "data/artefacts/scaler_state.pt",
        "umap_embedding_pt": "data/processed/umap2d.pt",
        "umap_model_joblib": "data/artefacts/umap_model.joblib",
        "hdbscan_model_pkl": "models/hdbscan/hdbscan_model.pkl",
        "evaluation_csv": "reports/evaluation.csv",
        "trained_models_dir": "agent_workspace/trained_models",
    },
    "seed": 42,
    "preprocess": {
        "batch_norm": {
            "eps": 1e-5,
            "momentum": 0.1,
            "track_running_stats": False,
        }
    },
    "umap": {
        "n_neighbors": 15,
        "min_dist": 0.1,
        "n_components": 2,
        "metric": "euclidean",
        "random_state": 42,
    },
    "hdbscan": {
        "min_cluster_size": 7,
        "min_samples": 5,
        "metric": "euclidean",
        "cluster_selection_method": "eom",
        "prediction_data": True,
        "core_dist_n_jobs": 1,
    },
}

# Resolve all relative paths to absolute paths (repo root = script parent directory)
REPO_ROOT = Path(__file__).resolve().parent
for key, rel_path in CONFIG["paths"].items():
    if key == "raw_csv_src":
        # keep absolute as‑is
        CONFIG["paths"][key] = Path(rel_path)
    else:
        CONFIG["paths"][key] = (REPO_ROOT / rel_path).resolve()

SEED = CONFIG["seed"]


# ----------------------------------------------------------------------
# 2️⃣  REPRODUCIBILITY
# ----------------------------------------------------------------------
def set_global_seed(seed: int) -> None:
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


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# 4️⃣  PIPELINE STEPS
# ----------------------------------------------------------------------
def copy_raw_csv() -> str:
    """Copy source CSV to ``data/raw`` and return its SHA‑256 checksum."""
    src = CONFIG["paths"]["raw_csv_src"]
    dst = CONFIG["paths"]["raw_csv"]
    ensure_dir(dst.parent)

    if not src.is_file():
        sys.exit(f"[ERROR] Source CSV not found at {src}")

    src_hash = sha256_of_file(src)

    if dst.is_file():
        dst_hash = sha256_of_file(dst)
        if src_hash == dst_hash:
            print(f"[INFO] Raw CSV already present and identical at {dst}")
            return src_hash

    shutil.copy2(src, dst)
    print(f"[INFO] Copied raw CSV to {dst}")
    return src_hash


def csv_to_tensors() -> None:
    """Read CSV, split features/target, store as Torch tensors."""
    csv_path = CONFIG["paths"]["raw_csv"]
    feats_path = CONFIG["paths"]["features_pt"]
    target_path = CONFIG["paths"]["target_pt"]
    ensure_dir(feats_path.parent)

    df = pd.read_csv(csv_path)

    # Identify target column – wine dataset uses "target" or "class"
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


def scale_features() -> None:
    """Scale with BatchNorm1d (no running stats) and persist the state dict."""
    feats_path = CONFIG["paths"]["features_pt"]
    scaler_state_path = CONFIG["paths"]["scaler_state_pt"]
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


def embed_umap() -> None:
    """Fit UMAP on scaled data, persist embedding and the fitted model."""
    feats_path = CONFIG["paths"]["features_pt"]
    embed_path = CONFIG["paths"]["umap_embedding_pt"]
    umap_model_path = CONFIG["paths"]["umap_model_joblib"]
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
    embed_path = CONFIG["paths"]["umap_embedding_pt"]
    model_path = CONFIG["paths"]["hdbscan_model_pkl"]
    ensure_dir(model_path.parent)

    X = torch.load(embed_path).numpy()
    hdb_cfg = CONFIG["hdbscan"]
    clusterer = HDBSCAN(
        min_cluster_size=hdb_cfg.get("min_cluster_size", 7),
        min_samples=hdb_cfg.get("min_samples", 5),
        metric=hdb_cfg.get("metric", "euclidean"),
        cluster_selection_method=hdb_cfg.get("cluster_selection_method", "eom"),
        prediction_data=hdb_cfg.get("prediction_data", True),
        core_dist_n_jobs=hdb_cfg.get("core_dist_n_jobs", 1),
    )
    labels = clusterer.fit_predict(X)
    joblib.dump(clusterer, model_path)

    print(f"[INFO] HDBSCAN model saved to {model_path}")
    return labels


def evaluate(labels: np.ndarray) -> None:
    """Compute metrics and write a one‑row CSV report."""
    target_path = CONFIG["paths"]["target_pt"]
    eval_csv = CONFIG["paths"]["evaluation_csv"]
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


def copy_artifacts_for_deployment(raw_checksum: str) -> None:
    """Gather scaler, UMAP, HDBSCAN artefacts into ./agent_workspace/trained_models."""
    dst_root = CONFIG["paths"]["trained_models_dir"]
    ensure_dir(dst_root)

    mapping = {
        CONFIG["paths"]["scaler_state_pt"]: dst_root / "scaler_state.pt",
        CONFIG["paths"]["umap_model_joblib"]: dst_root / "umap_model.joblib",
        CONFIG["paths"]["hdbscan_model_pkl"]: dst_root / "hdbscan_model.pkl",
    }

    for src, dst in mapping.items():
        shutil.copy2(src, dst)
        print(f"[INFO] Copied {src} -> {dst}")

    descriptor = {
        "model_type": "HDBSCAN+UMAP",
        "features": 13,
        "embedding_dim": 2,
        "seed": SEED,
        "raw_sha256": raw_checksum,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    descriptor_path = dst_root / "model_descriptor.json"
    with open(descriptor_path, "w", encoding="utf-8") as f:
        json.dump(descriptor, f, indent=2)
    print(f"[INFO] Model descriptor written to {descriptor_path}")


# ----------------------------------------------------------------------
# 5️⃣  MAIN ORCHESTRATOR
# ----------------------------------------------------------------------
def main() -> None:
    # 1. Copy raw CSV & compute checksum
    raw_checksum = copy_raw_csv()

    # 2. CSV → tensors
    csv_to_tensors()

    # 3. Scaling
    scale_features()

    # 4. UMAP embedding
    embed_umap()

    # 5. HDBSCAN clustering
    labels = cluster_hdbscan()

    # 6. Evaluation
    evaluate(labels)

    # 7. Package artefacts for deployment
    copy_artifacts_for_deployment(raw_checksum)

    print("[INFO] Pipeline finished successfully.")


if __name__ == "__main__":
    main()
