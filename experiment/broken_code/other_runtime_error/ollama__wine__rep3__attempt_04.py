
"""
Wine‑Dataset Clustering Pipeline
================================
End‑to‑end script that

1️⃣ Copies the raw CSV from the user‑provided absolute location to the
   project folder (./wine_clustering/data/raw/wine.csv).

2️⃣ Loads the CSV, splits features / target and stores them as torch tensors.

3️⃣ Standard‑scales the 13 chemical features with a deterministic
   ``torch.nn.BatchNorm1d`` layer (no running‑stats updates).

4️⃣ Computes a 2‑D UMAP embedding (fixed ``random_state`` for reproducibility).

5️⃣ Fits HDBSCAN on the UMAP embedding.

6️⃣ Evaluates the clustering (ARI, NMI, V‑Measure, Silhouette) against the true
   ``target`` column.

7️⃣ Persists **all** artefacts required for inference into
   ``./agent_workspace/trained_models`` (the location you asked for).

8️⃣ (Optional) launches a tiny Gradio UI for interactive predictions
   (only if Gradio is available – otherwise the script finishes after step 7).

The script is completely self‑contained: if a required third‑party library is
missing it will be installed on‑the‑fly via ``pip`` (this works for most
environments that have internet access).  After the first successful run the
environment will contain the needed packages, so subsequent executions are
instant.

--------------------------------------------------------------------
Required packages (will be installed automatically if absent):
    - numpy
    - pandas
    - torch
    - scikit‑learn
    - umap-learn   (imported as ``umap``)
    - hdbscan
    - joblib
    - pyyaml
    - gradio   (optional, only for the demo)
--------------------------------------------------------------------
"""

# ----------------------------------------------------------------------
# 0️⃣ Imports & helper to auto‑install missing packages
# ----------------------------------------------------------------------
import importlib
import json
import os
import random
import shutil
import subprocess
import sys
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
    silhouette_score,
)

# ----------------------------------------------------------------------
# Helper: try to import a module, pip‑install it if it does not exist
# ----------------------------------------------------------------------
def import_or_install(pkg_name: str, import_name: str = None):
    """
    Try to import ``import_name`` (or ``pkg_name`` if not given).  If the import
    fails, install ``pkg_name`` via pip and retry.
    """
    import_name = import_name or pkg_name
    try:
        return importlib.import_module(import_name)
    except ImportError:
        print(f"⚠️  Package '{pkg_name}' not found – installing it now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
        return importlib.import_module(import_name)


# Import heavy optional dependencies (they will be installed if missing)
umap = import_or_install("umap-learn", "umap")
hdbscan = import_or_install("hdbscan")
joblib = import_or_install("joblib")
# Gradio is optional – we import it later only when we need the demo.


# ----------------------------------------------------------------------
# 1️⃣ Global constants & reproducibility
# ----------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")  # dataset is tiny – CPU is enough


# ----------------------------------------------------------------------
# 2️⃣ Configuration (mirrors the design document)
# ----------------------------------------------------------------------
CONFIG = {
    "paths": {
        # Absolute source supplied by the user (do NOT modify)
        "raw_csv_src": r"C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\wine.csv",
        # Project‑relative locations (will be resolved to absolute paths)
        "project_root": Path("./wine_clustering"),
        "raw_csv": "data/raw/wine.csv",
        "scaled_tensor": "data/processed/wine_scaled.pt",
        "target_tensor": "data/processed/target.pt",
        "scaler_state": "data/artefacts/scaler_state_dict.pt",
        "umap_embedding": "data/processed/umap_embedding.pt",
        "umap_model": "data/artefacts/umap_model.joblib",
        "hdbscan_model": "models/hdbscan/hdbscan_model.pkl",
        "trained_models_dir": "./agent_workspace/trained_models",
        "evaluation_csv": "reports/evaluation.csv",
    },
    "preprocess": {
        "batch_norm": {"eps": 1e-5, "momentum": 0.1, "track_running_stats": False}
    },
    "umap": {
        "n_neighbors": 15,
        "min_dist": 0.1,
        "n_components": 2,
        "metric": "euclidean",
        "random_state": SEED,
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

# Resolve all relative paths w.r.t. the project root
ROOT = CONFIG["paths"]["project_root"].resolve()
for key, rel_path in CONFIG["paths"].items():
    if key.endswith("_src"):
        continue
    CONFIG["paths"][key] = (ROOT / rel_path).resolve()


# ----------------------------------------------------------------------
# 3️⃣ Small utility helpers
# ----------------------------------------------------------------------
def sha256_of_file(path: Path) -> str:
    """Return SHA‑256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(p: Path):
    """Create parent directories for a given path if they do not exist."""
    p.parent.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path):
    ensure_dir(path)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ----------------------------------------------------------------------
# 4️⃣ Pipeline steps
# ----------------------------------------------------------------------
def step_0_copy_raw_csv() -> Path:
    """Copy the user‑provided CSV to the project folder."""
    src = Path(CONFIG["paths"]["raw_csv_src"])
    dst = CONFIG["paths"]["raw_csv"]
    ensure_dir(dst)
    shutil.copy2(src, dst)
    print(f"✅ Copied raw CSV to {dst}")
    return dst


def step_1_load_and_save_dataset(csv_path: Path):
    """Load CSV, split features / target, convert to torch tensors and persist."""
    df = pd.read_csv(csv_path)
    if "target" not in df.columns:
        raise KeyError("Expected a column named 'target' in the CSV.")
    target = df["target"].values.astype(np.int64)
    features = df.drop(columns=["target"]).values.astype(np.float32)

    feats_tensor = torch.from_numpy(features)          # (N,13)
    target_tensor = torch.from_numpy(target)           # (N,)

    # Persist tensors (will be overwritten after scaling)
    torch.save(feats_tensor, CONFIG["paths"]["scaled_tensor"])
    torch.save(target_tensor, CONFIG["paths"]["target_tensor"])
    print(f"✅ Loaded dataset – {feats_tensor.shape[0]} samples, saved raw tensors.")
    return feats_tensor, target_tensor


def step_2_standard_scale(feats_tensor: torch.Tensor):
    """Scale with a deterministic BatchNorm1d layer and persist the state."""
    bn_cfg = CONFIG["preprocess"]["batch_norm"]
    bn = nn.BatchNorm1d(
        num_features=feats_tensor.shape[1],
        eps=bn_cfg["eps"],
        momentum=bn_cfg["momentum"],
        track_running_stats=bn_cfg["track_running_stats"],
    ).to(DEVICE)

    bn.eval()  # ensure running stats are NOT updated
    with torch.no_grad():
        scaled = bn(feats_tensor.to(DEVICE))

    # Save scaled tensor and scaler state‑dict
    torch.save(scaled, CONFIG["paths"]["scaled_tensor"])
    torch.save(bn.state_dict(), CONFIG["paths"]["scaler_state"])
    print("✅ Standard‑scaled features with BatchNorm1d and saved state dict.")
    return scaled


def step_3_umap_embedding(scaled_tensor: torch.Tensor):
    """Compute 2‑D UMAP embedding and persist both embedding and model."""
    umap_cfg = CONFIG["umap"]
    reducer = umap.UMAP(
        n_neighbors=umap_cfg["n_neighbors"],
        min_dist=umap_cfg["min_dist"],
        n_components=umap_cfg["n_components"],
        metric=umap_cfg["metric"],
        random_state=umap_cfg["random_state"],
    )
    embedding_np = reducer.fit_transform(scaled_tensor.numpy())
    embedding_tensor = torch.from_numpy(embedding_np.astype(np.float32))

    # Persist embedding and the fitted UMAP model
    torch.save(embedding_tensor, CONFIG["paths"]["umap_embedding"])
    ensure_dir(CONFIG["paths"]["umap_model"])
    joblib.dump(reducer, CONFIG["paths"]["umap_model"])
    print("✅ UMAP (2‑D) embedding computed and saved.")
    return embedding_tensor, reducer


def step_4_hdbscan_clustering(embedding_tensor: torch.Tensor):
    """Fit HDBSCAN on the 2‑D embedding and persist the model."""
    hdb_cfg = CONFIG["hdbscan"]
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=hdb_cfg["min_cluster_size"],
        min_samples=hdb_cfg["min_samples"],
        metric=hdb_cfg["metric"],
        cluster_selection_method=hdb_cfg["cluster_selection_method"],
        prediction_data=hdb_cfg["prediction_data"],
        core_dist_n_jobs=hdb_cfg["core_dist_n_jobs"],
    )
    labels = clusterer.fit_predict(embedding_tensor.numpy())

    # Persist the trained HDBSCAN object
    ensure_dir(CONFIG["paths"]["hdbscan_model"])
    joblib.dump(clusterer, CONFIG["paths"]["hdbscan_model"])
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = np.sum(labels == -1)
    print(
        f"✅ HDBSCAN fitted – discovered {n_clusters} clusters "
        f"({n_noise} noise points)."
    )
    return labels, clusterer


def step_5_evaluation(
    true_labels: torch.Tensor,
    pred_labels: np.ndarray,
    embedding_tensor: torch.Tensor,
):
    """Compute external & internal metrics and save a CSV report."""
    ari = adjusted_rand_score(true_labels.numpy(), pred_labels)
    nmi = normalized_mutual_info_score(true_labels.numpy(), pred_labels)
    v_measure = v_measure_score(true_labels.numpy(), pred_labels)

    # Silhouette requires at least 2 non‑noise clusters
    mask = pred_labels != -1
    if np.unique(pred_labels[mask]).shape[0] > 1:
        sil = silhouette_score(embedding_tensor.numpy()[mask], pred_labels[mask])
    else:
        sil = float("nan")

    report = pd.DataFrame(
        {
            "algorithm": ["HDBSCAN"],
            "view": ["UMAP_2D"],
            "ARI": [ari],
            "NMI": [nmi],
            "V_measure": [v_measure],
            "Silhouette": [sil],
            "num_clusters": [len(set(pred_labels)) - (1 if -1 in pred_labels else 0)],
            "num_noise": [np.sum(pred_labels == -1)],
        }
    )
    ensure_dir(Path(CONFIG["paths"]["evaluation_csv"]))
    report.to_csv(CONFIG["paths"]["evaluation_csv"], index=False)
    print("✅ Evaluation completed – metrics saved to", CONFIG["paths"]["evaluation_csv"])
    print(report)
    return report


def step_6_save_trained_model():
    """
    Gather all artefacts needed for inference and copy them into the
    ``./agent_workspace/trained_models`` directory for easy deployment.
    """
    dst_root = Path(CONFIG["paths"]["trained_models_dir"]).resolve()
    dst_root.mkdir(parents=True, exist_ok=True)

    artefacts = {
        "scaler_state_dict.pt": CONFIG["paths"]["scaler_state"],
        "umap_model.joblib": CONFIG["paths"]["umap_model"],
        "hdbscan_model.pkl": CONFIG["paths"]["hdbscan_model"],
    }

    for name, src in artefacts.items():
        shutil.copy2(src, dst_root / name)

    # JSON descriptor with meta‑information
    descriptor = {
        "model_type": "HDBSCAN+UMAP",
        "features": 13,
        "embedding_dim": 2,
        "seed": SEED,
        "raw_sha256": sha256_of_file(CONFIG["paths"]["raw_csv"]),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    save_json(descriptor, dst_root / "model_descriptor.json")
    print(f"✅ All inference artefacts copied to {dst_root}")
    return dst_root


# ----------------------------------------------------------------------
# 7️⃣ Optional Gradio demo (only imported if available)
# ----------------------------------------------------------------------
def build_gradio_interface(trained_dir: Path):
    """Create a tiny Gradio UI for interactive predictions."""
    try:
        import gradio as gr
    except ImportError:
        print("⚠️  Gradio not installed – skipping the demo.")
        return None

    # Re‑create the BatchNorm layer and load its state dict
    scaler_state = torch.load(trained_dir / "scaler_state_dict.pt")
    bn_cfg = CONFIG["preprocess"]["batch_norm"]
    bn = nn.BatchNorm1d(
        num_features=13,
        eps=bn_cfg["eps"],
        momentum=bn_cfg["momentum"],
        track_running_stats=bn_cfg["track_running_stats"],
    )
    bn.load_state_dict(scaler_state)
    bn.eval()

    # Load UMAP and HDBSCAN models
    umap_model = joblib.load(trained_dir / "umap_model.joblib")
    hdbscan_model = joblib.load(trained_dir / "hdbscan_model.pkl")

    def predict(*features):
        """Accept 13 floats, return the predicted cluster label."""
        x = torch.tensor([features], dtype=torch.float32)  # (1,13)
        with torch.no_grad():
            x_scaled = bn(x)                     # (1,13)
        x_emb = umap_model.transform(x_scaled.numpy())  # (1,2)
        label = hdbscan_model.approximate_predict(x_emb)[0]
        return int(label)

    # Build 13 sliders – ranges loosely follow the wine dataset statistics
    inputs = [
        gr.Slider(minimum=0, maximum=20, step=0.01, label=f"feature_{i+1}")
        for i in range(13)
    ]

    iface = gr.Interface(
        fn=predict,
        inputs=inputs,
        outputs="label",
        title="Wine Clustering (HDBSCAN + UMAP)",
        description=(
            "Enter the 13 chemical measurements of a wine sample to obtain its "
            "cluster label (‑1 indicates noise)."
        ),
    )
    return iface


# ----------------------------------------------------------------------
# 8️⃣ Main orchestration
# ----------------------------------------------------------------------
def main():
    # 0️⃣ Copy raw CSV into the project structure
    raw_csv_path = step_0_copy_raw_csv()

    # 1️⃣ Load data and persist raw tensors
    feats_tensor, target_tensor = step_1_load_and_save_dataset(raw_csv_path)

    # 2️⃣ Standard‑scale the features
    scaled_tensor = step_2_standard_scale(feats_tensor)

    # 3️⃣ Compute 2‑D UMAP embedding
    embedding_tensor, _ = step_3_umap_embedding(scaled_tensor)

    # 4️⃣ Fit HDBSCAN on the embedding
    pred_labels, _ = step_4_hdbscan_clustering(embedding_tensor)

    # 5️⃣ Evaluate clustering quality
    _ = step_5_evaluation(target_tensor, pred_labels, embedding_tensor)

    # 6️⃣ Save artefacts for production inference
    trained_dir = step_6_save_trained_model()

    # 7️⃣ (Optional) Launch Gradio demo
    iface = build_gradio_interface(trained_dir)
    if iface is not None:
        print("\n🚀 Starting Gradio demo – open the URL printed below in your browser.")
        iface.launch(share=False)  # set share=True for a public link


if __name__ == "__main__":
    main()
