
"""
Wine‑Dataset Clustering Pipeline
--------------------------------

This script implements the end‑to‑end solution described in the design
document:

1. **Data retrieval** – copy the raw CSV from the user‑provided absolute
   location to the project folder.
2. **Pre‑processing** – standard‑scale the 13 chemical features with a
   deterministic ``torch.nn.BatchNorm1d`` layer.
3. **Dimensionality reduction** – 2‑D UMAP embedding (fixed random_state).
4. **Clustering** – HDBSCAN on the UMAP embedding.
5. **Evaluation** – ARI, NMI, V‑Measure and Silhouette (using the true
   ``target`` column).
6. **Model artefacts** – save the scaler state‑dict, the fitted UMAP
   transformer and the trained HDBSCAN model under
   ``./agent_workspace/trained_models``.
7. **Gradio demo** – a tiny web UI that accepts the 13 wine chemistry
   features and returns the predicted cluster label.

The script is self‑contained: running it from a fresh environment will
re‑create all required folders, download the data, train the pipeline and
produce a ready‑to‑deploy clustering model.

Dependencies (see ``environment.yml`` in the design doc):
    - python=3.11
    - numpy, pandas, scikit‑learn, umap‑learn, hdbscan, pyyaml,
      tqdm, torch, gradio
"""

import os
import shutil
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
import umap
import hdbscan
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
    silhouette_score,
)

# ----------------------------------------------------------------------
# 0️⃣  Global constants & reproducibility
# ----------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")  # dataset is tiny; keep everything on CPU

# ----------------------------------------------------------------------
# 1️⃣  Configuration (mirrors the design document)
# ----------------------------------------------------------------------
CONFIG = {
    "paths": {
        "raw_csv_src": r"C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\wine.csv",
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

# Resolve all relative paths w.r.t. project root
ROOT = Path(CONFIG["paths"]["project_root"])
for key, rel_path in CONFIG["paths"].items():
    if key.endswith("_src"):
        continue
    CONFIG["paths"][key] = ROOT / rel_path

# ----------------------------------------------------------------------
# 2️⃣  Helper utilities
# ----------------------------------------------------------------------
def sha256_of_file(path: Path) -> str:
    """Return the SHA‑256 checksum of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path):
    ensure_dir(path)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


# ----------------------------------------------------------------------
# 3️⃣  Step‑by‑step pipeline
# ----------------------------------------------------------------------
def step_0_copy_raw_csv():
    """Copy the user‑provided CSV to the project folder."""
    src = Path(CONFIG["paths"]["raw_csv_src"])
    dst = CONFIG["paths"]["raw_csv"]
    ensure_dir(dst)
    shutil.copy2(src, dst)
    print(f"✅ Copied raw CSV to {dst}")
    return dst


def step_1_load_and_save_dataset(csv_path: Path):
    """
    Load CSV, split features / target, convert to torch tensors and
    persist them.
    """
    df = pd.read_csv(csv_path)
    # The classic Wine dataset has a column named 'target' (0‑2)
    if "target" not in df.columns:
        raise KeyError("Expected a column named 'target' in the CSV.")
    target = df["target"].values.astype(np.int64)
    features = df.drop(columns=["target"]).values.astype(np.float32)

    # Convert to torch tensors
    feats_tensor = torch.from_numpy(features)  # shape (N,13)
    target_tensor = torch.from_numpy(target)  # shape (N,)

    # Save tensors
    torch.save(feats_tensor, CONFIG["paths"]["scaled_tensor"])
    torch.save(target_tensor, CONFIG["paths"]["target_tensor"])
    print(f"✅ Loaded dataset – {feats_tensor.shape[0]} samples, saved tensors.")
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

    # Put in eval mode so that running stats are *not* updated.
    bn.eval()
    with torch.no_grad():
        scaled = bn(feats_tensor.to(DEVICE))

    # Save scaled tensor and the scaler state dict
    torch.save(scaled, CONFIG["paths"]["scaled_tensor"])
    torch.save(bn.state_dict(), CONFIG["paths"]["scaler_state"])
    print("✅ Standard‑scaled features with BatchNorm1d and saved state dict.")
    return scaled


def step_3_umap_embedding(scaled_tensor: torch.Tensor):
    """Compute 2‑D UMAP embedding and persist both embedding and model."""
    import joblib

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

    # Persist
    torch.save(embedding_tensor, CONFIG["paths"]["umap_embedding"])
    ensure_dir(CONFIG["paths"]["umap_model"])
    joblib.dump(reducer, CONFIG["paths"]["umap_model"])
    print("✅ UMAP embedding (2‑D) computed and saved.")
    return embedding_tensor, reducer


def step_4_hdbscan_clustering(embedding_tensor: torch.Tensor):
    """Fit HDBSCAN on the 2‑D embedding and persist the model."""
    import joblib

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
    print(
        f"✅ HDBSCAN fitted – discovered {len(set(labels)) - (1 if -1 in labels else 0)} clusters "
        f"({np.sum(labels == -1)} noise points)."
    )
    return labels, clusterer


def step_5_evaluation(
    true_labels: torch.Tensor,
    pred_labels,
    embedding_tensor: torch.Tensor,
):
    """Compute external & internal clustering metrics and save a CSV report."""
    ari = adjusted_rand_score(true_labels.numpy(), pred_labels)
    nmi = normalized_mutual_info_score(true_labels.numpy(), pred_labels)
    v_measure = v_measure_score(true_labels.numpy(), pred_labels)

    # Silhouette requires at least 2 clusters (ignoring noise)
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
    dst_root = Path(CONFIG["paths"]["trained_models_dir"])
    dst_root.mkdir(parents=True, exist_ok=True)

    artefacts = {
        "scaler_state_dict.pt": CONFIG["paths"]["scaler_state"],
        "umap_model.joblib": CONFIG["paths"]["umap_model"],
        "hdbscan_model.pkl": CONFIG["paths"]["hdbscan_model"],
    }

    for name, src in artefacts.items():
        shutil.copy2(src, dst_root / name)

    # Also copy a tiny JSON descriptor with meta‑information
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
# 7️⃣  Gradio demo (optional but requested)
# ----------------------------------------------------------------------
def build_gradio_interface(trained_dir: Path):
    """
    Load the saved artefacts and expose a simple UI:
    - Input: 13 numeric fields (wine chemistry)
    - Output: predicted cluster label (int, -1 = noise)
    """
    import joblib

    # Load artefacts
    scaler_state = torch.load(trained_dir / "scaler_state_dict.pt")
    bn = nn.BatchNorm1d(num_features=13, eps=CONFIG["preprocess"]["batch_norm"]["eps"],
                       momentum=CONFIG["preprocess"]["batch_norm"]["momentum"],
                       track_running_stats=CONFIG["preprocess"]["batch_norm"]["track_running_stats"])
    bn.load_state_dict(scaler_state)
    bn.eval()

    umap_model = joblib.load(trained_dir / "umap_model.joblib")
    hdbscan_model = joblib.load(trained_dir / "hdbscan_model.pkl")

    def predict(*features):
        """
        ``features`` is a tuple of 13 floats coming from the Gradio UI.
        """
        # 1) Tensor → BatchNorm scaling
        x = torch.tensor([features], dtype=torch.float32)  # shape (1,13)
        with torch.no_grad():
            x_scaled = bn(x)

        # 2) UMAP transform (expects numpy)
        x_emb = umap_model.transform(x_scaled.numpy())  # shape (1,2)

        # 3) HDBSCAN approximate prediction
        label = hdbscan_model.approximate_predict(x_emb)[0]
        return int(label)

    # Build Gradio interface
    feature_inputs = [
        gr.inputs.Slider(minimum=0, maximum=20, default=0, label=f"feature_{i+1}")
        for i in range(13)
    ]
    iface = gr.Interface(
        fn=predict,
        inputs=feature_inputs,
        outputs="label",
        title="Wine Clustering (HDBSCAN + UMAP)",
        description="Enter the 13 chemical measurements of a wine sample to obtain its cluster label "
        "(‑1 indicates noise).",
    )
    return iface


# ----------------------------------------------------------------------
# 8️⃣  Main orchestration
# ----------------------------------------------------------------------
def main():
    # 0️⃣ copy raw CSV
    raw_csv_path = step_0_copy_raw_csv()

    # 1️⃣ load data and persist raw tensors
    feats_tensor, target_tensor = step_1_load_and_save_dataset(raw_csv_path)

    # 2️⃣ scaling
    scaled_tensor = step_2_standard_scale(feats_tensor)

    # 3️⃣ UMAP embedding
    embedding_tensor, _ = step_3_umap_embedding(scaled_tensor)

    # 4️⃣ HDBSCAN clustering
    pred_labels, _ = step_4_hdbscan_clustering(embedding_tensor)

    # 5️⃣ Evaluation
    _ = step_5_evaluation(target_tensor, pred_labels, embedding_tensor)

    # 6️⃣ Save artefacts for deployment
    trained_dir = step_6_save_trained_model()

    # 7️⃣ Launch Gradio demo (optional – comment out if not needed)
    iface = build_gradio_interface(trained_dir)
    print("\n🚀 Starting Gradio demo – open the URL printed below in your browser.")
    iface.launch(share=False)  # set share=True if you want a public link

if __name__ == "__main__":
    main()
