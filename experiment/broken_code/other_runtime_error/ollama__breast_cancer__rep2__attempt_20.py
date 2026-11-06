
# -*- coding: utf-8 -*-
"""
End‑to‑end pipeline for a tabular classification task.

Features
--------
1. Load a public dataset (UCI Adult Income) via `sklearn.datasets.fetch_openml`.
2. Preprocess: missing‑value handling, categorical encoding (One‑Hot), numeric scaling.
3. Split into train / validation / test (70 % / 20 % / 10 %).
4. PyTorch `TabularNN` model (simple feed‑forward network).
5. Training loop with early‑stopping and a tiny grid‑search over learning‑rate & hidden size.
6. Evaluation (accuracy, macro‑F1) + inference‑time & model‑size reporting.
7. Model compression (dynamic quantisation) and saving to ``./agent_workspace/trained_models``.
8. Gradio demo for interactive inference.

The script is self‑contained – just run it with a Python 3 interpreter
(and internet access for the dataset download).  All required packages are
installed automatically if missing.
"""

import os
import sys
import time
import json
import random
import warnings
from pathlib import Path

# --------------------------------------------------------------------------- #
# Install missing dependencies on‑the‑fly (only executed the first time)
# --------------------------------------------------------------------------- #
def _install(package: str):
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

required = [
    "torch",
    "scikit-learn",
    "pandas",
    "numpy",
    "gradio",
    "joblib",
    "openml",
]
for pkg in required:
    try:
        __import__(pkg.split("=")[0].replace("-", "_"))
    except Exception:  # pragma: no cover
        _install(pkg)

# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_openml

import joblib
import gradio as gr

# --------------------------------------------------------------------------- #
# Global settings
# --------------------------------------------------------------------------- #
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Directory for saving artefacts
BASE_DIR = Path("./agent_workspace")
MODEL_DIR = BASE_DIR / "trained_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# 1️⃣ Dataset retrieval
# --------------------------------------------------------------------------- #
def load_dataset() -> pd.DataFrame:
    """
    Load the UCI Adult Income dataset (binary classification >50K).
    Returns a pandas DataFrame with the target column named ``target``.
    """
    print("Downloading Adult dataset from OpenML …")
    adult = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
    df = adult.frame.copy()
    # Rename target column for clarity
    df = df.rename(columns={"class": "target"})
    # Strip whitespace in string columns (OpenML artefacts)
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    return df


# --------------------------------------------------------------------------- #
# 2️⃣ Pre‑processing utilities
# --------------------------------------------------------------------------- #
def build_preprocess_pipeline(df: pd.DataFrame) -> Pipeline:
    """
    Construct a sklearn Pipeline that:
      * One‑Hot encodes categorical columns (handle_unknown='ignore')
      * Standard‑scales numeric columns
    The pipeline returns a NumPy array ready for PyTorch.
    """
    # Identify column types
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    categorical_cols = [c for c in categorical_cols if c != "target"]
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "target"]

    # Transformers
    cat_transformer = OneHotEncoder(handle_unknown="ignore", sparse=False)
    num_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", cat_transformer, categorical_cols),
            ("num", num_transformer, numeric_cols),
        ]
    )

    pipeline = Pipeline(steps=[("preprocess", preprocessor)])
    return pipeline


def preprocess_data(df: pd.DataFrame, pipeline: Pipeline = None):
    """
    Fit (if needed) and apply the preprocessing pipeline.
    Returns:
        X (np.ndarray) – feature matrix
        y (np.ndarray) – integer encoded labels (0/1)
        fitted pipeline (for later use in inference)
    """
    if pipeline is None:
        pipeline = build_preprocess_pipeline(df)

    X = pipeline.fit_transform(df.drop(columns=["target"]))
    # Encode target: '>50K' -> 1, '<=50K' -> 0
    y = (df["target"] == ">50K").astype(int).values
    return X.astype(np.float32), y.astype(np.int64), pipeline


# --------------------------------------------------------------------------- #
# 3️⃣ PyTorch Dataset & DataLoader
# --------------------------------------------------------------------------- #
class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 256,
    valid_size: float = 0.2,
    test_size: float = 0.1,
):
    """
    Split the data into train / valid / test and return DataLoaders.
    """
    # First split off test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )
    # Then split remaining into train / valid
    valid_ratio = valid_size / (1 - test_size)  # re‑scale
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp,
        y_temp,
        test_size=valid_ratio,
        random_state=SEED,
        stratify=y_temp,
    )

    train_loader = DataLoader(
        TabularDataset(X_train, y_train), batch_size=batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        TabularDataset(X_valid, y_valid), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TabularDataset(X_test, y_test), batch_size=batch_size, shuffle=False
    )
    return train_loader, valid_loader, test_loader


# --------------------------------------------------------------------------- #
# 4️⃣ Model definition
# --------------------------------------------------------------------------- #
class TabularNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # shape (batch,)


# --------------------------------------------------------------------------- #
# 5️⃣ Training utilities (including tiny hyper‑parameter search)
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer,
    device: torch.device,
):
    model.train()
    epoch_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device).float()
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * xb.size(0)
    return epoch_loss / len(loader.dataset)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = torch.sigmoid(logits) > 0.5
            all_preds.append(preds.cpu().numpy())
            all_labels.append(yb.numpy())
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    return acc, f1


def train_model(
    train_loader: DataLoader,
    valid_loader: DataLoader,
    input_dim: int,
    hyperparams: dict,
    device: torch.device,
    max_epochs: int = 30,
    early_stop_patience: int = 5,
):
    """
    Train the model with early stopping.
    Returns the best model (state_dict) and the validation metrics of the best epoch.
    """
    model = TabularNN(
        input_dim=input_dim,
        hidden_dim=hyperparams["hidden_dim"],
        dropout=hyperparams["dropout"],
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=hyperparams["lr"], weight_decay=hyperparams["weight_decay"]
    )

    best_val_f1 = -np.inf
    best_state = None
    epochs_no_improve = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_f1 = evaluate(model, valid_loader, device)

        print(
            f"Epoch {epoch:02d} | Train loss: {train_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}"
        )

        # Early‑stopping check
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stop_patience:
                print("Early stopping triggered.")
                break

    # Load best weights
    model.load_state_dict(best_state)
    return model, {"val_acc": val_acc, "val_f1": best_val_f1}


def hyperparameter_search(train_loader, valid_loader, input_dim, device):
    """
    Very small grid‑search over a few sensible values.
    Returns the best hyper‑parameter dict.
    """
    grid = {
        "lr": [1e-3, 5e-4],
        "hidden_dim": [128, 256],
        "dropout": [0.2, 0.3],
        "weight_decay": [0.0, 1e-4],
    }

    best_cfg = None
    best_f1 = -np.inf

    # Cartesian product
    import itertools

    for lr, hidden_dim, dropout, weight_decay in itertools.product(
        grid["lr"], grid["hidden_dim"], grid["dropout"], grid["weight_decay"]
    ):
        cfg = {
            "lr": lr,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "weight_decay": weight_decay,
        }
        print(f"\nTesting config: {cfg}")
        model, metrics = train_model(
            train_loader,
            valid_loader,
            input_dim,
            cfg,
            device,
            max_epochs=20,
            early_stop_patience=3,
        )
        if metrics["val_f1"] > best_f1:
            best_f1 = metrics["val_f1"]
            best_cfg = cfg
            # Save temporary best model for later reuse
            torch.save(model.state_dict(), MODEL_DIR / "tmp_best.pt")
    # Load the best temporary model back into memory
    best_model = TabularNN(input_dim, hidden_dim=best_cfg["hidden_dim"], dropout=best_cfg["dropout"]).to(device)
    best_model.load_state_dict(torch.load(MODEL_DIR / "tmp_best.pt"))
    os.remove(MODEL_DIR / "tmp_best.pt")
    print(f"\nBest hyper‑parameters: {best_cfg} with Val F1={best_f1:.4f}")
    return best_model, best_cfg


# --------------------------------------------------------------------------- #
# 6️⃣ Evaluation (performance + complexity)
# --------------------------------------------------------------------------- #
def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device):
    """
    Returns:
        performance_scores – dict with ACC and F1
        complexity_scores – dict with inference_time (ms per sample) and model_size (MB)
    """
    # ---- Performance ----
    acc, f1 = evaluate(model, test_loader, device)
    performance_scores = {"ACC": acc, "F1": f1}

    # ---- Complexity ----
    model.eval()
    total_time = 0.0
    n_samples = 0
    with torch.no_grad():
        for xb, _ in test_loader:
            xb = xb.to(device)
            start = time.time()
            _ = model(xb)
            torch.cuda.synchronize() if device.type == "cuda" else None
            total_time += (time.time() - start)
            n_samples += xb.size(0)

    inference_time_ms = (total_time / n_samples) * 1000.0
    model_path = MODEL_DIR / "model_float.pt"
    torch.save(model.state_dict(), model_path)
    model_size_mb = model_path.stat().st_size / (1024 * 1024)

    complexity_scores = {
        "inference_time_ms_per_sample": inference_time_ms,
        "model_size_mb": model_size_mb,
    }

    return performance_scores, complexity_scores


# --------------------------------------------------------------------------- #
# 7️⃣ Model compression & saving for deployment
# --------------------------------------------------------------------------- #
def prepare_model_for_deployment(model: nn.Module):
    """
    Apply dynamic quantisation (CPU‑only friendly) and save the quantised model.
    Returns the path to the deployable artefact.
    """
    model_cpu = model.to("cpu")
    quantised_model = torch.quantization.quantize_dynamic(
        model_cpu, {nn.Linear}, dtype=torch.qint8
    )
    deploy_path = MODEL_DIR / "model_quantised.pt"
    torch.save(quantised_model.state_dict(), deploy_path)
    print(f"Quantised model saved to {deploy_path}")
    return deploy_path


# --------------------------------------------------------------------------- #
# 8️⃣ Gradio demo
# --------------------------------------------------------------------------- #
def build_gradio_interface(pipeline: Pipeline, model_path: Path, input_dim: int):
    """
    Build a Gradio interface that accepts a single CSV row (or a JSON dict)
    and returns the predicted class (0/1) with probability.
    """
    # Load quantised model
    quantised_model = TabularNN(input_dim)
    quantised_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    quantised_model.eval()

    def predict(**kwargs):
        """
        kwargs: feature_name=value (strings)
        """
        # Convert dict to DataFrame (single row)
        df = pd.DataFrame([kwargs])
        # Apply same preprocessing pipeline
        X = pipeline.transform(df).astype(np.float32)
        X_tensor = torch.from_numpy(X)
        with torch.no_grad():
            logit = quantised_model(X_tensor).item()
            prob = 1 / (1 + np.exp(-logit))
            pred = int(prob > 0.5)
        return {"prediction": pred, "probability": float(prob)}

    # Build input components dynamically based on pipeline's expected columns
    # (We reuse the column list from the original dataset)
    # For simplicity we expose all original features (excluding target)
    example_df = load_dataset()
    feature_names = [c for c in example_df.columns if c != "target"]
    inputs = [gr.Textbox(label=name) for name in feature_names]

    iface = gr.Interface(
        fn=predict,
        inputs=inputs,
        outputs=[
            gr.Label(num_top_classes=2, label="Predicted class (0/1)"),
            gr.Number(label="Probability of class 1"),
        ],
        title="Adult Income Prediction (Tabular Classification)",
        description="Enter the attributes of a person and the model will predict whether the income exceeds $50K per year.",
        examples=[
            # Provide a single example row (values taken from the dataset)
            [
                "39",  # age
                "State-gov",  # workclass
                "77516",  # fnlwgt
                "Bachelors",  # education
                "13",  # education-num
                "Never-married",  # marital-status
                "Adm-clerical",  # occupation
                "Not-in-family",  # relationship
                "White",  # race
                "Male",  # sex
                "2174",  # capital-gain
                "0",  # capital-loss
                "40",  # hours-per-week
                "United-States",  # native-country
            ]
        ],
        allow_flagging="never",
        theme="default",
    )
    return iface


# --------------------------------------------------------------------------- #
# Main orchestration
# --------------------------------------------------------------------------- #
def main():
    # 1️⃣ Load raw data
    raw_df = load_dataset()

    # 2️⃣ Preprocess (fit pipeline on the whole data – later we will split)
    X, y, preprocess_pipe = preprocess_data(raw_df)

    # 3️⃣ Create DataLoaders
    train_loader, valid_loader, test_loader = create_dataloaders(X, y)

    # 4️⃣ Hyper‑parameter search & final training
    input_dim = X.shape[1]
    best_model, best_cfg = hyperparameter_search(
        train_loader, valid_loader, input_dim, DEVICE
    )

    # 5️⃣ Final evaluation on test set
    performance, complexity = evaluate_model(best_model, test_loader, DEVICE)
    print("\n=== Test set performance ===")
    print(json.dumps(performance, indent=2))
    print("\n=== Model complexity ===")
    print(json.dumps(complexity, indent=2))

    # 6️⃣ Prepare model for deployment (quantisation + saving)
    deployable_path = prepare_model_for_deployment(best_model)

    # 7️⃣ Build & launch Gradio demo
    iface = build_gradio_interface(preprocess_pipe, deployable_path, input_dim)
    # Launch in a blocking manner; change `share=True` if you need a public URL.
    iface.launch(share=False, server_name="0.0.0.0", server_port=7860)

    # Return artefacts for possible downstream usage
    return {
        "preprocess_pipeline": preprocess_pipe,
        "trained_model": best_model,
        "deployable_model_path": str(deployable_path),
        "performance": performance,
        "complexity": complexity,
        "best_hyperparameters": best_cfg,
    }


if __name__ == "__main__":
    artefacts = main()
