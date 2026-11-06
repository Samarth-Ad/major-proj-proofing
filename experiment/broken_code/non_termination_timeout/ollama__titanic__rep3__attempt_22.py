
# -*- coding: utf-8 -*-
"""
End‑to‑end tabular classification pipeline (Adult Income) using PyTorch and Gradio.

Features
--------
1. Loads the “Adult” dataset from OpenML.
2. Preprocesses categorical & numeric columns with a ColumnTransformer
   (One‑Hot + StandardScaler) and saves the transformer for later inference.
3. Trains a small feed‑forward network with early stopping.
4. Evaluates accuracy & F1 on a held‑out test set and reports model size &
   average inference time.
5. Converts the trained model to TorchScript for fast, portable inference.
6. Deploys a Gradio demo (compatible with the latest Gradio API).

All artefacts (preprocessor, checkpoint, TorchScript file) are stored under
`./agent_workspace/trained_models`.
"""

import os
import random
import time
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import gradio as gr
import joblib  # for saving/loading the ColumnTransformer

# --------------------------------------------------------------------------- #
# 1️⃣  GLOBAL SETTINGS & REPRODUCIBILITY
# --------------------------------------------------------------------------- #
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Directory for all artefacts
ARTIFACT_DIR = Path("./agent_workspace/trained_models")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# 2️⃣  DATA RETRIEVAL
# --------------------------------------------------------------------------- #
def load_dataset() -> pd.DataFrame:
    """Download the Adult Income dataset from OpenML and create a binary target."""
    print("Downloading Adult dataset from OpenML …")
    adult = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
    df = adult.frame.copy()
    # Create binary target: 1 if >50K else 0
    df["target"] = (df["class"] == ">50K").astype(int)
    df = df.drop(columns=["class"])
    return df

# --------------------------------------------------------------------------- #
# 3️⃣  PREPROCESSING & SPLITTING
# --------------------------------------------------------------------------- #
def preprocess_data(df: pd.DataFrame):
    """Split data, fit ColumnTransformer, and return torch tensors."""
    X = df.drop(columns=["target"])
    y = df["target"]

    # Train / validation / test split (70/20/10)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.10, random_state=SEED, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=0.2222, random_state=SEED, stratify=y_temp
    )  # ≈20% of original data

    # Identify column types
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    # ColumnTransformer: one‑hot for categoricals, standard scaling for numerics
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", StandardScaler(), numeric_cols),
        ]
    )

    # Fit on training data only
    X_train_enc = preprocessor.fit_transform(X_train)
    X_valid_enc = preprocessor.transform(X_valid)
    X_test_enc = preprocessor.transform(X_test)

    # Convert sparse matrices to dense torch tensors (float32)
    X_train_tensor = torch.tensor(X_train_enc.todense(), dtype=torch.float32)
    X_valid_tensor = torch.tensor(X_valid_enc.todense(), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_enc.todense(), dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_valid_tensor = torch.tensor(y_valid.values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # Save the preprocessor for inference
    preproc_path = ARTIFACT_DIR / "preprocessor.pkl"
    joblib.dump(preprocessor, preproc_path)
    print(f"Preprocessor saved to {preproc_path}")

    return (
        (X_train_tensor, y_train_tensor),
        (X_valid_tensor, y_valid_tensor),
        (X_test_tensor, y_test_tensor),
        preprocessor,
    )

# --------------------------------------------------------------------------- #
# 4️⃣  TORCH DATASET
# --------------------------------------------------------------------------- #
class TabularDataset(Dataset):
    """Simple torch Dataset for tabular data."""

    def __init__(self, X_tensor: torch.Tensor, y_tensor: torch.Tensor):
        self.X = X_tensor
        self.y = y_tensor

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]

# --------------------------------------------------------------------------- #
# 5️⃣  MODEL DEFINITION
# --------------------------------------------------------------------------- #
class FeedForwardNN(nn.Module):
    """Small fully‑connected network for binary classification."""

    def __init__(self, input_dim: int, hidden_dims=(128, 64), dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 2))  # two logits → binary
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# --------------------------------------------------------------------------- #
# 6️⃣  TRAINING LOOP
# --------------------------------------------------------------------------- #
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 5,
) -> nn.Module:
    """Train `model` with early stopping on validation accuracy."""
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc = 0.0
    epochs_no_improve = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        # ---------- Validation ----------
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                logits = model(xb)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())
                all_labels.append(yb.cpu())

        y_true = torch.cat(all_labels).numpy()
        y_pred = torch.cat(all_preds).numpy()
        val_acc = accuracy_score(y_true, y_pred)

        print(
            f"Epoch {epoch:02d} | Train loss: {epoch_loss/len(train_loader.dataset):.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # Early stopping logic
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    # Load best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save checkpoint
    ckpt_path = ARTIFACT_DIR / "best_model.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Best model checkpoint saved to {ckpt_path}")

    return model

# --------------------------------------------------------------------------- #
# 7️⃣  EVALUATION
# --------------------------------------------------------------------------- #
def evaluate_model(model: nn.Module, test_loader: DataLoader):
    """Return accuracy, F1 and simple complexity metrics."""
    model.eval()
    all_preds, all_labels = [], []
    inference_times = []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(DEVICE)
            start = time.time()
            logits = model(xb)
            end = time.time()
            inference_times.append(end - start)

            preds = torch.argmax(logits, dim=1).cpu()
            all_preds.append(preds)
            all_labels.append(yb)

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    perf = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, average="binary"),
    }

    avg_inf_ms = (np.mean(inference_times) / xb.shape[0]) * 1000  # per sample
    model_path = ARTIFACT_DIR / "best_model.pt"
    model_size_mb = model_path.stat().st_size / (1024 * 1024)

    comp = {
        "AvgInferenceTime_ms": avg_inf_ms,
        "ModelSize_MB": model_size_mb,
    }

    return perf, comp

# --------------------------------------------------------------------------- #
# 8️⃣  PREPARE TORCHSCRIPT FOR DEPLOYMENT
# --------------------------------------------------------------------------- #
def prepare_model_for_deployment(model: nn.Module, example_input: torch.Tensor) -> Path:
    """Trace the model with a single example and save the TorchScript file."""
    model.eval()
    scripted = torch.jit.trace(model, example_input.to(DEVICE))
    scripted_path = ARTIFACT_DIR / "model_scripted.pt"
    scripted.save(str(scripted_path))
    print(f"TorchScript model saved to {scripted_path}")
    return scripted_path

# --------------------------------------------------------------------------- #
# 9️⃣  GRADIO INFERENCE FUNCTION
# --------------------------------------------------------------------------- #
def load_preprocessor() -> ColumnTransformer:
    """Load the saved ColumnTransformer."""
    preproc_path = ARTIFACT_DIR / "preprocessor.pkl"
    return joblib.load(preproc_path)

def inference_fn(*inputs):
    """
    Gradio inference function.
    `inputs` are raw feature values (strings) in the order of the original columns.
    """
    # Lazy loading of artefacts
    if not hasattr(inference_fn, "model"):
        inference_fn.model = torch.jit.load(str(ARTIFACT_DIR / "model_scripted.pt"))
        inference_fn.model.to(DEVICE)
        inference_fn.preprocessor = load_preprocessor()
        # Original column order before transformation
        inference_fn.feature_names = inference_fn.preprocessor.feature_names_in_.tolist()

    # Build a one‑row DataFrame from the raw inputs
    data = dict(zip(inference_fn.feature_names, inputs))
    df = pd.DataFrame([data])

    # Apply same preprocessing pipeline
    X_enc = inference_fn.preprocessor.transform(df)
    X_tensor = torch.tensor(X_enc.todense(), dtype=torch.float32).to(DEVICE)

    # Predict
    with torch.no_grad():
        logits = inference_fn.model(X_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_label = int(np.argmax(probs))
        confidence = float(np.max(probs))

    label_str = ">50K" if pred_label == 1 else "<=50K"
    return label_str, f"{confidence:.4f}"

# --------------------------------------------------------------------------- #
# 🔟  GRADIO DEMO
# --------------------------------------------------------------------------- #
def launch_gradio_demo():
    """Create and launch the Gradio interface."""
    preprocessor = load_preprocessor()
    feature_names = preprocessor.feature_names_in_.tolist()

    # Use a Textbox for every feature (simple but works for all types)
    inputs = [gr.Textbox(label=name) for name in feature_names]
    outputs = [
        gr.Label(num_top_classes=2, label="Prediction"),
        gr.Textbox(label="Confidence (0‑1)"),
    ]

    iface = gr.Interface(
        fn=inference_fn,
        inputs=inputs,
        outputs=outputs,
        title="Adult Income Prediction",
        description="Predict whether a person makes >50K USD per year.",
        # `allow_flagging` was removed in newer Gradio versions; omit it.
    )
    # `share=False` runs locally; set to True if you need a public link.
    iface.launch(share=False, inline=False)

# --------------------------------------------------------------------------- #
# 📌  MAIN PIPELINE
# --------------------------------------------------------------------------- #
def main():
    # 1️⃣ Load raw data
    df = load_dataset()

    # 2️⃣ Preprocess & split
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test), _ = preprocess_data(df)

    # 3️⃣ Datasets & DataLoaders
    train_ds = TabularDataset(X_train, y_train)
    valid_ds = TabularDataset(X_valid, y_valid)
    test_ds = TabularDataset(X_test, y_test)

    batch_size = 256
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 4️⃣ Model instantiation
    model = FeedForwardNN(input_dim=X_train.shape[1]).to(DEVICE)

    # 5️⃣ Training
    model = train_model(
        model,
        train_loader,
        valid_loader,
        epochs=30,
        lr=5e-4,
        weight_decay=1e-5,
        patience=6,
    )

    # 6️⃣ Evaluation
    perf, comp = evaluate_model(model, test_loader)
    print("\n=== Test Performance ===")
    print(json.dumps(perf, indent=2))
    print("\n=== Model Complexity ===")
    print(json.dumps(comp, indent=2))

    # 7️⃣ TorchScript conversion
    example_input = X_train[:1]  # single sample for tracing
    _ = prepare_model_for_deployment(model, example_input)

    # 8️⃣ Launch Gradio demo
    launch_gradio_demo()

    # Return a summary dictionary (useful if the script is imported)
    return {
        "trained_checkpoint": str(ARTIFACT_DIR / "best_model.pt"),
        "torchscript_model": str(ARTIFACT_DIR / "model_scripted.pt"),
        "preprocessor": str(ARTIFACT_DIR / "preprocessor.pkl"),
        "performance": perf,
        "complexity": comp,
    }

if __name__ == "__main__":
    results = main()
