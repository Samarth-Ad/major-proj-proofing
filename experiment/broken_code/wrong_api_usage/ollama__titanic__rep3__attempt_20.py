
# -*- coding: utf-8 -*-
"""
Tabular classification pipeline (data loading → preprocessing → training → evaluation →
deployment) using PyTorch and Gradio.

The script is self‑contained and can be executed directly. It downloads the
“Adult” dataset from OpenML, builds a simple feed‑forward network, trains it,
evaluates accuracy/F1, compresses the model with TorchScript and finally
exposes a Gradio demo.

All artefacts (model checkpoints, TorchScript file) are saved under
./agent_workspace/trained_models .
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
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

import gradio as gr

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
def load_dataset():
    """
    Loads the Adult Income dataset (binary classification) from OpenML.
    Returns a pandas DataFrame with features and a Series with the target.
    """
    print("Downloading Adult dataset from OpenML …")
    adult = fetch_openml(name="adult", version=2, as_frame=True, parser="auto")
    df = adult.frame
    # The target column is named "class"
    df["target"] = (df["class"] == ">50K").astype(int)
    df.drop(columns=["class"], inplace=True)
    return df

# --------------------------------------------------------------------------- #
# 3️⃣  PREPROCESSING & FEATURE ENGINEERING
# --------------------------------------------------------------------------- #
def preprocess_data(df):
    """
    Splits the data, fits encoders/scalers, and returns ready‑to‑use tensors.
    Also returns the fitted ColumnTransformer and label encoder for later use.
    """
    # Separate features / target
    X = df.drop(columns=["target"])
    y = df["target"]

    # Train / validation / test split (70/20/10)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.10, random_state=SEED, stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=0.2222, random_state=SEED, stratify=y_temp
    )  # 0.2222 ≈ 20% of original data

    # Identify column types
    categorical_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
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

    # Convert to torch tensors (float32)
    X_train_tensor = torch.tensor(X_train_enc.todense(), dtype=torch.float32)
    X_valid_tensor = torch.tensor(X_valid_enc.todense(), dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_enc.todense(), dtype=torch.float32)

    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_valid_tensor = torch.tensor(y_valid.values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # Save preprocessing artefacts for inference
    preprocessing_path = ARTIFACT_DIR / "preprocessor.pkl"
    pd.to_pickle(preprocessor, preprocessing_path)

    return (
        (X_train_tensor, y_train_tensor),
        (X_valid_tensor, y_valid_tensor),
        (X_test_tensor, y_test_tensor),
        preprocessor,
    )


class TabularDataset(Dataset):
    """Simple torch Dataset for tabular data."""

    def __init__(self, X_tensor, y_tensor):
        self.X = X_tensor
        self.y = y_tensor

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# --------------------------------------------------------------------------- #
# 4️⃣  MODEL DEFINITION
# --------------------------------------------------------------------------- #
class FeedForwardNN(nn.Module):
    """A small fully‑connected network for binary classification."""

    def __init__(self, input_dim, hidden_dims=[128, 64], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 2))  # two logits → binary
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# --------------------------------------------------------------------------- #
# 5️⃣  TRAINING LOOP (with simple hyper‑parameter config)
# --------------------------------------------------------------------------- #
def train_model(
    model,
    train_loader,
    valid_loader,
    epochs=20,
    lr=1e-3,
    weight_decay=1e-5,
    patience=5,
):
    """
    Trains `model` and returns the best checkpoint (by validation accuracy).
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_acc = 0.0
    epochs_no_improve = 0
    best_state_dict = None

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

        # Validation
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

        # Early stopping check
        if val_acc > best_acc:
            best_acc = val_acc
            best_state_dict = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

    # Load best weights
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # Save checkpoint
    ckpt_path = ARTIFACT_DIR / "best_model.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"Best model saved to {ckpt_path}")

    return model


# --------------------------------------------------------------------------- #
# 6️⃣  EVALUATION (metrics + complexity)
# --------------------------------------------------------------------------- #
def evaluate_model(model, test_loader):
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

    performance_scores = {
        "ACC": accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, average="binary"),
    }

    # Complexity: average inference time per sample & model file size
    avg_inf_time_ms = (np.mean(inference_times) / xb.shape[0]) * 1000
    model_path = ARTIFACT_DIR / "best_model.pt"
    model_size_mb = model_path.stat().st_size / (1024 * 1024)

    complexity_scores = {
        "AvgInferenceTime_ms": avg_inf_time_ms,
        "ModelSize_MB": model_size_mb,
    }

    return performance_scores, complexity_scores


# --------------------------------------------------------------------------- #
# 7️⃣  PREPARATION FOR DEPLOYMENT (TorchScript)
# --------------------------------------------------------------------------- #
def prepare_model_for_deployment(model, example_input):
    """
    Converts the trained PyTorch model to TorchScript (scripted) for fast inference.
    Returns the path to the scripted model file.
    """
    model.eval()
    scripted_model = torch.jit.trace(model, example_input.to(DEVICE))
    scripted_path = ARTIFACT_DIR / "model_scripted.pt"
    scripted_model.save(str(scripted_path))
    print(f"Scripted model saved to {scripted_path}")
    return scripted_path


# --------------------------------------------------------------------------- #
# 8️⃣  GRADIO DEPLOYMENT
# --------------------------------------------------------------------------- #
def load_preprocessor():
    """Loads the ColumnTransformer saved during preprocessing."""
    preproc_path = ARTIFACT_DIR / "preprocessor.pkl"
    return pd.read_pickle(preproc_path)


def inference_fn(*inputs):
    """
    Gradio inference function.
    `inputs` are raw feature values in the order returned by `feature_names`.
    """
    # Load artefacts lazily (first call)
    if not hasattr(inference_fn, "model"):
        inference_fn.model = torch.jit.load(str(ARTIFACT_DIR / "model_scripted.pt"))
        inference_fn.model.to(DEVICE)
        inference_fn.preprocessor = load_preprocessor()
        inference_fn.feature_names = inference_fn.preprocessor.feature_names_in_.tolist()

    # Build a single‑row DataFrame
    data_dict = dict(zip(inference_fn.feature_names, inputs))
    df = pd.DataFrame([data_dict])

    # Transform with the same pipeline used during training
    X_enc = inference_fn.preprocessor.transform(df)
    X_tensor = torch.tensor(X_enc.todense(), dtype=torch.float32).to(DEVICE)

    # Predict
    with torch.no_grad():
        logits = inference_fn.model(X_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_label = int(np.argmax(probs))
        confidence = float(np.max(probs))

    # Human‑readable label (0 → "<=50K", 1 → ">50K")
    label_str = ">50K" if pred_label == 1 else "<=50K"
    return {"label": label_str, "confidence": confidence}


def deploy_model():
    """
    Builds and launches a Gradio demo.
    Returns the public URL (if launched with share=True) or local URL.
    """
    preprocessor = load_preprocessor()
    feature_names = preprocessor.feature_names_in_.tolist()
    # Determine input types (categorical → dropdown, numeric → slider)
    # For simplicity we treat everything as text boxes.
    inputs = [gr.inputs.Textbox(label=name) for name in feature_names]
    outputs = [
        gr.outputs.Label(num_top_classes=2, label="Prediction"),
        gr.outputs.Textbox(label="Confidence (0‑1)"),
    ]

    iface = gr.Interface(
        fn=inference_fn,
        inputs=inputs,
        outputs=outputs,
        title="Adult Income Prediction",
        description="Predict whether a person makes >50K USD per year.",
        allow_flagging="never",
    )
    # Launch with share=True for a public link (useful in notebooks)
    url = iface.launch(share=False, inline=False)
    return url


# --------------------------------------------------------------------------- #
# 9️⃣  MAIN PIPELINE
# --------------------------------------------------------------------------- #
def main():
    # 1️⃣ Load raw dataset
    df = load_dataset()

    # 2️⃣ Preprocess & split
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test), preprocessor = preprocess_data(df)

    # 3️⃣ Create torch Datasets / DataLoaders
    train_dataset = TabularDataset(X_train, y_train)
    valid_dataset = TabularDataset(X_valid, y_valid)
    test_dataset = TabularDataset(X_test, y_test)

    batch_size = 256
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # 4️⃣ Model instantiation
    input_dim = X_train.shape[1]
    model = FeedForwardNN(input_dim).to(DEVICE)

    # 5️⃣ Train (hyper‑parameters can be tweaked here)
    model = train_model(
        model,
        train_loader,
        valid_loader,
        epochs=30,
        lr=5e-4,
        weight_decay=1e-5,
        patience=6,
    )

    # 6️⃣ Evaluate
    perf, comp = evaluate_model(model, test_loader)
    print("\n=== Test Performance ===")
    print(json.dumps(perf, indent=2))
    print("\n=== Model Complexity ===")
    print(json.dumps(comp, indent=2))

    # 7️⃣ Prepare for deployment (TorchScript)
    example_input = X_train[:1]  # a single sample for tracing
    scripted_path = prepare_model_for_deployment(model, example_input)

    # 8️⃣ Deploy with Gradio
    url = deploy_model()
    print(f"\nGradio demo launched at: {url}")

    return {
        "preprocessed_data": "saved in ./agent_workspace/trained_models",
        "trained_model_path": str(ARTIFACT_DIR / "best_model.pt"),
        "scripted_model_path": str(scripted_path),
        "gradio_url": url,
        "performance": perf,
        "complexity": comp,
    }


if __name__ == "__main__":
    results = main()
