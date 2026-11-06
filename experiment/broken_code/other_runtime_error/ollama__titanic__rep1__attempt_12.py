
"""
Titanic Tabular Classification – End‑to‑End Modeling Pipeline
============================================================

This script performs the complete workflow required for the Titanic
tabular classification task:

1️⃣ Load the raw CSV (expects it at ``./data/raw/titanic.csv``).  
2️⃣ Clean column names (strip whitespace, lower‑case).  
3️⃣ Split the data stratified on the target (70 % train, 20 % validation,
   10 % test).  
4️⃣ Build a scikit‑learn preprocessing pipeline:
   * Feature engineering (FamilySize, IsAlone)
   * Numeric → IterativeImputer → RobustScaler
   * Categorical → SimpleImputer → OneHotEncoder (dense)
5️⃣ Fit the pipeline on the training split only and transform all splits.  
6️⃣ Train three tree‑based base learners (LightGBM, CatBoost, XGBoost) with
   fixed hyper‑parameters.  
7️⃣ Train a custom PyTorch MLP (early‑stopping, no hyper‑parameter search).  
8️⃣ Fit a Logistic‑Regression meta‑learner that stacks the four base
   predictions (using the validation set).  
9️⃣ Evaluate the stacked ensemble on the held‑out test set (ROC‑AUC,
   accuracy, F1).  
🔟 Persist **all** artefacts under ``./agent_workspace/trained_models`` so
   they can be loaded later for inference or deployment.

Run this script from the repository root after installing the dependencies
listed in ``requirements.txt``.
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports & Global Settings
# ──────────────────────────────────────────────────────────────────────────────
import json
import os
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# ── scikit‑learn (enable experimental IterativeImputer) ────────────────────────
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.compose import ColumnTransformer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# ── tree‑based models ────────────────────────────────────────────────────────
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# ── reproducibility ────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ── paths ───────────────────────────────────────────────────────────────────
BASE_DIR = Path.cwd()
RAW_CSV = BASE_DIR / "data" / "raw" / "titanic.csv"
MODEL_DIR = BASE_DIR / "agent_workspace" / "trained_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# 1️⃣ Load data & basic clean‑up
# ──────────────────────────────────────────────────────────────────────────────
if not RAW_CSV.is_file():
    raise FileNotFoundError(f"Dataset not found at {RAW_CSV}")

df_raw = pd.read_csv(RAW_CSV)

# Clean column names: strip whitespace, replace spaces with underscores, lower‑case
df_raw.columns = (
    df_raw.columns.str.strip()
    .str.replace(" ", "_")
    .str.replace("-", "_")
    .str.lower()
)

# Expected target column name after cleaning
TARGET_COL = "survived"
if TARGET_COL not in df_raw.columns:
    raise KeyError(f"Target column '{TARGET_COL}' not found after cleaning columns.")

# Separate features / label
y = df_raw[TARGET_COL].astype(int).values
X = df_raw.drop(columns=[TARGET_COL])

# ──────────────────────────────────────────────────────────────────────────────
# 2️⃣ Train / Validation / Test split (stratified)
# ──────────────────────────────────────────────────────────────────────────────
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=SEED,
)

# 30 % → split into 20 % validation / 10 % test
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp,
    y_tmp,
    test_size=1 / 3,  # 1/3 of 30 % = 10 %
    stratify=y_tmp,
    random_state=SEED,
)

# ──────────────────────────────────────────────────────────────────────────────
# 3️⃣ Feature engineering & preprocessing pipeline
# ──────────────────────────────────────────────────────────────────────────────
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add FamilySize and IsAlone columns (if the required columns exist)."""
    df = df.copy()
    # Guard against missing columns – use .get with default 0
    sibsp = df.get("sibsp", pd.Series(0, index=df.index))
    parch = df.get("parch", pd.Series(0, index=df.index))
    df["familysize"] = sibsp + parch + 1
    df["isalone"] = (df["familysize"] == 1).astype(int)
    return df

# Identify numeric vs categorical columns after cleaning
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

# Numeric sub‑pipeline
numeric_pipe = Pipeline(
    steps=[
        (
            "imputer",
            IterativeImputer(
                estimator=LinearRegression(),
                max_iter=5,
                random_state=SEED,
            ),
        ),
        ("scaler", RobustScaler()),
    ]
)

# Categorical sub‑pipeline
categorical_pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
        ),
    ]
)

# Full preprocessing pipeline
preprocess = Pipeline(
    steps=[
        (
            "feature_engineering",
            FunctionTransformer(add_engineered_features, validate=False),
        ),
        (
            "column_transform",
            ColumnTransformer(
                transformers=[
                    ("num", numeric_pipe, numeric_cols),
                    ("cat", categorical_pipe, categorical_cols),
                ],
                remainder="passthrough",  # engineered columns stay as‑is
            ),
        ),
    ]
)

# Fit only on the training split to avoid leakage
preprocess.fit(X_train, y_train)

# Transform all splits
X_train_proc = preprocess.transform(X_train).astype(np.float32)
X_val_proc = preprocess.transform(X_val).astype(np.float32)
X_test_proc = preprocess.transform(X_test).astype(np.float32)

# Persist the preprocessing pipeline
joblib.dump(preprocess, MODEL_DIR / "preprocess.pkl")

# ──────────────────────────────────────────────────────────────────────────────
# 4️⃣ Train tree‑based base learners
# ──────────────────────────────────────────────────────────────────────────────
# LightGBM
lgbm = LGBMClassifier(
    random_state=SEED,
    n_estimators=200,
    learning_rate=0.05,
    max_depth=-1,
    num_leaves=31,
)
lgbm.fit(X_train_proc, y_train)

# CatBoost
cat = CatBoostClassifier(
    random_seed=SEED,
    iterations=400,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    verbose=False,
)
cat.fit(X_train_proc, y_train)

# XGBoost
xgb = XGBClassifier(
    random_state=SEED,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    verbosity=0,
)
xgb.fit(X_train_proc, y_train)

# Persist tree models
joblib.dump(lgbm, MODEL_DIR / "lightgbm.pkl")
cat.save_model(str(MODEL_DIR / "catboost.cbm"))
joblib.dump(xgb, MODEL_DIR / "xgboost.pkl")

# ──────────────────────────────────────────────────────────────────────────────
# 5️⃣ Custom PyTorch MLP (early‑stopping)
# ──────────────────────────────────────────────────────────────────────────────
class TorchMLP(nn.Module):
    """Two‑layer feed‑forward network for binary classification."""
    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),  # logits
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # shape (N,)


def train_torch_mlp(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    input_dim: int,
    hidden_dims: list[int] = [64, 32],
    lr: float = 2e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 32,
    max_epochs: int = 100,
    patience: int = 10,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TorchMLP(input_dim, hidden_dims).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_ds = TensorDataset(
        torch.from_numpy(X_tr), torch.from_numpy(y_tr.astype(np.float32))
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_va), torch.from_numpy(y_va.astype(np.float32))
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_auc = 0.0
    epochs_no_improve = 0
    best_state = None

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = model(xb)
                all_logits.append(logits.cpu())
                all_labels.append(yb)

        probs = torch.sigmoid(torch.cat(all_logits)).numpy()
        true = torch.cat(all_labels).numpy()
        val_auc = roc_auc_score(true, probs)

        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
            best_state = model.state_dict()
        else:
            epochs_no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"[MLP] Epoch {epoch:03d} – val ROC‑AUC: {val_auc:.4f}")

        if epochs_no_improve >= patience:
            print(f"[MLP] Early stopping at epoch {epoch} (best ROC‑AUC={best_auc:.4f})")
            break

    # Load best weights
    model.load_state_dict(best_state)

    # Save checkpoint
    ckpt = {
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "best_val_auc": best_auc,
    }
    ckpt_path = MODEL_DIR / "pytorch_mlp_best.ckpt"
    torch.save(ckpt, ckpt_path)

    return model, best_auc, ckpt_path


# Train the MLP
input_dim = X_train_proc.shape[1]
mlp_model, mlp_best_auc, mlp_ckpt_path = train_torch_mlp(
    X_train_proc,
    y_train,
    X_val_proc,
    y_val,
    input_dim=input_dim,
    hidden_dims=[64, 32],
    lr=2e-3,
    weight_decay=1e-4,
    batch_size=32,
    max_epochs=100,
    patience=10,
)

# ──────────────────────────────────────────────────────────────────────────────
# 6️⃣ Helper to obtain class‑1 probabilities from any base learner
# ──────────────────────────────────────────────────────────────────────────────
def get_proba(model, X: np.ndarray) -> np.ndarray:
    """Return probability of class 1 for supported base models."""
    if isinstance(model, (LGBMClassifier, XGBClassifier)):
        return model.predict_proba(X)[:, 1]
    if isinstance(model, CatBoostClassifier):
        return model.predict_proba(X)[:, 1]
    if isinstance(model, nn.Module):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(X).to(device)
            logits = model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs
    raise TypeError(f"Unsupported model type: {type(model)}")

# ──────────────────────────────────────────────────────────────────────────────
# 7️⃣ Train stacking meta‑learner (Logistic Regression) on validation predictions
# ──────────────────────────────────────────────────────────────────────────────
val_preds = np.column_stack(
    [
        get_proba(lgbm, X_val_proc),
        get_proba(cat, X_val_proc),
        get_proba(xgb, X_val_proc),
        get_proba(mlp_model, X_val_proc),
    ]
)

meta_learner = LogisticRegression(
    solver="lbfgs", max_iter=1000, random_state=SEED
)
meta_learner.fit(val_preds, y_val)

# Persist meta‑learner
joblib.dump(meta_learner, MODEL_DIR / "stacking_meta.pkl")

# ──────────────────────────────────────────────────────────────────────────────
# 8️⃣ Final evaluation on the test set
# ──────────────────────────────────────────────────────────────────────────────
test_preds = np.column_stack(
    [
        get_proba(lgbm, X_test_proc),
        get_proba(cat, X_test_proc),
        get_proba(xgb, X_test_proc),
        get_proba(mlp_model, X_test_proc),
    ]
)

final_proba = meta_learner.predict_proba(test_preds)[:, 1]
final_pred = (final_proba >= 0.5).astype(int)

roc_auc = roc_auc_score(y_test, final_proba)
accuracy = accuracy_score(y_test, final_pred)
f1 = f1_score(y_test, final_pred)

print("\n=== Stacked Ensemble Test Performance ===")
print(f"ROC‑AUC : {roc_auc:.4f}")
print(f"Accuracy: {accuracy:.4f}")
print(f"F1‑Score: {f1:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# 9️⃣ Save a single “deployable” object (convenient for inference)
# ──────────────────────────────────────────────────────────────────────────────
deployable = {
    "preprocess": preprocess,
    "lightgbm": lgbm,
    "catboost": cat,
    "xgboost": xgb,
    "torch_mlp_ckpt_path": str(mlp_ckpt_path),
    "meta_learner": meta_learner,
}
joblib.dump(deployable, MODEL_DIR / "stacked_ensemble_deployable.pkl")

print(f"\nAll artefacts have been saved under: {MODEL_DIR}")
