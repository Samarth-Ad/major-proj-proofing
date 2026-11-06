
"""
Titanic Tabular Classification – End‑to‑End Modeling Pipeline
================================================================
This script implements the full workflow required for the task:

1️⃣ Load the raw Titanic CSV (expects it at ``./data/raw/titanic.csv``).  
2️⃣ Split the data stratified on the target (70 % train, 20 % validation, 10 % test).  
3️⃣ Build a scikit‑learn preprocessing pipeline (numeric imputation → robust scaling,
   categorical imputation → one‑hot encoding, plus two engineered features).  
4️⃣ Train four base learners:
   • LightGBM
   • CatBoost
   • XGBoost
   • Custom PyTorch MLP (trained with early‑stopping)
5️⃣ Fit a Logistic‑Regression meta‑learner that stacks the four base predictions.  
6️⃣ Evaluate the stacked model on the held‑out test set (ROC‑AUC, accuracy, F1).  
7️⃣ Persist **all** artefacts to ``./agent_workspace/trained_models`` so they can be
   loaded later for inference or deployment.

The script is self‑contained – just run it from the repository root after installing
the dependencies listed in ``requirements.txt`` (see the long specification that
accompanies this code).

Author: MLOps Engineer – AutoML solution
"""

import os
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, FunctionTransformer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
)

# ──────────────────────────────────────────────────────────────────────────────
# 1️⃣  GLOBAL SETTINGS & PATHS
# ──────────────────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR = Path.cwd()
RAW_CSV = BASE_DIR / "data" / "raw" / "titanic.csv"
MODEL_DIR = BASE_DIR / "agent_workspace" / "trained_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "Survived"          # column name in the original CSV
TARGET_COL_LOWER = "survived"    # we will rename it to lower‑case for convenience

# ──────────────────────────────────────────────────────────────────────────────
# 2️⃣  DATA LOADING & BASIC CLEAN‑UP
# ──────────────────────────────────────────────────────────────────────────────
if not RAW_CSV.is_file():
    raise FileNotFoundError(f"Dataset not found at {RAW_CSV}")

df = pd.read_csv(RAW_CSV)

# Ensure the target column exists (case‑insensitive handling)
if TARGET_COL not in df.columns:
    # try to find a case‑insensitive match
    cols_lower = {c.lower(): c for c in df.columns}
    if TARGET_COL.lower() in cols_lower:
        df = df.rename(columns={cols_lower[TARGET_COL.lower()]: TARGET_COL})
    else:
        raise KeyError(f"Target column '{TARGET_COL}' not found in CSV.")

# Lower‑case the target for downstream consistency
df = df.rename(columns={TARGET_COL: TARGET_COL_LOWER})

# Separate features / label
y = df[TARGET_COL_LOWER].values
X = df.drop(columns=[TARGET_COL_LOWER])

# ──────────────────────────────────────────────────────────────────────────────
# 3️⃣  TRAIN / VALID / TEST SPLIT (stratified)
# ──────────────────────────────────────────────────────────────────────────────
X_train, X_tmp, y_train, y_tmp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=SEED,
)

# 30 % of the data → split into 20 % val / 10 % test
X_val, X_test, y_val, y_test = train_test_split(
    X_tmp,
    y_tmp,
    test_size=1 / 3,          # 1/3 of 30 % = 10 %
    stratify=y_tmp,
    random_state=SEED,
)

# ──────────────────────────────────────────────────────────────────────────────
# 4️⃣  FEATURE ENGINEERING & PRE‑PROCESSING PIPELINE
# ──────────────────────────────────────────────────────────────────────────────
def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds FamilySize and IsAlone columns."""
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df

# Identify column types (simple heuristic)
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

# Build sub‑pipelines
numeric_pipe = Pipeline(
    steps=[
        ("imputer", IterativeImputer(estimator=LinearRegression(),
                                     max_iter=5,
                                     random_state=SEED)),
        ("scaler", RobustScaler()),
    ]
)

categorical_pipe = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore",
                                 sparse_output=False)),
    ]
)

preprocess = Pipeline(
    steps=[
        ("feature_engineering", FunctionTransformer(add_engineered_features,
                                                    validate=False)),
        (
            "column_transform",
            ColumnTransformer(
                transformers=[
                    ("num", numeric_pipe, numeric_cols),
                    ("cat", categorical_pipe, categorical_cols),
                ],
                remainder="passthrough",  # engineered cols stay as‑is
            ),
        ),
    ]
)

# Fit on the training set only (to avoid leakage)
preprocess.fit(X_train, y_train)

# Transform all splits
X_train_proc = preprocess.transform(X_train)
X_val_proc = preprocess.transform(X_val)
X_test_proc = preprocess.transform(X_test)

# Ensure numpy dtype is float32 (required for torch)
X_train_proc = X_train_proc.astype(np.float32)
X_val_proc = X_val_proc.astype(np.float32)
X_test_proc = X_test_proc.astype(np.float32)

# Save the preprocessing pipeline for later use
joblib.dump(preprocess, MODEL_DIR / "preprocess_pipeline.pkl")

# ──────────────────────────────────────────────────────────────────────────────
# 5️⃣  BASE LEARNERS (Tree‑based)
# ──────────────────────────────────────────────────────────────────────────────
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

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
# 6️⃣  CUSTOM PYTORCH MLP (simple early‑stopping)
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
        return self.net(x).squeeze(1)  # (N,)


def train_torch_mlp(
    X_tr,
    y_tr,
    X_va,
    y_va,
    input_dim,
    hidden_dims=[64, 32],
    lr=2e-3,
    weight_decay=1e-4,
    batch_size=32,
    max_epochs=100,
    patience=10,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TorchMLP(input_dim, hidden_dims).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Datasets / loaders
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

        # ---- validation ----
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

        # Early‑stopping check
        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
            best_state = model.state_dict()
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} (val AUC={best_auc:.4f})")
            break

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d} – val ROC‑AUC: {val_auc:.4f}")

    # Load best weights
    model.load_state_dict(best_state)

    # Save checkpoint
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dims": hidden_dims,
        "best_val_auc": best_auc,
    }
    torch.save(checkpoint, MODEL_DIR / "pytorch_mlp_best.ckpt")
    return model, best_auc


# Train the MLP
input_dim = X_train_proc.shape[1]
mlp_model, mlp_best_auc = train_torch_mlp(
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
# 7️⃣  STACKING META‑LEARNER (Logistic Regression)
# ──────────────────────────────────────────────────────────────────────────────
def get_base_probas(model, X):
    """Return probability of class 1 for any supported base model."""
    if isinstance(model, (LGBMClassifier, XGBClassifier)):
        return model.predict_proba(X)[:, 1]
    elif isinstance(model, CatBoostClassifier):
        return model.predict_proba(X)[:, 1]
    elif isinstance(model, torch.nn.Module):
        # PyTorch MLP – use the same inference routine as during training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        with torch.no_grad():
            tensor = torch.from_numpy(X).to(device)
            logits = model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        return probs
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")


# Validation predictions for stacking
val_preds = np.column_stack(
    [
        get_base_probas(lgbm, X_val_proc),
        get_base_probas(cat, X_val_proc),
        get_base_probas(xgb, X_val_proc),
        get_base_probas(mlp_model, X_val_proc),
    ]
)

meta_learner = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=SEED)
meta_learner.fit(val_preds, y_val)

# Persist meta‑learner
joblib.dump(meta_learner, MODEL_DIR / "stacking_meta.pkl")

# ──────────────────────────────────────────────────────────────────────────────
# 8️⃣  FINAL EVALUATION ON TEST SET
# ──────────────────────────────────────────────────────────────────────────────
test_preds = np.column_stack(
    [
        get_base_probas(lgbm, X_test_proc),
        get_base_probas(cat, X_test_proc),
        get_base_probas(xgb, X_test_proc),
        get_base_probas(mlp_model, X_test_proc),
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
# 9️⃣  OPTIONAL: SAVE A SINGLE “Deployable” Object
# ──────────────────────────────────────────────────────────────────────────────
# For many serving scenarios it is convenient to bundle everything into one
# dictionary that can be loaded with ``joblib.load``.
deployable = {
    "preprocess": preprocess,
    "lightgbm": lgbm,
    "catboost": cat,
    "xgboost": xgb,
    "torch_mlp_ckpt_path": str(MODEL_DIR / "pytorch_mlp_best.ckpt"),
    "meta_learner": meta_learner,
}
joblib.dump(deployable, MODEL_DIR / "stacked_ensemble_deployable.pkl")

print(f"\nAll artefacts saved under: {MODEL_DIR}")

# ---------------------------------------------------------------------------
# End of script
# ---------------------------------------------------------------------------
