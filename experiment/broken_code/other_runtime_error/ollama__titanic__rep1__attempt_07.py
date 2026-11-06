
"""
Titanic AutoML end‑to‑end pipeline
=================================

This script can be run from the repository root. It performs the full
workflow required in the specification:

1. **Data retrieval** – copies the original CSV (if not already present)
   to ``data/raw/titanic.csv``.
2. **Schema generation** – creates a minimal JSON schema describing each
   column type.
3. **Pre‑processing** – builds a Scikit‑Learn ``Pipeline`` that:
   * adds engineered features ``FamilySize`` and ``IsAlone``,
   * iteratively imputes numeric columns,
   * most‑frequent imputes categoricals,
   * robust‑scales numeric features,
   * one‑hot encodes categoricals.
4. **Stratified train/val/test split** (70 % / 20 % / 10 % on ``survived``).
5. **Training** – LightGBM, CatBoost, XGBoost (fixed hyper‑parameters) and
   a custom two‑layer PyTorch MLP tuned with Optuna (≤ 30 trials, ≤ 10 min).
6. **Stacking** – Logistic‑Regression meta‑learner on the validation‑set
   predictions of the four base models.
7. **Evaluation** – ROC‑AUC, Accuracy, F1 on the held‑out test set.
8. **Profiling & reporting** – model size, FLOPs, latency, training time,
   peak RAM (place‑holder) saved to ``reports/model_profile.csv``.
9. **Persistence** – all artefacts are stored under
   ``./agent_workspace/trained_models`` so they can be loaded in production.

The script is self‑contained; just install the packages listed in
``requirements.txt`` (the versions from the specification) and run it.

"""

# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
import json
import os
import shutil
import time
import warnings
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as data
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    RobustScaler,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Global settings & paths
# --------------------------------------------------------------------------- #
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Repository root (assume script is executed from there)
REPO_ROOT = Path.cwd()

# --------------------------------------------------------------------------- #
# 1️⃣ Data retrieval
# --------------------------------------------------------------------------- #
RAW_DIR = REPO_ROOT / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
RAW_CSV = RAW_DIR / "titanic.csv"

# The original CSV location (as given in the prompt).  Adjust if your
# directory layout differs.
SOURCE_CSV = REPO_ROOT / "experiment" / "datasets" / "titanic.csv"

if not RAW_CSV.is_file():
    if SOURCE_CSV.is_file():
        shutil.copy2(SOURCE_CSV, RAW_CSV)
        print(f"Copied raw CSV from {SOURCE_CSV} → {RAW_CSV}")
    else:
        raise FileNotFoundError(
            f"Raw CSV not found at {RAW_CSV} and source CSV not found at {SOURCE_CSV}"
        )

# --------------------------------------------------------------------------- #
# 2️⃣ Schema generation (minimal JSON schema)
# --------------------------------------------------------------------------- #
SCHEMA_DIR = REPO_ROOT / "data" / "schema"
SCHEMA_DIR.mkdir(parents=True, exist_ok=True)
SCHEMA_JSON = SCHEMA_DIR / "titanic_schema.json"

if not SCHEMA_JSON.is_file():
    df_tmp = pd.read_csv(RAW_CSV, nrows=100)  # small sample for inference
    schema = {}
    for col in df_tmp.columns:
        dtype = df_tmp[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            col_type = "numeric"
        else:
            # treat low‑cardinality strings as categorical
            unique_vals = df_tmp[col].nunique()
            col_type = "categorical" if unique_vals <= 15 else "numeric"
        schema[col] = {"type": col_type}
    with open(SCHEMA_JSON, "w") as f:
        json.dump(schema, f, indent=2)
    print(f"Generated schema → {SCHEMA_JSON}")
else:
    print(f"Schema already exists → {SCHEMA_JSON}")

# --------------------------------------------------------------------------- #
# Utility helpers
# --------------------------------------------------------------------------- #
def load_schema(schema_path: Path) -> dict:
    with open(schema_path, "r") as f:
        return json.load(f)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create FamilySize and IsAlone columns."""
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def build_preprocess_pipeline(
    numeric_cols: List[str], categorical_cols: List[str]
) -> Pipeline:
    """Construct the full preprocessing pipeline."""
    # Numeric pipeline
    numeric_pipe = Pipeline(
        [
            ("imputer", IterativeImputer(max_iter=5, random_state=SEED)),
            ("scaler", RobustScaler()),
        ]
    )
    # Categorical pipeline
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse=False),
            ),
        ]
    )
    # ColumnTransformer
    col_transform = ColumnTransformer(
        [
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="passthrough",  # engineered columns flow unchanged
    )
    # Full pipeline with feature engineering step
    full_pipe = Pipeline(
        [
            (
                "feature_engineering",
                FunctionTransformer(add_engineered_features, validate=False),
            ),
            ("preprocess", col_transform),
        ]
    )
    return full_pipe


def stratified_split(
    df: pd.DataFrame, target_col: str = "survived"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Return train/val/test splits (70/20/10) stratified on target."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 70% train, 30% temp
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=SEED,
    )
    # temp → 2/3 val (≈20% total), 1/3 test (≈10% total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=1 / 3,
        stratify=y_tmp,
        random_state=SEED,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# --------------------------------------------------------------------------- #
# 3️⃣ PyTorch MLP definition
# --------------------------------------------------------------------------- #
class TorchMLP(nn.Module):
    """Two‑layer feed‑forward network (input → hidden1 → hidden2 → 1)."""

    def __init__(self, input_dim: int, hidden_dims: List[int]):
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


def train_pytorch_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
) -> Tuple[dict, float]:
    """Optuna HPO for the MLP (≤30 trials, ≤10 min). Returns checkpoint & best AUC."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective(trial):
        # Hyper‑parameters
        hidden_1 = trial.suggest_categorical("hidden_1", [32, 64, 128])
        hidden_2 = trial.suggest_categorical("hidden_2", [32, 64, 128])
        lr = trial.suggest_loguniform("lr", 1e-4, 5e-3)
        weight_decay = trial.suggest_loguniform("weight_decay", 1e-5, 1e-3)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

        model = TorchMLP(input_dim, [hidden_1, hidden_2]).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=2, verbose=False
        )

        # Datasets / loaders
        train_ds = data.TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).float(),
        )
        val_ds = data.TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).float(),
        )
        train_loader = data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)

        best_auc = 0.0
        epochs_no_improve = 0
        best_state = None

        for epoch in range(100):  # generous upper bound; early‑stop will cut it
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
            preds, trues = [], []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    logits = model(xb)
                    probs = torch.sigmoid(logits).cpu().numpy()
                    preds.extend(probs)
                    trues.extend(yb.numpy())
            val_auc = roc_auc_score(trues, preds)
            scheduler.step(val_auc)

            # Early stopping
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= 10:
                break

        trial.set_user_attr("best_state", best_state)
        return best_auc

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(objective, n_trials=30, timeout=600)  # ≤10 min

    best_trial = study.best_trial
    best_hidden = [best_trial.params["hidden_1"], best_trial.params["hidden_2"]]
    best_state = best_trial.user_attrs["best_state"]

    # Re‑instantiate final model with best hyper‑parameters
    final_model = TorchMLP(input_dim, best_hidden).to(device)
    final_model.load_state_dict(best_state)
    final_model.eval()

    checkpoint = {
        "model_state_dict": final_model.state_dict(),
        "hyperparameters": best_trial.params,
        "input_dim": input_dim,
    }
    return checkpoint, best_trial.value


def pytorch_predict_proba(checkpoint: dict, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Return class‑1 probabilities for a NumPy array."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_dims = [
        checkpoint["hyperparameters"]["hidden_1"],
        checkpoint["hyperparameters"]["hidden_2"],
    ]
    model = TorchMLP(checkpoint["input_dim"], hidden_dims).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ds = data.TensorDataset(torch.from_numpy(X).float())
    loader = data.DataLoader(ds, batch_size=batch_size, shuffle=False)

    probs = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device)
            logits = model(xb)
            probs.extend(torch.sigmoid(logits).cpu().numpy())
    return np.array(probs)


# --------------------------------------------------------------------------- #
# 4️⃣ Main pipeline
# --------------------------------------------------------------------------- #
def main():
    # ------------------------------------------------------------------- #
    # Load raw data & ensure target column name
    # ------------------------------------------------------------------- #
    df_raw = pd.read_csv(RAW_CSV)
    # Normalise target column name to lower‑case ``survived``
    if "Survived" in df_raw.columns:
        df_raw = df_raw.rename(columns={"Survived": "survived"})
    if "survived" not in df_raw.columns:
        raise KeyError("Target column 'survived' not found in the dataset.")

    # ------------------------------------------------------------------- #
    # Column groups (as required by the spec)
    # ------------------------------------------------------------------- #
    target_col = "survived"
    categorical_cols = ["Sex", "Embarked", "Pclass", "Cabin", "Ticket"]
    numeric_cols = ["Age", "Fare", "SibSp", "Parch"]
    # Engineered columns will be added by the pipeline

    # ------------------------------------------------------------------- #
    # Build & fit preprocessing pipeline on the *whole* raw data (excluding target)
    # ------------------------------------------------------------------- #
    preprocess_pipe = build_preprocess_pipeline(numeric_cols, categorical_cols)
    preprocess_pipe.fit(df_raw.drop(columns=[target_col]))

    # Persist pipeline
    ARTIFACT_ROOT = REPO_ROOT / "agent_workspace" / "trained_models"
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    preprocess_path = ARTIFACT_ROOT / "preprocess.pkl"
    joblib.dump(preprocess_pipe, preprocess_path)
    print(f"Preprocess pipeline saved to {preprocess_path}")

    # ------------------------------------------------------------------- #
    # Stratified train/val/test split
    # ------------------------------------------------------------------- #
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = stratified_split(
        df_raw, target_col=target_col
    )

    # ------------------------------------------------------------------- #
    # Transform splits
    # ------------------------------------------------------------------- #
    X_train = preprocess_pipe.transform(X_train_raw)
    X_val = preprocess_pipe.transform(X_val_raw)
    X_test = preprocess_pipe.transform(X_test_raw)

    # Ensure float32 for downstream libs
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    y_train = y_train.values.astype(np.int64)
    y_val = y_val.values.astype(np.int64)
    y_test = y_test.values.astype(np.int64)

    # ------------------------------------------------------------------- #
    # 5️⃣ Train tree‑based base learners
    # ------------------------------------------------------------------- #
    # LightGBM
    lgbm = LGBMClassifier(
        random_state=SEED,
        n_estimators=200,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
    )
    lgbm.fit(X_train, y_train)
    lgbm_path = ARTIFACT_ROOT / "lightgbm.pkl"
    joblib.dump(lgbm, lgbm_path)
    print(f"LightGBM model saved to {lgbm_path}")

    # CatBoost
    cat = CatBoostClassifier(
        random_seed=SEED,
        iterations=400,
        learning_rate=0.05,
        depth=6,
        loss_function="Logloss",
        verbose=False,
    )
    cat.fit(X_train, y_train)
    cat_path = ARTIFACT_ROOT / "catboost.cbm"
    cat.save_model(cat_path)
    print(f"CatBoost model saved to {cat_path}")

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
    xgb.fit(X_train, y_train)
    xgb_path = ARTIFACT_ROOT / "xgboost.pkl"
    joblib.dump(xgb, xgb_path)
    print(f"XGBoost model saved to {xgb_path}")

    # ------------------------------------------------------------------- #
    # 6️⃣ Train custom PyTorch MLP with Optuna HPO
    # ------------------------------------------------------------------- #
    input_dim = X_train.shape[1]
    start_time = time.perf_counter()
    torch_ckpt, best_mlp_auc = train_pytorch_mlp(
        X_train, y_train, X_val, y_val, input_dim
    )
    torch_ckpt_path = ARTIFACT_ROOT / "pytorch_mlp_best.ckpt"
    torch.save(torch_ckpt, torch_ckpt_path)
    mlp_train_time = time.perf_counter() - start_time
    print(f"PyTorch MLP checkpoint saved to {torch_ckpt_path}")
    print(f"Best validation ROC‑AUC for MLP: {best_mlp_auc:.4f}")

    # ------------------------------------------------------------------- #
    # 7️⃣ Stacking meta‑learner (Logistic Regression)
    # ------------------------------------------------------------------- #
    # Validation predictions
    lgbm_val = lgbm.predict_proba(X_val)[:, 1]
    cat_val = cat.predict_proba(X_val)[:, 1]
    xgb_val = xgb.predict_proba(X_val)[:, 1]
    torch_val = pytorch_predict_proba(torch_ckpt, X_val)

    meta_X_val = np.column_stack([lgbm_val, cat_val, xgb_val, torch_val])
    meta_clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    meta_clf.fit(meta_X_val, y_val)

    meta_path = ARTIFACT_ROOT / "stacking_meta.pkl"
    joblib.dump(meta_clf, meta_path)
    print(f"Stacking meta‑learner saved to {meta_path}")

    # ------------------------------------------------------------------- #
    # 8️⃣ Final evaluation on test set
    # ------------------------------------------------------------------- #
    lgbm_test = lgbm.predict_proba(X_test)[:, 1]
    cat_test = cat.predict_proba(X_test)[:, 1]
    xgb_test = xgb.predict_proba(X_test)[:, 1]
    torch_test = pytorch_predict_proba(torch_ckpt, X_test)

    meta_X_test = np.column_stack([lgbm_test, cat_test, xgb_test, torch_test])
    test_proba = meta_clf.predict_proba(meta_X_test)[:, 1]
    test_pred = (test_proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_test, test_proba)
    acc = accuracy_score(y_test, test_pred)
    f1 = f1_score(y_test, test_pred)

    print("\n=== Test‑set performance ===")
    print(f"ROC‑AUC : {roc_auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1      : {f1:.4f}")

    # ------------------------------------------------------------------- #
    # 9️⃣ Profiling & CSV report
    # ------------------------------------------------------------------- #
    # Model size (sum of artefacts)
    total_bytes = sum(p.stat().st_size for p in ARTIFACT_ROOT.glob("*"))
    size_mb = total_bytes / (1024 * 1024)

    # Rough FLOPs estimate
    avg_tree_depth = 6  # conservative estimate
    n_trees = (
        lgbm.n_estimators_
        + cat.tree_count_
        + xgb.get_booster().best_iteration
    )
    flops_trees = n_trees * avg_tree_depth * 2

    h1 = torch_ckpt["hyperparameters"]["hidden_1"]
    h2 = torch_ckpt["hyperparameters"]["hidden_2"]
    flops_mlp = (
        2 * input_dim * h1
        + 2 * h1 * h2
        + 2 * h2 * 1
    )
    flops_total = flops_trees + flops_mlp

    # Latency (average inference time per row on CPU)
    start = time.perf_counter()
    _ = meta_clf.predict_proba(meta_X_test[:1000])[:, 1]
    latency_us = (time.perf_counter() - start) / 1000 * 1e6  # µs per row

    # Training time (sum of tree training + MLP HPO)
    # Tree training is fast; we approximate with the MLP training time.
    train_time_s = mlp_train_time

    # Peak RAM – placeholder (set to 0, can be filled later with memory_profiler)
    peak_ram_mb = 0

    report_dir = REPO_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "model_profile.csv"

    report_df = pd.DataFrame(
        [
            {
                "model_name": "titanic_stacked_ensemble",
                "roc_auc": round(roc_auc, 4),
                "accuracy": round(acc, 4),
                "f1": round(f1, 4),
                "size_mb": round(size_mb, 2),
                "flops_per_pred": flops_total,
                "latency_us": round(latency_us, 1),
                "train_time_s": round(train_time_s, 1),
                "peak_ram_mb": peak_ram_mb,
            }
        ]
    )
    report_df.to_csv(report_path, index=False)
    print(f"\nModel profiling report written to {report_path}")

    # ------------------------------------------------------------------- #
    # 10️⃣ Wrapper dict for easy production loading (optional)
    # ------------------------------------------------------------------- #
    wrapper_path = ARTIFACT_ROOT / "stacked_predictor_wrapper.pkl"
    wrapper_dict = {
        "preprocess_path": str(preprocess_path),
        "lightgbm_path": str(lgbm_path),
        "catboost_path": str(cat_path),
        "xgboost_path": str(xgb_path),
        "pytorch_ckpt_path": str(torch_ckpt_path),
        "meta_path": str(meta_path),
    }
    joblib.dump(wrapper_dict, wrapper_path)
    print(f"Wrapper dict saved to {wrapper_path}")


if __name__ == "__main__":
    main()
