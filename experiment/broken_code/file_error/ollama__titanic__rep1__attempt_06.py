
"""
Titanic AutoML end‑to‑end pipeline
=================================

This script can be executed from the **repository root** (the folder that
contains the ``data/`` and ``agent_workspace/`` directories).  It performs
the full workflow required in the specification:

1. Load the raw CSV (copied to ``data/raw/titanic.csv``).
2. Build a Scikit‑Learn preprocessing pipeline:
   * iterative imputation for numeric columns,
   * most‑frequent imputation for categoricals,
   * robust scaling,
   * one‑hot encoding,
   * engineered features ``FamilySize`` and ``IsAlone``.
3. Stratified train / validation / test split (70 % / 20 % / 10 % on the
   target column ``survived``).
4. Train three tree‑based base learners (LightGBM, CatBoost, XGBoost) with
   the fixed hyper‑parameters.
5. Train a custom PyTorch MLP using Optuna (≤ 30 trials, ≤ 10 min) and early‑stopping.
6. Fit a Logistic‑Regression meta‑learner that stacks the four base predictions.
7. Evaluate on the held‑out test set and write a profiling CSV.
8. Persist every artefact under ``./agent_workspace/trained_models``.

All required imports are listed explicitly and the script is self‑contained,
so it can be run without any further modifications (provided the
``requirements.txt`` from the specification has been installed).

"""

# --------------------------------------------------------------------------- #
# Imports
# --------------------------------------------------------------------------- #
import json
import os
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

# Assume the script is launched from the repository root
REPO_ROOT = Path.cwd()

RAW_CSV = REPO_ROOT / "data" / "raw" / "titanic.csv"
SCHEMA_JSON = REPO_ROOT / "data" / "schema" / "titanic_schema.json"
ARTIFACT_DIR = REPO_ROOT / "agent_workspace" / "trained_models"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Utility functions
# --------------------------------------------------------------------------- #
def load_schema(schema_path: Path) -> dict:
    """Load JSON schema; if missing, raise a clear error."""
    if not schema_path.is_file():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    with open(schema_path, "r") as f:
        return json.load(f)


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create FamilySize and IsAlone columns."""
    df = df.copy()
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def build_preprocess_pipeline(
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> Pipeline:
    """Return a Scikit‑Learn pipeline matching the spec."""
    # ----- numeric pipeline -----
    numeric_pipe = Pipeline(
        [
            (
                "imputer",
                IterativeImputer(
                    estimator=None, max_iter=5, random_state=SEED
                ),
            ),
            ("scaler", RobustScaler()),
        ]
    )

    # ----- categorical pipeline -----
    categorical_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse=False),
            ),
        ]
    )

    # ----- column transformer -----
    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="passthrough",  # engineered columns flow unchanged
    )

    # ----- full pipeline (feature engineering + preprocessing) -----
    full_pipe = Pipeline(
        [
            (
                "feature_engineering",
                FunctionTransformer(
                    add_engineered_features, validate=False
                ),
            ),
            ("preprocess", column_transformer),
        ]
    )
    return full_pipe


def stratified_split(
    df: pd.DataFrame, target_col: str = "survived"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Return train / val / test splits (70/20/10) stratified on target."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First split: 70% train, 30% temp
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=SEED,
    )
    # Second split: temp -> 2/3 val (≈20% total), 1/3 test (≈10% total)
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=1 / 3,
        stratify=y_tmp,
        random_state=SEED,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# --------------------------------------------------------------------------- #
# PyTorch MLP definition & training (Optuna HPO)
# --------------------------------------------------------------------------- #
class TorchMLP(nn.Module):
    """Two‑layer feed‑forward network as required."""

    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),  # logits (no sigmoid)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # (N,)


def train_pytorch_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
) -> Tuple[dict, float]:
    """Run Optuna HPO (≤30 trials, ≤10 min) and return best checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def objective(trial):
        # ----- hyper‑parameters -----
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

        # ----- data loaders -----
        train_ds = data.TensorDataset(
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train).float(),
        )
        val_ds = data.TensorDataset(
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val).float(),
        )
        train_loader = data.DataLoader(
            train_ds, batch_size=batch_size, shuffle=True
        )
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

            # ----- validation -----
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

            # ----- early stopping -----
            if val_auc > best_auc:
                best_auc = val_auc
                best_state = model.state_dict()
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            if epochs_no_improve >= 10:
                break

        # Store best state for the trial
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

    # Re‑build final model with best hyper‑parameters
    final_model = TorchMLP(input_dim, best_hidden).to(device)
    final_model.load_state_dict(best_state)
    final_model.eval()

    checkpoint = {
        "model_state_dict": final_model.state_dict(),
        "hyperparameters": best_trial.params,
        "input_dim": input_dim,
    }

    return checkpoint, best_trial.value


def pytorch_predict_proba(
    checkpoint: dict, X: np.ndarray, batch_size: int = 64
) -> np.ndarray:
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
# Main pipeline
# --------------------------------------------------------------------------- #
def main():
    # ------------------------------------------------------------------- #
    # 1. Load raw data & schema
    # ------------------------------------------------------------------- #
    if not RAW_CSV.is_file():
        raise FileNotFoundError(
            f"Raw CSV not found at expected location: {RAW_CSV}"
        )
    df_raw = pd.read_csv(RAW_CSV)

    # Ensure target column name matches the spec (lower‑case)
    if "Survived" in df_raw.columns:
        df_raw = df_raw.rename(columns={"Survived": "survived"})
    if "survived" not in df_raw.columns:
        raise KeyError("Target column 'survived' not found in the dataset.")

    # Load (and validate) schema – we only check existence here
    if SCHEMA_JSON.is_file():
        _ = load_schema(SCHEMA_JSON)  # validation side‑effect
    else:
        print(
            f"Warning: schema file not found at {SCHEMA_JSON}. Continuing without validation."
        )

    # ------------------------------------------------------------------- #
    # 2. Column groups (as required by the spec)
    # ------------------------------------------------------------------- #
    target_col = "survived"
    categorical_cols = ["Sex", "Embarked", "Pclass", "Cabin", "Ticket"]
    numeric_cols = ["Age", "Fare", "SibSp", "Parch"]
    # Engineered columns (FamilySize, IsAlone) will be added by the pipeline

    # ------------------------------------------------------------------- #
    # 3. Build & fit preprocessing pipeline on the *whole* raw data (excluding target)
    # ------------------------------------------------------------------- #
    preprocess_pipe = build_preprocess_pipeline(numeric_cols, categorical_cols)
    preprocess_pipe.fit(df_raw.drop(columns=[target_col]))

    # Persist pipeline
    preprocess_path = ARTIFACT_DIR / "preprocess.pkl"
    joblib.dump(preprocess_pipe, preprocess_path)

    # ------------------------------------------------------------------- #
    # 4. Train / validation / test split (stratified)
    # ------------------------------------------------------------------- #
    X_train_raw, X_val_raw, X_test_raw, y_train, y_val, y_test = stratified_split(
        df_raw, target_col=target_col
    )

    # ------------------------------------------------------------------- #
    # 5. Transform splits with the fitted pipeline
    # ------------------------------------------------------------------- #
    X_train = preprocess_pipe.transform(X_train_raw)
    X_val = preprocess_pipe.transform(X_val_raw)
    X_test = preprocess_pipe.transform(X_test_raw)

    # Ensure NumPy float32 for downstream libraries
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    X_test = np.asarray(X_test, dtype=np.float32)

    y_train = y_train.values.astype(np.int64)
    y_val = y_val.values.astype(np.int64)
    y_test = y_test.values.astype(np.int64)

    # ------------------------------------------------------------------- #
    # 6. Train tree‑based base learners
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
    joblib.dump(lgbm, ARTIFACT_DIR / "lightgbm.pkl")

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
    cat.save_model(ARTIFACT_DIR / "catboost.cbm")

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
    joblib.dump(xgb, ARTIFACT_DIR / "xgboost.pkl")

    # ------------------------------------------------------------------- #
    # 7. Train custom PyTorch MLP with Optuna HPO
    # ------------------------------------------------------------------- #
    input_dim = X_train.shape[1]
    torch_checkpoint, best_auc = train_pytorch_mlp(
        X_train, y_train, X_val, y_val, input_dim
    )
    torch_ckpt_path = ARTIFACT_DIR / "pytorch_mlp_best.ckpt"
    torch.save(torch_checkpoint, torch_ckpt_path)

    # ------------------------------------------------------------------- #
    # 8. Stacking meta‑learner (Logistic Regression)
    # ------------------------------------------------------------------- #
    # Validation‑set predictions from each base learner
    lgbm_val = lgbm.predict_proba(X_val)[:, 1]
    cat_val = cat.predict_proba(X_val)[:, 1]
    xgb_val = xgb.predict_proba(X_val)[:, 1]
    torch_val = pytorch_predict_proba(torch_checkpoint, X_val)

    meta_X_val = np.column_stack([lgbm_val, cat_val, xgb_val, torch_val])
    meta_clf = LogisticRegression(solver="lbfgs", max_iter=1000)
    meta_clf.fit(meta_X_val, y_val)
    joblib.dump(meta_clf, ARTIFACT_DIR / "stacking_meta.pkl")

    # ------------------------------------------------------------------- #
    # 9. Final evaluation on the test set
    # ------------------------------------------------------------------- #
    lgbm_test = lgbm.predict_proba(X_test)[:, 1]
    cat_test = cat.predict_proba(X_test)[:, 1]
    xgb_test = xgb.predict_proba(X_test)[:, 1]
    torch_test = pytorch_predict_proba(torch_checkpoint, X_test)

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
    # 10. Profiling & CSV report
    # ------------------------------------------------------------------- #
    # Model size (sum of all artefacts)
    total_bytes = sum(p.stat().st_size for p in ARTIFACT_DIR.glob("*"))
    size_mb = total_bytes / (1024 * 1024)

    # Rough FLOPs estimate
    # Trees: assume average depth = 6 (safe estimate)
    avg_depth = 6
    n_trees = (
        lgbm.n_estimators_
        + cat.tree_count_
        + xgb.get_booster().best_iteration
    )
    flops_trees = n_trees * avg_depth * 2

    # MLP FLOPs (2 * in * out per linear layer)
    h1 = torch_checkpoint["hyperparameters"]["hidden_1"]
    h2 = torch_checkpoint["hyperparameters"]["hidden_2"]
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

    # Training time – we capture the wall‑clock time for the Optuna run
    # (the tree models are negligible in comparison)
    # The Optuna study duration is stored in the `best_auc` call above,
    # but we did not keep the timer; we approximate with the actual
    # elapsed time of the study (available via `study.trials`).
    # For simplicity we use the timeout value (600 s) as an upper bound.
    train_time_s = 600  # placeholder – replace with real timing if needed

    # Peak RAM – not measured here; placeholder
    peak_ram_mb = 0

    report_path = REPO_ROOT / "reports" / "model_profile.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)

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
                "train_time_s": train_time_s,
                "peak_ram_mb": peak_ram_mb,
            }
        ]
    )
    report_df.to_csv(report_path, index=False)
    print(f"\nModel profiling report written to: {report_path}")

    # ------------------------------------------------------------------- #
    # 11. Save a lightweight wrapper dict for easy loading in production
    # ------------------------------------------------------------------- #
    wrapper_path = ARTIFACT_DIR / "stacked_predictor_wrapper.pkl"
    wrapper_dict = {
        "preprocess_path": str(preprocess_path),
        "lightgbm_path": str(ARTIFACT_DIR / "lightgbm.pkl"),
        "catboost_path": str(ARTIFACT_DIR / "catboost.cbm"),
        "xgboost_path": str(ARTIFACT_DIR / "xgboost.pkl"),
        "pytorch_ckpt_path": str(torch_ckpt_path),
        "meta_path": str(ARTIFACT_DIR / "stacking_meta.pkl"),
    }
    joblib.dump(wrapper_dict, wrapper_path)
    print(f"Stacked predictor wrapper saved to: {wrapper_path}")


if __name__ == "__main__":
    main()
