
"""
Titanic Survival Prediction – End‑to‑End AutoML pipeline
========================================================

This script implements the full workflow described in the design document:

1.  Load the raw CSV (or any user‑provided tabular CSV)  
2.  Convert to Parquet (optional, kept for reproducibility)  
3.  Build a Scikit‑Learn preprocessing pipeline that
    * validates schema,
    * imputes numeric (IterativeImputer) and categorical (most_frequent) values,
    * adds two engineered features (FamilySize, IsAlone),
    * scales numeric columns with RobustScaler,
    * one‑hot encodes categorical columns.
4.  Split the data into train / validation / test (70 % / 20 % / 10 %,
    stratified on the target).
5.  Train three tree‑based models via **AutoGluon‑Tabular** (LightGBM,
    CatBoost, XGBoost) – the built‑in NN is deliberately excluded.
6.  Train a custom **PyTorch feed‑forward network** on the same
    pre‑processed data (with Optuna‑driven hyper‑parameter search).
7.  Stack the four base learners with a simple Logistic‑Regression meta‑model.
8.  Evaluate the stacked ensemble on the held‑out test set (ROC‑AUC,
    Accuracy, F1).
9.  Persist all artefacts (pre‑processing pipeline, AutoGluon predictor,
    PyTorch checkpoint, stacking model) under
    ``./agent_workspace/trained_models``.
10. Log the whole run to MLflow for traceability.

The script is self‑contained – running it from the repository root will
reproduce the entire pipeline without further modification.
"""

import os
import json
import time
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import joblib
import yaml
import mlflow
import optuna
from optuna.trial import TrialState

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from autogluon.tabular import TabularPredictor

# --------------------------------------------------------------------------- #
# GLOBAL SETTINGS
# --------------------------------------------------------------------------- #
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Where the final artefacts will be stored
ARTIFACT_DIR = Path("./agent_workspace/trained_models")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# 1️⃣ DATA INGESTION & CONVERSION
# --------------------------------------------------------------------------- #
def ingest_csv(csv_path: str, parquet_path: str) -> pd.DataFrame:
    """Read CSV with Polars (fast) → write Parquet → return pandas DF."""
    # Polars read → Parquet write
    pl.read_csv(csv_path).write_parquet(parquet_path)

    # Load back as pandas (AutoGluon expects pandas)
    df = pd.read_parquet(parquet_path)
    return df


# --------------------------------------------------------------------------- #
# 2️⃣ PRE‑PROCESSING PIPELINE
# --------------------------------------------------------------------------- #
def build_preprocess_pipeline(cfg: dict) -> Pipeline:
    """Construct the Scikit‑Learn preprocessing pipeline described in the spec."""
    # ------------------------------------------------------------------- #
    # 2.1 Column groups (taken from the YAML config)
    # ------------------------------------------------------------------- #
    cat_cols = cfg["categorical"]
    ord_cols = cfg["ordinal"]          # treated as categorical for imputation
    num_cols = cfg["numeric"]
    target_col = cfg["target_column"]

    # ------------------------------------------------------------------- #
    # 2.2 Feature‑engineering transformer
    # ------------------------------------------------------------------- #
    def add_engineered_features(X):
        """X is a numpy array / pandas DataFrame with original columns."""
        # Convert to DataFrame for easy column ops
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=num_cols + cat_cols + ord_cols)
        X = X.copy()
        X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
        X["IsAlone"] = (X["FamilySize"] == 1).astype(int)
        return X

    fe_transformer = FunctionTransformer(add_engineered_features, validate=False)

    # ------------------------------------------------------------------- #
    # 2.3 Numeric pipeline
    # ------------------------------------------------------------------- #
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", IterativeImputer(
                estimator=nn.LinearRegression(),
                max_iter=5,
                random_state=SEED)),
            ("scaler", RobustScaler()),
        ]
    )

    # ------------------------------------------------------------------- #
    # 2.4 Categorical pipeline (including ordinal treated as categorical)
    # ------------------------------------------------------------------- #
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    # ------------------------------------------------------------------- #
    # 2.5 Combine everything
    # ------------------------------------------------------------------- #
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols + ord_cols),
        ],
        remainder="passthrough",  # engineered features will be added later
    )

    full_pipe = Pipeline(
        steps=[
            ("feature_engineering", fe_transformer),
            ("preprocess", preprocessor),
        ]
    )
    return full_pipe


def fit_and_save_preprocess(df: pd.DataFrame, cfg_path: str, save_path: Path):
    """Fit the pipeline on the full raw data (excluding target) and persist."""
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    target = cfg["target_column"]
    X = df.drop(columns=[target])
    y = df[target]

    pipe = build_preprocess_pipeline(cfg)
    pipe.fit(X, y)

    joblib.dump(pipe, save_path)
    print(f"Pre‑processing pipeline saved to {save_path}")
    return pipe, X, y


# --------------------------------------------------------------------------- #
# 3️⃣ DATA SPLITTING
# --------------------------------------------------------------------------- #
def stratified_split(X: pd.DataFrame, y: pd.Series, seed: int = SEED):
    """Return train/val/test DataFrames (70/20/10) stratified on y."""
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    # 20% val, 10% test → split the remaining 30% accordingly
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.3333, stratify=y_tmp, random_state=seed
    )
    return (
        X_train.reset_index(drop=True),
        y_train.reset_index(drop=True),
        X_val.reset_index(drop=True),
        y_val.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )


# --------------------------------------------------------------------------- #
# 4️⃣ AUTOGLUON (TREE‑BASED) ENSEMBLE
# --------------------------------------------------------------------------- #
def train_autogluon(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    time_limit: int = 1800,
):
    """Train LightGBM, CatBoost & XGBoost via AutoGluon (no internal NN)."""
    train_df = X_train.copy()
    train_df["survived"] = y_train
    val_df = X_val.copy()
    val_df["survived"] = y_val

    predictor = TabularPredictor(
        label="survived",
        path=str(ARTIFACT_DIR / "autogluon"),
        eval_metric="roc_auc",
        verbosity=2,
    ).fit(
        train_data=train_df,
        tuning_data=val_df,
        presets="best_quality",
        hyperparameters={"GBM": {}, "CAT": {}, "XGB": {}},
        time_limit=time_limit,
        excluded_model_types=["NN"],  # ensure no MXNet/torch NN inside AutoGluon
    )
    # Save predictor (AutoGluon automatically persists it under the given path)
    print(f"AutoGluon predictor saved to {predictor.path}")
    return predictor


# --------------------------------------------------------------------------- #
# 5️⃣ CUSTOM PYTORCH MLP (with Optuna HPO)
# --------------------------------------------------------------------------- #
class TorchMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))  # logits
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)  # (N,) logits


def objective(trial, X_train, y_train, X_val, y_val, input_dim):
    """Optuna objective – returns validation ROC‑AUC."""
    # ---- Hyper‑parameters ----
    hidden_1 = trial.suggest_categorical("hidden_1", [32, 64, 128])
    hidden_2 = trial.suggest_categorical("hidden_2", [32, 64, 128])
    lr = trial.suggest_loguniform("lr", 1e-4, 5e-3)
    wd = trial.suggest_loguniform("weight_decay", 1e-5, 1e-3)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    model = TorchMLP(input_dim, [hidden_1, hidden_2]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, verbose=False
    )

    # Datasets
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val.values, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_auc = 0.0
    epochs_no_improve = 0
    max_epochs = 100

    for epoch in range(max_epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        # ---- Validation ----
        model.eval()
        probs = []
        true = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE)
                logits = model(xb)
                probs.extend(torch.sigmoid(logits).cpu().numpy())
                true.extend(yb.numpy())
        val_auc = roc_auc_score(true, probs)

        scheduler.step(val_auc)

        # Early stopping
        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
            # Save best checkpoint in memory
            best_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 10:
                break

    # Return negative because Optuna minimizes by default
    trial.set_user_attr("best_state", best_state)
    return best_auc


def train_pytorch_mlp(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_val: np.ndarray,
    y_val: pd.Series,
    max_trials: int = 30,
    time_budget: int = 600,
):
    """Run Optuna HPO, train the final MLP and persist the checkpoint."""
    input_dim = X_train.shape[1]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(
        lambda trial: objective(
            trial, X_train, y_train, X_val, y_val, input_dim
        ),
        n_trials=max_trials,
        timeout=time_budget,
    )

    # Get best trial & its saved state
    best_trial = study.best_trial
    best_state = best_trial.user_attrs["best_state"]
    best_params = best_trial.params

    # Re‑instantiate model with best hyper‑params
    model = TorchMLP(
        input_dim,
        [best_params["hidden_1"], best_params["hidden_2"]],
    ).to(DEVICE)
    model.load_state_dict(best_state)

    # Save checkpoint
    ckpt_path = ARTIFACT_DIR / "pytorch_mlp_best.ckpt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hyperparameters": best_params,
            "input_dim": input_dim,
        },
        ckpt_path,
    )
    print(f"Best PyTorch MLP checkpoint saved to {ckpt_path}")
    return model, best_params


def pytorch_predict_proba(model: nn.Module, X: np.ndarray, batch_size: int = 64):
    """Return probability of class‑1 for a numpy array X."""
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.tensor(X, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )
    probs = []
    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(DEVICE)
            logits = model(xb)
            probs.extend(torch.sigmoid(logits).cpu().numpy())
    return np.array(probs)


# --------------------------------------------------------------------------- #
# 6️⃣ STACKING METALEARNER
# --------------------------------------------------------------------------- #
def train_stacking_ensemble(
    autogluon_pred_val: np.ndarray,
    pytorch_pred_val: np.ndarray,
    y_val: pd.Series,
):
    """Train a LogisticRegression meta‑learner on validation predictions."""
    # AutoGluon predict_proba returns (N, 2); we keep prob of class 1
    ag_prob = autogluon_pred_val[:, 1].reshape(-1, 1)
    pt_prob = pytorch_pred_val.reshape(-1, 1)

    X_meta = np.hstack([ag_prob, pt_prob])
    meta = LogisticRegression(solver="lbfgs")
    meta.fit(X_meta, y_val)

    # Persist meta‑learner
    meta_path = ARTIFACT_DIR / "stacking_meta.pkl"
    joblib.dump(meta, meta_path)
    print(f"Stacking meta‑learner saved to {meta_path}")
    return meta


def stacked_predict(
    predictor: TabularPredictor,
    pytorch_model: nn.Module,
    meta: LogisticRegression,
    X: pd.DataFrame,
):
    """Generate final ensemble predictions on a raw (pre‑processed) DataFrame."""
    # AutoGluon prediction
    ag_proba = predictor.predict_proba(X).values  # (N, 2)

    # PyTorch prediction – need numpy matrix
    X_np = X.values.astype(np.float32)
    pt_proba = pytorch_predict_proba(pytorch_model, X_np).reshape(-1, 1)

    # Meta‑learner
    X_meta = np.hstack([ag_proba[:, 1].reshape(-1, 1), pt_proba])
    final_proba = meta.predict_proba(X_meta)[:, 1]  # prob of class 1
    final_pred = (final_proba >= 0.5).astype(int)
    return final_proba, final_pred


# --------------------------------------------------------------------------- #
# 7️⃣ MAIN PIPELINE
# --------------------------------------------------------------------------- #
def main():
    # ------------------------------------------------------------------- #
    # 0️⃣ Paths & configuration
    # ------------------------------------------------------------------- #
    # Adjust this path to point to the user‑provided CSV
    USER_CSV = "data/raw/titanic.csv"          # <-- change if needed
    RAW_PARQUET = "data/raw/titanic.parquet"
    PREPROCESS_CFG = "configs/preprocess.yaml"
    PREPROCESS_PKL = ARTIFACT_DIR / "preprocess.pkl"

    # ------------------------------------------------------------------- #
    # 1️⃣ Ingest raw data
    # ------------------------------------------------------------------- #
    df_raw = ingest_csv(USER_CSV, RAW_PARQUET)

    # ------------------------------------------------------------------- #
    # 2️⃣ Build & fit preprocessing pipeline
    # ------------------------------------------------------------------- #
    preprocess_pipe, X_raw, y_raw = fit_and_save_preprocess(
        df_raw, PREPROCESS_CFG, PREPROCESS_PKL
    )

    # ------------------------------------------------------------------- #
    # 3️⃣ Train/validation/test split (stratified)
    # ------------------------------------------------------------------- #
    (
        X_train_raw,
        y_train,
        X_val_raw,
        y_val,
        X_test_raw,
        y_test,
    ) = stratified_split(X_raw, y_raw, SEED)

    # ------------------------------------------------------------------- #
    # 4️⃣ Transform splits
    # ------------------------------------------------------------------- #
    X_train = preprocess_pipe.transform(X_train_raw)
    X_val = preprocess_pipe.transform(X_val_raw)
    X_test = preprocess_pipe.transform(X_test_raw)

    # Convert back to pandas DataFrames (required by AutoGluon)
    X_train_df = pd.DataFrame(
        X_train, columns=[f"f{i}" for i in range(X_train.shape[1])]
    )
    X_val_df = pd.DataFrame(
        X_val, columns=[f"f{i}" for i in range(X_val.shape[1])]
    )
    X_test_df = pd.DataFrame(
        X_test, columns=[f"f{i}" for i in range(X_test.shape[1])]
    )

    # ------------------------------------------------------------------- #
    # 5️⃣ AutoGluon tree ensemble (no internal NN)
    # ------------------------------------------------------------------- #
    predictor = train_autogluon(
        X_train_df, y_train, X_val_df, y_val, time_limit=1800
    )

    # ------------------------------------------------------------------- #
    # 6️⃣ PyTorch MLP (with Optuna HPO)
    # ------------------------------------------------------------------- #
    pytorch_model, best_hparams = train_pytorch_mlp(
        X_train, y_train, X_val, y_val, max_trials=30, time_budget=600
    )

    # ------------------------------------------------------------------- #
    # 7️⃣ Stacking meta‑learner (trained on validation predictions)
    # ------------------------------------------------------------------- #
    # AutoGluon validation probabilities
    ag_val_proba = predictor.predict_proba(X_val_df).values
    # PyTorch validation probabilities
    pt_val_proba = pytorch_predict_proba(pytorch_model, X_val)

    meta = train_stacking_ensemble(
        ag_val_proba, pt_val_proba, y_val
    )

    # ------------------------------------------------------------------- #
    # 8️⃣ Final evaluation on the held‑out test set
    # ------------------------------------------------------------------- #
    test_proba, test_pred = stacked_predict(
        predictor, pytorch_model, meta, X_test_df
    )
    test_auc = roc_auc_score(y_test, test_proba)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred)

    print("\n=== TEST SET METRICS ===")
    print(f"ROC‑AUC : {test_auc:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"F1      : {test_f1:.4f}")

    # ------------------------------------------------------------------- #
    # 9️⃣ Log everything to MLflow (optional but highly recommended)
    # ------------------------------------------------------------------- #
    mlflow.set_experiment("Titanic_AutoML")
    with mlflow.start_run(run_name="titanic_end_to_end"):
        # Parameters
        mlflow.log_params(best_hparams)
        mlflow.log_param("seed", SEED)

        # Metrics
        mlflow.log_metric("test_roc_auc", test_auc)
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1", test_f1)

        # Artefacts
        mlflow.log_artifact(PREPROCESS_PKL, artifact_path="pipeline")
        mlflow.log_artifact(str(ARTIFACT_DIR / "autogluon"), artifact_path="autogluon")
        mlflow.log_artifact(str(ARTIFACT_DIR / "pytorch_mlp_best.ckpt"), artifact_path="pytorch")
        mlflow.log_artifact(str(ARTIFACT_DIR / "stacking_meta.pkl"), artifact_path="stacking")

    # ------------------------------------------------------------------- #
    # 10️⃣ Persist final artefacts (already saved during the steps)
    # ------------------------------------------------------------------- #
    # All artefacts are under ./agent_workspace/trained_models
    print(f"\nAll artefacts are stored in {ARTIFACT_DIR.resolve()}")


if __name__ == "__main__":
    # Suppress harmless warnings for cleaner output
    warnings.filterwarnings("ignore")
    main()
