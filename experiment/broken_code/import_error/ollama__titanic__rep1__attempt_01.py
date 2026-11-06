
"""
Titanic Survival – End‑to‑End AutoML pipeline
================================================

This script implements a reproducible, production‑ready workflow for the
Titanic tabular classification problem:

1️⃣  Load the raw CSV (any user‑provided path) and store a Parquet copy.
2️⃣  Generate a simple JSON schema (numeric vs categorical) for validation.
3️⃣  Build a Scikit‑Learn preprocessing pipeline that:
    • validates the schema,
    • imputes numeric (IterativeImputer) and categorical (most_frequent) values,
    • adds two engineered features (FamilySize, IsAlone),
    • scales numeric columns with RobustScaler,
    • one‑hot encodes categorical columns.
4️⃣  Split the data (70 % train / 20 % validation / 10 % test) stratified on the target.
5️⃣  Fit the preprocessing pipeline on the full training set and persist it.
6️⃣  Transform the three splits with the fitted pipeline.
7️⃣  Train three tree‑based models via **AutoGluon‑Tabular** (LightGBM, CatBoost,
    XGBoost) – the internal MXNet/torch NN is deliberately excluded.
8️⃣  Train a custom PyTorch feed‑forward network (with Optuna hyper‑parameter
    optimisation) on the same pre‑processed data.
9️⃣  Stack the AutoGluon predictions and the PyTorch predictions with a
    Logistic‑Regression meta‑learner.
🔟  Evaluate the stacked ensemble on the held‑out test set (ROC‑AUC, Accuracy,
    F1) and persist all artefacts under ``./agent_workspace/trained_models``.
"""

import os
import json
import warnings
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import joblib
import yaml
import mlflow
import optuna

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, FunctionTransformer
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

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

# Directory where all artefacts will be stored
ARTIFACT_DIR = Path("./agent_workspace/trained_models")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# 1️⃣ DATA INGESTION & CONVERSION
# --------------------------------------------------------------------------- #
def ingest_csv(csv_path: str, parquet_path: str) -> pd.DataFrame:
    """
    Load a CSV with pandas, write a Parquet copy (for reproducibility) and
    return the DataFrame.
    """
    df = pd.read_csv(csv_path)
    # Ensure the parquet directory exists
    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
    # Write parquet – requires pyarrow (installed as a pandas extra)
    df.to_parquet(parquet_path, engine="pyarrow", index=False)
    return df


# --------------------------------------------------------------------------- #
# 2️⃣ SCHEMA GENERATION
# --------------------------------------------------------------------------- #
def generate_schema(df: pd.DataFrame, schema_path: str, threshold: int = 15) -> None:
    """
    Very simple schema: columns with > ``threshold`` unique values are considered
    numeric, everything else categorical.  The result is saved as JSON.
    """
    schema = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_type = "numeric"
        else:
            # treat low‑cardinality as categorical, high‑cardinality as numeric
            col_type = "categorical" if df[col].nunique() <= threshold else "numeric"
        schema[col] = {"type": col_type}
    Path(schema_path).parent.mkdir(parents=True, exist_ok=True)
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)


# --------------------------------------------------------------------------- #
# 3️⃣ PRE‑PROCESSING PIPELINE
# --------------------------------------------------------------------------- #
def _add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering: FamilySize and IsAlone."""
    X = X.copy()
    X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
    X["IsAlone"] = (X["FamilySize"] == 1).astype(int)
    return X


def build_preprocess_pipeline(cfg: dict) -> Pipeline:
    """
    Build the Scikit‑Learn pipeline according to ``configs/preprocess.yaml``.
    """
    cat_cols: List[str] = cfg["categorical"]
    ord_cols: List[str] = cfg["ordinal"]
    num_cols: List[str] = cfg["numeric"]
    target_col: str = cfg["target_column"]

    # ------------------------------------------------------------------- #
    # Numeric sub‑pipeline
    # ------------------------------------------------------------------- #
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

    # ------------------------------------------------------------------- #
    # Categorical sub‑pipeline (ordinal columns are treated as categorical)
    # ------------------------------------------------------------------- #
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    # ------------------------------------------------------------------- #
    # ColumnTransformer – note that engineered features are added *before*
    # the transformer, so they are included in the ``remainder``.
    # ------------------------------------------------------------------- #
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols + ord_cols),
        ],
        remainder="passthrough",  # engineered features will pass through unchanged
    )

    # ------------------------------------------------------------------- #
    # Full pipeline: feature engineering → column transformer
    # ------------------------------------------------------------------- #
    full_pipe = Pipeline(
        steps=[
            ("feature_engineering", FunctionTransformer(_add_engineered_features, validate=False)),
            ("preprocess", preprocessor),
        ]
    )
    return full_pipe


def fit_preprocess_and_save(
    df: pd.DataFrame,
    cfg_path: str,
    save_path: Path,
) -> Tuple[Pipeline, pd.DataFrame, pd.Series]:
    """Fit the pipeline on the full raw data (excluding target) and persist it."""
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    target = cfg["target_column"]
    X = df.drop(columns=[target])
    y = df[target]

    pipe = build_preprocess_pipeline(cfg)
    pipe.fit(X, y)

    joblib.dump(pipe, save_path)
    print(f"✅ Pre‑processing pipeline saved to {save_path}")
    return pipe, X, y


# --------------------------------------------------------------------------- #
# 4️⃣ DATA SPLITTING
# --------------------------------------------------------------------------- #
def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Return train / validation / test splits with a 70/20/10 ratio,
    stratified on the target.
    """
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    # 20% of the original data → 20/70 ≈ 0.2857 of the remaining 30%
    val_frac_of_tmp = 0.20 / 0.30  # ≈ 0.6667
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=1 - val_frac_of_tmp,
        stratify=y_tmp,
        random_state=seed,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


# --------------------------------------------------------------------------- #
# 5️⃣ AUTOGLUON (TREE‑BASED) ENSEMBLE
# --------------------------------------------------------------------------- #
def train_autogluon(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    time_limit: int = 1800,
) -> TabularPredictor:
    """
    Train LightGBM, CatBoost and XGBoost via AutoGluon.
    The internal NN is excluded (we will use a custom PyTorch model later).
    """
    # AutoGluon expects the label column to be present in the DataFrame
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
    print(f"✅ AutoGluon predictor saved to {predictor.path}")
    return predictor


# --------------------------------------------------------------------------- #
# 6️⃣ CUSTOM PYTORCH MLP (with Optuna HPO)
# --------------------------------------------------------------------------- #
class TorchMLP(nn.Module):
    """Simple feed‑forward network as described in the design doc."""

    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))  # logits (no sigmoid)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)  # (N,)


def _objective(
    trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
) -> float:
    """Optuna objective – maximise validation ROC‑AUC."""
    # Hyper‑parameters
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

    # Datasets / loaders
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32),
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

        # Validation
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
            best_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 10:
                break

    # Store the best state so the outer routine can retrieve it
    trial.set_user_attr("best_state", best_state)
    return best_auc


def train_pytorch_mlp(
    X_train: np.ndarray,
    y_train: pd.Series,
    X_val: np.ndarray,
    y_val: pd.Series,
    max_trials: int = 30,
    time_budget: int = 600,
) -> Tuple[TorchMLP, dict]:
    """
    Run Optuna HPO, re‑train the best model on the full training data and
    persist the checkpoint.
    """
    input_dim = X_train.shape[1]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(
        lambda trial: _objective(
            trial,
            X_train,
            y_train.values,
            X_val,
            y_val.values,
            input_dim,
        ),
        n_trials=max_trials,
        timeout=time_budget,
    )

    best_trial = study.best_trial
    best_state = best_trial.user_attrs["best_state"]
    best_params = best_trial.params

    # Re‑instantiate the model with the best hyper‑parameters
    model = TorchMLP(input_dim, [best_params["hidden_1"], best_params["hidden_2"]]).to(DEVICE)
    model.load_state_dict(best_state)

    # Persist checkpoint
    ckpt_path = ARTIFACT_DIR / "pytorch_mlp_best.ckpt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hyperparameters": best_params,
            "input_dim": input_dim,
        },
        ckpt_path,
    )
    print(f"✅ PyTorch MLP checkpoint saved to {ckpt_path}")

    return model, best_params


def pytorch_predict_proba(model: nn.Module, X: np.ndarray, batch_size: int = 64) -> np.ndarray:
    """Return class‑1 probabilities for a NumPy matrix."""
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
# 7️⃣ STACKING META‑LEARNER
# --------------------------------------------------------------------------- #
def train_meta_learner(
    ag_val_proba: np.ndarray,
    pt_val_proba: np.ndarray,
    y_val: pd.Series,
) -> LogisticRegression:
    """
    Train a simple LogisticRegression on the validation‑set predictions of
    AutoGluon and the PyTorch MLP.
    """
    # AutoGluon returns (N, 2); keep probability of class 1
    ag_prob = ag_val_proba[:, 1].reshape(-1, 1)
    pt_prob = pt_val_proba.reshape(-1, 1)

    X_meta = np.hstack([ag_prob, pt_prob])
    meta = LogisticRegression(solver="lbfgs")
    meta.fit(X_meta, y_val)
    meta_path = ARTIFACT_DIR / "stacking_meta.pkl"
    joblib.dump(meta, meta_path)
    print(f"✅ Stacking meta‑learner saved to {meta_path}")
    return meta


def stacked_predict(
    predictor: TabularPredictor,
    pt_model: nn.Module,
    meta: LogisticRegression,
    X: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Produce final probabilities and binary predictions for a pre‑processed
    DataFrame ``X``.
    """
    # AutoGluon probabilities
    ag_proba = predictor.predict_proba(X).values  # (N, 2)

    # PyTorch probabilities
    pt_proba = pytorch_predict_proba(pt_model, X.values.astype(np.float32)).reshape(-1, 1)

    # Meta‑learner input
    X_meta = np.hstack([ag_proba[:, 1].reshape(-1, 1), pt_proba])
    final_proba = meta.predict_proba(X_meta)[:, 1]
    final_pred = (final_proba >= 0.5).astype(int)
    return final_proba, final_pred


# --------------------------------------------------------------------------- #
# 8️⃣ MAIN PIPELINE
# --------------------------------------------------------------------------- #
def main():
    warnings.filterwarnings("ignore")

    # ------------------------------------------------------------------- #
    # Paths & configuration
    # ------------------------------------------------------------------- #
    USER_CSV = "data/raw/titanic.csv"          # <-- adjust if needed
    RAW_PARQUET = "data/raw/titanic.parquet"
    SCHEMA_JSON = "data/schema/titanic_schema.json"
    PREPROCESS_CFG = "configs/preprocess.yaml"
    PREPROCESS_PKL = ARTIFACT_DIR / "preprocess.pkl"

    # ------------------------------------------------------------------- #
    # 1️⃣ Load raw data & write parquet
    # ------------------------------------------------------------------- #
    df_raw = ingest_csv(USER_CSV, RAW_PARQUET)

    # ------------------------------------------------------------------- #
    # 2️⃣ Generate a minimal JSON schema (optional but useful)
    # ------------------------------------------------------------------- #
    generate_schema(df_raw, SCHEMA_JSON)

    # ------------------------------------------------------------------- #
    # 3️⃣ Build & fit preprocessing pipeline
    # ------------------------------------------------------------------- #
    preprocess_pipe, X_raw, y_raw = fit_preprocess_and_save(
        df_raw, PREPROCESS_CFG, PREPROCESS_PKL
    )

    # ------------------------------------------------------------------- #
    # 4️⃣ Train / validation / test split (70/20/10)
    # ------------------------------------------------------------------- #
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test = stratified_split(
        X_raw, y_raw, SEED
    )

    # ------------------------------------------------------------------- #
    # 5️⃣ Transform splits using the fitted pipeline
    # ------------------------------------------------------------------- #
    X_train = preprocess_pipe.transform(X_train_raw)
    X_val = preprocess_pipe.transform(X_val_raw)
    X_test = preprocess_pipe.transform(X_test_raw)

    # Convert to DataFrames with generic column names (required by AutoGluon)
    col_names = [f"f{i}" for i in range(X_train.shape[1])]
    X_train_df = pd.DataFrame(X_train, columns=col_names)
    X_val_df = pd.DataFrame(X_val, columns=col_names)
    X_test_df = pd.DataFrame(X_test, columns=col_names)

    # ------------------------------------------------------------------- #
    # 6️⃣ AutoGluon tree ensemble (no internal NN)
    # ------------------------------------------------------------------- #
    predictor = train_autogluon(
        X_train_df, y_train, X_val_df, y_val, time_limit=1800
    )

    # ------------------------------------------------------------------- #
    # 7️⃣ Train custom PyTorch MLP (Optuna HPO)
    # ------------------------------------------------------------------- #
    pt_model, best_hparams = train_pytorch_mlp(
        X_train, y_train, X_val, y_val, max_trials=30, time_budget=600
    )

    # ------------------------------------------------------------------- #
    # 8️⃣ Train stacking meta‑learner on validation predictions
    # ------------------------------------------------------------------- #
    ag_val_proba = predictor.predict_proba(X_val_df).values
    pt_val_proba = pytorch_predict_proba(pt_model, X_val)

    meta = train_meta_learner(ag_val_proba, pt_val_proba, y_val)

    # ------------------------------------------------------------------- #
    # 9️⃣ Final evaluation on the held‑out test set
    # ------------------------------------------------------------------- #
    test_proba, test_pred = stacked_predict(predictor, pt_model, meta, X_test_df)

    test_auc = roc_auc_score(y_test, test_proba)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred)

    print("\n=== TEST SET METRICS ===")
    print(f"ROC‑AUC : {test_auc:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"F1      : {test_f1:.4f}")

    # ------------------------------------------------------------------- #
    # 🔟 Log artefacts & metrics to MLflow (optional but recommended)
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
        mlflow.log_artifact(str(PREPROCESS_PKL), artifact_path="pipeline")
        mlflow.log_artifact(str(ARTIFACT_DIR / "autogluon"), artifact_path="autogluon")
        mlflow.log_artifact(str(ARTIFACT_DIR / "pytorch_mlp_best.ckpt"), artifact_path="pytorch")
        mlflow.log_artifact(str(ARTIFACT_DIR / "stacking_meta.pkl"), artifact_path="stacking")

    print(f"\n✅ All artefacts are stored under {ARTIFACT_DIR.resolve()}")


if __name__ == "__main__":
    main()
