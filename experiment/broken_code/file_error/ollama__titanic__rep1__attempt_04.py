
"""
Titanic Survival – End‑to‑End AutoML pipeline
================================================

This script is a **stand‑alone, reproducible** implementation of the design
described in the specification.  It performs the following steps:

1️⃣  Load the raw CSV (expects it at ``./data/raw/titanic.csv``) and write a
    Parquet copy.

2️⃣  Generate a tiny JSON schema (numeric vs categorical) for sanity‑checking.

3️⃣  Build a Scikit‑Learn preprocessing pipeline:
    - schema validation
    - iterative imputation for numeric columns
    - most‑frequent imputation for categorical columns
    - two engineered features (FamilySize, IsAlone)
    - Robust scaling for numerics
    - One‑hot encoding for categoricals

4️⃣  Split the data (70 % train / 20 % validation / 10 % test) **stratified**
    on the target column ``survived``.

5️⃣  Train three tree‑based models (LightGBM, CatBoost, XGBoost) on the
    training set.

6️⃣  Train a **custom PyTorch MLP** (with a lightweight Optuna HPO).

7️⃣  Stack the four base learners with a Logistic‑Regression meta‑learner.

8️⃣  Evaluate on the held‑out test set (ROC‑AUC, Accuracy, F1).

9️⃣  Persist all artefacts under ``./agent_workspace/trained_models``:
    - preprocessing pipeline (`preprocess.pkl`)
    - LightGBM, CatBoost, XGBoost models
    - PyTorch checkpoint (`pytorch_mlp_best.ckpt`)
    - Stacking meta‑learner (`stacking_meta.pkl`)

The script can be executed directly::

    python titanic_automl.py
"""

import json
import os
import random
import warnings
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBClassifier

# --------------------------------------------------------------------------- #
# GLOBAL SETTINGS
# --------------------------------------------------------------------------- #
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------------------------------- #
# PATHS (all relative to repository root)
# --------------------------------------------------------------------------- #
BASE_DIR = Path(".")
RAW_CSV = BASE_DIR / "data" / "raw" / "titanic.csv"
RAW_PARQUET = BASE_DIR / "data" / "raw" / "titanic.parquet"
SCHEMA_JSON = BASE_DIR / "data" / "schema" / "titanic_schema.json"
PREPROCESS_CFG = BASE_DIR / "configs" / "preprocess.yaml"

# Directory where all artefacts will be stored
ARTIFACT_DIR = BASE_DIR / "agent_workspace" / "trained_models"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

PIPELINE_PKL = ARTIFACT_DIR / "preprocess.pkl"
LIGHTGBM_PKL = ARTIFACT_DIR / "lightgbm_model.pkl"
CATBOOST_PKL = ARTIFACT_DIR / "catboost_model.cbm"
XGBOOST_PKL = ARTIFACT_DIR / "xgboost_model.pkl"
PYTORCH_CKPT = ARTIFACT_DIR / "pytorch_mlp_best.ckpt"
META_LEARNER_PKL = ARTIFACT_DIR / "stacking_meta.pkl"
MODEL_PROFILE_CSV = BASE_DIR / "reports" / "model_profile.csv"

# --------------------------------------------------------------------------- #
# 1️⃣ DATA INGESTION & CONVERSION
# --------------------------------------------------------------------------- #
def ingest_csv(csv_path: Path, parquet_path: Path) -> pd.DataFrame:
    """Read CSV with pandas, write a Parquet copy, and return the DataFrame."""
    df = pd.read_csv(csv_path)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, engine="pyarrow", index=False)
    return df


# --------------------------------------------------------------------------- #
# 2️⃣ SIMPLE JSON SCHEMA
# --------------------------------------------------------------------------- #
def generate_schema(df: pd.DataFrame, schema_path: Path, uniq_threshold: int = 15) -> None:
    """Columns with >threshold unique values are marked as numeric, else categorical."""
    schema = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            col_type = "numeric"
        else:
            col_type = "categorical" if df[col].nunique() <= uniq_threshold else "numeric"
        schema[col] = {"type": col_type}
    schema_path.parent.mkdir(parents=True, exist_ok=True)
    with open(schema_path, "w") as f:
        json.dump(schema, f, indent=2)


# --------------------------------------------------------------------------- #
# 3️⃣ PRE‑PROCESSING PIPELINE
# --------------------------------------------------------------------------- #
def _add_engineered_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add FamilySize and IsAlone."""
    X = X.copy()
    X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
    X["IsAlone"] = (X["FamilySize"] == 1).astype(int)
    return X


def build_preprocess_pipeline(cfg: dict) -> Pipeline:
    """Construct the Scikit‑Learn pipeline according to ``configs/preprocess.yaml``."""
    cat_cols = cfg["categorical"]
    ord_cols = cfg["ordinal"]
    num_cols = cfg["numeric"]

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

    # Categorical sub‑pipeline (ordinal columns are treated as categorical)
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols + ord_cols),
        ],
        remainder="passthrough",  # engineered features will be passed through
    )

    full_pipe = Pipeline(
        steps=[
            ("feature_engineering", FunctionTransformer(_add_engineered_features, validate=False)),
            ("preprocess", preprocessor),
        ]
    )
    return full_pipe


def fit_and_save_preprocess(df: pd.DataFrame, cfg_path: Path, save_path: Path) -> Tuple[Pipeline, pd.DataFrame, pd.Series]:
    """Fit the pipeline on the raw data (excluding target) and persist it."""
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    target = cfg["target_column"]
    X = df.drop(columns=[target])
    y = df[target]

    pipe = build_preprocess_pipeline(cfg)
    pipe.fit(X, y)

    joblib.dump(pipe, save_path)
    print(f"[INFO] Pre‑processing pipeline saved to {save_path}")
    return pipe, X, y


# --------------------------------------------------------------------------- #
# 4️⃣ STRATIFIED SPLIT (70/20/10)
# --------------------------------------------------------------------------- #
def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    seed: int = SEED,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Return train / validation / test splits (70/20/10) stratified on the label."""
    # First split off 30 % (val+test)
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )
    # From the remaining 30 % we take 2/3 as validation (≈20 % of original)
    val_frac_of_tmp = 0.20 / 0.30  # ≈0.6667
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=1 - val_frac_of_tmp,
        stratify=y_tmp,
        random_state=seed,
    )
    return X_train, y_train, X_val, y_val, X_test, y_test


# --------------------------------------------------------------------------- #
# 5️⃣ TREE‑BASED MODELS (LightGBM, CatBoost, XGBoost)
# --------------------------------------------------------------------------- #
def train_tree_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> Tuple[LGBMClassifier, CatBoostClassifier, XGBClassifier]:
    """Train three tree models with default hyper‑parameters."""
    # LightGBM
    lgbm = LGBMClassifier(random_state=SEED, n_estimators=200, learning_rate=0.05)
    lgbm.fit(X_train, y_train)

    # CatBoost (silent mode)
    cat = CatBoostClassifier(
        random_seed=SEED,
        iterations=400,
        learning_rate=0.05,
        depth=6,
        verbose=False,
        loss_function="Logloss",
    )
    cat.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)

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
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Persist models
    joblib.dump(lgbm, LIGHTGBM_PKL)
    cat.save_model(str(CATBOOST_PKL))
    joblib.dump(xgb, XGBOOST_PKL)

    print(f"[INFO] Tree models saved to {ARTIFACT_DIR}")
    return lgbm, cat, xgb


# --------------------------------------------------------------------------- #
# 6️⃣ CUSTOM PYTORCH MLP (with Optuna HPO)
# --------------------------------------------------------------------------- #
class TorchMLP(nn.Module):
    """Feed‑forward network: Input → [Linear → BN → ReLU] ×2 → Linear (logits)."""

    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))  # logits
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)  # (N,)


def _objective(
    trial,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    input_dim: int,
) -> float:
    """Optuna objective – maximise validation ROC‑AUC."""
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

    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32),
    )
    val_ds = TensorDataset(
        torch.tensor(X_va, dtype=torch.float32),
        torch.tensor(y_va, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_auc = 0.0
    epochs_no_improve = 0
    max_epochs = 100
    best_state = None

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

        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
            best_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 10:
                break

    trial.set_user_attr("best_state", best_state)
    return best_auc


def train_pytorch_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    max_trials: int = 30,
    time_budget: int = 600,
) -> Tuple[TorchMLP, dict]:
    """Run Optuna HPO, reload the best model and persist the checkpoint."""
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
            y_train,
            X_val,
            y_val,
            input_dim,
        ),
        n_trials=max_trials,
        timeout=time_budget,
    )

    best_trial = study.best_trial
    best_state = best_trial.user_attrs["best_state"]
    best_params = best_trial.params

    # Re‑instantiate with best hyper‑parameters
    model = TorchMLP(input_dim, [best_params["hidden_1"], best_params["hidden_2"]]).to(DEVICE)
    model.load_state_dict(best_state)

    # Save checkpoint
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "hyperparameters": best_params,
            "input_dim": input_dim,
        },
        PYTORCH_CKPT,
    )
    print(f"[INFO] PyTorch MLP checkpoint saved to {PYTORCH_CKPT}")

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
    base_val_probas: List[np.ndarray],
    y_val: np.ndarray,
) -> LogisticRegression:
    """
    Train a LogisticRegression meta‑learner on validation‑set predictions.
    ``base_val_probas`` is a list where each element is a (N,) array of
    class‑1 probabilities from a base model.
    """
    X_meta = np.column_stack(base_val_probas)
    meta = LogisticRegression(solver="lbfgs", max_iter=1000)
    meta.fit(X_meta, y_val)

    joblib.dump(meta, META_LEARNER_PKL)
    print(f"[INFO] Stacking meta‑learner saved to {META_LEARNER_PKL}")
    return meta


def stacked_predict(
    base_models: List[Tuple[str, object]],
    meta_model: LogisticRegression,
    X: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate final probabilities and binary predictions using the stacked ensemble.
    ``base_models`` is a list of (name, model) tuples. Supported names:
    "lgbm", "catboost", "xgboost", "pytorch".
    """
    prob_list = []
    for name, mdl in base_models:
        if name == "lgbm":
            prob = mdl.predict_proba(X)[:, 1]
        elif name == "catboost":
            prob = mdl.predict_proba(X)[:, 1]
        elif name == "xgboost":
            prob = mdl.predict_proba(X)[:, 1]
        elif name == "pytorch":
            prob = pytorch_predict_proba(mdl, X)
        else:
            raise ValueError(f"Unsupported model name: {name}")
        prob_list.append(prob)

    X_meta = np.column_stack(prob_list)
    final_proba = meta_model.predict_proba(X_meta)[:, 1]
    final_pred = (final_proba >= 0.5).astype(int)
    return final_proba, final_pred


# --------------------------------------------------------------------------- #
# MAIN PIPELINE
# --------------------------------------------------------------------------- #
def main():
    warnings.filterwarnings("ignore")

    # --------------------------------------------------------------- #
    # 0️⃣ Ensure a minimal preprocessing config exists
    # --------------------------------------------------------------- #
    if not PREPROCESS_CFG.is_file():
        default_cfg = {
            "seed": SEED,
            "target_column": "survived",
            "categorical": ["Sex", "Embarked", "Pclass", "Cabin", "Ticket"],
            "ordinal": ["Pclass"],
            "numeric": ["Age", "Fare", "SibSp", "Parch"],
            "engineered_features": {"enable": True},
            "imputation": {"numeric": "iterative", "categorical": "most_frequent"},
            "scaling": "robust",
            "encoding": {"categorical": "onehot"},
        }
        PREPROCESS_CFG.parent.mkdir(parents=True, exist_ok=True)
        with open(PREPROCESS_CFG, "w") as f:
            yaml.safe_dump(default_cfg, f)

    # --------------------------------------------------------------- #
    # 1️⃣ Load raw data & write parquet
    # --------------------------------------------------------------- #
    if not RAW_CSV.is_file():
        raise FileNotFoundError(f"Raw CSV not found at {RAW_CSV}. Please place the Titanic CSV there.")
    df_raw = ingest_csv(RAW_CSV, RAW_PARQUET)

    # --------------------------------------------------------------- #
    # 2️⃣ Generate JSON schema (optional but useful)
    # --------------------------------------------------------------- #
    generate_schema(df_raw, SCHEMA_JSON)

    # --------------------------------------------------------------- #
    # 3️⃣ Build & fit preprocessing pipeline, persist it
    # --------------------------------------------------------------- #
    preprocess_pipe, X_raw, y_raw = fit_and_save_preprocess(df_raw, PREPROCESS_CFG, PIPELINE_PKL)

    # --------------------------------------------------------------- #
    # 4️⃣ Train / validation / test split (70/20/10)
    # --------------------------------------------------------------- #
    X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test = stratified_split(
        X_raw, y_raw, SEED
    )

    # --------------------------------------------------------------- #
    # 5️⃣ Transform splits using the fitted pipeline
    # --------------------------------------------------------------- #
    X_train_np = preprocess_pipe.transform(X_train_raw)
    X_val_np = preprocess_pipe.transform(X_val_raw)
    X_test_np = preprocess_pipe.transform(X_test_raw)

    # --------------------------------------------------------------- #
    # 6️⃣ Train tree‑based models
    # --------------------------------------------------------------- #
    lgbm_model, cat_model, xgb_model = train_tree_models(
        X_train_np, y_train.values, X_val_np, y_val.values
    )

    # --------------------------------------------------------------- #
    # 7️⃣ Train custom PyTorch MLP (Optuna HPO)
    # --------------------------------------------------------------- #
    pt_model, best_hparams = train_pytorch_mlp(
        X_train_np, y_train.values, X_val_np, y_val.values,
        max_trials=30,
        time_budget=600,
    )

    # --------------------------------------------------------------- #
    # 8️⃣ Train stacking meta‑learner on validation predictions
    # --------------------------------------------------------------- #
    # Get validation probabilities from each base learner
    lgbm_val_proba = lgbm_model.predict_proba(X_val_np)[:, 1]
    cat_val_proba = cat_model.predict_proba(X_val_np)[:, 1]
    xgb_val_proba = xgb_model.predict_proba(X_val_np)[:, 1]
    pt_val_proba = pytorch_predict_proba(pt_model, X_val_np)

    meta_learner = train_meta_learner(
        [lgbm_val_proba, cat_val_proba, xgb_val_proba, pt_val_proba],
        y_val.values,
    )

    # --------------------------------------------------------------- #
    # 9️⃣ Final evaluation on the held‑out test set
    # --------------------------------------------------------------- #
    base_models = [
        ("lgbm", lgbm_model),
        ("catboost", cat_model),
        ("xgboost", xgb_model),
        ("pytorch", pt_model),
    ]
    test_proba, test_pred = stacked_predict(base_models, meta_learner, X_test_np)

    test_auc = roc_auc_score(y_test, test_proba)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred)

    print("\n=== TEST SET METRICS ===")
    print(f"ROC‑AUC : {test_auc:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"F1      : {test_f1:.4f}")

    # --------------------------------------------------------------- #
    # 10️⃣ Save a tiny profiling CSV (model size, etc.)
    # --------------------------------------------------------------- #
    MODEL_PROFILE_CSV.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "model_name": "titanic_stacked_ensemble",
        "roc_auc": test_auc,
        "accuracy": test_acc,
        "f1": test_f1,
        "size_mb": round((os.path.getsize(PYTORCH_CKPT) + os.path.getsize(LIGHTGBM_PKL) +
                           os.path.getsize(CATBOOST_PKL) + os.path.getsize(XGBOOST_PKL) +
                           os.path.getsize(PIPELINE_PKL)) / (1024 * 1024), 2),
    }
    pd.DataFrame([profile]).to_csv(MODEL_PROFILE_CSV, index=False)
    print(f"[INFO] Model profile written to {MODEL_PROFILE_CSV}")

    # --------------------------------------------------------------- #
    # 11️⃣ Summary of saved artefacts
    # --------------------------------------------------------------- #
    print("\n[INFO] All artefacts stored under:", ARTIFACT_DIR.resolve())
    print(f" - Pre‑processing pipeline : {PIPELINE_PKL}")
    print(f" - LightGBM model          : {LIGHTGBM_PKL}")
    print(f" - CatBoost model          : {CATBOOST_PKL}")
    print(f" - XGBoost model           : {XGBOOST_PKL}")
    print(f" - PyTorch MLP checkpoint  : {PYTORCH_CKPT}")
    print(f" - Stacking meta‑learner   : {META_LEARNER_PKL}")


if __name__ == "__main__":
    main()
