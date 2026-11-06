
# =============================================================================
# titanic_pipeline.py
# =============================================================================
# End‑to‑end Titanic survival pipeline:
#   1️⃣ Load configuration (config.yaml) – searched upward from this file.
#   2️⃣ Load raw CSV, resolve target column (case‑insensitive).
#   3️⃣ Cast numeric columns, engineer features.
#   4️⃣ Stratified 70/20/10 train/valid/test split.
#   5️⃣ Build a Scikit‑Learn preprocessing pipeline (impute → scale → encode).
#   6️⃣ Fit pipeline on train+valid, transform all splits, persist them.
#   7️⃣ Train LightGBM with the exact hyper‑parameters from the play‑book.
#   8️⃣ Evaluate on the test split (ROC‑AUC, Accuracy, F1, Brier score).
#   9️⃣ Persist artefacts:
#        - preprocessing pipeline (joblib)
#        - feature order (json)
#        - LightGBM model (joblib)
#        - copy LightGBM model to ./agent_workspace/trained_models/
#        - ONNX export + TorchScript wrapper
# =============================================================================

import os
import json
import pathlib
import warnings
import random
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import joblib
import yaml

# -------------------- 3rd‑party ML libraries --------------------
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    brier_score_loss,
)

# ONNX / Torch imports – optional fallback if the packages are missing
try:
    import onnx
    import onnxruntime as ort
except Exception:  # pragma: no cover
    onnx = None
    ort = None

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None


# =============================================================================
# 0️⃣ Helper utilities
# =============================================================================
def set_random_seed(seed: int = 42) -> None:
    """Make the whole pipeline deterministic."""
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_target_column(df: pd.DataFrame, target_name: str = "Survived") -> str:
    """Return the exact column name from the CSV (case‑insensitive)."""
    mapping = {c.lower(): c for c in df.columns}
    if target_name.lower() not in mapping:
        raise KeyError(f"Target column '{target_name}' not found in CSV.")
    return mapping[target_name.lower()]


def extract_title(name: str) -> str:
    """Extract title (Mr., Mrs., etc.) from the passenger name."""
    if pd.isna(name):
        return "Unknown"
    parts = name.split(",")
    if len(parts) < 2:
        return "Unknown"
    title_part = parts[1].strip()
    return title_part.split(".")[0].strip()


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create FamilySize, Title, LogFare and drop raw columns that are not needed."""
    df = df.copy()

    # FamilySize = SibSp + Parch + 1
    if {"SibSp", "Parch"}.issubset(df.columns):
        df["FamilySize"] = df["SibSp"].astype(float) + df["Parch"].astype(float) + 1

    # Title from Name
    if "Name" in df.columns:
        df["Title"] = df["Name"].apply(extract_title)

    # LogFare (guard against zero / missing)
    if "Fare" in df.columns:
        df["LogFare"] = np.log1p(df["Fare"].astype(float))

    # Drop columns that we will not feed directly to the model
    drop_cols = ["Name", "Ticket", "Cabin"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")
    return df


def split_data(
    df: pd.DataFrame,
    target_col: str,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame,
           pd.Series, pd.Series, pd.Series]:
    """Stratified 70/20/10 split (train / valid / test)."""
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=seed,
    )
    # Remaining 30 % → 20 % validation, 10 % test (2:1 ratio)
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=1 / 3,
        stratify=y_tmp,
        random_state=seed,
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


# ------------------------------------------------------------------
# Simple fallback TargetEncoder (used when category_encoders is unavailable)
# ------------------------------------------------------------------
class SimpleTargetEncoder:
    """Very small target encoder that works inside a ColumnTransformer."""

    def __init__(self):
        self.mapping_: Dict[str, Dict] = {}
        self.global_mean_: float = 0.0

    def fit(self, X, y):
        # X may be a DataFrame or a Series
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
        y_numeric = pd.to_numeric(y, errors="coerce")
        self.global_mean_ = y_numeric.mean()

        for col in X.columns:
            df = pd.DataFrame({col: X[col], "target": y_numeric})
            self.mapping_[col] = (
                df.groupby(col)["target"]
                .mean()
                .fillna(self.global_mean_)
                .to_dict()
            )
        return self

    def transform(self, X):
        X_out = X.copy()
        for col in X_out.columns:
            X_out[col] = X_out[col].map(self.mapping_.get(col, {})).fillna(self.global_mean_)
        return X_out.values.astype(np.float32)


# ------------------------------------------------------------------
# Age binning transformer (creates 5 quantile bins)
# ------------------------------------------------------------------
class AgeBinner:
    """Creates 5 quantile bins for Age (if Age exists)."""

    def __init__(self, n_bins: int = 5):
        self.n_bins = n_bins
        self.bin_edges_: np.ndarray = None

    def fit(self, X, y=None):
        if "Age" in X.columns:
            ages = X["Age"].astype(float).dropna()
            if not ages.empty:
                self.bin_edges_ = np.quantile(
                    ages, q=np.linspace(0, 1, self.n_bins + 1)
                )
        return self

    def transform(self, X):
        X = X.copy()
        if "Age" in X.columns and self.bin_edges_ is not None:
            X["AgeBin"] = pd.cut(
                X["Age"].astype(float),
                bins=self.bin_edges_,
                labels=False,
                duplicates="drop",
            ).astype(float)
        return X


def build_preprocess_pipeline(
    numeric_features: List[str],
    low_card_cat: List[str],
    high_card_cat: List[str],
) -> Pipeline:
    """Construct the Scikit‑Learn preprocessing pipeline."""
    # ---- numeric sub‑pipeline -------------------------------------------------
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    # ---- low‑cardinality categorical (One‑Hot) -------------------------------
    low_cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # ---- high‑cardinality categorical (Target Encoding) --------------------
    high_cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("target_enc", SimpleTargetEncoder()),
        ]
    )

    # ---- ColumnTransformer ---------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("lowcat", low_cat_pipe, low_card_cat),
            ("highcat", high_cat_pipe, high_card_cat),
        ],
        remainder="drop",
    )

    # ---- Full pipeline (AgeBinner first) ------------------------------------
    full_pipe = Pipeline(
        steps=[
            ("age_bin", AgeBinner()),
            ("preprocess", preprocessor),
        ]
    )
    return full_pipe


def get_feature_names_from_ct(ct: ColumnTransformer) -> List[str]:
    """Re‑create the feature order after a fitted ColumnTransformer."""
    out: List[str] = []
    for name, transformer, cols in ct.transformers_:
        if name == "remainder":
            continue
        # If the transformer is a Pipeline, fetch the last step
        if hasattr(transformer, "named_steps"):
            last_step = list(transformer.named_steps.values())[-1]
            if hasattr(last_step, "get_feature_names_out"):
                names = last_step.get_feature_names_out(cols)
            else:
                names = cols
        else:
            names = cols
        out.extend(list(names))
    return out


def train_lightgbm(
    X_train,
    y_train,
    X_valid,
    y_valid,
    params: Dict,
) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        early_stopping_rounds=params.get("early_stopping_rounds", 20),
        verbose=False,
    )
    return model


def evaluate_model(
    model: lgb.LGBMClassifier,
    X,
    y_true: pd.Series,
) -> Dict[str, float]:
    prob = model.predict_proba(X)[:, 1]
    pred = (prob >= 0.5).astype(int)

    return {
        "roc_auc": roc_auc_score(y_true, prob),
        "accuracy": accuracy_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "brier_score": brier_score_loss(y_true, prob),
    }


def export_to_onnx(
    model: lgb.LGBMClassifier,
    feature_order: List[str],
    onnx_path: pathlib.Path,
) -> None:
    """Export LightGBM model to ONNX (CPU)."""
    if onnx is None:
        raise RuntimeError("ONNX package not installed.")
    booster = model.booster_
    booster.save_model(str(onnx_path), format="onnx")

    # Verify / fix input name
    onnx_model = onnx.load_model(str(onnx_path))
    input_name = onnx_model.graph.input[0].name
    if input_name != feature_order[0]:
        onnx_model.graph.input[0].name = feature_order[0]
        onnx.save_model(onnx_model, str(onnx_path))


class TorchScriptWrapper(nn.Module):
    """Tiny wrapper that runs an ONNX model via onnxruntime inside TorchScript."""

    def __init__(self, onnx_path: str):
        super().__init__()
        if ort is None:
            raise RuntimeError("onnxruntime not installed.")
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        np_input = x.detach().cpu().numpy().astype(np.float32)
        if np_input.ndim == 1:
            np_input = np_input.reshape(1, -1)
        ort_out = self.session.run(None, {self.input_name: np_input})
        prob = ort_out[0].squeeze()
        return torch.from_numpy(prob.astype(np.float32))


# =============================================================================
# 1️⃣ Main orchestration
# =============================================================================
def find_repo_root(start_path: pathlib.Path) -> pathlib.Path:
    """Search upward from start_path until a config.yaml is found."""
    cur = start_path.resolve()
    while True:
        if (cur / "config.yaml").is_file():
            return cur
        if cur.parent == cur:
            raise FileNotFoundError("Could not locate config.yaml in any parent directory.")
        cur = cur.parent


def main() -> None:
    # ------------------------------------------------------------------
    # Locate repository root (where config.yaml lives)
    # ------------------------------------------------------------------
    repo_root = find_repo_root(pathlib.Path(__file__))
    cfg_path = repo_root / "config.yaml"
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Set deterministic seed
    # ------------------------------------------------------------------
    set_random_seed(cfg["resources"]["seed"])

    # ------------------------------------------------------------------
    # Load raw CSV (as strings → later cast numeric columns)
    # ------------------------------------------------------------------
    raw_csv_path = pathlib.Path(cfg["paths"]["raw_csv"])
    if not raw_csv_path.is_file():
        raise FileNotFoundError(f"Raw CSV not found at {raw_csv_path}")
    df_raw = pd.read_csv(raw_csv_path, dtype=str)

    # ------------------------------------------------------------------
    # Resolve exact target column name (case‑insensitive)
    # ------------------------------------------------------------------
    target_col = resolve_target_column(df_raw, "Survived")

    # ------------------------------------------------------------------
    # Cast known numeric columns to proper dtypes
    # ------------------------------------------------------------------
    numeric_cols = ["Age", "Fare", "SibSp", "Parch", "Pclass"]
    for col in numeric_cols:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    # ------------------------------------------------------------------
    # Feature engineering (FamilySize, Title, LogFare, drop raw cols)
    # ------------------------------------------------------------------
    df_fe = add_engineered_features(df_raw)

    # ------------------------------------------------------------------
    # Train/valid/test split (stratified)
    # ------------------------------------------------------------------
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(
        df_fe, target_col, seed=cfg["resources"]["seed"]
    )

    # ------------------------------------------------------------------
    # Identify numeric vs categorical columns for the pipeline
    # ------------------------------------------------------------------
    all_features = X_train.columns.tolist()

    numeric_features = [
        c for c in all_features if X_train[c].dtype.kind in "iufc"
    ]

    # Ensure engineered numeric columns are present
    for extra in ["FamilySize", "LogFare", "AgeBin"]:
        if extra in X_train.columns and extra not in numeric_features:
            numeric_features.append(extra)

    categorical_features = [c for c in all_features if c not in numeric_features]

    low_card_cat = [c for c in categorical_features if X_train[c].nunique() <= 9]
    high_card_cat = [c for c in categorical_features if X_train[c].nunique() > 9]

    # ------------------------------------------------------------------
    # Build preprocessing pipeline
    # ------------------------------------------------------------------
    pipeline = build_preprocess_pipeline(numeric_features, low_card_cat, high_card_cat)

    # Fit on train + valid (maximise data for imputation/encoding)
    pipeline.fit(
        pd.concat([X_train, X_valid], axis=0),
        pd.concat([y_train, y_valid], axis=0),
    )

    # Transform each split
    X_train_enc = pipeline.transform(X_train)
    X_valid_enc = pipeline.transform(X_valid)
    X_test_enc = pipeline.transform(X_test)

    # ------------------------------------------------------------------
    # Retrieve feature order after ColumnTransformer (needed for ONNX)
    # ------------------------------------------------------------------
    feature_order = get_feature_names_from_ct(pipeline.named_steps["preprocess"])

    # ------------------------------------------------------------------
    # LightGBM hyper‑parameters (exactly as prescribed)
    # ------------------------------------------------------------------
    lgb_params = {
        "learning_rate": 0.07,
        "num_leaves": 31,
        "max_depth": -1,
        "n_estimators": 300,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l2": 0.5,
        "objective": "binary",
        "metric": "auc",
        "random_state": cfg["resources"]["seed"],
        "n_jobs": cfg["resources"]["n_jobs"],
        "early_stopping_rounds": 20,
    }

    # ------------------------------------------------------------------
    # Train LightGBM
    # ------------------------------------------------------------------
    model = train_lightgbm(
        X_train_enc,
        y_train,
        X_valid_enc,
        y_valid,
        lgb_params,
    )

    # ------------------------------------------------------------------
    # Evaluate on the test split
    # ------------------------------------------------------------------
    test_metrics = evaluate_model(model, X_test_enc, y_test)
    print("\n🔎 Test set metrics")
    for k, v in test_metrics.items():
        print(f"  {k:12}: {v:.4f}")

    # ------------------------------------------------------------------
    # Helper to ensure a directory exists
    # ------------------------------------------------------------------
    def ensure_dir(p: pathlib.Path) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Persist artefacts (paths are relative to repo_root)
    # ------------------------------------------------------------------
    pipeline_path = repo_root / cfg["paths"]["pipeline"]
    feature_order_path = repo_root / cfg["paths"]["feature_order"]
    lgb_model_path = repo_root / cfg["paths"]["lightgbm_model"]
    onnx_path = repo_root / cfg["paths"]["onnx_model"]
    torch_path = repo_root / cfg["paths"]["torch_script"]
    splits_dir = repo_root / cfg["paths"]["splits_dir"]
    trained_models_dir = repo_root / "agent_workspace" / "trained_models"

    # 1️⃣  Pipeline
    ensure_dir(pipeline_path)
    joblib.dump(pipeline, pipeline_path)
    print(f"✅ Preprocessing pipeline saved → {pipeline_path}")

    # 2️⃣  Feature order
    ensure_dir(feature_order_path)
    with open(feature_order_path, "w") as f:
        json.dump(feature_order, f)
    print(f"✅ Feature order saved → {feature_order_path}")

    # 3️⃣  LightGBM model
    ensure_dir(lgb_model_path)
    joblib.dump(model, lgb_model_path)
    print(f"✅ LightGBM model saved → {lgb_model_path}")

    # 4️⃣  Copy LightGBM model to the user‑requested folder
    trained_models_dir.mkdir(parents=True, exist_ok=True)
    copy_dest = trained_models_dir / "lightgbm_best.pkl"
    joblib.dump(model, copy_dest)
    print(f"✅ Model also copied to → {copy_dest}")

    # 5️⃣  ONNX export
    ensure_dir(onnx_path)
    export_to_onnx(model, feature_order, onnx_path)
    print(f"✅ ONNX model exported → {onnx_path}")

    # 6️⃣  TorchScript wrapper
    if torch is None:
        raise RuntimeError("PyTorch not installed – cannot create TorchScript wrapper.")
    ensure_dir(torch_path)
    wrapper = TorchScriptWrapper(str(onnx_path))
    scripted = torch.jit.script(wrapper)
    scripted.save(str(torch_path))
    print(f"✅ TorchScript wrapper saved → {torch_path}")

    # ------------------------------------------------------------------
    # Save transformed splits as Parquet (optional but useful)
    # ------------------------------------------------------------------
    splits_dir.mkdir(parents=True, exist_ok=True)
    for name, X_enc, y in [
        ("train", X_train_enc, y_train),
        ("valid", X_valid_enc, y_valid),
        ("test", X_test_enc, y_test),
    ]:
        df_out = pd.DataFrame(X_enc, columns=feature_order)
        df_out[target_col] = y.values
        parquet_path = splits_dir / f"{name}.parquet"
        df_out.to_parquet(parquet_path, index=False)
        print(f"✅ {name.capitalize()} split written → {parquet_path}")

    # ------------------------------------------------------------------
    # Done!
    # ------------------------------------------------------------------
    print("\n🚀 Titanic pipeline completed successfully!")


if __name__ == "__main__":
    # Suppress non‑critical warnings (e.g., LightGBM on Windows)
    warnings.filterwarnings("ignore")
    main()
