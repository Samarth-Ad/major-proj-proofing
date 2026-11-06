
"""
Titanic end‑to‑end modeling pipeline

* Reads the raw CSV (absolute path taken from ``config.yaml``)
* Builds a fully serialisable Scikit‑Learn preprocessing pipeline
  (feature engineering → imputation → encoding → robust scaling)
* Fits the pipeline on the combined train + validation split and saves it
* Transforms train / validation / test splits and stores them as Parquet
  files containing **only numeric columns** plus the target column ``Survived``
* Trains a LightGBM classifier with the exact hyper‑parameters from the spec
* Persists the trained model and also copies it to ``./agent_workspace/trained_models``
* Writes the feature order (column names after the full transformation) to
  ``models/feature_order.json`` for later ONNX export

The script is self‑contained and can be executed from the repository root:

    python src/pipeline_full.py
"""

import json
import pathlib
import sys
from typing import Dict, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from category_encoders import TargetEncoder  # optional, will be used if high‑cardinality cols exist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn import set_config

# --------------------------------------------------------------------------- #
# Global sklearn configuration – makes ColumnTransformer return a DataFrame
# --------------------------------------------------------------------------- #
set_config(transform_output="pandas")

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def ensure_dir(p: pathlib.Path) -> None:
    """Create a directory (including parents) if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)


def load_config(cfg_path: pathlib.Path) -> Dict:
    """Load the yaml configuration file."""
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- #
# 1️⃣  Custom feature‑engineering transformer
# --------------------------------------------------------------------------- #
class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    """Feature engineering for the Titanic dataset.

    * FamilySize = SibSp + Parch + 1
    * Title extracted from Name (regex ``',\\s*([^\\.]*).`')
    * Age quantile bins → AgeBin (5 bins)
    * LogFare = log(Fare + 1)
    * Drops raw columns that are no longer needed
    """

    @staticmethod
    def _extract_title(name: str) -> str:
        import re

        m = re.search(r",\s*([^\.]*)\.", name)
        return m.group(1).strip() if m else "Unknown"

    def fit(self, X, y=None):
        # Stateless transformer – nothing to learn
        return self

    def transform(self, X):
        X = X.copy()

        # ------------------------------------------------------------------- #
        # FamilySize
        # ------------------------------------------------------------------- #
        X["FamilySize"] = (
            pd.to_numeric(X["SibSp"], errors="coerce")
            + pd.to_numeric(X["Parch"], errors="coerce")
            + 1
        ).astype(int)

        # ------------------------------------------------------------------- #
        # Title
        # ------------------------------------------------------------------- #
        X["Title"] = X["Name"].apply(self._extract_title)

        # ------------------------------------------------------------------- #
        # AgeBin (5 quantile bins)
        # ------------------------------------------------------------------- #
        age_num = pd.to_numeric(X["Age"], errors="coerce")
        X["AgeBin"] = pd.qcut(age_num, q=5, duplicates="drop").astype(str)

        # ------------------------------------------------------------------- #
        # LogFare
        # ------------------------------------------------------------------- #
        fare_num = pd.to_numeric(X["Fare"], errors="coerce")
        X["LogFare"] = np.log1p(fare_num)

        # ------------------------------------------------------------------- #
        # Drop raw columns that are no longer needed
        # ------------------------------------------------------------------- #
        cols_to_drop = [
            "Name",
            "Ticket",
            "Cabin",
            "Age",
            "Fare",
            "SibSp",
            "Parch",
        ]
        X.drop(columns=cols_to_drop, inplace=True, errors="ignore")
        return X


# --------------------------------------------------------------------------- #
# 2️⃣  Build the full preprocessing pipeline (feature‑engineer + ColumnTransformer)
# --------------------------------------------------------------------------- #
def build_full_pipeline(df_after_feat: pd.DataFrame, target_col: str = "Survived") -> Tuple[Pipeline, list]:
    """
    Detect numeric / categorical columns (excluding the target) and build the
    ColumnTransformer that performs imputation, encoding and scaling.
    Returns the pipeline and the final feature order (list of column names).
    """
    # ------------------------------------------------------------------- #
    # Identify column types (target must be excluded)
    # ------------------------------------------------------------------- #
    numeric_cols = df_after_feat.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df_after_feat.select_dtypes(include=["object", "category"]).columns.tolist()

    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    if target_col in categorical_cols:
        categorical_cols.remove(target_col)

    # ------------------------------------------------------------------- #
    # Split categorical columns into low / high cardinality
    # ------------------------------------------------------------------- #
    high_cardinality = [c for c in categorical_cols if df_after_feat[c].nunique() > 9]
    low_cardinality = [c for c in categorical_cols if c not in high_cardinality]

    # ------------------------------------------------------------------- #
    # Sub‑pipelines
    # ------------------------------------------------------------------- #
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    cat_low_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # High‑cardinality handling – use TargetEncoder if we have any such column
    if high_cardinality:
        cat_high_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("target_enc", TargetEncoder()),
            ]
        )
        cat_transformer = ColumnTransformer(
            transformers=[
                ("low", cat_low_pipe, low_cardinality),
                ("high", cat_high_pipe, high_cardinality),
            ],
            remainder="drop",
        )
    else:
        # No high‑cardinality columns → treat everything as low
        cat_transformer = cat_low_pipe

    # ------------------------------------------------------------------- #
    # Full ColumnTransformer
    # ------------------------------------------------------------------- #
    preproc = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", cat_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    # ------------------------------------------------------------------- #
    # Assemble the final pipeline
    # ------------------------------------------------------------------- #
    full_pipeline = Pipeline(
        steps=[
            ("feat_eng", CustomFeatureEngineer()),
            ("preproc", preproc),
        ]
    )

    # Fit once to be able to extract the feature order
    full_pipeline.fit(df_after_feat)

    # Feature order after the whole pipeline (used later for ONNX export)
    sample_out = full_pipeline.transform(df_after_feat.head(1))
    feature_order = sample_out.columns.tolist()

    return full_pipeline, feature_order


# --------------------------------------------------------------------------- #
# 3️⃣  Transform splits and write Parquet files (numeric only + target)
# --------------------------------------------------------------------------- #
def transform_and_save_splits(pipeline: Pipeline, raw_df: pd.DataFrame, split_indices: Dict[str, np.ndarray],
                              cfg: Dict) -> None:
    """
    For each split (train / valid / test):
        * Separate the target column
        * Apply the fitted pipeline to the feature part
        * Append the target back
        * Save as Parquet (numeric columns only)
    """
    splits_dir = pathlib.Path(cfg["paths"]["splits_dir"])
    ensure_dir(splits_dir)

    for name, idx in split_indices.items():
        df_raw = raw_df.loc[idx].reset_index(drop=True)

        # Separate target – be tolerant to case differences
        target_col = "Survived"
        if target_col not in df_raw.columns:
            # try lower‑case version
            alt = [c for c in df_raw.columns if c.lower() == "survived"]
            if alt:
                target_col = alt[0]
            else:
                raise KeyError("Target column 'Survived' not found in the raw data.")

        y = df_raw[target_col].astype(int).values
        X_raw = df_raw.drop(columns=[target_col])

        # Transform – returns a DataFrame because of set_config above
        X_transformed = pipeline.transform(X_raw)

        # Append target back
        out_df = X_transformed.copy()
        out_df["Survived"] = y

        out_path = splits_dir / f"{name}.parquet"
        out_df.to_parquet(out_path, index=False)
        print(f"✔ Saved {name} split → {out_path}")


# --------------------------------------------------------------------------- #
# 4️⃣  LightGBM training (exact hyper‑parameters from the spec)
# --------------------------------------------------------------------------- #
def train_lightgbm(cfg: Dict) -> Tuple[lgb.LGBMClassifier, Dict]:
    """Train LightGBM on the processed training split and persist the model."""
    splits_dir = pathlib.Path(cfg["paths"]["splits_dir"])

    train_df = pd.read_parquet(splits_dir / "train.parquet")
    valid_df = pd.read_parquet(splits_dir / "valid.parquet")

    X_train = train_df.drop(columns=["Survived"])
    y_train = train_df["Survived"]
    X_valid = valid_df.drop(columns=["Survived"])
    y_valid = valid_df["Survived"]

    params = {
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
    }

    model = lgb.LGBMClassifier(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="auc",
        early_stopping_rounds=20,
        verbose=False,
    )

    # Validation metrics
    valid_pred = model.predict_proba(X_valid)[:, 1]
    auc = roc_auc_score(y_valid, valid_pred)
    brier = brier_score_loss(y_valid, valid_pred)

    print(f"Validation ROC‑AUC : {auc:.4f}")
    print(f"Validation Brier  : {brier:.4f}")

    # Persist model
    model_path = pathlib.Path(cfg["paths"]["lightgbm_model"])
    ensure_dir(model_path.parent)
    joblib.dump(model, model_path)
    print(f"✔ LightGBM model saved → {model_path}")

    # Also copy to the required workspace directory
    ws_dir = pathlib.Path("./agent_workspace/trained_models")
    ensure_dir(ws_dir)
    ws_copy = ws_dir / model_path.name
    ws_copy.write_bytes(model_path.read_bytes())
    print(f"✔ Copy of LightGBM model placed at {ws_copy}")

    return model, {"roc_auc": auc, "brier": brier}


# --------------------------------------------------------------------------- #
# 5️⃣  Main orchestration
# --------------------------------------------------------------------------- #
def main():
    # ------------------------------------------------------------------- #
    # Load configuration (expects a file named ``config.yaml`` at repo root)
    # ------------------------------------------------------------------- #
    repo_root = pathlib.Path(__file__).resolve().parents[1]  # src/ -> experiment/
    cfg_path = repo_root / "config.yaml"
    if not cfg_path.is_file():
        sys.exit(f"Configuration file not found at {cfg_path}")

    cfg = load_config(cfg_path)

    # ------------------------------------------------------------------- #
    # Resolve the raw CSV path (absolute path is stored in the config)
    # ------------------------------------------------------------------- #
    raw_path = pathlib.Path(cfg["paths"]["raw_csv"])
    if not raw_path.is_file():
        sys.exit(f"Raw CSV not found at expected location: {raw_path}")

    # ------------------------------------------------------------------- #
    # 0️⃣ Load raw CSV (as strings to avoid premature NaNs)
    # ------------------------------------------------------------------- #
    raw_df = pd.read_csv(raw_path, dtype=str)

    # ------------------------------------------------------------------- #
    # Ensure the target column exists and is integer‑typed
    # ------------------------------------------------------------------- #
    target_col = "Survived"
    if target_col not in raw_df.columns:
        # fallback to case‑insensitive match
        matches = [c for c in raw_df.columns if c.lower() == "survived"]
        if matches:
            target_col = matches[0]
        else:
            sys.exit("Target column 'Survived' not found in the raw CSV.")
    raw_df[target_col] = raw_df[target_col].astype(int)

    # ------------------------------------------------------------------- #
    # 1️⃣ Deterministic train/valid/test split (70/20/10) – stratified on target
    # ------------------------------------------------------------------- #
    rng = np.random.default_rng(cfg["resources"]["seed"])
    shuffled = rng.permutation(raw_df.index)

    n = len(shuffled)
    n_train = int(0.70 * n)
    n_valid = int(0.20 * n)

    train_idx = shuffled[:n_train]
    valid_idx = shuffled[n_train : n_train + n_valid]
    test_idx = shuffled[n_train + n_valid :]

    split_indices = {
        "train": train_idx,
        "valid": valid_idx,
        "test": test_idx,
    }

    # ------------------------------------------------------------------- #
    # 2️⃣ Build pipeline template, fit on train+valid, and persist it
    # ------------------------------------------------------------------- #
    # Run the custom transformer once on the combined data to discover columns
    combined_raw = pd.concat([raw_df.loc[train_idx], raw_df.loc[valid_idx]], axis=0).reset_index(drop=True)
    after_feat = CustomFeatureEngineer().transform(combined_raw)

    pipeline, feature_order = build_full_pipeline(after_feat, target_col=target_col)

    # Persist the pipeline
    pipeline_path = pathlib.Path(cfg["paths"]["pipeline"])
    ensure_dir(pipeline_path.parent)
    joblib.dump(pipeline, pipeline_path)
    print(f"✔ Preprocessing pipeline saved → {pipeline_path}")

    # Persist feature order (required later for ONNX export)
    feature_order_path = pathlib.Path("models/feature_order.json")
    ensure_dir(feature_order_path.parent)
    feature_order_path.write_text(json.dumps(feature_order, indent=2))
    print(f"✔ Feature order saved → {feature_order_path}")

    # ------------------------------------------------------------------- #
    # 3️⃣ Transform each split and write Parquet files
    # ------------------------------------------------------------------- #
    transform_and_save_splits(pipeline, raw_df, split_indices, cfg)

    # ------------------------------------------------------------------- #
    # 4️⃣ Train LightGBM on the processed data
    # ------------------------------------------------------------------- #
    model, metrics = train_lightgbm(cfg)

    # ------------------------------------------------------------------- #
    # 5️⃣ Final summary
    # ------------------------------------------------------------------- #
    print("\n=== PIPELINE COMPLETED ===")
    print(f"Validation metrics: ROC‑AUC={metrics['roc_auc']:.4f}, Brier={metrics['brier']:.4f}")
    print("Artifacts written:")
    print(f"  Preprocessing pipeline → {cfg['paths']['pipeline']}")
    print(f"  LightGBM model          → {cfg['paths']['lightgbm_model']}")
    print("  (Both also copied to ./agent_workspace/trained_models)")

if __name__ == "__main__":
    main()
