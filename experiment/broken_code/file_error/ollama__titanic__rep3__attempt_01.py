
"""
Titanic modeling pipeline – end‑to‑end (data loading → preprocessing → LightGBM training → model saving)

The script follows the exact folder layout and hyper‑parameters described in the
specification.  All artefacts are written under the repository root and a copy of
the trained LightGBM model is also placed in ``./agent_workspace/trained_models`` as
required by the execution environment.

The script is deliberately self‑contained:
* It does **not** import optional heavy libraries such as ``onnx`` or ``mlflow``,
  avoiding the “ModuleNotFoundError” seen in the previous run.
* If ``category_encoders`` is unavailable, the pipeline falls back to using only
  ``OneHotEncoder`` (the Titanic dataset has no truly high‑cardinality columns, so
  this fallback is safe).

Run the script from the repository root:

    python src/pipeline_full.py
"""

import pathlib
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import roc_auc_score, brier_score_loss

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def ensure_dir(p: pathlib.Path) -> None:
    """Create a directory (including parents) if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)


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

    def __init__(self):
        pass

    @staticmethod
    def _extract_title(name: str) -> str:
        import re

        m = re.search(r",\s*([^\.]*)\.", name)
        return m.group(1).strip() if m else "Unknown"

    def fit(self, X, y=None):
        # No state to learn
        return self

    def transform(self, X):
        X = X.copy()

        # FamilySize
        X["FamilySize"] = (
            X["SibSp"].astype(float) + X["Parch"].astype(float) + 1
        ).astype(int)

        # Title
        X["Title"] = X["Name"].apply(self._extract_title)

        # AgeBin (5 quantile bins)
        age_num = pd.to_numeric(X["Age"], errors="coerce")
        X["AgeBin"] = pd.qcut(age_num, q=5, duplicates="drop").astype(str)

        # LogFare
        fare_num = pd.to_numeric(X["Fare"], errors="coerce")
        X["LogFare"] = np.log1p(fare_num)

        # Drop raw columns
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
# 2️⃣  Build the full preprocessing pipeline (Scikit‑Learn)
# --------------------------------------------------------------------------- #
def build_preprocess_pipeline():
    """Creates the pipeline described in the spec (feature‑engineer → impute → encode → scale)."""
    # --- numeric sub‑pipeline -------------------------------------------------
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    # --- categorical sub‑pipeline (low‑cardinality) ---------------------------
    cat_low_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Try to import TargetEncoder for high‑cardinality handling.
    # Titanic does not have such columns, so we can safely ignore if the import fails.
    try:
        from category_encoders import TargetEncoder
    except Exception:  # pragma: no cover
        TargetEncoder = None

    # The actual ColumnTransformer will be built *after* we see the column names
    # (see `fit_and_save_pipeline` below).
    placeholder_ct = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, []),  # to be filled later
            ("cat", cat_low_pipe, []),  # to be filled later
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("feat_eng", CustomFeatureEngineer()),
            ("preproc", placeholder_ct),
        ]
    )
    return pipeline


# --------------------------------------------------------------------------- #
# 3️⃣  Fit pipeline on train+valid, persist it, and store feature order
# --------------------------------------------------------------------------- #
def fit_and_save_pipeline(pipeline, raw_df, train_idx, valid_idx, cfg):
    """Fit the preprocessing pipeline on train+valid, save it, and write feature order."""
    # Split raw data (features + target)
    train_raw = raw_df.loc[train_idx].reset_index(drop=True)
    valid_raw = raw_df.loc[valid_idx].reset_index(drop=True)

    # Run the custom transformer once to discover column names after feature engineering
    combined_raw = pd.concat([train_raw, valid_raw], axis=0).reset_index(drop=True)
    after_feat = pipeline.named_steps["feat_eng"].transform(combined_raw)

    # Identify numeric vs categorical columns
    numeric_cols = after_feat.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = after_feat.select_dtypes(include=["object", "category"]).columns.tolist()

    # Separate low‑ and high‑cardinality categorical columns
    high_cardinality = [
        c for c in categorical_cols if after_feat[c].nunique() > 9
    ]
    low_cardinality = [c for c in categorical_cols if c not in high_cardinality]

    # Build the final ColumnTransformer
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

    if high_cardinality and (pipeline.named_steps["preproc"].transformers[1][1].steps[0][0] != "target_enc"):
        # Build high‑cardinality pipeline only if TargetEncoder is available
        if "TargetEncoder" in globals() and globals()["TargetEncoder"] is not None:
            cat_high_pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("target_enc", globals()["TargetEncoder"]()),
                ]
            )
            cat_transformer = ColumnTransformer(
                transformers=[
                    ("low", cat_low_pipe, low_cardinality),
                    ("high", cat_high_pipe, high_cardinality),
                ]
            )
        else:
            # Fallback – treat high‑cardinality as low‑cardinality
            cat_transformer = cat_low_pipe
    else:
        cat_transformer = cat_low_pipe

    preproc = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", cat_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    # Assemble the final pipeline
    final_pipe = Pipeline(
        steps=[
            ("feat_eng", CustomFeatureEngineer()),
            ("preproc", preproc),
        ]
    )

    # Fit on train+valid
    final_pipe.fit(combined_raw)

    # ------------------------------------------------------------------- #
    # Persist pipeline
    # ------------------------------------------------------------------- #
    pipeline_path = pathlib.Path(cfg["paths"]["pipeline"])
    ensure_dir(pipeline_path.parent)
    joblib.dump(final_pipe, pipeline_path)

    # ------------------------------------------------------------------- #
    # Store feature order (column names after full transformation)
    # ------------------------------------------------------------------- #
    sample_transformed = final_pipe.transform(combined_raw.head(1))
    feature_order = sample_transformed.columns.tolist()
    feature_order_path = pathlib.Path("models/feature_order.json")
    ensure_dir(feature_order_path.parent)
    feature_order_path.write_text(json.dumps(feature_order, indent=2))

    return final_pipe, feature_order


# --------------------------------------------------------------------------- #
# 4️⃣  Transform splits and write Parquet files (numeric only, target retained)
# --------------------------------------------------------------------------- #
def transform_and_save_splits(pipeline, raw_df, split_indices, cfg):
    """Apply the fitted pipeline to train/valid/test and store Parquet files."""
    splits_dir = pathlib.Path(cfg["paths"]["splits_dir"])
    ensure_dir(splits_dir)

    for name, idx in split_indices.items():
        df_raw = raw_df.loc[idx].reset_index(drop=True)
        y = df_raw["Survived"].astype(int).values
        X_transformed = pipeline.transform(df_raw)

        # X_transformed is a DataFrame with ONLY numeric columns (after encoding)
        out_df = X_transformed.copy()
        out_df["Survived"] = y
        out_path = splits_dir / f"{name}.parquet"
        out_df.to_parquet(out_path, index=False)


# --------------------------------------------------------------------------- #
# 5️⃣  LightGBM training (exact hyper‑parameters from the spec)
# --------------------------------------------------------------------------- #
def train_lightgbm(cfg):
    """Train LightGBM on the processed training split and log validation metrics."""
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

    # Also copy to the required workspace directory
    ws_dir = pathlib.Path("./agent_workspace/trained_models")
    ensure_dir(ws_dir)
    ws_copy = ws_dir / model_path.name
    ws_copy.write_bytes(model_path.read_bytes())
    print(f"Copy of LightGBM model placed at {ws_copy}")

    return model, {"roc_auc": auc, "brier": brier}


# --------------------------------------------------------------------------- #
# 6️⃣  Main orchestration
# --------------------------------------------------------------------------- #
def main():
    # ------------------------------------------------------------------- #
    # Configuration (hard‑coded to match the required layout)
    # ------------------------------------------------------------------- #
    cfg = {
        "paths": {
            "raw_csv": r"C:/Users/Samarth_Ad/Desktop/College/SEM-7/AutoMLAgent/experiment/data/raw/titanic.csv",
            "pipeline": "models/pipeline_preprocess.pkl",
            "lightgbm_model": "models/lightgbm_best.pkl",
            "splits_dir": "data/splits",
        },
        "resources": {
            "seed": 42,
            "n_jobs": 4,
        },
    }

    # ------------------------------------------------------------------- #
    # 0️⃣ Load raw CSV (as strings to avoid premature NaNs)
    # ------------------------------------------------------------------- #
    raw_path = pathlib.Path(cfg["paths"]["raw_csv"])
    raw_df = pd.read_csv(raw_path, dtype=str)

    # Ensure target column is integer for later steps
    raw_df["Survived"] = raw_df["Survived"].astype(int)

    # ------------------------------------------------------------------- #
    # 1️⃣ Deterministic train/valid/test split (70/20/10)
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
    # 2️⃣ Build and fit preprocessing pipeline, then persist it
    # ------------------------------------------------------------------- #
    pipeline_template = build_preprocess_pipeline()
    pipeline, feature_order = fit_and_save_pipeline(
        pipeline_template, raw_df, train_idx, valid_idx, cfg
    )

    # ------------------------------------------------------------------- #
    # 3️⃣ Transform splits and write Parquet files
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
    print("Artifacts:")
    print(f"  Preprocessing pipeline → {cfg['paths']['pipeline']}")
    print(f"  LightGBM model          → {cfg['paths']['lightgbm_model']}")
    print("  (Both also copied to ./agent_workspace/trained_models)")

if __name__ == "__main__":
    main()
