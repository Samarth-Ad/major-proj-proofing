
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
titanic_full_pipeline.py
-----------------------

A **stand‑alone** reproducible pipeline for the Titanic survival classification
problem.  The script follows the exact folder layout and configuration described
in the project documentation:

1️⃣  Load the raw CSV (path taken from ``config.yaml``).  
2️⃣  Resolve the target column name case‑insensitively.  
3️⃣  Perform a deterministic stratified split (70 % train / 20 % valid / 10 % test).  
4️⃣  Build a Scikit‑Learn preprocessing pipeline that  
    * engineers ``FamilySize``, ``Title``, ``AgeBin`` and ``LogFare``;  
    * drops the raw columns that become redundant;  
    * imputes & scales numeric columns;  
    * one‑hot‑encodes low‑cardinality categoricals (≤ 9 distinct values);  
    * target‑encodes high‑cardinality categoricals (> 9 distinct values).  
5️⃣  Fit the pipeline on **train + valid**, persist the pipeline and the ordered
    feature list.  
6️⃣  Transform the three splits, re‑attach the target column and write them as
    Parquet files (numeric only + target).  
7️⃣  Train a LightGBM binary classifier with the hyper‑parameters that gave the
    best ROC‑AUC in the “top‑3” playbook.  
8️⃣  Evaluate on the validation split (ROC‑AUC, Accuracy, F1, Brier score).  
9️⃣  Persist the trained model to ``models/lightgbm_best.pkl`` **and** copy it to
    ``./agent_workspace/trained_models`` (required by the deployment contract).

All artefacts are stored under paths that are *relative to the repository root*,
so the script can be executed from the repository root without any further
modifications.
"""

# --------------------------------------------------------------------- #
# 0️⃣  Imports & helper utilities
# --------------------------------------------------------------------- #
from __future__ import annotations

import json
import pathlib
import re
import sys
from typing import List, Tuple, Union

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# --------------------------------------------------------------------- #
# 1️⃣  Load configuration (single source of truth)
# --------------------------------------------------------------------- #
def load_config() -> dict:
    """Load ``config.yaml`` from the repository root."""
    cfg_path = pathlib.Path("config.yaml")
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config.yaml not found at {cfg_path.resolve()}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------- #
# 2️⃣  Resolve target column name (case‑insensitive)
# --------------------------------------------------------------------- #
def resolve_target_column(df: pd.DataFrame) -> str:
    """Return the exact column name that matches ``Survived`` case‑insensitively."""
    candidates = [c for c in df.columns if c.lower() == "survived"]
    if not candidates:
        raise KeyError("Target column 'Survived' not found in the dataset.")
    return candidates[0]


# --------------------------------------------------------------------- #
# 3️⃣  Custom feature‑engineering transformer
# --------------------------------------------------------------------- #
class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    """Creates engineered columns and drops the raw ones."""

    def __init__(self):
        self.age_bins_: Union[np.ndarray, None] = None

    # --------------------------------------------------------------- #
    # Fit – compute quantile edges for Age (if present)
    # --------------------------------------------------------------- #
    def fit(self, X: pd.DataFrame, y=None):
        if "Age" in X.columns:
            ages = X["Age"].astype(float)
            # ``qcut`` returns the bin edges; ``duplicates='drop'`` protects against
            # too many identical ages.
            _, bins = pd.qcut(
                ages.dropna(),
                q=5,
                duplicates="drop",
                retbins=True,
            )
            self.age_bins_ = bins
        else:
            self.age_bins_ = None
        return self

    # --------------------------------------------------------------- #
    # Transform – add engineered columns, drop raw ones
    # --------------------------------------------------------------- #
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # ---- FamilySize ------------------------------------------------
        if {"SibSp", "Parch"}.issubset(X.columns):
            X["FamilySize"] = (
                X["SibSp"].astype(float) + X["Parch"].astype(float) + 1
            ).astype(int)

        # ---- Title ----------------------------------------------------
        if "Name" in X.columns:
            X["Title"] = X["Name"].fillna("").apply(
                lambda n: re.search(r",\s*([^.]*)\.", n).group(1).strip()
                if re.search(r",\s*([^.]*)\.", n)
                else "Unknown"
            )

        # ---- AgeBin ----------------------------------------------------
        if self.age_bins_ is not None and "Age" in X.columns:
            X["AgeBin"] = pd.cut(
                X["Age"].astype(float),
                bins=self.age_bins_,
                labels=False,
                duplicates="drop",
            ).astype(str)

        # ---- LogFare ---------------------------------------------------
        if "Fare" in X.columns:
            X["LogFare"] = np.log1p(X["Fare"].astype(float))

        # ---- Drop raw columns -----------------------------------------
        cols_to_drop = [
            "Name",
            "Ticket",
            "Cabin",
            "Age",
            "Fare",
            "SibSp",
            "Parch",
        ]
        X.drop(columns=[c for c in cols_to_drop if c in X.columns],
               inplace=True, errors="ignore")
        return X


# --------------------------------------------------------------------- #
# 4️⃣  Fallback TargetEncoder (used only if ``category_encoders`` is missing)
# --------------------------------------------------------------------- #
try:
    from category_encoders import TargetEncoder  # type: ignore
except Exception:  # pragma: no cover
    class TargetEncoder(BaseEstimator, TransformerMixin):
        """Mean‑target encoding for each categorical column."""

        def __init__(self):
            self.mapping_: dict = {}
            self.global_mean_: float = 0.0

        def fit(self, X: pd.DataFrame, y: pd.Series):
            # Ensure ``y`` is numeric – the raw CSV is read as strings.
            y_numeric = pd.to_numeric(y, errors="coerce")
            self.global_mean_ = float(y_numeric.mean())
            for col in X.columns:
                # Compute mean target per category
                self.mapping_[col] = (
                    pd.DataFrame({"x": X[col], "y": y_numeric})
                    .groupby("x")["y"]
                    .mean()
                    .to_dict()
                )
            return self

        def transform(self, X: pd.DataFrame) -> pd.DataFrame:
            X_out = X.copy()
            for col, mapping in self.mapping_.items():
                X_out[col] = X_out[col].map(mapping).fillna(self.global_mean_)
            return X_out


# --------------------------------------------------------------------- #
# 5️⃣  Build the full preprocessing pipeline (unfitted)
# --------------------------------------------------------------------- #
def build_preprocess_pipeline(df: pd.DataFrame) -> Tuple[Pipeline, List[str]]:
    target_col = resolve_target_column(df)

    # Run the feature engineer once to discover post‑FE column types
    fe = CustomFeatureEngineer()
    fe.fit(df.drop(columns=[target_col]))
    engineered = fe.transform(df.drop(columns=[target_col]))

    # --------------------------------------------------------------- #
    # Identify numeric vs categorical columns after FE
    # --------------------------------------------------------------- #
    numeric_cols = engineered.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = engineered.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    # --------------------------------------------------------------- #
    # Split categorical columns by cardinality
    # --------------------------------------------------------------- #
    low_cardinality = [c for c in categorical_cols if engineered[c].nunique() <= 9]
    high_cardinality = [c for c in categorical_cols if engineered[c].nunique() > 9]

    # --------------------------------------------------------------- #
    # Sub‑pipelines
    # --------------------------------------------------------------- #
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    low_cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    high_cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("target_enc", TargetEncoder()),
        ]
    )

    # --------------------------------------------------------------- #
    # ColumnTransformer – apply the right sub‑pipeline to each group
    # --------------------------------------------------------------- #
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("low_cat", low_cat_pipe, low_cardinality),
            ("high_cat", high_cat_pipe, high_cardinality),
        ],
        remainder="drop",
    )

    # --------------------------------------------------------------- #
    # Full pipeline (FeatureEngineer → ColumnTransformer)
    # --------------------------------------------------------------- #
    full_pipe = Pipeline(
        steps=[
            ("feature_engineer", fe),
            ("preprocess", preprocessor),
        ]
    )

    # Return the pipeline and the *raw* feature list (useful for debugging)
    raw_feature_names = numeric_cols + low_cardinality + high_cardinality
    return full_pipe, raw_feature_names


# --------------------------------------------------------------------- #
# 6️⃣  Deterministic stratified split (70 % train / 20 % valid / 10 % test)
# --------------------------------------------------------------------- #
def stratified_split(df: pd.DataFrame, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_col = resolve_target_column(df)
    y = pd.to_numeric(df[target_col], errors="coerce").astype(int).values

    # 70 % train, 30 % temporary
    train_idx, temp_idx, _, _ = train_test_split(
        np.arange(len(df)),
        y,
        test_size=0.30,
        stratify=y,
        random_state=seed,
    )
    # From the 30 % temporary, take 1/3 as test → 10 % overall
    y_temp = y[temp_idx]
    valid_idx, test_idx, _, _ = train_test_split(
        temp_idx,
        y_temp,
        test_size=1 / 3,
        stratify=y_temp,
        random_state=seed,
    )
    return train_idx, valid_idx, test_idx


# --------------------------------------------------------------------- #
# 7️⃣  Fit pipeline, persist artefacts, write Parquet splits
# --------------------------------------------------------------------- #
def fit_pipeline_and_write_splits(
    cfg: dict,
    df_raw: pd.DataFrame,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    test_idx: np.ndarray,
) -> Tuple[pathlib.Path, List[str]]:
    target_col = resolve_target_column(df_raw)

    # Build *unfitted* pipeline
    pipeline, _ = build_preprocess_pipeline(df_raw)

    # -----------------------------------------------------------------
    # Fit on the union of train + valid (features only)
    # -----------------------------------------------------------------
    train_features = df_raw.iloc[train_idx].drop(columns=[target_col])
    valid_features = df_raw.iloc[valid_idx].drop(columns=[target_col])

    # Convert target to numeric before passing to ``fit`` (required for TargetEncoder)
    y_train = pd.to_numeric(df_raw.iloc[train_idx][target_col], errors="coerce").astype(int)
    y_valid = pd.to_numeric(df_raw.iloc[valid_idx][target_col], errors="coerce").astype(int)

    pipeline.fit(
        pd.concat([train_features, valid_features], axis=0),
        pd.concat([y_train, y_valid], axis=0),
    )

    # -----------------------------------------------------------------
    # Persist the pipeline
    # -----------------------------------------------------------------
    pipeline_path = pathlib.Path(cfg["paths"]["pipeline"])
    pipeline_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, pipeline_path)
    print(f"✅ Pipeline saved → {pipeline_path}")

    # -----------------------------------------------------------------
    # Extract ordered feature names after the ColumnTransformer
    # -----------------------------------------------------------------
    dummy = df_raw.iloc[train_idx[:1]].drop(columns=[target_col])
    transformed_dummy = pipeline.transform(dummy)

    if hasattr(pipeline.named_steps["preprocess"], "get_feature_names_out"):
        feature_order = pipeline.named_steps["preprocess"].get_feature_names_out().tolist()
    else:
        # Fallback – the transformer returns a NumPy array
        if isinstance(transformed_dummy, np.ndarray):
            feature_order = [f"f{i}" for i in range(transformed_dummy.shape[1])]
        else:  # pandas DataFrame
            feature_order = transformed_dummy.columns.tolist()

    # -----------------------------------------------------------------
    # Save feature order (required later for ONNX export)
    # -----------------------------------------------------------------
    feature_path = pathlib.Path(cfg["paths"]["feature_order"])
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    with feature_path.open("w", encoding="utf-8") as f:
        json.dump(feature_order, f, indent=2)
    print(f"✅ Feature order saved → {feature_path}")

    # -----------------------------------------------------------------
    # Transform each split, re‑attach the target, write Parquet
    # -----------------------------------------------------------------
    splits_dir = pathlib.Path(cfg["paths"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    for name, idx in [("train", train_idx), ("valid", valid_idx), ("test", test_idx)]:
        split_raw = df_raw.iloc[idx].reset_index(drop=True)
        X = split_raw.drop(columns=[target_col])
        y = pd.to_numeric(split_raw[target_col], errors="coerce").astype(int)

        X_transformed = pipeline.transform(X)

        if isinstance(X_transformed, np.ndarray):
            X_transformed = pd.DataFrame(
                X_transformed,
                columns=feature_order,
                index=split_raw.index,
            )
        else:  # already a DataFrame
            X_transformed = pd.DataFrame(
                X_transformed,
                columns=feature_order,
                index=split_raw.index,
            )

        # Append the target column using the *exact* name from the CSV
        X_transformed[target_col] = y.values

        parquet_path = splits_dir / f"{name}.parquet"
        X_transformed.to_parquet(parquet_path, index=False)
        print(f"✅ {name.capitalize()} split written → {parquet_path}")

    return pipeline_path, feature_order


# --------------------------------------------------------------------- #
# 8️⃣  LightGBM training & validation
# --------------------------------------------------------------------- #
def train_lightgbm(cfg: dict, feature_order: List[str]) -> pathlib.Path:
    splits_dir = pathlib.Path(cfg["paths"]["splits_dir"])

    train_df = pd.read_parquet(splits_dir / "train.parquet")
    valid_df = pd.read_parquet(splits_dir / "valid.parquet")

    target_col = resolve_target_column(train_df)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].astype(int)

    X_valid = valid_df.drop(columns=[target_col])
    y_valid = valid_df[target_col].astype(int)

    # -----------------------------------------------------------------
    # Hyper‑parameters from the “top‑3” playbook
    # -----------------------------------------------------------------
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
        "n_jobs": cfg["resources"]["n_jobs"],
    }

    model = lgb.LGBMClassifier(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        early_stopping_rounds=20,
        verbose=False,
    )

    # -----------------------------------------------------------------
    # Validation metrics
    # -----------------------------------------------------------------
    prob_valid = model.predict_proba(X_valid)[:, 1]
    pred_valid = (prob_valid >= 0.5).astype(int)

    auc = roc_auc_score(y_valid, prob_valid)
    acc = accuracy_score(y_valid, pred_valid)
    f1 = f1_score(y_valid, pred_valid)
    brier = brier_score_loss(y_valid, prob_valid)

    print("\n🔎 Validation metrics")
    print(f"  ROC‑AUC : {auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1      : {f1:.4f}")
    print(f"  Brier  : {brier:.4f}")

    # -----------------------------------------------------------------
    # Persist the model
    # -----------------------------------------------------------------
    model_path = pathlib.Path(cfg["paths"]["lightgbm_model"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    print(f"\n✅ LightGBM model saved → {model_path}")

    # -----------------------------------------------------------------
    # Copy to the required deployment folder
    # -----------------------------------------------------------------
    deploy_dir = pathlib.Path("./agent_workspace/trained_models")
    deploy_dir.mkdir(parents=True, exist_ok=True)
    copy_path = deploy_dir / model_path.name
    copy_path.write_bytes(model_path.read_bytes())
    print(f"✅ Model also copied to {copy_path}")

    return model_path


# --------------------------------------------------------------------- #
# 9️⃣  Main orchestration
# --------------------------------------------------------------------- #
def main() -> None:
    cfg = load_config()

    # ------------------------------------------------------------- #
    # 1️⃣ Load raw CSV (all columns as strings → prevents premature NaN conversion)
    # ------------------------------------------------------------- #
    raw_path = pathlib.Path(cfg["paths"]["raw_csv"])
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw CSV not found at {raw_path}")
    df_raw = pd.read_csv(raw_path, dtype=str)

    # ------------------------------------------------------------- #
    # 2️⃣ Deterministic stratified split
    # ------------------------------------------------------------- #
    seed = cfg["resources"]["seed"]
    train_idx, valid_idx, test_idx = stratified_split(df_raw, seed)

    # ------------------------------------------------------------- #
    # 3️⃣ Build pipeline, fit on train+valid, persist artefacts, write splits
    # ------------------------------------------------------------- #
    _, feature_order = fit_pipeline_and_write_splits(
        cfg, df_raw, train_idx, valid_idx, test_idx
    )

    # ------------------------------------------------------------- #
    # 4️⃣ Train LightGBM and evaluate on validation set
    # ------------------------------------------------------------- #
    _ = train_lightgbm(cfg, feature_order)

    print("\n🎉 End‑to‑end pipeline completed successfully!")


if __name__ == "__main__":
    # The script is intended to be run from the repository root.
    main()
