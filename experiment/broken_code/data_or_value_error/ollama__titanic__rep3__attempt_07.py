
"""
Titanic survival end‑to‑end pipeline
-----------------------------------

This script is self‑contained and can be executed from **any** working directory.
It performs the following steps:

1. Load a configuration (``config.yaml``) if it exists; otherwise fall back to a
   built‑in default configuration that points to the standard repository layout.
2. Read the raw Titanic CSV (everything as ``str`` to avoid premature type conversion).
3. Resolve the target column name in a case‑insensitive way.
4. Build a preprocessing pipeline:
   * ``CustomFeatureEngineer`` – creates ``FamilySize``, ``Title``, ``AgeBin``,
     ``LogFare`` and drops the raw columns.
   * Numeric → median imputer → robust scaler.
   * Low‑cardinality categorical (≤ 9 distinct) → most‑frequent imputer → one‑hot.
   * High‑cardinality categorical (> 9 distinct) → most‑frequent imputer → target encoder
     (fallback to a tiny ordinal encoder if the *category‑encoders* package is missing).
5. Perform a stratified split (70 % train / 20 % validation / 10 % test) using the seed
   from the configuration.
6. Fit the pipeline on the **train + validation** data, transform all three splits,
   persist the pipeline, the feature order and the Parquet files.
7. Train a LightGBM classifier on the processed training split, evaluate on validation,
   and save the model.
8. (Optional) Export the trained model to ONNX and wrap it in a TorchScript module
   if the required libraries are installed.
9. Copy the final LightGBM model to ``./agent_workspace/trained_models`` – this is the
   location required by the downstream deployment step.

All artefacts are written under the repository root, matching the layout described
in the original specification.  No manual path editing is required.
"""

# --------------------------------------------------------------------------- #
# Imports (standard + optional)
# --------------------------------------------------------------------------- #
import json
import pathlib
import re
import sys
from typing import List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml

# Optional third‑party imports – graceful fall‑backs
try:
    from category_encoders import TargetEncoder  # type: ignore
except Exception:  # pragma: no cover
    # Very small fallback: ordinal encoding based on pandas factorize
    class TargetEncoder:  # pylint: disable=too-few-public-methods
        """Fallback that mimics the fit/transform API of category_encoders.TargetEncoder."""
        def __init__(self):
            self.mapping_ = {}

        def fit(self, X, y=None):
            if isinstance(X, pd.DataFrame):
                for col in X.columns:
                    _, uniques = pd.factorize(X[col])
                    self.mapping_[col] = {cat: idx for idx, cat in enumerate(uniques)}
            else:  # Series
                _, uniques = pd.factorize(X)
                self.mapping_ = {cat: idx for idx, cat in enumerate(uniques)}
            return self

        def transform(self, X):
            X = X.copy()
            if isinstance(X, pd.DataFrame):
                for col, mapping in self.mapping_.items():
                    X[col] = X[col].map(mapping).fillna(-1).astype(int)
            else:  # Series
                X = X.map(self.mapping_).fillna(-1).astype(int)
            return X

        def fit_transform(self, X, y=None):
            return self.fit(X, y).transform(X)

try:
    import onnx  # type: ignore
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    onnx = None
    ort = None

# Scikit‑Learn imports
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# --------------------------------------------------------------------------- #
# 1️⃣  Configuration loader (with sensible defaults)
# --------------------------------------------------------------------------- #
def load_config() -> dict:
    """
    Look for ``config.yaml`` in three conventional locations.
    If none is found, return a built‑in default configuration that matches the
    repository layout described in the specification.
    """
    possible = [
        pathlib.Path(__file__).resolve().parents[0] / "config.yaml",   # same dir as script
        pathlib.Path.cwd() / "config.yaml",                           # cwd
        pathlib.Path.cwd() / "experiment" / "config.yaml",           # fallback used in notebooks
    ]
    for p in possible:
        if p.is_file():
            with p.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)

    # ---------- Default configuration (no external file) ----------
    repo_root = pathlib.Path.cwd()
    default_cfg = {
        "paths": {
            "raw_csv": str(repo_root / "data" / "raw" / "titanic.csv"),
            "pipeline": "models/pipeline_preprocess.pkl",
            "lightgbm_model": "models/lightgbm_best.pkl",
            "onnx_model": "onnx/lightgbm_best.onnx",
            "torch_script": "onnx/lightgbm_onnx.pt",
            "splits_dir": "data/splits",
        },
        "mlflow": {
            "tracking_uri": f"file://{repo_root / 'mlflow'}",
            "experiment_name": "titanic_survival",
        },
        "resources": {
            "seed": 42,
            "n_jobs": 4,
            "max_memory_gb": 8,
        },
    }
    return default_cfg


# --------------------------------------------------------------------------- #
# 2️⃣  Helper to resolve the target column name (case‑insensitive)
# --------------------------------------------------------------------------- #
def resolve_target_column(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if c.lower() == "survived"]
    if not candidates:
        raise KeyError("Target column 'Survived' not found in the raw CSV.")
    return candidates[0]  # exact spelling as it appears in the file


# --------------------------------------------------------------------------- #
# 3️⃣  Custom feature‑engineering transformer (sklearn compatible)
# --------------------------------------------------------------------------- #
class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Adds:
        * FamilySize = SibSp + Parch + 1
        * Title      = extracted from Name (regex)
        * AgeBin     = 5 quantile bins (as string)
        * LogFare    = log1p(Fare)
    and drops the raw columns that are no longer needed.
    """

    def __init__(self):
        self.age_bins_ = None  # will hold the bin edges after fit

    # ------------------------------------------------------------------- #
    # Fit – compute quantile edges for Age (ignoring NaNs)
    # ------------------------------------------------------------------- #
    def fit(self, X: pd.DataFrame, y=None):
        ages = X["Age"].astype(float)
        # ``duplicates='drop'`` protects against too many identical ages
        _, bins = pd.qcut(ages.dropna(), q=5, duplicates="drop", retbins=True)
        self.age_bins_ = bins
        return self

    # ------------------------------------------------------------------- #
    # Transform – create new features and drop the originals
    # ------------------------------------------------------------------- #
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # FamilySize
        X["FamilySize"] = (
            X["SibSp"].astype(float) + X["Parch"].astype(float) + 1
        ).astype(int)

        # Title (regex on the Name column)
        X["Title"] = X["Name"].fillna("").apply(
            lambda n: re.search(r",\s*([^.]*)\.", n).group(1).strip()
            if re.search(r",\s*([^.]*)\.", n)
            else "Unknown"
        )

        # AgeBin – use the bins learned during fit; unknown ages become NaN → later imputed
        X["AgeBin"] = pd.cut(
            X["Age"].astype(float), bins=self.age_bins_, labels=False, duplicates="drop"
        ).astype(str)

        # LogFare
        X["LogFare"] = np.log1p(X["Fare"].astype(float))

        # Drop raw columns that are no longer needed
        cols_to_drop = ["Name", "Ticket", "Cabin", "Age", "Fare", "SibSp", "Parch"]
        X.drop(columns=cols_to_drop, inplace=True, errors="ignore")

        return X


# --------------------------------------------------------------------------- #
# 4️⃣  Build the full preprocessing pipeline (returns unfitted pipeline)
# --------------------------------------------------------------------------- #
def build_preprocess_pipeline(df: pd.DataFrame) -> Tuple[Pipeline, List[str]]:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe **including** the target column.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Unfitted pipeline that expects only feature columns.
    feature_names : List[str]
        List of column names after feature engineering (used later for ONNX).
    """
    target_col = resolve_target_column(df)

    # Run the feature engineer once to discover column types
    fe = CustomFeatureEngineer()
    fe.fit(df.drop(columns=[target_col]))
    engineered = fe.transform(df.drop(columns=[target_col]))

    numeric_cols = engineered.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = engineered.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    # Split categorical columns by cardinality
    low_cardinality = [c for c in categorical_cols if engineered[c].nunique() <= 9]
    high_cardinality = [c for c in categorical_cols if engineered[c].nunique() > 9]

    # Sub‑pipelines
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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("low_cat", low_cat_pipe, low_cardinality),
            ("high_cat", high_cat_pipe, high_cardinality),
        ],
        remainder="drop",
    )

    full_pipe = Pipeline(
        steps=[
            ("feature_engineer", fe),
            ("preprocess", preprocessor),
        ]
    )

    # Return also the *raw* feature list (used only for sanity checks)
    feature_names = numeric_cols + low_cardinality + high_cardinality
    return full_pipe, feature_names


# --------------------------------------------------------------------------- #
# 5️⃣  Stratified split helper (70/20/10)
# --------------------------------------------------------------------------- #
def stratified_split(df: pd.DataFrame, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns three index arrays: train_idx, valid_idx, test_idx.
    """
    target_col = resolve_target_column(df)
    y = df[target_col].astype(int).values

    # First split 70% train / 30% temp
    train_idx, temp_idx, _, _ = train_test_split(
        np.arange(len(df)),
        y,
        test_size=0.30,
        stratify=y,
        random_state=seed,
    )
    # From the 30% temp we take 1/3 as test → 10% overall
    y_temp = y[temp_idx]
    valid_idx, test_idx, _, _ = train_test_split(
        temp_idx,
        y_temp,
        test_size=1 / 3,
        stratify=y_temp,
        random_state=seed,
    )
    return train_idx, valid_idx, test_idx


# --------------------------------------------------------------------------- #
# 6️⃣  Fit pipeline, persist artefacts and write Parquet splits
# --------------------------------------------------------------------------- #
def fit_pipeline_and_save(
    cfg: dict,
    df: pd.DataFrame,
    train_idx: np.ndarray,
    valid_idx: np.ndarray,
    test_idx: np.ndarray,
) -> pathlib.Path:
    """
    Fits the preprocessing pipeline on train+valid,
    saves the pipeline, the feature order and the three splits.
    Returns the path to the saved pipeline.
    """
    target_col = resolve_target_column(df)

    # Split dataframes
    train_df = df.iloc[train_idx].reset_index(drop=True)
    valid_df = df.iloc[valid_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    # Build *unfitted* pipeline
    pipeline, _ = build_preprocess_pipeline(df)

    # Fit on the union of train+valid (features only)
    pipeline.fit(
        pd.concat([train_df.drop(columns=[target_col]), valid_df.drop(columns=[target_col])]),
        pd.concat([train_df[target_col], valid_df[target_col]]),
    )

    # ------------------------------------------------------------------- #
    # Persist the pipeline
    # ------------------------------------------------------------------- #
    pipeline_path = pathlib.Path(cfg["paths"]["pipeline"])
    pipeline_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, pipeline_path)

    # ------------------------------------------------------------------- #
    # Determine the feature order (column names after the ColumnTransformer)
    # ------------------------------------------------------------------- #
    dummy = pipeline.transform(train_df.drop(columns=[target_col]).head(1))
    if isinstance(dummy, np.ndarray):
        feature_order = [f"f{i}" for i in range(dummy.shape[1])]
    else:  # pandas DataFrame
        feature_order = dummy.columns.tolist()

    feature_order_path = pathlib.Path("models/feature_order.json")
    feature_order_path.parent.mkdir(parents=True, exist_ok=True)
    with feature_order_path.open("w", encoding="utf-8") as f:
        json.dump(feature_order, f, indent=2)

    # ------------------------------------------------------------------- #
    # Transform each split and write Parquet files (features + target)
    # ------------------------------------------------------------------- #
    splits_dir = pathlib.Path(cfg["paths"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    for name, split_df in [("train", train_df), ("valid", valid_df), ("test", test_df)]:
        X_transformed = pipeline.transform(split_df.drop(columns=[target_col]))
        if isinstance(X_transformed, np.ndarray):
            X_transformed = pd.DataFrame(
                X_transformed, columns=feature_order, index=split_df.index
            )
        else:  # already a DataFrame
            X_transformed = pd.DataFrame(
                X_transformed, index=split_df.index
            )  # column names are already set

        X_transformed["Survived"] = split_df[target_col].astype(int).values
        parquet_path = splits_dir / f"{name}.parquet"
        X_transformed.to_parquet(parquet_path, index=False)

    print(f"✅ Pipeline saved to {pipeline_path}")
    print(f"✅ Feature order saved to {feature_order_path}")
    print(f"✅ Parquet splits written to {splits_dir}")

    return pipeline_path


# --------------------------------------------------------------------------- #
# 7️⃣  LightGBM training
# --------------------------------------------------------------------------- #
def train_lightgbm(cfg: dict) -> pathlib.Path:
    """
    Trains LightGBM on the processed training split,
    evaluates on validation, and saves the model.
    Returns the path of the saved model.
    """
    splits_dir = pathlib.Path(cfg["paths"]["splits_dir"])
    train_path = splits_dir / "train.parquet"
    valid_path = splits_dir / "valid.parquet"

    train_df = pd.read_parquet(train_path)
    valid_df = pd.read_parquet(valid_path)

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

    # ------------------------------------------------------------------- #
    # Validation metrics
    # ------------------------------------------------------------------- #
    prob_valid = model.predict_proba(X_valid)[:, 1]
    pred_valid = (prob_valid >= 0.5).astype(int)

    auc = roc_auc_score(y_valid, prob_valid)
    acc = accuracy_score(y_valid, pred_valid)
    f1 = f1_score(y_valid, pred_valid)
    brier = brier_score_loss(y_valid, prob_valid)

    print(
        f"🔎 LightGBM validation → AUC: {auc:.4f}, ACC: {acc:.4f}, "
        f"F1: {f1:.4f}, Brier: {brier:.4f}"
    )

    # ------------------------------------------------------------------- #
    # Persist model
    # ------------------------------------------------------------------- #
    model_path = pathlib.Path(cfg["paths"]["lightgbm_model"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    # Also copy to the required deployment folder
    deploy_dir = pathlib.Path("./agent_workspace/trained_models")
    deploy_dir.mkdir(parents=True, exist_ok=True)
    copy_path = deploy_dir / model_path.name
    copy_path.write_bytes(model_path.read_bytes())

    print(f"✅ LightGBM model saved to {model_path}")
    print(f"✅ Model also copied to {copy_path}")

    return model_path


# --------------------------------------------------------------------------- #
# 8️⃣  Optional ONNX export + TorchScript wrapper
# --------------------------------------------------------------------------- #
def export_to_onnx_if_possible(cfg: dict, model_path: pathlib.Path) -> pathlib.Path | None:
    """
    Exports the LightGBM model to ONNX (if onnx/onnxruntime are installed)
    and returns the path to the ONNX file, otherwise returns None.
    """
    if onnx is None or ort is None:
        print("⚠️ ONNX/ONNX‑Runtime not installed – skipping ONNX export.")
        return None

    model = joblib.load(model_path)

    onnx_path = pathlib.Path(cfg["paths"]["onnx_model"])
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    # LightGBM's native ONNX exporter works via the Booster object
    booster = model.booster_
    booster.save_model(onnx_path, format="onnx")

    # Quick sanity check
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    print(f"✅ ONNX model exported to {onnx_path}")
    return onnx_path


def create_torchscript_wrapper_if_possible(
    cfg: dict, onnx_path: pathlib.Path | None
) -> pathlib.Path | None:
    """
    Wraps the ONNX model in a tiny torch.nn.Module and scripts it.
    Returns the path to the TorchScript file, or None if ONNX was not produced.
    """
    if onnx_path is None or ort is None:
        print("⚠️ ONNX model unavailable – skipping TorchScript wrapper.")
        return None

    import torch  # Imported lazily to avoid unnecessary dependency if not used

    class LightGBMOnnxModel(torch.nn.Module):
        def __init__(self, onnx_file: str):
            super().__init__()
            self.session = ort.InferenceSession(onnx_file)
            self.input_name = self.session.get_inputs()[0].name

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            np_input = x.detach().cpu().numpy().astype(np.float32)
            ort_out = self.session.run(None, {self.input_name: np_input})
            prob = torch.from_numpy(ort_out[0]).squeeze(1)
            return prob

    wrapper = LightGBMOnnxModel(str(onnx_path))
    scripted = torch.jit.script(wrapper)

    torch_path = pathlib.Path(cfg["paths"]["torch_script"])
    torch_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(torch_path))

    print(f"✅ TorchScript wrapper saved to {torch_path}")
    return torch_path


# --------------------------------------------------------------------------- #
# 9️⃣  Main orchestration
# --------------------------------------------------------------------------- #
def main() -> None:
    cfg = load_config()

    # ------------------------------------------------------------------- #
    # Load raw CSV (everything as string to avoid premature type conversion)
    # ------------------------------------------------------------------- #
    raw_path = pathlib.Path(cfg["paths"]["raw_csv"])
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw CSV not found at {raw_path}")
    df_raw = pd.read_csv(raw_path, dtype=str)

    # ------------------------------------------------------------------- #
    # Create stratified splits
    # ------------------------------------------------------------------- #
    seed = cfg["resources"]["seed"]
    train_idx, valid_idx, test_idx = stratified_split(df_raw, seed)

    # ------------------------------------------------------------------- #
    # Fit preprocessing pipeline and persist artefacts
    # ------------------------------------------------------------------- #
    _ = fit_pipeline_and_save(cfg, df_raw, train_idx, valid_idx, test_idx)

    # ------------------------------------------------------------------- #
    # Train LightGBM
    # ------------------------------------------------------------------- #
    model_path = train_lightgbm(cfg)

    # ------------------------------------------------------------------- #
    # Optional: ONNX export + TorchScript wrapper
    # ------------------------------------------------------------------- #
    onnx_path = export_to_onnx_if_possible(cfg, model_path)
    _ = create_torchscript_wrapper_if_possible(cfg, onnx_path)

    print("\n🚀 Pipeline finished successfully!")


if __name__ == "__main__":
    # The script can be launched from any directory; all path handling is robust.
    main()
