
"""
Titanic survival end‑to‑end pipeline

What it does
------------
1️⃣ Load a single source of truth configuration (config.yaml).  
2️⃣ Read the raw Titanic CSV (path defined in the config).  
3️⃣ Resolve the exact target column name in a case‑insensitive way.  
4️⃣ Build a preprocessing pipeline:
   • CustomFeatureEngineer – creates FamilySize, Title, AgeBin, LogFare and drops raw columns.
   • Numeric columns → Median imputer → RobustScaler.
   • Low‑cardinality categorical (≤ 9 distinct) → Most‑frequent imputer → OneHotEncoder.
   • High‑cardinality categorical (> 9 distinct) → Most‑frequent imputer → TargetEncoder
     (if the *category‑encoders* package is unavailable a very simple ordinal encoder is used).
5️⃣ Perform a stratified split (70 % train, 20 % validation, 10 % test).  
6️⃣ Fit the pipeline on the *train + validation* data, transform all splits and
   persist the pipeline and the transformed splits (Parquet).  
7️⃣ Train a LightGBM classifier on the processed training set, evaluate on validation,
   and log a few metrics.  
8️⃣ Save the trained model to ``./agent_workspace/trained_models``.  
9️⃣ (Optional) Export the model to ONNX and wrap it in a TorchScript module
   – this step is executed only if the required libraries are present.

The script is self‑contained: it contains all imports inside ``try/except`` blocks,
uses ``pathlib`` for robust path handling and never hard‑codes absolute paths
(except the raw CSV path that lives in ``config.yaml``).  Running the script from
the repository root (or any sub‑directory) works out‑of‑the‑box.
"""

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

# --------------------------------------------------------------------------- #
# Optional third‑party imports – we provide graceful fall‑backs
# --------------------------------------------------------------------------- #
try:
    from category_encoders import TargetEncoder  # type: ignore
except Exception:  # pragma: no cover
    # Very small fallback: ordinal encoding based on pandas factorize
    class TargetEncoder:  # pylint: disable=too-few-public-methods
        """Fallback that mimics the fit/transform API of category_encoders.TargetEncoder."""
        def __init__(self):
            self.mapping_ = {}

        def fit(self, X, y=None):
            # X is a DataFrame or Series; we store the factorization mapping
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

# --------------------------------------------------------------------------- #
# 1️⃣  Configuration loader
# --------------------------------------------------------------------------- #
def load_config() -> dict:
    """
    Looks for ``config.yaml`` in three possible locations:
    1. The directory of this script (repo root)
    2. The current working directory
    3. A parent directory named ``experiment`` (fallback used in some notebooks)
    """
    possible = [
        pathlib.Path(__file__).resolve().parents[0] / "config.yaml",
        pathlib.Path.cwd() / "config.yaml",
        pathlib.Path.cwd() / "experiment" / "config.yaml",
    ]
    for p in possible:
        if p.is_file():
            with p.open("r", encoding="utf-8") as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(
        "Could not locate config.yaml. Searched paths:\n"
        + "\n".join(str(p) for p in possible)
    )


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
from sklearn.base import BaseEstimator, TransformerMixin


class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Adds:
        * FamilySize = SibSp + Parch + 1
        * Title      = extracted from Name
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
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


def build_preprocess_pipeline(df: pd.DataFrame) -> Tuple[Pipeline, List[str]]:
    """
    Parameters
    ----------
    df : pd.DataFrame
        The *raw* dataframe (including the target column).

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        Unfitted pipeline that expects only feature columns.
    feature_names : List[str]
        List of column names *after* feature engineering (used later for ONNX export).
    """
    target_col = resolve_target_column(df)

    # ------------------------------------------------------------------- #
    # Run the feature‑engineer once to discover column types
    # ------------------------------------------------------------------- #
    fe = CustomFeatureEngineer()
    fe.fit(df.drop(columns=[target_col]))
    engineered = fe.transform(df.drop(columns=[target_col]))

    numeric_cols = engineered.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = engineered.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    # Split categorical columns by cardinality
    low_cardinality = [
        c for c in categorical_cols if engineered[c].nunique() <= 9
    ]
    high_cardinality = [
        c for c in categorical_cols if engineered[c].nunique() > 9
    ]

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

    # The order of columns after the ColumnTransformer is not directly exposed.
    # We will obtain it later after fitting (by transforming a single row).
    return full_pipe, numeric_cols + low_cardinality + high_cardinality


# --------------------------------------------------------------------------- #
# 5️⃣  Stratified split helper (70/20/10)
# --------------------------------------------------------------------------- #
from sklearn.model_selection import train_test_split


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
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score


def train_lightgbm(cfg: dict) -> pathlib.Path:
    """
    Trains LightGBM on the processed training split,
    evaluates on validation, and saves the model to
    ``./agent_workspace/trained_models``.
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

    # Load feature order – needed only for sanity checks
    with open("models/feature_order.json", "r", encoding="utf-8") as f:
        feature_order = json.load(f)

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


def create_torchscript_wrapper_if_possible(cfg: dict, onnx_path: pathlib.Path | None) -> pathlib.Path | None:
    """
    Wraps the ONNX model in a tiny torch.nn.Module and scripts it.
    Returns the path to the TorchScript file, or None if ONNX was not produced.
    """
    if onnx_path is None or ort is None:
        print("⚠️ ONNX model unavailable – skipping TorchScript wrapper.")
        return None

    # Load feature order (used only for documentation; the wrapper does not need it)
    with open("models/feature_order.json", "r", encoding="utf-8") as f:
        feature_order = json.load(f)

    class LightGBMOnnxModel(torch.nn.Module):
        def __init__(self, onnx_file: str):
            super().__init__()
            self.session = ort.InferenceSession(onnx_file)
            self.input_name = self.session.get_inputs()[0].name

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            np_input = x.detach().cpu().numpy().astype(np.float32)
            ort_out = self.session.run(None, {self.input_name: np_input})
            # ort_out[0] shape: (batch, 1) – probability of class 1
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
