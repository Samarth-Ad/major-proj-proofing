
"""
Titanic Survival – end‑to‑end modeling pipeline
================================================

This script implements the complete workflow described in the “Titanic Survival”
MLOps play‑book:

1️⃣ Load the single source of truth `config.yaml`.
2️⃣ Read the raw CSV (as strings) and resolve the target column case‑insensitively.
3️⃣ Perform a stratified 70/20/10 split (train / valid / test) using the seed from the config.
4️⃣ Build a Scikit‑Learn preprocessing pipeline that:
   • Engineers `FamilySize`, `Title`, `AgeBin` (5 quantile bins) and `LogFare`.
   • Drops raw columns that are no longer needed.
   • Imputes numeric columns with median, categorical with most‑frequent.
   • Scales numerics with `RobustScaler`.
   • One‑Hot encodes low‑cardinality categoricals (≤ 9 distinct values).
   • Target‑encodes high‑cardinality categoricals (> 9 distinct values) – a fallback
     implementation is provided if `category_encoders` is unavailable.
5️⃣ Fit the pipeline on *train + validation* and transform all three splits.
6️⃣ Persist the processed splits as Parquet files.
7️⃣ Train a LightGBM binary classifier with the exact hyper‑parameters from the play‑book.
8️⃣ Evaluate on the validation split (ROC‑AUC, Accuracy, F1, Brier score) and print the metrics.
9️⃣ Persist artefacts:
   • `models/pipeline_preprocess.pkl`
   • `models/feature_order.json`
   • `models/lightgbm_best.pkl`
   • copy of the model to `./agent_workspace/trained_models/`
   • ONNX export (`onnx/lightgbm_best.onnx`)
   • TorchScript wrapper (`onnx/lightgbm_onnx.pt`)

The script is **self‑contained** – place it anywhere inside the repository (e.g.
`src/pipeline/run_titanic_pipeline.py`) and execute it from the repository root:

    python src/pipeline/run_titanic_pipeline.py
"""

import json
import shutil
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                             roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------


def load_config() -> dict:
    """Load the YAML configuration file located at the repository root."""
    cfg_path = Path(__file__).resolve().parents[3] / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config.yaml not found at {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_target_column(df: pd.DataFrame, target_name: str = "Survived") -> str:
    """
    Return the exact column name from ``df`` that matches ``target_name`` case‑insensitively.
    Raises ``KeyError`` if not found.
    """
    matches = [c for c in df.columns if c.lower() == target_name.lower()]
    if not matches:
        raise KeyError(f"Target column '{target_name}' not found in DataFrame columns.")
    return matches[0]


def stratified_split(df: pd.DataFrame, target_col: str, seed: int):
    """Return train/valid/test DataFrames (70/20/10) using stratified sampling."""
    from sklearn.model_selection import StratifiedShuffleSplit

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.30, random_state=seed
    )  # 30% will become valid+test
    train_idx, temp_idx = next(splitter.split(df, df[target_col]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    # Now split temp into valid (2/3 of 30% = 20%) and test (1/3 of 30% = 10%)
    splitter2 = StratifiedShuffleSplit(
        n_splits=1, test_size=1 / 3, random_state=seed
    )
    valid_idx, test_idx = next(splitter2.split(temp_df, temp_df[target_col]))
    valid_df = temp_df.iloc[valid_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    return train_df, valid_df, test_df


# ----------------------------------------------------------------------
# Fallback TargetEncoder (used when `category_encoders` is not installed)
# ----------------------------------------------------------------------


class SimpleTargetEncoder:
    """
    Very lightweight target encoder.
    For each categorical column it computes the mean of the target for each category.
    Unseen categories are replaced with the global target mean.
    """

    def __init__(self, cols: list, target_name: str):
        self.cols = cols
        self.target_name = target_name
        self.mapping_ = {}
        self.global_mean_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Ensure y is numeric
        y_numeric = pd.to_numeric(y, errors="coerce")
        self.global_mean_ = y_numeric.mean()
        for col in self.cols:
            # Compute mean target per category
            means = pd.concat([X[col], y_numeric], axis=1).groupby(col)[self.target_name].mean()
            self.mapping_[col] = means.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_enc = X.copy()
        for col in self.cols:
            mapping = self.mapping_.get(col, {})
            X_enc[col] = X_enc[col].map(mapping).fillna(self.global_mean_)
        return X_enc

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        return self.fit(X, y).transform(X)


# ----------------------------------------------------------------------
# Feature engineering helpers
# ----------------------------------------------------------------------


def extract_title(name: str) -> str:
    """Extract title from a passenger name (e.g., 'Mr.', 'Mrs.', 'Miss.', etc.)."""
    if pd.isna(name):
        return "Unknown"
    # Names are like "Braund, Mr. Owen Harris"
    parts = name.split(",")
    if len(parts) < 2:
        return "Unknown"
    title_part = parts[1].strip()
    title = title_part.split(".")[0].strip()
    return title if title else "Unknown"


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create FamilySize, Title, AgeBin, LogFare and drop raw columns later."""
    df = df.copy()

    # FamilySize = SibSp + Parch + 1 (the passenger themselves)
    if {"SibSp", "Parch"}.issubset(df.columns):
        df["FamilySize"] = (
            pd.to_numeric(df["SibSp"], errors="coerce")
            + pd.to_numeric(df["Parch"], errors="coerce")
            + 1
        )

    # Title extraction from Name
    if "Name" in df.columns:
        df["Title"] = df["Name"].apply(extract_title)

    # LogFare (guard against zero / missing)
    if "Fare" in df.columns:
        fare = pd.to_numeric(df["Fare"], errors="coerce").fillna(0)
        df["LogFare"] = np.log1p(fare)

    # Age binning – will be performed later inside the pipeline (needs numeric Age)
    return df


# ----------------------------------------------------------------------
# Main pipeline
# ----------------------------------------------------------------------


def main():
    warnings.filterwarnings("ignore")  # suppress noisy sklearn warnings

    # ------------------------------------------------------------------
    # 1️⃣ Load configuration
    # ------------------------------------------------------------------
    cfg = load_config()
    paths_cfg = cfg["paths"]
    resources_cfg = cfg["resources"]
    seed = resources_cfg["seed"]
    n_jobs = resources_cfg["n_jobs"]

    # Resolve absolute paths
    raw_csv_path = Path(paths_cfg["raw_csv"])
    splits_dir = Path(paths_cfg["splits_dir"])
    model_dir = Path("models")
    onnx_dir = Path("onnx")
    agent_workspace_dir = Path("agent_workspace") / "trained_models"

    # Ensure output directories exist
    for d in [splits_dir, model_dir, onnx_dir, agent_workspace_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2️⃣ Load raw CSV (as strings to avoid dtype inference issues)
    # ------------------------------------------------------------------
    df_raw = pd.read_csv(raw_csv_path, dtype=str)

    # Resolve target column name (case‑insensitive)
    target_col = resolve_target_column(df_raw, "Survived")

    # ------------------------------------------------------------------
    # 3️⃣ Feature engineering (adds new columns, keeps raw ones for later dropping)
    # ------------------------------------------------------------------
    df_fe = add_engineered_features(df_raw)

    # ------------------------------------------------------------------
    # 4️⃣ Train/valid/test split (stratified)
    # ------------------------------------------------------------------
    train_df, valid_df, test_df = stratified_split(df_fe, target_col, seed)

    # ------------------------------------------------------------------
    # 5️⃣ Build preprocessing pipeline
    # ------------------------------------------------------------------
    # Identify column groups after engineering
    numeric_cols = [
        c
        for c in train_df.select_dtypes(include=["int64", "float64"]).columns
        if c != target_col
    ]
    # Some columns may still be object because we read everything as str; force conversion
    for col in ["Age", "Fare", "SibSp", "Parch"]:
        if col in train_df.columns:
            train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
            valid_df[col] = pd.to_numeric(valid_df[col], errors="coerce")
            test_df[col] = pd.to_numeric(test_df[col], errors="coerce")
    # Re‑detect numeric after conversion
    numeric_cols = [
        c
        for c in train_df.columns
        if c != target_col and pd.api.types.is_numeric_dtype(train_df[c])
    ]

    # Categorical columns (object dtype)
    categorical_cols = [
        c
        for c in train_df.columns
        if c != target_col and train_df[c].dtype == "object"
    ]

    # Determine low‑ vs high‑cardinality categoricals
    low_cardinality = [
        c for c in categorical_cols if train_df[c].nunique() <= 9
    ]
    high_cardinality = [
        c for c in categorical_cols if train_df[c].nunique() > 9
    ]

    # Age binning transformer (5 quantile bins) – only if Age exists
    from sklearn.base import BaseEstimator, TransformerMixin

    class AgeBinner(BaseEstimator, TransformerMixin):
        def __init__(self, n_bins=5):
            self.n_bins = n_bins
            self.bin_edges_ = None

        def fit(self, X, y=None):
            if "Age" not in X.columns:
                return self
            ages = pd.to_numeric(X["Age"], errors="coerce")
            self.bin_edges_ = np.quantile(
                ages.dropna(), q=np.linspace(0, 1, self.n_bins + 1)
            )
            return self

        def transform(self, X):
            X = X.copy()
            if "Age" not in X.columns or self.bin_edges_ is None:
                X["AgeBin"] = np.nan
                return X
            ages = pd.to_numeric(X["Age"], errors="coerce")
            X["AgeBin"] = pd.cut(
                ages,
                bins=self.bin_edges_,
                include_lowest=True,
                duplicates="drop",
            ).astype(str)
            return X

    # ColumnTransformer pieces
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    low_card_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Try to import the real TargetEncoder; fall back otherwise
    try:
        from category_encoders import TargetEncoder as CatBoostTargetEncoder
    except Exception:  # pragma: no cover
        CatBoostTargetEncoder = None

    if high_cardinality:
        if CatBoostTargetEncoder is not None:
            high_card_pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("target_enc", CatBoostTargetEncoder(cols=high_cardinality, smoothing=1.0)),
                ]
            )
        else:
            # Use our simple fallback
            high_card_pipe = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("target_enc", SimpleTargetEncoder(cols=high_cardinality, target_name=target_col)),
                ]
            )
    else:
        high_card_pipe = "passthrough"

    # Assemble full ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("age_bin", AgeBinner(), ["Age"] if "Age" in train_df.columns else []),
            ("num", numeric_pipe, numeric_cols),
            ("low_cat", low_card_pipe, low_cardinality),
            ("high_cat", high_card_pipe, high_cardinality),
        ],
        remainder="drop",  # drop columns not listed (e.g., raw Name, Ticket, Cabin, etc.)
    )

    # Full pipeline (preprocessor only – target column handled separately)
    pipeline = Pipeline(steps=[("preprocess", preprocessor)])

    # ------------------------------------------------------------------
    # 6️⃣ Fit pipeline on train + validation (to maximise data for imputation/encoding)
    # ------------------------------------------------------------------
    train_valid_df = pd.concat([train_df, valid_df], axis=0).reset_index(drop=True)
    X_train_valid = train_valid_df.drop(columns=[target_col])
    y_train_valid = train_valid_df[target_col]

    # Fit
    pipeline.fit(X_train_valid, y_train_valid)

    # Persist pipeline
    pipeline_path = model_dir / "pipeline_preprocess.pkl"
    joblib.dump(pipeline, pipeline_path)
    print(f"✅ Pipeline saved → {pipeline_path}")

    # ------------------------------------------------------------------
    # 7️⃣ Save feature order (the column order after ColumnTransformer)
    # ------------------------------------------------------------------
    # We can obtain the order by transforming a single row and inspecting columns
    dummy_row = X_train_valid.head(1)
    transformed_dummy = pipeline.transform(dummy_row)
    if isinstance(transformed_dummy, np.ndarray):
        # ColumnTransformer with OneHotEncoder returns a sparse matrix; convert to dense
        transformed_dummy = transformed_dummy.toarray()
    feature_names = []
    # Retrieve names from each transformer
    for name, transformer, cols in pipeline.named_steps["preprocess"].transformers_:
        if transformer == "drop" or transformer == "passthrough":
            continue
        if name == "age_bin":
            feature_names.append("AgeBin")
        elif name == "num":
            feature_names.extend(cols)
        elif name == "low_cat":
            # OneHotEncoder feature names
            ohe = transformer.named_steps["onehot"]
            ohe_features = ohe.get_feature_names_out(cols)
            feature_names.extend(ohe_features.tolist())
        elif name == "high_cat":
            # TargetEncoder returns numeric columns with same names
            feature_names.extend(cols)

    feature_order_path = model_dir / "feature_order.json"
    with open(feature_order_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)
    print(f"✅ Feature order saved → {feature_order_path}")

    # ------------------------------------------------------------------
    # 8️⃣ Transform and persist splits as Parquet
    # ------------------------------------------------------------------
    def transform_and_save(df_split: pd.DataFrame, split_name: str):
        X = df_split.drop(columns=[target_col])
        y = df_split[target_col]
        X_trans = pipeline.transform(X)
        # Ensure dense numpy array
        if hasattr(X_trans, "toarray"):
            X_trans = X_trans.toarray()
        # Build DataFrame with correct column order
        df_out = pd.DataFrame(X_trans, columns=feature_names)
        df_out[target_col] = pd.to_numeric(y, errors="coerce")
        out_path = splits_dir / f"{split_name}.parquet"
        df_out.to_parquet(out_path, index=False)
        print(f"✅ {split_name.capitalize()} split written → {out_path}")

    for split_df, name in [(train_df, "train"), (valid_df, "valid"), (test_df, "test")]:
        transform_and_save(split_df, name)

    # ------------------------------------------------------------------
    # 9️⃣ Train LightGBM on the processed training data
    # ------------------------------------------------------------------
    # Load the processed train/valid splits (they are small, so we can read them again)
    train_parquet = splits_dir / "train.parquet"
    valid_parquet = splits_dir / "valid.parquet"

    train_processed = pd.read_parquet(train_parquet)
    valid_processed = pd.read_parquet(valid_parquet)

    X_train = train_processed.drop(columns=[target_col])
    y_train = train_processed[target_col]

    X_valid = valid_processed.drop(columns=[target_col])
    y_valid = valid_processed[target_col]

    # LightGBM dataset objects
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_valid = lgb.Dataset(X_valid, label=y_valid, reference=lgb_train)

    # Hyper‑parameters from the play‑book
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
        "random_state": seed,
        "n_jobs": n_jobs,
    }

    # Train with early stopping
    gbm = lgb.train(
        lgb_params,
        lgb_train,
        valid_sets=[lgb_valid],
        early_stopping_rounds=20,
        verbose_eval=False,
    )

    # ------------------------------------------------------------------
    # 10️⃣ Evaluate on validation set
    # ------------------------------------------------------------------
    y_valid_pred_proba = gbm.predict(X_valid, num_iteration=gbm.best_iteration)
    y_valid_pred = (y_valid_pred_proba >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_valid, y_valid_pred_proba)
    acc = accuracy_score(y_valid, y_valid_pred)
    f1 = f1_score(y_valid, y_valid_pred)
    brier = brier_score_loss(y_valid, y_valid_pred_proba)

    print("\n🔎 Validation metrics")
    print(f"  ROC‑AUC : {roc_auc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1      : {f1:.4f}")
    print(f"  Brier  : {brier:.4f}")

    # ------------------------------------------------------------------
    # 11️⃣ Persist LightGBM model
    # ------------------------------------------------------------------
    lgb_model_path = model_dir / "lightgbm_best.pkl"
    joblib.dump(gbm, lgb_model_path)
    print(f"\n✅ LightGBM model saved → {lgb_model_path}")

    # Copy to agent_workspace (required by downstream deployment)
    copy_dest = agent_workspace_dir / "lightgbm_best.pkl"
    shutil.copy2(lgb_model_path, copy_dest)
    print(f"✅ Model also copied to {copy_dest}")

    # ------------------------------------------------------------------
    # 12️⃣ Export to ONNX
    # ------------------------------------------------------------------
    onnx_path = onnx_dir / "lightgbm_best.onnx"
    # LightGBM's native export
    gbm.booster_.save_model(str(onnx_path), format="onnx")
    print(f"\n✅ ONNX model exported → {onnx_path}")

    # ------------------------------------------------------------------
    # 13️⃣ Verify ONNX input name matches feature order (rename if needed)
    # ------------------------------------------------------------------
    import onnx

    onnx_model = onnx.load(str(onnx_path))
    # The first input should be the feature tensor
    onnx_input = onnx_model.graph.input[0]
    expected_name = feature_names[0]  # first feature name
    if onnx_input.name != expected_name:
        print(
            f"⚠️ ONNX input name ('{onnx_input.name}') does not match expected ('{expected_name}'). Renaming..."
        )
        onnx_input.name = expected_name
        onnx.save(onnx_model, str(onnx_path))
        print("✅ ONNX input name corrected.")
    else:
        print("✅ ONNX input name matches the feature order.")

    # ------------------------------------------------------------------
    # 14️⃣ Build tiny TorchScript wrapper around ONNX Runtime
    # ------------------------------------------------------------------
    import onnxruntime as ort

    class ONNXWrapper(torch.nn.Module):
        def __init__(self, onnx_path: str):
            super().__init__()
            self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
            # Input name is taken from the ONNX graph
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x is expected to be of shape (batch, n_features)
            np_input = x.detach().cpu().numpy().astype(np.float32)
            ort_out = self.session.run([self.output_name], {self.input_name: np_input})
            # LightGBM ONNX returns shape (batch, 1)
            probs = np.squeeze(ort_out[0])
            return torch.from_numpy(probs.astype(np.float32))

    # Instantiate and script
    wrapper = ONNXWrapper(str(onnx_path))
    scripted = torch.jit.script(wrapper)
    torchscript_path = onnx_dir / "lightgbm_onnx.pt"
    scripted.save(str(torchscript_path))
    print(f"\n✅ TorchScript wrapper saved → {torchscript_path}")

    # ------------------------------------------------------------------
    # End of pipeline
    # ------------------------------------------------------------------
    print("\n🎉 Pipeline completed successfully!")


if __name__ == "__main__":
    main()
