
"""
Titanic survival end‑to‑end tabular classification pipeline

Steps performed:
1️⃣ Load configuration (config.yaml)
2️⃣ Read the raw Titanic CSV (path defined in the config)
3️⃣ Resolve the exact target column name (case‑insensitive)
4️⃣ Build a preprocessing pipeline:
   - CustomFeatureEngineer (FamilySize, Title, AgeBin, LogFare, column drops)
   - Numeric → MedianImputer → RobustScaler
   - Low‑cardinality categorical → MostFrequentImputer → OneHotEncoder
   - High‑cardinality categorical → MostFrequentImputer → TargetEncoder
5️⃣ Create stratified train / validation / test splits (70/20/10)
6️⃣ Fit the pipeline on train+valid, transform all splits and persist them as Parquet
7️⃣ Train a LightGBM classifier on the transformed training data
8️⃣ Persist the trained model to ``./agent_workspace/trained_models``
9️⃣ Export the LightGBM model to ONNX and create a tiny TorchScript wrapper
10️⃣ Provide a minimal Gradio demo that accepts raw Titanic rows,
    runs the preprocessing pipeline, feeds the data to the TorchScript model,
    and returns the survival probability.

All artefacts (pipeline, feature order, LightGBM model, ONNX model,
TorchScript wrapper) are saved under the repository‑relative ``models`` /
``onnx`` folders as described in the cookbook.
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
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
import onnx
import onnxruntime as ort
import torch
import gradio as gr

# --------------------------------------------------------------------------- #
# 1️⃣  Utility helpers
# --------------------------------------------------------------------------- #
def load_config() -> dict:
    """Load ``config.yaml`` located at the repository root."""
    repo_root = pathlib.Path(__file__).resolve().parents[0]  # assumes script is in repo root
    cfg_path = repo_root / "config.yaml"
    if not cfg_path.is_file():
        raise FileNotFoundError(f"config.yaml not found at {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def resolve_target_column(df: pd.DataFrame) -> str:
    """Return the exact column name that matches 'survived' case‑insensitively."""
    candidates = [c for c in df.columns if c.lower() == "survived"]
    if not candidates:
        raise KeyError("Target column 'Survived' not found in the raw CSV.")
    return candidates[0]  # the column name as it appears in the file


# --------------------------------------------------------------------------- #
# 2️⃣  Custom feature engineering transformer
# --------------------------------------------------------------------------- #
class CustomFeatureEngineer:
    """
    Scikit‑Learn compatible transformer that creates:
    * FamilySize = SibSp + Parch + 1
    * Title      = extracted from Name
    * AgeBin     = 5 quantile bins (as string)
    * LogFare    = log1p(Fare)
    and drops the raw columns that are no longer needed.
    """

    def __init__(self):
        self.age_bins_ = None  # will hold the bin edges after fit

    def fit(self, X: pd.DataFrame, y=None):
        # Compute quantile edges for Age (ignore missing values)
        ages = X["Age"].astype(float)
        self.age_bins_ = pd.qcut(
            ages.dropna(), q=5, duplicates="drop", retbins=True
        )[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # ---- FamilySize ----------------------------------------------------
        X["FamilySize"] = (
            X["SibSp"].astype(float) + X["Parch"].astype(float) + 1
        ).astype(int)

        # ---- Title ---------------------------------------------------------
        # Extract the word before the first '.' in the Name column
        X["Title"] = X["Name"].fillna("").apply(
            lambda n: re.search(r",\s*([^.]*)\.", n).group(1).strip()
            if re.search(r",\s*([^.]*)\.", n)
            else "Unknown"
        )

        # ---- AgeBin --------------------------------------------------------
        # Bin ages using the edges learned during fit; unknown ages become NaN
        X["AgeBin"] = pd.cut(
            X["Age"].astype(float), bins=self.age_bins_, labels=False, duplicates="drop"
        ).astype(str)

        # ---- LogFare -------------------------------------------------------
        X["LogFare"] = np.log1p(X["Fare"].astype(float))

        # ---- Drop raw columns ------------------------------------------------
        cols_to_drop = ["Name", "Ticket", "Cabin", "Age", "Fare", "SibSp", "Parch"]
        X.drop(columns=cols_to_drop, inplace=True, errors="ignore")

        return X

    # Scikit‑Learn compatibility helpers
    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


# --------------------------------------------------------------------------- #
# 3️⃣  Build the full preprocessing pipeline
# --------------------------------------------------------------------------- #
def build_preprocess_pipeline(df: pd.DataFrame) -> Tuple[Pipeline, List[str]]:
    """
    Returns:
        pipeline: fitted sklearn Pipeline (including CustomFeatureEngineer)
        feature_order: list of column names after the ColumnTransformer step
    """
    # ------------------------------------------------------------------- #
    # Detect target column and separate it (the pipeline works on features only)
    # ------------------------------------------------------------------- #
    target_col = resolve_target_column(df)
    y = df[target_col].astype(int)

    # ------------------------------------------------------------------- #
    # Build the pipeline
    # ------------------------------------------------------------------- #
    # 1️⃣ Custom feature engineering
    fe = CustomFeatureEngineer()

    # 2️⃣ After feature engineering we need to know which columns are numeric / categorical
    # We'll fit the FE on the whole dataframe (including target) just to get column types.
    fe.fit(df.drop(columns=[target_col]))
    engineered = fe.transform(df.drop(columns=[target_col]))

    # Identify column types
    numeric_cols = engineered.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = engineered.select_dtypes(include=["object", "category", "string"]).columns.tolist()

    # Split categorical into low / high cardinality
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

    # Fit on train+valid later – here we just return the unfitted pipeline
    # To obtain the feature order we need a fitted pipeline; this will be done
    # after the train/valid split (see `fit_and_save_pipeline` below).
    return full_pipe, numeric_cols + low_cardinality + high_cardinality


def fit_and_save_pipeline(
    cfg: dict, df: pd.DataFrame, train_idx: np.ndarray, valid_idx: np.ndarray
):
    """
    Fit the preprocessing pipeline on train+valid, persist it,
    and write transformed splits as Parquet.
    """
    target_col = resolve_target_column(df)
    y = df[target_col].astype(int)

    # Split indices
    train_df = df.iloc[train_idx].reset_index(drop=True)
    valid_df = df.iloc[valid_idx].reset_index(drop=True)

    # Build pipeline (unfitted)
    pipeline, _ = build_preprocess_pipeline(df)

    # Fit on combined train+valid
    pipeline.fit(
        pd.concat([train_df.drop(columns=[target_col]), valid_df.drop(columns=[target_col])]),
        pd.concat([train_df[target_col], valid_df[target_col]]),
    )

    # Persist pipeline
    pipeline_path = pathlib.Path(cfg["paths"]["pipeline"])
    pipeline_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, pipeline_path)

    # ------------------------------------------------------------------- #
    # Save feature order (column names after transformation)
    # ------------------------------------------------------------------- #
    # Transform a single row to get the column names
    dummy = pipeline.transform(train_df.drop(columns=[target_col]).head(1))
    if isinstance(dummy, np.ndarray):
        feature_order = [f"f{i}" for i in range(dummy.shape[1])]
    else:  # pandas DataFrame
        feature_order = dummy.columns.tolist()

    feature_order_path = pathlib.Path("models/feature_order.json")
    feature_order_path.parent.mkdir(parents=True, exist_ok=True)
    with open(feature_order_path, "w", encoding="utf-8") as f:
        json.dump(feature_order, f, indent=2)

    # ------------------------------------------------------------------- #
    # Transform each split and write Parquet (features + target)
    # ------------------------------------------------------------------- #
    splits_dir = pathlib.Path(cfg["paths"]["splits_dir"])
    splits_dir.mkdir(parents=True, exist_ok=True)

    for name, split_df in [("train", train_df), ("valid", valid_df), ("test", df.iloc[test_idx])]:
        X_transformed = pipeline.transform(split_df.drop(columns=[target_col]))
        # Ensure we have a DataFrame for easier concatenation
        if isinstance(X_transformed, np.ndarray):
            X_transformed = pd.DataFrame(
                X_transformed, columns=feature_order, index=split_df.index
            )
        else:
            X_transformed = pd.DataFrame(
                X_transformed, index=split_df.index
            )  # column names already set

        X_transformed["Survived"] = split_df[target_col].values
        parquet_path = splits_dir / f"{name}.parquet"
        X_transformed.to_parquet(parquet_path, index=False)

    print(f"✅ Preprocessing pipeline saved to {pipeline_path}")
    print(f"✅ Feature order saved to {feature_order_path}")
    print(f"✅ Split Parquet files written to {splits_dir}")


# --------------------------------------------------------------------------- #
# 4️⃣  Train LightGBM on the processed data
# --------------------------------------------------------------------------- #
def train_lightgbm(cfg: dict) -> pathlib.Path:
    """Trains LightGBM, logs metrics, and saves the model. Returns model path."""
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

    # Validation metrics
    y_pred_proba = model.predict_proba(X_valid)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc = roc_auc_score(y_valid, y_pred_proba)
    brier = brier_score_loss(y_valid, y_pred_proba)
    acc = accuracy_score(y_valid, y_pred)
    f1 = f1_score(y_valid, y_pred)

    print(f"🔎 LightGBM validation metrics → AUC: {auc:.4f}, ACC: {acc:.4f}, F1: {f1:.4f}, Brier: {brier:.4f}")

    # Persist model
    model_path = pathlib.Path(cfg["paths"]["lightgbm_model"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    # Also copy to the required deployment folder
    deploy_dir = pathlib.Path("./agent_workspace/trained_models")
    deploy_dir.mkdir(parents=True, exist_ok=True)
    copy_path = deploy_dir / model_path.name
    copy_path.write_bytes(model_path.read_bytes())

    print(f"✅ LightGBM model saved to {model_path} and copied to {copy_path}")
    return model_path


# --------------------------------------------------------------------------- #
# 5️⃣  Export LightGBM → ONNX
# --------------------------------------------------------------------------- #
def export_to_onnx(cfg: dict, model_path: pathlib.Path) -> pathlib.Path:
    """Converts the LightGBM model to ONNX and saves it."""
    # Load model
    model = joblib.load(model_path)

    # Load feature order – this determines the input shape
    with open("models/feature_order.json", "r", encoding="utf-8") as f:
        feature_order = json.load(f)

    # LightGBM's built‑in ONNX exporter works via the Booster object
    booster = model.booster_
    onnx_path = pathlib.Path(cfg["paths"]["onnx_model"])
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    # Export – the input name will be "input"
    booster.save_model(onnx_path, format="onnx")

    # Verify that the ONNX graph expects the correct number of inputs
    onnx_model = onnx.load(onnx_path)
    # LightGBM creates a single input tensor named "input". Its shape is (None, n_features)
    # No further renaming is required for our wrapper.
    onnx.checker.check_model(onnx_model)

    print(f"✅ ONNX model exported to {onnx_path}")
    return onnx_path


# --------------------------------------------------------------------------- #
# 6️⃣  TorchScript wrapper around the ONNX model
# --------------------------------------------------------------------------- #
class LightGBMOnnxModel(torch.nn.Module):
    """
    Tiny wrapper that holds an ONNX Runtime session.
    The forward method receives a ``torch.Tensor`` of shape (batch, n_features)
    and returns a ``torch.Tensor`` of probabilities for class 1.
    """

    def __init__(self, onnx_path: str, feature_order: List[str]):
        super().__init__()
        self.session = ort.InferenceSession(onnx_path)
        self.input_name = self.session.get_inputs()[0].name
        self.feature_order = feature_order

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure numpy array of type float32
        np_input = x.detach().cpu().numpy().astype(np.float32)
        ort_out = self.session.run(None, {self.input_name: np_input})
        # ort_out[0] shape: (batch, 1) – probability of class 1
        prob = torch.from_numpy(ort_out[0]).squeeze(1)  # (batch,)
        return prob


def create_torchscript_wrapper(cfg: dict) -> pathlib.Path:
    """Creates a TorchScript file (lightgbm_onnx.pt) that can be loaded in production."""
    # Load feature order
    with open("models/feature_order.json", "r", encoding="utf-8") as f:
        feature_order = json.load(f)

    onnx_path = pathlib.Path(cfg["paths"]["onnx_model"])
    wrapper = LightGBMOnnxModel(str(onnx_path), feature_order)

    scripted = torch.jit.script(wrapper)
    torch_path = pathlib.Path(cfg["paths"]["torch_script"])
    torch_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(torch_path))

    print(f"✅ TorchScript wrapper saved to {torch_path}")
    return torch_path


# --------------------------------------------------------------------------- #
# 7️⃣  Gradio demo (raw JSON → probability)
# --------------------------------------------------------------------------- #
def load_artifacts(cfg: dict):
    """Load pipeline, torchscript model and feature order for inference."""
    pipeline = joblib.load(cfg["paths"]["pipeline"])
    torch_path = pathlib.Path(cfg["paths"]["torch_script"])
    model = torch.jit.load(str(torch_path), map_location="cpu")
    # The model expects a torch.Tensor of shape (batch, n_features)
    # Feature order is stored inside the wrapper (accessible via model.feature_order)
    return pipeline, model


def predict_survival(raw_json: dict) -> dict:
    """
    Gradio callable.
    ``raw_json`` must contain the original Titanic columns (except the target).
    Returns a dict with probability and binary prediction.
    """
    # Load config only once (module‑level caching could be added for speed)
    cfg = load_config()
    pipeline, torch_model = load_artifacts(cfg)

    # Convert incoming dict to DataFrame (single row)
    input_df = pd.DataFrame([raw_json])

    # Apply preprocessing pipeline (produces a numpy array)
    X_processed = pipeline.transform(input_df)

    # Ensure we have a 2‑D numpy array
    if isinstance(X_processed, pd.DataFrame):
        X_processed = X_processed.values.astype(np.float32)
    else:
        X_processed = X_processed.astype(np.float32)

    # Torch inference
    with torch.no_grad():
        tensor_input = torch.from_numpy(X_processed)
        prob = torch_model(tensor_input).numpy().item()  # scalar

    pred = int(prob >= 0.5)
    return {"survival_probability": float(prob), "prediction": pred}


def launch_gradio_demo():
    """Creates and launches a minimal Gradio interface."""
    # Load a sample row to auto‑populate the UI
    sample_row = {
        "PassengerId": 1,
        "Pclass": 3,
        "Name": "Braund, Mr. Owen Harris",
        "Sex": "male",
        "Age": 22,
        "SibSp": 1,
        "Parch": 0,
        "Ticket": "A/5 21171",
        "Fare": 7.25,
        "Cabin": "",
        "Embarked": "S",
    }

    iface = gr.Interface(
        fn=predict_survival,
        inputs=gr.JSON(label="Raw Titanic row (JSON)"),
        outputs=gr.JSON(label="Prediction"),
        examples=[[sample_row]],
        title="Titanic Survival Prediction (Tabular Classification)",
        description="Upload a JSON object containing the original Titanic columns (except `Survived`). The model returns the probability of survival and a binary prediction (0 = did not survive, 1 = survived).",
        allow_flagging="never",
    )
    iface.launch(share=False)


# --------------------------------------------------------------------------- #
# 8️⃣  Main orchestration
# --------------------------------------------------------------------------- #
def main():
    cfg = load_config()

    # ------------------------------------------------------------------- #
    # Load raw CSV
    # ------------------------------------------------------------------- #
    raw_csv_path = pathlib.Path(cfg["paths"]["raw_csv"])
    if not raw_csv_path.is_file():
        raise FileNotFoundError(f"Raw CSV not found at {raw_csv_path}")
    df_raw = pd.read_csv(raw_csv_path, dtype=str)  # read everything as string

    # ------------------------------------------------------------------- #
    # Create stratified splits (70/20/10) using the resolved target column
    # ------------------------------------------------------------------- #
    target_col = resolve_target_column(df_raw)
    y = df_raw[target_col].astype(int).values
    rng = np.random.default_rng(cfg["resources"]["seed"])

    # Shuffle indices
    indices = np.arange(len(df_raw))
    rng.shuffle(indices)

    # Stratified split – we will use sklearn's `train_test_split` twice
    from sklearn.model_selection import train_test_split

    train_idx, temp_idx, _, _ = train_test_split(
        indices,
        y,
        test_size=0.30,
        stratify=y,
        random_state=cfg["resources"]["seed"],
    )
    valid_idx, test_idx, _, _ = train_test_split(
        temp_idx,
        y[train_idx.shape[0] :],  # corresponding labels for temp set
        test_size=1 / 3,  # 0.10 overall (since temp is 30%)
        stratify=y[temp_idx],
        random_state=cfg["resources"]["seed"],
    )

    # ------------------------------------------------------------------- #
    # Fit preprocessing pipeline, persist artefacts and write Parquet splits
    # ------------------------------------------------------------------- #
    fit_and_save_pipeline(cfg, df_raw, train_idx, valid_idx)

    # ------------------------------------------------------------------- #
    # Train LightGBM
    # ------------------------------------------------------------------- #
    model_path = train_lightgbm(cfg)

    # ------------------------------------------------------------------- #
    # Export to ONNX and create TorchScript wrapper
    # ------------------------------------------------------------------- #
    onnx_path = export_to_onnx(cfg, model_path)
    _ = create_torchscript_wrapper(cfg)

    # ------------------------------------------------------------------- #
    # (Optional) launch Gradio demo – comment out if you only need artefacts
    # ------------------------------------------------------------------- #
    print("\n🚀 Launching Gradio demo – close the browser tab to finish.")
    launch_gradio_demo()


if __name__ == "__main__":
    # The script assumes it is executed from the repository root.
    # If you run it from elsewhere, adjust the path resolution in `load_config`.
    main()
