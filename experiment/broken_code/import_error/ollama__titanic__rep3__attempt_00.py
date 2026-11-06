
"""
Titanic survival prediction – end‑to‑end MLOps pipeline

This script follows the exact folder layout and hyper‑parameters described in the
specification.  It:

1️⃣  Builds & saves a Scikit‑Learn preprocessing pipeline.
2️⃣  Transforms the train / valid / test splits and stores them as Parquet.
3️⃣  Trains a LightGBM classifier on the processed features.
4️⃣  Exports the trained model to ONNX.
5️⃣  Wraps the ONNX model in a tiny TorchScript‑compatible module.
6️⃣  (Optional) launches a Gradio demo that uses the pipeline + Torch wrapper.

All artefacts are written under the repository‑wide paths **and** a copy of the
final TorchScript model is placed in ``./agent_workspace/trained_models`` as
required by the execution environment.
"""

import os
import json
import yaml
import joblib
import pathlib
import numpy as np
import pandas as pd
import lightgbm as lgb
import onnx
import onnxruntime as ort
import torch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from category_encoders import TargetEncoder
from sklearn.metrics import roc_auc_score, brier_score_loss
from scipy import stats

# --------------------------------------------------------------------------- #
# 1️⃣  Utility helpers
# --------------------------------------------------------------------------- #
def load_config(cfg_path: str = "config.yaml") -> dict:
    """Load the central YAML configuration."""
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(path: pathlib.Path):
    """Create a directory if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# 2️⃣  Custom feature‑engineering transformer
# --------------------------------------------------------------------------- #
class CustomFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Implements the feature‑engineering steps required for the Titanic data:

    * FamilySize = SibSp + Parch + 1
    * Title extracted from Name
    * Age quantile bins (5 bins) → AgeBin
    * LogFare = log(Fare + 1)
    * Drops raw columns that are no longer needed
    """

    def __init__(self):
        # nothing to initialise
        pass

    @staticmethod
    def _extract_title(name: str) -> str:
        """Extract title using the regex ', <title>.'."""
        import re

        match = re.search(r",\s*([^\.]*)\.", name)
        return match.group(1).strip() if match else "Unknown"

    def fit(self, X: pd.DataFrame, y=None):
        # No fitting needed – all operations are deterministic
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # 1. FamilySize
        X["FamilySize"] = (
            X["SibSp"].astype(float) + X["Parch"].astype(float) + 1
        ).astype(int)

        # 2. Title
        X["Title"] = X["Name"].apply(self._extract_title)

        # 3. AgeBin (5 quantile bins)
        # We compute bins on the whole column (including NaNs) – pandas will
        # assign NaN to a separate bin which we later impute.
        age_series = pd.to_numeric(X["Age"], errors="coerce")
        X["AgeBin"] = pd.qcut(
            age_series, q=5, duplicates="drop"
        ).astype(str)

        # 4. LogFare
        fare_series = pd.to_numeric(X["Fare"], errors="coerce")
        X["LogFare"] = np.log1p(fare_series)

        # 5. Drop raw columns
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
# 3️⃣  Build the full preprocessing pipeline
# --------------------------------------------------------------------------- #
def build_preprocess_pipeline() -> Pipeline:
    """Creates the Scikit‑Learn pipeline described in the spec."""
    # The pipeline will be fitted later – we just declare the structure.
    # Placeholder – actual column lists are discovered after the custom step.
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # High‑cardinality handling (TargetEncoder) will be added dynamically.
    # For now we create a ColumnTransformer with empty lists; they will be
    # replaced after we run the custom transformer once on a sample.
    preproc = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, []),  # to be filled later
            ("cat", categorical_pipe, []),  # to be filled later
        ],
        remainder="drop",
    )

    pipeline = Pipeline(
        steps=[
            ("feat_eng", CustomFeatureEngineer()),
            ("preproc", preproc),
        ]
    )
    return pipeline


# --------------------------------------------------------------------------- #
# 4️⃣  Fit pipeline on train+valid and persist it
# --------------------------------------------------------------------------- #
def fit_and_save_pipeline(
    cfg: dict,
    raw_df: pd.DataFrame,
    train_idx: pd.Index,
    valid_idx: pd.Index,
):
    """Fits the preprocessing pipeline on train+valid and saves artefacts."""
    # Split raw data into train/valid (we only need the features here)
    train_raw = raw_df.loc[train_idx].reset_index(drop=True)
    valid_raw = raw_df.loc[valid_idx].reset_index(drop=True)

    # Build initial pipeline to run the custom transformer and infer column types
    pipeline = build_preprocess_pipeline()

    # Run the custom transformer on a combined sample to discover column names
    temp_feat = pipeline.named_steps["feat_eng"].transform(
        pd.concat([train_raw, valid_raw], axis=0)
    )
    # Determine numeric vs categorical after feature engineering
    numeric_cols = temp_feat.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = temp_feat.select_dtypes(include=["object", "category"]).columns.tolist()

    # Identify high‑cardinality categorical columns (>9 distinct values)
    high_cardinality = [
        col
        for col in categorical_cols
        if temp_feat[col].nunique() > 9
    ]
    low_cardinality = [c for c in categorical_cols if c not in high_cardinality]

    # Re‑build the ColumnTransformer with the correct column lists
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )
    # Low‑cardinality → OneHotEncoder
    cat_low_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    # High‑cardinality → TargetEncoder (fallback to identity if none)
    if high_cardinality:
        cat_high_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("target_enc", TargetEncoder()),
            ]
        )
        # Combine both categorical pipelines via ColumnTransformer
        cat_transformer = ColumnTransformer(
            transformers=[
                ("low", cat_low_pipe, low_cardinality),
                ("high", cat_high_pipe, high_cardinality),
            ]
        )
    else:
        cat_transformer = cat_low_pipe  # only low‑cardinality case

    # Final ColumnTransformer
    preproc = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", cat_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    # Assemble final pipeline
    pipeline = Pipeline(
        steps=[
            ("feat_eng", CustomFeatureEngineer()),
            ("preproc", preproc),
        ]
    )

    # Fit on train+valid
    combined = pd.concat([train_raw, valid_raw], axis=0).reset_index(drop=True)
    pipeline.fit(combined)

    # Persist pipeline
    model_dir = pathlib.Path(cfg["paths"]["pipeline"]).parent
    ensure_dir(model_dir)
    joblib.dump(pipeline, cfg["paths"]["pipeline"])

    # Store feature order (the order after transformation) for ONNX export
    # We obtain it by transforming a single row and inspecting column names.
    sample_transformed = pipeline.transform(combined.head(1))
    feature_order = sample_transformed.columns.tolist()
    feature_order_path = pathlib.Path("models/feature_order.json")
    ensure_dir(feature_order_path.parent)
    feature_order_path.write_text(json.dumps(feature_order, indent=2))

    return pipeline, feature_order


# --------------------------------------------------------------------------- #
# 5️⃣  Transform splits and write Parquet files (numeric only)
# --------------------------------------------------------------------------- #
def transform_and_save_splits(
    cfg: dict,
    pipeline: Pipeline,
    raw_df: pd.DataFrame,
    split_indices: dict,
):
    """Applies the fitted pipeline to train/valid/test and stores Parquet."""
    splits_dir = pathlib.Path(cfg["paths"]["splits_dir"])
    ensure_dir(splits_dir)

    for split_name, idx in split_indices.items():
        df_raw = raw_df.loc[idx].reset_index(drop=True)
        # Preserve the target column for later use
        y = df_raw["Survived"].astype(int).values
        X_transformed = pipeline.transform(df_raw)
        # X_transformed is a DataFrame with only numeric columns (after encoding)
        # Append the target as a column for convenience
        out_df = X_transformed.copy()
        out_df["Survived"] = y
        out_path = splits_dir / f"{split_name}.parquet"
        out_df.to_parquet(out_path, index=False)


# --------------------------------------------------------------------------- #
# 6️⃣  LightGBM training
# --------------------------------------------------------------------------- #
def train_lightgbm(cfg: dict, feature_order: list):
    """Trains LightGBM on the processed train/valid splits and logs artefacts."""
    splits_dir = pathlib.Path(cfg["paths"]["splits_dir"])

    # Load Parquet files
    train_df = pd.read_parquet(splits_dir / "train.parquet")
    valid_df = pd.read_parquet(splits_dir / "valid.parquet")

    X_train = train_df.drop(columns=["Survived"])
    y_train = train_df["Survived"]
    X_valid = valid_df.drop(columns=["Survived"])
    y_valid = valid_df["Survived"]

    # LightGBM hyper‑parameters (exact as specified)
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

    print(f"Validation ROC‑AUC: {auc:.4f}")
    print(f"Validation Brier score: {brier:.4f}")

    # Persist model
    model_path = pathlib.Path(cfg["paths"]["lightgbm_model"])
    ensure_dir(model_path.parent)
    joblib.dump(model, model_path)

    # Save feature order for ONNX export (already saved earlier)
    # Return model and metrics for downstream steps
    return model, {"roc_auc": auc, "brier": brier}


# --------------------------------------------------------------------------- #
# 7️⃣  Export LightGBM → ONNX
# --------------------------------------------------------------------------- #
def export_to_onnx(cfg: dict, model: lgb.LGBMClassifier, feature_order: list):
    """Converts the trained LightGBM model to ONNX and saves it."""
    # LightGBM's native ONNX exporter works via the Booster object
    booster = model.booster_
    onnx_path = pathlib.Path(cfg["paths"]["onnx_model"])
    ensure_dir(onnx_path.parent)

    # The input name must match the column order expected by the model.
    # We use the feature_order list (numeric column names after preprocessing).
    input_name = "input"
    # Export
    onnx_bytes = booster.save_model(
        onnx_path,
        format="onnx",
        # Provide the column names to the exporter
        # LightGBM expects a list of strings for the input feature names.
        # The order must be exactly the same as during training.
        # The exporter will embed these names in the graph.
        # The `input_name` argument is not directly exposed, so we rely on
        # the default naming; later we will reference the first input.
        # No extra arguments needed for a simple binary classifier.
    )
    print(f"ONNX model saved to {onnx_path}")
    return onnx_path


# --------------------------------------------------------------------------- #
# 8️⃣  Torch wrapper (scriptable)
# --------------------------------------------------------------------------- #
class LightGBMOnnxModel(torch.nn.Module):
    """
    Tiny wrapper that forwards a torch.Tensor to an ONNX Runtime session
    and returns the probability of class 1 as a torch.Tensor.
    """

    def __init__(self, onnx_path: str, input_name: str = "input"):
        super().__init__()
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        # Determine the actual input name from the graph (usually the first)
        self.input_name = self.session.get_inputs()[0].name or input_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ONNX Runtime expects numpy array of type float32
        np_input = x.detach().cpu().numpy().astype(np.float32)
        ort_outs = self.session.run(None, {self.input_name: np_input})
        # Output is a list; for binary classification LightGBM returns shape (N, 2)
        probs = ort_outs[0]  # shape (N, 2)
        # Return probability of class 1
        prob_pos = probs[:, 1]
        return torch.from_numpy(prob_pos).to(x.device)


def script_and_save_torch_model(cfg: dict, onnx_path: pathlib.Path):
    """Creates a TorchScript version of the ONNX wrapper and saves it."""
    wrapper = LightGBMOnnxModel(str(onnx_path))
    # Example dummy input for tracing (batch size 1)
    dummy = torch.randn(1, len(json.loads(open("models/feature_order.json").read()))).float()
    scripted = torch.jit.script(wrapper)
    torch_path = pathlib.Path(cfg["paths"]["torch_script"])
    ensure_dir(torch_path.parent)
    scripted.save(str(torch_path))
    print(f"TorchScript model saved to {torch_path}")

    # Also copy to the required workspace directory for the final deliverable
    ws_dir = pathlib.Path("./agent_workspace/trained_models")
    ensure_dir(ws_dir)
    ws_copy = ws_dir / torch_path.name
    ws_copy.write_bytes(torch_path.read_bytes())
    print(f"Copy of TorchScript model placed at {ws_copy}")


# --------------------------------------------------------------------------- #
# 9️⃣  Gradio demo (optional but convenient for quick testing)
# --------------------------------------------------------------------------- #
def launch_gradio_demo(cfg: dict):
    """Runs a minimal Gradio UI that accepts raw Titanic fields."""
    # Load artefacts once
    pipeline = joblib.load(cfg["paths"]["pipeline"])
    torch_script_path = pathlib.Path(cfg["paths"]["torch_script"])
    model = torch.jit.load(str(torch_script_path), map_location="cpu")
    model.eval()

    # Helper to turn a dict of raw inputs into a DataFrame
    def dict_to_df(record: dict) -> pd.DataFrame:
        return pd.DataFrame([record])

    def predict(**kwargs):
        raw_df = dict_to_df(kwargs)
        # Apply preprocessing – result is a DataFrame with numeric columns only
        processed = pipeline.transform(raw_df)
        # Convert to torch tensor
        tensor = torch.from_numpy(processed.values).float()
        with torch.no_grad():
            prob = model(tensor).numpy().item()
        label = int(prob > 0.5)
        return {"Survival probability": prob, "Predicted label": label}

    # Build the interface – we expose all original Titanic columns except the target
    # (the column list is taken from the raw CSV header)
    raw_csv_path = cfg["paths"]["raw_csv"]
    raw_columns = pd.read_csv(raw_csv_path, nrows=0).columns.tolist()
    raw_columns.remove("Survived")  # target not needed for inference

    iface = gr.Interface(
        fn=predict,
        inputs=[gr.Textbox(label=c) for c in raw_columns],
        outputs=[
            gr.Number(label="Survival probability"),
            gr.Number(label="Predicted label (0/1)"),
        ],
        title="Titanic Survival – PyTorch ONNX Wrapper",
        description="Enter raw passenger information; the model returns the survival probability.",
    )
    iface.launch(share=False)


# --------------------------------------------------------------------------- #
# 🔧  Main orchestration
# --------------------------------------------------------------------------- #
def main():
    cfg = load_config()  # expects config.yaml in the repo root

    # ------------------------------------------------------------------- #
    # 0️⃣ Load raw CSV (as strings to avoid premature NaNs)
    # ------------------------------------------------------------------- #
    raw_path = pathlib.Path(cfg["paths"]["raw_csv"])
    raw_df = pd.read_csv(raw_path, dtype=str)

    # Ensure the target column is integer for later steps
    raw_df["Survived"] = raw_df["Survived"].astype(int)

    # ------------------------------------------------------------------- #
    # 1️⃣ Create train / valid / test splits (70/20/10) – deterministic seed
    # ------------------------------------------------------------------- #
    rng = np.random.default_rng(cfg["resources"]["seed"])
    shuffled_idx = rng.permutation(raw_df.index)
    n = len(shuffled_idx)
    n_train = int(0.70 * n)
    n_valid = int(0.20 * n)
    train_idx = shuffled_idx[:n_train]
    valid_idx = shuffled_idx[n_train : n_train + n_valid]
    test_idx = shuffled_idx[n_train + n_valid :]

    split_indices = {
        "train": train_idx,
        "valid": valid_idx,
        "test": test_idx,
    }

    # ------------------------------------------------------------------- #
    # 2️⃣ Fit preprocessing pipeline on train+valid and persist it
    # ------------------------------------------------------------------- #
    pipeline, feature_order = fit_and_save_pipeline(
        cfg, raw_df, train_idx, valid_idx
    )

    # ------------------------------------------------------------------- #
    # 3️⃣ Transform splits and write Parquet files (numeric only)
    # ------------------------------------------------------------------- #
    transform_and_save_splits(cfg, pipeline, raw_df, split_indices)

    # ------------------------------------------------------------------- #
    # 4️⃣ Train LightGBM on the processed training data
    # ------------------------------------------------------------------- #
    model, metrics = train_lightgbm(cfg, feature_order)

    # ------------------------------------------------------------------- #
    # 5️⃣ Export the trained model to ONNX
    # ------------------------------------------------------------------- #
    onnx_path = export_to_onnx(cfg, model, feature_order)

    # ------------------------------------------------------------------- #
    # 6️⃣ Wrap ONNX model in a TorchScript module and save it
    # ------------------------------------------------------------------- #
    script_and_save_torch_model(cfg, onnx_path)

    # ------------------------------------------------------------------- #
    # 7️⃣ (Optional) launch a quick Gradio demo for manual testing
    # ------------------------------------------------------------------- #
    # Uncomment the line below if you want an interactive UI after the script runs.
    # launch_gradio_demo(cfg)

    # Final printout
    print("\n=== PIPELINE FINISHED ===")
    print(f"Validation metrics: {metrics}")
    print(f"Artifacts stored under the repository as defined in config.yaml.")
    print(
        "TorchScript model also copied to ./agent_workspace/trained_models for the final deliverable."
    )


if __name__ == "__main__":
    main()
