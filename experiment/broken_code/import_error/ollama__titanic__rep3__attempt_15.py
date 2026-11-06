
# --------------------------------------------------------------
# run_titanic_pipeline.py
# --------------------------------------------------------------
# End‑to‑end Titanic survival pipeline (data → preprocess → train → export → demo)
# --------------------------------------------------------------

import os
import json
import pathlib
import warnings
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import joblib
import yaml
import lightgbm as lgb
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import gradio as gr

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

# --------------------------------------------------------------
# 0️⃣  Configuration (single source of truth)
# --------------------------------------------------------------
CONFIG = {
    "paths": {
        # Absolute location of the raw CSV (change only if you store it elsewhere)
        "raw_csv": "C:/Users/Samarth_Ad/Desktop/College/SEM-7/AutoMLAgent/experiment/data/raw/titanic.csv",
        # Where artefacts will be persisted
        "pipeline": "models/pipeline_preprocess.pkl",
        "feature_order": "models/feature_order.json",
        "lightgbm_model": "models/lightgbm_best.pkl",
        "onnx_model": "onnx/lightgbm_best.onnx",
        "torch_script": "onnx/lightgbm_onnx.pt",
        # Destination required by the user
        "trained_models_dir": "./agent_workspace/trained_models",
    },
    "resources": {
        "seed": 42,
        "n_jobs": 4,
    },
    "lightgbm_params": {
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
        "random_state": 42,
        "n_jobs": 4,
        "early_stopping_rounds": 20,
    },
}
# --------------------------------------------------------------
# Helper utilities
# --------------------------------------------------------------
def set_random_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    np.random.RandomState(seed)
    import random

    random.seed(seed)
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_target_column(df: pd.DataFrame, target_name: str = "Survived") -> str:
    """Return the exact column name from the CSV (case‑insensitive)."""
    cols = {c.lower(): c for c in df.columns}
    if target_name.lower() not in cols:
        raise KeyError(f"Target column '{target_name}' not found in CSV.")
    return cols[target_name.lower()]


def extract_title(name: str) -> str:
    """Extract title from the passenger name (e.g., 'Mr.', 'Mrs.', 'Miss.', etc.)."""
    if pd.isna(name):
        return "Unknown"
    # Title is the word after the comma and before the dot
    parts = name.split(",")
    if len(parts) < 2:
        return "Unknown"
    title_part = parts[1].strip()
    title = title_part.split(".")[0].strip()
    return title


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create FamilySize, Title, AgeBin, LogFare and drop raw columns."""
    df = df.copy()

    # FamilySize = SibSp + Parch + 1 (the passenger themselves)
    if {"SibSp", "Parch"}.issubset(df.columns):
        df["FamilySize"] = df["SibSp"].astype(float) + df["Parch"].astype(float) + 1

    # Title extraction from Name
    if "Name" in df.columns:
        df["Title"] = df["Name"].apply(extract_title)

    # LogFare (guard against zero / missing)
    if "Fare" in df.columns:
        df["LogFare"] = np.log1p(df["Fare"].astype(float))

    # Age quantile bins (5 bins) – will be created later in the pipeline
    # Keep the raw Age column for binning; we do not drop it here because the
    # pipeline will handle the binning step conditionally.

    # Drop columns that we do not want to feed directly to the model
    drop_cols = ["Name", "Ticket", "Cabin"]
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True, errors="ignore")
    return df


def build_preprocess_pipeline(
    numeric_features: List[str], low_card_cat: List[str], high_card_cat: List[str]
) -> Pipeline:
    """Construct the Scikit‑Learn preprocessing pipeline."""
    # Numeric sub‑pipeline
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    # Low‑cardinality categorical – One‑Hot
    low_card_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # High‑cardinality categorical – Target Encoding (fallback implementation)
    class SimpleTargetEncoder:
        """Very small target‑encoder that works inside a ColumnTransformer."""

        def __init__(self):
            self.mapping_ = {}
            self.global_mean_ = None

        def fit(self, X, y):
            # X is a DataFrame / Series with a single column
            if isinstance(X, np.ndarray):
                X = pd.Series(X.ravel())
            self.global_mean_ = pd.to_numeric(y, errors="coerce").mean()
            for col in X.columns if isinstance(X, pd.DataFrame) else [X.name]:
                # Compute mean target per category
                df = pd.DataFrame({col: X[col], "target": y})
                mapping = (
                    df.groupby(col)["target"]
                    .mean()
                    .fillna(self.global_mean_)
                    .to_dict()
                )
                self.mapping_[col] = mapping
            return self

        def transform(self, X):
            X_out = X.copy()
            for col in X_out.columns:
                X_out[col] = X_out[col].map(self.mapping_.get(col, {})).fillna(self.global_mean_)
            return X_out.values.astype(np.float32)

    high_card_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("target_enc", SimpleTargetEncoder()),
        ]
    )

    # Assemble ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("lowcat", low_card_pipe, low_card_cat),
            ("highcat", high_card_pipe, high_card_cat),
        ],
        remainder="drop",  # everything else is dropped
    )

    # Full pipeline (adds the AgeBin step before the ColumnTransformer)
    class AgeBinner:
        """Creates 5 quantile bins for Age (if Age exists)."""

        def __init__(self, n_bins: int = 5):
            self.n_bins = n_bins
            self.bin_edges_ = None

        def fit(self, X, y=None):
            if "Age" in X.columns:
                self.bin_edges_ = np.quantile(
                    X["Age"].astype(float).replace({np.nan: None}), q=np.linspace(0, 1, self.n_bins + 1)
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

    pipeline = Pipeline(
        steps=[
            ("age_bin", AgeBinner()),
            ("preprocess", preprocessor),
        ]
    )
    return pipeline


def split_data(
    df: pd.DataFrame, target_col: str, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
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
    # From the remaining 30 % split into valid / test (2/1 ratio → 20 % / 10 %)
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=1 / 3,
        stratify=y_tmp,
        random_state=seed,
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


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


def evaluate(
    model: lgb.LGBMClassifier,
    X: np.ndarray,
    y_true: pd.Series,
) -> Dict[str, float]:
    """Return ROC‑AUC, Accuracy, F1, Brier score."""
    prob = model.predict_proba(X)[:, 1]
    y_pred = (prob >= 0.5).astype(int)

    metrics = {
        "roc_auc": roc_auc_score(y_true, prob),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "brier_score": brier_score_loss(y_true, prob),
    }
    return metrics


def export_to_onnx(
    model: lgb.LGBMClassifier,
    feature_order: List[str],
    onnx_path: str,
) -> None:
    """Export LightGBM model to ONNX (CPU version)."""
    # LightGBM's native exporter works with the Booster object
    booster = model.booster_
    booster.save_model(onnx_path, format="onnx")
    # Verify that the input name matches the first feature name
    onnx_model = onnx.load(onnx_path)
    input_name = onnx_model.graph.input[0].name
    if input_name != feature_order[0]:
        # Rename the first input tensor to match the feature order
        onnx_model.graph.input[0].name = feature_order[0]
        onnx.save(onnx_model, onnx_path)


class TorchScriptWrapper(nn.Module):
    """Tiny wrapper that runs an ONNX model via onnxruntime inside TorchScript."""

    def __init__(self, onnx_path: str):
        super().__init__()
        self.session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        # ONNX input name (assume single input)
        self.input_name = self.session.get_inputs()[0].name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to numpy float32
        np_input = x.detach().cpu().numpy().astype(np.float32)
        # ONNX expects shape (batch, n_features)
        if np_input.ndim == 1:
            np_input = np_input.reshape(1, -1)
        ort_outs = self.session.run(None, {self.input_name: np_input})
        # Output is a list with one array (probability of class 1)
        prob = ort_outs[0].squeeze()
        return torch.from_numpy(prob.astype(np.float32))


def save_artifacts(
    pipeline: Pipeline,
    model: lgb.LGBMClassifier,
    feature_order: List[str],
    cfg: Dict,
) -> None:
    """Persist pipeline, model, feature order and copy model to the required folder."""
    os.makedirs(os.path.dirname(cfg["paths"]["pipeline"]), exist_ok=True)
    os.makedirs(os.path.dirname(cfg["paths"]["lightgbm_model"]), exist_ok=True)
    os.makedirs(os.path.dirname(cfg["paths"]["onnx_model"]), exist_ok=True)
    os.makedirs(os.path.dirname(cfg["paths"]["torch_script"]), exist_ok=True)
    os.makedirs(cfg["paths"]["trained_models_dir"], exist_ok=True)

    # 1️⃣ Pipeline
    joblib.dump(pipeline, cfg["paths"]["pipeline"])
    print(f"✅ Preprocessing pipeline saved → {cfg['paths']['pipeline']}")

    # 2️⃣ Feature order (after ColumnTransformer)
    with open(cfg["paths"]["feature_order"], "w") as f:
        json.dump(feature_order, f)
    print(f"✅ Feature order saved → {cfg['paths']['feature_order']}")

    # 3️⃣ LightGBM model (joblib)
    joblib.dump(model, cfg["paths"]["lightgbm_model"])
    print(f"✅ LightGBM model saved → {cfg['paths']['lightgbm_model']}")

    # 4️⃣ Copy model to the user‑requested folder
    copy_dest = os.path.join(cfg["paths"]["trained_models_dir"], "lightgbm_best.pkl")
    joblib.dump(model, copy_dest)
    print(f"✅ Model also copied to → {copy_dest}")

    # 5️⃣ ONNX export
    export_to_onnx(model, feature_order, cfg["paths"]["onnx_model"])
    print(f"✅ ONNX model exported → {cfg['paths']['onnx_model']}")

    # 6️⃣ TorchScript wrapper
    wrapper = TorchScriptWrapper(cfg["paths"]["onnx_model"])
    scripted = torch.jit.script(wrapper)
    scripted.save(cfg["paths"]["torch_script"])
    print(f"✅ TorchScript wrapper saved → {cfg['paths']['torch_script']}")


# --------------------------------------------------------------
# 1️⃣  Main orchestration
# --------------------------------------------------------------
def main():
    set_random_seed(CONFIG["resources"]["seed"])

    # ------------------------------------------------------------------
    # Load raw CSV (as strings → later we cast where needed)
    # ------------------------------------------------------------------
    raw_path = pathlib.Path(CONFIG["paths"]["raw_csv"])
    if not raw_path.is_file():
        raise FileNotFoundError(f"Raw CSV not found at {raw_path}")
    df_raw = pd.read_csv(raw_path, dtype=str)

    # Resolve exact target column name (case‑insensitive)
    target_col = resolve_target_column(df_raw, "Survived")

    # ------------------------------------------------------------------
    # Basic cleaning: convert numeric columns that we know are numeric
    # ------------------------------------------------------------------
    numeric_cast = ["Age", "Fare", "SibSp", "Parch", "Pclass"]
    for col in numeric_cast:
        if col in df_raw.columns:
            df_raw[col] = pd.to_numeric(df_raw[col], errors="coerce")

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    df_fe = add_engineered_features(df_raw)

    # ------------------------------------------------------------------
    # Split data (stratified)
    # ------------------------------------------------------------------
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(
        df_fe, target_col, seed=CONFIG["resources"]["seed"]
    )

    # ------------------------------------------------------------------
    # Determine column groups for the pipeline
    # ------------------------------------------------------------------
    # After feature engineering we have:
    #   - numeric: Age, Fare, SibSp, Parch, Pclass, FamilySize, LogFare, AgeBin (if created)
    #   - categorical low‑cardinality (≤9 distinct): Sex, Embarked, Title
    #   - categorical high‑cardinality (>9): none for Titanic, but we keep the logic generic
    # ------------------------------------------------------------------
    all_features = X_train.columns.tolist()
    numeric_features = [
        c
        for c in all_features
        if X_train[c].dtype.kind in "iufc" and c != target_col
    ]
    # Ensure engineered numeric columns are present
    for extra in ["FamilySize", "LogFare", "AgeBin"]:
        if extra in X_train.columns and extra not in numeric_features:
            numeric_features.append(extra)

    # Categorical columns
    categorical_features = [c for c in all_features if c not in numeric_features]

    low_card_cat = [
        c for c in categorical_features if X_train[c].nunique() <= 9
    ]
    high_card_cat = [
        c for c in categorical_features if X_train[c].nunique() > 9
    ]

    # ------------------------------------------------------------------
    # Build preprocessing pipeline
    # ------------------------------------------------------------------
    pipeline = build_preprocess_pipeline(numeric_features, low_card_cat, high_card_cat)

    # Fit on train + valid (so we use as much data as possible for imputation/encoding)
    pipeline.fit(pd.concat([X_train, X_valid], axis=0), pd.concat([y_train, y_valid], axis=0))

    # Transform splits
    X_train_enc = pipeline.transform(X_train)
    X_valid_enc = pipeline.transform(X_valid)
    X_test_enc = pipeline.transform(X_test)

    # Retrieve the exact column order after the ColumnTransformer (needed for ONNX)
    # ColumnTransformer returns a numpy array; we can reconstruct order by
    # concatenating the names from each transformer.
    # Helper to get feature names from a transformer:
    def get_feature_names(ct: ColumnTransformer) -> List[str]:
        out = []
        for name, trans, cols in ct.transformers_:
            if name == "remainder":
                continue
            if hasattr(trans, "named_steps"):
                # Pipeline inside ColumnTransformer
                last_step = list(trans.named_steps.values())[-1]
                if hasattr(last_step, "get_feature_names_out"):
                    names = last_step.get_feature_names_out(cols)
                else:
                    names = cols
            else:
                # Simple transformer (e.g., SimpleImputer)
                names = cols
            out.extend(list(names))
        return out

    feature_order = get_feature_names(pipeline.named_steps["preprocess"])

    # ------------------------------------------------------------------
    # Train LightGBM
    # ------------------------------------------------------------------
    model = train_lightgbm(
        X_train_enc,
        y_train,
        X_valid_enc,
        y_valid,
        CONFIG["lightgbm_params"],
    )

    # ------------------------------------------------------------------
    # Evaluate on test set
    # ------------------------------------------------------------------
    test_metrics = evaluate(model, X_test_enc, y_test)
    print("\n🔎 Test set metrics")
    for k, v in test_metrics.items():
        print(f"  {k:12}: {v:.4f}")

    # ------------------------------------------------------------------
    # Persist artefacts (pipeline, model, ONNX, TorchScript, feature order)
    # ------------------------------------------------------------------
    save_artifacts(pipeline, model, feature_order, CONFIG)

    # ------------------------------------------------------------------
    # OPTIONAL: launch a Gradio demo (you can comment this out if you only need the artefacts)
    # ------------------------------------------------------------------
    def predict_survival(**kwargs):
        """Accept raw Titanic columns, run pipeline + TorchScript, return prob & class."""
        # Build a one‑row DataFrame preserving original dtypes (strings → later cast)
        input_df = pd.DataFrame([kwargs])
        # Cast numeric columns (same list as before)
        for col in numeric_cast:
            if col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors="coerce")
        # Feature engineering (must be identical to training)
        input_df = add_engineered_features(input_df)

        # Transform
        X_arr = pipeline.transform(input_df)  # numpy array, shape (1, n_features)

        # TorchScript inference
        torch_model = TorchScriptWrapper(CONFIG["paths"]["onnx_model"])
        torch_model.eval()
        with torch.no_grad():
            prob = torch_model(torch.from_numpy(X_arr.astype(np.float32))).item()
        pred = int(prob >= 0.5)
        return {"survival_probability": round(float(prob), 4), "prediction": pred}

    # Build Gradio interface (only the columns that exist after feature engineering)
    demo_inputs = []
    for col in X_train.columns:
        # Choose appropriate widget based on dtype / known domain
        if col in ["Sex", "Embarked", "Title"]:
            choices = sorted(df_raw[col].dropna().unique().tolist())
            demo_inputs.append(gr.inputs.Dropdown(choices=choices, label=col))
        elif col in ["Pclass"]:
            demo_inputs.append(gr.inputs.Slider(minimum=1, maximum=3, step=1, label=col))
        elif col in ["Age", "Fare", "FamilySize", "LogFare", "SibSp", "Parch", "AgeBin"]:
            # Use numeric input; allow None for missing
            demo_inputs.append(gr.inputs.Number(label=col, optional=True))
        else:
            demo_inputs.append(gr.inputs.Textbox(label=col, optional=True))

    iface = gr.Interface(
        fn=predict_survival,
        inputs=demo_inputs,
        outputs=[
            gr.outputs.Label(num_top_classes=1, label="Survival Probability"),
            gr.outputs.Label(num_top_classes=1, label="Prediction (0=No, 1=Yes)"),
        ],
        title="Titanic Survival Prediction (LightGBM + TorchScript)",
        description="Enter the original Titanic passenger fields and get a survival probability.",
        allow_flagging="never",
        theme="default",
    )
    iface.launch(share=False)  # set share=True if you want a public link


if __name__ == "__main__":
    # Suppress harmless warnings (e.g., from LightGBM on Windows)
    warnings.filterwarnings("ignore")
    main()
