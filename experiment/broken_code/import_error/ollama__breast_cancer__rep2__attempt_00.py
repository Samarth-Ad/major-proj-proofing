
"""
Breast‑Cancer Wisconsin Tabular Classification Pipeline
=======================================================

This script implements the full end‑to‑end MLOps pipeline described in the
project blueprint:

1. **Data ingestion & validation**
2. **Pre‑processing pipeline** (median imputer → winsoriser → standard‑scaler →
   optional 5‑feature interaction block)
3. **Stratified train/val/test split** (70 % / 20 % / 10 %)
4. **LightGBM training** with the exact Optuna‑derived hyper‑parameters
5. **ONNX export**
6. **Test‑set evaluation** (ROC‑AUC, Brier, …)
7. **System profiling & metadata generation**
8. **MLflow tracking** (experiment: ``breast-cancer-tabular-auto-ml``)
9. **Saving the final artefacts** to ``./agent_workspace/trained_models``

The script is self‑contained – just run it from the project root after
installing the required packages (see the blueprint for exact versions).

Author: MLOps Engineer
Date: 2026‑10‑31
"""

import os
import json
import hashlib
import random
import time
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import joblib
import mlflow
import lightgbm as lgb
import onnxruntime as ort
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from scipy.stats import pointbiserialr

# --------------------------------------------------------------------------- #
# Global configuration & reproducibility
# --------------------------------------------------------------------------- #
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# MLflow experiment name
MLFLOW_EXPERIMENT = "breast-cancer-tabular-auto-ml"
mlflow.set_experiment(MLFLOW_EXPERIMENT)

# --------------------------------------------------------------------------- #
# Path configuration (all relative to project root)
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).parent.resolve()  # folder containing this script

# Input / validation
RAW_CSV = ROOT / "data" / "raw" / "breast_cancer_wisconsin.csv"
VALIDATED_CSV = ROOT / "data" / "validated" / "breast_cancer.csv"
VALIDATION_REPORT = ROOT / "data" / "report" / "data_validation_report.json"
RAW_HASH_FILE = ROOT / "data" / "raw" / "breast_cancer.sha256"

# Pre‑processing artefacts
PIPELINE_PATH = ROOT / "models" / "preprocess.pkl"
TRAIN_PARQUET = ROOT / "data" / "processed" / "train.parquet"
VAL_PARQUET = ROOT / "data" / "processed" / "val.parquet"
TEST_PARQUET = ROOT / "data" / "processed" / "test.parquet"
PREPROCESS_REPORT = ROOT / "reports" / "preprocess_report.json"

# Model artefacts
MODEL_TXT = ROOT / "models" / "lightgbm_best.txt"
MODEL_ONNX = ROOT / "models" / "lightgbm_best.onnx"
MODEL_PROFILE = ROOT / "reports" / "model_profile.json"

# Final deployment folder (as required by the user)
DEPLOY_DIR = ROOT / "agent_workspace" / "trained_models"
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def sha256_of_file(path: Path) -> str:
    """Return the SHA‑256 hash of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# 1️⃣ Data ingestion & validation
# --------------------------------------------------------------------------- #
def validate_and_copy(raw_path: Path, dest_path: Path, report_path: Path) -> str:
    """
    Validate the raw CSV and copy it to the validated location.
    Returns the SHA‑256 hash of the raw file.
    """
    df = pd.read_csv(raw_path)

    # ---- Validation checks ----
    report = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_perc_per_column": {},
        "target_unique_values": None,
        "validation_passed": False,
    }

    # 1. Column count (should be 31: 30 features + target)
    if df.shape[1] != 31:
        raise ValueError(f"Expected 31 columns, found {df.shape[1]}")

    # 2. Target column existence & binary values
    if "target" not in df.columns:
        raise ValueError("Column 'target' not found in dataset")
    target_vals = sorted(df["target"].dropna().unique())
    if target_vals != [0, 1]:
        raise ValueError(f"'target' must contain only 0/1, found {target_vals}")
    report["target_unique_values"] = target_vals

    # 3. Duplicates
    if df.duplicated().any():
        df = df.drop_duplicates()
        report["duplicate_rows"] = int(df.duplicated().sum())

    # 4. Missing values statistics
    missing_perc = (df.isna().mean() * 100).round(2).to_dict()
    report["missing_perc_per_column"] = missing_perc

    # All checks passed → copy
    ensure_dir(dest_path.parent)
    shutil.copy2(raw_path, dest_path)

    # Write JSON report
    ensure_dir(report_path.parent)
    with report_path.open("w") as f:
        json.dump(report, f, indent=4)

    # Write SHA‑256 hash file
    hash_val = sha256_of_file(raw_path)
    with (RAW_HASH_FILE).open("w") as f:
        f.write(hash_val)

    # Log to MLflow
    with mlflow.start_run(run_name="data-validation") as run:
        mlflow.log_artifact(str(raw_path), artifact_path="raw")
        mlflow.log_artifact(str(dest_path), artifact_path="validated")
        mlflow.log_artifact(str(report_path), artifact_path="reports")
        mlflow.log_text(hash_val, "raw/breast_cancer.sha256")
        mlflow.log_metric("n_rows", report["n_rows"])
        mlflow.log_metric("n_columns", report["n_columns"])
        mlflow.log_metric("duplicate_rows", report["duplicate_rows"])

    report["validation_passed"] = True
    return hash_val


# --------------------------------------------------------------------------- #
# 2️⃣ Custom transformers for the preprocessing pipeline
# --------------------------------------------------------------------------- #
class Winsorizer(BaseEstimator, TransformerMixin):
    """Cap values at lower/upper percentiles (computed on fit data)."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        self.lower_bounds_ = np.percentile(X, self.lower * 100, axis=0)
        self.upper_bounds_ = np.percentile(X, self.upper * 100, axis=0)
        return self

    def transform(self, X, y=None):
        X_clipped = np.clip(X, self.lower_bounds_, self.upper_bounds_)
        return X_clipped


class InteractionGenerator(BaseEstimator, TransformerMixin):
    """
    Generate pairwise product interactions between the top‑N features (by
    absolute point‑biserial correlation with the target) and all other numeric
    features.
    """

    def __init__(self, n_top=5, feature_names: List[str] = None):
        self.n_top = n_top
        self.feature_names = feature_names

    def fit(self, X, y):
        # Compute absolute point‑biserial correlation for each feature
        corrs = {}
        for i, col in enumerate(self.feature_names):
            corr, _ = pointbiserialr(X[:, i], y)
            corrs[col] = abs(corr)
        # Select top‑N
        self.top_features_ = sorted(corrs, key=corrs.get, reverse=True)[
            : self.n_top
        ]
        # Store column indices for later use
        self.top_idx_ = [self.feature_names.index(f) for f in self.top_features_]
        return self

    def transform(self, X, y=None):
        # X is a numpy array (already scaled / imputed)
        interactions = []
        col_names = []
        for top_i in self.top_idx_:
            top_col = X[:, top_i]
            for j in range(X.shape[1]):
                if j == top_i:
                    continue
                prod = top_col * X[:, j]
                interactions.append(prod.reshape(-1, 1))
                col_names.append(
                    f"{self.feature_names[top_i]}_x_{self.feature_names[j]}"
                )
        if interactions:
            inter_arr = np.hstack(interactions)
            X_aug = np.hstack([X, inter_arr])
            self.interaction_names_ = col_names
            return X_aug
        else:
            # No interactions added
            self.interaction_names_ = []
            return X

    def get_feature_names_out(self, input_features=None):
        base = list(self.feature_names)
        return base + self.interaction_names_


# --------------------------------------------------------------------------- #
# 3️⃣ Pre‑processing + train/val/test split
# --------------------------------------------------------------------------- #
def build_and_apply_preprocess(
    validated_csv: Path,
    pipeline_path: Path,
    train_path: Path,
    val_path: Path,
    test_path: Path,
    report_path: Path,
) -> Tuple[Pipeline, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads the validated CSV, creates a stratified split, fits the preprocessing
    pipeline on the training data, transforms all splits and writes them to
    Parquet.
    Returns the fitted pipeline and the three DataFrames (with target column).
    """
    df = pd.read_csv(validated_csv)

    # Separate features / target
    X = df.drop(columns=["target"]).values.astype(float)
    y = df["target"].values
    feature_names = list(df.drop(columns=["target"]).columns)

    # Stratified split 70/20/10
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=0.30, random_state=SEED
    )  # 30 % hold‑out (val+test)
    train_idx, holdout_idx = next(sss1.split(X, y))

    X_train, y_train = X[train_idx], y[train_idx]
    X_holdout, y_holdout = X[holdout_idx], y[holdout_idx]

    # Second split for val/test (20/10)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=1 / 3, random_state=SEED
    )  # 1/3 of 30 % → 10 % test, 20 % val
    val_idx, test_idx = next(sss2.split(X_holdout, y_holdout))

    X_val, y_val = X_holdout[val_idx], y_holdout[val_idx]
    X_test, y_test = X_holdout[test_idx], y_holdout[test_idx]

    # Build pipeline
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("winsor", Winsorizer(lower=0.01, upper=0.99)),
            ("scaler", StandardScaler()),
            (
                "interact",
                InteractionGenerator(
                    n_top=5, feature_names=feature_names
                ),
            ),
        ]
    )

    # Fit on training data only
    pipeline.fit(X_train, y_train)

    # Transform splits
    X_train_t = pipeline.transform(X_train)
    X_val_t = pipeline.transform(X_val)
    X_test_t = pipeline.transform(X_test)

    # Re‑attach target column for storage
    train_df = pd.DataFrame(
        X_train_t, columns=pipeline.get_feature_names_out()
    )
    train_df["target"] = y_train

    val_df = pd.DataFrame(
        X_val_t, columns=pipeline.get_feature_names_out()
    )
    val_df["target"] = y_val

    test_df = pd.DataFrame(
        X_test_t, columns=pipeline.get_feature_names_out()
    )
    test_df["target"] = y_test

    # Save artefacts
    ensure_dir(pipeline_path.parent)
    joblib.dump(pipeline, pipeline_path)

    for p, df_ in zip(
        [train_path, val_path, test_path], [train_df, val_df, test_df]
    ):
        ensure_dir(p.parent)
        df_.to_parquet(p, index=False)

    # Write preprocessing report
    report = {
        "feature_names_before": feature_names,
        "feature_names_after": pipeline.get_feature_names_out(),
        "winsorizer_percentiles": {"lower": 0.01, "upper": 0.99},
        "interaction_top_features": pipeline.named_steps[
            "interact"
        ].top_features_,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
    }
    ensure_dir(report_path.parent)
    with report_path.open("w") as f:
        json.dump(report, f, indent=4)

    # Log to MLflow
    with mlflow.start_run(run_name="preprocess") as run:
        mlflow.log_artifact(str(pipeline_path), artifact_path="models")
        mlflow.log_artifact(str(train_path), artifact_path="data/processed")
        mlflow.log_artifact(str(val_path), artifact_path="data/processed")
        mlflow.log_artifact(str(test_path), artifact_path="data/processed")
        mlflow.log_artifact(str(report_path), artifact_path="reports")
        mlflow.log_params(
            {
                "winsor_lower": 0.01,
                "winsor_upper": 0.99,
                "interaction_n_top": 5,
            }
        )

    return pipeline, train_df, val_df, test_df


# --------------------------------------------------------------------------- #
# 4️⃣ LightGBM training
# --------------------------------------------------------------------------- #
LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.07,
    "n_estimators": 400,
    "num_leaves": 31,
    "feature_fraction": 0.90,
    "bagging_fraction": 0.80,
    "bagging_freq": 5,
    "lambda_l2": 0.5,
    "min_child_samples": 15,
    "verbosity": -1,
    "seed": SEED,
}


def train_lightgbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    model_txt_path: Path,
    model_onnx_path: Path,
) -> Tuple[lgb.Booster, dict]:
    """
    Trains LightGBM on the pre‑processed data, logs metrics & artefacts,
    exports the model to native and ONNX formats.
    Returns the Booster object and a dict of validation metrics.
    """
    X_train = train_df.drop(columns=["target"]).values
    y_train = train_df["target"].values
    X_val = val_df.drop(columns=["target"]).values
    y_val = val_df["target"].values

    lgb_train = lgb.Dataset(
        X_train, label=y_train, free_raw_data=False
    )
    lgb_val = lgb.Dataset(
        X_val, label=y_val, reference=lgb_train, free_raw_data=False
    )

    start_time = time.time()
    booster = lgb.train(
        LGB_PARAMS,
        lgb_train,
        valid_sets=[lgb_val],
        valid_names=["validation"],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    training_time = time.time() - start_time

    # Validation predictions
    val_pred = booster.predict(X_val, num_iteration=booster.best_iteration)
    val_auc = roc_auc_score(y_val, val_pred)
    val_brier = brier_score_loss(y_val, val_pred)

    # Save native model
    ensure_dir(model_txt_path.parent)
    booster.save_model(str(model_txt_path))

    # Export to ONNX (LightGBM >= 4.0 supports direct export)
    booster.save_model(str(model_onnx_path), format="onnx")

    # Log to MLflow
    with mlflow.start_run(run_name="train-lightgbm") as run:
        mlflow.log_artifact(str(model_txt_path), artifact_path="models")
        mlflow.log_artifact(str(model_onnx_path), artifact_path="models")
        mlflow.log_metrics(
            {"val_auc": val_auc, "val_brier": val_brier, "training_time_s": training_time}
        )
        mlflow.log_params(LGB_PARAMS)

    metrics = {
        "val_auc": val_auc,
        "val_brier": val_brier,
        "training_time_s": training_time,
    }
    return booster, metrics


# --------------------------------------------------------------------------- #
# 5️⃣ Test‑set evaluation
# --------------------------------------------------------------------------- #
def evaluate_on_test(
    booster: lgb.Booster,
    test_df: pd.DataFrame,
    report_path: Path,
    predictions_path: Path,
) -> dict:
    X_test = test_df.drop(columns=["target"]).values
    y_test = test_df["target"].values

    # Predict probabilities of class 1 (malignant)
    prob = booster.predict(X_test, num_iteration=booster.best_iteration)

    # Metrics
    metrics = {
        "test_auc": roc_auc_score(y_test, prob),
        "test_brier": brier_score_loss(y_test, prob),
        "test_accuracy": accuracy_score(y_test, (prob >= 0.5).astype(int)),
        "test_precision": precision_score(y_test, (prob >= 0.5).astype(int)),
        "test_recall": recall_score(y_test, (prob >= 0.5).astype(int)),
        "test_f1": f1_score(y_test, (prob >= 0.5).astype(int)),
    }

    # Confusion matrix
    cm = confusion_matrix(y_test, (prob >= 0.5).astype(int)).tolist()
    metrics["confusion_matrix"] = cm

    # Save predictions
    pred_df = test_df.copy()
    pred_df["malignant_probability"] = prob
    ensure_dir(predictions_path.parent)
    pred_df.to_parquet(predictions_path, index=False)

    # Save JSON report
    ensure_dir(report_path.parent)
    with report_path.open("w") as f:
        json.dump(metrics, f, indent=4)

    # Log to MLflow
    with mlflow.start_run(run_name="test-evaluation") as run:
        mlflow.log_artifact(str(report_path), artifact_path="reports")
        mlflow.log_artifact(str(predictions_path), artifact_path="data")
        mlflow.log_metrics(metrics)

    return metrics


# --------------------------------------------------------------------------- #
# 6️⃣ Profiling & metadata generation
# --------------------------------------------------------------------------- #
def profile_model(
    booster: lgb.Booster,
    pipeline: Pipeline,
    validation_df: pd.DataFrame,
    hash_val: str,
    profile_path: Path,
):
    """
    Generates a JSON file with model size, FLOPs estimate, latency, RAM usage,
    hyper‑parameters, etc.
    """
    # Model size on disk
    model_size_bytes = MODEL_TXT.stat().st_size

    # Number of leaves (approximate FLOPs)
    num_trees = booster.num_trees()
    avg_depth = np.mean([tree["max_depth"] for tree in booster.dump_model()["tree_info"]])
    flops_est = num_trees * avg_depth * 2  # each node: compare + branch

    # Inference latency (mean over 10 000 rows of validation set)
    X_val = validation_df.drop(columns=["target"]).values
    n_samples = min(10_000, X_val.shape[0])
    X_sample = X_val[:n_samples]

    start = time.time()
    _ = booster.predict(X_sample, num_iteration=booster.best_iteration)
    latency_ms = (time.time() - start) / n_samples * 1000

    # Peak RAM during inference (simple approximation)
    process = None
    try:
        import psutil

        process = psutil.Process(os.getpid())
        _ = booster.predict(X_sample, num_iteration=booster.best_iteration)
        peak_ram_mb = process.memory_info().rss / (1024 ** 2)
    except Exception:
        peak_ram_mb = None

    profile = {
        "model_size_bytes": model_size_bytes,
        "model_size_mb": round(model_size_bytes / (1024 ** 2), 3),
        "num_trees": int(num_trees),
        "avg_tree_depth": float(avg_depth),
        "estimated_flops_per_prediction": int(flops_est),
        "mean_inference_latency_ms": round(latency_ms, 3),
        "peak_ram_mb_during_inference": round(peak_ram_mb, 3) if peak_ram_mb else None,
        "hyperparameters": LGB_PARAMS,
        "seed": SEED,
        "data_sha256": hash_val,
        "pipeline_feature_count": len(pipeline.get_feature_names_out()),
        "interaction_features_added": len(pipeline.named_steps["interact"].interaction_names_)
        if hasattr(pipeline.named_steps["interact"], "interaction_names_")
        else 0,
    }

    ensure_dir(profile_path.parent)
    with profile_path.open("w") as f:
        json.dump(profile, f, indent=4)

    # Log to MLflow (attach to the training run)
    with mlflow.start_run(run_name="model-profiling") as run:
        mlflow.log_artifact(str(profile_path), artifact_path="reports")
        mlflow.log_metrics(
            {
                "model_size_mb": profile["model_size_mb"],
                "mean_latency_ms": profile["mean_inference_latency_ms"],
                "val_auc": booster.best_score["validation"]["auc"],
            }
        )
        mlflow.log_params(LGB_PARAMS)


# --------------------------------------------------------------------------- #
# 7️⃣ Main orchestration
# --------------------------------------------------------------------------- #
def main():
    # ------------------------------------------------------------------- #
    # Step 1 – Validation
    # ------------------------------------------------------------------- #
    print("🔎 Validating raw CSV …")
    data_hash = validate_and_copy(
        RAW_CSV, VALIDATED_CSV, VALIDATION_REPORT
    )
    print(f"✅ Validation complete – SHA256: {data_hash}")

    # ------------------------------------------------------------------- #
    # Step 2 – Pre‑processing & split
    # ------------------------------------------------------------------- #
    print("⚙️ Building preprocessing pipeline and creating splits …")
    pipeline, train_df, val_df, test_df = build_and_apply_preprocess(
        VALIDATED_CSV,
        PIPELINE_PATH,
        TRAIN_PARQUET,
        VAL_PARQUET,
        TEST_PARQUET,
        PREPROCESS_REPORT,
    )
    print("✅ Pre‑processing done – artefacts stored.")

    # ------------------------------------------------------------------- #
    # Step 3 – LightGBM training
    # ------------------------------------------------------------------- #
    print("🚂 Training LightGBM model …")
    booster, val_metrics = train_lightgbm(
        train_df, val_df, MODEL_TXT, MODEL_ONNX
    )
    print(
        f"✅ Training finished – Val AUC: {val_metrics['val_auc']:.4f}, "
        f"Brier: {val_metrics['val_brier']:.4f}"
    )

    # ------------------------------------------------------------------- #
    # Step 4 – Test‑set evaluation
    # ------------------------------------------------------------------- #
    print("🔬 Evaluating on held‑out test set …")
    test_metrics = evaluate_on_test(
        booster,
        test_df,
        MODEL_PROFILE,  # reuse path for test metrics (will be overwritten later)
        ROOT / "data" / "processed" / "test_predictions.parquet",
    )
    print(f"✅ Test AUC: {test_metrics['test_auc']:.4f}")

    # ------------------------------------------------------------------- #
    # Step 5 – Profiling & metadata
    # ------------------------------------------------------------------- #
    print("📊 Profiling model …")
    profile_model(
        booster,
        pipeline,
        val_df,
        data_hash,
        MODEL_PROFILE,
    )
    print("✅ Profiling complete – report saved.")

    # ------------------------------------------------------------------- #
    # Step 6 – Copy final artefacts to deployment folder
    # ------------------------------------------------------------------- #
    print("📦 Copying final model artefacts to deployment directory …")
    shutil.copy2(MODEL_TXT, DEPLOY_DIR / MODEL_TXT.name)
    shutil.copy2(MODEL_ONNX, DEPLOY_DIR / MODEL_ONNX.name)
    shutil.copy2(PIPELINE_PATH, DEPLOY_DIR / PIPELINE_PATH.name)
    print(f"✅ Models available at: {DEPLOY_DIR}")

    print("\n🎉 Pipeline execution finished successfully!")


if __name__ == "__main__":
    main()
