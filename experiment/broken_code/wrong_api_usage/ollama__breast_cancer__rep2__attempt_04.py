
"""
Breast‑Cancer Wisconsin – end‑to‑end modeling pipeline
=======================================================

This script follows the blueprint you provided and implements the full
workflow:

1. Copy the user‑provided CSV into ``data/raw/`` (if it is not already there).
2. Validate the raw file (column count, target values, duplicates, SHA‑256).
3. Stratified split (70 % train / 20 % validation / 10 % test).
4. Build a scikit‑learn ``Pipeline``:
   - Median imputer
   - Winsoriser (1 % / 99 % caps)
   - StandardScaler
   - Interaction block (top‑5 features × all other numeric features)
5. Fit the pipeline on the training set, transform all splits and store them as
   Parquet files.
6. Train LightGBM with the exact hyper‑parameters (early‑stopping).
7. Export the model to native LightGBM text format **and** ONNX.
8. Evaluate on the held‑out test set (ROC‑AUC, Brier, …) and store predictions.
9. Profile the model (size, FLOPs, latency, RAM, …) and write a JSON report.
10. Copy the final artefacts to ``./agent_workspace/trained_models`` for
    downstream serving.

All paths are **relative to the script location**, making the code portable
across local runs, CI pipelines and Docker containers.
"""

# --------------------------------------------------------------------------- #
# Imports & version‑pinning (install with the supplied requirements.txt)
# --------------------------------------------------------------------------- #
import os
import json
import hashlib
import random
import time
import shutil
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from scipy.stats import pointbiserialr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# --------------------------------------------------------------------------- #
# Global configuration & reproducibility
# --------------------------------------------------------------------------- #
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# --------------------------------------------------------------------------- #
# Path configuration (all relative to the script location)
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).parent.resolve()

# ── Input / validation ────────────────────────────────────────────────────── #
RAW_CSV = ROOT / "data" / "raw" / "breast_cancer_wisconsin.csv"
VALIDATED_CSV = ROOT / "data" / "validated" / "breast_cancer.csv"
VALIDATION_REPORT = ROOT / "data" / "report" / "data_validation_report.json"
RAW_HASH_FILE = ROOT / "data" / "raw" / "breast_cancer.sha256"

# ── User‑provided source (hard‑coded in the description) ── #
USER_SOURCE_CSV = Path(
    r"C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\experiment\datasets\breast_cancer.csv"
)

# ── Pre‑processing artefacts ──────────────────────────────────────────────── #
PIPELINE_PATH = ROOT / "models" / "preprocess.pkl"
TRAIN_PARQUET = ROOT / "data" / "processed" / "train.parquet"
VAL_PARQUET = ROOT / "data" / "processed" / "val.parquet"
TEST_PARQUET = ROOT / "data" / "processed" / "test.parquet"
PREPROCESS_REPORT = ROOT / "reports" / "preprocess_report.json"

# ── Model artefacts ─────────────────────────────────────────────────────── #
MODEL_TXT = ROOT / "models" / "lightgbm_best.txt"
MODEL_ONNX = ROOT / "models" / "lightgbm_best.onnx"
TEST_PREDICTIONS = ROOT / "data" / "processed" / "test_predictions.parquet"
TEST_METRICS = ROOT / "reports" / "test_metrics.json"
MODEL_PROFILE = ROOT / "reports" / "model_profile.json"

# ── Final deployment folder ─────────────────────────────────────────────── #
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
    """Create a directory (including parents) if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)


def copy_user_raw_csv():
    """Copy the CSV from the user‑provided location into ``data/raw/``."""
    if not RAW_CSV.exists():
        if not USER_SOURCE_CSV.exists():
            raise FileNotFoundError(
                f"User‑provided source CSV not found at {USER_SOURCE_CSV}"
            )
        ensure_dir(RAW_CSV.parent)
        shutil.copy2(USER_SOURCE_CSV, RAW_CSV)
        print(f"Copied raw CSV from user source to {RAW_CSV}")
    else:
        print(f"Raw CSV already present at {RAW_CSV}")


# --------------------------------------------------------------------------- #
# 1️⃣ Data ingestion & validation
# --------------------------------------------------------------------------- #
def validate_and_copy(
    raw_path: Path, dest_path: Path, report_path: Path
) -> str:
    """Validate the raw CSV and copy a clean version to the validated folder."""
    df = pd.read_csv(raw_path)

    report = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_perc_per_column": (df.isna().mean() * 100).round(2).to_dict(),
        "target_unique_values": None,
        "validation_passed": False,
    }

    # ---- Checks ----------------------------------------------------------- #
    if df.shape[1] != 31:
        raise ValueError(f"Expected 31 columns, found {df.shape[1]}")
    if "target" not in df.columns:
        raise ValueError("Column 'target' not found")
    target_vals = sorted(df["target"].dropna().unique())
    if target_vals != [0, 1]:
        raise ValueError(f"'target' must contain only 0/1, found {target_vals}")
    report["target_unique_values"] = target_vals

    # Remove duplicates (keep first occurrence)
    if report["duplicate_rows"] > 0:
        df = df.drop_duplicates()
        report["duplicate_rows"] = int(df.duplicated().sum())

    # ---- Persist validated copy ------------------------------------------- #
    ensure_dir(dest_path.parent)
    df.to_csv(dest_path, index=False)

    # ---- Write validation report ------------------------------------------ #
    ensure_dir(report_path.parent)
    with report_path.open("w") as f:
        json.dump(report, f, indent=4)

    # ---- SHA‑256 hash ------------------------------------------------------ #
    hash_val = sha256_of_file(raw_path)
    ensure_dir(RAW_HASH_FILE.parent)
    RAW_HASH_FILE.write_text(hash_val)

    report["validation_passed"] = True
    # (Optionally rewrite the report with the final flag – not required)
    return hash_val


# --------------------------------------------------------------------------- #
# 2️⃣ Custom transformers for the preprocessing pipeline
# --------------------------------------------------------------------------- #
class Winsorizer(BaseEstimator, TransformerMixin):
    """Cap values at lower/upper percentiles (computed on fit data)."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        self.lower_bounds_ = np.percentile(X, self.lower * 100, axis=0)
        self.upper_bounds_ = np.percentile(X, self.upper * 100, axis=0)
        return self

    def transform(self, X, y=None):
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)


class InteractionGenerator(BaseEstimator, TransformerMixin):
    """
    Generate pairwise product interactions between the top‑N features
    (by absolute point‑biserial correlation with the target) and all other
    numeric features.
    """

    def __init__(self, n_top: int = 5, feature_names: List[str] = None):
        self.n_top = n_top
        self.feature_names = feature_names

    def fit(self, X, y):
        # Compute absolute point‑biserial correlation for each feature
        corrs = {}
        for i, name in enumerate(self.feature_names):
            corr, _ = pointbiserialr(X[:, i], y)
            corrs[name] = abs(corr)
        # Keep the top‑N features
        self.top_features_ = sorted(corrs, key=corrs.get, reverse=True)[
            : self.n_top
        ]
        self.top_idx_ = [self.feature_names.index(f) for f in self.top_features_]
        return self

    def transform(self, X, y=None):
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
            self.interaction_names_ = []
            return X

    def get_feature_names_out(self, input_features=None):
        base = list(self.feature_names)
        return base + getattr(self, "interaction_names_", [])


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
):
    """Fit the preprocessing pipeline on the training split and store artefacts."""
    df = pd.read_csv(validated_csv)

    X = df.drop(columns=["target"]).values.astype(float)
    y = df["target"].values
    feature_names = list(df.drop(columns=["target"]).columns)

    # ---- Stratified split 70/20/10 --------------------------------------- #
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=0.30, random_state=SEED
    )
    train_idx, holdout_idx = next(sss1.split(X, y))

    X_train, y_train = X[train_idx], y[train_idx]
    X_holdout, y_holdout = X[holdout_idx], y[holdout_idx]

    # Second split for validation / test (20% / 10%)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=1 / 3, random_state=SEED
    )
    val_idx, test_idx = next(sss2.split(X_holdout, y_holdout))

    X_val, y_val = X_holdout[val_idx], y_holdout[val_idx]
    X_test, y_test = X_holdout[test_idx], y_holdout[test_idx]

    # ---- Build pipeline ---------------------------------------------------- #
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("winsor", Winsorizer(lower=0.01, upper=0.99)),
            ("scaler", StandardScaler()),
            ("interact", InteractionGenerator(n_top=5, feature_names=feature_names)),
        ]
    )

    # Fit only on training data
    pipeline.fit(X_train, y_train)

    # Transform all splits
    X_train_t = pipeline.transform(X_train)
    X_val_t = pipeline.transform(X_val)
    X_test_t = pipeline.transform(X_test)

    # Store as DataFrames (including target)
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

    # ---- Persist artefacts ------------------------------------------------ #
    ensure_dir(pipeline_path.parent)
    joblib.dump(pipeline, pipeline_path)

    for p, df_ in zip([train_path, val_path, test_path],
                      [train_df, val_df, test_df]):
        ensure_dir(p.parent)
        df_.to_parquet(p, index=False)

    # ---- Preprocess report ------------------------------------------------ #
    report = {
        "original_feature_names": feature_names,
        "final_feature_names": pipeline.get_feature_names_out(),
        "winsorizer_percentiles": {"lower": 0.01, "upper": 0.99},
        "interaction_top_features": pipeline.named_steps["interact"].top_features_,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
    }
    ensure_dir(report_path.parent)
    with report_path.open("w") as f:
        json.dump(report, f, indent=4)

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
):
    """Train LightGBM, export native and ONNX models, and return the booster."""
    X_train = train_df.drop(columns=["target"]).values
    y_train = train_df["target"].values
    X_val = val_df.drop(columns=["target"]).values
    y_val = val_df["target"].values

    lgb_train = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    lgb_val = lgb.Dataset(
        X_val, label=y_val, reference=lgb_train, free_raw_data=False
    )

    start = time.time()
    booster = lgb.train(
        LGB_PARAMS,
        lgb_train,
        valid_sets=[lgb_val],
        valid_names=["validation"],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    training_time = time.time() - start

    # Validation metrics
    val_pred = booster.predict(X_val, num_iteration=booster.best_iteration)
    val_auc = roc_auc_score(y_val, val_pred)
    val_brier = brier_score_loss(y_val, val_pred)

    # ---- Save native model ------------------------------------------------ #
    ensure_dir(model_txt_path.parent)
    booster.save_model(str(model_txt_path))

    # ---- Export to ONNX --------------------------------------------------- #
    try:
        booster.save_model(str(model_onnx_path), format="onnx")
        onnx_exported = True
    except Exception as e:
        print(f"WARNING: ONNX export failed ({e}); continuing without ONNX.")
        onnx_exported = False

    # ---- Training report ------------------------------------------------- #
    training_report = {
        "val_auc": val_auc,
        "val_brier": val_brier,
        "training_time_s": training_time,
        "best_iteration": booster.best_iteration,
        "onnx_exported": onnx_exported,
    }
    ensure_dir(model_txt_path.parent)
    with (model_txt_path.parent / "training_report.json").open("w") as f:
        json.dump(training_report, f, indent=4)

    return booster, training_report


# --------------------------------------------------------------------------- #
# 5️⃣ Test‑set evaluation
# --------------------------------------------------------------------------- #
def evaluate_on_test(
    booster: lgb.Booster,
    test_df: pd.DataFrame,
    metrics_path: Path,
    predictions_path: Path,
):
    """Compute test metrics, store predictions and a JSON report."""
    X_test = test_df.drop(columns=["target"]).values
    y_test = test_df["target"].values

    prob = booster.predict(X_test, num_iteration=booster.best_iteration)
    preds_binary = (prob >= 0.5).astype(int)

    metrics = {
        "test_auc": roc_auc_score(y_test, prob),
        "test_brier": brier_score_loss(y_test, prob),
        "test_accuracy": accuracy_score(y_test, preds_binary),
        "test_precision": precision_score(
            y_test, preds_binary, zero_division=0
        ),
        "test_recall": recall_score(y_test, preds_binary, zero_division=0),
        "test_f1": f1_score(y_test, preds_binary, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, preds_binary).tolist(),
    }

    # Save predictions
    pred_df = test_df.copy()
    pred_df["malignant_probability"] = prob
    ensure_dir(predictions_path.parent)
    pred_df.to_parquet(predictions_path, index=False)

    # Save metrics
    ensure_dir(metrics_path.parent)
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


# --------------------------------------------------------------------------- #
# 6️⃣ Profiling & metadata generation
# --------------------------------------------------------------------------- #
def profile_model(
    booster: lgb.Booster,
    pipeline: Pipeline,
    validation_df: pd.DataFrame,
    data_hash: str,
    profile_path: Path,
):
    """Create a JSON file with size, FLOPs estimate, latency, RAM usage, etc."""
    # Model size on disk (bytes → MB)
    model_size_bytes = MODEL_TXT.stat().st_size
    model_size_mb = round(model_size_bytes / (1024 ** 2), 3)

    # Approximate FLOPs: num_trees * avg_depth * 2
    dump = booster.dump_model()
    num_trees = booster.num_trees()
    depths = [tree["max_depth"] for tree in dump["tree_info"]]
    avg_depth = float(np.mean(depths)) if depths else 0.0
    flops_est = int(num_trees * avg_depth * 2)

    # Inference latency (mean over up to 10 000 validation rows)
    X_val = validation_df.drop(columns=["target"]).values
    n_samples = min(10_000, X_val.shape[0])
    X_sample = X_val[:n_samples]

    start = time.time()
    _ = booster.predict(X_sample, num_iteration=booster.best_iteration)
    latency_ms = (time.time() - start) / n_samples * 1000

    # Peak RAM during inference (simple approximation)
    try:
        import psutil

        process = psutil.Process(os.getpid())
        _ = booster.predict(X_sample, num_iteration=booster.best_iteration)
        peak_ram_mb = round(process.memory_info().rss / (1024 ** 2), 3)
    except Exception:
        peak_ram_mb = None

    profile = {
        "model_size_bytes": model_size_bytes,
        "model_size_mb": model_size_mb,
        "num_trees": int(num_trees),
        "avg_tree_depth": float(avg_depth),
        "estimated_flops_per_prediction": flops_est,
        "mean_inference_latency_ms": round(latency_ms, 3),
        "peak_ram_mb_during_inference": peak_ram_mb,
        "hyperparameters": LGB_PARAMS,
        "seed": SEED,
        "data_sha256": data_hash,
        "pipeline_feature_count": len(pipeline.get_feature_names_out()),
        "interaction_features_added": len(
            getattr(
                pipeline.named_steps["interact"], "interaction_names_", []
            )
        ),
    }

    ensure_dir(profile_path.parent)
    with profile_path.open("w") as f:
        json.dump(profile, f, indent=4)

    return profile


# --------------------------------------------------------------------------- #
# 7️⃣ Main orchestration
# --------------------------------------------------------------------------- #
def main():
    # ------------------------------------------------------------------- #
    # Step 0 – Ensure raw CSV is present (copy from user location if needed)
    # ------------------------------------------------------------------- #
    copy_user_raw_csv()

    # ------------------------------------------------------------------- #
    # Step 1 – Validation
    # ------------------------------------------------------------------- #
    print("Validating raw CSV ...")
    data_hash = validate_and_copy(RAW_CSV, VALIDATED_CSV, VALIDATION_REPORT)
    print(f"Validation completed – SHA256: {data_hash}")

    # ------------------------------------------------------------------- #
    # Step 2 – Pre‑processing & split
    # ------------------------------------------------------------------- #
    print("Building preprocessing pipeline and creating splits ...")
    pipeline, train_df, val_df, test_df = build_and_apply_preprocess(
        VALIDATED_CSV,
        PIPELINE_PATH,
        TRAIN_PARQUET,
        VAL_PARQUET,
        TEST_PARQUET,
        PREPROCESS_REPORT,
    )
    print("Pre‑processing finished – artefacts stored.")

    # ------------------------------------------------------------------- #
    # Step 3 – LightGBM training
    # ------------------------------------------------------------------- #
    print("Training LightGBM model ...")
    booster, train_report = train_lightgbm(
        train_df, val_df, MODEL_TXT, MODEL_ONNX
    )
    print(
        f"Training done – Validation AUC: {train_report['val_auc']:.4f}, "
        f"Brier: {train_report['val_brier']:.4f}"
    )

    # ------------------------------------------------------------------- #
    # Step 4 – Test‑set evaluation
    # ------------------------------------------------------------------- #
    print("Evaluating on held‑out test set ...")
    test_metrics = evaluate_on_test(
        booster, test_df, TEST_METRICS, TEST_PREDICTIONS
    )
    print(f"Test AUC: {test_metrics['test_auc']:.4f}")

    # ------------------------------------------------------------------- #
    # Step 5 – Profiling & metadata
    # ------------------------------------------------------------------- #
    print("Profiling model ...")
    profile = profile_model(
        booster, pipeline, val_df, data_hash, MODEL_PROFILE
    )
    print(f"Profiling complete – model size: {profile['model_size_mb']} MiB")

    # ------------------------------------------------------------------- #
    # Step 6 – Copy final artefacts to deployment folder
    # ------------------------------------------------------------------- #
    print("Copying artefacts to ./agent_workspace/trained_models ...")
    artefacts = [
        (MODEL_TXT, MODEL_TXT.name),
        (MODEL_ONNX, MODEL_ONNX.name),
        (PIPELINE_PATH, PIPELINE_PATH.name),
        (MODEL_PROFILE, MODEL_PROFILE.name),
        (TEST_METRICS, TEST_METRICS.name),
    ]
    for src, dst_name in artefacts:
        if src.exists():
            shutil.copy2(src, DEPLOY_DIR / dst_name)

    print(f"All artefacts are now available in: {DEPLOY_DIR}")

    print("\nPipeline execution finished successfully!")


if __name__ == "__main__":
    main()
