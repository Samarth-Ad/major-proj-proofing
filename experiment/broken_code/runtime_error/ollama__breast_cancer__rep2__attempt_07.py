
"""
pipeline.py

End‑to‑end AutoML pipeline for the Breast‑Cancer tabular classification task.
All file paths are **relative** to the repository root, so the script can be
executed from any working directory with:

    python -m src.pipeline

The script performs the following steps:

1️⃣  Copy the user‑provided CSV (path given by the environment variable
    USER_CSV_PATH) to ``data/raw/breast_cancer.csv``.
2️⃣  Validate the raw file, write a cleaned copy to ``data/validated`` and a
    JSON validation report to ``data/report``.
3️⃣  Build a scikit‑learn preprocessing pipeline:
        MedianImputer → Winsorizer (1 %/99 %) → StandardScaler →
        InteractionGenerator (top‑5 features).
   The pipeline is saved as ``models/preprocess.pkl``.
4️⃣  Stratified split (70 % train / 20 % val / 10 % test) and store the
    transformed splits as Parquet files under ``data/processed``.
5️⃣  Train a LightGBM model with the exact hyper‑parameters given in the
    specification, export it to native LightGBM text and ONNX formats.
6️⃣  Evaluate the model on the held‑out test set, write predictions
    (including ``malignant_probability``) and a JSON metrics report.
7️⃣  Profile the model (size, FLOPs, latency, RAM, …) and write a JSON
    profile.
8️⃣  Assemble the final artefacts required for serving in
    ``agent_workspace/trained_models`` (model files, pipeline, profile,
    test metrics).

All prints use plain ASCII characters to avoid UnicodeEncodeError on
Windows consoles.
"""

import os
import json
import hashlib
import shutil
import time
import warnings
from pathlib import Path
from typing import Tuple, List

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
# Global constants (all paths are relative to the repository root)
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]          # repository root
RAW_DIR = ROOT / "data" / "raw"
VALIDATED_DIR = ROOT / "data" / "validated"
REPORT_DIR = ROOT / "data" / "report"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
AGENT_WORKSPACE = ROOT / "agent_workspace" / "trained_models"

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #
def ensure_dir(p: Path) -> None:
    """Create a directory (including parents) if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)


def sha256_of_file(p: Path) -> str:
    """Return the SHA‑256 hex digest of a file."""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# 1️⃣  Data ingestion
# --------------------------------------------------------------------------- #
def copy_user_csv() -> Path:
    """Copy the user CSV (path given by env var USER_CSV_PATH) to the repo."""
    src_path = os.getenv("USER_CSV_PATH")
    if not src_path:
        raise RuntimeError(
            "Environment variable USER_CSV_PATH is not set. "
            "It must point to the original breast_cancer.csv file."
        )
    src = Path(src_path)
    if not src.is_file():
        raise FileNotFoundError(f"Provided USER_CSV_PATH does not exist: {src}")

    dst = RAW_DIR / "breast_cancer.csv"
    ensure_dir(RAW_DIR)

    if dst.is_file():
        print("=== RAW CSV already present, skipping copy ===")
    else:
        shutil.copy2(src, dst)
        print(f"=== Copied raw CSV to {dst} ===")
    return dst


# --------------------------------------------------------------------------- #
# 2️⃣  Validation
# --------------------------------------------------------------------------- #
def validate_raw_csv(raw_path: Path) -> Tuple[Path, dict]:
    """Validate the raw CSV, write a cleaned copy and a JSON report."""
    df = pd.read_csv(raw_path)

    # ---- schema checks ----------------------------------------------------- #
    expected_n_cols = 31  # 30 features + target
    if df.shape[1] != expected_n_cols:
        raise ValueError(
            f"Expected {expected_n_cols} columns, found {df.shape[1]}."
        )
    if "target" not in df.columns:
        raise ValueError("Column 'target' is missing from the dataset.")
    if not set(df["target"].unique()).issubset({0, 1}):
        raise ValueError("Target column must contain only 0 and 1.")

    # ---- duplicate handling ------------------------------------------------ #
    n_dup = int(df.duplicated().sum())
    if n_dup > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"=== Removed {n_dup} duplicate rows ===")
    else:
        print("=== No duplicate rows found ===")

    # ---- missing values ----------------------------------------------------- #
    missing_perc = (df.isna().mean() * 100).round(2).to_dict()

    # ---- SHA‑256 of the original raw file ----------------------------------- #
    raw_sha = sha256_of_file(raw_path)

    # ---- write validated CSV ------------------------------------------------ #
    ensure_dir(VALIDATED_DIR)
    validated_path = VALIDATED_DIR / "breast_cancer.csv"
    df.to_csv(validated_path, index=False)
    print(f"=== Validated CSV written to {validated_path} ===")

    # ---- build validation report -------------------------------------------- #
    report = {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "duplicate_rows": n_dup,
        "missing_perc_per_column": {
            k: float(v) for k, v in missing_perc.items()
        },
        "target_unique_values": sorted(df["target"].unique().tolist()),
        "sha256_raw": raw_sha,
        "validation_passed": True,
    }

    ensure_dir(REPORT_DIR)
    report_path = REPORT_DIR / "validation_report.json"
    with report_path.open("w") as f:
        json.dump(report, f, indent=4)
    print(f"=== Validation report written to {report_path} ===")

    return validated_path, report


# --------------------------------------------------------------------------- #
# 3️⃣  Pre‑processing pipeline components
# --------------------------------------------------------------------------- #
class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip numeric columns at the given lower/upper percentiles."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.lower_bounds_ = np.percentile(
            X, self.lower * 100, axis=0
        )
        self.upper_bounds_ = np.percentile(
            X, self.upper * 100, axis=0
        )
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)


class InteractionGenerator(BaseEstimator, TransformerMixin):
    """
    Generate interaction features between the top‑k features (by absolute
    point‑biserial correlation with the target) and every other numeric
    feature.
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def fit(self, X, y):
        X_df = pd.DataFrame(X)
        corrs = {}
        for col in X_df.columns:
            corr, _ = pointbiserialr(X_df[col], y)
            corrs[col] = abs(corr) if not np.isnan(corr) else 0.0
        # select top‑k features
        self.top_features_ = sorted(
            corrs, key=corrs.get, reverse=True
        )[: self.top_k]
        self.feature_names_ = list(X_df.columns)

        # pre‑compute interaction column names for later use
        self.interaction_names_ = []
        for top in self.top_features_:
            for other in self.feature_names_:
                if other == top:
                    continue
                self.interaction_names_.append(f"{top}_x_{other}")
        return self

    def transform(self, X):
        X_df = pd.DataFrame(X, columns=self.feature_names_)
        interactions = {}
        for top in self.top_features_:
            for other in self.feature_names_:
                if other == top:
                    continue
                interactions[f"{top}_x_{other}"] = X_df[top] * X_df[other]
        inter_df = pd.DataFrame(interactions)
        return pd.concat([X_df, inter_df], axis=1).values

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_
        return list(input_features) + self.interaction_names_


def build_preprocess_pipeline() -> Pipeline:
    """Create the full preprocessing pipeline."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("winsorizer", Winsorizer(lower=0.01, upper=0.99)),
            ("scaler", StandardScaler()),
        ]
    )
    full_pipe = Pipeline(
        steps=[
            ("numeric", numeric_transformer),
            ("interact", InteractionGenerator(top_k=5)),
        ]
    )
    return full_pipe


# --------------------------------------------------------------------------- #
# 4️⃣  Stratified split & transformation
# --------------------------------------------------------------------------- #
def stratified_split(df: pd.DataFrame, random_state: int = 42):
    """Return train/val/test splits (features + target)."""
    X = df.drop(columns=["target"])
    y = df["target"]
    sss = StratifiedShuffleSplit(
        n_splits=1, test_size=0.30, random_state=random_state
    )
    train_idx, holdout_idx = next(sss.split(X, y))

    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_holdout, y_holdout = X.iloc[holdout_idx], y.iloc[holdout_idx]

    # holdout → val (2/3) + test (1/3)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=1 / 3, random_state=random_state
    )
    val_idx, test_idx = next(sss2.split(X_holdout, y_holdout))
    X_val, y_val = X_holdout.iloc[val_idx], y_holdout.iloc[val_idx]
    X_test, y_test = X_holdout.iloc[test_idx], y_holdout.iloc[test_idx]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def transform_and_save(
    pipeline: Pipeline,
    splits: Tuple[
        Tuple[pd.DataFrame, pd.Series],
        Tuple[pd.DataFrame, pd.Series],
        Tuple[pd.DataFrame, pd.Series],
    ],
):
    """Fit on train, transform all splits, and write them as Parquet."""
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = splits

    # Fit only on training data
    pipeline.fit(X_train.values, y_train.values)

    # Get final column names from the InteractionGenerator
    final_feature_names = pipeline.named_steps["interact"].get_feature_names_out(
        X_train.columns
    )

    def to_parquet(X_arr, y_series, filename):
        df = pd.DataFrame(X_arr, columns=final_feature_names)
        df["target"] = y_series.values
        df.to_parquet(PROCESSED_DIR / filename, index=False)

    ensure_dir(PROCESSED_DIR)

    X_train_t = pipeline.transform(X_train.values)
    X_val_t = pipeline.transform(X_val.values)
    X_test_t = pipeline.transform(X_test.values)

    to_parquet(X_train_t, y_train, "train.parquet")
    to_parquet(X_val_t, y_val, "val.parquet")
    to_parquet(X_test_t, y_test, "test.parquet")

    print("=== Processed splits saved as Parquet ===")
    return pipeline, (X_train_t, y_train.values), (X_val_t, y_val.values), (X_test_t, y_test.values)


# --------------------------------------------------------------------------- #
# 5️⃣  LightGBM training & export
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
    "seed": 42,
}


def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> lgb.Booster:
    """Train LightGBM with early stopping and return the booster."""
    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, free_raw_data=False)

    start = time.time()
    booster = lgb.train(
        LGB_PARAMS,
        dtrain,
        valid_sets=[dval],
        valid_names=["validation"],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    elapsed = time.time() - start
    print(f"=== LightGBM training finished in {elapsed:.2f}s ===")
    print(f"=== Best iteration: {booster.best_iteration} ===")
    return booster


def export_lightgbm(booster: lgb.Booster) -> None:
    """Export the trained model to native LightGBM text and ONNX formats."""
    ensure_dir(MODEL_DIR)
    txt_path = MODEL_DIR / "lightgbm_best.txt"
    onnx_path = MODEL_DIR / "lightgbm_best.onnx"

    booster.save_model(str(txt_path))
    print(f"=== LightGBM native model saved to {txt_path} ===")

    try:
        booster.save_model(str(onnx_path), format="onnx")
        print(f"=== LightGBM ONNX model saved to {onnx_path} ===")
    except Exception as exc:
        warnings.warn(
            f"ONNX export failed; the model will still be usable via the "
            f"native text file. Error: {exc}"
        )


# --------------------------------------------------------------------------- #
# 6️⃣  Test‑set evaluation
# --------------------------------------------------------------------------- #
def evaluate_on_test(
    booster: lgb.Booster,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Compute metrics, write predictions Parquet, and return a dict."""
    # LightGBM predicts probability of the positive class (label 1)
    prob_pos = booster.predict(
        X_test, num_iteration=booster.best_iteration
    )
    prob_malignant = 1.0 - prob_pos  # class 0 = malignant

    # ---- metrics ----------------------------------------------------------- #
    auc = roc_auc_score(y_test, prob_pos)
    brier = brier_score_loss(y_test, prob_pos)
    y_pred = (prob_pos >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()  # list of lists for JSON

    metrics = {
        "test_auc": float(auc),
        "test_brier": float(brier),
        "test_accuracy": float(acc),
        "test_precision": float(prec),
        "test_recall": float(rec),
        "test_f1": float(f1),
        "confusion_matrix": cm,
    }

    # ---- write predictions ------------------------------------------------- #
    ensure_dir(PROCESSED_DIR)
    pred_df = pd.DataFrame(
        X_test,
        columns=booster.feature_name(),
    )
    pred_df["target"] = y_test
    pred_df["malignant_probability"] = prob_malignant
    pred_path = PROCESSED_DIR / "test_predictions.parquet"
    pred_df.to_parquet(pred_path, index=False)
    print(f"=== Test predictions written to {pred_path} ===")

    # ---- write metrics JSON ------------------------------------------------ #
    ensure_dir(REPORTS_DIR)
    metrics_path = REPORTS_DIR / "test_metrics.json"
    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=4)
    print(f"=== Test metrics saved to {metrics_path} ===")
    return metrics


# --------------------------------------------------------------------------- #
# 7️⃣  Model profiling
# --------------------------------------------------------------------------- #
def profile_model(
    booster: lgb.Booster,
    pipeline: Pipeline,
    data_hash: str,
) -> dict:
    """Collect size, FLOPs, latency, RAM, etc., and write JSON."""
    # ---- model size -------------------------------------------------------- #
    model_path = MODEL_DIR / "lightgbm_best.txt"
    model_size_bytes = model_path.stat().st_size

    # ---- tree statistics --------------------------------------------------- #
    num_trees = booster.num_trees()
    dump = booster.dump_model()
    depths = [tree["max_depth"] for tree in dump["tree_info"]]
    avg_depth = float(np.mean(depths))

    # ---- FLOPs estimate (very rough) -------------------------------------- #
    flops_est = int(num_trees * avg_depth * 2)

    # ---- inference latency (using validation set) -------------------------- #
    val_path = PROCESSED_DIR / "val.parquet"
    val_df = pd.read_parquet(val_path)
    X_val = val_df.drop(columns=["target"]).values
    n_samples = min(10_000, X_val.shape[0])

    # Optional psutil import – if unavailable we skip RAM measurement
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
    except Exception:
        process = None
        mem_before = None

    start = time.time()
    _ = booster.predict(
        X_val[:n_samples], num_iteration=booster.best_iteration
    )
    elapsed = time.time() - start

    if process is not None:
        mem_after = process.memory_info().rss
        peak_ram_mb = round((mem_after - mem_before) / (1024 ** 2), 3)
    else:
        peak_ram_mb = None

    latency_ms = (elapsed / n_samples) * 1000

    # ---- feature count ----------------------------------------------------- #
    final_feature_count = len(
        pipeline.named_steps["interact"].get_feature_names_out(
            X_val.shape[1] and list(pipeline.named_steps["numeric"]
                                   .named_steps["scaler"]
                                   .get_feature_names_out())
        )
    )

    profile = {
        "model_size_bytes": int(model_size_bytes),
        "model_size_mb": round(model_size_bytes / (1024 ** 2), 3),
        "num_trees": int(num_trees),
        "avg_tree_depth": round(avg_depth, 3),
        "estimated_flops_per_prediction": int(flops_est),
        "mean_inference_latency_ms": round(latency_ms, 3),
        "peak_ram_mb_during_inference": peak_ram_mb,
        "hyperparameters": LGB_PARAMS,
        "seed": 42,
        "data_sha256": data_hash,
        "pipeline_feature_count": final_feature_count,
        "interaction_features_added": len(
            pipeline.named_steps["interact"].interaction_names_
        ),
    }

    ensure_dir(REPORTS_DIR)
    profile_path = REPORTS_DIR / "model_profile.json"
    with profile_path.open("w") as f:
        json.dump(profile, f, indent=4)
    print(f"=== Model profile saved to {profile_path} ===")
    return profile


# --------------------------------------------------------------------------- #
# 8️⃣  Assemble deployment artefacts
# --------------------------------------------------------------------------- #
def assemble_deployment_package() -> None:
    """Copy required artefacts into agent_workspace/trained_models."""
    ensure_dir(AGENT_WORKSPACE)

    src_to_dst = {
        MODEL_DIR / "lightgbm_best.txt": AGENT_WORKSPACE / "lightgbm_best.txt",
        MODEL_DIR / "lightgbm_best.onnx": AGENT_WORKSPACE / "lightgbm_best.onnx",
        MODEL_DIR / "preprocess.pkl": AGENT_WORKSPACE / "preprocess.pkl",
        REPORTS_DIR / "model_profile.json": AGENT_WORKSPACE / "model_profile.json",
        REPORTS_DIR / "test_metrics.json": AGENT_WORKSPACE / "test_metrics.json",
    }

    for src, dst in src_to_dst.items():
        if not src.is_file():
            raise FileNotFoundError(f"Expected artefact not found: {src}")
        shutil.copy2(src, dst)
        print(f"=== Copied {src.name} to deployment package ===")


# --------------------------------------------------------------------------- #
# Main orchestration
# --------------------------------------------------------------------------- #
def main():
    # 1️⃣  Copy raw CSV
    raw_csv_path = copy_user_csv()

    # 2️⃣  Validate and obtain a clean copy
    validated_path, validation_report = validate_raw_csv(raw_csv_path)

    # 3️⃣  Load validated data
    df = pd.read_csv(validated_path)

    # 4️⃣  Build and persist preprocessing pipeline
    pipeline = build_preprocess_pipeline()
    ensure_dir(MODEL_DIR)
    pipeline_path = MODEL_DIR / "preprocess.pkl"
    joblib.dump(pipeline, pipeline_path)
    print(f"=== Preprocessing pipeline saved to {pipeline_path} ===")

    # 5️⃣  Stratified split
    splits = stratified_split(df, random_state=42)

    # 6️⃣  Fit pipeline on train, transform all splits, write Parquet
    pipeline, train_t, val_t, test_t = transform_and_save(pipeline, splits)

    # 7️⃣  Train LightGBM
    booster = train_lightgbm(train_t[0], train_t[1], val_t[0], val_t[1])

    # 8️⃣  Export model (native + ONNX)
    export_lightgbm(booster)

    # 9️⃣  Evaluate on test set
    test_metrics = evaluate_on_test(booster, test_t[0], test_t[1])

    # 🔟  Profile the model
    model_profile = profile_model(
        booster,
        pipeline,
        data_hash=validation_report["sha256_raw"],
    )

    # 1️⃣1️⃣ Assemble deployment artefacts
    assemble_deployment_package()

    # ------------------------------------------------------------------- #
    print("\n=== PIPELINE FINISHED SUCCESSFULLY ===")
    print(f"Test ROC-AUC : {test_metrics['test_auc']:.4f}")
    print(f"Model size   : {model_profile['model_size_mb']:.2f} MB")
    # ------------------------------------------------------------------- #


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
