
"""
src/pipeline.py

End‑to‑end AutoML pipeline for the breast‑cancer tabular classification task.
It implements the exact artefact layout described in the project brief.

Usage (from the repository root):
    python -m src.pipeline

If the environment variable ``USER_CSV_PATH`` is not set the script falls back
to the synthetic sklearn breast‑cancer dataset (so CI can run without the
real CSV).  All paths are built relative to the repository root, making the
code portable.
"""

# --------------------------------------------------------------------------- #
# Standard library
# --------------------------------------------------------------------------- #
import os
import sys
import json
import time
import hashlib
import shutil
from pathlib import Path
from typing import Dict, Any, Tuple, List

# --------------------------------------------------------------------------- #
# Third‑party imports – these must be installed via ``requirements.txt``.
# --------------------------------------------------------------------------- #
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
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
# Helper utilities
# --------------------------------------------------------------------------- #
def _sha256_file(path: Path) -> str:
    """Return the SHA‑256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_dir(p: Path) -> None:
    """Create a directory (including parents) if it does not exist."""
    p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# 1️⃣ Ingest & validate
# --------------------------------------------------------------------------- #
def ingest_and_validate() -> Tuple[Path, str]:
    """
    - Copies the raw CSV (or creates a synthetic fallback).
    - Validates column count, target column, target values, duplicates,
      missing‑value percentages.
    - Writes:
        data/raw/breast_cancer.csv
        data/validated/breast_cancer.csv
        data/report/validation_report.json
    Returns
    -------
    clean_path : Path
        Path to the validated CSV.
    raw_sha256 : str
        SHA‑256 hash of the *original* raw file (before cleaning).
    """
    root = Path(__file__).parents[1]          # repository root
    raw_dir = root / "data" / "raw"
    validated_dir = root / "data" / "validated"
    report_dir = root / "data" / "report"

    _ensure_dir(raw_dir)
    _ensure_dir(validated_dir)
    _ensure_dir(report_dir)

    raw_path = raw_dir / "breast_cancer.csv"

    # --------------------------------------------------- #
    # Copy user CSV or generate synthetic fallback
    # --------------------------------------------------- #
    user_csv = os.getenv("USER_CSV_PATH", "").strip()
    if user_csv:
        src = Path(user_csv)
        if not src.is_file():
            raise RuntimeError(
                f"USER_CSV_PATH is set to '{user_csv}' but the file does not exist."
            )
        shutil.copy(src, raw_path)
    else:
        # Synthetic fallback – sklearn's breast_cancer dataset
        from sklearn.datasets import load_breast_cancer

        data = load_breast_cancer(as_frame=True)
        df = data.frame.rename(columns={"target": "target"})
        df.to_csv(raw_path, index=False)

    # --------------------------------------------------- #
    # Validation
    # --------------------------------------------------- #
    df_raw = pd.read_csv(raw_path)

    validation: Dict[str, Any] = {
        "column_count_ok": int(df_raw.shape[1] == 31),
        "target_column_present": int("target" in df_raw.columns),
        "target_values_ok": int(set(df_raw["target"].unique()) <= {0, 1}),
        "duplicate_rows": int(df_raw.duplicated().sum()),
        "missing_percent_per_column": {
            col: float(round(df_raw[col].isna().mean() * 100, 4))
            for col in df_raw.columns
        },
        "sha256_raw": _sha256_file(raw_path),
    }

    # Drop duplicates (if any) – keep NaNs for later preprocessing
    df_clean = df_raw.drop_duplicates().reset_index(drop=True)

    # Write cleaned CSV
    clean_path = validated_dir / "breast_cancer.csv"
    df_clean.to_csv(clean_path, index=False)

    # Write validation JSON (native Python types only)
    report_path = report_dir / "validation_report.json"
    with report_path.open("w") as f:
        json.dump(validation, f, indent=2)

    return clean_path, validation["sha256_raw"]


# --------------------------------------------------------------------------- #
# 2️⃣ Preprocess pipeline & split
# --------------------------------------------------------------------------- #
class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip numeric columns to lower/upper percentiles."""

    def __init__(self, lower: float = 0.01, upper: float = 0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.lower_vals_ = X.quantile(self.lower)
        self.upper_vals_ = X.quantile(self.upper)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        X_clipped = X.clip(lower=self.lower_vals_, upper=self.upper_vals_, axis=1)
        return X_clipped.values

    def get_feature_names_out(self, input_features=None):
        return input_features


class InteractionGenerator(BaseEstimator, TransformerMixin):
    """
    Generate pairwise product interactions using the top‑k features most
    correlated with the binary target.
    """

    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def fit(self, X, y):
        X = pd.DataFrame(X)
        # Compute absolute point‑biserial correlation for each column
        corr = {}
        for col in X.columns:
            r, _ = pointbiserialr(X[col], y)
            corr[col] = abs(r) if not np.isnan(r) else 0.0
        # Store top‑k feature names
        self.top_features_ = sorted(corr, key=corr.get, reverse=True)[: self.top_k]
        self.original_features_ = list(X.columns)

        # Pre‑compute interaction column names
        self.interaction_names_ = []
        for top in self.top_features_:
            for other in self.original_features_:
                if other == top:
                    continue
                self.interaction_names_.append(f"{top}_x_{other}")
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        interactions = []
        for top in self.top_features_:
            for other in self.original_features_:
                if other == top:
                    continue
                interactions.append(X[top] * X[other])
        if interactions:
            inter_df = pd.concat(interactions, axis=1)
            inter_df.columns = self.interaction_names_
            out = pd.concat([X, inter_df], axis=1)
        else:
            out = X
        return out.values

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.original_features_
        return list(input_features) + self.interaction_names_


def build_and_apply_pipeline(validated_csv: Path) -> Dict[str, Any]:
    """
    - Stratified 70/20/10 split.
    - Build preprocessing pipeline.
    - Fit on training split only.
    - Transform all splits and write them as Parquet.
    - Persist the pipeline as ``models/preprocess.pkl``.
    Returns a dict with artefact paths and useful objects.
    """
    root = Path(__file__).parents[1]
    processed_dir = root / "data" / "processed"
    _ensure_dir(processed_dir)

    df = pd.read_csv(validated_csv)
    y = df["target"].values
    X = df.drop(columns=["target"]).values

    # --------------------------------------------------- #
    # Stratified split 70 % train / 20 % val / 10 % test
    # --------------------------------------------------- #
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, holdout_idx = next(sss1.split(X, y))

    X_train, y_train = X[train_idx], y[train_idx]
    X_holdout, y_holdout = X[holdout_idx], y[holdout_idx]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1 / 3, random_state=42)
    val_idx, test_idx = next(sss2.split(X_holdout, y_holdout))

    X_val, y_val = X_holdout[val_idx], y_holdout[val_idx]
    X_test, y_test = X_holdout[test_idx], y_holdout[test_idx]

    # --------------------------------------------------- #
    # Build pipeline (imputer → winsor → scaler → interactions)
    # --------------------------------------------------- #
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("winsor", Winsorizer(lower=0.01, upper=0.99)),
            ("scaler", StandardScaler()),
            ("interact", InteractionGenerator(top_k=5)),
        ]
    )

    # Fit only on training data
    pipeline.fit(X_train, y_train)

    # Transform splits
    X_train_t = pipeline.transform(X_train)
    X_val_t = pipeline.transform(X_val)
    X_test_t = pipeline.transform(X_test)

    # Helper to write a Parquet file (engine = pyarrow, which must be installed)
    def _save_parquet(arr: np.ndarray, target: np.ndarray, path: Path) -> None:
        df_out = pd.DataFrame(arr)
        df_out["target"] = target
        df_out.to_parquet(path, index=False, engine="pyarrow")

    _save_parquet(X_train_t, y_train, processed_dir / "train.parquet")
    _save_parquet(X_val_t, y_val, processed_dir / "val.parquet")
    _save_parquet(X_test_t, y_test, processed_dir / "test.parquet")

    # Persist pipeline
    models_dir = root / "models"
    _ensure_dir(models_dir)
    pipeline_path = models_dir / "preprocess.pkl"
    joblib.dump(pipeline, pipeline_path)

    # Preprocess report (native Python types)
    preprocess_report = {
        "original_feature_names": list(df.drop(columns=["target"]).columns),
        "final_feature_names": pipeline.named_steps["interact"].get_feature_names_out(),
        "winsorizer_percentiles": {"lower": 0.01, "upper": 0.99},
        "interaction_top_features": pipeline.named_steps["interact"].top_features_,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
    }
    report_path = root / "reports" / "preprocess_report.json"
    _ensure_dir(report_path.parent)
    with report_path.open("w") as f:
        json.dump(preprocess_report, f, indent=2)

    return {
        "pipeline_path": pipeline_path,
        "train_path": processed_dir / "train.parquet",
        "val_path": processed_dir / "val.parquet",
        "test_path": processed_dir / "test.parquet",
        "pipeline": pipeline,
        "data_sha256": _sha256_file(validated_csv),
    }


# --------------------------------------------------------------------------- #
# 3️⃣ Train LightGBM
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


def train_lightgbm(train_path: Path, val_path: Path, root: Path) -> Tuple[lgb.Booster, Dict[str, Any]]:
    """Train LightGBM, save native & ONNX models, write a training report."""
    df_train = pd.read_parquet(train_path)
    df_val = pd.read_parquet(val_path)

    X_train = df_train.drop(columns=["target"]).values
    y_train = df_train["target"].values
    X_val = df_val.drop(columns=["target"]).values
    y_val = df_val["target"].values

    train_data = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    val_data = lgb.Dataset(
        X_val, label=y_val, free_raw_data=False, reference=train_data
    )

    start = time.time()
    booster = lgb.train(
        LGB_PARAMS,
        train_data,
        valid_sets=[val_data],
        valid_names=["validation"],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    train_time = time.time() - start

    # Validation metrics
    val_pred = booster.predict(X_val, num_iteration=booster.best_iteration)
    val_auc = roc_auc_score(y_val, val_pred)
    val_brier = brier_score_loss(y_val, val_pred)

    # Save native LightGBM model (text format)
    models_dir = root / "models"
    _ensure_dir(models_dir)
    txt_path = models_dir / "lightgbm_best.txt"
    booster.save_model(str(txt_path), num_iteration=booster.best_iteration, format="text")

    # Export ONNX (optional – do not abort if it fails)
    onnx_path = models_dir / "lightgbm_best.onnx"
    try:
        booster.save_model(str(onnx_path), format="onnx")
    except Exception as exc:  # pragma: no cover
        print(f"⚠️ ONNX export failed: {exc}. Continuing without ONNX model.", file=sys.stderr)
        if onnx_path.is_file():
            onnx_path.unlink()

    # Training report
    training_report = {
        "train_time_seconds": float(round(train_time, 4)),
        "best_iteration": int(booster.best_iteration),
        "validation_auc": float(round(val_auc, 6)),
        "validation_brier": float(round(val_brier, 6)),
    }
    with (root / "reports" / "training_report.json").open("w") as f:
        json.dump(training_report, f, indent=2)

    return booster, training_report


# --------------------------------------------------------------------------- #
# 4️⃣ Test‑set evaluation
# --------------------------------------------------------------------------- #
def evaluate_test(
    booster: lgb.Booster, test_path: Path, root: Path
) -> Tuple[Dict[str, Any], Path]:
    """Predict on the test split, write predictions + metrics JSON."""
    df_test = pd.read_parquet(test_path)
    X_test = df_test.drop(columns=["target"]).values
    y_test = df_test["target"].values

    # LightGBM returns probability of class 1 (benign)
    prob_class1 = booster.predict(X_test, num_iteration=booster.best_iteration)
    malignant_prob = 1.0 - prob_class1

    # Metrics
    test_auc = roc_auc_score(y_test, prob_class1)
    test_brier = brier_score_loss(y_test, prob_class1)
    y_pred = (prob_class1 >= 0.5).astype(int)
    test_acc = accuracy_score(y_test, y_pred)
    test_prec = precision_score(y_test, y_pred, zero_division=0)
    test_rec = recall_score(y_test, y_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()  # list of lists for JSON

    # Save enriched test set
    df_out = df_test.copy()
    df_out["malignant_probability"] = malignant_prob
    out_path = root / "data" / "processed" / "test_predictions.parquet"
    df_out.to_parquet(out_path, index=False, engine="pyarrow")

    # Save metrics JSON
    metrics = {
        "test_auc": float(round(test_auc, 6)),
        "test_brier": float(round(test_brier, 6)),
        "test_accuracy": float(round(test_acc, 6)),
        "test_precision": float(round(test_prec, 6)),
        "test_recall": float(round(test_rec, 6)),
        "test_f1": float(round(test_f1, 6)),
        "confusion_matrix": cm,
    }
    with (root / "reports" / "test_metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    return metrics, out_path


# --------------------------------------------------------------------------- #
# 5️⃣ Model profiling
# --------------------------------------------------------------------------- #
def profile_model(
    booster: lgb.Booster,
    pipeline: Pipeline,
    val_path: Path,
    root: Path,
    data_sha256: str,
) -> Dict[str, Any]:
    """Collect size, FLOPs, latency, RAM, feature counts, etc., and write JSON."""
    # Model size
    model_path = root / "models" / "lightgbm_best.txt"
    size_bytes = model_path.stat().st_size
    size_mb = round(size_bytes / (1024 ** 2), 6)

    # Tree statistics
    dump = booster.dump_model()
    depths = [tree["max_depth"] for tree in dump["tree_info"]]
    avg_depth = float(round(np.mean(depths), 4))
    num_trees = booster.num_trees()

    # FLOPs estimate (very rough)
    flops = int(num_trees * avg_depth * 2)

    # Inference latency on validation split (mean per sample)
    df_val = pd.read_parquet(val_path)
    X_val = df_val.drop(columns=["target"]).values
    n_samples = min(10_000, len(X_val))
    X_sample = X_val[:n_samples]

    start = time.time()
    _ = booster.predict(X_sample, num_iteration=booster.best_iteration)
    elapsed = time.time() - start
    latency_ms = round((elapsed / n_samples) * 1000, 6)

    # Peak RAM during inference (optional)
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        _ = booster.predict(X_sample, num_iteration=booster.best_iteration)
        peak_ram_mb = round(proc.memory_info().rss / (1024 ** 2), 3)
    except Exception:  # pragma: no cover
        peak_ram_mb = None

    # Feature counts
    final_feature_names = pipeline.named_steps["interact"].get_feature_names_out()
    pipeline_feature_count = len(final_feature_names)
    interaction_features_added = len(pipeline.named_steps["interact"].interaction_names_)

    profile = {
        "model_size_bytes": int(size_bytes),
        "model_size_mb": float(size_mb),
        "num_trees": int(num_trees),
        "avg_tree_depth": float(avg_depth),
        "estimated_flops_per_prediction": int(flops),
        "mean_inference_latency_ms": float(latency_ms),
        "peak_ram_mb_during_inference": peak_ram_mb,
        "hyperparameters": LGB_PARAMS,
        "seed": 42,
        "data_sha256": data_sha256,
        "pipeline_feature_count": int(pipeline_feature_count),
        "interaction_features_added": int(interaction_features_added),
    }

    with (root / "reports" / "model_profile.json").open("w") as f:
        json.dump(profile, f, indent=2)

    return profile


# --------------------------------------------------------------------------- #
# 6️⃣ Assemble final artefacts for the serving container
# --------------------------------------------------------------------------- #
def assemble_deployment(root: Path) -> None:
    """
    Copies exactly the required artefacts into ``agent_workspace/trained_models``.
    Raises FileNotFoundError if any source artefact is missing.
    """
    src_to_dst = {
        "models/lightgbm_best.txt": "agent_workspace/trained_models/lightgbm_best.txt",
        "models/lightgbm_best.onnx": "agent_workspace/trained_models/lightgbm_best.onnx",
        "models/preprocess.pkl": "agent_workspace/trained_models/preprocess.pkl",
        "reports/model_profile.json": "agent_workspace/trained_models/model_profile.json",
        "reports/test_metrics.json": "agent_workspace/trained_models/test_metrics.json",
    }

    for src_rel, dst_rel in src_to_dst.items():
        src = root / src_rel
        dst = root / dst_rel
        if not src.is_file():
            raise FileNotFoundError(f"Required artefact missing: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)


# --------------------------------------------------------------------------- #
# 7️⃣ Main orchestration
# --------------------------------------------------------------------------- #
def main() -> Dict[str, Any]:
    """
    Executes the full pipeline:
        1. Ingest & validate
        2. Preprocess + split
        3. Train LightGBM
        4. Test‑set evaluation
        5. Model profiling
        6. Assemble deployment artefacts
    Returns a dictionary with SHA‑256, test metrics and model profile for quick
    inspection.
    """
    root = Path(__file__).parents[1]  # repository root

    # 1️⃣ Ingest & validate
    validated_csv, raw_sha256 = ingest_and_validate()

    # 2️⃣ Preprocess & split
    split_info = build_and_apply_pipeline(validated_csv)

    # 3️⃣ Train LightGBM
    booster, _ = train_lightgbm(
        split_info["train_path"], split_info["val_path"], root=root
    )

    # 4️⃣ Evaluate on test set
    test_metrics, _ = evaluate_test(booster, split_info["test_path"], root=root)

    # 5️⃣ Profile model
    profile = profile_model(
        booster,
        split_info["pipeline"],
        split_info["val_path"],
        root=root,
        data_sha256=raw_sha256,
    )

    # 6️⃣ Assemble deployment artefacts
    assemble_deployment(root)

    # Simple console summary (useful when run manually)
    print("\n=== Pipeline finished ===")
    print(f"Test ROC‑AUC : {test_metrics['test_auc']:.4f}")
    print(f"Model size   : {profile['model_size_mb']:.3f} MiB")
    print("Deployment artefacts are available under ./agent_workspace/trained_models/")

    return {
        "validation_sha256": raw_sha256,
        "test_metrics": test_metrics,
        "model_profile": profile,
    }


if __name__ == "__main__":
    main()
