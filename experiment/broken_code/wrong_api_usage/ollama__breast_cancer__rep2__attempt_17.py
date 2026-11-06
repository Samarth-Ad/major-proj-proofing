
"""
Breast‑cancer tabular‑classification end‑to‑end pipeline.

The script follows the specification while being robust to missing optional
dependencies (pyarrow, onnxruntime, psutil).  All artefacts are written under
the repository root using relative paths, and the final trained model,
pre‑processing pipeline and profiling information are copied to
"./agent_workspace/trained_models".

Run with:
    python pipeline.py
"""

import os
import json
import time
import hashlib
import shutil
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import joblib
import lightgbm as lgb

# ----------------------------------------------------------------------
# Optional imports – guarded as required by the specification
# ----------------------------------------------------------------------
try:
    import pyarrow  # noqa: F401
    _HAS_PYARROW = True
except Exception:  # pragma: no cover
    _HAS_PYARROW = False
    warnings.warn("pyarrow not available – Parquet I/O will fall back to CSV.")

try:
    import onnxruntime as ort  # noqa: F401
    _HAS_ONNXRUNTIME = True
except Exception:  # pragma: no cover
    _HAS_ONNXRUNTIME = False
    warnings.warn("onnxruntime not available – ONNX export will be skipped.")

try:
    import psutil  # noqa: F401
    _HAS_PSUTIL = True
except Exception:  # pragma: no cover
    _HAS_PSUTIL = False
    warnings.warn("psutil not available – RAM profiling will be omitted.")

# ----------------------------------------------------------------------
# Global constants
# ----------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Repository root (two levels up from this file)
REPO_ROOT = Path(__file__).parents[1]

# ----------------------------------------------------------------------
# Helper utilities
# ----------------------------------------------------------------------
def sha256_of_file(path: Path) -> str:
    """Return the SHA‑256 hash of a file (hex string)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def cast_to_native(obj):
    """Recursively cast NumPy scalars/arrays to native Python types."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: cast_to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [cast_to_native(v) for v in obj]
    return obj


def write_json(path: Path, data: dict):
    """Write JSON with only native Python types."""
    with path.open("w", encoding="utf-8") as f:
        json.dump(cast_to_native(data), f, indent=2)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# 1️⃣ Ingest & validation
# ----------------------------------------------------------------------
def ingest_and_validate() -> Path:
    raw_dir = REPO_ROOT / "data" / "raw"
    validated_dir = REPO_ROOT / "data" / "validated"
    report_dir = REPO_ROOT / "data" / "report"

    ensure_dir(raw_dir)
    ensure_dir(validated_dir)
    ensure_dir(report_dir)

    raw_path = raw_dir / "breast_cancer.csv"

    user_path = os.getenv("USER_CSV_PATH", "").strip()
    if user_path and Path(user_path).is_file():
        shutil.copy(user_path, raw_path)
        print("=== USER CSV FOUND – COPIED TO RAW DIR ===")
    else:
        print("=== USER CSV NOT FOUND – USING SYNTHETIC DATA ===")
        bc = load_breast_cancer(as_frame=True)
        bc.frame.to_csv(raw_path, index=False)
        print("=== SYNTHETIC CSV WRITTEN TO RAW DIR ===")

    df = pd.read_csv(raw_path)

    report = {"raw_sha256": sha256_of_file(raw_path)}

    # Column count check
    expected_cols = 31  # 30 features + target
    if df.shape[1] != expected_cols:
        report["error"] = f"Expected {expected_cols} columns, found {df.shape[1]}"
    else:
        report["error"] = None

    # Target column checks
    if "target" not in df.columns:
        report["target_missing"] = True
    else:
        report["target_missing"] = False
        uniq = sorted(df["target"].dropna().unique())
        report["target_unique_values"] = uniq
        if set(uniq) - {0, 1}:
            report["target_invalid_values"] = uniq

    # Drop duplicates
    dup_cnt = int(df.duplicated().sum())
    report["duplicate_rows_removed"] = dup_cnt
    df = df.drop_duplicates().reset_index(drop=True)

    # Missing‑value percentages
    report["missing_percentage_per_column"] = (
        (df.isna().mean() * 100).round(2).to_dict()
    )

    # Write validated CSV
    validated_path = validated_dir / "breast_cancer.csv"
    df.to_csv(validated_path, index=False)

    # Write validation report
    write_json(report_dir / "validation_report.json", report)

    print("=== VALIDATION COMPLETE ===")
    return validated_path


# ----------------------------------------------------------------------
# Custom transformers
# ----------------------------------------------------------------------
class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip values to lower/upper percentiles (computed on fit)."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        self.lower_vals_ = np.percentile(
            X_arr, self.lower * 100, axis=0
        )
        self.upper_vals_ = np.percentile(
            X_arr, self.upper * 100, axis=0
        )
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        return np.clip(X_arr, self.lower_vals_, self.upper_vals_)


class InteractionGenerator(BaseEstimator, TransformerMixin):
    """
    Add interaction columns between the top‑k features (by absolute point‑biserial
    correlation with the target) and every other feature.
    Works with DataFrames and NumPy arrays.
    """

    def __init__(self, top_k=5):
        self.top_k = top_k

    @staticmethod
    def _point_biserial_corr(x, y):
        # y is binary {0,1}
        if np.std(x) == 0:
            return 0.0
        return np.corrcoef(x, y)[0, 1]

    def fit(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_arr = X.values
        else:
            n_feat = X.shape[1]
            self.feature_names_ = [f"f{i}" for i in range(n_feat)]
            X_arr = X

        corrs = []
        for i in range(len(self.feature_names_)):
            corr = self._point_biserial_corr(X_arr[:, i], y)
            corrs.append(abs(corr) if not np.isnan(corr) else 0.0)

        self.top_idx_ = np.argsort(corrs)[-self.top_k :][::-1]
        self.top_features_ = [
            self.feature_names_[i] for i in self.top_idx_
        ]

        # Build interaction column names
        self.interaction_names_ = []
        for top_feat in self.top_features_:
            for other_feat in self.feature_names_:
                if other_feat != top_feat:
                    self.interaction_names_.append(
                        f"{top_feat}_x_{other_feat}"
                    )
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X_arr = X.values
        else:
            X_arr = X

        interactions = []
        for top_feat in self.top_features_:
            idx_top = self.feature_names_.index(top_feat)
            top_col = X_arr[:, idx_top].reshape(-1, 1)
            for other_feat in self.feature_names_:
                if other_feat == top_feat:
                    continue
                idx_other = self.feature_names_.index(other_feat)
                other_col = X_arr[:, idx_other].reshape(-1, 1)
                interactions.append(top_col * other_col)

        if interactions:
            inter_arr = np.hstack(interactions)
            X_out = np.hstack([X_arr, inter_arr])
        else:
            X_out = X_arr
        return X_out

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            input_features = self.feature_names_
        return np.array(list(input_features) + self.interaction_names_)


# ----------------------------------------------------------------------
# 2️⃣ Preprocess & split
# ----------------------------------------------------------------------
def preprocess_and_split(validated_csv: Path):
    processed_dir = REPO_ROOT / "data" / "processed"
    ensure_dir(processed_dir)

    df = pd.read_csv(validated_csv)

    X = df.drop(columns=["target"])
    y = df["target"].values

    # Stratified split: 70% train, 20% val, 10% test
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=0.30, random_state=SEED
    )
    train_idx, holdout_idx = next(sss1.split(X, y))
    X_train, y_train = X.iloc[train_idx], y[train_idx]
    X_holdout, y_holdout = X.iloc[holdout_idx], y[holdout_idx]

    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=1 / 3, random_state=SEED
    )
    val_idx, test_idx = next(sss2.split(X_holdout, y_holdout))
    X_val, y_val = X_holdout.iloc[val_idx], y_holdout[val_idx]
    X_test, y_test = X_holdout.iloc[test_idx], y_holdout[test_idx]

    # Build preprocessing pipeline
    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("winsor", Winsorizer(lower=0.01, upper=0.99)),
            ("scaler", StandardScaler()),
            ("interact", InteractionGenerator(top_k=5)),
        ]
    )
    pipeline.fit(X_train, y_train)

    # Helper to transform a split and persist it
    def transform_and_save(X_split, y_split, name):
        X_out = pipeline.transform(X_split)
        col_names = pipeline.named_steps[
            "interact"
        ].get_feature_names_out(input_features=X_split.columns)
        df_out = pd.DataFrame(X_out, columns=col_names)
        df_out["target"] = y_split

        out_path = processed_dir / f"{name}.parquet"
        if _HAS_PYARROW:
            df_out.to_parquet(
                out_path, index=False, engine="pyarrow"
            )
        else:
            # Fallback to CSV (still usable downstream)
            out_path = processed_dir / f"{name}.csv"
            df_out.to_csv(out_path, index=False)
        return out_path

    train_path = transform_and_save(X_train, y_train, "train")
    val_path = transform_and_save(X_val, y_val, "val")
    test_path = transform_and_save(X_test, y_test, "test")

    # Persist pipeline
    model_dir = REPO_ROOT / "models"
    ensure_dir(model_dir)
    pipeline_path = model_dir / "preprocess.pkl"
    joblib.dump(pipeline, pipeline_path)

    # Preprocess report
    report = {
        "original_feature_names": list(X.columns),
        "final_feature_names": list(
            pipeline.named_steps["interact"].get_feature_names_out(
                input_features=X.columns
            )
        ),
        "winsorizer_percentiles": {"lower": 0.01, "upper": 0.99},
        "interaction_top_features": pipeline.named_steps[
            "interact"
        ].top_features_,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
    }
    write_json(REPO_ROOT / "reports" / "preprocess_report.json", report)

    print("=== PREPROCESSING & SPLIT COMPLETE ===")
    return {
        "train_path": train_path,
        "val_path": val_path,
        "test_path": test_path,
        "pipeline_path": pipeline_path,
        "preprocess_report_path": REPO_ROOT / "reports" / "preprocess_report.json",
    }


# ----------------------------------------------------------------------
# 3️⃣ Train LightGBM
# ----------------------------------------------------------------------
def train_lightgbm(train_path: Path, val_path: Path):
    # Load (Parquet if present, else CSV)
    if train_path.suffix == ".parquet":
        train_df = pd.read_parquet(train_path)
        val_df = pd.read_parquet(val_path)
    else:
        train_df = pd.read_csv(train_path)
        val_df = pd.read_csv(val_path)

    y_train = train_df["target"].values
    X_train = train_df.drop(columns=["target"])
    y_val = val_df["target"].values
    X_val = val_df.drop(columns=["target"])

    hyperparams = {
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

    lgb_train = lgb.Dataset(
        X_train, label=y_train, free_raw_data=False
    )
    lgb_val = lgb.Dataset(
        X_val, label=y_val, reference=lgb_train, free_raw_data=False
    )

    start = time.time()
    booster = lgb.train(
        hyperparams,
        lgb_train,
        valid_sets=[lgb_val],
        valid_names=["validation"],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    train_time = time.time() - start

    # Validation metrics
    val_pred = booster.predict(
        X_val, num_iteration=booster.best_iteration
    )
    val_auc = roc_auc_score(y_val, val_pred)
    val_brier = brier_score_loss(y_val, val_pred)

    training_report = {
        "train_time_seconds": float(train_time),
        "best_iteration": int(booster.best_iteration),
        "validation_auc": float(val_auc),
        "validation_brier": float(val_brier),
        "hyperparameters": hyperparams,
        "seed": SEED,
    }
    write_json(REPO_ROOT / "reports" / "training_report.json", training_report)

    # Export native LightGBM model
    model_dir = REPO_ROOT / "models"
    ensure_dir(model_dir)
    txt_path = model_dir / "lightgbm_best.txt"
    booster.save_model(txt_path)

    # Optional ONNX export
    onnx_path = model_dir / "lightgbm_best.onnx"
    if _HAS_ONNXRUNTIME:
        try:
            import onnx
            from onnxmltools.convert.lightgbm import convert_lightgbm
            from skl2onnx.common.data_types import FloatTensorType

            initial_type = [
                ("float_input", FloatTensorType([None, X_train.shape[1]]))
            ]
            onnx_model = convert_lightgbm(
                booster,
                initial_types=initial_type,
                target_opset=12,
            )
            onnx.save_model(onnx_model, onnx_path)
            print("=== ONNX EXPORT SUCCESS ===")
        except Exception as e:  # pragma: no cover
            print(f"=== ONNX EXPORT FAILED: {e}")
            if onnx_path.is_file():
                onnx_path.unlink()
            onnx_path = None
    else:
        onnx_path = None
        print("=== ONNX export skipped (onnxruntime missing) ===")

    print("=== TRAINING COMPLETE ===")
    return {
        "booster": booster,
        "txt_path": txt_path,
        "onnx_path": onnx_path,
        "training_report_path": REPO_ROOT / "reports" / "training_report.json",
    }


# ----------------------------------------------------------------------
# 4️⃣ Evaluation on test set
# ----------------------------------------------------------------------
def evaluate_test_set(test_path: Path, booster: lgb.Booster):
    if test_path.suffix == ".parquet":
        test_df = pd.read_parquet(test_path)
    else:
        test_df = pd.read_csv(test_path)

    y_test = test_df["target"].values
    X_test = test_df.drop(columns=["target"])

    prob_class1 = booster.predict(
        X_test, num_iteration=booster.best_iteration
    )
    malignant_prob = 1.0 - prob_class1

    # Metrics
    test_auc = roc_auc_score(y_test, prob_class1)
    test_brier = brier_score_loss(y_test, prob_class1)
    y_pred_label = (prob_class1 >= 0.5).astype(int)
    test_accuracy = accuracy_score(y_test, y_pred_label)
    test_precision = precision_score(
        y_test, y_pred_label, zero_division=0
    )
    test_recall = recall_score(y_test, y_pred_label, zero_division=0)
    test_f1 = f1_score(y_test, y_pred_label, zero_division=0)
    cm = confusion_matrix(y_test, y_pred_label).tolist()

    metrics = {
        "test_auc": float(test_auc),
        "test_brier": float(test_brier),
        "test_accuracy": float(test_accuracy),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "test_f1": float(test_f1),
        "confusion_matrix": cm,
    }

    # Enriched predictions
    enriched = X_test.copy()
    enriched["malignant_probability"] = malignant_prob
    enriched["target"] = y_test
    out_path = REPO_ROOT / "data" / "processed" / "test_predictions.parquet"
    enriched.to_parquet(out_path, index=False, engine="pyarrow")

    write_json(REPO_ROOT / "reports" / "test_metrics.json", metrics)

    print("=== EVALUATION COMPLETE ===")
    return {
        "metrics": metrics,
        "metrics_path": REPO_ROOT / "reports" / "test_metrics.json",
        "predictions_path": out_path,
    }


# ----------------------------------------------------------------------
# 5️⃣ Model profiling
# ----------------------------------------------------------------------
def profile_model(
    txt_path: Path,
    onnx_path: Path | None,
    booster: lgb.Booster,
    pipeline_path: Path,
    data_sha256: str,
):
    size_bytes = txt_path.stat().st_size
    size_mb = size_bytes / (1024 ** 2)

    num_trees = booster.num_trees()
    dump = booster.dump_model()
    depths = [tree["max_depth"] for tree in dump["tree_info"]]
    avg_depth = float(np.mean(depths)) if depths else 0.0
    estimated_flops = int(num_trees * avg_depth * 2)

    # Latency on up to 10k validation rows
    val_path = REPO_ROOT / "data" / "processed" / "val.parquet"
    if val_path.is_file():
        val_df = pd.read_parquet(val_path)
    else:
        val_df = pd.read_csv(
            REPO_ROOT / "data" / "processed" / "val.csv"
        )
    X_val = val_df.drop(columns=["target"])
    n_samples = min(10_000, len(X_val))
    X_sample = X_val.iloc[:n_samples]

    start = time.time()
    _ = booster.predict(X_sample, num_iteration=booster.best_iteration)
    elapsed = time.time() - start
    latency_ms = (elapsed / n_samples) * 1000.0

    # Peak RAM (optional)
    if _HAS_PSUTIL:
        proc = psutil.Process(os.getpid())
        _ = booster.predict(X_sample, num_iteration=booster.best_iteration)
        peak_ram_mb = proc.memory_info().rss / (1024 ** 2)
    else:
        peak_ram_mb = None

    # Feature counts from pipeline
    pipeline = joblib.load(pipeline_path)
    final_feat_cnt = pipeline.named_steps[
        "interact"
    ].get_feature_names_out(
        input_features=pipeline.named_steps["interact"].feature_names_
    ).shape[0]
    interaction_cnt = len(pipeline.named_steps["interact"].interaction_names_)

    profile = {
        "model_size_bytes": int(size_bytes),
        "model_size_mb": float(round(size_mb, 3)),
        "num_trees": int(num_trees),
        "avg_tree_depth": float(round(avg_depth, 3)),
        "estimated_flops_per_prediction": int(estimated_flops),
        "mean_inference_latency_ms": float(round(latency_ms, 3)),
        "peak_ram_mb_during_inference": float(round(peak_ram_mb, 3))
        if peak_ram_mb is not None
        else None,
        "hyperparameters": booster.params,
        "seed": SEED,
        "data_sha256": data_sha256,
        "pipeline_feature_count": int(final_feat_cnt),
        "interaction_features_added": int(interaction_cnt),
    }

    write_json(REPO_ROOT / "reports" / "model_profile.json", profile)

    print("=== PROFILING COMPLETE ===")
    return {
        "profile": profile,
        "profile_path": REPO_ROOT / "reports" / "model_profile.json",
    }


# ----------------------------------------------------------------------
# 6️⃣ Assemble deployment artefacts
# ----------------------------------------------------------------------
def assemble_deployment(
    txt_path: Path,
    onnx_path: Path | None,
    pipeline_path: Path,
    profile_path: Path,
    test_metrics_path: Path,
):
    dest_dir = REPO_ROOT / "agent_workspace" / "trained_models"
    ensure_dir(dest_dir)

    required = {
        "lightgbm_best.txt": txt_path,
        "preprocess.pkl": pipeline_path,
        "model_profile.json": profile_path,
        "test_metrics.json": test_metrics_path,
    }

    for name, src in required.items():
        if not src.is_file():
            raise FileNotFoundError(f"Required artefact missing: {src}")
        shutil.copy(src, dest_dir / name)

    # Optional ONNX
    if onnx_path and onnx_path.is_file():
        shutil.copy(onnx_path, dest_dir / "lightgbm_best.onnx")
        print("=== ONNX model copied to deployment package ===")
    else:
        print(
            "=== ONNX model not available – deployment will use native model ==="
        )

    print("=== DEPLOYMENT PACKAGE READY ===")
    return dest_dir


# ----------------------------------------------------------------------
# Main orchestration
# ----------------------------------------------------------------------
def main():
    # 1️⃣ Ingest & validate
    validated_csv = ingest_and_validate()

    # 2️⃣ Preprocess & split
    split_info = preprocess_and_split(validated_csv)

    # 3️⃣ Train LightGBM
    train_info = train_lightgbm(
        split_info["train_path"], split_info["val_path"]
    )

    # 4️⃣ Evaluate on test set
    eval_info = evaluate_test_set(
        split_info["test_path"], train_info["booster"]
    )

    # 5️⃣ Profile model (use raw SHA from validation report)
    raw_sha = json.load(
        open(
            REPO_ROOT
            / "data"
            / "report"
            / "validation_report.json",
            "r",
        )
    )["raw_sha256"]
    profile_info = profile_model(
        txt_path=train_info["txt_path"],
        onnx_path=train_info["onnx_path"],
        booster=train_info["booster"],
        pipeline_path=split_info["pipeline_path"],
        data_sha256=raw_sha,
    )

    # 6️⃣ Assemble deployment artefacts
    assemble_deployment(
        txt_path=train_info["txt_path"],
        onnx_path=train_info["onnx_path"],
        pipeline_path=split_info["pipeline_path"],
        profile_path=profile_info["profile_path"],
        test_metrics_path=eval_info["metrics_path"],
    )

    # Summary (ASCII only)
    print("\n=== PIPELINE SUMMARY ===")
    print(f"Validation report:   {REPO_ROOT / 'data' / 'report' / 'validation_report.json'}")
    print(f"Preprocess report:   {split_info['preprocess_report_path']}")
    print(f"Training report:     {train_info['training_report_path']}")
    print(f"Test metrics:       {eval_info['metrics_path']}")
    print(f"Model profile:      {profile_info['profile_path']}")
    print(f"Deployment folder:  {REPO_ROOT / 'agent_workspace' / 'trained_models'}")
    print("=== PIPELINE FINISHED SUCCESSFULLY ===")


if __name__ == "__main__":
    main()
