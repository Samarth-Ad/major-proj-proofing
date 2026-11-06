"""Classical sklearn baselines per dataset for the semantic 'runs-but-wrong' check.

For each dataset we record a trivial FLOOR (what a dumb predictor achieves) and a
simple TRAINED REFERENCE (for context). A run that exits 0 but reports a metric at
or below the floor is flagged as a silent/semantic failure in analyze.py:

  classification : floor = majority-class accuracy; a run must beat it.
  regression     : floor = mean-predictor (R^2 = 0); a run's R^2 must be > 0,
                   or its RMSE/RMSLE must be below the mean-predictor's.
  clustering     : floor = random labelling (ARI = 0, NMI ~ 0); a run must beat it.

All splits/seeds are fixed so the baselines are reproducible.

Run:  python -m experiment.baselines
"""
from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd

from experiment import config

SEED = 42
TEST_SIZE = 0.25


def _load(spec: config.DatasetSpec) -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_csv(config.dataset_csv(spec.id))
    return frame.drop(columns=[spec.target]), frame[spec.target]


def _preprocessor(features: pd.DataFrame):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in features.columns if c not in numeric]
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="most_frequent")),
         ("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )
    return ColumnTransformer(
        [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categorical)]
    )


def _classification(features: pd.DataFrame, target: pd.Series) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    n_classes = int(target.nunique())
    avg = "binary" if n_classes == 2 else "macro"
    x_tr, x_te, y_tr, y_te = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=SEED, stratify=target
    )
    majority = y_tr.value_counts().idxmax()
    dummy = [majority] * len(y_te)
    floor_acc = accuracy_score(y_te, dummy)
    floor_f1 = f1_score(y_te, dummy, average=avg, zero_division=0)

    clf = Pipeline(
        [("pre", _preprocessor(features)), ("model", LogisticRegression(max_iter=2000))]
    ).fit(x_tr, y_tr)
    pred = clf.predict(x_te)
    return {
        "metric": "accuracy/F1",
        "n_classes": n_classes,
        "majority_class_accuracy": round(float(floor_acc), 4),
        "majority_class_f1": round(float(floor_f1), 4),
        "reference_logreg_accuracy": round(float(accuracy_score(y_te, pred)), 4),
        "reference_logreg_f1": round(float(f1_score(y_te, pred, average=avg, zero_division=0)), 4),
        "semantic_rule": "fail if reported accuracy <= majority_class_accuracy (or metric missing/NaN)",
    }


def _regression(features: pd.DataFrame, target: pd.Series) -> dict:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline

    x_tr, x_te, y_tr, y_te = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=SEED
    )
    mean_pred = np.full(len(y_te), float(y_tr.mean()))
    mean_rmse = math.sqrt(mean_squared_error(y_te, mean_pred))
    can_rmsle = bool((y_te >= 0).all() and (mean_pred >= 0).all())
    mean_rmsle = (
        math.sqrt(float(np.mean((np.log1p(y_te) - np.log1p(mean_pred)) ** 2)))
        if can_rmsle else None
    )
    rf = Pipeline(
        [("pre", _preprocessor(features)), ("model", RandomForestRegressor(random_state=SEED))]
    ).fit(x_tr, y_tr)
    pred = rf.predict(x_te)
    return {
        "metric": "R2/RMSE/RMSLE",
        "floor_r2": 0.0,
        "mean_predictor_rmse": round(mean_rmse, 4),
        "mean_predictor_rmsle": round(mean_rmsle, 4) if mean_rmsle is not None else None,
        "reference_rf_r2": round(float(r2_score(y_te, pred)), 4),
        "reference_rf_rmse": round(math.sqrt(mean_squared_error(y_te, pred)), 4),
        "semantic_rule": "fail if reported R2 < 0, or reported RMSE/RMSLE >= mean-predictor value (or metric missing/NaN)",
    }


def _clustering(features: pd.DataFrame, target: pd.Series) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.preprocessing import StandardScaler

    numeric = features.select_dtypes(include="number")
    scaled = StandardScaler().fit_transform(numeric)
    k = int(target.nunique())
    labels = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(scaled)
    return {
        "metric": "ARI/NMI/RI",
        "n_clusters": k,
        "floor_ari": 0.0,
        "floor_nmi": 0.02,
        "reference_kmeans_ari": round(float(adjusted_rand_score(target, labels)), 4),
        "reference_kmeans_nmi": round(float(normalized_mutual_info_score(target, labels)), 4),
        "semantic_rule": "fail if reported ARI <= 0 or NMI < 0.02 (or metric missing/NaN)",
    }


_BY_TASK = {
    "tabular_classification": _classification,
    "tabular_regression": _regression,
    "tabular_clustering": _clustering,
}


def main() -> None:
    baselines: dict[str, dict] = {}
    for spec in config.DATASETS:
        features, target = _load(spec)
        entry = {"task": spec.task, "target": spec.target, **_BY_TASK[spec.task](features, target)}
        baselines[spec.id] = entry
        print(f"[{spec.id:14s}] {spec.task:22s} {json.dumps({k: v for k, v in entry.items() if k not in ('task', 'target', 'semantic_rule')})}")

    with open(config.BASELINES_PATH, "w", encoding="utf-8") as handle:
        json.dump(baselines, handle, indent=2)
    print(f"\nWrote baselines -> {config.BASELINES_PATH}")


if __name__ == "__main__":
    main()
