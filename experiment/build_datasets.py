"""Build the 6 study datasets as local CSVs + a manifest.

All 6 are regenerated deterministically (fixed seed) so the evidence tree is
self-contained. Four come from sklearn offline; California-housing and Titanic
try their real source and fall back to a deterministic synthetic set (recorded
in the manifest) if the download is unavailable.

Run:  python -m experiment.build_datasets
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiment import config

SEED = 42

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


def _bunch_frame(loader) -> pd.DataFrame:
    bunch = loader()
    frame = pd.DataFrame(bunch.data, columns=list(bunch.feature_names))
    frame["target"] = bunch.target
    return frame


def build_breast_cancer() -> tuple[pd.DataFrame, str]:
    from sklearn.datasets import load_breast_cancer
    return _bunch_frame(load_breast_cancer), "sklearn"


def build_diabetes() -> tuple[pd.DataFrame, str]:
    from sklearn.datasets import load_diabetes
    return _bunch_frame(load_diabetes), "sklearn"


def build_wine() -> tuple[pd.DataFrame, str]:
    from sklearn.datasets import load_wine
    return _bunch_frame(load_wine), "sklearn"


def build_iris() -> tuple[pd.DataFrame, str]:
    from sklearn.datasets import load_iris
    return _bunch_frame(load_iris), "sklearn"


def build_california() -> tuple[pd.DataFrame, str]:
    try:
        from sklearn.datasets import fetch_california_housing

        bunch = fetch_california_housing(as_frame=True)
        frame = bunch.frame.rename(columns={bunch.target_names[0]: "target"})
        frame = frame.sample(n=1500, random_state=SEED).reset_index(drop=True)
        return frame, "sklearn-fetch"
    except Exception as exc:  # offline / download blocked -> deterministic synthetic
        print(f"[california] fetch failed ({exc!r}); using synthetic fallback.")
        rng = np.random.default_rng(SEED)
        n = 1500
        med_inc = rng.gamma(shape=2.0, scale=2.0, size=n)
        house_age = rng.uniform(1, 52, size=n)
        rooms = rng.normal(5.4, 1.2, size=n).clip(1)
        pop = rng.gamma(2.0, 700, size=n)
        latitude = rng.uniform(32.5, 42.0, size=n)
        longitude = rng.uniform(-124.3, -114.3, size=n)
        target = (
            0.42 * med_inc
            - 0.01 * house_age
            + 0.07 * rooms
            - 0.00002 * pop
            + rng.normal(0, 0.3, size=n)
        ).clip(0.15, 5.0)
        frame = pd.DataFrame(
            {
                "MedInc": med_inc,
                "HouseAge": house_age,
                "AveRooms": rooms,
                "Population": pop,
                "Latitude": latitude,
                "Longitude": longitude,
                "target": target,
            }
        )
        return frame, "synthetic-fallback"


def build_titanic() -> tuple[pd.DataFrame, str]:
    keep = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked", "survived"]
    try:
        from sklearn.datasets import fetch_openml

        # liac-arff parser: the default "pandas" parser trips over this dataset
        # under pandas 3.x ("expected string or bytes-like object, got 'int'").
        bunch = fetch_openml("titanic", version=1, as_frame=True, parser="liac-arff")
        frame = bunch.frame.copy()
        frame.columns = [c.lower() for c in frame.columns]
        frame = frame[[c for c in keep if c in frame.columns]].copy()
        # Coerce numeric columns (OpenML often returns them as object/category).
        for col in ("age", "fare", "sibsp", "parch", "pclass"):
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame["survived"] = pd.to_numeric(frame["survived"], errors="coerce").astype("Int64")
        frame = frame.dropna(subset=["survived"]).reset_index(drop=True)
        frame["survived"] = frame["survived"].astype(int)
        return frame, "openml"
    except Exception as exc:  # offline / download blocked -> deterministic synthetic
        print(f"[titanic] fetch failed ({exc!r}); using synthetic mixed-type fallback.")
        rng = np.random.default_rng(SEED)
        n = 800
        sex = rng.choice(["male", "female"], size=n, p=[0.64, 0.36])
        pclass = rng.choice([1, 2, 3], size=n, p=[0.24, 0.21, 0.55])
        embarked = rng.choice(["S", "C", "Q", None], size=n, p=[0.70, 0.19, 0.08, 0.03])
        age = rng.normal(29, 13, size=n).clip(0.5, 80)
        age[rng.random(n) < 0.20] = np.nan            # ~20% missing age
        sibsp = rng.poisson(0.5, size=n)
        parch = rng.poisson(0.38, size=n)
        fare = (rng.gamma(2.0, 15.0, size=n) / pclass).round(2)
        fare[rng.random(n) < 0.01] = np.nan           # a few missing fares
        logit = (
            1.1 * (sex == "female")
            - 0.9 * (pclass - 1)
            - 0.015 * np.nan_to_num(age, nan=29.0)
            + 0.002 * np.nan_to_num(fare, nan=15.0)
            + rng.normal(0, 0.6, size=n)
        )
        survived = (1 / (1 + np.exp(-logit)) > 0.5).astype(int)
        frame = pd.DataFrame(
            {
                "pclass": pclass,
                "sex": sex,
                "age": age,
                "sibsp": sibsp,
                "parch": parch,
                "fare": fare,
                "embarked": embarked,
                "survived": survived,
            }
        )
        return frame, "synthetic-fallback"


BUILDERS = {
    "breast_cancer": build_breast_cancer,
    "diabetes": build_diabetes,
    "wine": build_wine,
    "iris": build_iris,
    "california": build_california,
    "titanic": build_titanic,
}


def _column_summary(frame: pd.DataFrame, target: str) -> dict:
    features = frame.drop(columns=[target])
    numeric = features.select_dtypes(include="number").columns
    return {
        "n_rows": int(len(frame)),
        "n_columns": int(frame.shape[1]),
        "n_features": int(features.shape[1]),
        "n_numeric_features": int(len(numeric)),
        "n_categorical_features": int(features.shape[1] - len(numeric)),
        "has_missing": bool(frame.isna().any().any()),
        "columns": list(frame.columns),
    }


def main() -> None:
    config.DATASET_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict] = []

    for spec in config.DATASETS:
        frame, source = BUILDERS[spec.builder]()
        if spec.target not in frame.columns:
            raise ValueError(f"[{spec.id}] missing target column '{spec.target}'")
        out_path = config.dataset_csv(spec.id)
        frame.to_csv(out_path, index=False)
        entry = {
            "id": spec.id,
            "task": spec.task,
            "target": spec.target,
            "path": str(out_path.relative_to(config.PROJECT_ROOT)).replace("\\", "/"),
            "source_realized": source,
            "seed": SEED,
            **_column_summary(frame, spec.target),
        }
        manifest.append(entry)
        print(f"[{spec.id:14s}] {entry['n_rows']:>5d} rows x {entry['n_columns']:>2d} cols "
              f"| missing={entry['has_missing']} | source={source}")

    with open(config.MANIFEST_PATH, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(f"\nWrote {len(manifest)} datasets + manifest -> {config.MANIFEST_PATH}")


if __name__ == "__main__":
    main()
