
"""
Titanic Survival – end‑to‑end modeling pipeline
================================================

This script implements **all** steps required for the Titanic tabular‑classification
play‑book described in the “Titanic Survival” MLOps guide:

1. Load the single source of truth `config.yaml`.
2. Read the raw CSV (UTF‑8 safe) and resolve the target column case‑insensitively.
3. Cast numeric columns, engineer features, and perform a stratified 70/20/10 split.
4. Build a Scikit‑Learn preprocessing pipeline:
   * Age binning (5 quantile bins)
   * Numeric imputation → Robust scaling
   * Low‑cardinality categorical one‑hot encoding
   * High‑cardinality categorical target encoding (fallback implementation)
5. Fit the pipeline on *train + validation*, transform all splits and persist them as Parquet.
6. Train a LightGBM binary classifier with the exact hyper‑parameters from the play‑book.
7. Evaluate on the test split (ROC‑AUC, Accuracy, F1, Brier score) and print the results.
8. Persist artefacts:
   * `models/pipeline_preprocess.pkl`
   * `models/feature_order.json`
   * `models/lightgbm_best.pkl`
   * copy of the model to `./agent_workspace/trained_models/`
   * ONNX export (`onnx/lightgbm_best.onnx`)
   * TorchScript wrapper (`onnx/lightgbm_onnx.pt`)

The script is **self‑contained** – just place it at
`src/pipeline/run_titanic_pipeline.py` (or any location inside the repository) and
execute it from the repository root:

