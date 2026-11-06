
"""
California Housing – End‑to‑End Modeling Pipeline
=================================================

This script implements the complete production‑ready workflow described in the
specification:

1️⃣  Load the raw CSV (immutable source) and compute a SHA‑256 checksum.  
2️⃣  Validate the schema and generate a data profile.  
3️⃣  Split the data into train / validation / test (70 % / 20 % / 10 %).  
4️⃣  Build a **versioned Scikit‑Learn pipeline** that performs  

   * Median imputation  
   * Isolation‑Forest outlier clipping (1 % contamination)  
   * Robust scaling  
   * Interaction‑only polynomial features (degree 2) for the three most
     correlated features w.r.t. the target  

   The pipeline is fitted on the training split only and persisted with
   ``joblib``.  The transformed splits are stored as Parquet files.

5️⃣  Train a **CatBoostRegressor** with the frozen hyper‑parameters, export the
   native model (``.cbm``) and an ONNX version (``.onnx``).

6️⃣  Train a small **feed‑forward PyTorch MLP** on the same pre‑processed data,
   export it as a TorchScript module (``mlp_model.pt``).

7️⃣  Log the most important artefacts and metrics to a local MLflow tracking
   server (experiment name: ``CaliforniaHousing_ModelTraining``).

All artefacts (pipeline, models, JSON metadata) are written under the
project‑root ``C:\\mlops\\cali_housing_project``.  In addition, the trained
models are copied to ``./agent_workspace/trained_models`` as required by the
evaluation harness.

The script can be executed directly:

