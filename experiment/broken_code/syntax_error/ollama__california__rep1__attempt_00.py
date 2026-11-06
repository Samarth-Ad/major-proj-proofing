
"""
California Housing – Tabular Regression End‑to‑End Pipeline
================================================================
Implements the full workflow described in the specification:

1️⃣  Data ingestion & checksum + pandera validation
2️⃣  Train/validation/test split
3️⃣  Sklearn preprocessing pipeline
    – Winsorising (1st/99th percentile)
    – Log‑skew handling (log1p for |skew|>1)
    – Engineered ratios
    – StandardScaler
4️⃣  LightGBM training with the fixed hyper‑parameters
5️⃣  Model & pipeline persistence
6️⃣  Export to ONNX + tiny PyTorch wrapper
7️⃣  Simple Gradio UI for demo inference
8️⃣  All artefacts are saved under ``./agent_workspace/trained_models``

Run the script directly (``python run_pipeline.py``).  It will create the
required folder structure, log progress to the console and produce the
following files:

