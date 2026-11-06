
"""
Wine‑DEC clustering pipeline
---------------------------

This script is a **stand‑alone** implementation of the end‑to‑end solution
described in the executive summary.  It follows the folder layout and
configuration that were prescribed, but everything is kept in a single
file so that it can be executed immediately after cloning the repository.

The pipeline performs:

1.  Data ingestion (copy the user‑provided CSV, validation, split)
2.  Pre‑processing (SimpleImputer + StandardScaler) wrapped in a
    `sklearn.pipeline.Pipeline`
3.  Deep‑Embedded Clustering (DEC) implemented with
    **PyTorch‑Lightning**
4.  Evaluation (ARI, AMI, Silhouette) on the whole data set
5.  Saving of the preprocessing pipeline and the best DEC checkpoint
    to ``./agent_workspace/trained_models``
6.  A tiny Gradio demo that lets a user upload a CSV and returns the
    predicted cluster ids.

The script can be run from the project root:

