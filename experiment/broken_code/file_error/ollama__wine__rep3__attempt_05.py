
# src/config.py
import yaml
from pathlib import Path

# Resolve repository root (the folder that contains this file's parent)
ROOT = Path(__file__).resolve().parents[1]

# Load YAML
with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
    _raw_cfg = yaml.safe_load(f)

# Convert every path under `paths` to an absolute Path object
PATHS = {}
for key, rel_path in _raw_cfg["paths"].items():
    PATHS[key] = (ROOT / rel_path).resolve()

# Build a single dictionary that mimics the original YAML structure,
# but with absolute Path objects for the `paths` section.
CONFIG = _raw_cfg.copy()
CONFIG["paths"] = PATHS
