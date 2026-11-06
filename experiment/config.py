"""Single source of truth for the scaled AutoML-Agent reliability study.

Holds the one full-strength config applied to every grid cell, the model
(backend) roster, the dataset specs, task->metric map, and every filesystem path.
All other modules in `experiment/` import from here so there is exactly one place
to change a knob, a model id, or a dataset.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# --- Filesystem paths --------------------------------------------------------
PKG_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PKG_DIR.parent
DATASET_DIR = PKG_DIR / "datasets"
MANIFEST_PATH = DATASET_DIR / "manifest.json"
BASELINES_PATH = PKG_DIR / "baselines.json"
RUNS_DIR = PKG_DIR / "runs"              # archived scripts + per-cell logs/results
LEDGER_PATH = PKG_DIR / "ledger.jsonl"   # one line per execute_script call
RESULTS_DIR = PKG_DIR / "results"        # analyze/report outputs
FIGURES_DIR = RESULTS_DIR / "figures"

# --- The one full-strength config (applied to every cell) --------------------
# These are the AgentManager constructor defaults for the quality knobs
# (agent_manager/__init__.py:71-84), with two documented, justified deviations:
#   full_pipeline=False -- a *scope* knob, not a safeguard. True makes the coder
#     build a Gradio app that launches a server then dies on the 300s exec
#     timeout (operation_agent/__init__.py:97-101): a pure artifact, not
#     code-brokenness.
#   rap=False -- keeps runs offline/reproducible and removes an uncontrolled
#     external-knowledge confound + Kaggle/SerpApi crash risk. The single
#     explicit limitation; every *reasoning* safeguard stays ON.
FULL_CONFIG: dict[str, object] = {
    "verification": True,   # input gating + pre-exec Pass/Fail judge ON
    "decomp": True,         # per-agent plan decomposition ON
    "n_plans": 3,           # multi-plan exploration ON
    "n_candidates": 3,      # multi-model selection ON
    "n_revise": 3,          # repair + plan-revision loops ON
    "interactive": False,   # headless/batch
    "full_pipeline": False,
    "rap": False,
}

REPS = int(os.getenv("AMLA_EXP_REPS", "3"))
SNAPSHOT_DATE = os.getenv("AMLA_SNAPSHOT_DATE", "2026-10-31")

# Per-cell wall-clock watchdog (seconds). Covers the empty-completion
# non-increment loop (operation_agent/__init__.py:121-123) and the unbounded
# provider-retry `while True` loops in agent_manager (e.g. :216, :293, :693).
# Default 30 min: throttled free-tier runs make ~40 calls/cell, each spaced.
CELL_TIMEOUT_SEC = int(os.getenv("AMLA_CELL_TIMEOUT", "1800"))

# Transport-layer throttle: minimum seconds between the agent's LLM calls, to
# keep free tiers under their per-minute request caps. This changes only call
# TIMING (not agent decision logic). 0 = no throttle (e.g. local Ollama, which
# has no per-minute cap). Override globally with AMLA_MIN_CALL_INTERVAL.
DEFAULT_THROTTLE_SEC: dict[str, float] = {
    "gemini": 5.0, "openrouter": 5.0, "openrouter2": 5.0, "groq": 3.0,
    "ollama": 0.0, "ollama2": 0.0,
}


def throttle_for(backend: str) -> float:
    override = os.getenv("AMLA_MIN_CALL_INTERVAL")
    if override is not None:
        return float(override)
    return DEFAULT_THROTTLE_SEC.get(backend, 5.0)


# --- Models (backends) -------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    key: str        # short id used in paths/results
    backend: str    # AMLA_LLM value + AgentManager llm= (gemini|groq|openrouter|ollama)
    model_env: str  # env var configs.py reads the model id from for this backend
    model_id: str   # pinned model id


def _model(key: str, backend: str, model_env: str, override_env: str, default: str) -> ModelSpec:
    return ModelSpec(key=key, backend=backend, model_env=model_env,
                     model_id=os.getenv(override_env, default))


# Default roster (4 backends). Groq is at-risk: its 8000 TPM per-request cap has
# already 413'd this system's ~9k-token prompts, and full-strength prompts are
# larger. Contingency: drop groq and use a 2nd OpenRouter free model by setting
#   AMLA_EXP_MODELS="gemini,openrouter,ollama,openrouter2"
ALL_MODELS: dict[str, ModelSpec] = {
    "gemini": _model("gemini", "gemini", "GEMINI_MODEL",
                     "GEMINI_MODEL", "gemini-flash-latest"),
    "openrouter": _model("openrouter", "openrouter", "OPENROUTER_MODEL",
                         "OPENROUTER_MODEL", "nvidia/nemotron-3.5-super-vl:free"),
    "ollama": _model("ollama", "ollama", "OLLAMA_MODEL",
                     "OLLAMA_MODEL", "gpt-oss:120b-cloud"),
    "groq": _model("groq", "groq", "GROQ_MODEL",
                   "GROQ_MODEL", "openai/gpt-oss-120b"),
    # Contingency 2nd OpenRouter model (same backend, different model id).
    "openrouter2": _model("openrouter2", "openrouter", "OPENROUTER_MODEL",
                          "OPENROUTER_MODEL_2", "meta-llama/llama-3.3-70b-instruct:free"),
    # Ollama cloud models (no per-minute/daily/size caps). Distinct families via
    # one gateway; the reliable roster when free provider tiers are capped.
    # "ollama" above = gpt-oss:120b-cloud (OpenAI family); this adds Google Gemma.
    "ollama2": _model("ollama2", "ollama", "OLLAMA_MODEL",
                      "OLLAMA_MODEL_2", "gemma4:cloud"),
}


def selected_models() -> list[ModelSpec]:
    """Roster for this run, chosen by AMLA_EXP_MODELS (comma-separated keys)."""
    keys = os.getenv("AMLA_EXP_MODELS", "gemini,openrouter,ollama,groq")
    out: list[ModelSpec] = []
    for key in (k.strip() for k in keys.split(",") if k.strip()):
        if key not in ALL_MODELS:
            raise KeyError(f"Unknown model key '{key}'; known: {sorted(ALL_MODELS)}")
        out.append(ALL_MODELS[key])
    return out


# --- Datasets ----------------------------------------------------------------
@dataclass(frozen=True)
class DatasetSpec:
    id: str          # file stem + id in manifest/results
    task: str        # tabular_classification | tabular_regression | tabular_clustering
    target: str      # target column name in the CSV
    builder: str     # builder key in build_datasets.py
    area: str        # requirements["problem"]["area"]
    domain: str      # requirements["problem"]["application_domain"]
    description: str # requirements["problem"]["description"]
    ds_name: str     # requirements["dataset"][0]["name"]
    data_desc: str   # requirements["dataset"][0]["description"]
    prompt: str      # free-text user prompt handed to initiate_chat


DATASETS: list[DatasetSpec] = [
    DatasetSpec(
        id="breast_cancer", task="tabular_classification", target="target",
        builder="breast_cancer", area="tabular data analysis",
        domain="healthcare diagnostics",
        description="Binary classification of breast tumors (malignant=0, benign=1) from 30 numeric features.",
        ds_name="Breast Cancer Wisconsin",
        data_desc="Local CSV, 569 rows, 30 numeric feature columns + binary 'target'.",
        prompt="Build a model that classifies breast tumors as malignant or benign from the local CSV.",
    ),
    DatasetSpec(
        id="titanic", task="tabular_classification", target="survived",
        builder="titanic", area="tabular data analysis",
        domain="passenger survival analysis",
        description="Binary survival classification from mixed-type features with missing values (a preprocessing stressor).",
        ds_name="Titanic",
        data_desc="Local CSV with categorical columns (e.g. sex, embarked, pclass), numeric columns with missing values (e.g. age, fare), and a binary 'survived' target.",
        prompt="Build a model that predicts passenger survival from the local CSV. The data has categorical columns and missing values that need preprocessing.",
    ),
    DatasetSpec(
        id="diabetes", task="tabular_regression", target="target",
        builder="diabetes", area="tabular data analysis", domain="healthcare",
        description="Predict a continuous diabetes disease-progression score from 10 numeric features.",
        ds_name="Diabetes",
        data_desc="Local CSV, 442 rows, 10 numeric feature columns + continuous positive 'target'.",
        prompt="Build a regression model that predicts the diabetes progression score from the local CSV.",
    ),
    DatasetSpec(
        id="california", task="tabular_regression", target="target",
        builder="california", area="tabular data analysis", domain="real estate",
        description="Predict median California house value (in $100k units) from 8 numeric features.",
        ds_name="California Housing (subsample)",
        data_desc="Local CSV, ~1500 rows, 8 numeric feature columns + continuous positive 'target'.",
        prompt="Build a regression model that predicts the median house value from the local CSV.",
    ),
    DatasetSpec(
        id="wine", task="tabular_clustering", target="target",
        builder="wine", area="tabular data analysis", domain="chemistry",
        description="Cluster wine samples into groups from 13 numeric chemical features; 'target' holds the true cluster for evaluation.",
        ds_name="Wine",
        data_desc="Local CSV, 178 rows, 13 numeric feature columns + integer 'target' (0/1/2) usable as ground-truth clusters.",
        prompt="Cluster the wine samples into groups using the local CSV and evaluate against the true labels.",
    ),
    DatasetSpec(
        id="iris", task="tabular_clustering", target="target",
        builder="iris", area="tabular data analysis", domain="botany",
        description="Cluster iris flowers into groups from 4 numeric measurements; 'target' holds the true cluster for evaluation.",
        ds_name="Iris",
        data_desc="Local CSV, 150 rows, 4 numeric feature columns + integer 'target' (0/1/2) usable as ground-truth clusters.",
        prompt="Cluster the iris samples into groups using the local CSV and evaluate against the true labels.",
    ),
]

# Primary metric name per task (matches configs.TASK_METRICS conventions).
TASK_METRIC: dict[str, str] = {
    "tabular_classification": "F1",
    "tabular_regression": "RMSLE",
    "tabular_clustering": "RI",
}


def dataset_by_id(dataset_id: str) -> DatasetSpec:
    for spec in DATASETS:
        if spec.id == dataset_id:
            return spec
    raise KeyError(f"Unknown dataset id '{dataset_id}'")


def dataset_csv(dataset_id: str) -> Path:
    """Absolute path to a dataset CSV inside the experiment's evidence tree."""
    return DATASET_DIR / f"{dataset_id}.csv"


def task_phrase(task: str) -> str:
    """'tabular_classification' -> 'tabular classification'."""
    return task.replace("_", " ")


def requirements_for(spec: DatasetSpec) -> dict:
    """Build the AutoML-Agent user_requirements JSON for a dataset.

    source='user-upload' + model=[] route the run to the local CSV and skip all
    retrieval (data_agent/retriever.py:24, model_agent/retriever.py:16).
    """
    return {
        "user": {"intent": "build", "expertise": "low"},
        "problem": {
            "area": spec.area,
            "downstream_task": task_phrase(spec.task),
            "application_domain": spec.domain,
            "description": spec.description,
            "performance_metrics": [],
            "complexity_metrics": [],
        },
        "dataset": [
            {
                "name": spec.ds_name,
                "modality": ["tabular"],
                "target_variables": [spec.target],
                "description": spec.data_desc,
                "preprocessing": [],
                "augmentation": [],
                "visualization": [],
                "source": "user-upload",
            }
        ],
        "model": [],
        "confidence": {"rationale": "Fully specified; data provided locally.", "score": 0.9},
    }


def grid_cells() -> list[dict]:
    """Every (model x dataset x rep) cell as a plain dict, in run order."""
    cells: list[dict] = []
    for model in selected_models():
        for spec in DATASETS:
            for rep in range(1, REPS + 1):
                cells.append(
                    {
                        "cell_id": f"{model.key}__{spec.id}__rep{rep}",
                        "model_key": model.key,
                        "backend": model.backend,
                        "model_env": model.model_env,
                        "model_id": model.model_id,
                        "dataset": spec.id,
                        "task": spec.task,
                        "rep": rep,
                    }
                )
    return cells
