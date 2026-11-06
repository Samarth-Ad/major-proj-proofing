"""Central configuration for AutoML-Agent.

This version removes the hard OpenAI dependency. Every agent talks to the LLM
through an OpenAI-compatible client (see ``utils.get_client``), so any provider
that exposes an OpenAI-compatible ``/v1`` endpoint works by adding an entry to
``AVAILABLE_LLMs`` whose key does NOT start with ``"gpt"`` (keys starting with
``gpt`` are routed to api.openai.com without a base_url).

Secrets are read from a local ``.env`` file (see ``.env.example``). Pick the
active backend with ``AMLA_LLM`` (gemini | groq | openrouter | ollama).
"""

import os

from dotenv import load_dotenv

load_dotenv()


class Configs:
    # OpenAI kept only for backward-compatibility; unused unless AMLA_LLM="gpt-*".
    OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
    HF_KEY = os.getenv("HF_TOKEN", "")
    PWC_KEY = os.getenv("PWC_KEY", "")
    SEARCHAPI_API_KEY = os.getenv("SERPAPI_KEY", "")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")


# Which backend the pipeline uses. Must match a key in AVAILABLE_LLMs below.
ACTIVE_LLM = os.getenv("AMLA_LLM", "gemini")


AVAILABLE_LLMs = {
    # --- Free / self-hosted OpenAI-compatible backends -----------------------
    "gemini": {
        "api_key": os.getenv("GEMINI_API_KEY", ""),
        "model": os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
    },
    "groq": {
        "api_key": os.getenv("GROQ_API_KEY", ""),
        "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        "base_url": "https://api.groq.com/openai/v1",
    },
    "openrouter": {
        "api_key": os.getenv("OPENROUTER_API_KEY", ""),
        "model": os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-chat-v3.1:free"),
        "base_url": "https://openrouter.ai/api/v1",
    },
    "ollama": {
        # Ollama ignores the key, but the OpenAI client requires a non-empty one.
        "api_key": os.getenv("OLLAMA_API_KEY", "ollama"),
        # Cloud models (e.g. "gpt-oss:120b-cloud") are served through the local
        # Ollama endpoint after `ollama signin` + `ollama pull <model>`.
        "model": os.getenv("OLLAMA_MODEL", "gpt-oss:120b-cloud"),
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
    },
    # --- OpenAI (kept for reference; needs OPENAI_API_KEY) -------------------
    "gpt-4.1": {"api_key": Configs.OPENAI_KEY, "model": "gpt-4.1"},
    "gpt-4": {"api_key": Configs.OPENAI_KEY, "model": "gpt-4o"},
    "gpt-3.5": {"api_key": Configs.OPENAI_KEY, "model": "gpt-3.5-turbo"},
}

# The Prompt Agent builds its own client from the "prompt-llm" entry. Point it
# at the active backend so it no longer needs the local vLLM/Mixtral server.
AVAILABLE_LLMs["prompt-llm"] = dict(AVAILABLE_LLMs[ACTIVE_LLM])

TASK_METRICS = {
    "image_classification": "accuracy",
    "text_classification": "accuracy",
    "tabular_classification": "F1",
    "tabular_regression": "RMSLE",
    "tabular_clustering": "RI",
    "node_classification": "accuracy",
    "ts_forecasting": "RMSLE",
}
