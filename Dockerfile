# AutoML-Agent proof-recording image.
#
# Builds a single CPU-only image with the framework + the common ML libraries
# the generated code reaches for. Interactive by default, so the press-Enter
# pacing in demo_live works inside `docker compose run`.
FROM python:3.11-slim-bookworm

# System deps: build tools for ML wheels that don't ship binaries, libgomp for
# xgboost/lightgbm, and git for packages that pull from VCS.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        libgomp1 \
        libstdc++6 \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # The demo_live module reads AMLA_LLM + a per-backend key from .env; nothing
    # is baked into the image. The ANSI colours rely on utf-8 stdio.
    PYTHONIOENCODING=utf-8

WORKDIR /app

# Install Python deps first (leverages Docker layer cache on code changes).
COPY requirements-docker.txt /app/requirements-docker.txt
RUN pip install --upgrade pip \
 && pip install -r /app/requirements-docker.txt

# Then the project source.
COPY . /app

# Build-time smoke test: fail the build early if a pip wheel is missing / broken
# (so the recorder sees the problem at `docker build`, not mid-take). Only
# side-effect-free imports are tested here; framework modules that read env at
# import time are exercised at runtime by the demo.
RUN python -c "import openai, pandas, numpy, sklearn, bs4, dotenv, requests, validators, num2words, torch, xgboost, lightgbm, catboost, scipy, matplotlib, seaborn, pyarrow, fastparquet; \
from experiment import config, instrumentation; \
print('[smoke] dependencies OK')"

# Interactive by default — run the live demo. Override at run time with e.g.
#   docker compose run --rm demo python -m experiment.demo_broken --auto
CMD ["python", "-m", "experiment.demo_live", "--dataset", "titanic"]
