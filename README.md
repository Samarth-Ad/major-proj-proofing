# Proof Recording — AutoML-Agent Generated-Code Reliability

**Handover packet for the proof-recording task.**

This repo exists for one deliverable: a **screen recording that proves, live, that the
LLM-powered AutoML system AutoML-Agent generates broken code** — you hand it a dataset,
it writes code from scratch with an LLM, and that code fails in real time.

**Everything is in this one repo** — the full AutoML-Agent system *and* the demo tooling
*and* a Docker image so you don't have to fight Python dependencies. Clone, build, record.

---

## 1. The task, in one line

> Record the terminal while AutoML-Agent is given a dataset and a plain-English task,
> generates a Python pipeline with an LLM, runs it, and the code **breaks on camera**.

The "money shot" is: dataset in → agent generates code → the code throws a real error
(ImportError / NameError / timeout / …) in the terminal, live.

---

## 2. Two ways to record (pick based on whether you have a working LLM)

| | **A. LIVE from zero** (preferred) | **B. Real re-execution of captured code** (fallback) |
|---|---|---|
| What it shows | The agent *generating* code live, then failing | A script the agent *already* generated, re-run live so the real error reappears |
| Needs a working LLM backend? | **Yes** | No |
| "From zero"? | Yes | No (replay of captured evidence) |
| Command (Docker) | `docker compose run --rm demo` | `docker compose run --rm demo python -m experiment.demo_broken` |

Do **A** if you can get any working LLM backend (see §4). If no backend is available,
**B** still shows genuinely broken generated code executing and erroring live — just not
the generation step.

---

## 3. Setup (Docker — recommended, no Python/pip pain)

Why Docker: the project's upstream `requirements.txt` is a kitchen-sink listing that will
not install cleanly on a fresh venv (ghost packages, mismatched torch/torchvision pins,
GPU-only deps). The Docker image uses a tested, trimmed requirements file and installs in
one go.

### Prerequisites
- **Docker Desktop** (Windows/macOS) or Docker Engine + `docker compose` v2 (Linux).
- About 4 GB of disk for the image and ~2 GB of RAM for the demo.

### Steps

```powershell
# 1. Clone and enter the repo
git clone https://github.com/Samarth-Ad/major-proj-proofing.git
cd major-proj-proofing

# 2. Create your .env from the template, then edit it (set ONE backend, see §4)
#    PowerShell:
copy .env.example .env
#    macOS / Linux:
#    cp .env.example .env

# 3. Build the image (one-off, ~5–15 min first time; cached after)
docker compose build

# 4. Run the LIVE demo (interactive — press Enter to advance)
docker compose run --rm demo
```

That's it. No venv, no pip, no "could not find a version that satisfies".

> **Fallback B** needs only steps 1 and 3 — no `.env` required. Run it with
> `docker compose run --rm demo python -m experiment.demo_broken`.

### If you prefer a bare-metal setup (not recommended)

Skip Docker only if you really want to: you will need Python 3.11, then
`pip install -r requirements-docker.txt` into a fresh venv (NOT the original
`requirements.txt` — that one does not install cleanly).

---

## 4. The LLM backend (read this — it's the only hard part)

The study ran on Ollama's cloud models, but **those free/monthly quotas are now
exhausted**, and the free Gemini / Groq / OpenRouter tiers were already rate/size-capped.
So to record the **live** demo you must supply one reachable model. Any of:

- **Local Ollama model (free, no quota, recommended if your machine can run a 7B model).**
  Install Ollama on your host (not inside the container), then:
  ```powershell
  ollama pull qwen2.5-coder:7b
  ```
  In `.env` set:
  ```
  AMLA_LLM=ollama
  OLLAMA_MODEL=qwen2.5-coder:7b
  OLLAMA_BASE_URL=http://host.docker.internal:11434/v1
  ```
  *(That `host.docker.internal` hostname lets the container reach the Ollama server
  running on your host. On Docker Desktop for Windows/macOS it works out of the box.
  On Linux it works because of `extra_hosts` in `docker-compose.yml`.)*
  A smaller local model tends to break code *more* often — good for the demo.

- **A paid / topped-up Ollama cloud** account → in `.env` keep
  `OLLAMA_MODEL=gpt-oss:120b-cloud` (the exact model from the study) and
  `OLLAMA_BASE_URL=https://ollama.com/v1`.

- **A fresh API key** for Gemini / Groq / OpenRouter — paste it into `.env`
  (`GEMINI_API_KEY=…` etc.) and set `AMLA_LLM=gemini` (or `groq` / `openrouter`).
  A single demo run makes ~40 LLM calls, so one reset daily free quota may cover
  one take.

Backend selection is read from `.env` at container start, so edit `.env` and rerun.

---

## 5. How to record — LIVE (option A)

```powershell
# full walkthrough: dataset → task → agent generates & runs code
docker compose run --rm demo

# pick the dataset (titanic & breast_cancer broke most reliably in the study)
docker compose run --rm demo python -m experiment.demo_live --dataset breast_cancer

# pick the backend/model key (ollama, ollama2, gemini, groq, openrouter)
docker compose run --rm demo python -m experiment.demo_live --model gemini --dataset titanic

# show the FULL generated script each attempt (default caps the preview at 45 lines)
docker compose run --rm demo python -m experiment.demo_live --code-lines 0
```

What you'll see, in order:
1. **STEP 1** — the dataset handed to the system (shape, columns, first rows).
2. **STEP 2** — the plain-English task/prompt (the agent gets *only* this + the CSV).
3. **STEP 3** — the real agent launches; each time it writes a script and runs it, the
   terminal prints the generated code and whether it **RAN** or **✗ BROKE** with the error.

A full run keeps retrying and can take 10–15 minutes. **Once you've captured a couple of
broken attempts, press `Ctrl-C`** — that's enough for the proof.

---

## 6. How to record — FALLBACK (option B, no backend needed)

If you cannot get a working LLM, this replays scripts the agent **already generated** and
re-runs them live, so the real tracebacks appear on camera:

```powershell
# curated showcase: 5 different failure types, press Enter between each
docker compose run --rm demo python -m experiment.demo_broken

# hands-off (no pauses)
docker compose run --rm demo python -m experiment.demo_broken --auto

# one failure category, or the full list
docker compose run --rm demo python -m experiment.demo_broken --category import_error --n 1
docker compose run --rm demo python -m experiment.demo_broken --list
```

No `.env` needed; no LLM needed.

---

## 7. Recording tips

- Use **Windows Terminal / PowerShell**, maximized, a **large font**, and a **dark
  theme** — the demo is colour-coded (yellow banners, cyan code, **red errors**).
- `Clear-Host` before you start for a clean first frame.
- Narrate the three steps: "here's the dataset → here's the task → watch it generate and
  fail." The press-Enter pacing gives you time to talk.
- Capture at least: one full generated script + its live error, and the final
  "✗ CRASHED / BROKEN" line with the exit code.
- If a live take happens to succeed on the first try (≈8% chance per the study), just
  run it again or switch `--dataset` — first-attempt breakage was ~92%.

---

## 8. What's already proven (say this on camera / in the write-up)

From the full 36-cell study (2 model families × 6 tabular datasets × 3 reps), all with
Wilson 95% CIs:

- **First generated script broken: 92%** [78–97]
- **Per-execution broken: 91%** [88–94] (309 executions)
- **Still unrecoverable after all repair loops: 25%** [14–41]
- **Semantic "runs-but-wrong" (of clean exits): 33%** [19–52]
- Net: only **18 of 36 runs (50%)** produced code that both ran and was correct.

Dominant failure modes: missing/hallucinated imports (94), runtime crashes (45), and
non-terminating code (37).

---

## 9. What's in this repo

This one repo bundles the whole system plus the demo and the proof:

```
major-proj-proofing/
├── README.md              ← this handover doc
├── Dockerfile             ← build target: python:3.11-slim + the trimmed deps
├── docker-compose.yml     ← `docker compose run --rm demo` entry point
├── requirements-docker.txt← the pragmatic, actually-installs-clean dep list
├── requirements.txt       ← original paper requirements (kept for reference only)
├── .env.example           ← copy to .env and set ONE backend
├── configs.py             ← backend/LLM routing (AMLA_LLM, model ids, base URLs)
├── agent_manager/ operation_agent/ data_agent/ model_agent/ prompt_agent/
│                          ← the AutoML-Agent system under test (unmodified)
├── prompt_pool/ utils/ experiments/ example_plans/   ← supporting modules
└── experiment/            ← the study harness + demo tooling + proof
    ├── demo_live.py       ← LIVE demo: dataset → real agent → code generated & broken (A)
    ├── demo_broken.py     ← fallback: re-run captured broken scripts live (B)
    ├── config.py · instrumentation.py · run_grid.py · analyze.py · report.py · …
    ├── datasets/          ← the 6 tabular CSVs + manifest
    ├── broken_code/       ← 282 captured broken scripts by error type (+ INDEX.md)
    ├── results/           ← reports, figures, summary.json (the study findings)
    ├── docs/              ← HOW_IT_WORKS, FILE_INVENTORY, REPLICATION_GUIDE
    └── ledger.jsonl · runs/   ← per-attempt records + outcomes/provenance
```

Start with `experiment/docs/HOW_IT_WORKS.md` for the plain-language A→Z of the whole
experiment, and `experiment/broken_code/INDEX.md` for the catalogue of captured failures.

---

## 10. Quick checklist for the recorder

- [ ] Installed Docker Desktop (Windows/macOS) or docker+compose (Linux).
- [ ] Cloned this repo.
- [ ] `cp .env.example .env` and set ONE backend (§4) — or decide to use fallback B.
- [ ] `docker compose build` (one-off, 5–15 min first time).
- [ ] Terminal maximized, dark theme, big font, `Clear-Host`.
- [ ] Recording started.
- [ ] Ran `docker compose run --rm demo` (or the demo_broken command for B).
- [ ] Captured: dataset → task → generated code → **live error** → crash line.
- [ ] `Ctrl-C` after a couple of broken attempts; stop recording.

---

## 11. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `docker: command not found` | Install Docker Desktop and restart your shell. |
| Build fails with a torch wheel error | Make sure Docker has at least 4 GB RAM and 10 GB disk available (Settings → Resources on Docker Desktop). |
| Container can't reach local Ollama | Set `OLLAMA_BASE_URL=http://host.docker.internal:11434/v1` in `.env`, and make sure `ollama serve` is running on the host. |
| `402 Add usage credits` from Ollama | Your Ollama cloud quota is exhausted; switch to a local model (see §4) or use fallback B. |
| 429 / 413 from Gemini / Groq / OpenRouter | Free-tier limit hit; wait for the daily quota to reset, or use fallback B. |
| Live demo succeeds on the first try | Just run it again with a different `--dataset`; first-attempt breakage was ~92% in the study, not 100%. |
| ANSI colours look like escape codes | Use Windows Terminal (not the legacy `cmd.exe` console) or any modern terminal on macOS / Linux. |
