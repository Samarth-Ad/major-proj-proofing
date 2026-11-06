# Proof Recording — AutoML-Agent Generated-Code Reliability

**Handover packet for the proof-recording task.**

This repo exists for one deliverable: a **screen recording that proves, live, that
the LLM-powered AutoML system AutoML-Agent generates broken code** — you hand it a
dataset, it writes code from scratch, and that code fails in real time.

**This one repo contains everything** — the full AutoML-Agent system *and* the demo
tooling (under `experiment/`), the datasets, and the already-captured proof. Clone it,
read this file, record. Nothing else to download.

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
| Command | `python -m experiment.demo_live` | `python -m experiment.demo_broken` |

Do **A** if you can get any working LLM backend (see §4). If no backend is available,
**B** still shows genuinely broken generated code executing and erroring live — just not
the generation step.

---

## 3. Setup (one clone, ~5 minutes)

Everything is in this one repo — the full, unmodified AutoML-Agent system **and** the
demo tooling under `experiment/`. No other download needed.

1. **Clone and enter the repo:**
   ```powershell
   git clone https://github.com/Samarth-Ad/major-proj-proofing.git
   cd major-proj-proofing
   ```
2. **Create a virtual environment and install dependencies:**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
3. **Provide a working LLM backend** — copy the template and fill in ONE backend (see §4):
   ```powershell
   copy .env.example .env
   # then edit .env: set AMLA_LLM and the matching key + model
   ```

> Fallback **B** (§6) needs steps 1–2 only — no backend, no `.env`.

---

## 4. The LLM backend (read this — it's the only hard part)

The study ran on Ollama's cloud models, but **those free/monthly quotas are now
exhausted**, and the free Gemini / Groq / OpenRouter tiers were already rate/size-capped.
So to record the **live** demo you must supply one reachable model. Any of:

- **Local Ollama model (free, no quota, recommended if the machine can run it).**
  ```powershell
  ollama pull qwen2.5-coder:7b
  $env:OLLAMA_MODEL="qwen2.5-coder:7b"
  ```
  The `ollama` backend already points at the local server (`http://localhost:11434/v1`
  in `configs.py`). A smaller local model tends to break code *more* often — good for
  the demo.
- **A paid / topped-up Ollama cloud** account → keep `OLLAMA_MODEL=gpt-oss:120b-cloud`
  (the exact model from the study).
- **A fresh API key** for Gemini / Groq / OpenRouter in `.env`
  (`GEMINI_API_KEY=…`, etc.) and select it with `--model gemini` (see §5). A single
  demo run is ~40 LLM calls, so a reset daily free quota may be enough for one take.

Backend selection is read from the environment at launch, so set it **before** running.

---

## 5. How to record — LIVE (option A)

From the repo root (venv activated, backend set in `.env`):

```powershell
# full walkthrough: shows the dataset, the task, then the agent generating + running code
.\.venv\Scripts\python.exe -m experiment.demo_live --dataset titanic
```

Useful variants:

```powershell
# pick the dataset (titanic & breast_cancer broke most reliably in the study)
.\.venv\Scripts\python.exe -m experiment.demo_live --dataset breast_cancer

# pick the backend/model key (ollama, ollama2, gemini, groq, openrouter)
.\.venv\Scripts\python.exe -m experiment.demo_live --model gemini --dataset titanic

# show the FULL generated script each attempt (default caps the preview at 45 lines)
.\.venv\Scripts\python.exe -m experiment.demo_live --code-lines 0
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
.\.venv\Scripts\python.exe -m experiment.demo_broken

# hands-off (no pauses)
.\.venv\Scripts\python.exe -m experiment.demo_broken --auto

# one failure category, or the full list
.\.venv\Scripts\python.exe -m experiment.demo_broken --category import_error --n 1
.\.venv\Scripts\python.exe -m experiment.demo_broken --list
```

This needs steps 1–2 of setup (the venv — it executes the scripts), but **no LLM backend** and no `.env`.

---

## 7. Recording tips

- Use **Windows Terminal / PowerShell**, maximized, a **large font**, and a **dark
  theme** — the demo is colour-coded (yellow banners, cyan code, **red errors**).
- `Clear-Host` before you start for a clean first frame.
- Narrate the three steps: "here's the dataset → here's the task → watch it generate and
  fail." The press-Enter pacing (omit `--auto`) gives you time to talk.
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
├── requirements.txt       ← install into a venv
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

- [ ] Cloned this repo.
- [ ] venv created, `requirements.txt` installed.
- [ ] A working LLM backend set (local Ollama model, or a key in `.env`) — or decide to
      use fallback **B**.
- [ ] Terminal maximized, dark theme, big font, `Clear-Host`.
- [ ] Recording started.
- [ ] Ran `python -m experiment.demo_live --dataset titanic` (or `demo_broken` for B).
- [ ] Captured: dataset → task → generated code → **live error** → crash line.
- [ ] `Ctrl-C` after a couple of broken attempts; stop recording.
