# How This Experiment Works — A to Z (plain language)

This document explains, step by step and in simple terms, exactly how the
AutoML-Agent reliability study runs — from an empty folder to the final report.

---

## The one-sentence idea

> Run AutoML-Agent normally, at its strongest settings, many times — and quietly
> record **every** piece of code it writes and whether that code ran — so we can
> measure how often its generated code is broken, **without changing the agent at all.**

The hard part is the word *every*. AutoML-Agent overwrites its own script file each
time it rewrites the code, and throws away old error logs when it retries. So if you
only look at the end, you massively undercount the failures. The trick of this study
is to **tap the execution step** and save a copy of each attempt as it happens.

---

## The vocabulary (5 words)

- **Backend / model** — which LLM writes the code (e.g. `gpt-oss:120b`, `gemma4`).
- **Dataset** — a small tabular CSV (iris, titanic, …).
- **Task** — classification, regression, or clustering.
- **Rep** — a repetition (we run each combination 3 times, because LLMs are random).
- **Cell** — one single run = one (backend × dataset × rep). The grid has 36 cells.

---

## The big picture (one diagram)

```
                 config.py  (the one place that defines everything)
                     │
   build_datasets.py │ baselines.py      ← run once: make the 6 CSVs + the "floor" scores
                     ▼
            ┌──────────────────────┐
            │     run_grid.py      │  ← the ORCHESTRATOR (parent process)
            │  loops over 36 cells │
            └──────────┬───────────┘
                       │  for each cell: launch a fresh child process,
                       │  watch it with a 30-minute timer, then record the outcome
                       ▼
            ┌──────────────────────┐
            │  run_grid.py --cell  │  ← the CHILD (one cell, its own process)
            │  1. pick the LLM     │
            │  2. install the TAP  │  ← instrumentation.py (archiver + throttle)
            │  3. run AutoML-Agent │  ← agent_manager.AgentManager (UNMODIFIED)
            └──────────┬───────────┘
                       │  the TAP writes one line per execution →
                       ▼
             ledger.jsonl   +   runs/.../attempt_NN.py   +   cell_status.json
                       │
                       ▼
            ┌──────────────────────┐
            │     analyze.py       │  ← turn raw logs into numbers (with error bars)
            │     report.py        │  ← turn numbers into a written report
            └──────────┬───────────┘
                       ▼
        results/summary.json · tables.md · report.md · figures/*.png
```

---

## Step 0 — Set up the inputs (done once)

**`build_datasets.py`** downloads/loads 6 classic tabular datasets, fixes a random
seed (42) so they're identical every time, and saves them as plain CSVs in
`datasets/`. It also writes `manifest.json` noting exactly how many rows/columns
each has and whether it's the real dataset or a synthetic fallback.

**`baselines.py`** computes a "how good is trivial?" score for each dataset:
- Classification → accuracy you'd get by always guessing the majority class.
- Regression → the error you'd get by always predicting the average value.
- Clustering → the agreement a basic KMeans gets vs. the true labels.

These floors are saved to `baselines.json` and used much later to judge whether a
result that *ran* is actually any good.

---

## Step 1 — Define the grid

**`config.py`** holds one list of models, one list of datasets, and the number of
reps. `grid_cells()` multiplies them out into 36 cells, each with a unique id like
`ollama__titanic__rep2`. It also holds the **one full-strength config** — the exact
AutoML-Agent settings used for every cell (all its reasoning/verification/revision
features ON). There is exactly one config so the study is clean: nothing varies
except the model, the data, and the repetition.

---

## Step 2 — The orchestrator loops over the cells

**`run_grid.py`** (parent mode) walks the 36 cells one at a time. For each cell it:

1. **Checks if it's already done** (reads `cell_status.json`). If yes, it skips it.
   This is what makes the whole run *resumable* — stop and restart anytime, nothing
   is lost or repeated.
2. **Launches the cell as a brand-new subprocess.** Each cell gets its own fresh
   Python process. Why? Isolation — one crashing or hanging run can't poison the
   next, and memory/state never leaks between cells.
3. **Watches it with a 30-minute timer.** If a cell runs past 30 minutes (e.g. the
   agent is stuck in a retry loop, or generated code that never finishes), the
   orchestrator force-kills the **entire process tree** (the cell and any scripts it
   spawned) and moves on.
4. **Records the outcome** into `cell_status.json`, then waits a few seconds and
   does the next cell.

If a cell fails because the LLM provider itself refused (rate limit, quota, request
too big), the orchestrator backs off and retries a couple of times before giving up.

---

## Step 3 — Inside one cell (the child process)

**`run_grid.py`** (child mode, `--cell`) does the actual run. Order matters here:

1. **Point the pipeline at this cell's LLM.** It sets two environment variables
   (`AMLA_LLM` = the backend, plus the model-id variable). This must happen *before*
   AutoML-Agent is imported, because AutoML-Agent reads those variables at import time.
2. **Install the tap** (from `instrumentation.py`) — see Step 4.
3. **Build the agent** exactly as a normal user would: `AgentManager(task=…,
   data_path=<the CSV>, user_requirements=<the JSON>, **full_strength_config)`.
4. **Press go:** `manager.initiate_chat(prompt)` with a plain-English request like
   *"Build a model that predicts passenger survival from the local CSV."*
5. From here AutoML-Agent does its normal thing — plan, decompose, write code, run
   it, judge it, repair it, repeat — completely unaware it's being watched.
6. When it finishes (or crashes), the child writes a `cell_result.json` with the
   final state, exit code, token usage, and timing.

A detail that keeps the study honest: the dataset is passed as a local file with
`source="user-upload"`, which tells AutoML-Agent to use our CSV and skip all its
internet/dataset-retrieval steps. So the only thing being measured is code
generation, not web lookups.

---

## Step 4 — The tap (how we catch every attempt)

This is the heart of the study, in **`instrumentation.py`**.

AutoML-Agent has one function that actually runs generated code: `execute_script`.
The tap replaces that function with a thin wrapper that:

1. Lets the **original** function run the code (so the agent behaves identically).
2. Then **copies the script** that just ran to `runs/<model>/<dataset>/rep<N>/attempt_NN.py`.
3. And **appends one line** to `ledger.jsonl` recording: which cell, which attempt
   number, the exit code (0 = ran, non-zero = broken), a fingerprint (SHA-256) of the
   code, how many tokens it cost, and the tail of the execution log (which contains
   the error message, if any).

Because this fires on **every** call — inside the inner coding loop *and* the outer
revision loop — we capture attempts the agent itself overwrites and forgets. That's
why the study counts 309 executions when the agent's own final record would show
only a handful.

The same module also installs a **throttle**: a minimum delay between LLM calls. This
only changes *timing* (to stay under free-tier rate limits); it never changes what the
agent decides to do.

---

## Step 5 — Deciding "infra problem" vs "broken code"

Not every failure is the agent's fault. If the LLM provider is down or rejects the
request, that's an **infrastructure** problem, not bad code. The orchestrator looks at
each finished cell and tags it `infra_error = true` **only if** it produced *zero* code
attempts *and* shows provider-error signals (like "429", "quota", "rate limit"). Those
cells are set aside and **excluded** from all the code-failure percentages, and
reported separately. This "validity guard" is what stops a flaky free API from being
mistaken for AutoML-Agent writing bad code.

---

## Step 6 — Turn logs into numbers

**`analyze.py`** reads `ledger.jsonl` + `cell_status.json` + `baselines.json` and
computes the five measures:

1. **First-attempt failure** — did the *very first* script a cell generated fail?
2. **Per-execution failure** — across all 309 executions, what fraction failed?
3. **Final outcome** — did the cell end with working code, or not (unrecoverable /
   incomplete)?
4. **Repair attempts to success** — how many broken scripts came before the first
   one that worked?
5. **Semantic "runs-but-wrong"** — for cells that *did* run cleanly, read the printed
   metric (accuracy, R², ARI…) and compare it to the baseline floor. A run that exits
   cleanly but reports no metric, a NaN, or a score worse than trivial is counted as a
   *semantic* failure — "it ran, but the answer is useless."

Every percentage is reported with a **Wilson 95% confidence interval**, which is the
statistically correct error bar for "k out of n" when n is small — so the report never
overclaims precision.

It also builds the **error taxonomy** (bucketing each failure by its Python exception:
import error, timeout, wrong API usage, …) and three figures.

Outputs: `summary.json` (all numbers), `tables.md` (tables), and `figures/*.png`.

---

## Step 7 — Turn numbers into a report

**`report.py`** reads `summary.json` and writes `results/report.md` — a ready-to-read
Methods + Results write-up. (The richer `DETAILED_REPORT.md` was assembled on top of
the same numbers with extra per-task/per-dataset cuts and real failure examples.)

---

## How to run the whole thing (5 commands)

```bash
python -m experiment.build_datasets     # 1. make the 6 CSVs
python -m experiment.baselines          # 2. compute classical floors
AMLA_EXP_MODELS="ollama,ollama2" python -m experiment.run_grid   # 3. run the 36-cell grid (resumable)
python -m experiment.analyze            # 4. compute metrics + figures
python -m experiment.report             # 5. write the report
```

Steps 1–2 run once. Step 3 is the long one (~6 hours here). Steps 4–5 can be re-run
anytime; they only read the saved logs.

---

## Why each design choice matters (the short version)

| Choice | Why |
|--------|-----|
| **Don't modify the agent** | So the result reflects the real system, not our edits. We only observe. |
| **Archive every attempt** | The agent overwrites its own files; the end state hides most failures. |
| **One subprocess per cell** | Isolation + a hard watchdog kill for hangs and infinite loops. |
| **One fixed full-strength config** | A clean study: only model/data/rep vary. The result is a floor, not a strawman. |
| **Infra-vs-code separation** | A flaky free API must never be miscounted as broken code. |
| **Local CSV, no retrieval** | Measure code generation, not internet lookups; keeps runs reproducible. |
| **Wilson CIs + 3 reps** | Honest error bars for small samples and non-deterministic models. |
| **Semantic check** | "It ran" ≠ "it's right"; a clean exit with a garbage metric still counts as a failure. |
