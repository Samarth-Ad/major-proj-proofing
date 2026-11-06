# Generated-Code Reliability of AutoML-Agent at Full Strength
### A scaled, instrumentation-only reliability study

> **Thesis.** *Even run at its full as-designed configuration — with every reasoning, verification, and revision safeguard enabled — the LLM-supported AutoML system AutoML-Agent produces broken code at a significant rate: not always, but often, across multiple LLM backends and tabular tasks.*

| | |
|---|---|
| AutoML-Agent commit | `9e0d842959a09d96c00768c14d73bd9df6d39023` |
| Provider snapshot | `2026-10-31` |
| Grid started | `2026-10-31T21:08:17` |
| Cells recorded | 39 (36 code-bearing + 3 infra-excluded) |
| Executions logged | 309 |
| Total LLM tokens | 4,117,016 |
| Total wall-clock | 6.36 h (mean 587 s/cell) |

---

## Contents

1. [Executive summary](#1-executive-summary)
2. [Background and motivation](#2-background-and-motivation)
3. [Methodology](#3-methodology)
4. [Results](#4-results)
5. [Concrete failure examples](#5-concrete-failure-examples)
6. [Discussion](#6-discussion)
7. [Limitations](#7-limitations)
8. [Reproducibility and evidence tree](#8-reproducibility-and-evidence-tree)
9. [Conclusion](#9-conclusion)
10. [Appendix: per-cell outcomes](#10-appendix-per-cell-outcomes)

---

## 1. Executive summary

AutoML-Agent was run **unmodified**, at its **full as-designed configuration**, across a grid of 36 cells (2 LLM backends × 6 tabular datasets × 3 repetitions) spanning classification, regression, and clustering. A logging-only harness captured **every** generated script and execution outcome — 309 executions in all — without touching the agent's decision logic.

**Headline findings (Wilson 95% confidence intervals):**

| Measure | Rate | 95% CI | n |
|---------|------|--------|---|
| **First generated script broken** | **91.7%** | [78.2 – 97.1] | 36 cells |
| **Per-execution broken** (every attempt) | **91.3%** | [87.6 – 93.9] | 309 executions |
| **Still unrecoverable after all repair loops** | **25.0%** | [13.8 – 41.1] | 36 cells |
| **Semantic "runs-but-wrong"** (of clean exits) | **33.3%** | [18.6 – 52.2] | 27 cells |

The single most important number for the thesis: of the 36 runs, only **18 (50%)** ended with code that both executed cleanly **and** produced a correct, plausible result. The other half either never produced working code (9 cells, 25%) or produced code that ran but reported a missing or implausible metric (9 cells).

The agent's repair and revision machinery is doing real work — it rescues the large majority of the 92% first-attempt failures down to a 25% final-failure rate — but it does so at a cost of **8.6 execution attempts per cell on average**, and a quarter of the time it never converges.

---

## 2. Background and motivation

AutoML-Agent is a multi-agent LLM system (Manager, Prompt, Data, Model, and Operation agents) that turns a natural-language ML request plus a dataset into an executable end-to-end pipeline. Like all LLM code generators, its usefulness hinges on a question that is rarely measured directly: **how often is the code it writes actually broken?**

Marketing-style demos show the happy path. A reliability study needs the opposite: a fixed, honest configuration run at scale, with every generated attempt captured — including the ones the system silently regenerates over the top of. This study measures code-brokenness at the system's *strongest* setting, so the result is a floor on how bad things are, not a strawman.

---

## 3. Methodology

### 3.1 System under test — one full-strength configuration

Every cell used the identical configuration below. These are AutoML-Agent's own constructor defaults for the quality knobs, with exactly two documented deviations.

| Knob | Value | Role |
|------|-------|------|
| `verification` | `True` | input gating + pre-execution Pass/Fail judge **ON** |
| `decomp` | `True` | per-agent plan decomposition **ON** |
| `n_plans` | `3` | multi-plan exploration **ON** |
| `n_candidates` | `3` | multi-candidate model selection **ON** |
| `n_revise` | `3` | repair + plan-revision loops **ON** |
| `interactive` | `False` | headless/batch operation |
| `full_pipeline` | **`False`** | *Scope knob, not a safeguard.* `True` makes the coder build a Gradio web app that launches a server and is then killed by the 300 s execution timeout — a pure measurement artifact, not code brokenness. |
| `rap` | **`False`** | *The single explicit limitation.* Retrieval-augmented planning is disabled to keep runs offline and reproducible and to remove an uncontrolled external-knowledge confound plus Kaggle/SerpApi crash risk. Every **reasoning** safeguard remains on. |

Nothing in the agent's decision logic was modified. The two deviations are properties of *scope* and *reproducibility*, not of the agent's ability to write correct code.

### 3.2 Instrumentation — logging only

A monkey-patch wraps `execute_script` so that on every call it (a) archives the exact script to `attempt_NN.py`, (b) appends a structured record (exit code, SHA-256, token usage, log tail) to a JSONL ledger, then (c) delegates to the original and returns its result unchanged.

This is **required** because each regeneration overwrites the single working script file, and each manager revision spawns a fresh Operation agent whose logs replace the prior ones — so the system's own final state *undercounts* attempts. Only by wrapping execution can every attempt across both the inner 5-iteration coding loop and the outer revision loop be preserved. A transport-layer throttle enforces a minimum interval between LLM calls (timing only — it never changes what the agent decides).

### 3.3 Models (backends)

| Key | Backend | Model id | Family |
|-----|---------|----------|--------|
| `ollama` | ollama-cloud | `gpt-oss:120b-cloud` | OpenAI gpt-oss |
| `ollama2` | ollama-cloud | `gemma4:cloud` | Google Gemma |

Two distinct model families were exercised through the Ollama cloud gateway. (See [Limitations](#7-limitations) for why the originally planned free gemini/groq/openrouter backends could not sustain full-strength call volume and were excluded by the validity guard.)

### 3.4 Datasets (local CSVs, `source=user-upload` so no retrieval fires)

| Dataset | Task | Target | Stressor |
|---------|------|--------|----------|
| breast_cancer | classification | `target` | 30 numeric features, binary |
| titanic | classification | `survived` | mixed types + missing values (preprocessing stressor) |
| diabetes | regression | `target` | continuous progression score |
| california | regression | `target` | ~1500 rows, continuous |
| wine | clustering | `target` | 13 chemical features, 3 true clusters |
| iris | clustering | `target` | 4 measurements, 3 true clusters |

### 3.5 Measures and statistics

1. **First-attempt failure rate** — fraction of cells whose *first* generated script failed execution (before any repair).
2. **Per-execution failure rate** — broken executions ÷ total executions, counting every repair/revision attempt.
3. **Final-after-repair outcome** — `working` / `unrecoverable` / `incomplete`; the first-attempt-vs-final contrast shows how much the safeguards recover.
4. **Repair attempts to success** — distribution of the number of broken scripts preceding the first success.
5. **Semantic "runs-but-wrong" rate** — of cells that exit 0, the fraction whose reported metric is missing/NaN or no better than a classical floor (classification ≤ majority-class accuracy; regression R² < 0 / worse than the mean predictor; clustering ARI ≤ 0 / NMI ≈ 0).

All proportions are reported with **Wilson 95% confidence intervals** (appropriate for small n and rates near 0/1).

### 3.6 Validity guard — infra vs. code

A cell in which the LLM API itself failed or hung with **no code produced** (quota 429, size-limit 413, network) is tagged `infra_error` and **excluded** from all code-failure numerators, reported separately. This ensures provider limits are never miscounted as code brokenness. Three such cells (one each from gemini, groq, openrouter during roster probing) were excluded.

### 3.7 Scale and resumability

Each cell runs in its own subprocess under a 30-minute wall-clock watchdog that force-kills the whole process tree on timeout. A status ledger makes the grid resumable: completed cells are skipped on restart. The grid completed 36/36 cells in a single ~6.4-hour pass.

---

## 4. Results

### 4.1 Headline failure rates, overall and per model

| Model | First-attempt broken | Per-execution broken | Final unrecoverable | cells |
|-------|----------------------|----------------------|---------------------|-------|
| `gpt-oss:120b` (ollama) | 94% [74–99] (n=18) | 95% [91–97] (n=240) | 28% [12–51] (n=18) | 18 |
| `gemma4` (ollama2) | 89% [67–97] (n=18) | 80% [69–88] (n=69) | 22% [9–45] (n=18) | 18 |
| **Overall** | **92% [78–97] (n=36)** | **91% [88–94] (n=309)** | **25% [14–41] (n=36)** | 36 |

Both model families show the same qualitative picture — near-universal first-attempt breakage — so the result is not an artifact of a single backend. The two differ in *how* they fail (§4.4) and in total attempt volume: `gpt-oss` generated 240 executions (it keeps retrying within the inner loop), while `gemma4` generated only 69 (it more often either succeeded early or hit the 300 s execution timeout).

### 4.2 First-attempt vs. final — how much the safeguards recover

- First generated script broken: **92% [78–97]**
- Still unrecoverable after the repair + revision loops: **25% [14–41]**
- Final outcome counts: **working 27 · unrecoverable 5 · incomplete 4**

The safeguards are clearly valuable: they convert a 92% first-attempt failure rate into a 75% eventual-success rate. But a quarter of all cells never reach working code even after the full 3-revision budget, and recovery is expensive (§4.3).

### 4.3 Repair effort — attempts to first success

Distribution of the number of broken scripts before the first clean execution (only the 27 eventually-working cells have a success index):

| Broken attempts before success | # cells |
|---|---|
| 0 (first try worked) | 3 |
| 1 | 5 |
| 2 | 7 |
| 3 | 2 |
| 5 | 1 |
| 8 | 4 |
| 9 | 1 |
| 10 | 1 |
| 13 | 1 |
| 15 | 2 |

Attempts per cell: **min 1, max 25, mean 8.6**. Five cells exhausted the inner 25-attempt cap without ever succeeding. The long right tail (cells needing 8–15 broken scripts first) shows that "eventually works" can mean a dozen failed generations along the way.

### 4.4 Error taxonomy — why the code breaks

Categories assigned to the 282 failed executions:

| Category | Overall | `gpt-oss` | `gemma4` |
|----------|---------|-----------|----------|
| import_error | 94 | 91 | 3 |
| other_runtime_error | 45 | 44 | 1 |
| non_termination_timeout | 37 | 3 | 34 |
| file_error | 33 | 33 | 0 |
| wrong_api_usage | 30 | 27 | 3 |
| data_or_value_error | 26 | 13 | 13 |
| syntax_error | 10 | 10 | 0 |
| runtime_error | 6 | 6 | 0 |
| undefined_name | 1 | 0 | 1 |

Two sharply different failure signatures:

- **`gpt-oss`** fails overwhelmingly through **import errors** (91) — missing/hallucinated packages, wrong optional engines — plus file, API-usage, and other runtime errors. It writes a lot of code, much of it reaching for libraries that are not present or not used correctly.
- **`gemma4`** fails mostly through **non-termination / timeout** (34 of its 55 failures) — it generates code that runs past the 300 s execution limit and is killed. Its import hygiene is far better, but it hangs.

### 4.5 Semantic "runs-but-wrong"

Of the 27 cells that exited cleanly:

| | Overall | `gpt-oss` | `gemma4` |
|---|---|---|---|
| Checked (ran cleanly) | 27 | 13 | 14 |
| **Semantic failure** | **33% [19–52]** (9) | **54% [29–77]** (7) | **14% [4–40]** (2) |
| — implausible / NaN | 1 | 1 | 0 |
| — missing metric | 8 | 6 | 2 |
| — plausible | 18 | 6 | 12 |

A clean exit code is **not** the same as a correct result. A third of the "successful" runs either printed no recognizable evaluation metric at all (8 cases) or reported a metric worse than a trivial baseline (1 case: `gpt-oss` on `diabetes` rep 3 reported **R² = −3.66**, far worse than simply predicting the mean). `gpt-oss` is especially prone to finishing without ever reporting how well its model did.

> **Honesty note on the parser.** Plausibility is judged on each task's primary metric (accuracy/F1 vs. majority floor; R² vs. 0 / RMSE vs. mean-predictor; ARI/NMI vs. 0). The metric extractor is heuristic; a few `gemma4` regression cells report an identical `rmse = 64.0` that looks like a placeholder, so their verdicts lean on the R² value rather than the RMSE. This does not affect the pass/fail counts above, which key on the primary metric.

### 4.6 By task

| Task | First-attempt broken | Per-execution broken | Final unrecoverable | Outcomes |
|------|----------------------|----------------------|---------------------|----------|
| Classification | 92% [65–99] | 96% [92–98] | **50% [25–75]** | working 6 · unrecoverable 4 · incomplete 2 |
| Regression | 92% [65–99] | 89% [80–94] | 25% [9–53] | working 9 · unrecoverable 1 · incomplete 2 |
| Clustering | 92% [65–99] | 82% [71–89] | **0% [0–24]** | working 12 |

First-attempt breakage is identical across tasks (92%), but **final** success varies strongly: clustering always eventually worked, while **classification failed to converge half the time** — the hardest task here, driven by the preprocessing-heavy datasets (missing values, mixed types) and the semantic-reporting gaps on `gpt-oss`.

### 4.7 By dataset

| Dataset | Task | First-attempt broken | Final outcomes |
|---------|------|----------------------|----------------|
| breast_cancer | clf | 83% [44–97] | working 3 · unrecoverable 2 · incomplete 1 |
| titanic | clf | 100% [61–100] | working 3 · unrecoverable 2 · incomplete 1 |
| diabetes | reg | 83% [44–97] | working 4 · unrecoverable 1 · incomplete 1 |
| california | reg | 100% [61–100] | working 5 · incomplete 1 |
| wine | clustering | 83% [44–97] | working 6 |
| iris | clustering | 100% [61–100] | working 6 |

The two datasets whose first attempt **always** broke (titanic, california) are the two with the most preprocessing or scale demands; titanic (missing values + categoricals) also produced the most unrecoverable cells.

### 4.8 Cost and wall time

- **Total LLM tokens:** 4,117,016
- **Total wall-clock:** 22,908.7 s (6.36 h) across 36 cells
- **Mean per cell:** 587 s (~9.8 min)

Producing one working-or-not pipeline costs, on average, ~114k tokens and ~10 minutes of wall time — a direct consequence of the multi-plan, multi-candidate, multi-revision machinery and the repeated regeneration driven by the high failure rate.

---

## 5. Concrete failure examples

Real first-or-mid-loop errors pulled verbatim from the evidence tree, one per category:

| Category | Cell / attempt | Error (verbatim) |
|----------|----------------|------------------|
| import_error | `ollama__iris__rep1` a2 | `ImportError: Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'.` — tried to read Parquet with no engine installed, for a CSV task. |
| undefined_name | `ollama2__wine__rep1` a0 | `NameError: name 'torch' is not defined` — reached for PyTorch (unimported, unneeded) on a 178-row tabular clustering problem. |
| data_or_value_error | `ollama__iris__rep1` a1 | `ValueError: CSV columns mismatch. Expected {'sepal_width','sepal_length',...}, got {'sepal length (cm)',...}` — hallucinated the column names instead of reading the actual header. |
| wrong_api_usage | `ollama__breast_cancer__rep2` a4 | `TypeError: Object of type int64 is not JSON serializable` — the classic NumPy-to-JSON bug. |
| syntax_error | `ollama__breast_cancer__rep1` a20 | `SyntaxError: '(' was never closed` — malformed code emitted on the 21st attempt. |
| non_termination_timeout | `ollama__titanic__rep3` a22 | `Execution ... timed out after 300s` — script never terminated and was killed by the per-execution watchdog. |

These are not exotic edge cases — they are mainstream LLM-codegen failure modes: wrong/absent imports, hallucinated schema, NumPy/JSON type mismatches, and non-terminating code.

---

## 6. Discussion

**The safeguards help, but the raw generator is unreliable.** At full strength, the *first* thing AutoML-Agent writes is broken 92% of the time. The verification/revision stack pulls eventual success up to 75%, which is a genuine and measurable benefit — but it is a correction layer compensating for a base generator that rarely gets it right the first time, and it still fails outright a quarter of the time.

**"It ran" is a misleading success signal.** A third of clean exits are semantically empty or wrong. If success were defined (as a user would) by "ran *and* reported a sensible result," the true success rate drops from 75% to **50%**.

**Failure mode depends on the model.** `gpt-oss` fails loudly and often (imports, APIs, runtime), generating many attempts; `gemma4` fails quietly by hanging. A single-backend study would have mischaracterized the phenomenon — the cross-family design matters.

**Task difficulty shows up in convergence, not first-attempt rate.** Everything breaks first try at the same ~92%, but preprocessing-heavy classification is where the repair loop most often fails to converge (50% unrecoverable), while clustering always recovered.

---

## 7. Limitations

- **Roster narrowed to two backends.** The study was designed for four cross-provider free backends (gemini, groq, openrouter, Ollama). In practice the free tiers of gemini/groq/openrouter could not sustain full-strength's ~40-call-per-cell burst — they returned 429 (rate/quota) and 413 (per-request size) immediately. Their probe cells are correctly tagged `infra_error` and excluded. The final roster is two distinct model families (OpenAI gpt-oss, Google Gemma) via Ollama cloud. Generality beyond these two is not claimed.
- **No ablation (by design).** Brokenness cannot be attributed to individual safeguards; this is mitigated by the first-attempt-vs-final contrast *within* the full-strength runs.
- **`rap=False`.** Retrieval-augmented planning is the one safeguard not exercised, disabled for reproducibility and confound/cost/crash control.
- **Single framework.** Findings are scoped to AutoML-Agent.
- **Modest N (36 code cells).** All point estimates are reported with Wilson 95% CIs; the CIs are wide for per-task/per-dataset cuts and the claims are kept loose accordingly.
- **Heuristic semantic parser.** Metric extraction is regex-based; plausibility keys on the primary metric per task (see §4.5 note).

---

## 8. Reproducibility and evidence tree

Everything needed to re-run or audit the study is committed:

| Artifact | Path |
|----------|------|
| Single-source config (knobs, roster, grid) | `experiment/config.py` |
| Dataset builders + 6 CSVs + provenance | `experiment/build_datasets.py`, `experiment/datasets/` (incl. `manifest.json`) |
| Classical baselines for the semantic check | `experiment/baselines.py`, `experiment/baselines.json` |
| Logging-only instrumentation + throttle | `experiment/instrumentation.py` |
| Grid orchestrator (subprocess, watchdog, resume, infra/code split) | `experiment/run_grid.py` |
| Analysis + report generators | `experiment/analyze.py`, `experiment/report.py` |
| **Per-attempt ledger (309 records)** | `experiment/ledger.jsonl` |
| **Archived scripts + logs (every attempt)** | `experiment/runs/<model>/<dataset>/<rep>/attempt_NN.py` |
| Machine-readable results | `experiment/results/summary.json` |
| Human-readable tables / report / figures | `experiment/results/tables.md`, `report.md`, `figures/*.png` |

To reproduce: `python -m experiment.build_datasets` → `python -m experiment.baselines` → `AMLA_EXP_MODELS="ollama,ollama2" python -m experiment.run_grid` → `python -m experiment.analyze` → `python -m experiment.report`. Exact model ids, the provider snapshot date, the AutoML-Agent commit, dataset seeds, and prompt-template SHA-256 hashes are all recorded in `summary.json`. Free-model outputs are non-deterministic and endpoints drift — hence the snapshot date and the archived artifacts.

---

## 9. Conclusion

Run at its **full, as-designed strength**, AutoML-Agent's first generated script is broken **92%** of the time; **91%** of all executions fail; **25%** of tasks never yield working code even after the complete repair-and-revision budget; and of the runs that do exit cleanly, **a third** report a missing or implausible result. Taken together, only **half** of the runs produced code that both ran and was right. The system's safeguards materially improve on the raw generator, but they are compensating for a generator that is broken far more often than not — confirming the thesis that **even at full strength, this LLM-supported AutoML system produces broken code at a significant rate, across multiple LLM backends and tabular tasks.**

---

## 10. Appendix: per-cell outcomes

Final state per cell (B = broken first attempt, ✓/✗ = eventually working/unrecoverable, `attempts` = scripts archived). Full detail in `experiment/runs/cell_status.json` and `experiment/ledger.jsonl`.

| Cell | Task | First attempt | Final | Attempts |
|------|------|---------------|-------|----------|
| ollama · breast_cancer · rep1 | clf | broken | ✗ unrecoverable | 25 |
| ollama · breast_cancer · rep2 | clf | broken | ✗ unrecoverable | 25 |
| ollama · breast_cancer · rep3 | clf | broken | ✓ working (no metric) | 16 |
| ollama · titanic · rep1 | clf | broken | ✓ working | 14 |
| ollama · titanic · rep2 | clf | broken | ✗ unrecoverable | 25 |
| ollama · titanic · rep3 | clf | broken | ✗ unrecoverable | 25 |
| ollama · diabetes · rep1 | reg | broken | ✗ unrecoverable | 25 |
| ollama · diabetes · rep2 | reg | broken | ✓ working | 9 |
| ollama · diabetes · rep3 | reg | ok | ✓ working (R²=−3.66, implausible) | 1 |
| ollama · california · rep1 | reg | broken | ✓ working | 9 |
| ollama · california · rep2 | reg | broken | ✓ working | 4 |
| ollama · california · rep3 | reg | broken | ✓ working | 9 |
| ollama · wine · rep1 | clustering | broken | ✓ working (no metric) | 9 |
| ollama · wine · rep2 | clustering | broken | ✓ working (no metric) | 16 |
| ollama · wine · rep3 | clustering | broken | ✓ working (no metric) | 11 |
| ollama · iris · rep1 | clustering | broken | ✓ working (no metric) | 4 |
| ollama · iris · rep2 | clustering | broken | ✓ working (no metric) | 10 |
| ollama · iris · rep3 | clustering | broken | ✓ working | 3 |
| ollama2 · breast_cancer · rep1 | clf | broken | ✓ working | 3 |
| ollama2 · breast_cancer · rep2 | clf | broken | ✗ incomplete (timeout) | 7 |
| ollama2 · breast_cancer · rep3 | clf | ok | ✓ working | 2 |
| ollama2 · titanic · rep1 | clf | broken | ✗ incomplete (timeout) | 15 |
| ollama2 · titanic · rep2 | clf | broken | ✓ working (no metric) | 3 |
| ollama2 · titanic · rep3 | clf | broken | ✓ working (no metric) | 2 |
| ollama2 · diabetes · rep1 | reg | broken | ✗ incomplete (timeout) | 6 |
| ollama2 · diabetes · rep2 | reg | broken | ✓ working | 2 |
| ollama2 · diabetes · rep3 | reg | broken | ✓ working | 3 |
| ollama2 · california · rep1 | reg | broken | ✓ working | 6 |
| ollama2 · california · rep2 | reg | broken | ✗ incomplete (timeout) | 5 |
| ollama2 · california · rep3 | reg | ok | ✓ working | 2 |
| ollama2 · wine · rep1 | clustering | broken | ✓ working | 3 |
| ollama2 · wine · rep2 | clustering | ok | ✓ working | 1 |
| ollama2 · wine · rep3 | clustering | broken | ✓ working | 2 |
| ollama2 · iris · rep1 | clustering | broken | ✓ working | 3 |
| ollama2 · iris · rep2 | clustering | broken | ✓ working | 2 |
| ollama2 · iris · rep3 | clustering | broken | ✓ working | 3 |

*Report generated from `experiment/results/summary.json`, `experiment/ledger.jsonl`, and `experiment/runs/cell_status.json`.*
