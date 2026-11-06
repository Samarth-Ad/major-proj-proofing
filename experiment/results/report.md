# Generated-Code Reliability of AutoML-Agent at Full Strength

**Thesis.** *Even run at its full as-designed configuration, the LLM-supported AutoML system AutoML-Agent produces broken code at a significant rate — not always, but often — across multiple LLM backends and tabular tasks.*

- AutoML-Agent commit: `9e0d842959a09d96c00768c14d73bd9df6d39023`  ·  provider snapshot: `2026-10-31`
- Grid: 39/72 cells recorded (2 models × 6 datasets × 3 reps).

## Methods

### Configuration (single, full-strength)

AutoML-Agent is run **unmodified** at its as-designed settings: all reasoning, verification, and revision safeguards are ON. Only two knobs deviate, both documented:

| Knob | Value | Rationale |
|------|-------|-----------|
| verification | True | input gating + pre-exec Pass/Fail judge ON |
| decomp | True | per-agent plan decomposition ON |
| n_plans | 3 | multi-plan exploration ON |
| n_candidates | 3 | multi-model selection ON |
| n_revise | 3 | repair + plan-revision loops ON |
| full_pipeline | **False** | *scope*, not a safeguard: True builds a Gradio server killed by the 300s exec timeout — a pure artifact, not code-brokenness |
| rap | **False** | keeps runs offline/reproducible; removes Kaggle/SerpApi crash risk and an uncontrolled external-knowledge confound. The single explicit limitation; every *reasoning* safeguard stays on |

### Models (backends)

| Key | Backend | Model id |
|-----|---------|----------|
| ollama | ollama | `gpt-oss:120b-cloud` |
| ollama2 | ollama | `gemma4:cloud` |

### Datasets (local CSVs, `source=user-upload`, so no retrieval fires)

| Dataset | Task | Rows × Cols | Missing | Source |
|---------|------|-------------|---------|--------|
| breast_cancer | classification | 569×31 | False | sklearn |
| titanic | classification | 1309×8 | True | openml |
| diabetes | regression | 442×11 | False | sklearn |
| california | regression | 1500×9 | False | sklearn-fetch |
| wine | clustering | 178×14 | False | sklearn |
| iris | clustering | 150×5 | False | sklearn |

### Metrics

All measured from the system's own execution outputs; proportions with Wilson 95% CIs.

1. **First-attempt failure rate** — fraction of cells whose *first* generated script failed execution (measured before any repair).
2. **Per-execution failure rate** — broken executions ÷ total executions (every repair/revision attempt included).
3. **Final-after-repair outcome** — working vs unrecoverable/incomplete; the first-attempt-vs-final contrast shows how much the safeguards recover.
4. **Repair attempts to success** — distribution of broken scripts preceding success.
5. **Semantic 'runs-but-wrong' rate** — of cells that exit 0, the fraction whose reported metric is missing/NaN or no better than a classical floor (clf ≤ majority-class accuracy; reg R² < 0 / worse than mean-predictor; clustering ARI ≤ 0 / NMI ≈ 0).

**Validity guard.** A cell where the LLM API itself failed or hung with no code produced is tagged `infra_error` and **excluded** from the code-failure numerators (reported separately), so provider limits are never miscounted as code brokenness.

### Instrumentation & reproducibility

A logging-only wrapper around `execute_script` archives **every** generated script (`experiment/runs/<model>/<dataset>/<rep>/attempt_k.py`) and appends each execution to a JSONL ledger; the agent's decision logic is untouched. This is required because each regeneration overwrites the single script file and each manager revision spawns a fresh operation agent, so the final result undercounts attempts. Each cell runs in its own subprocess under a wall-clock watchdog. Exact model ids + snapshot date are pinned; the AutoML-Agent commit, dataset seeds, and prompt-template hashes are recorded. Free-model outputs are non-deterministic and endpoints may drift — hence the snapshot date and the archived artifacts.

## Results

### Headline

- **First generated script broken: 92% [78-97] (n=36).**
- **Per-execution broken: 91% [88-94] (n=309).**
- Still unrecoverable after the repair + revision loops: 25% [14-41] (n=36).
- Of the scripts that ran cleanly, semantic failure (missing/NaN/implausible metric): 33% [19-52] (n=27) (implausible/NaN=1, missing=8, plausible=18).
- Infra/provider-error cells excluded from the above: 3.

### Per-model and taxonomy

See `results/tables.md` for the full per-model table (first-attempt, per-execution, final-unrecoverable with CIs) and the error-taxonomy breakdown; see `results/figures/` for the failure-rate bars, taxonomy stacked bars, and repair-attempt histogram.

- Final outcome counts (overall): `{'working': 27, 'unrecoverable': 5, 'incomplete': 4}`
- Repair attempts to first success: `{'0': 3, '1': 5, '2': 7, '3': 2, '5': 1, '8': 4, '9': 1, '10': 1, '13': 1, '15': 2}`
- Error taxonomy (overall): `{'import_error': 94, 'other_runtime_error': 45, 'non_termination_timeout': 37, 'file_error': 33, 'wrong_api_usage': 30, 'data_or_value_error': 26, 'syntax_error': 10, 'runtime_error': 6, 'undefined_name': 1}`

## Limitations

- **No ablation** (by design): brokenness cannot be attributed to specific safeguards; mitigated by the first-attempt-vs-final contrast within the full-strength runs.
- **`rap=False`**: retrieval-augmented planning is the one safeguard not exercised; documented and justified (confound/cost/crash control).
- **Free-tier flakiness**: quotas/TPM/non-determinism; mitigated by backoff, resume, the `infra_error` separation, and multiple reps.
- **Single framework**: generality is limited to AutoML-Agent (thesis scoped accordingly).
- **Light N (72 cells)**: CIs are reported; point claims are kept loose.

## Evidence tree

Every generated script + its execution log is archived under `experiment/runs/`; the complete per-attempt ledger is `experiment/ledger.jsonl`; dataset provenance is `experiment/datasets/manifest.json`; classical baselines are `experiment/baselines.json`.
