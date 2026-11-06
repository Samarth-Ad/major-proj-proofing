# AutoML-Agent — Generated Code Reliability (full-strength config)

- Snapshot date: `2026-10-31` · AutoML-Agent commit: `9e0d842959`
- Cells recorded: 39/72 (infra/provider-error cells excluded from code rates: 3)
- Config: `{"verification": true, "decomp": true, "n_plans": 3, "n_candidates": 3, "n_revise": 3, "interactive": false, "full_pipeline": false, "rap": false}`

## Primary failure rates (per model), Wilson 95% CI

| Model | First-attempt broken | Per-execution broken | Final unrecoverable | cells |
|-------|----------------------|----------------------|---------------------|-------|
| gemini | n/a | n/a | n/a | 0 |
| groq | n/a | n/a | n/a | 0 |
| openrouter | n/a | n/a | n/a | 0 |
| ollama | 94% [74-99] (n=18) | 95% [91-97] (n=240) | 28% [12-51] (n=18) | 18 |
| ollama2 | 89% [67-97] (n=18) | 80% [69-88] (n=69) | 22% [9-45] (n=18) | 18 |
| **overall** | 92% [78-97] (n=36) | 91% [88-94] (n=309) | 25% [14-41] (n=36) | 36 |

## First-attempt vs final (how much the safeguards recover)

- First generated script broken: **92% [78-97] (n=36)**
- Still unrecoverable after the repair + revision loops: **25% [14-41] (n=36)**
- Final outcome counts: `{'working': 27, 'unrecoverable': 5, 'incomplete': 4}`
- Repair attempts to first success (index -> #cells): `{'0': 3, '1': 5, '2': 7, '3': 2, '5': 1, '8': 4, '9': 1, '10': 1, '13': 1, '15': 2}`

## Error taxonomy (failed execution attempts)

| Category | Count |
|----------|-------|
| import_error | 94 |
| other_runtime_error | 45 |
| non_termination_timeout | 37 |
| file_error | 33 |
| wrong_api_usage | 30 |
| data_or_value_error | 26 |
| syntax_error | 10 |
| runtime_error | 6 |
| undefined_name | 1 |

## Semantic 'runs-but-wrong' (of cells that exit 0)

- Checked (ran cleanly): 27
- Semantic failure (missing/NaN/implausible metric): **33% [19-52] (n=27)**
- Breakdown: implausible/NaN=1 · missing metric=8 · plausible=18

## Cost & wall time

- Total LLM tokens: 4,117,016
- Mean wall time/cell: 587.4s