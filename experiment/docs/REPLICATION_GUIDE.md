# Replication Guide — Running This Study on Another LLM-Powered AutoML System

This guide explains how to take the methodology built here for **AutoML-Agent** and
apply it to a *different* LLM-powered AutoML / code-generating ML system (e.g. AIDE,
OpenHands-for-ML, Data-Interpreter / MetaGPT-DI, AutoKaggle, H2O/AutoGen-style agents,
or your own).

The study's **design is system-agnostic**. Only a handful of concrete "binding points"
change per system. This guide tells you exactly which parts you reuse verbatim and
which parts you rewire.

---

## 1. Is the target system a candidate?

The methodology applies to any system that:

- takes a natural-language ML task + a dataset, and
- **generates code** with an LLM, and
- **executes** that code, and (ideally)
- has some **repair / revision loop** when code fails.

If the system generates code but you can't intercept its execution step, you can still
measure first/final outcomes from its logs — but you lose the per-attempt ledger, which
is the most valuable part. Prefer systems where the execution path is reachable.

---

## 2. The mental model: 7 binding points

Everything system-specific reduces to finding these 7 things in the target codebase.
Fill in the right-hand column for your system; that *is* the port.

| # | Binding point | In AutoML-Agent | In YOUR system |
|---|---------------|-----------------|----------------|
| 1 | **Programmatic entry point** — how to launch one run from Python | `AgentManager(...).initiate_chat(prompt)` | `________` |
| 2 | **LLM selection** — how to pin which model writes the code | env `AMLA_LLM` + model-id env var, read by `configs.py` at import | `________` |
| 3 | **The execute function to wrap** — the single call that runs generated code | `operation_agent.execute_script` | `________` |
| 4 | **Task/requirements format** — the structured input | `user_requirements` JSON | `________` |
| 5 | **Dataset routing / skip retrieval** — point it at a local file, no internet | `source="user-upload"`, `model=[]` | `________` |
| 6 | **Config knobs ("full strength")** — the settings that turn on all safeguards | `verification/decomp/n_plans/n_candidates/n_revise=…` | `________` |
| 7 | **Final result / state / token usage** — where the run stores its outcome | `manager.implementation_result`, `.state`, `.money`, `.timer` | `________` |

Find these 7, and the rest of the harness drops in.

---

## 3. What you reuse almost verbatim

These modules are generic and change little or not at all:

| Module | Reuse | Notes |
|--------|-------|-------|
| `build_datasets.py` | ✅ as-is | Same tabular datasets work for any system. Swap in your own if needed. |
| `baselines.py` | ✅ as-is | Classical floors are task-defined, not system-defined. |
| `analyze.py` | ✅ as-is | Consumes the ledger + status format, which you keep identical. |
| `report.py` | ✅ as-is | Reads `summary.json`. |
| `instrumentation.py` | 🔧 one line | Only the **wrap target** (binding point #3) changes. |
| `run_grid.py` | 🔧 small | Only `run_cell()` (binding points #1, #2, #4, #6, #7) changes; the orchestrator/watchdog/resume/infra logic stays. |
| `config.py` | 🔧 rewired | Your models, your knob names, your paths. |

The key insight: **the data contract is fixed.** As long as your instrumentation writes
the same `ledger.jsonl` records and your orchestrator writes the same `cell_status.json`
fields, `analyze.py` and `report.py` work unchanged.

---

## 4. The fixed data contract (keep these identical)

### `ledger.jsonl` — one JSON object per execution attempt

```json
{"cell_id": "...", "model_key": "...", "backend": "...", "model_id": "...",
 "dataset": "...", "task": "...", "rep": 1,
 "attempt": 0, "rcode": 0, "ok": true,
 "code_chars": 12345, "code_sha256": "...", "log_tail": "...(last ~6000 chars)..."}
```

- `attempt` must increase from 0 within a cell.
- `rcode == 0` means the script ran; non-zero (or -1 for a kill/timeout) means broken.
- `log_tail` must include the error message / final metric print — the analyzer reads it
  for both the taxonomy and the semantic check.

### `cell_status.json` — one object per cell, keyed by `cell_id`

Minimum fields `analyze.py` needs: `model_key`, `dataset`, `task`, `rep`,
`termination` (`completed` / `timeout_after_attempts` / `subprocess_error` /
`pipeline_exception`), `final_rcode`, `infra_error` (bool), `money` (token usage dict),
`wall_sec`.

If you produce these two files, you inherit the entire analysis + reporting layer.

---

## 5. Step-by-step port

### Step A — Rebind `config.py`
- Replace the model roster with your system's backends/model ids and the env/API your
  system uses to select a model.
- Replace the knob dict (`FULL_CONFIG`) with your system's equivalent of "all safeguards
  on." Document any deviation and *why* (as we did for `full_pipeline`/`rap`).
- Keep the datasets, task→metric map, and paths.

### Step B — Rebind the wrap target in `instrumentation.py`
Change the two patched names to your system's execute function. The wrapper body (archive
+ ledger append) stays the same. If your system runs code via `subprocess` with no single
Python function, wrap that call site instead, or wrap the lowest-level "run this file"
helper.

> ⚠️ Patch the **binding the caller actually uses.** In AutoML-Agent the caller imports
> `from operation_agent import execute_script`, so patching only
> `operation_agent.execution.execute_script` would miss it. Patch both the module *and*
> any re-exported name.

### Step C — Rebind `run_cell()` in `run_grid.py`
- Set your LLM-selection env/vars **before importing** the target system (many systems
  read config at import time).
- Build and launch your system the way a user would (binding point #1).
- After it finishes, read its final state/exit/tokens (binding point #7) into the
  `cell_result.json` fields.
- Leave the orchestrator (`orchestrate`, `_launch_cell`, `_kill_tree`, `_classify_cell`,
  resume, backoff) untouched.

### Step D — Point it at local data, skip retrieval (binding point #5)
Make sure your task spec routes to the local CSV and disables any web/dataset search, so
you measure code generation, not lookups. If the system can't skip retrieval, document it
as a confound.

### Step E — Smoke test one cell, then scale
```bash
python -m experiment.run_grid --list                     # confirm the grid expands
python -m experiment.run_grid --cells <one_cell_id>      # run exactly one cell
# inspect runs/<...>/attempt_00.py and ledger.jsonl — did the tap capture it?
python -m experiment.run_grid                            # full grid (resumable)
python -m experiment.analyze && python -m experiment.report
```
Verify the tap first: if `ledger.jsonl` stays empty after a cell, your wrap target is
wrong (Step B).

---

## 6. Validity rules to preserve (don't skip these)

These are what make the result defensible, independent of the system:

1. **Archive every attempt.** If the system overwrites its script file or discards logs
   on retry, the end-state undercounts failures. The tap is non-negotiable.
2. **Separate infra from code.** Tag and exclude cells where the LLM API failed with no
   code produced (rate limit / quota / size / network). Otherwise a flaky provider
   inflates your "broken code" number.
3. **One fixed full-strength config.** Run the system at its *strongest* setting so the
   failure rate is a floor, not a strawman. No cherry-picking.
4. **Repetitions + Wilson CIs.** LLMs are non-deterministic; run ≥3 reps and report
   confidence intervals, not bare percentages.
5. **Pin everything.** Model ids, a snapshot date, the system's git commit, dataset seeds,
   and prompt-template hashes — all recorded in `run_meta.json`. Free endpoints drift.
6. **"Ran" ≠ "right".** Keep the semantic check: a clean exit with a missing/NaN/implausible
   metric is still a failure.
7. **Watchdog every cell.** LLM code hangs; a per-cell tree-kill timer prevents one run
   from stalling the whole grid.

---

## 7. Common pitfalls

| Pitfall | Fix |
|---------|-----|
| Ledger stays empty | Wrong wrap target (Step B); patch the name the caller imports. |
| Model never changes | The system read its config at import *before* you set the env var; set env first, import later (deferred import). |
| "Broken code" rate looks absurdly high | You're counting provider errors as code failures; enforce the infra guard. |
| Grid stalls for hours | No watchdog, or the kill doesn't reach child processes; use a whole-tree kill. |
| Numbers change every run and you can't explain it | Non-determinism; that's expected — report CIs and archive artifacts, pin a snapshot date. |
| Can't reproduce a result | You didn't record the model id / commit / seed / template hash; always write `run_meta.json`. |
| Clean exits with no metric | Expected and interesting — the semantic check catches these; don't treat exit-0 as success. |

---

## 8. Minimal checklist

- [ ] Found all 7 binding points in the target system.
- [ ] `config.py` rewired (models, knobs, paths).
- [ ] `instrumentation.py` wrap target points at the real execute call.
- [ ] `run_cell()` sets the LLM env **before** import and launches the system.
- [ ] Local-data routing confirmed; retrieval disabled or documented.
- [ ] One-cell smoke test shows a non-empty `ledger.jsonl` + archived script.
- [ ] Infra-vs-code separation verified on a deliberately rate-limited run.
- [ ] Full grid resumable (stop/restart loses nothing).
- [ ] `analyze.py` + `report.py` produce `summary.json` and the report unchanged.
- [ ] `run_meta.json` pins model ids, commit, seeds, template hashes, snapshot date.

If every box is ticked, you have a faithful, comparable replication on the new system —
and because the data contract is identical, you can put both systems' `summary.json`
side by side.
