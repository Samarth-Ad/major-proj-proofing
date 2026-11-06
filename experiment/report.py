"""Assemble the paper-ready Markdown report from the analysis outputs.

Supersedes the pilot's broken_code_report.md: a Methods-ready writeup covering the
config, models, datasets, metrics, stats, headline results, taxonomy, the semantic
check, limitations, and a pointer to the archived-script evidence tree.

Run (after analyze):  python -m experiment.report
"""
from __future__ import annotations

import json

from experiment import config
from experiment.analyze import _fmt

THESIS = (
    "Even run at its full as-designed configuration, the LLM-supported AutoML system "
    "AutoML-Agent produces broken code at a significant rate — not always, but often — "
    "across multiple LLM backends and tabular tasks."
)


def _load(path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def build_report() -> str:
    summary = _load(config.RESULTS_DIR / "summary.json", {})
    manifest = _load(config.MANIFEST_PATH, [])
    baselines = _load(config.BASELINES_PATH, {})
    meta = summary.get("meta", {})
    ov = summary.get("overall", {})

    out: list[str] = []
    w = out.append

    w("# Generated-Code Reliability of AutoML-Agent at Full Strength")
    w("")
    w(f"**Thesis.** *{THESIS}*")
    w("")
    w(f"- AutoML-Agent commit: `{meta.get('git_commit', 'n/a')}`  ·  provider snapshot: "
      f"`{meta.get('snapshot_date', 'n/a')}`")
    w(f"- Grid: {summary.get('n_cells_recorded', 0)}/{summary.get('n_cells_total', 0)} cells recorded "
      f"({len(meta.get('models', []))} models × {len(config.DATASETS)} datasets × {meta.get('reps', config.REPS)} reps).")
    w("")

    # --- Methods -------------------------------------------------------------
    w("## Methods")
    w("")
    w("### Configuration (single, full-strength)")
    w("")
    w("AutoML-Agent is run **unmodified** at its as-designed settings: all reasoning, "
      "verification, and revision safeguards are ON. Only two knobs deviate, both documented:")
    w("")
    w("| Knob | Value | Rationale |")
    w("|------|-------|-----------|")
    w("| verification | True | input gating + pre-exec Pass/Fail judge ON |")
    w("| decomp | True | per-agent plan decomposition ON |")
    w("| n_plans | 3 | multi-plan exploration ON |")
    w("| n_candidates | 3 | multi-model selection ON |")
    w("| n_revise | 3 | repair + plan-revision loops ON |")
    w("| full_pipeline | **False** | *scope*, not a safeguard: True builds a Gradio server killed by the 300s exec timeout — a pure artifact, not code-brokenness |")
    w("| rap | **False** | keeps runs offline/reproducible; removes Kaggle/SerpApi crash risk and an uncontrolled external-knowledge confound. The single explicit limitation; every *reasoning* safeguard stays on |")
    w("")
    w("### Models (backends)")
    w("")
    w("| Key | Backend | Model id |")
    w("|-----|---------|----------|")
    for model in meta.get("models", []):
        w(f"| {model.get('key')} | {model.get('backend')} | `{model.get('model_id')}` |")
    w("")
    w("### Datasets (local CSVs, `source=user-upload`, so no retrieval fires)")
    w("")
    w("| Dataset | Task | Rows × Cols | Missing | Source |")
    w("|---------|------|-------------|---------|--------|")
    for entry in manifest:
        w(f"| {entry['id']} | {entry['task'].replace('tabular_', '')} | "
          f"{entry['n_rows']}×{entry['n_columns']} | {entry['has_missing']} | {entry['source_realized']} |")
    w("")
    w("### Metrics")
    w("")
    w("All measured from the system's own execution outputs; proportions with Wilson 95% CIs.")
    w("")
    w("1. **First-attempt failure rate** — fraction of cells whose *first* generated script "
      "failed execution (measured before any repair).")
    w("2. **Per-execution failure rate** — broken executions ÷ total executions (every repair/"
      "revision attempt included).")
    w("3. **Final-after-repair outcome** — working vs unrecoverable/incomplete; the first-attempt-"
      "vs-final contrast shows how much the safeguards recover.")
    w("4. **Repair attempts to success** — distribution of broken scripts preceding success.")
    w("5. **Semantic 'runs-but-wrong' rate** — of cells that exit 0, the fraction whose reported "
      "metric is missing/NaN or no better than a classical floor (clf ≤ majority-class accuracy; "
      "reg R² < 0 / worse than mean-predictor; clustering ARI ≤ 0 / NMI ≈ 0).")
    w("")
    w("**Validity guard.** A cell where the LLM API itself failed or hung with no code produced is "
      "tagged `infra_error` and **excluded** from the code-failure numerators (reported separately), "
      "so provider limits are never miscounted as code brokenness.")
    w("")
    w("### Instrumentation & reproducibility")
    w("")
    w("A logging-only wrapper around `execute_script` archives **every** generated script "
      "(`experiment/runs/<model>/<dataset>/<rep>/attempt_k.py`) and appends each execution to a "
      "JSONL ledger; the agent's decision logic is untouched. This is required because each "
      "regeneration overwrites the single script file and each manager revision spawns a fresh "
      "operation agent, so the final result undercounts attempts. Each cell runs in its own "
      "subprocess under a wall-clock watchdog. Exact model ids + snapshot date are pinned; the "
      "AutoML-Agent commit, dataset seeds, and prompt-template hashes are recorded. Free-model "
      "outputs are non-deterministic and endpoints may drift — hence the snapshot date and the "
      "archived artifacts.")
    w("")

    # --- Results -------------------------------------------------------------
    w("## Results")
    w("")
    if ov:
        w("### Headline")
        w("")
        w(f"- **First generated script broken: {_fmt(ov['first_attempt_failure'])}.**")
        w(f"- **Per-execution broken: {_fmt(ov['per_execution_failure'])}.**")
        w(f"- Still unrecoverable after the repair + revision loops: {_fmt(ov['final_unrecoverable'])}.")
        sem = summary.get("semantic", {}).get("overall", {})
        if sem.get("n_checked"):
            w(f"- Of the scripts that ran cleanly, semantic failure (missing/NaN/implausible "
              f"metric): {_fmt(sem['semantic_failure'])} "
              f"(implausible/NaN={sem['implausible_or_nan']}, missing={sem['missing_metric']}, "
              f"plausible={sem['plausible']}).")
        w(f"- Infra/provider-error cells excluded from the above: {summary.get('n_infra_cells', 0)}.")
        w("")
        w("### Per-model and taxonomy")
        w("")
        w("See `results/tables.md` for the full per-model table (first-attempt, per-execution, "
          "final-unrecoverable with CIs) and the error-taxonomy breakdown; see `results/figures/` "
          "for the failure-rate bars, taxonomy stacked bars, and repair-attempt histogram.")
        w("")
        w(f"- Final outcome counts (overall): `{ov.get('final_outcome_counts', {})}`")
        w(f"- Repair attempts to first success: `{ov.get('repair_attempts_to_success', {})}`")
        w(f"- Error taxonomy (overall): `{summary.get('taxonomy', {}).get('overall', {})}`")
        w("")
    else:
        w("_No results yet. Run `python -m experiment.run_grid` then `python -m experiment.analyze`._")
        w("")

    # --- Limitations ---------------------------------------------------------
    w("## Limitations")
    w("")
    w("- **No ablation** (by design): brokenness cannot be attributed to specific safeguards; "
      "mitigated by the first-attempt-vs-final contrast within the full-strength runs.")
    w("- **`rap=False`**: retrieval-augmented planning is the one safeguard not exercised; "
      "documented and justified (confound/cost/crash control).")
    w("- **Free-tier flakiness**: quotas/TPM/non-determinism; mitigated by backoff, resume, the "
      "`infra_error` separation, and multiple reps.")
    w("- **Single framework**: generality is limited to AutoML-Agent (thesis scoped accordingly).")
    w(f"- **Light N ({summary.get('n_cells_total', 72)} cells)**: CIs are reported; point claims are kept loose.")
    w("")
    w("## Evidence tree")
    w("")
    w("Every generated script + its execution log is archived under `experiment/runs/`; the "
      "complete per-attempt ledger is `experiment/ledger.jsonl`; dataset provenance is "
      "`experiment/datasets/manifest.json`; classical baselines are `experiment/baselines.json`.")
    w("")
    return "\n".join(out)


def main() -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    report = build_report()
    path = config.RESULTS_DIR / "report.md"
    path.write_text(report, encoding="utf-8")
    print(f"Wrote report -> {path}")


if __name__ == "__main__":
    main()
