"""Compute the study's metrics from the harness outputs.

Reads the attempt ledger (every execution), the per-cell status (outcomes +
infra/code separation), and the classical baselines, then writes:
  results/summary.json  -- all metrics, per model and overall, with Wilson 95% CIs
  results/tables.md     -- paper-ready Markdown tables
  results/figures/*.png -- failure-rate bars, error-taxonomy bars, repair histogram

Validity guard: cells flagged infra_error (the LLM API itself failed / hung with
no code produced) are EXCLUDED from code-failure numerators and reported separately.

Run:  python -m experiment.analyze
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict

from experiment import config
from experiment.run_grid import STATUS_PATH, RUN_META_PATH

Z = 1.96  # 95%

# --- Error taxonomy (ported from run_batch.py:111-143) -----------------------
_EXC_CATEGORY = {
    "ModuleNotFoundError": "import_error",
    "ImportError": "import_error",
    "SyntaxError": "syntax_error",
    "IndentationError": "syntax_error",
    "TabError": "syntax_error",
    "NameError": "undefined_name",
    "AttributeError": "wrong_api_usage",
    "TypeError": "wrong_api_usage",
    "ValueError": "data_or_value_error",
    "KeyError": "data_or_value_error",
    "IndexError": "data_or_value_error",
    "FileNotFoundError": "file_error",
    "RuntimeError": "runtime_error",
    "RecursionError": "recursion_error",
    "AssertionError": "assertion_self_test",
}


def classify_error(log: str) -> dict:
    if not log or not log.strip():
        return {"exception": "EmptyOutput", "category": "empty_output", "last_line": "", "at": ""}
    if "timed out" in log.lower():
        return {"exception": "Timeout", "category": "non_termination_timeout",
                "last_line": log.strip().splitlines()[-1][:300], "at": ""}
    exc_lines = re.findall(r"^([A-Za-z_][\w.]*(?:Error|Exception|Warning|Interrupt)):?.*$",
                           log, re.MULTILINE)
    last = exc_lines[-1] if exc_lines else log.strip().splitlines()[-1]
    match = re.match(r"^([A-Za-z_][\w.]*)", last)
    short = (match.group(1).split(".")[-1]) if match else "Unknown"
    category = _EXC_CATEGORY.get(short, "other_runtime_error")
    locs = re.findall(r'File "([^"]+)", line (\d+)', log)
    at = f"{os.path.basename(locs[-1][0])}:{locs[-1][1]}" if locs else ""
    return {"exception": short, "category": category, "last_line": last.strip()[:300], "at": at}


# --- Stats -------------------------------------------------------------------
def wilson(k: int, n: int) -> dict:
    if n == 0:
        return {"k": k, "n": n, "rate": None, "ci_low": None, "ci_high": None}
    p = k / n
    denom = 1 + Z * Z / n
    center = (p + Z * Z / (2 * n)) / denom
    margin = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / denom
    return {"k": k, "n": n, "rate": round(p, 4),
            "ci_low": round(max(0.0, center - margin), 4),
            "ci_high": round(min(1.0, center + margin), 4)}


# --- Loaders -----------------------------------------------------------------
def load_ledger() -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    if not config.LEDGER_PATH.exists():
        return groups
    with open(config.LEDGER_PATH, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            groups[record["cell_id"]].append(record)
    for records in groups.values():
        records.sort(key=lambda r: r.get("attempt", 0))
    return groups


def _sum_tokens(obj) -> int:
    total = 0
    if isinstance(obj, dict):
        if "total_tokens" in obj and isinstance(obj["total_tokens"], (int, float)):
            total += int(obj["total_tokens"])
        for value in obj.values():
            if isinstance(value, (dict, list)):
                total += _sum_tokens(value)
    elif isinstance(obj, list):
        for value in obj:
            total += _sum_tokens(value)
    return total


# --- Semantic 'runs-but-wrong' check -----------------------------------------
_NUM = r"(-?\d+(?:\.\d+)?)"
_METRIC_PATTERNS = {
    "accuracy": rf"(?:accuracy|\bacc\b)\b[^\d\-\n]{{0,15}}{_NUM}",
    "f1": rf"\bf1\b[^\d\-\n]{{0,15}}{_NUM}",
    "r2": rf"(?:r2|r\^2|r-squared|r_squared)\b[^\d\-\n]{{0,15}}{_NUM}",
    "rmse": rf"\brmse\b[^\d\-\n]{{0,15}}{_NUM}",
    "rmsle": rf"\brmsle\b[^\d\-\n]{{0,15}}{_NUM}",
    "ari": rf"(?:ari|adjusted[ _]rand)\b[^\d\-\n]{{0,15}}{_NUM}",
    "nmi": rf"\bnmi\b[^\d\-\n]{{0,15}}{_NUM}",
    "ri": rf"(?:\bri\b|rand[ _]index)\b[^\d\-\n]{{0,15}}{_NUM}",
}


def _extract_metrics(log: str) -> dict[str, float]:
    low = (log or "").lower()
    found: dict[str, float] = {}
    for name, pattern in _METRIC_PATTERNS.items():
        values = [float(v) for v in re.findall(pattern, low)]
        if values:
            # accuracy/f1/ari/nmi/ri: the best reported; rmse/rmsle: the smallest
            found[name] = max(values) if name in ("accuracy", "f1", "ari", "nmi", "ri") else min(values)
    found["_has_nan"] = bool(re.search(r"(accuracy|acc|f1|r2|rmse|rmsle|ari|nmi|ri)[^\n]{0,15}(nan|inf)", low))
    return found


def semantic_check(task: str, dataset: str, log: str, baselines: dict) -> dict:
    base = baselines.get(dataset, {})
    metrics = _extract_metrics(log)
    status, reason = "no_metric_parsed", "no recognizable metric in output"
    if metrics.get("_has_nan"):
        return {"status": "runs_but_wrong", "reason": "NaN/inf metric", "parsed": metrics}

    if task == "tabular_classification":
        floor = base.get("majority_class_accuracy")
        if "accuracy" in metrics and floor is not None:
            ok = metrics["accuracy"] > floor
            status = "plausible" if ok else "runs_but_wrong"
            reason = f"accuracy {metrics['accuracy']} vs majority floor {floor}"
        elif "f1" in metrics and base.get("majority_class_f1") is not None:
            ok = metrics["f1"] > base["majority_class_f1"]
            status = "plausible" if ok else "runs_but_wrong"
            reason = f"F1 {metrics['f1']} vs majority F1 {base['majority_class_f1']}"
    elif task == "tabular_regression":
        if "r2" in metrics:
            ok = metrics["r2"] > 0
            status = "plausible" if ok else "runs_but_wrong"
            reason = f"R2 {metrics['r2']} vs floor 0"
        elif "rmse" in metrics and base.get("mean_predictor_rmse") is not None:
            ok = metrics["rmse"] < base["mean_predictor_rmse"]
            status = "plausible" if ok else "runs_but_wrong"
            reason = f"RMSE {metrics['rmse']} vs mean-predictor {base['mean_predictor_rmse']}"
        elif "rmsle" in metrics and base.get("mean_predictor_rmsle") is not None:
            ok = metrics["rmsle"] < base["mean_predictor_rmsle"]
            status = "plausible" if ok else "runs_but_wrong"
            reason = f"RMSLE {metrics['rmsle']} vs mean-predictor {base['mean_predictor_rmsle']}"
    elif task == "tabular_clustering":
        if "ari" in metrics:
            ok = metrics["ari"] > base.get("floor_ari", 0.0)
            status = "plausible" if ok else "runs_but_wrong"
            reason = f"ARI {metrics['ari']} vs floor {base.get('floor_ari', 0.0)}"
        elif "nmi" in metrics:
            ok = metrics["nmi"] >= base.get("floor_nmi", 0.02)
            status = "plausible" if ok else "runs_but_wrong"
            reason = f"NMI {metrics['nmi']} vs floor {base.get('floor_nmi', 0.02)}"
    return {"status": status, "reason": reason, "parsed": metrics}


# --- Core aggregation --------------------------------------------------------
def _final_outcome(cell: dict, attempts: list[dict]) -> str:
    term = cell.get("termination")
    if term == "completed":
        return "working" if cell.get("final_rcode") == 0 else "unrecoverable"
    if term in ("timeout_after_attempts", "subprocess_error"):
        return "incomplete"
    if term == "pipeline_exception":
        return "unrecoverable"
    return "incomplete"


def _metrics_block(cells: list[dict], ledger: dict[str, list[dict]]) -> dict:
    code_cells = [c for c in cells if not c.get("infra_error") and ledger.get(c["cell_id"])]
    first_fail = sum(1 for c in code_cells if ledger[c["cell_id"]][0]["rcode"] != 0)

    all_attempts = [a for c in code_cells for a in ledger[c["cell_id"]]]
    broken_exec = sum(1 for a in all_attempts if a["rcode"] != 0)

    outcomes = Counter(_final_outcome(c, ledger[c["cell_id"]]) for c in code_cells)
    repair_hist: Counter = Counter()
    for c in code_cells:
        ok_indices = [a["attempt"] for a in ledger[c["cell_id"]] if a["rcode"] == 0]
        if ok_indices:
            repair_hist[min(ok_indices)] += 1

    n_outcome = len(code_cells)
    unrecoverable = outcomes["unrecoverable"] + outcomes["incomplete"]
    return {
        "n_cells_with_code": len(code_cells),
        "n_infra_cells": sum(1 for c in cells if c.get("infra_error")),
        "first_attempt_failure": wilson(first_fail, len(code_cells)),
        "per_execution_failure": wilson(broken_exec, len(all_attempts)),
        "final_outcome_counts": dict(outcomes),
        "final_unrecoverable": wilson(unrecoverable, n_outcome),
        "repair_attempts_to_success": {str(k): v for k, v in sorted(repair_hist.items())},
    }


def analyze() -> dict:
    status = json.loads(STATUS_PATH.read_text(encoding="utf-8")) if STATUS_PATH.exists() else {}
    baselines = json.loads(config.BASELINES_PATH.read_text(encoding="utf-8")) if config.BASELINES_PATH.exists() else {}
    meta = json.loads(RUN_META_PATH.read_text(encoding="utf-8")) if RUN_META_PATH.exists() else {}
    ledger = load_ledger()

    cells = list(status.values())
    by_model: dict[str, list[dict]] = defaultdict(list)
    for cell in cells:
        by_model[cell["model_key"]].append(cell)

    # Taxonomy over every FAILED execution attempt.
    tax_overall: Counter = Counter()
    tax_by_model: dict[str, Counter] = defaultdict(Counter)
    for cid, attempts in ledger.items():
        model_key = attempts[0]["model_key"]
        for a in attempts:
            if a["rcode"] != 0:
                cat = classify_error(a.get("log_tail", ""))["category"]
                tax_overall[cat] += 1
                tax_by_model[model_key][cat] += 1

    # Semantic check over cells with a successful execution.
    sem_details: list[dict] = []
    sem_overall: Counter = Counter()
    sem_by_model: dict[str, Counter] = defaultdict(Counter)
    for cell in cells:
        attempts = ledger.get(cell["cell_id"], [])
        ok_attempts = [a for a in attempts if a["rcode"] == 0]
        if not ok_attempts:
            continue
        result = semantic_check(cell["task"], cell["dataset"], ok_attempts[-1].get("log_tail", ""), baselines)
        sem_overall[result["status"]] += 1
        sem_by_model[cell["model_key"]][result["status"]] += 1
        sem_details.append({"cell_id": cell["cell_id"], "dataset": cell["dataset"],
                            "task": cell["task"], **result})

    def sem_block(counter: Counter) -> dict:
        checked = sum(counter.values())
        implausible = counter.get("runs_but_wrong", 0)
        missing = counter.get("no_metric_parsed", 0)
        return {
            "n_checked": checked,
            # Plan definition: a clean-exit run is a semantic failure if its metric
            # is missing / NaN / implausible. Breakdown kept for spot-checking, since
            # 'missing' can be a genuine non-evaluation OR a parser miss.
            "semantic_failure": wilson(implausible + missing, checked),
            "implausible_or_nan": implausible,
            "missing_metric": missing,
            "plausible": counter.get("plausible", 0),
        }

    # Cost + wall time.
    tokens = sum(_sum_tokens(c.get("money", {})) for c in cells)
    walls = [c.get("wall_sec", 0) for c in cells if c.get("wall_sec")]

    summary = {
        "meta": meta,
        "n_cells_recorded": len(cells),
        "n_cells_total": len(config.grid_cells()),
        "n_infra_cells": sum(1 for c in cells if c.get("infra_error")),
        "overall": _metrics_block(cells, ledger),
        "per_model": {m: _metrics_block(cs, ledger) for m, cs in by_model.items()},
        "taxonomy": {"overall": dict(tax_overall.most_common()),
                     "per_model": {m: dict(c.most_common()) for m, c in tax_by_model.items()}},
        "semantic": {"overall": sem_block(sem_overall),
                     "per_model": {m: sem_block(c) for m, c in sem_by_model.items()},
                     "details": sem_details},
        "cost_walltime": {
            "total_llm_tokens": tokens,
            "total_wall_sec": round(sum(walls), 1),
            "mean_wall_sec_per_cell": round(sum(walls) / len(walls), 1) if walls else None,
        },
    }
    return summary


# --- Rendering ---------------------------------------------------------------
def _fmt(block: dict) -> str:
    if block.get("rate") is None:
        return "n/a"
    return f"{block['rate'] * 100:.0f}% [{block['ci_low'] * 100:.0f}-{block['ci_high'] * 100:.0f}] (n={block['n']})"


def write_tables(summary: dict) -> None:
    lines = [
        "# AutoML-Agent — Generated Code Reliability (full-strength config)",
        "",
        f"- Snapshot date: `{summary['meta'].get('snapshot_date', 'n/a')}` · "
        f"AutoML-Agent commit: `{summary['meta'].get('git_commit', 'n/a')[:10]}`",
        f"- Cells recorded: {summary['n_cells_recorded']}/{summary['n_cells_total']} "
        f"(infra/provider-error cells excluded from code rates: {summary['n_infra_cells']})",
        f"- Config: `{json.dumps(summary['meta'].get('full_config', config.FULL_CONFIG))}`",
        "",
        "## Primary failure rates (per model), Wilson 95% CI",
        "",
        "| Model | First-attempt broken | Per-execution broken | Final unrecoverable | cells |",
        "|-------|----------------------|----------------------|---------------------|-------|",
    ]
    rows = list(summary["per_model"].items()) + [("**overall**", summary["overall"])]
    for name, block in rows:
        lines.append(
            f"| {name} | {_fmt(block['first_attempt_failure'])} | "
            f"{_fmt(block['per_execution_failure'])} | {_fmt(block['final_unrecoverable'])} | "
            f"{block['n_cells_with_code']} |"
        )

    lines += ["", "## First-attempt vs final (how much the safeguards recover)", ""]
    ov = summary["overall"]
    lines.append(f"- First generated script broken: **{_fmt(ov['first_attempt_failure'])}**")
    lines.append(f"- Still unrecoverable after the repair + revision loops: **{_fmt(ov['final_unrecoverable'])}**")
    lines.append(f"- Final outcome counts: `{ov['final_outcome_counts']}`")
    lines.append(f"- Repair attempts to first success (index -> #cells): `{ov['repair_attempts_to_success']}`")

    lines += ["", "## Error taxonomy (failed execution attempts)", "",
              "| Category | Count |", "|----------|-------|"]
    for cat, count in summary["taxonomy"]["overall"].items():
        lines.append(f"| {cat} | {count} |")

    lines += ["", "## Semantic 'runs-but-wrong' (of cells that exit 0)", ""]
    so = summary["semantic"]["overall"]
    lines.append(f"- Checked (ran cleanly): {so['n_checked']}")
    lines.append(f"- Semantic failure (missing/NaN/implausible metric): **{_fmt(so['semantic_failure'])}**")
    lines.append(f"- Breakdown: implausible/NaN={so['implausible_or_nan']} · "
                 f"missing metric={so['missing_metric']} · plausible={so['plausible']}")

    lines += ["", "## Cost & wall time", "",
              f"- Total LLM tokens: {summary['cost_walltime']['total_llm_tokens']:,}",
              f"- Mean wall time/cell: {summary['cost_walltime']['mean_wall_sec_per_cell']}s"]

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "tables.md").write_text("\n".join(lines), encoding="utf-8")


def write_figures(summary: dict) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[figures] matplotlib unavailable ({exc!r}); skipping figures.")
        return

    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    models = list(summary["per_model"].keys())

    # 1) failure-rate bars with CI whiskers
    if models:
        fig, ax = plt.subplots(figsize=(1.6 * len(models) + 3, 4))
        width = 0.38
        xs = range(len(models))
        for offset, key, label in [(-width / 2, "first_attempt_failure", "first-attempt"),
                                   (width / 2, "per_execution_failure", "per-execution")]:
            rates, lo, hi = [], [], []
            for m in models:
                b = summary["per_model"][m][key]
                rates.append((b["rate"] or 0) * 100)
                lo.append(((b["rate"] or 0) - (b["ci_low"] or 0)) * 100)
                hi.append(((b["ci_high"] or 0) - (b["rate"] or 0)) * 100)
            ax.bar([x + offset for x in xs], rates, width, label=label,
                   yerr=[lo, hi], capsize=4)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.set_ylabel("broken scripts (%)")
        ax.set_title("AutoML-Agent broken-code rate by backend (Wilson 95% CI)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(config.FIGURES_DIR / "failure_rates.png", dpi=150)
        plt.close(fig)

    # 2) error-taxonomy stacked bars per model
    tax = summary["taxonomy"]["per_model"]
    cats = list(summary["taxonomy"]["overall"].keys())
    if models and cats:
        fig, ax = plt.subplots(figsize=(1.6 * len(models) + 3, 4))
        bottoms = [0] * len(models)
        for cat in cats:
            vals = [tax.get(m, {}).get(cat, 0) for m in models]
            ax.bar(models, vals, bottom=bottoms, label=cat)
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        ax.set_ylabel("failed attempts")
        ax.set_title("Error taxonomy by backend")
        ax.legend(fontsize=7, ncol=2)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(config.FIGURES_DIR / "error_taxonomy.png", dpi=150)
        plt.close(fig)

    # 3) repair-attempts histogram (overall)
    hist = summary["overall"]["repair_attempts_to_success"]
    if hist:
        fig, ax = plt.subplots(figsize=(6, 4))
        keys = sorted(hist, key=int)
        ax.bar(keys, [hist[k] for k in keys])
        ax.set_xlabel("attempt index of first success (0 = first try)")
        ax.set_ylabel("# cells")
        ax.set_title("Repair attempts to first working script")
        fig.tight_layout()
        fig.savefig(config.FIGURES_DIR / "repair_attempts.png", dpi=150)
        plt.close(fig)


def main() -> None:
    summary = analyze()
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_tables(summary)
    write_figures(summary)
    ov = summary["overall"]
    print(f"Recorded {summary['n_cells_recorded']}/{summary['n_cells_total']} cells "
          f"({summary['n_infra_cells']} infra-excluded).")
    print(f"First-attempt broken: {_fmt(ov['first_attempt_failure'])}")
    print(f"Per-execution broken: {_fmt(ov['per_execution_failure'])}")
    print(f"Final unrecoverable:  {_fmt(ov['final_unrecoverable'])}")
    print(f"Outputs -> {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
