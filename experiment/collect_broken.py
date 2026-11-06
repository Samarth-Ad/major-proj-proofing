"""Segregate the broken-code evidence into one browsable folder.

Reads the attempt ledger, finds every execution that FAILED (rcode != 0), copies
the exact archived script into ``experiment/broken_code/<category>/`` under a unique
name, and writes an index (Markdown + JSON) mapping each file to the error it threw.

This does not re-run anything; it only reorganizes copies of the already-archived
scripts so the broken ones can be inspected on their own. Re-runnable: the output
folder is rebuilt from scratch each time.

Run:  python -m experiment.collect_broken
"""
from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict

from experiment import config
from experiment.analyze import classify_error, load_ledger

BROKEN_DIR = config.PKG_DIR / "broken_code"


def _reset_output() -> None:
    if BROKEN_DIR.exists():
        shutil.rmtree(BROKEN_DIR)
    BROKEN_DIR.mkdir(parents=True, exist_ok=True)


def collect() -> dict:
    ledger = load_ledger()
    entries: list[dict] = []
    missing: list[str] = []

    for cell_id, attempts in ledger.items():
        for a in attempts:
            if a.get("rcode") == 0:
                continue  # keep only the broken executions
            info = classify_error(a.get("log_tail", ""))
            category = info["category"]
            source = config.PROJECT_ROOT / a["script"]
            dest_name = f"{cell_id}__attempt_{a['attempt']:02d}.py"
            dest = BROKEN_DIR / category / dest_name

            dest.parent.mkdir(parents=True, exist_ok=True)
            if source.exists():
                shutil.copyfile(source, dest)
            else:
                missing.append(a["script"])

            entries.append({
                "file": str(dest.relative_to(config.PROJECT_ROOT)).replace("\\", "/"),
                "cell_id": cell_id,
                "model_key": a["model_key"],
                "dataset": a["dataset"],
                "task": a["task"],
                "rep": a["rep"],
                "attempt": a["attempt"],
                "rcode": a["rcode"],
                "category": category,
                "exception": info["exception"],
                "error": info["last_line"],
                "source": a["script"],
            })

    entries.sort(key=lambda e: (e["category"], e["cell_id"], e["attempt"]))
    by_cat: Counter = Counter(e["category"] for e in entries)
    by_model: Counter = Counter(e["model_key"] for e in entries)
    return {"entries": entries, "by_cat": by_cat, "by_model": by_model, "missing": missing}


def _write_json(result: dict) -> None:
    payload = {
        "total_broken": len(result["entries"]),
        "by_category": dict(result["by_cat"].most_common()),
        "by_model": dict(result["by_model"].most_common()),
        "files": result["entries"],
    }
    (BROKEN_DIR / "broken_index.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")


def _write_index_md(result: dict) -> None:
    entries = result["entries"]
    lines = [
        "# Broken-code evidence",
        "",
        f"Every generated script that **failed execution** (exit code != 0) in the "
        f"study, copied out of `experiment/runs/` and grouped by error type. "
        f"**{len(entries)} broken scripts** in total.",
        "",
        "Each filename is `<model>__<dataset>__rep<N>__attempt_<NN>.py`; the original "
        "lives at the `source` path in `broken_index.json`. Nothing here was re-run — "
        "these are verbatim copies of the archived attempts.",
        "",
        "## Counts by error type",
        "",
        "| Error type (folder) | Broken scripts |",
        "|---------------------|---------------:|",
    ]
    for cat, n in result["by_cat"].most_common():
        lines.append(f"| `{cat}/` | {n} |")
    lines += [f"| **total** | **{len(entries)}** |", "",
              "## Counts by model", "", "| Model | Broken scripts |",
              "|-------|---------------:|"]
    for model, n in result["by_model"].most_common():
        lines.append(f"| {model} | {n} |")
    lines.append("")

    grouped: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        grouped[e["category"]].append(e)

    for cat, _ in result["by_cat"].most_common():
        lines += ["", f"## `{cat}/` ({len(grouped[cat])})", "",
                  "| File | Error |", "|------|-------|"]
        for e in grouped[cat]:
            err = (e["error"] or "").replace("|", "\\|")[:160]
            lines.append(f"| `{e['file'].split('/', 2)[-1]}` | {err} |")

    if result["missing"]:
        lines += ["", "## Missing source scripts (archived path absent)", ""]
        lines += [f"- `{m}`" for m in result["missing"]]

    (BROKEN_DIR / "INDEX.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    _reset_output()
    result = collect()
    _write_json(result)
    _write_index_md(result)
    print(f"Collected {len(result['entries'])} broken scripts -> {BROKEN_DIR}")
    for cat, n in result["by_cat"].most_common():
        print(f"  {cat:26s} {n}")
    if result["missing"]:
        print(f"  ({len(result['missing'])} source scripts missing on disk)")


if __name__ == "__main__":
    main()
