"""Live terminal demo: show AutoML-Agent's generated code, then run it and watch it break.

For each chosen broken script it prints a banner (which model / dataset / task / attempt),
shows the generated code, then EXECUTES the script live so the real traceback appears in
the terminal — ideal for a screen recording. Colour + press-Enter pacing let you narrate.

Nothing is generated on the fly; these are the verbatim scripts the agent produced during
the study (from experiment/broken_code/, built by collect_broken.py).

Examples:
    python -m experiment.demo_broken                 # curated showcase (press Enter between)
    python -m experiment.demo_broken --auto          # no pauses (good for an unattended capture)
    python -m experiment.demo_broken --list          # list every broken script you can demo
    python -m experiment.demo_broken --category import_error --n 2
    python -m experiment.demo_broken --file experiment/broken_code/syntax_error/<name>.py
    python -m experiment.demo_broken --captured      # show the originally-recorded error (don't re-run)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from experiment import config

BROKEN_DIR = config.PKG_DIR / "broken_code"
INDEX_PATH = BROKEN_DIR / "broken_index.json"

# ANSI colours
R = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
RED = "\033[91m"; GRN = "\033[92m"; YEL = "\033[93m"; BLU = "\033[94m"; CYN = "\033[96m"; GRY = "\033[90m"

# A hand-picked, reliable, vivid set (one per failure mode). Paths are relative to repo root.
SHOWCASE = [
    "experiment/broken_code/undefined_name/ollama2__wine__rep1__attempt_00.py",
    "experiment/broken_code/import_error/ollama__iris__rep1__attempt_02.py",
    "experiment/broken_code/data_or_value_error/ollama__iris__rep1__attempt_01.py",
    "experiment/broken_code/syntax_error/ollama__breast_cancer__rep1__attempt_20.py",
    "experiment/broken_code/wrong_api_usage/ollama__breast_cancer__rep2__attempt_04.py",
]


def _enable_ansi() -> None:
    if sys.platform == "win32":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetConsoleMode(k.GetStdHandle(-11), 7)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
        except Exception:
            pass
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except Exception:
        pass


def _load_index() -> dict:
    if not INDEX_PATH.exists():
        print(f"{RED}No broken_index.json found. Run `python -m experiment.collect_broken` first.{R}")
        sys.exit(1)
    data = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    return {e["file"]: e for e in data["files"]}


def _rule(ch: str = "=", color: str = BLU) -> str:
    return f"{color}{ch * 78}{R}"


def _banner(entry: dict, n: int, total: int) -> None:
    print("\n" + _rule("="))
    print(f"{BOLD}{YEL}  DEMO {n}/{total}  ·  AutoML-Agent generated code — and it's broken{R}")
    print(_rule("="))
    print(f"  {BOLD}Model:{R}   {entry['model_key']}  ({config.ALL_MODELS.get(entry['model_key']).model_id if entry['model_key'] in config.ALL_MODELS else entry['model_key']})")
    print(f"  {BOLD}Task:{R}    {entry['task'].replace('tabular_', '')} on '{entry['dataset']}'")
    print(f"  {BOLD}Attempt:{R} #{entry['attempt']} of this run   ·   {BOLD}recorded error:{R} {RED}{entry['category']}{R}")
    print(f"  {DIM}source: {entry['source']}{R}")
    print(_rule("-", GRY))


def _print_code(path: Path, max_lines: int) -> None:
    print(f"{BOLD}{CYN}  ┌─ GENERATED CODE {('(' + path.name + ')')}{R}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    shown = lines if max_lines <= 0 else lines[:max_lines]
    width = len(str(len(lines)))
    for i, line in enumerate(shown, 1):
        print(f"{GRY}{str(i).rjust(width)}│{R} {CYN}{line}{R}")
    if max_lines > 0 and len(lines) > max_lines:
        print(f"{GRY}  … {len(lines) - max_lines} more lines "
              f"(full file: {path}){R}")
    print(f"{BOLD}{CYN}  └─ {len(lines)} lines total{R}")


def _run_live(path: Path, timeout: int) -> None:
    print(f"\n{BOLD}{YEL}  ▶ RUNNING IT LIVE …{R}  {DIM}(python {path.name}){R}\n")
    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(path)],
            cwd=str(config.PROJECT_ROOT),
            capture_output=True, text=True, timeout=timeout,
            encoding="utf-8", errors="replace",
        )
        out = (proc.stdout or "")[-1500:]
        err = (proc.stderr or "")[-3000:]
        if out.strip():
            print(DIM + out.rstrip() + R)
        if err.strip():
            for line in err.rstrip().splitlines():
                print(f"{RED}{line}{R}")
        elapsed = time.time() - start
        if proc.returncode == 0:
            print(f"\n{GRN}  exit code 0 — (this attempt happened to run){R}")
        else:
            print(f"\n{BOLD}{RED}  ✗ CRASHED — exit code {proc.returncode}  "
                  f"({elapsed:.1f}s){R}")
    except subprocess.TimeoutExpired:
        print(f"\n{BOLD}{RED}  ✗ NEVER TERMINATED — killed after {timeout}s "
              f"(non-termination failure){R}")


def _show_captured(entry: dict) -> None:
    print(f"\n{BOLD}{YEL}  ▶ ERROR RECORDED DURING THE STUDY:{R}")
    print(f"{RED}  {entry['error']}{R}")
    print(f"{DIM}  (exit code {entry['rcode']}){R}")


def _resolve_targets(args, index: dict) -> list[dict]:
    if args.list:
        return []
    if args.file:
        rel = str(Path(args.file)).replace("\\", "/")
        rel = rel.split("AutoMLAgent/", 1)[-1]
        entry = index.get(rel)
        if not entry:
            print(f"{RED}Not in the broken set: {rel}{R}")
            sys.exit(1)
        return [entry]
    if args.category:
        picks = [e for e in index.values() if e["category"] == args.category]
        if not picks:
            print(f"{RED}No broken scripts in category '{args.category}'.{R}")
            sys.exit(1)
        return picks[: args.n]
    # default: the curated showcase
    out = []
    for rel in SHOWCASE:
        if rel in index:
            out.append(index[rel])
    return out


def main() -> None:
    _enable_ansi()
    parser = argparse.ArgumentParser(description="Live demo of AutoML-Agent's broken generated code")
    parser.add_argument("--file", help="demo one specific broken script (path under broken_code/)")
    parser.add_argument("--category", help="demo scripts from one error category")
    parser.add_argument("--n", type=int, default=1, help="how many from --category (default 1)")
    parser.add_argument("--auto", action="store_true", help="no press-Enter pauses")
    parser.add_argument("--captured", action="store_true", help="show the recorded error instead of re-running")
    parser.add_argument("--full", action="store_true", help="print the whole script (no line cap)")
    parser.add_argument("--max-code-lines", type=int, default=70, help="max code lines shown (default 70; 0 = all)")
    parser.add_argument("--timeout", type=int, default=60, help="seconds before a live run is killed (default 60)")
    parser.add_argument("--list", action="store_true", help="list every broken script and exit")
    args = parser.parse_args()

    index = _load_index()

    if args.list:
        for cat in sorted({e["category"] for e in index.values()}):
            picks = [e for e in index.values() if e["category"] == cat]
            print(f"\n{BOLD}{cat}{R} ({len(picks)})")
            for e in picks:
                print(f"  {e['file']}")
        return

    targets = _resolve_targets(args, index)
    max_lines = 0 if args.full else args.max_code_lines
    total = len(targets)
    print(f"{BOLD}{GRN}AutoML-Agent — live broken-code demo  ·  {total} example(s){R}")
    print(f"{DIM}Each script below is exactly what the agent generated. We show it, then run it.{R}")

    for i, entry in enumerate(targets, 1):
        _banner(entry, i, total)
        path = config.PROJECT_ROOT / entry["file"]
        _print_code(path, max_lines)
        if args.captured:
            _show_captured(entry)
        else:
            _run_live(path, args.timeout)
        if not args.auto and i < total:
            try:
                input(f"\n{DIM}  [press Enter for the next example] {R}")
            except (EOFError, KeyboardInterrupt):
                break

    print(f"\n{_rule('=')}")
    print(f"{BOLD}{GRN}  End of demo.{R}  {DIM}Full evidence: experiment/broken_code/ · experiment/ledger.jsonl{R}")
    print(_rule("="))


if __name__ == "__main__":
    main()
