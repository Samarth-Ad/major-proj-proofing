"""LIVE end-to-end demo: hand AutoML-Agent a dataset and watch it generate broken code.

This is the real thing, from zero — not a replay. It:
  1. shows the dataset you're handing the system,
  2. shows the plain-English task,
  3. launches the UNMODIFIED AutoML-Agent on it (real LLM calls, real code generation),
  4. echoes each freshly generated script and its live pass/fail + error as it happens.

Perfect for a screen recording: start recording, run one command, narrate as the agent
plans, writes code, runs it, and the code breaks in real time.

Requirements: the chosen backend must be reachable (e.g. `ollama signin` done and the
cloud model available) and network access for the LLM calls.

Examples:
    python -m experiment.demo_live                         # gpt-oss on titanic (reliable breaker)
    python -m experiment.demo_live --dataset breast_cancer
    python -m experiment.demo_live --model ollama2 --dataset diabetes
    python -m experiment.demo_live --code-lines 0          # show the FULL generated script each time

Tip: once you've shown a couple of broken attempts, press Ctrl-C to stop — a full run can
take 10-15 minutes as the agent keeps retrying.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from experiment import config

_R = "\033[0m"; _BOLD = "\033[1m"; _DIM = "\033[2m"
_RED = "\033[91m"; _GRN = "\033[92m"; _YEL = "\033[93m"; _BLU = "\033[94m"; _CYN = "\033[96m"; _GRY = "\033[90m"

DEMO_DIR = config.PKG_DIR / "demo_runs"


def _enable_ansi() -> None:
    if sys.platform == "win32":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetConsoleMode(k.GetStdHandle(-11), 7)
        except Exception:
            pass
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass


def _rule(color: str = _BLU, ch: str = "=") -> str:
    return f"{color}{ch * 74}{_R}"


def _pause(auto: bool, msg: str) -> None:
    if auto:
        return
    try:
        input(f"{_DIM}  [press Enter {msg}] {_R}")
    except (EOFError, KeyboardInterrupt):
        raise SystemExit(0)


def _show_dataset(spec) -> None:
    csv = config.dataset_csv(spec.id)
    print(f"\n{_rule('=')}")
    print(f"{_BOLD}{_YEL}  STEP 1 — the dataset we hand the system{_R}")
    print(_rule("="))
    print(f"  {_BOLD}File:{_R} {csv}")
    try:
        import pandas as pd
        df = pd.read_csv(csv)
        print(f"  {_BOLD}Shape:{_R} {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  {_BOLD}Columns:{_R} {list(df.columns)}")
        print(f"  {_BOLD}Target:{_R} {spec.target}")
        print(f"\n{_DIM}  first 5 rows:{_R}")
        with pd.option_context("display.max_columns", 12, "display.width", 120):
            for line in df.head(5).to_string(index=False).splitlines():
                print(f"{_CYN}  {line}{_R}")
    except Exception as exc:
        print(f"  {_DIM}(preview unavailable: {exc!r}){_R}")


def _show_task(spec) -> None:
    print(f"\n{_rule('=')}")
    print(f"{_BOLD}{_YEL}  STEP 2 — the plain-English task we give the system{_R}")
    print(_rule("="))
    print(f"  {_BOLD}Task type:{_R} {spec.task.replace('tabular_', 'tabular ')}")
    print(f"  {_BOLD}Prompt:{_R}")
    print(f'  {_CYN}"{spec.prompt}"{_R}')
    print(f"\n{_DIM}  The agent gets ONLY this prompt + the CSV. It must figure out the rest.{_R}")


def main() -> None:
    _enable_ansi()
    parser = argparse.ArgumentParser(description="Live demo: AutoML-Agent generates broken code from a dataset")
    parser.add_argument("--dataset", default="titanic",
                        help="dataset id (default titanic). Options: " + ", ".join(d.id for d in config.DATASETS))
    parser.add_argument("--model", default="ollama",
                        help="model key from config.ALL_MODELS (default ollama = gpt-oss:120b)")
    parser.add_argument("--code-lines", type=int, default=45,
                        help="max lines of each generated script to print (0 = full script)")
    parser.add_argument("--auto", action="store_true", help="no press-Enter pauses before the run")
    args = parser.parse_args()

    if args.model not in config.ALL_MODELS:
        print(f"{_RED}Unknown model '{args.model}'. Known: {sorted(config.ALL_MODELS)}{_R}")
        sys.exit(1)
    model = config.ALL_MODELS[args.model]
    spec = config.dataset_by_id(args.dataset)

    print(f"{_BOLD}{_GRN}AutoML-Agent — LIVE broken-code demo (real run, from zero){_R}")
    print(f"{_DIM}Backend: {model.backend} · model: {model.model_id} · full-strength config{_R}")

    _show_dataset(spec)
    _show_task(spec)

    print(f"\n{_rule('=')}")
    print(f"{_BOLD}{_YEL}  STEP 3 — launch AutoML-Agent and watch it generate & run code{_R}")
    print(_rule("="))
    print(f"{_DIM}  Every time the agent writes a script and executes it, we print the script{_R}")
    print(f"{_DIM}  and whether it ran or BROKE. (Ctrl-C any time once you've seen enough.){_R}")
    _pause(args.auto, "to start the live run")

    # Point the pipeline at the chosen LLM BEFORE importing the agent (config reads env at import).
    os.environ["AMLA_LLM"] = model.backend
    os.environ[model.model_env] = model.model_id

    from experiment import instrumentation

    archive_dir = DEMO_DIR / f"{model.key}__{spec.id}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    requirements = config.requirements_for(spec)
    req_path = archive_dir / "requirements.json"
    req_path.write_text(json.dumps(requirements, indent=2), encoding="utf-8")

    cell = {"cell_id": f"demo__{model.key}__{spec.id}", "model_key": model.key,
            "backend": model.backend, "model_id": model.model_id,
            "dataset": spec.id, "task": spec.task, "rep": 1}

    # Live echo ON; demo artifacts go to their own folder/ledger (study data untouched).
    instrumentation.install_instrumentation(
        cell, archive_dir, DEMO_DIR / "demo_ledger.jsonl",
        echo=True, echo_code_lines=args.code_lines)
    instrumentation.install_rate_limiter(config.throttle_for(model.backend))

    from agent_manager import AgentManager  # deferred: must follow the env setup above

    start = time.time()
    broken = 0
    try:
        manager = AgentManager(
            task=spec.task, llm=model.backend,
            data_path=str(config.dataset_csv(spec.id)),
            user_requirements=str(req_path), uid=cell["cell_id"], device=0,
            **config.FULL_CONFIG,
        )
        manager.initiate_chat(spec.prompt)
        state = getattr(manager, "state", "?")
        impl = getattr(manager, "implementation_result", None) or {}
        final_rcode = impl.get("rcode")
    except KeyboardInterrupt:
        print(f"\n{_YEL}  (stopped by you){_R}")
        state, final_rcode = "interrupted", None
    except Exception as exc:
        print(f"\n{_RED}  pipeline raised: {type(exc).__name__}: {str(exc)[:200]}{_R}")
        state, final_rcode = "exception", None

    # Count broken from the demo ledger for this run.
    try:
        for line in (DEMO_DIR / "demo_ledger.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                if rec.get("cell_id") == cell["cell_id"] and rec.get("rcode") != 0:
                    broken += 1
    except Exception:
        pass

    print(f"\n{_rule('=')}")
    print(f"{_BOLD}{_GRN}  DEMO COMPLETE{_R}  ({time.time() - start:.0f}s)")
    print(f"  Broken execution attempts seen this run: {_BOLD}{_RED}{broken}{_R}")
    print(f"  Final agent state: {state} · final exit code: {final_rcode}")
    print(f"{_DIM}  Scripts archived under: {archive_dir}{_R}")
    print(_rule("="))


if __name__ == "__main__":
    main()
