"""Orchestrator for the scaled AutoML-Agent reliability study.

Dual mode:

* Orchestrator (default): iterate the model x dataset x rep grid. Each cell runs
  in its OWN subprocess with a wall-clock watchdog (kill-on-exceed, whole process
  tree). Resumable via a status ledger; provider errors get exponential backoff +
  bounded retries and are classified separately from code brokenness.

* Cell runner (``--cell <spec.json> --out <result.json>``): set the backend env,
  install the logging-only instrumentation, build the AgentManager at the single
  full-strength config, run it once, and write a per-cell result JSON.

Usage:
    python -m experiment.run_grid --list
    python -m experiment.run_grid --cells gemini__wine__rep1        # smoke test
    python -m experiment.run_grid                                   # full grid (resumable)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from experiment import config

THIS_FILE = Path(__file__).resolve()
STATUS_PATH = config.RUNS_DIR / "cell_status.json"
RUN_META_PATH = config.RUNS_DIR / "run_meta.json"

PROVIDER_MARKERS = (
    "rate limit", "ratelimit", "rate_limit", "429", "413", "quota",
    "insufficient_quota", "authenticationerror", "permissiondenied", "unauthorized",
    "invalid api key", "invalid_api_key", "connection error", "connecterror",
    "apiconnectionerror", "overloaded", "service unavailable", "503", "502",
    "apitimeouterror", "too many requests",
)


# ---------------------------------------------------------------------------
# Child: run exactly one cell.
# ---------------------------------------------------------------------------
def run_cell(cell: dict, out_path: Path) -> None:
    # 1. Point the whole pipeline at this cell's backend/model BEFORE importing
    #    the agent (configs.py reads env at import; load_dotenv does not override).
    os.environ["AMLA_LLM"] = cell["backend"]
    os.environ[cell["model_env"]] = cell["model_id"]

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except Exception:
            pass

    from experiment import instrumentation

    spec = config.dataset_by_id(cell["dataset"])
    archive_dir = config.RUNS_DIR / cell["model_key"] / cell["dataset"] / f"rep{cell['rep']}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    requirements = config.requirements_for(spec)
    req_path = archive_dir / "requirements.json"
    with open(req_path, "w", encoding="utf-8") as handle:
        json.dump(requirements, handle, indent=2)

    stats = instrumentation.install_instrumentation(cell, archive_dir, config.LEDGER_PATH)
    instrumentation.install_rate_limiter(config.throttle_for(cell["backend"]))

    from agent_manager import AgentManager  # deferred: must follow the env setup above

    result: dict = {
        "cell_id": cell["cell_id"],
        "model_key": cell["model_key"],
        "backend": cell["backend"],
        "model_id": cell["model_id"],
        "dataset": cell["dataset"],
        "task": cell["task"],
        "rep": cell["rep"],
        "config": config.FULL_CONFIG,
    }
    start = time.time()
    manager = AgentManager(
        task=cell["task"],
        llm=cell["backend"],
        data_path=str(config.dataset_csv(cell["dataset"])),
        user_requirements=str(req_path),
        uid=cell["cell_id"],
        device=0,
        **config.FULL_CONFIG,
    )
    try:
        manager.initiate_chat(spec.prompt)
        impl = getattr(manager, "implementation_result", None) or {}
        result.update(
            {
                "termination": "completed",
                "final_state": manager.state,
                "final_rcode": impl.get("rcode"),
                "manager_n_attempts": getattr(manager, "n_attempts", None),
                "n_revise_remaining": getattr(manager, "n_revise", None),
                "final_code_chars": len(impl.get("code", "") or ""),
                "inner_error_logs": len(impl.get("error_logs", []) or []),
            }
        )
    except Exception as exc:
        import traceback

        trace = traceback.format_exc()
        result.update(
            {
                "termination": "pipeline_exception",
                "final_state": getattr(manager, "state", None),
                "final_rcode": None,
                "exception_type": type(exc).__name__,
                "exception_msg": str(exc)[:500],
                "looks_like_provider_error": _has_provider_marker(trace),
                "traceback_tail": trace[-2000:],
            }
        )
    result["attempts_archived"] = stats["attempts"]
    result["money"] = getattr(manager, "money", {})
    result["timers"] = getattr(manager, "timer", {})
    result["wall_sec"] = round(time.time() - start, 1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)


def _has_provider_marker(text: str) -> bool:
    low = (text or "").lower()
    return any(marker in low for marker in PROVIDER_MARKERS)


# ---------------------------------------------------------------------------
# Parent: orchestrate the grid.
# ---------------------------------------------------------------------------
def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            import signal

            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def _launch_cell(cell: dict, archive_dir: Path) -> dict:
    """Run one cell as a watched subprocess; return the orchestrator-side outcome."""
    archive_dir.mkdir(parents=True, exist_ok=True)
    spec_path = archive_dir / "cell_spec.json"
    result_path = archive_dir / "cell_result.json"
    console_path = archive_dir / "console.log"
    if result_path.exists():
        result_path.unlink()
    with open(spec_path, "w", encoding="utf-8") as handle:
        json.dump(cell, handle, indent=2)

    popen_kwargs: dict = {}
    if sys.platform != "win32":
        popen_kwargs["start_new_session"] = True

    start = time.time()
    with open(console_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            # Launch as a module (-m) so the project root is on sys.path and
            # `import experiment` resolves; a bare script path would not.
            [sys.executable, "-m", "experiment.run_grid", "--cell", str(spec_path), "--out", str(result_path)],
            cwd=str(config.PROJECT_ROOT),
            stdout=logf,
            stderr=subprocess.STDOUT,
            env={**os.environ},
            **popen_kwargs,
        )
        killed = False
        try:
            proc.wait(timeout=config.CELL_TIMEOUT_SEC)
        except subprocess.TimeoutExpired:
            _kill_tree(proc)
            killed = True
            try:
                proc.wait(timeout=30)
            except Exception:
                pass
    wall = round(time.time() - start, 1)

    console_text = console_path.read_text(encoding="utf-8", errors="replace") if console_path.exists() else ""
    attempts = _count_ledger_attempts(cell["cell_id"])
    return _classify_cell(cell, result_path, killed, wall, attempts, console_text)


def _count_ledger_attempts(cell_id: str) -> int:
    if not config.LEDGER_PATH.exists():
        return 0
    count = 0
    with open(config.LEDGER_PATH, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                if json.loads(line).get("cell_id") == cell_id:
                    count += 1
            except json.JSONDecodeError:
                continue
    return count


def _classify_cell(cell: dict, result_path: Path, killed: bool, wall: float,
                   attempts: int, console_text: str) -> dict:
    """Merge child result (if any) with watchdog state into one outcome record.

    infra_error=True means the cell yielded NO usable code-failure data (it is
    excluded from code numerators and reported separately).
    """
    outcome = {k: cell[k] for k in
               ("cell_id", "model_key", "backend", "model_id", "dataset", "task", "rep")}
    outcome["wall_sec"] = wall
    outcome["attempts_archived"] = attempts
    outcome["provider_markers"] = sorted({m for m in PROVIDER_MARKERS if m in console_text.lower()})

    child: dict = {}
    if result_path.exists():
        try:
            child = json.loads(result_path.read_text(encoding="utf-8"))
        except Exception:
            child = {}

    if killed:
        # Watchdog fired. Attempts already archived are valid code evidence.
        outcome["termination"] = "timeout_after_attempts" if attempts else "timeout_no_code"
        outcome["final_state"] = child.get("final_state")
        outcome["final_rcode"] = None
        outcome["infra_error"] = attempts == 0
    elif child:
        outcome["termination"] = child.get("termination", "completed")
        outcome["final_state"] = child.get("final_state")
        outcome["final_rcode"] = child.get("final_rcode")
        outcome["manager_n_attempts"] = child.get("manager_n_attempts")
        outcome["n_revise_remaining"] = child.get("n_revise_remaining")
        outcome["final_code_chars"] = child.get("final_code_chars")
        outcome["money"] = child.get("money", {})
        outcome["timers"] = child.get("timers", {})
        if outcome["termination"] == "pipeline_exception":
            outcome["exception_type"] = child.get("exception_type")
            outcome["exception_msg"] = child.get("exception_msg")
            # provider-side exception with no code produced => infra, not code failure
            outcome["infra_error"] = bool(child.get("looks_like_provider_error")) and attempts == 0
        else:
            outcome["infra_error"] = False
    else:
        # Subprocess exited without writing a result and was not killed by us.
        outcome["termination"] = "subprocess_error"
        outcome["final_state"] = None
        outcome["final_rcode"] = None
        outcome["infra_error"] = attempts == 0

    # A cell that produced provider markers and never ran code is an infra failure.
    if attempts == 0 and outcome.get("provider_markers"):
        outcome["infra_error"] = True
    return outcome


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(config.PROJECT_ROOT),
            capture_output=True, text=True, timeout=10,
        ).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _template_hashes() -> dict:
    hashes = {}
    for task in ("tabular_classification", "tabular_regression", "tabular_clustering"):
        path = config.PROJECT_ROOT / "prompt_pool" / f"{task}.py"
        if path.exists():
            hashes[task] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _write_run_meta() -> None:
    meta = {
        "snapshot_date": config.SNAPSHOT_DATE,
        "git_commit": _git_commit(),
        "full_config": config.FULL_CONFIG,
        "reps": config.REPS,
        "cell_timeout_sec": config.CELL_TIMEOUT_SEC,
        "models": [vars(m) for m in config.selected_models()],
        "datasets": [d.id for d in config.DATASETS],
        "prompt_template_sha256": _template_hashes(),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RUN_META_PATH, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2)


def _load_status() -> dict:
    if STATUS_PATH.exists():
        try:
            return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_status(status: dict) -> None:
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STATUS_PATH, "w", encoding="utf-8") as handle:
        json.dump(status, handle, indent=2)


def orchestrate(args: argparse.Namespace) -> None:
    cells = config.grid_cells()
    if args.cells:
        wanted = {c.strip() for c in args.cells.split(",") if c.strip()}
        cells = [c for c in cells if c["cell_id"] in wanted]
    if args.limit:
        cells = cells[: args.limit]

    if args.list:
        for cell in cells:
            print(cell["cell_id"], "->", cell["backend"], cell["model_id"])
        print(f"\n{len(cells)} cell(s).")
        return

    _write_run_meta()
    status = _load_status()
    retry_infra = os.getenv("AMLA_EXP_RETRY_INFRA") == "1"
    provider_retries = int(os.getenv("AMLA_PROVIDER_RETRIES", "2"))
    backoff_base = float(os.getenv("AMLA_BACKOFF_BASE", "30"))
    inter_cell = float(os.getenv("AMLA_INTER_CELL_DELAY", "3"))

    total = len(cells)
    for index, cell in enumerate(cells, 1):
        cid = cell["cell_id"]
        prior = status.get(cid)
        if prior and not (retry_infra and prior.get("infra_error")):
            print(f"[{index}/{total}] SKIP (done): {cid} -> {prior.get('termination')}")
            continue

        archive_dir = config.RUNS_DIR / cell["model_key"] / cell["dataset"] / f"rep{cell['rep']}"
        attempt = 0
        while True:
            print(f"[{index}/{total}] RUN: {cid} (backend={cell['backend']}, try={attempt + 1})")
            outcome = _launch_cell(cell, archive_dir)
            outcome["retries_used"] = attempt
            is_infra = outcome.get("infra_error")
            if is_infra and attempt < provider_retries:
                delay = backoff_base * (2 ** attempt)
                print(f"    infra/provider error ({outcome.get('provider_markers') or outcome.get('termination')}); "
                      f"backing off {delay:.0f}s then retrying.")
                time.sleep(delay)
                attempt += 1
                continue
            break

        status[cid] = outcome
        _save_status(status)
        print(f"    => {outcome['termination']} | attempts={outcome['attempts_archived']} "
              f"| final_rcode={outcome.get('final_rcode')} | infra={outcome.get('infra_error')} "
              f"| {outcome['wall_sec']}s")
        time.sleep(inter_cell)

    done = sum(1 for c in cells if c["cell_id"] in status)
    print(f"\n===== GRID PASS COMPLETE: {done}/{total} cells recorded =====")
    print(f"Ledger: {config.LEDGER_PATH}\nStatus: {STATUS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AutoML-Agent reliability study grid runner")
    parser.add_argument("--cell", type=str, help="(child) path to a single cell spec JSON")
    parser.add_argument("--out", type=str, help="(child) path to write the cell result JSON")
    parser.add_argument("--cells", type=str, help="comma-separated cell_ids to run (subset)")
    parser.add_argument("--limit", type=int, help="run only the first N cells")
    parser.add_argument("--list", action="store_true", help="list the grid and exit")
    args = parser.parse_args()

    if args.cell:
        cell = json.loads(Path(args.cell).read_text(encoding="utf-8"))
        out_path = Path(args.out) if args.out else (
            config.RUNS_DIR / cell["model_key"] / cell["dataset"] / f"rep{cell['rep']}" / "cell_result.json"
        )
        run_cell(cell, out_path)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
