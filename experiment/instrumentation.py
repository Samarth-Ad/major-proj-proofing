"""Logging-only instrumentation for the AutoML-Agent reliability study.

Monkey-patches `execute_script` so that on EVERY call it (a) archives the current
generated script to a uniquely named file, (b) appends an attempt record to a
JSONL ledger, then (c) delegates to the original and returns its result unchanged.
The agent's decision logic is NOT modified.

Why this is required: each regeneration overwrites the single script file
(operation_agent/__init__.py:124-129) and each manager revision spawns a fresh
OperationAgent whose error_logs replace the prior ones, so the final
implementation_result undercounts attempts. Wrapping execution captures every
attempt across the inner 5-loop AND the outer revision loop, and preserves every
broken script as evidence.

The patch targets `operation_agent.execute_script` (the binding that
self_validation actually calls) as well as `operation_agent.execution.execute_script`.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import threading
import time
from pathlib import Path

from experiment import config

_LEDGER_LOCK = threading.Lock()
LOG_TAIL_CHARS = 6000  # keep the TAIL: both the final metric print and tracebacks live there

# ANSI colours for the optional live demo echo.
_R = "\033[0m"; _BOLD = "\033[1m"; _DIM = "\033[2m"
_RED = "\033[91m"; _GRN = "\033[92m"; _YEL = "\033[93m"; _CYN = "\033[96m"; _GRY = "\033[90m"


def _last_error_line(log: str) -> str:
    """Pull the most informative error line out of an execution log, for the live echo."""
    if not log or not log.strip():
        return ""
    if "timed out" in log.lower():
        return "Timeout — the script never terminated."
    errs = re.findall(r"^[A-Za-z_][\w.]*(?:Error|Exception):.*$", log, re.MULTILINE)
    if errs:
        return errs[-1].strip()
    return log.strip().splitlines()[-1][:200]


def _echo_attempt(index: int, code_text: str, rcode: int, log: str, code_lines: int) -> None:
    """Print one freshly generated attempt and its live result to the console."""
    bar = "─" * 74
    print(f"\n{_YEL}{bar}{_R}")
    print(f"{_BOLD}{_YEL}  ⚙  ATTEMPT {index + 1}  ·  AutoML-Agent just generated a script and ran it{_R}")
    print(f"{_YEL}{bar}{_R}")
    if code_text:
        lines = code_text.splitlines()
        shown = lines if code_lines <= 0 else lines[:code_lines]
        width = len(str(len(lines)))
        print(f"{_DIM}  generated {len(lines)} lines of Python:{_R}")
        for i, line in enumerate(shown, 1):
            print(f"{_GRY}{str(i).rjust(width)}│{_R} {_CYN}{line}{_R}")
        if code_lines > 0 and len(lines) > code_lines:
            print(f"{_GRY}  … {len(lines) - code_lines} more lines{_R}")
    else:
        print(f"{_DIM}  (no script file found on disk for this attempt){_R}")
    print(f"{_DIM}  ── ran it ──{_R}")
    if rcode == 0:
        print(f"{_BOLD}{_GRN}  ✓ RAN (exit 0){_R}")
    else:
        print(f"{_BOLD}{_RED}  ✗ BROKEN (exit {rcode}){_R}")
        err = _last_error_line(log)
        if err:
            print(f"{_RED}  {err}{_R}")


def install_instrumentation(cell: dict, archive_dir: Path, ledger_path: Path,
                            echo: bool = False, echo_code_lines: int = 40) -> dict:
    """Install the execute_script wrapper for one grid cell.

    Returns a mutable stats dict whose ``attempts`` count is updated in place after
    each execution (the child process reads it to report total archived attempts).

    When ``echo`` is True (the live demo), each attempt is also printed to the
    console: a banner, a capped preview of the freshly generated script, and the
    live pass/fail + error. This only prints; it changes nothing about the run.
    """
    import operation_agent
    import operation_agent.execution as opexec

    original = opexec.execute_script
    archive_dir.mkdir(parents=True, exist_ok=True)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    stats = {"attempts": 0}

    def wrapper(script_name: str, work_dir: str = ".", device: str = "0"):
        start = time.time()
        rcode, log = original(script_name, work_dir=work_dir, device=device)
        try:
            index = stats["attempts"]
            stats["attempts"] = index + 1

            source = (Path(work_dir) / script_name).resolve()
            dest = archive_dir / f"attempt_{index:02d}.py"
            code_text = ""
            sha = ""
            if source.exists():
                code_text = source.read_text(encoding="utf-8", errors="replace")
                shutil.copyfile(source, dest)
                sha = hashlib.sha256(code_text.encode("utf-8", "replace")).hexdigest()

            if echo:
                _echo_attempt(index, code_text, rcode, log, echo_code_lines)

            try:
                script_rel = str(dest.relative_to(config.PROJECT_ROOT)).replace("\\", "/")
            except ValueError:
                script_rel = str(dest)

            record = {
                "cell_id": cell["cell_id"],
                "model_key": cell["model_key"],
                "backend": cell["backend"],
                "model_id": cell["model_id"],
                "dataset": cell["dataset"],
                "task": cell["task"],
                "rep": cell["rep"],
                "attempt": index,
                "rcode": rcode,
                "ok": rcode == 0,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "elapsed_sec": round(time.time() - start, 2),
                "script": script_rel,
                "code_chars": len(code_text),
                "code_sha256": sha,
                "log_tail": (log or "")[-LOG_TAIL_CHARS:],
            }
            with _LEDGER_LOCK, open(ledger_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except Exception as exc:  # instrumentation must never break the agent run
            print(f"[instrumentation] ledger write failed: {exc!r}")
        return rcode, log

    opexec.execute_script = wrapper
    operation_agent.execute_script = wrapper
    return stats


def install_rate_limiter(min_interval_sec: float) -> None:
    """Enforce a minimum interval between OpenAI-compatible chat calls (timing only).

    Patches ``Completions.create`` process-wide so every client the agent builds is
    throttled, keeping free tiers under their per-minute request caps. Does NOT alter
    any agent decision; it only delays calls. No-op when ``min_interval_sec`` <= 0.
    """
    if min_interval_sec <= 0:
        return
    try:
        import openai.resources.chat.completions as chat_completions
    except Exception as exc:
        print(f"[throttle] could not locate openai Completions to patch: {exc!r}")
        return

    state = {"next_allowed": 0.0}
    lock = threading.Lock()
    original = chat_completions.Completions.create

    def throttled(self, *args, **kwargs):
        with lock:
            scheduled = max(time.time(), state["next_allowed"])
            state["next_allowed"] = scheduled + min_interval_sec
        delay = scheduled - time.time()
        if delay > 0:
            time.sleep(delay)
        return original(self, *args, **kwargs)

    chat_completions.Completions.create = throttled
    print(f"[throttle] min interval between LLM calls: {min_interval_sec}s")
