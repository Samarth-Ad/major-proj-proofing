import os
import sys
import subprocess

# Max seconds a generated script may run before it is killed (prevents hangs).
EXEC_TIMEOUT = int(os.getenv("AMLA_EXEC_TIMEOUT", "300"))


def execute_script(script_name, work_dir=".", device="0"):
    """Run a generated Python script as a subprocess and return (return_code, log).

    Cross-platform rewrite (see operation_agent/execution.py for details).
    """
    if not os.path.exists(os.path.join(work_dir, script_name)):
        raise Exception(f"The file {script_name} does not exist.")
    try:
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(device)}
        proc = subprocess.run(
            [sys.executable, "-u", script_name],
            cwd=work_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=EXEC_TIMEOUT,
        )
        return_code = proc.returncode
        if return_code != 0:
            observation = proc.stderr
        else:
            observation = proc.stdout or proc.stderr
        return return_code, "The script has been executed. Here is the output:\n" + observation

    except subprocess.TimeoutExpired as e:
        return -1, f"Execution of {script_name} timed out after {EXEC_TIMEOUT}s: {e}"
    except Exception as e:
        print("++++", "Wrong!")
        return -1, f"Something went wrong in executing {script_name}: {e}. Please check if it is ready to be executed."
