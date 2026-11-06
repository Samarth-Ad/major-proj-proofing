import os
import sys
import subprocess

# Max seconds a generated script may run before it is killed (prevents hangs,
# e.g. a Gradio server launch or an accidental infinite loop).
EXEC_TIMEOUT = int(os.getenv("AMLA_EXEC_TIMEOUT", "300"))


def execute_script(script_name, work_dir=".", device="0"):
    """Run a generated Python script as a subprocess and return (return_code, log).

    Cross-platform: unlike the original, this does not use the Unix-only
    ``VAR=value cmd`` shell prefix nor ``selectors`` on pipes (which is not
    supported for pipes on Windows). The GPU is selected via ``env`` and output
    is captured with a hard timeout.
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
        return -1, f"Something went wrong in executing {script_name}: {e}. Please check if it is ready to be executed."
