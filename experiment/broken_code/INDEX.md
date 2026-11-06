# Broken-code evidence

Every generated script that **failed execution** (exit code != 0) in the study, copied out of `experiment/runs/` and grouped by error type. **282 broken scripts** in total.

Each filename is `<model>__<dataset>__rep<N>__attempt_<NN>.py`; the original lives at the `source` path in `broken_index.json`. Nothing here was re-run — these are verbatim copies of the archived attempts.

## Counts by error type

| Error type (folder) | Broken scripts |
|---------------------|---------------:|
| `import_error/` | 94 |
| `other_runtime_error/` | 45 |
| `non_termination_timeout/` | 37 |
| `file_error/` | 33 |
| `wrong_api_usage/` | 30 |
| `data_or_value_error/` | 26 |
| `syntax_error/` | 10 |
| `runtime_error/` | 6 |
| `undefined_name/` | 1 |
| **total** | **282** |

## Counts by model

| Model | Broken scripts |
|-------|---------------:|
| ollama | 227 |
| ollama2 | 55 |


## `import_error/` (94)

| File | Error |
|------|-------|
| `import_error/ollama2__breast_cancer__rep2__attempt_00.py` | ImportError |
| `import_error/ollama2__breast_cancer__rep2__attempt_05.py` | ImportError |
| `import_error/ollama2__titanic__rep2__attempt_00.py` | ImportError |
| `import_error/ollama__breast_cancer__rep1__attempt_00.py` | ModuleNotFoundError |
| `import_error/ollama__breast_cancer__rep1__attempt_02.py` | ModuleNotFoundError |
| `import_error/ollama__breast_cancer__rep1__attempt_04.py` | ModuleNotFoundError |
| `import_error/ollama__breast_cancer__rep1__attempt_05.py` | ModuleNotFoundError |
| `import_error/ollama__breast_cancer__rep1__attempt_06.py` | ModuleNotFoundError |
| `import_error/ollama__breast_cancer__rep1__attempt_07.py` | ModuleNotFoundError |
| `import_error/ollama__breast_cancer__rep1__attempt_08.py` | ModuleNotFoundError |
| `import_error/ollama__breast_cancer__rep1__attempt_09.py` | ModuleNotFoundError |
| `import_error/ollama__breast_cancer__rep1__attempt_10.py` | ModuleNotFoundError |
| `import_error/ollama__breast_cancer__rep1__attempt_13.py` | ModuleNotFoundError |
| `import_error/ollama__breast_cancer__rep1__attempt_15.py` | ModuleNotFoundError |
| `import_error/ollama__breast_cancer__rep1__attempt_16.py` | ModuleNotFoundError |
| `import_error/ollama__breast_cancer__rep1__attempt_18.py` | ModuleNotFoundError |
| `import_error/ollama__breast_cancer__rep2__attempt_00.py` | ModuleNotFoundError |
| `import_error/ollama__breast_cancer__rep2__attempt_05.py` | ModuleNotFoundError |
| `import_error/ollama__breast_cancer__rep2__attempt_06.py` | ModuleNotFoundError |
| `import_error/ollama__breast_cancer__rep2__attempt_09.py` | ImportError |
| `import_error/ollama__breast_cancer__rep2__attempt_10.py` | ImportError |
| `import_error/ollama__breast_cancer__rep2__attempt_11.py` | ImportError |
| `import_error/ollama__breast_cancer__rep2__attempt_12.py` | ImportError |
| `import_error/ollama__breast_cancer__rep2__attempt_14.py` | ImportError |
| `import_error/ollama__breast_cancer__rep2__attempt_16.py` | ImportError |
| `import_error/ollama__breast_cancer__rep2__attempt_18.py` | ImportError |
| `import_error/ollama__breast_cancer__rep3__attempt_00.py` | ModuleNotFoundError |
| `import_error/ollama__breast_cancer__rep3__attempt_01.py` | ModuleNotFoundError |
| `import_error/ollama__breast_cancer__rep3__attempt_03.py` | ImportError |
| `import_error/ollama__california__rep1__attempt_01.py` | ModuleNotFoundError |
| `import_error/ollama__california__rep1__attempt_02.py` | ModuleNotFoundError |
| `import_error/ollama__california__rep1__attempt_03.py` | ImportError |
| `import_error/ollama__california__rep1__attempt_04.py` | ModuleNotFoundError |
| `import_error/ollama__california__rep1__attempt_05.py` | ModuleNotFoundError |
| `import_error/ollama__california__rep3__attempt_01.py` | ModuleNotFoundError |
| `import_error/ollama__diabetes__rep1__attempt_00.py` | ModuleNotFoundError |
| `import_error/ollama__diabetes__rep1__attempt_01.py` | ImportError |
| `import_error/ollama__diabetes__rep1__attempt_02.py` | ImportError |
| `import_error/ollama__diabetes__rep1__attempt_03.py` | ImportError |
| `import_error/ollama__diabetes__rep1__attempt_04.py` | ModuleNotFoundError |
| `import_error/ollama__diabetes__rep1__attempt_05.py` | ModuleNotFoundError |
| `import_error/ollama__diabetes__rep1__attempt_06.py` | ImportError |
| `import_error/ollama__diabetes__rep1__attempt_10.py` | ModuleNotFoundError |
| `import_error/ollama__diabetes__rep1__attempt_12.py` | ModuleNotFoundError |
| `import_error/ollama__diabetes__rep1__attempt_13.py` | ModuleNotFoundError |
| `import_error/ollama__diabetes__rep1__attempt_14.py` | ModuleNotFoundError |
| `import_error/ollama__diabetes__rep1__attempt_17.py` | ModuleNotFoundError |
| `import_error/ollama__diabetes__rep1__attempt_18.py` | ModuleNotFoundError |
| `import_error/ollama__diabetes__rep1__attempt_21.py` | ModuleNotFoundError |
| `import_error/ollama__diabetes__rep1__attempt_23.py` | ImportError |
| `import_error/ollama__diabetes__rep1__attempt_24.py` | ModuleNotFoundError |
| `import_error/ollama__diabetes__rep2__attempt_00.py` | ModuleNotFoundError |
| `import_error/ollama__iris__rep1__attempt_02.py` | ImportError |
| `import_error/ollama__iris__rep2__attempt_00.py` | ModuleNotFoundError |
| `import_error/ollama__iris__rep2__attempt_01.py` | ModuleNotFoundError |
| `import_error/ollama__iris__rep2__attempt_04.py` | ImportError |
| `import_error/ollama__iris__rep2__attempt_07.py` | ImportError |
| `import_error/ollama__iris__rep3__attempt_01.py` | ImportError |
| `import_error/ollama__titanic__rep1__attempt_00.py` | ModuleNotFoundError |
| `import_error/ollama__titanic__rep1__attempt_01.py` | ModuleNotFoundError |
| `import_error/ollama__titanic__rep1__attempt_02.py` | ImportError |
| `import_error/ollama__titanic__rep1__attempt_03.py` | ModuleNotFoundError |
| `import_error/ollama__titanic__rep1__attempt_10.py` | ImportError |
| `import_error/ollama__titanic__rep2__attempt_00.py` | ModuleNotFoundError |
| `import_error/ollama__titanic__rep2__attempt_02.py` | ImportError |
| `import_error/ollama__titanic__rep2__attempt_03.py` | ImportError |
| `import_error/ollama__titanic__rep2__attempt_04.py` | ImportError |
| `import_error/ollama__titanic__rep2__attempt_07.py` | ImportError |
| `import_error/ollama__titanic__rep2__attempt_09.py` | ImportError |
| `import_error/ollama__titanic__rep2__attempt_10.py` | ImportError |
| `import_error/ollama__titanic__rep2__attempt_11.py` | ImportError |
| `import_error/ollama__titanic__rep2__attempt_18.py` | ImportError |
| `import_error/ollama__titanic__rep2__attempt_20.py` | ModuleNotFoundError |
| `import_error/ollama__titanic__rep2__attempt_24.py` | ImportError |
| `import_error/ollama__titanic__rep3__attempt_00.py` | ModuleNotFoundError |
| `import_error/ollama__titanic__rep3__attempt_03.py` | ModuleNotFoundError |
| `import_error/ollama__titanic__rep3__attempt_05.py` | ModuleNotFoundError |
| `import_error/ollama__titanic__rep3__attempt_15.py` | ModuleNotFoundError |
| `import_error/ollama__wine__rep1__attempt_02.py` | ImportError |
| `import_error/ollama__wine__rep2__attempt_01.py` | ModuleNotFoundError |
| `import_error/ollama__wine__rep2__attempt_02.py` | ModuleNotFoundError |
| `import_error/ollama__wine__rep2__attempt_03.py` | ModuleNotFoundError |
| `import_error/ollama__wine__rep2__attempt_05.py` | ModuleNotFoundError |
| `import_error/ollama__wine__rep2__attempt_10.py` | ModuleNotFoundError |
| `import_error/ollama__wine__rep2__attempt_11.py` | ModuleNotFoundError |
| `import_error/ollama__wine__rep2__attempt_12.py` | ModuleNotFoundError |
| `import_error/ollama__wine__rep2__attempt_14.py` | ModuleNotFoundError |
| `import_error/ollama__wine__rep3__attempt_00.py` | ModuleNotFoundError |
| `import_error/ollama__wine__rep3__attempt_01.py` | ModuleNotFoundError |
| `import_error/ollama__wine__rep3__attempt_03.py` | ImportError |
| `import_error/ollama__wine__rep3__attempt_06.py` | ModuleNotFoundError |
| `import_error/ollama__wine__rep3__attempt_07.py` | ModuleNotFoundError |
| `import_error/ollama__wine__rep3__attempt_08.py` | ModuleNotFoundError |
| `import_error/ollama__wine__rep3__attempt_09.py` | ModuleNotFoundError |

## `other_runtime_error/` (45)

| File | Error |
|------|-------|
| `other_runtime_error/ollama2__titanic__rep1__attempt_10.py` | _catboost.CatBoostError |
| `other_runtime_error/ollama__breast_cancer__rep1__attempt_01.py` | UnicodeEncodeError |
| `other_runtime_error/ollama__breast_cancer__rep1__attempt_03.py` | subprocess.CalledProcessError |
| `other_runtime_error/ollama__breast_cancer__rep1__attempt_14.py` | subprocess.CalledProcessError |
| `other_runtime_error/ollama__breast_cancer__rep1__attempt_17.py` | subprocess.CalledProcessError |
| `other_runtime_error/ollama__breast_cancer__rep1__attempt_19.py` | subprocess.CalledProcessError |
| `other_runtime_error/ollama__breast_cancer__rep1__attempt_22.py` | UnicodeEncodeError |
| `other_runtime_error/ollama__breast_cancer__rep1__attempt_24.py` | UnicodeEncodeError |
| `other_runtime_error/ollama__breast_cancer__rep2__attempt_01.py` | UnicodeEncodeError |
| `other_runtime_error/ollama__breast_cancer__rep2__attempt_03.py` | UnicodeEncodeError |
| `other_runtime_error/ollama__breast_cancer__rep2__attempt_13.py` | subprocess.CalledProcessError |
| `other_runtime_error/ollama__breast_cancer__rep2__attempt_20.py` | subprocess.CalledProcessError |
| `other_runtime_error/ollama__breast_cancer__rep2__attempt_22.py` | UnicodeEncodeError |
| `other_runtime_error/ollama__breast_cancer__rep2__attempt_24.py` | UnicodeEncodeError |
| `other_runtime_error/ollama__breast_cancer__rep3__attempt_02.py` | UnicodeEncodeError |
| `other_runtime_error/ollama__breast_cancer__rep3__attempt_04.py` | UnicodeEncodeError |
| `other_runtime_error/ollama__breast_cancer__rep3__attempt_07.py` | UnicodeEncodeError |
| `other_runtime_error/ollama__breast_cancer__rep3__attempt_09.py` | UnicodeEncodeError |
| `other_runtime_error/ollama__breast_cancer__rep3__attempt_10.py` | UnicodeEncodeError |
| `other_runtime_error/ollama__breast_cancer__rep3__attempt_12.py` | sklearn.utils._param_validation.InvalidParameterError |
| `other_runtime_error/ollama__breast_cancer__rep3__attempt_13.py` | UnicodeEncodeError |
| `other_runtime_error/ollama__california__rep1__attempt_06.py` | _pickle.PicklingError |
| `other_runtime_error/ollama__california__rep2__attempt_00.py` | EOFError |
| `other_runtime_error/ollama__diabetes__rep1__attempt_08.py` | UnicodeEncodeError |
| `other_runtime_error/ollama__diabetes__rep1__attempt_11.py` | subprocess.CalledProcessError |
| `other_runtime_error/ollama__diabetes__rep1__attempt_16.py` | If the file lives elsewhere, copy it to that location before running the script. |
| `other_runtime_error/ollama__diabetes__rep1__attempt_19.py` | subprocess.CalledProcessError |
| `other_runtime_error/ollama__iris__rep2__attempt_08.py` | UnicodeEncodeError |
| `other_runtime_error/ollama__titanic__rep1__attempt_07.py` | UnicodeEncodeError |
| `other_runtime_error/ollama__titanic__rep1__attempt_09.py` | UnicodeEncodeError |
| `other_runtime_error/ollama__titanic__rep1__attempt_12.py` | UnicodeEncodeError |
| `other_runtime_error/ollama__titanic__rep2__attempt_08.py` | UnicodeEncodeError |
| `other_runtime_error/ollama__titanic__rep2__attempt_17.py` | UnicodeEncodeError |
| `other_runtime_error/ollama__titanic__rep2__attempt_19.py` | UnicodeEncodeError |
| `other_runtime_error/ollama__titanic__rep2__attempt_21.py` | subprocess.CalledProcessError |
| `other_runtime_error/ollama__titanic__rep2__attempt_22.py` | Please install it (e.g. `pip install -r requirements.txt`). |
| `other_runtime_error/ollama__titanic__rep2__attempt_23.py` | subprocess.CalledProcessError |
| `other_runtime_error/ollama__titanic__rep3__attempt_04.py` | Configuration file not found at C:\Users\Samarth_Ad\Desktop\College\SEM-7\AutoMLAgent\agent_workspace\config.yaml |
| `other_runtime_error/ollama__titanic__rep3__attempt_17.py` | UnicodeDecodeError |
| `other_runtime_error/ollama__wine__rep1__attempt_05.py` | UnicodeEncodeError |
| `other_runtime_error/ollama__wine__rep1__attempt_07.py` | UnicodeEncodeError |
| `other_runtime_error/ollama__wine__rep2__attempt_04.py` | UnicodeEncodeError |
| `other_runtime_error/ollama__wine__rep2__attempt_13.py` | subprocess.CalledProcessError |
| `other_runtime_error/ollama__wine__rep3__attempt_02.py` | subprocess.CalledProcessError |
| `other_runtime_error/ollama__wine__rep3__attempt_04.py` | UnicodeEncodeError |

## `non_termination_timeout/` (37)

| File | Error |
|------|-------|
| `non_termination_timeout/ollama2__breast_cancer__rep1__attempt_00.py` | Execution of ./agent_workspace/exp/ollama2__breast_cancer__rep1_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\Colle |
| `non_termination_timeout/ollama2__breast_cancer__rep1__attempt_01.py` | Execution of ./agent_workspace/exp/ollama2__breast_cancer__rep1_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\Colle |
| `non_termination_timeout/ollama2__breast_cancer__rep2__attempt_01.py` | Execution of ./agent_workspace/exp/ollama2__breast_cancer__rep2_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\Colle |
| `non_termination_timeout/ollama2__breast_cancer__rep2__attempt_02.py` | Execution of ./agent_workspace/exp/ollama2__breast_cancer__rep2_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\Colle |
| `non_termination_timeout/ollama2__breast_cancer__rep2__attempt_03.py` | Execution of ./agent_workspace/exp/ollama2__breast_cancer__rep2_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\Colle |
| `non_termination_timeout/ollama2__breast_cancer__rep2__attempt_04.py` | Execution of ./agent_workspace/exp/ollama2__breast_cancer__rep2_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\Colle |
| `non_termination_timeout/ollama2__breast_cancer__rep2__attempt_06.py` | Execution of ./agent_workspace/exp/ollama2__breast_cancer__rep2_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\Colle |
| `non_termination_timeout/ollama2__california__rep1__attempt_01.py` | Execution of ./agent_workspace/exp/ollama2__california__rep1_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\College\ |
| `non_termination_timeout/ollama2__california__rep1__attempt_02.py` | Execution of ./agent_workspace/exp/ollama2__california__rep1_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\College\ |
| `non_termination_timeout/ollama2__california__rep1__attempt_03.py` | Execution of ./agent_workspace/exp/ollama2__california__rep1_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\College\ |
| `non_termination_timeout/ollama2__california__rep1__attempt_04.py` | Execution of ./agent_workspace/exp/ollama2__california__rep1_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\College\ |
| `non_termination_timeout/ollama2__california__rep2__attempt_00.py` | Execution of ./agent_workspace/exp/ollama2__california__rep2_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\College\ |
| `non_termination_timeout/ollama2__california__rep2__attempt_01.py` | Execution of ./agent_workspace/exp/ollama2__california__rep2_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\College\ |
| `non_termination_timeout/ollama2__california__rep2__attempt_02.py` | Execution of ./agent_workspace/exp/ollama2__california__rep2_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\College\ |
| `non_termination_timeout/ollama2__california__rep2__attempt_03.py` | Execution of ./agent_workspace/exp/ollama2__california__rep2_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\College\ |
| `non_termination_timeout/ollama2__california__rep2__attempt_04.py` | Execution of ./agent_workspace/exp/ollama2__california__rep2_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\College\ |
| `non_termination_timeout/ollama2__diabetes__rep1__attempt_01.py` | Execution of ./agent_workspace/exp/ollama2__diabetes__rep1_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\College\\S |
| `non_termination_timeout/ollama2__diabetes__rep1__attempt_02.py` | Execution of ./agent_workspace/exp/ollama2__diabetes__rep1_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\College\\S |
| `non_termination_timeout/ollama2__diabetes__rep1__attempt_03.py` | Execution of ./agent_workspace/exp/ollama2__diabetes__rep1_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\College\\S |
| `non_termination_timeout/ollama2__diabetes__rep1__attempt_04.py` | Execution of ./agent_workspace/exp/ollama2__diabetes__rep1_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\College\\S |
| `non_termination_timeout/ollama2__diabetes__rep1__attempt_05.py` | Execution of ./agent_workspace/exp/ollama2__diabetes__rep1_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\College\\S |
| `non_termination_timeout/ollama2__diabetes__rep2__attempt_00.py` | Execution of ./agent_workspace/exp/ollama2__diabetes__rep2_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\College\\S |
| `non_termination_timeout/ollama2__diabetes__rep3__attempt_00.py` | Execution of ./agent_workspace/exp/ollama2__diabetes__rep3_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\College\\S |
| `non_termination_timeout/ollama2__diabetes__rep3__attempt_01.py` | Execution of ./agent_workspace/exp/ollama2__diabetes__rep3_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\College\\S |
| `non_termination_timeout/ollama2__iris__rep1__attempt_01.py` | Execution of ./agent_workspace/exp/ollama2__iris__rep1_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\College\\SEM-7 |
| `non_termination_timeout/ollama2__iris__rep2__attempt_00.py` | Execution of ./agent_workspace/exp/ollama2__iris__rep2_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\College\\SEM-7 |
| `non_termination_timeout/ollama2__iris__rep3__attempt_00.py` | Execution of ./agent_workspace/exp/ollama2__iris__rep3_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\College\\SEM-7 |
| `non_termination_timeout/ollama2__iris__rep3__attempt_01.py` | Execution of ./agent_workspace/exp/ollama2__iris__rep3_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\College\\SEM-7 |
| `non_termination_timeout/ollama2__titanic__rep1__attempt_11.py` | Execution of ./agent_workspace/exp/ollama2__titanic__rep1_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\College\\SE |
| `non_termination_timeout/ollama2__titanic__rep1__attempt_12.py` | Execution of ./agent_workspace/exp/ollama2__titanic__rep1_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\College\\SE |
| `non_termination_timeout/ollama2__titanic__rep1__attempt_13.py` | Execution of ./agent_workspace/exp/ollama2__titanic__rep1_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\College\\SE |
| `non_termination_timeout/ollama2__titanic__rep1__attempt_14.py` | Execution of ./agent_workspace/exp/ollama2__titanic__rep1_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\College\\SE |
| `non_termination_timeout/ollama2__wine__rep1__attempt_01.py` | Execution of ./agent_workspace/exp/ollama2__wine__rep1_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\College\\SEM-7 |
| `non_termination_timeout/ollama2__wine__rep3__attempt_00.py` | Execution of ./agent_workspace/exp/ollama2__wine__rep3_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\College\\SEM-7 |
| `non_termination_timeout/ollama__california__rep2__attempt_02.py` | Execution of ./agent_workspace/exp/ollama__california__rep2_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\College\\ |
| `non_termination_timeout/ollama__diabetes__rep2__attempt_07.py` | Execution of ./agent_workspace/exp/ollama__diabetes__rep2_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\College\\SE |
| `non_termination_timeout/ollama__titanic__rep3__attempt_22.py` | Execution of ./agent_workspace/exp/ollama__titanic__rep3_ollama_p3__decomp_ver_.py timed out after 300s: Command '['C:\\Users\\Samarth_Ad\\Desktop\\College\\SEM |

## `file_error/` (33)

| File | Error |
|------|-------|
| `file_error/ollama__breast_cancer__rep1__attempt_11.py` | FileNotFoundError |
| `file_error/ollama__breast_cancer__rep1__attempt_12.py` | FileNotFoundError |
| `file_error/ollama__breast_cancer__rep1__attempt_21.py` | FileNotFoundError |
| `file_error/ollama__breast_cancer__rep1__attempt_23.py` | FileNotFoundError |
| `file_error/ollama__breast_cancer__rep2__attempt_02.py` | FileNotFoundError |
| `file_error/ollama__breast_cancer__rep3__attempt_06.py` | FileNotFoundError |
| `file_error/ollama__california__rep3__attempt_02.py` | FileNotFoundError |
| `file_error/ollama__diabetes__rep1__attempt_07.py` | FileNotFoundError |
| `file_error/ollama__diabetes__rep1__attempt_15.py` | FileNotFoundError |
| `file_error/ollama__diabetes__rep1__attempt_20.py` | FileNotFoundError |
| `file_error/ollama__diabetes__rep1__attempt_22.py` | FileNotFoundError |
| `file_error/ollama__iris__rep1__attempt_00.py` | FileNotFoundError |
| `file_error/ollama__iris__rep2__attempt_02.py` | FileNotFoundError |
| `file_error/ollama__iris__rep2__attempt_03.py` | FileNotFoundError |
| `file_error/ollama__titanic__rep1__attempt_04.py` | FileNotFoundError |
| `file_error/ollama__titanic__rep1__attempt_05.py` | FileNotFoundError |
| `file_error/ollama__titanic__rep1__attempt_06.py` | FileNotFoundError |
| `file_error/ollama__titanic__rep2__attempt_01.py` | FileNotFoundError |
| `file_error/ollama__titanic__rep2__attempt_05.py` | FileNotFoundError |
| `file_error/ollama__titanic__rep2__attempt_06.py` | FileNotFoundError |
| `file_error/ollama__titanic__rep2__attempt_12.py` | FileNotFoundError |
| `file_error/ollama__titanic__rep2__attempt_13.py` | FileNotFoundError |
| `file_error/ollama__titanic__rep2__attempt_14.py` | FileNotFoundError |
| `file_error/ollama__titanic__rep2__attempt_15.py` | FileNotFoundError |
| `file_error/ollama__titanic__rep2__attempt_16.py` | FileNotFoundError |
| `file_error/ollama__titanic__rep3__attempt_01.py` | FileNotFoundError |
| `file_error/ollama__titanic__rep3__attempt_06.py` | FileNotFoundError |
| `file_error/ollama__titanic__rep3__attempt_10.py` | FileNotFoundError |
| `file_error/ollama__titanic__rep3__attempt_12.py` | FileNotFoundError |
| `file_error/ollama__titanic__rep3__attempt_16.py` | FileNotFoundError |
| `file_error/ollama__wine__rep1__attempt_00.py` | FileNotFoundError |
| `file_error/ollama__wine__rep1__attempt_01.py` | FileNotFoundError |
| `file_error/ollama__wine__rep3__attempt_05.py` | FileNotFoundError |

## `wrong_api_usage/` (30)

| File | Error |
|------|-------|
| `wrong_api_usage/ollama2__california__rep1__attempt_00.py` | AttributeError |
| `wrong_api_usage/ollama2__diabetes__rep1__attempt_00.py` | TypeError |
| `wrong_api_usage/ollama2__iris__rep1__attempt_00.py` | TypeError |
| `wrong_api_usage/ollama__breast_cancer__rep2__attempt_04.py` | TypeError |
| `wrong_api_usage/ollama__breast_cancer__rep2__attempt_15.py` | AttributeError |
| `wrong_api_usage/ollama__breast_cancer__rep2__attempt_17.py` | TypeError |
| `wrong_api_usage/ollama__breast_cancer__rep2__attempt_21.py` | TypeError |
| `wrong_api_usage/ollama__breast_cancer__rep3__attempt_08.py` | TypeError |
| `wrong_api_usage/ollama__breast_cancer__rep3__attempt_11.py` | TypeError |
| `wrong_api_usage/ollama__breast_cancer__rep3__attempt_14.py` | TypeError |
| `wrong_api_usage/ollama__california__rep1__attempt_07.py` | TypeError |
| `wrong_api_usage/ollama__california__rep2__attempt_01.py` | TypeError |
| `wrong_api_usage/ollama__california__rep3__attempt_03.py` | TypeError |
| `wrong_api_usage/ollama__california__rep3__attempt_04.py` | TypeError |
| `wrong_api_usage/ollama__california__rep3__attempt_06.py` | TypeError |
| `wrong_api_usage/ollama__california__rep3__attempt_07.py` | TypeError |
| `wrong_api_usage/ollama__diabetes__rep1__attempt_09.py` | AttributeError |
| `wrong_api_usage/ollama__diabetes__rep2__attempt_01.py` | TypeError |
| `wrong_api_usage/ollama__diabetes__rep2__attempt_03.py` | TypeError |
| `wrong_api_usage/ollama__diabetes__rep2__attempt_05.py` | TypeError |
| `wrong_api_usage/ollama__diabetes__rep2__attempt_06.py` | TypeError |
| `wrong_api_usage/ollama__titanic__rep1__attempt_08.py` | TypeError |
| `wrong_api_usage/ollama__titanic__rep3__attempt_11.py` | TypeError |
| `wrong_api_usage/ollama__titanic__rep3__attempt_14.py` | AttributeError |
| `wrong_api_usage/ollama__titanic__rep3__attempt_20.py` | AttributeError |
| `wrong_api_usage/ollama__titanic__rep3__attempt_21.py` | TypeError |
| `wrong_api_usage/ollama__titanic__rep3__attempt_23.py` | TypeError |
| `wrong_api_usage/ollama__titanic__rep3__attempt_24.py` | TypeError |
| `wrong_api_usage/ollama__wine__rep1__attempt_03.py` | AttributeError |
| `wrong_api_usage/ollama__wine__rep1__attempt_04.py` | AttributeError |

## `data_or_value_error/` (26)

| File | Error |
|------|-------|
| `data_or_value_error/ollama2__california__rep3__attempt_00.py` | KeyError |
| `data_or_value_error/ollama2__titanic__rep1__attempt_00.py` | KeyError |
| `data_or_value_error/ollama2__titanic__rep1__attempt_01.py` | KeyError |
| `data_or_value_error/ollama2__titanic__rep1__attempt_02.py` | KeyError |
| `data_or_value_error/ollama2__titanic__rep1__attempt_03.py` | KeyError |
| `data_or_value_error/ollama2__titanic__rep1__attempt_04.py` | KeyError |
| `data_or_value_error/ollama2__titanic__rep1__attempt_05.py` | KeyError |
| `data_or_value_error/ollama2__titanic__rep1__attempt_06.py` | ValueError |
| `data_or_value_error/ollama2__titanic__rep1__attempt_07.py` | ValueError |
| `data_or_value_error/ollama2__titanic__rep1__attempt_08.py` | ValueError |
| `data_or_value_error/ollama2__titanic__rep1__attempt_09.py` | ValueError |
| `data_or_value_error/ollama2__titanic__rep2__attempt_01.py` | KeyError |
| `data_or_value_error/ollama2__titanic__rep3__attempt_00.py` | KeyError |
| `data_or_value_error/ollama__breast_cancer__rep2__attempt_19.py` | KeyError |
| `data_or_value_error/ollama__california__rep3__attempt_05.py` | KeyError |
| `data_or_value_error/ollama__iris__rep1__attempt_01.py` | ValueError |
| `data_or_value_error/ollama__iris__rep2__attempt_05.py` | ValueError |
| `data_or_value_error/ollama__iris__rep2__attempt_06.py` | ValueError |
| `data_or_value_error/ollama__iris__rep3__attempt_00.py` | ValueError |
| `data_or_value_error/ollama__titanic__rep1__attempt_11.py` | KeyError |
| `data_or_value_error/ollama__titanic__rep3__attempt_02.py` | KeyError |
| `data_or_value_error/ollama__titanic__rep3__attempt_07.py` | KeyError |
| `data_or_value_error/ollama__titanic__rep3__attempt_08.py` | ValueError |
| `data_or_value_error/ollama__titanic__rep3__attempt_09.py` | ValueError |
| `data_or_value_error/ollama__titanic__rep3__attempt_13.py` | KeyError |
| `data_or_value_error/ollama__titanic__rep3__attempt_19.py` | KeyError |

## `syntax_error/` (10)

| File | Error |
|------|-------|
| `syntax_error/ollama__breast_cancer__rep1__attempt_20.py` | SyntaxError |
| `syntax_error/ollama__breast_cancer__rep3__attempt_05.py` | SyntaxError |
| `syntax_error/ollama__california__rep1__attempt_00.py` | SyntaxError |
| `syntax_error/ollama__california__rep3__attempt_00.py` | SyntaxError |
| `syntax_error/ollama__titanic__rep3__attempt_18.py` | SyntaxError |
| `syntax_error/ollama__wine__rep2__attempt_00.py` | SyntaxError |
| `syntax_error/ollama__wine__rep2__attempt_06.py` | SyntaxError |
| `syntax_error/ollama__wine__rep2__attempt_07.py` | SyntaxError |
| `syntax_error/ollama__wine__rep2__attempt_08.py` | SyntaxError |
| `syntax_error/ollama__wine__rep2__attempt_09.py` | SyntaxError |

## `runtime_error/` (6)

| File | Error |
|------|-------|
| `runtime_error/ollama__breast_cancer__rep2__attempt_07.py` | RuntimeError |
| `runtime_error/ollama__breast_cancer__rep2__attempt_08.py` | RuntimeError |
| `runtime_error/ollama__breast_cancer__rep2__attempt_23.py` | RuntimeError |
| `runtime_error/ollama__diabetes__rep2__attempt_02.py` | RuntimeError |
| `runtime_error/ollama__diabetes__rep2__attempt_04.py` | RuntimeError |
| `runtime_error/ollama__wine__rep1__attempt_06.py` | RuntimeError |

## `undefined_name/` (1)

| File | Error |
|------|-------|
| `undefined_name/ollama2__wine__rep1__attempt_00.py` | NameError |