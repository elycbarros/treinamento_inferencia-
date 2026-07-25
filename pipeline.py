from __future__ import annotations

import subprocess
from pathlib import Path
import sys

BASE = Path("/Users/elydocarmobarros/Desktop/ESTUDOS TEC/JUPYTER PROJECTS/treinamento_inferencia")
NOTEBOOKS = BASE / "notebooks"
OUTPUT = BASE / "data" / "output"

PIPELINE = [
    {
        "notebook": "aula_01_coleta_unitarizacao.ipynb",
        "executed": "aula_01_coleta_unitarizacao.executed.ipynb",
        "expected_file": OUTPUT / "aula_01_amostra_unitarizada.csv",
    },
    {
        "notebook": "aula_02_sanitizacao_chauvenet.ipynb",
        "executed": "aula_02_sanitizacao_chauvenet.executed.ipynb",
        "expected_file": OUTPUT / "aula_02_amostra_saneada.csv",
    },
    {
        "notebook": "aula_03_regressao_ols.ipynb",
        "executed": "aula_03_regressao_ols.executed.ipynb",
        "expected_file": OUTPUT / "aula_03_amostra_com_residuos.csv",
    },
    {
        "notebook": "aula_04_diagnosticos_nbr.ipynb",
        "executed": "aula_04_diagnosticos_nbr.executed.ipynb",
        "expected_file": None,
    },
]

def run_notebook(notebook_name: str, executed_name: str) -> None:
    notebook_path = NOTEBOOKS / notebook_name
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook não encontrado: {notebook_path}")

    cmd = [
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        "--ExecutePreprocessor.kernel_name=treinamento-inferencia",
        str(notebook_path),
        "--output", executed_name,
    ]

    print(f"\\n>>> Executando: {notebook_name}")
    subprocess.run(cmd, check=True, cwd=str(NOTEBOOKS))
    print(f">>> Concluído: {executed_name}")

def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for step in PIPELINE:
        run_notebook(step["notebook"], step["executed"])

        expected = step["expected_file"]
        if expected is not None and not expected.exists():
            raise FileNotFoundError(
                f"Saída esperada não foi gerada: {expected}"
            )

        if expected is not None:
            print(f">>> Saída validada: {expected}")

    print("\\nPipeline concluído com sucesso.")
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"\\nERRO NO PIPELINE: {exc}")
        raise
