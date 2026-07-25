from __future__ import annotations

import subprocess
from pathlib import Path
import sys

# Constantes que apontam para as pastas estruturais do projeto.
# A separação entre BASE, NOTEBOOKS e OUTPUT torna o pipeline
# previsível e facilita ajustes de caminho.
BASE = Path("/Users/elydocarmobarros/Desktop/ESTUDOS TEC/JUPYTER PROJECTS/treinamento_inferencia")
NOTEBOOKS = BASE / "notebooks"
OUTPUT = BASE / "data" / "output"

# A PIPELINE é uma lista ordenada de dicionários.
# Cada dicionário descreve uma etapa: nome do notebook, nome do arquivo
# executado e o arquivo CSV de saída esperado.
# Essa abordagem declarativa permite alterar a ordem ou adicionar novas
# etapas sem mexer na lógica de execução.
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
    """Executa um notebook via jupyter nbconvert.

    O comando gera um novo arquivo .executed.ipynb, que contém
    tanto o código quanto as saídas geradas durante a execução.
    """
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

    print(f"\n>>> Executando: {notebook_name}")
    subprocess.run(cmd, check=True, cwd=str(NOTEBOOKS))
    print(f">>> Concluído: {executed_name}")


def main() -> int:
    """Roda a pipeline completa e valida saídas.

    Para cada etapa, executamos o notebook e verificamos se o CSV
    esperado foi gerado. Se não, o pipeline aborta.
    """
    # Garantimos que a pasta de saída exista antes de começar.
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for step in PIPELINE:
        # A execução do notebook é o núcleo da função.
        run_notebook(step["notebook"], step["executed"])

        # Após a execução, confirmamos que o arquivo de saída existe.
        # Isso protege a integridade do pipeline: uma etapa sem saída
        # indica que o notebook falhou silenciosamente.
        expected = step["expected_file"]
        if expected is not None and not expected.exists():
            raise FileNotFoundError(
                f"Saída esperada não foi gerada: {expected}"
            )

        if expected is not None:
            print(f">>> Saída validada: {expected}")

    print("\nPipeline concluído com sucesso.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # A mensagem de erro explícita facilita o diagnóstico
        # quando o notebook não executa corretamente.
        print(f"\nERRO NO PIPELINE: {exc}")
        raise
