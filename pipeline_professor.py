from __future__ import annotations

import subprocess
from pathlib import Path

# Mesma estrutura do pipeline.py, mas apontando para notebooks
# da versão professor. Isso permite que o instrutor tenha uma
# base de referência separada da versão aluno.
BASE = Path("/Users/elydocarmobarros/Desktop/ESTUDOS TEC/JUPYTER PROJECTS/treinamento_inferencia")
NOTEBOOKS = BASE / "notebooks"
OUTPUT = BASE / "data" / "output"
KERNEL_NAME = "treinamento-inferencia"

# A PIPELINE professor segue a mesma sequência lógica,
# mas com notebooks que já contêm as resoluções esperadas.
# Cada etapa aponta para o notebook professor e para o arquivo
# de saída que ele deve produzir.
PIPELINE = [
    {
        "notebook": "aula_01_coleta_unitarizacao_professor.ipynb",
        "executed": "aula_01_coleta_unitarizacao_professor.executed.ipynb",
        "expected_file": OUTPUT / "aula_01_amostra_unitarizada.csv",
    },
    {
        "notebook": "aula_02_sanitizacao_chauvenet_professor.ipynb",
        "executed": "aula_02_sanitizacao_chauvenet_professor.executed.ipynb",
        "expected_file": OUTPUT / "aula_02_amostra_saneada.csv",
    },
    {
        "notebook": "aula_03_regressao_ols_professor.ipynb",
        "executed": "aula_03_regressao_ols_professor.executed.ipynb",
        "expected_file": OUTPUT / "aula_03_amostra_com_residuos.csv",
    },
    {
        "notebook": "aula_04_diagnosticos_nbr_professor.ipynb",
        "executed": "aula_04_diagnosticos_nbr_professor.executed.ipynb",
        "expected_file": OUTPUT / "relatorio_final_treinamento_inferencial.html",
    },
]


def run_notebook(notebook_name: str, executed_name: str) -> None:
    """Executa um notebook via jupyter nbconvert.

    O comando converte o notebook para um novo arquivo executado,
    mantendo o kernel fixo para garantir reprodutibilidade.
    """
    notebook_path = NOTEBOOKS / notebook_name
    if not notebook_path.exists():
        raise FileNotFoundError(f"Notebook não encontrado: {notebook_path}")

    cmd = [
        "jupyter", "nbconvert",
        "--to", "notebook",
        "--execute",
        f"--ExecutePreprocessor.kernel_name={KERNEL_NAME}",
        str(notebook_path),
        "--output", executed_name,
    ]

    print(f"\n>>> Executando: {notebook_name}")
    subprocess.run(cmd, check=True, cwd=str(NOTEBOOKS))
    print(f">>> Concluído: {executed_name}")


def main() -> int:
    """Roda a pipeline professor e valida saídas.

    A lógica de validação é idêntica ao pipeline.py:
    após cada execução, verificamos se o arquivo esperado existe.
    """
    # Cria a pasta de saída caso ela ainda não exista.
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for step in PIPELINE:
        # Executa o notebook professor.
        run_notebook(step["notebook"], step["executed"])

        # Verifica se o arquivo de saída foi gerado.
        # A ausência indica que a etapa falhou ou não produziu o artefato.
        expected = step["expected_file"]
        if expected is not None and not expected.exists():
            raise FileNotFoundError(
                f"Saída esperada não foi gerada: {expected}"
            )

        if expected is not None:
            print(f">>> Saída validada: {expected}")

    print("\nPipeline professor concluído com sucesso.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        # A mensagem de erro explica claramente que o problema
        # ocorreu na versão professor, facilitando o diagnóstico.
        print(f"\nERRO NO PIPELINE PROFESSOR: {exc}")
        raise
