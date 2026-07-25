"""Orquestrador principal do treinamento inferencial.

Este script centraliza a execução das etapas do projeto,
permitindo rodar aulas isoladas ou o fluxo completo do treinamento.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from servicos.carregamento import resolve_project_root


def locate_default_dataset(project_root: Path) -> Path:
    """Localiza a base padrão dentro da pasta data.

    Em contexto didático, o aluno pode não saber o nome exato do CSV.
    Por isso, tentamos uma lista de candidatos comuns no projeto.
    """
    # Os candidatos são nomes típicos usados em execuções passadas.
    # A função retorna o primeiro que existir, facilitando o uso padrão.
    candidates = (
        "amostras_residencial35.csv",
        "amostrasresidencial35.csv",
        "amostras_residencial.csv",
    )

    for filename in candidates:
        candidate = project_root / "data" / filename
        if candidate.exists():
            return candidate

    searched = ", ".join(candidates)
    raise FileNotFoundError(
        "Nenhum dataset padrão foi encontrado na pasta `data`. "
        f"Arquivos procurados: {searched}."
    )


def run_aula_1(dataset_path: Path | None = None) -> None:
    """Executa a Aula 1.

    A Aula 1 introduz coleta e unitarização de dados,
    transformando o CSV bruto em uma amostra pronta para análise.
    """
    # Importação tardia evita que o Python carregue o módulo
    # antes de a função ser de fato invocada.
    from aulas.aula_01_coleta_unitarizacao import main as aula_01_main

    aula_01_main(dataset_path)


def run_aula_2(dataset_path: Path | None = None) -> None:
    """Executa a Aula 2.

    A Aula 2 trabalha a sanitização estatística da amostra
    usando o critério de Chauvenet de forma iterativa.
    """
    from aulas.aula_02_sanitizacao_chauvenet import main as aula_02_main

    aula_02_main(dataset_path)


def run_aula_3(dataset_path: Path | None = None) -> None:
    """Executa a Aula 3.

    A Aula 3 aplica a regressão linear por mínimos quadrados (OLS)
    e produz os coeficientes, resíduos e métricas de ajuste.
    """
    from aulas.aula_03_regressao_ols import main as aula_03_main

    aula_03_main(dataset_path)


def run_aula_4(dataset_path: Path | None = None) -> None:
    """Executa a Aula 4.

    A Aula 4 realiza os diagnósticos estatísticos do modelo:
    poder explicativo, significância, normalidade e independência dos resíduos.
    """
    from aulas.aula_04_diagnosticos_nbr import main as aula_04_main

    aula_04_main(dataset_path)


def run_relatorio() -> None:
    """Executa a geração do relatório final consolidado.

    O relatório resume os resultados das quatro aulas
    em um documento único para revisão do aluno.
    """
    from relatorios.gerador_apostila import main as relatorio_main

    relatorio_main()


def run_full_training(dataset_path: Path | None = None) -> None:
    """Executa a trilha completa do treinamento.

    Chama as quatro aulas em sequência e, ao final, gera o relatório.
    Essa função é o ponto de entrada para quem deseja executar tudo de uma vez.
    """
    # A separação visual com linhas ajuda o aluno a perceber
    # o início de uma execução longa no terminal.
    print("\n" + "=" * 72)
    print("Execução completa do treinamento inferencial")
    print("=" * 72)

    run_aula_1(dataset_path)
    run_aula_2(dataset_path)
    run_aula_3(dataset_path)
    run_aula_4(dataset_path)
    run_relatorio()


def build_parser() -> argparse.ArgumentParser:
    """Cria o parser de argumentos do CLI principal.

    O CLI é uma porta de entrada simples para quem prefere
    executar o projeto diretamente do terminal, sem abrir notebooks.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Orquestra a execução das aulas e relatórios do projeto "
            "treinamento_inferencia."
        )
    )

    # O argumento posicional etapa define qual parte do treinamento será executada.
    parser.add_argument(
        "etapa",
        choices=[
            "aula1",
            "aula2",
            "aula3",
            "aula4",
            "relatorio",
            "completo",
        ],
        help="Etapa do treinamento a ser executada.",
    )

    # O argumento opcional --dataset permite apontar para um CSV diferente
    # do padrão, útil quando o aluno quer testar outras bases.
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Caminho opcional para o dataset CSV de entrada.",
    )

    return parser


def main() -> None:
    """Ponto de entrada principal do projeto.

    Aqui ocorre a leitura dos argumentos, a resolução automática
    do dataset padrão e o despacho para a função correta.
    """
    parser = build_parser()
    args = parser.parse_args()

    # resolve_project_root() sabe encontrar a raiz do projeto
    # mesmo se o script for executado de uma subpasta.
    project_root = resolve_project_root()
    dataset_path = args.dataset

    # Se o usuário não informou um dataset, tentamos localizar o padrão.
    # O relatório não depende de dataset, então pulamos essa etapa para ele.
    if args.etapa != "relatorio" and dataset_path is None:
        dataset_path = locate_default_dataset(project_root)

    # O despacho usa o valor da etapa para chamar a função apropriada.
    # Cada etapa imprime sua própria saída, deixando o terminal organizado.
    if args.etapa == "aula1":
        run_aula_1(dataset_path)
    elif args.etapa == "aula2":
        run_aula_2(dataset_path)
    elif args.etapa == "aula3":
        run_aula_3(dataset_path)
    elif args.etapa == "aula4":
        run_aula_4(dataset_path)
    elif args.etapa == "relatorio":
        run_relatorio()
    elif args.etapa == "completo":
        run_full_training(dataset_path)


if __name__ == "__main__":
    main()
