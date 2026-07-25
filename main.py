"""Orquestrador principal do treinamento inferencial.

Este script centraliza a execução das etapas do projeto,
permitindo rodar aulas isoladas ou o fluxo completo do treinamento.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from servicos.carregamento import resolve_project_root


def locate_default_dataset(project_root: Path) -> Path:
    """Localiza a base padrão dentro da pasta data."""
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
    """Executa a Aula 1."""
    from aulas.aula_01_coleta_unitarizacao import main as aula_01_main

    aula_01_main(dataset_path)


def run_aula_2(dataset_path: Path | None = None) -> None:
    """Executa a Aula 2."""
    from aulas.aula_02_sanitizacao_chauvenet import main as aula_02_main

    aula_02_main(dataset_path)


def run_aula_3(dataset_path: Path | None = None) -> None:
    """Executa a Aula 3."""
    from aulas.aula_03_regressao_ols import main as aula_03_main

    aula_03_main(dataset_path)


def run_aula_4(dataset_path: Path | None = None) -> None:
    """Executa a Aula 4."""
    from aulas.aula_04_diagnosticos_nbr import main as aula_04_main

    aula_04_main(dataset_path)


def run_relatorio() -> None:
    """Executa a geração do relatório final consolidado."""
    from relatorios.gerador_apostila import main as relatorio_main

    relatorio_main()


def run_full_training(dataset_path: Path | None = None) -> None:
    """Executa a trilha completa do treinamento."""
    print("\n" + "=" * 72)
    print("Execução completa do treinamento inferencial")
    print("=" * 72)

    run_aula_1(dataset_path)
    run_aula_2(dataset_path)
    run_aula_3(dataset_path)
    run_aula_4(dataset_path)
    run_relatorio()


def build_parser() -> argparse.ArgumentParser:
    """Cria o parser de argumentos do CLI principal."""
    parser = argparse.ArgumentParser(
        description=(
            "Orquestra a execução das aulas e relatórios do projeto "
            "treinamento_inferencia."
        )
    )

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

    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Caminho opcional para o dataset CSV de entrada.",
    )

    return parser


def main() -> None:
    """Ponto de entrada principal do projeto."""
    parser = build_parser()
    args = parser.parse_args()

    project_root = resolve_project_root()
    dataset_path = args.dataset

    if args.etapa != "relatorio" and dataset_path is None:
        dataset_path = locate_default_dataset(project_root)

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
