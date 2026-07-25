"""Aula 02 - Sanitização amostral por Chauvenet.

Script executável da segunda aula do treinamento inferencial.
O objetivo é carregar a amostra já unitarizada, aplicar o
critério de Chauvenet de forma iterativa e registrar, com
transparência, quais observações foram removidas da série.

Entregáveis didáticos:
- base com valor unitário calculado;
- amostra saneada;
- tabela de removidos;
- histórico de iterações da sanitização.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from servicos.carregamento import load_raw_dataset, resolve_project_root
from servicos.unitarizacao import UNIT_PRICE_COLUMN, add_unit_price_column
from servicos.sanitizacao import build_sanitization_report

DATASET_CANDIDATES: Final[tuple[str, ...]] = (
    "amostras_residencial35.csv",
    "amostrasresidencial35.csv",
    "amostras_residencial.csv",
)


def locate_default_dataset(project_root: Path) -> Path:
    """Localiza automaticamente a base padrão das aulas iniciais."""
    data_dir = project_root / "data"

    for filename in DATASET_CANDIDATES:
        candidate = data_dir / filename
        if candidate.exists():
            return candidate

    searched = ", ".join(DATASET_CANDIDATES)
    raise FileNotFoundError(
        "Nenhum arquivo padrão foi encontrado na pasta `data`. "
        f"Arquivos procurados: {searched}."
    )


def print_section_title(title: str) -> None:
    """Imprime um separador visual para as seções da aula."""
    print(f"\n{'=' * 72}")
    print(title)
    print(f"{'=' * 72}")


def print_sample_overview(df: pd.DataFrame) -> None:
    """Exibe uma visão geral inicial da amostra antes da sanitização."""
    print_section_title("Amostra inicial")
    print(f"Quantidade de registros: {len(df)}")

    preview_columns = [
        column
        for column in ("id", "preco", "areaprivativa", UNIT_PRICE_COLUMN)
        if column in df.columns
    ]
    print(df[preview_columns].head(8).to_string(index=False))


def print_iteration_history(history_df: pd.DataFrame) -> None:
    """Exibe o histórico consolidado das iterações de Chauvenet."""
    print_section_title("Histórico de iterações")
    print(history_df.to_string(index=False))


def print_removed_rows(removed_df: pd.DataFrame) -> None:
    """Exibe as observações removidas durante a sanitização."""
    print_section_title("Observações removidas")

    if removed_df.empty:
        print("Nenhuma observação foi removida pelo critério de Chauvenet.")
        return

    preferred_columns = [
        column
        for column in (
            "iteracao_remocao",
            "id",
            "preco",
            "areaprivativa",
            UNIT_PRICE_COLUMN,
            "z_score",
            "z_critical",
        )
        if column in removed_df.columns
    ]
    print(removed_df[preferred_columns].to_string(index=False))


def print_cleaned_summary(clean_df: pd.DataFrame) -> None:
    """Exibe estatísticas da amostra saneada."""
    print_section_title("Amostra saneada")
    print(f"Quantidade final de registros: {len(clean_df)}")

    numeric_columns = [
        column
        for column in ("preco", "areaprivativa", UNIT_PRICE_COLUMN)
        if column in clean_df.columns
    ]
    if numeric_columns:
        print(clean_df[numeric_columns].describe().round(2).to_string())


def print_final_report(report: dict[str, object]) -> None:
    """Exibe um resumo executivo da Aula 2."""
    print_section_title("Resumo executivo da sanitização")
    print(f"Amostra inicial: {report['amostra_inicial']}")
    print(f"Amostra saneada: {report['amostra_saneada']}")
    print(f"Total removido: {report['total_removido']}")

    if report["total_removido"] == 0:
        print("Resultado: a amostra permaneceu íntegra após o teste de Chauvenet.")
    else:
        print(
            "Resultado: foram identificadas observações discrepantes "
            "compatíveis com remoção estatística pelo critério de Chauvenet."
        )


def export_outputs(
    project_root: Path,
    report: dict[str, object],
) -> None:
    """Exporta os principais artefatos da aula para a pasta data/output."""
    output_dir = project_root / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = output_dir / "aula_02_amostra_saneada.csv"
    removed_path = output_dir / "aula_02_amostras_removidas.csv"
    history_path = output_dir / "aula_02_historico_chauvenet.csv"

    clean_df = report["df_saneado"]
    removed_df = report["df_removidos"]
    history_df = report["historico_iteracoes"]

    clean_df.to_csv(cleaned_path, index=False)

    if isinstance(removed_df, pd.DataFrame) and not removed_df.empty:
        removed_df.to_csv(removed_path, index=False)

    history_df.to_csv(history_path, index=False)

    print_section_title("Arquivos exportados")
    print(f"Amostra saneada: {cleaned_path}")
    print(f"Histórico de iterações: {history_path}")
    if isinstance(removed_df, pd.DataFrame) and not removed_df.empty:
        print(f"Observações removidas: {removed_path}")
    else:
        print("Observações removidas: nenhuma exportação, pois não houve exclusões.")


def main(csv_path: Path | None = None) -> None:
    """Executa o fluxo principal da Aula 2."""
    project_root = resolve_project_root()
    dataset_path = csv_path or locate_default_dataset(project_root)

    print_section_title("Aula 02 - Sanitização amostral por Chauvenet")
    print(f"Raiz do projeto: {project_root}")
    print(f"Arquivo utilizado: {dataset_path}")

    df_raw = load_raw_dataset(dataset_path)
    df_unitized = add_unit_price_column(df_raw)

    print_sample_overview(df_unitized)

    report = build_sanitization_report(
        df=df_unitized,
        target_col=UNIT_PRICE_COLUMN,
        id_col="id",
    )

    print_iteration_history(report["historico_iteracoes"])
    print_removed_rows(report["df_removidos"])
    print_cleaned_summary(report["df_saneado"])
    print_final_report(report)
    export_outputs(project_root, report)


if __name__ == "__main__":
    main()
