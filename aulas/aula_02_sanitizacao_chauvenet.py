"""Aula 2 - Sanitização estatística da amostra com o critério de Chauvenet.

Esta aula mostra como a série unitarizada pode ser refinada por meio
da exclusão justificada de observações discrepantes. O foco didático
está em tornar visível o que foi removido, em que rodada e por quê.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from servicos.carregamento import load_raw_dataset, resolve_project_root
from servicos.sanitizacao import build_sanitization_report, sanitize_sample_chauvenet
from servicos.unitarizacao import add_unit_price_column


def locate_default_dataset(project_root: Path) -> Path:
    """Localiza o CSV padrão usado na Aula 2."""
    return project_root / "data" / "amostras_imoveis.csv"


def print_section_title(title: str) -> None:
    """Imprime um título de seção no terminal."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_sample_overview(df: pd.DataFrame) -> None:
    """Mostra visão geral da amostra antes da sanitização."""
    print(f"Total de amostras: {len(df)}")
    print(f"Média do valor unitário: R$ {df['valor_unitario'].mean():.2f}/m²")
    print(f"Mediana do valor unitário: R$ {df['valor_unitario'].median():.2f}/m²")
    print(
        "Desvio-padrão do valor unitário: "
        f"R$ {df['valor_unitario'].std(ddof=1):.2f}/m²"
    )


def print_iteration_history(history_df: pd.DataFrame) -> None:
    """Exibe o histórico iterativo do critério de Chauvenet."""
    # Esta tabela é pedagogicamente importante porque mostra
    # como o saneamento evolui rodada a rodada.
    if history_df.empty:
        print("Nenhum histórico de sanitização disponível.")
        return
    print(history_df.to_string(index=False))


def print_removed_rows(removed_df: pd.DataFrame) -> None:
    """Mostra as amostras removidas durante a sanitização."""
    # Exibir explicitamente as linhas removidas ajuda o aluno
    # a entender que a sanitização não é uma “caixa-preta”.
    if removed_df.empty:
        print("Nenhuma amostra foi removida.")
        return
    print(removed_df.to_string(index=False))


def print_cleaned_summary(clean_df: pd.DataFrame) -> None:
    """Exibe resumo da amostra saneada."""
    print(f"Total de amostras após saneamento: {len(clean_df)}")
    print(f"Média saneada: R$ {clean_df['valor_unitario'].mean():.2f}/m²")
    print(f"Mediana saneada: R$ {clean_df['valor_unitario'].median():.2f}/m²")
    print(
        f"Desvio-padrão saneado: R$ {clean_df['valor_unitario'].std(ddof=1):.2f}/m²"
    )


def print_final_report(report: dict[str, object]) -> None:
    """Exibe o consolidado final da sanitização."""
    print(f"Total original: {report['total_original']}")
    print(f"Total saneado: {report['total_saneado']}")
    print(f"Total removido: {report['total_removido']}")
    print(f"Percentual removido: {report['percentual_removido']:.2f}%")


def export_outputs(
    project_root: Path,
    clean_df: pd.DataFrame,
    removed_df: pd.DataFrame,
    history_df: pd.DataFrame,
) -> None:
    """Exporta os artefatos gerados pela Aula 2."""
    # A exportação preserva rastreabilidade e permite que
    # as etapas seguintes trabalhem sobre a amostra já saneada.
    output_dir = project_root / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    clean_df.to_csv(output_dir / "aula_02_amostra_saneada.csv", index=False)
    removed_df.to_csv(output_dir / "aula_02_amostras_removidas.csv", index=False)
    history_df.to_csv(output_dir / "aula_02_historico_chauvenet.csv", index=False)


def main(csv_path: Path | None = None) -> None:
    """Executa o fluxo completo da Aula 2."""
    project_root = resolve_project_root()
    dataset_path = csv_path or locate_default_dataset(project_root)

    print_section_title("Aula 2 - Leitura e unitarização da base")
    raw_df = load_raw_dataset(dataset_path)
    unitized_df = add_unit_price_column(raw_df)
    print_sample_overview(unitized_df)

    # Nesta etapa aplicamos o saneamento iterativo sobre o valor unitário.
    clean_df, removed_df, history_df = sanitize_sample_chauvenet(unitized_df)

    print_section_title("Aula 2 - Histórico do critério de Chauvenet")
    print_iteration_history(history_df)

    print_section_title("Aula 2 - Amostras removidas")
    print_removed_rows(removed_df)

    print_section_title("Aula 2 - Resumo da amostra saneada")
    print_cleaned_summary(clean_df)

    report = build_sanitization_report(
        original_df=unitized_df,
        clean_df=clean_df,
        removed_df=removed_df,
        history_df=history_df,
    )

    print_section_title("Aula 2 - Consolidado final")
    print_final_report(report)

    export_outputs(project_root, clean_df, removed_df, history_df)


if __name__ == "__main__":
    main()
