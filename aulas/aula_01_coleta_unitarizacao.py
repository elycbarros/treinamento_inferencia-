"""Aula 01 - Coleta e unitarização.

Script executável da primeira aula do treinamento inferencial.
O objetivo é carregar a base, validar sua estrutura mínima,
unitarizar os preços em R$/m² e apresentar um diagnóstico
estatístico inicial da série de mercado.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from servicos.carregamento import load_raw_dataset, resolve_project_root
from servicos.unitarizacao import (
    UNIT_PRICE_COLUMN,
    build_unitization_report,
)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ImportError:
    plt = None
    sns = None


DATASET_CANDIDATES: Final[tuple[str, ...]] = (
    "amostras_residencial35.csv",
    "amostrasresidencial35.csv",
    "amostras_residencial.csv",
)


def locate_default_dataset(project_root: Path) -> Path:
    """Localiza automaticamente a base padrão da Aula 1."""
    data_dir = project_root / "data"

    for filename in DATASET_CANDIDATES:
        candidate = data_dir / filename
        if candidate.exists():
            return candidate

    searched = ", ".join(DATASET_CANDIDATES)
    raise FileNotFoundError(
        "Nenhum arquivo padrão da Aula 1 foi encontrado na pasta `data`. "
        f"Arquivos procurados: {searched}."
    )


def setup_plot_style() -> None:
    """Configura um estilo visual limpo para gráficos didáticos."""
    if plt is None or sns is None:
        return

    sns.set_theme(style="whitegrid", palette="muted")
    plt.rcParams["figure.figsize"] = (10, 5)


def print_section_title(title: str) -> None:
    """Imprime um título padronizado de seção no terminal."""
    print(f"\n{'=' * 72}")
    print(title)
    print(f"{'=' * 72}")


def print_dataset_preview(df: pd.DataFrame, rows: int = 5) -> None:
    """Exibe uma prévia textual da base carregada."""
    print_section_title("Prévia da base carregada")
    print(df.head(rows).to_string(index=False))


def print_report(report: dict[str, object]) -> None:
    """Exibe o relatório sintético de unitarização da Aula 1."""
    print_section_title("Resumo estatístico inicial")
    print(report["resumo_estatistico"])

    print_section_title("Diagnóstico de homogeneidade inicial")
    print(f"Total de amostras: {report['total_amostras']}")
    print(
        "Coeficiente de variação do valor unitário bruto: "
        f"{report['coeficiente_variacao_percentual']}%"
    )


def plot_exploratory_charts(df: pd.DataFrame) -> None:
    """Gera gráficos exploratórios semelhantes aos usados no notebook."""
    if plt is None or sns is None:
        print(
            "\nAviso: matplotlib/seaborn não estão instalados. "
            "Os gráficos exploratórios foram ignorados."
        )
        return

    setup_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.scatterplot(
        data=df,
        x="areaprivativa",
        y="preco",
        hue="vagas",
        palette="viridis",
        s=100,
        ax=axes[0],
    )
    axes[0].set_title("Preço total vs. área privativa")
    axes[0].set_xlabel("Área privativa (m²)")
    axes[0].set_ylabel("Preço total (R$)")

    sns.boxplot(y=df[UNIT_PRICE_COLUMN], ax=axes[1], color="lightblue")
    sns.stripplot(y=df[UNIT_PRICE_COLUMN], ax=axes[1], color="red", size=6, jitter=True)
    axes[1].set_title("Distribuição do valor unitário")
    axes[1].set_ylabel("Valor unitário (R$/m²)")

    plt.tight_layout()
    plt.show()


def main(csv_path: Path | None = None) -> None:
    """Executa o fluxo principal da Aula 1."""
    project_root = resolve_project_root()
    dataset_path = csv_path or locate_default_dataset(project_root)

    print_section_title("Aula 01 - Coleta e unitarização")
    print(f"Raiz do projeto: {project_root}")
    print(f"Arquivo utilizado: {dataset_path}")

    df_raw = load_raw_dataset(dataset_path)
    report = build_unitization_report(df_raw)
    df_unitized = report["dataframe_unitarizado"]

    print_dataset_preview(df_unitized)
    print_report(report)
    plot_exploratory_charts(df_unitized)


if __name__ == "__main__":
    main()
