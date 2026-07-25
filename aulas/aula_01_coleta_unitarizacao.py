"""Aula 1 - Coleta, inspeção inicial e unitarização da amostra.

Esta aula apresenta a transição entre a base bruta de mercado e a série
unitarizada em R$/m². O foco didático é mostrar que a análise inferencial
começa pela organização correta da amostra, e não apenas pela regressão.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from servicos.carregamento import load_raw_dataset, resolve_project_root
from servicos.unitarizacao import build_unitization_report


def locate_default_dataset(project_root: Path) -> Path:
    """Localiza o CSV padrão usado na Aula 1."""
    return project_root / "data" / "amostras_imoveis.csv"


def setup_plot_style() -> None:
    """Configura o estilo visual dos gráficos exploratórios."""
    # A padronização visual facilita leitura em aula e em notebook.
    sns.set_theme(style="whitegrid", context="talk")


def print_section_title(title: str) -> None:
    """Imprime um título de seção no terminal."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def print_dataset_preview(df: pd.DataFrame, rows: int = 5) -> None:
    """Mostra as primeiras linhas da base."""
    # Esta prévia ajuda o aluno a enxergar a estrutura original
    # antes de qualquer transformação estatística.
    print(df.head(rows).to_string(index=False))


def print_report(report: dict[str, object]) -> None:
    """Exibe os principais resultados da unitarização."""
    print(f"Total de amostras: {report['total_amostras']}")
    print(f"Média do valor unitário: R$ {report['media_valor_unitario']:.2f}/m²")
    print(f"Mediana do valor unitário: R$ {report['mediana_valor_unitario']:.2f}/m²")
    print(
        "Desvio-padrão do valor unitário: "
        f"R$ {report['desvio_padrao_valor_unitario']:.2f}/m²"
    )
    print(
        "Coeficiente de variação: "
        f"{report['coeficiente_variacao_percentual']:.2f}%"
    )
    print("\nResumo estatístico:")
    print(report["resumo_estatistico"].to_string())


def plot_exploratory_charts(df: pd.DataFrame) -> None:
    """Gera gráficos exploratórios básicos da amostra unitarizada."""
    # Os gráficos desta aula têm função de leitura inicial:
    # distribuição do valor unitário, relação preço-área e dispersões básicas.
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.histplot(df["valor_unitario"], kde=True, ax=axes[0], color="steelblue")
    axes[0].set_title("Distribuição do valor unitário")
    axes[0].set_xlabel("R$/m²")

    sns.scatterplot(
        data=df,
        x="areaprivativa",
        y="preco",
        ax=axes[1],
        color="darkgreen",
    )
    axes[1].set_title("Preço vs área privativa")
    axes[1].set_xlabel("Área privativa (m²)")
    axes[1].set_ylabel("Preço (R$)")

    sns.boxplot(y=df["valor_unitario"], ax=axes[2], color="salmon")
    axes[2].set_title("Boxplot do valor unitário")
    axes[2].set_ylabel("R$/m²")

    plt.tight_layout()
    plt.show()


def main(csv_path: Path | None = None) -> None:
    """Executa o fluxo completo da Aula 1."""
    project_root = resolve_project_root()
    dataset_path = csv_path or locate_default_dataset(project_root)

    print_section_title("Aula 1 - Leitura da base bruta")
    raw_df = load_raw_dataset(dataset_path)
    print_dataset_preview(raw_df)

    # A partir da base bruta, montamos um relatório de unitarização
    # que já inclui DataFrame transformado e indicadores descritivos.
    report = build_unitization_report(raw_df)
    unitized_df = report["dataframe_unitarizado"]

    print_section_title("Aula 1 - Resultados da unitarização")
    print_report(report)

    print_section_title("Aula 1 - Gráficos exploratórios")
    plot_exploratory_charts(unitized_df)


if __name__ == "__main__":
    main()
