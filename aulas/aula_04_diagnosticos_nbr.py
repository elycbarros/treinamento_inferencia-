"""Aula 04 - Diagnósticos estatísticos e aderência à NBR 14653.

Script executável da quarta aula do treinamento inferencial.
O objetivo é validar o modelo ajustado por OLS com base em métricas
de significância, normalidade e independência dos resíduos,
organizando um relatório automatizado de aprovação técnica.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd

from servicos.carregamento import load_raw_dataset, resolve_project_root
from servicos.unitarizacao import add_unit_price_column
from servicos.regressao import fit_ols_regression
from servicos.diagnosticos import (
    build_nbr_diagnostics_report,
    format_coefficients_for_display,
    format_diagnostics_for_display,
)

DATASET_CANDIDATES: Final[tuple[str, ...]] = (
    "amostras_residencial35.csv",
    "amostrasresidencial35.csv",
    "amostras_residencial.csv",
)

TARGET_COLUMN: Final[str] = "preco"
PREFERRED_FEATURES: Final[tuple[str, ...]] = (
    "areaprivativa",
    "vagas",
    "distanciacentrokm",
    "dist_praia",
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
    """Imprime um separador visual para a execução da aula."""
    print(f"\n{'=' * 72}")
    print(title)
    print(f"{'=' * 72}")


def print_diagnostics_report(report_df: pd.DataFrame) -> None:
    """Exibe o relatório consolidado de diagnóstico."""
    print_section_title("Relatório de diagnóstico NBR 14653")
    print(format_diagnostics_for_display(report_df).to_string(index=False))


def print_coefficients_report(coeff_df: pd.DataFrame) -> None:
    """Exibe a tabela de significância individual dos coeficientes."""
    print_section_title("Teste t - significância individual")
    print(format_coefficients_for_display(coeff_df).to_string(index=False))


def print_overall_summary(summary: dict[str, object]) -> None:
    """Exibe um resumo executivo da aprovação técnica."""
    print_section_title("Resumo executivo")
    print(f"Status geral do modelo: {summary['status_geral_modelo']}")
    print(
        "Itens aprovados ou normais: "
        f"{summary['itens_aprovados_ou_normais']} de {summary['itens_avaliados']}"
    )
    print(f"Proporção de aprovação: {summary['proporcao_aprovacao']:.2%}")
    print(
        "Coeficientes significativos a 10%: "
        f"{summary['coeficientes_significativos_10pct']} de "
        f"{summary['coeficientes_totais']}"
    )
    print(
        "Observação sobre normalidade: "
        f"{summary['teste_normalidade_observacao']}"
    )
    print(
        "Faixa usada para Durbin-Watson: "
        f"{summary['durbin_watson_faixa']}"
    )
    print(
        "Valor observado de Durbin-Watson: "
        f"{summary['durbin_watson_valor']:.6f}"
    )


def export_outputs(
    project_root: Path,
    diagnostics_report: pd.DataFrame,
    coefficients_report: pd.DataFrame,
    summary: dict[str, object],
) -> None:
    """Exporta os artefatos da Aula 4 para a pasta data/output."""
    # Esta exportação fecha o ciclo do treinamento,
    # preservando o laudo objetivo dos diagnósticos do modelo.
    output_dir = project_root / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    diagnostics_path = output_dir / "aula_04_diagnosticos_nbr.csv"
    coefficients_path = output_dir / "aula_04_significancia_individual.csv"
    summary_path = output_dir / "aula_04_resumo_aprovacao.csv"

    diagnostics_report.to_csv(diagnostics_path, index=False)
    coefficients_report.to_csv(coefficients_path, index=False)
    pd.DataFrame([summary]).to_csv(summary_path, index=False)

    print_section_title("Arquivos exportados")
    print(f"Diagnósticos NBR: {diagnostics_path}")
    print(f"Significância individual: {coefficients_path}")
    print(f"Resumo de aprovação: {summary_path}")


def main(csv_path: Path | None = None) -> None:
    """Executa o fluxo principal da Aula 4."""
    project_root = resolve_project_root()
    dataset_path = csv_path or locate_default_dataset(project_root)

    print_section_title("Aula 04 - Diagnósticos estatísticos")
    print(f"Raiz do projeto: {project_root}")
    print(f"Arquivo utilizado: {dataset_path}")

    # A Aula 4 recebe a base bruta, prepara as variáveis unitárias
    # e então ajusta o modelo para validar seus pressupostos.
    df_raw = load_raw_dataset(dataset_path)
    df_prepared = add_unit_price_column(df_raw)

    artifacts = fit_ols_regression(
        df_prepared,
        target_col=TARGET_COLUMN,
        feature_columns=[col for col in PREFERRED_FEATURES if col in df_prepared.columns],
    )

    diagnostics_report, coefficients_report, summary = build_nbr_diagnostics_report(
        artifacts
    )

    print_diagnostics_report(diagnostics_report)
    print_coefficients_report(coefficients_report)
    print_overall_summary(summary)
    export_outputs(project_root, diagnostics_report, coefficients_report, summary)


if __name__ == "__main__":
    main()
