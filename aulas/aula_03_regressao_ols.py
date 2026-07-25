"""Aula 3 - Ajuste de regressão linear múltipla por OLS.

Esta aula apresenta a transição entre a amostra saneada e o modelo
inferencial propriamente dito. O foco didático está em mostrar como
as variáveis explicativas entram no modelo e como interpretar o ajuste.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from servicos.carregamento import load_raw_dataset, resolve_project_root
from servicos.regressao import build_regression_report
from servicos.sanitizacao import sanitize_sample_chauvenet
from servicos.unitarizacao import add_unit_price_column


def locate_default_dataset(project_root: Path) -> Path:
    """Localiza o CSV padrão usado na Aula 3."""
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


def print_section_title(title: str) -> None:
    """Imprime um título de seção no terminal."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def build_regression_base(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Prepara a base regressiva a partir da base bruta."""
    # A base regressiva nasce da sequência lógica do projeto:
    # leitura, unitarização e depois sanitização da amostra.
    unitized_df = add_unit_price_column(raw_df)
    clean_df, _, _ = sanitize_sample_chauvenet(unitized_df)
    return clean_df


def print_base_overview(df: pd.DataFrame) -> None:
    """Mostra visão geral da base que entrará no modelo."""
    print(f"Total de amostras para regressão: {len(df)}")
    print(f"Preço médio: R$ {df['preco'].mean():.2f}")
    print(f"Área privativa média: {df['areaprivativa'].mean():.2f} m²")

    if "vagas" in df.columns:
        print(f"Vagas médias: {df['vagas'].mean():.2f}")

    if "distanciacentrokm" in df.columns:
        print(f"Distância média ao centro: {df['distanciacentrokm'].mean():.2f} km")


def print_model_metrics(report: dict[str, object]) -> None:
    """Exibe as métricas centrais do ajuste OLS."""
    print(f"Número de observações: {report['n_observacoes']}")
    print(f"Número de parâmetros: {report['n_parametros']}")
    print(f"R²: {report['r_quadrado']:.6f}")
    print(f"R² ajustado: {report['r_quadrado_ajustado']:.6f}")
    print(f"RMSE: {report['rmse']:.6f}")
    print(f"Estatística F: {report['estatistica_f']:.6f}")
    print(f"p-valor de F: {report['p_valor_f']:.6f}")


def print_regression_summary(report: dict[str, object]) -> None:
    """Mostra a tabela resumo dos coeficientes."""
    # Esta tabela permite discutir sinal, magnitude e significância
    # de cada variável explicativa no contexto do modelo.
    summary_df = report["resumo_regressao"]
    print(summary_df.to_string())


def print_residual_preview(report: dict[str, object], rows: int = 10) -> None:
    """Mostra uma prévia da base com estimados e resíduos."""
    regression_df = report["dataframe_regressao"]
    available_columns = [
        column
        for column in [
            "preco",
            "valor_estimado",
            "residuo",
            "residuo_percentual",
            "areaprivativa",
            "vagas",
            "distanciacentrokm",
        ]
        if column in regression_df.columns
    ]
    print(regression_df.loc[:, available_columns].head(rows).to_string(index=False))


def export_outputs(project_root: Path, report: dict[str, object]) -> None:
    """Exporta os artefatos gerados pela Aula 3."""
    # A exportação desta aula prepara a base para os diagnósticos
    # posteriores e preserva evidências do ajuste realizado.
    output_dir = project_root / "data" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    report["dataframe_regressao"].to_csv(
        output_dir / "aula_03_amostra_com_residuos.csv",
        index=False,
    )
    report["resumo_regressao"].to_csv(
        output_dir / "aula_03_resumo_regressao.csv"
    )
    report["coeficientes"].to_csv(
        output_dir / "aula_03_coeficientes.csv",
        header=True,
    )


def main(csv_path: Path | None = None) -> None:
    """Executa o fluxo completo da Aula 3."""
    project_root = resolve_project_root()
    dataset_path = csv_path or locate_default_dataset(project_root)

    print_section_title("Aula 3 - Leitura e preparação da base")
    raw_df = load_raw_dataset(dataset_path)
    regression_base = build_regression_base(raw_df)
    print_base_overview(regression_base)

    # Nesta etapa, a base saneada alimenta o ajuste do modelo OLS.
    report = build_regression_report(regression_base)

    print_section_title("Aula 3 - Métricas do modelo")
    print_model_metrics(report)

    print_section_title("Aula 3 - Resumo dos coeficientes")
    print_regression_summary(report)

    print_section_title("Aula 3 - Prévia dos resíduos")
    print_residual_preview(report)

    export_outputs(project_root, report)


if __name__ == "__main__":
    main()
