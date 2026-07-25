"""Serviços de unitarização e preparo inicial da amostra.

Este módulo contém funções que traduzem a lógica da Aula 1:
- calcular valor unitário;
- gerar amostra unitarizada;
- resumir estatisticamente a série;
- medir homogeneidade inicial por coeficiente de variação.

A intenção é que o aluno entenda que unitarizar não é apenas dividir
preço por área, mas preparar uma série comparável para inferência.
"""

from typing import Final

import pandas as pd

from dominio.models import RawRealEstateSample, UnitizedRealEstateSample

UNIT_PRICE_COLUMN: Final[str] = "valor_unitario"


def calculate_unit_price(preco: float, areaprivativa: float) -> float:
    """Calcula o valor unitário em R$/m²."""
    if areaprivativa <= 0:
        raise ValueError("A área privativa deve ser maior que zero para unitarização.")

    return preco / areaprivativa


def add_unit_price_column(
    df: pd.DataFrame,
    price_column: str = "preco",
    area_column: str = "areaprivativa",
    unit_price_column: str = UNIT_PRICE_COLUMN,
) -> pd.DataFrame:
    """Retorna uma cópia do DataFrame com a coluna de valor unitário."""
    if price_column not in df.columns:
        raise ValueError(f"A coluna '{price_column}' não existe no DataFrame.")

    if area_column not in df.columns:
        raise ValueError(f"A coluna '{area_column}' não existe no DataFrame.")

    unitized_df = df.copy()
    unitized_df[unit_price_column] = (
        unitized_df[price_column] / unitized_df[area_column]
    )

    return unitized_df


def convert_samples_to_unitized(
    samples: list[RawRealEstateSample],
) -> list[UnitizedRealEstateSample]:
    """Converte amostras brutas em amostras unitarizadas tipadas."""
    unitized_samples: list[UnitizedRealEstateSample] = []

    for sample in samples:
        valor_unitario = calculate_unit_price(
            preco=sample.preco,
            areaprivativa=sample.areaprivativa,
        )

        unitized_sample = UnitizedRealEstateSample(
            id=sample.id,
            preco=sample.preco,
            areaprivativa=sample.areaprivativa,
            valor_unitario=valor_unitario,
            vagas=sample.vagas,
            idadeaparente=sample.idadeaparente,
            distanciacentrokm=sample.distanciacentrokm,
            fontelink=sample.fontelink,
            observacoes=sample.observacoes,
        )
        unitized_samples.append(unitized_sample)

    return unitized_samples


def build_descriptive_summary(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Gera um resumo estatístico descritivo da amostra."""
    selected_columns = columns or ["preco", "areaprivativa", UNIT_PRICE_COLUMN]
    return df[selected_columns].describe().round(2)


def calculate_coefficient_of_variation(
    series: pd.Series,
) -> float:
    """Calcula o coeficiente de variação em percentual."""
    mean_value = float(series.mean())
    std_value = float(series.std(ddof=1))

    if mean_value == 0:
        raise ValueError("A média da série é zero; o CV não pode ser calculado.")

    return (std_value / mean_value) * 100


def build_unitization_report(df: pd.DataFrame) -> dict[str, object]:
    """Consolida os principais resultados didáticos da Aula 1."""
    unitized_df = add_unit_price_column(df)

    summary = build_descriptive_summary(unitized_df)
    cv_percent = calculate_coefficient_of_variation(unitized_df[UNIT_PRICE_COLUMN])

    return {
        "dataframe_unitarizado": unitized_df,
        "resumo_estatistico": summary,
        "coeficiente_variacao_percentual": round(cv_percent, 2),
        "total_amostras": int(len(unitized_df)),
        "media_valor_unitario": float(unitized_df[UNIT_PRICE_COLUMN].mean()),
        "mediana_valor_unitario": float(unitized_df[UNIT_PRICE_COLUMN].median()),
        "desvio_padrao_valor_unitario": float(unitized_df[UNIT_PRICE_COLUMN].std(ddof=1)),
    }
