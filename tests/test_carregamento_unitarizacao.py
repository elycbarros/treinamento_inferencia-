"""Testes para carga da base e cálculo de valor unitário."""

from __future__ import annotations

import pytest

from servicos.carregamento import load_raw_dataset
from servicos.unitarizacao import add_unit_price_column


def test_load_raw_dataset_reads_csv_with_expected_columns(market_csv_path) -> None:
    """Valida a leitura do CSV e a presença das colunas essenciais."""
    df = load_raw_dataset(market_csv_path)

    assert len(df) == 12
    assert {"preco", "areaprivativa", "vagas", "distanciacentrokm"}.issubset(df.columns)


def test_add_unit_price_column_creates_vu(sample_market_df) -> None:
    """Valida a criação da coluna de valor unitário."""
    result = add_unit_price_column(sample_market_df)

    assert "valor_unitario" in result.columns
    expected_vu = result.loc[0, "preco"] / result.loc[0, "areaprivativa"]
    assert result.loc[0, "valor_unitario"] == pytest.approx(expected_vu)


def test_add_unit_price_column_preserves_row_count(sample_market_df) -> None:
    """Garante que o enriquecimento não altere a quantidade de linhas."""
    result = add_unit_price_column(sample_market_df)
    assert len(result) == len(sample_market_df)
