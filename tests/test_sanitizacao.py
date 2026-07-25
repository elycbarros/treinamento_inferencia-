"""Testes para sanitização amostral por Chauvenet."""

from __future__ import annotations

from servicos.sanitizacao import sanitize_sample_chauvenet
from servicos.unitarizacao import add_unit_price_column


def test_sanitize_sample_chauvenet_removes_discrepant_observation(
    sample_market_df_with_outlier,
) -> None:
    """Espera remoção de pelo menos um ponto discrepante."""
    df = add_unit_price_column(sample_market_df_with_outlier)
    clean_df, outliers_df = sanitize_sample_chauvenet(df, target_col="vu")

    assert len(clean_df) < len(df)
    assert not outliers_df.empty
    assert "AP-999" in set(outliers_df["id"].astype(str))


def test_sanitize_sample_chauvenet_reduces_extreme_unit_values(
    sample_market_df_with_outlier,
) -> None:
    """Verifica redução do extremo superior após sanitização."""
    df = add_unit_price_column(sample_market_df_with_outlier)
    clean_df, outliers_df = sanitize_sample_chauvenet(df, target_col="vu")

    assert outliers_df["vu"].max() > clean_df["vu"].max()
