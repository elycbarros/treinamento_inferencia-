"""Testes para diagnósticos estatísticos do modelo."""

from __future__ import annotations

import pandas as pd

from servicos.diagnosticos import (
    build_nbr_diagnostics_report,
    compute_durbin_watson,
)
from servicos.regressao import fit_ols_regression
from servicos.unitarizacao import add_unit_price_column


def test_compute_durbin_watson_returns_value_between_zero_and_four() -> None:
    """Durbin-Watson válido deve ficar no intervalo teórico usual."""
    residuals = pd.Series([1.0, -0.5, 0.25, -0.25, 0.1])
    dw = compute_durbin_watson(residuals)

    assert 0.0 <= dw <= 4.0


def test_build_nbr_diagnostics_report_returns_four_core_checks(sample_market_df) -> None:
    """Valida o relatório consolidado de diagnóstico."""
    df = add_unit_price_column(sample_market_df)

    artifacts = fit_ols_regression(
        df=df,
        target_col="preco",
        feature_columns=["areaprivativa", "vagas", "distanciacentrokm"],
        add_intercept=True,
    )

    report_df, coeff_df, summary = build_nbr_diagnostics_report(artifacts)

    expected_checks = {
        "Poder Explicativo",
        "Significancia Global",
        "Normalidade dos Residuos",
        "Independencia de Residuos",
    }

    assert set(report_df["pressuposto"]) == expected_checks
    assert "significativo_10pct" in coeff_df.columns
    assert summary["itens_avaliados"] == 4
    assert "status_geral_modelo" in summary


def test_build_nbr_diagnostics_report_exposes_acceptance_columns(sample_market_df) -> None:
    """Confirma colunas de critério e status no relatório NBR."""
    df = add_unit_price_column(sample_market_df)

    artifacts = fit_ols_regression(
        df=df,
        target_col="preco",
        feature_columns=["areaprivativa", "vagas", "distanciacentrokm"],
        add_intercept=True,
    )

    report_df, _, _ = build_nbr_diagnostics_report(artifacts)

    assert {
        "pressuposto",
        "metrica_teste",
        "valor_obtido",
        "criterio_aceitacao",
        "status",
    }.issubset(report_df.columns)
