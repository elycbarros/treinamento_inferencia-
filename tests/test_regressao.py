"""Testes para ajuste regressivo por OLS."""

from __future__ import annotations

from servicos.regressao import (
    build_coefficients_table,
    build_model_summary,
    fit_ols_regression,
)
from servicos.unitarizacao import add_unit_price_column


def test_fit_ols_regression_returns_core_artifacts(sample_market_df) -> None:
    """Valida estrutura básica do ajuste OLS."""
    df = add_unit_price_column(sample_market_df)

    artifacts = fit_ols_regression(
        df=df,
        target_col="preco",
        feature_columns=["areaprivativa", "vagas", "distanciacentrokm"],
        add_intercept=True,
    )

    assert artifacts.n_obs == len(df)
    assert artifacts.n_params == 4
    assert artifacts.degrees_of_freedom_model == 3
    assert artifacts.degrees_of_freedom_resid == len(df) - 4
    assert "const" in artifacts.coefficients.index
    assert artifacts.adjusted_r_squared > 0.90


def test_build_coefficients_table_has_expected_columns(sample_market_df) -> None:
    """Valida a tabela didática de coeficientes."""
    df = add_unit_price_column(sample_market_df)

    artifacts = fit_ols_regression(
        df=df,
        target_col="preco",
        feature_columns=["areaprivativa", "vagas", "distanciacentrokm"],
        add_intercept=True,
    )
    table = build_coefficients_table(artifacts)

    assert {
        "variavel",
        "coeficiente",
        "erro_padrao",
        "estatistica_t",
        "p_valor",
        "significativo_10pct",
    }.issubset(table.columns)


def test_build_model_summary_returns_expected_keys(sample_market_df) -> None:
    """Valida o dicionário-resumo do ajuste."""
    df = add_unit_price_column(sample_market_df)

    artifacts = fit_ols_regression(
        df=df,
        target_col="preco",
        feature_columns=["areaprivativa", "vagas", "distanciacentrokm"],
        add_intercept=True,
    )
    summary = build_model_summary(artifacts)

    assert {
        "n_obs",
        "n_params",
        "gl_modelo",
        "gl_residuos",
        "r2",
        "r2_ajustado",
        "rmse",
        "f_statistic",
        "f_p_value",
    }.issubset(summary.keys())
