"""Teste integrado do fluxo principal do treinamento."""

from __future__ import annotations

from servicos.diagnosticos import build_nbr_diagnostics_report
from servicos.regressao import fit_ols_regression
from servicos.sanitizacao import sanitize_sample_chauvenet
from servicos.unitarizacao import add_unit_price_column


def test_end_to_end_training_flow(sample_market_df_with_outlier) -> None:
    """Executa unitarização, sanitização, regressão e diagnósticos."""
    df = add_unit_price_column(sample_market_df_with_outlier)
    clean_df, outliers_df = sanitize_sample_chauvenet(df, target_col="vu")

    artifacts = fit_ols_regression(
        df=clean_df,
        target_col="preco",
        feature_columns=["areaprivativa", "vagas", "distanciacentrokm"],
        add_intercept=True,
    )

    report_df, coeff_df, summary = build_nbr_diagnostics_report(artifacts)

    assert not outliers_df.empty
    assert len(clean_df) < len(df)
    assert not report_df.empty
    assert not coeff_df.empty
    assert summary["itens_avaliados"] == 4
