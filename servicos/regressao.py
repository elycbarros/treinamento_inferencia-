"""Serviços de ajuste regressivo por OLS.

Este módulo concentra funções para preparação da matriz de projeto,
ajuste do modelo linear múltiplo e extração de medidas centrais para
interpretação didática do modelo em avaliação imobiliária.
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_TARGET_COLUMN = "preco"
DEFAULT_FEATURE_COLUMNS = ("areaprivativa", "vagas", "distanciacentrokm")


@dataclass(frozen=True)
class OLSRegressionArtifacts:
    """Agrupa os principais artefatos do ajuste OLS."""

    coefficients: pd.Series
    fitted_values: pd.Series
    residuals: pd.Series
    design_matrix: pd.DataFrame
    response_vector: pd.Series
    n_obs: int
    n_params: int
    degrees_of_freedom_model: int
    degrees_of_freedom_resid: int
    r_squared: float
    adjusted_r_squared: float
    mse: float
    rmse: float
    sse: float
    ssr: float
    sst: float
    variance_covariance_matrix: pd.DataFrame
    standard_errors: pd.Series
    t_statistics: pd.Series
    p_values: pd.Series
    f_statistic: float
    f_p_value: float
    sigma2: float


def _normal_survival_function(value: float) -> float:
    """Retorna a cauda superior da normal padrão."""
    return 0.5 * math.erfc(value / np.sqrt(2.0))


def _two_tailed_normal_p_value(t_statistic: float) -> float:
    """Aproxima p-valor bicaudal via normal padrão.

    Observação didática:
    Para manter o projeto leve e sem dependência obrigatória de SciPy,
    esta implementação usa a normal padrão como aproximação para o
    teste t. Em amostras maiores isso tende a funcionar bem para fins
    instrutivos.
    """
    tail = _normal_survival_function(abs(float(t_statistic)))
    return 2.0 * tail


def _regularized_beta_continued_fraction(a: float, b: float, x: float) -> float:
    """Calcula a fração contínua usada na beta incompleta regularizada."""
    max_iter = 200
    epsilon = 3.0e-14
    fpmin = 1.0e-300

    qab = a + b
    qap = a + 1.0
    qam = a - 1.0

    c = 1.0
    d = 1.0 - (qab * x / qap)
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m

        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c

        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta

        if abs(delta - 1.0) < epsilon:
            break

    return h


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Calcula a beta incompleta regularizada I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    bt = np.exp(log_bt)

    if x < (a + 1.0) / (a + b + 2.0):
        cf = _regularized_beta_continued_fraction(a, b, x)
        return bt * cf / a

    cf = _regularized_beta_continued_fraction(b, a, 1.0 - x)
    return 1.0 - (bt * cf / b)


def f_survival_function(f_value: float, dfn: int, dfd: int) -> float:
    """Retorna a cauda superior da distribuição F."""
    if f_value <= 0:
        return 1.0

    x = (dfn * f_value) / ((dfn * f_value) + dfd)
    cdf = regularized_incomplete_beta(x, dfn / 2.0, dfd / 2.0)
    return max(0.0, min(1.0, 1.0 - cdf))


def resolve_regression_columns(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET_COLUMN,
    preferred_features: tuple[str, ...] = DEFAULT_FEATURE_COLUMNS,
) -> tuple[str, list[str]]:
    """Resolve automaticamente a variável dependente e as explicativas."""
    if target_col not in df.columns:
        raise KeyError(
            f"A coluna dependente `{target_col}` não foi encontrada no DataFrame."
        )

    feature_candidates = list(preferred_features)

    if "dist_praia" in df.columns and "dist_praia" not in feature_candidates:
        feature_candidates.append("dist_praia")

    available_features = [column for column in feature_candidates if column in df.columns]

    if not available_features:
        raise KeyError(
            "Nenhuma coluna explicativa padrão foi encontrada. "
            f"Tentativas: {feature_candidates}."
        )

    return target_col, available_features


def build_design_matrix(
    df: pd.DataFrame,
    feature_columns: list[str],
    *,
    add_intercept: bool = True,
) -> pd.DataFrame:
    """Monta a matriz de projeto para regressão."""
    X = df.loc[:, feature_columns].copy()

    if add_intercept:
        X.insert(0, "const", 1.0)

    return X.astype(float)


def fit_ols_regression(
    df: pd.DataFrame,
    *,
    target_col: str = DEFAULT_TARGET_COLUMN,
    feature_columns: list[str] | None = None,
    add_intercept: bool = True,
) -> OLSRegressionArtifacts:
    """Ajusta uma regressão linear múltipla por OLS."""
    resolved_target, resolved_features = resolve_regression_columns(
        df=df,
        target_col=target_col,
        preferred_features=tuple(feature_columns or DEFAULT_FEATURE_COLUMNS),
    )

    required_columns = [resolved_target, *resolved_features]
    clean_df = df.loc[:, required_columns].dropna().copy()

    X = build_design_matrix(clean_df, resolved_features, add_intercept=add_intercept)
    y = clean_df[resolved_target].astype(float)

    X_values = X.to_numpy(dtype=float)
    y_values = y.to_numpy(dtype=float)

    xtx = X_values.T @ X_values
    xtx_inv = np.linalg.inv(xtx)
    beta = xtx_inv @ (X_values.T @ y_values)

    fitted = X_values @ beta
    residuals = y_values - fitted

    n_obs = len(clean_df)
    n_params = X.shape[1]
    degrees_of_freedom_model = n_params - 1 if add_intercept else n_params
    degrees_of_freedom_resid = n_obs - n_params

    if degrees_of_freedom_resid <= 0:
        raise ValueError(
            "Graus de liberdade residuais insuficientes para ajuste OLS. "
            "A amostra precisa ser maior que o número de parâmetros."
        )

    y_mean = float(np.mean(y_values))
    sse = float(np.sum(residuals**2))
    sst = float(np.sum((y_values - y_mean) ** 2))
    ssr = float(sst - sse)

    r_squared = 1.0 - (sse / sst) if sst > 0 else 0.0
    adjusted_r_squared = 1.0 - (
        ((1.0 - r_squared) * (n_obs - 1)) / degrees_of_freedom_resid
    )

    sigma2 = sse / degrees_of_freedom_resid
    variance_covariance_matrix = sigma2 * xtx_inv
    standard_errors = np.sqrt(np.diag(variance_covariance_matrix))
    t_statistics = beta / standard_errors
    p_values = np.array([_two_tailed_normal_p_value(t) for t in t_statistics])

    mse = sse / n_obs
    rmse = float(np.sqrt(mse))

    if degrees_of_freedom_model > 0:
        msr = ssr / degrees_of_freedom_model
        f_statistic = msr / sigma2 if sigma2 > 0 else np.inf
        f_p_value = f_survival_function(
            f_value=float(f_statistic),
            dfn=degrees_of_freedom_model,
            dfd=degrees_of_freedom_resid,
        )
    else:
        f_statistic = np.nan
        f_p_value = np.nan

    coefficient_index = X.columns

    return OLSRegressionArtifacts(
        coefficients=pd.Series(beta, index=coefficient_index, name="coeficiente"),
        fitted_values=pd.Series(fitted, index=clean_df.index, name="valor_ajustado"),
        residuals=pd.Series(residuals, index=clean_df.index, name="residuo"),
        design_matrix=X,
        response_vector=y,
        n_obs=n_obs,
        n_params=n_params,
        degrees_of_freedom_model=degrees_of_freedom_model,
        degrees_of_freedom_resid=degrees_of_freedom_resid,
        r_squared=float(r_squared),
        adjusted_r_squared=float(adjusted_r_squared),
        mse=float(mse),
        rmse=float(rmse),
        sse=float(sse),
        ssr=float(ssr),
        sst=float(sst),
        variance_covariance_matrix=pd.DataFrame(
            variance_covariance_matrix,
            index=coefficient_index,
            columns=coefficient_index,
        ),
        standard_errors=pd.Series(
            standard_errors,
            index=coefficient_index,
            name="erro_padrao",
        ),
        t_statistics=pd.Series(
            t_statistics,
            index=coefficient_index,
            name="estatistica_t",
        ),
        p_values=pd.Series(
            p_values,
            index=coefficient_index,
            name="p_valor",
        ),
        f_statistic=float(f_statistic),
        f_p_value=float(f_p_value),
        sigma2=float(sigma2),
    )


def build_coefficients_table(artifacts: OLSRegressionArtifacts) -> pd.DataFrame:
    """Monta tabela didática de coeficientes e significância individual."""
    table = pd.concat(
        [
            artifacts.coefficients,
            artifacts.standard_errors,
            artifacts.t_statistics,
            artifacts.p_values,
        ],
        axis=1,
    ).reset_index()

    table = table.rename(
        columns={
            "index": "variavel",
            "coeficiente": "coeficiente",
            "erro_padrao": "erro_padrao",
            "estatistica_t": "estatistica_t",
            "p_valor": "p_valor",
        }
    )

    table["significativo_10pct"] = np.where(table["p_valor"] <= 0.10, "SIM", "NAO")
    return table


def build_model_summary(artifacts: OLSRegressionArtifacts) -> dict[str, Any]:
    """Consolida métricas centrais do ajuste OLS."""
    return {
        "n_obs": artifacts.n_obs,
        "n_params": artifacts.n_params,
        "gl_modelo": artifacts.degrees_of_freedom_model,
        "gl_residuos": artifacts.degrees_of_freedom_resid,
        "r2": artifacts.r_squared,
        "r2_ajustado": artifacts.adjusted_r_squared,
        "rmse": artifacts.rmse,
        "sse": artifacts.sse,
        "ssr": artifacts.ssr,
        "sst": artifacts.sst,
        "f_statistic": artifacts.f_statistic,
        "f_p_value": artifacts.f_p_value,
    }


def attach_predictions_and_residuals(
    df: pd.DataFrame,
    artifacts: OLSRegressionArtifacts,
) -> pd.DataFrame:
    """Anexa valores ajustados e resíduos ao DataFrame original filtrado."""
    enriched = df.loc[artifacts.response_vector.index].copy()
    enriched["valor_ajustado"] = artifacts.fitted_values
    enriched["residuo"] = artifacts.residuals
    return enriched
