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


# Variável dependente padrão do modelo.
# Neste projeto, a regressão busca explicar o preço do imóvel.
DEFAULT_TARGET_COLUMN = "preco"

# Conjunto preferencial de variáveis explicativas.
# Elas representam atributos com leitura econômica direta na avaliação.
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
    # O valor absoluto do t é usado porque o teste é bicaudal:
    # extremos positivos e negativos são igualmente relevantes.
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

    # A estatística F é usada para testar a significância global do modelo.
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

    # Partimos das variáveis preferenciais do projeto e,
    # se existir, incorporamos também a distância à praia.
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
    # A matriz X contém apenas as variáveis explicativas.
    X = df.loc[:, feature_columns].copy()

    # A constante representa o intercepto do modelo,
    # isto é, o valor esperado quando todas as explicativas são zero.
    if add_intercept:
        X.insert(0, "const", 1.0)

    return X.astype(float)


def build_response_vector(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET_COLUMN,
) -> pd.Series:
    """Monta o vetor resposta da regressão."""
    return df[target_col].astype(float).copy()


def prepare_regression_dataframe(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET_COLUMN,
    feature_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, str, list[str]]:
    """Prepara a base final da regressão removendo nulos essenciais."""
    target_col, resolved_features = (
        (target_col, feature_columns)
        if feature_columns is not None
        else resolve_regression_columns(df, target_col=target_col)
    )

    required_columns = [target_col, *resolved_features]

    # A regressão só pode usar linhas com todas as variáveis necessárias preenchidas.
    regression_df = df.loc[:, required_columns].dropna().copy()

    if regression_df.empty:
        raise ValueError(
            "A base para regressão ficou vazia após a remoção de valores ausentes."
        )

    return regression_df, target_col, resolved_features



def build_coefficients_table(artifacts: OLSRegressionArtifacts) -> pd.DataFrame:
    """Monta uma tabela didática com os coeficientes do modelo.

    Esta função organiza, em formato tabular, os principais números
    usados na leitura inferencial de cada parâmetro estimado.
    O objetivo é oferecer uma saída simples para aulas, testes e relatórios.
    """
    # Cada índice da série de coeficientes representa uma variável do modelo,
    # como a constante (intercepto) e as variáveis explicativas.
    table = pd.DataFrame(
        {
            "variavel": artifacts.coefficients.index,
            "coeficiente": artifacts.coefficients.values,
            "erro_padrao": artifacts.standard_errors.values,
            "estatistica_t": artifacts.t_statistics.values,
            "p_valor": artifacts.p_values.values,
        }
    )

    # A coluna abaixo resume, de forma didática, se o coeficiente
    # é estatisticamente significativo ao nível de 10%.
    table["significativo_10pct"] = np.where(table["p_valor"] <= 0.10, "SIM", "NAO")
    return table


def fit_ols_regression(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET_COLUMN,
    feature_columns: list[str] | None = None,
    *,
    add_intercept: bool = True,
) -> OLSRegressionArtifacts:
    """Ajusta uma regressão linear múltipla via mínimos quadrados ordinários."""
    regression_df, target_col, resolved_features = prepare_regression_dataframe(
        df,
        target_col=target_col,
        feature_columns=feature_columns,
    )

    # O parâmetro add_intercept é exposto nesta função para manter
    # compatibilidade com testes e chamadas didáticas do projeto.
    X = build_design_matrix(
        regression_df,
        resolved_features,
        add_intercept=add_intercept,
    )
    y = build_response_vector(regression_df, target_col=target_col)

    x_matrix = X.to_numpy(dtype=float)
    y_vector = y.to_numpy(dtype=float)

    n_obs = int(x_matrix.shape[0])
    n_params = int(x_matrix.shape[1])

    if n_obs <= n_params:
        raise ValueError(
            "Não há graus de liberdade residuais suficientes para ajustar o modelo OLS."
        )

    # OLS estima beta resolvendo o problema de mínimos quadrados:
    # minimizar a soma dos resíduos ao quadrado.
    xtx = x_matrix.T @ x_matrix
    xtx_inv = np.linalg.inv(xtx)
    beta = xtx_inv @ x_matrix.T @ y_vector

    fitted = x_matrix @ beta
    residuals = y_vector - fitted

    y_mean = float(np.mean(y_vector))
    sse = float(np.sum(residuals**2))
    ssr = float(np.sum((fitted - y_mean) ** 2))
    sst = float(np.sum((y_vector - y_mean) ** 2))

    # SST mede a variabilidade total do preço.
    # SSR mede a parcela explicada pelo modelo.
    # SSE mede a parcela que sobrou nos resíduos.
    r_squared = 0.0 if sst == 0 else ssr / sst

    degrees_of_freedom_model = n_params - 1
    degrees_of_freedom_resid = n_obs - n_params

    adjusted_r_squared = 1.0 - (
        ((1.0 - r_squared) * (n_obs - 1)) / degrees_of_freedom_resid
    )

    sigma2 = sse / degrees_of_freedom_resid
    mse = sigma2
    rmse = float(np.sqrt(mse))

    variance_covariance = sigma2 * xtx_inv
    standard_errors = np.sqrt(np.diag(variance_covariance))
    t_statistics = beta / standard_errors
    p_values = np.array([_two_tailed_normal_p_value(value) for value in t_statistics])

    if degrees_of_freedom_model > 0 and mse > 0:
        msr = ssr / degrees_of_freedom_model
        f_statistic = msr / mse
        f_p_value = f_survival_function(
            f_statistic,
            degrees_of_freedom_model,
            degrees_of_freedom_resid,
        )
    else:
        f_statistic = float("nan")
        f_p_value = float("nan")

    coefficient_index = X.columns

    return OLSRegressionArtifacts(
        coefficients=pd.Series(beta, index=coefficient_index, name="coeficiente"),
        fitted_values=pd.Series(fitted, index=regression_df.index, name="valor_estimado"),
        residuals=pd.Series(residuals, index=regression_df.index, name="residuo"),
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
            variance_covariance,
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



def build_model_summary(artifacts: OLSRegressionArtifacts) -> dict[str, float | int]:
    """Monta um dicionário-resumo com as métricas globais do modelo.

    Esta função organiza os indicadores centrais do ajuste OLS
    em formato simples, adequado para testes, relatórios e leitura didática.
    """
    # O resumo abaixo concentra tamanho da amostra, graus de liberdade,
    # qualidade de ajuste e significância global do modelo.
    return {
        "n_obs": int(artifacts.n_obs),
        "n_params": int(artifacts.n_params),
        "gl_modelo": int(artifacts.degrees_of_freedom_model),
        "gl_residuos": int(artifacts.degrees_of_freedom_resid),
        "r2": float(artifacts.r_squared),
        "r2_ajustado": float(artifacts.adjusted_r_squared),
        "rmse": float(artifacts.rmse),
        "f_statistic": float(artifacts.f_statistic),
        "f_p_value": float(artifacts.f_p_value),
    }


def build_regression_summary(artifacts: OLSRegressionArtifacts) -> pd.DataFrame:
    """Consolida coeficientes e estatísticas em tabela resumo."""
    # Esta tabela é o coração interpretativo da regressão:
    # reúne coeficiente, erro padrão, estatística t e p-valor.
    summary = pd.concat(
        [
            artifacts.coefficients,
            artifacts.standard_errors,
            artifacts.t_statistics,
            artifacts.p_values,
        ],
        axis=1,
    )
    return summary


def attach_regression_outputs(
    df: pd.DataFrame,
    artifacts: OLSRegressionArtifacts,
) -> pd.DataFrame:
    """Acopla valores estimados e resíduos ao DataFrame base da regressão."""
    result = df.copy()
    result.loc[artifacts.fitted_values.index, "valor_estimado"] = artifacts.fitted_values
    result.loc[artifacts.residuals.index, "residuo"] = artifacts.residuals
    result.loc[artifacts.residuals.index, "residuo_percentual"] = (
        artifacts.residuals / artifacts.response_vector
    ) * 100.0
    return result


def build_regression_report(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET_COLUMN,
    feature_columns: list[str] | None = None,
) -> dict[str, Any]:
    """Executa o ajuste e devolve um relatório completo da regressão."""
    artifacts = fit_ols_regression(
        df,
        target_col=target_col,
        feature_columns=feature_columns,
    )
    regression_df = attach_regression_outputs(df, artifacts)
    summary_df = build_regression_summary(artifacts)

    # O relatório final entrega tanto os artefatos numéricos
    # quanto a base enriquecida para inspeção posterior.
    return {
        "target_column": target_col,
        "feature_columns": feature_columns or list(DEFAULT_FEATURE_COLUMNS),
        "n_observacoes": artifacts.n_obs,
        "n_parametros": artifacts.n_params,
        "graus_liberdade_modelo": artifacts.degrees_of_freedom_model,
        "graus_liberdade_residuos": artifacts.degrees_of_freedom_resid,
        "r_quadrado": artifacts.r_squared,
        "r_quadrado_ajustado": artifacts.adjusted_r_squared,
        "mse": artifacts.mse,
        "rmse": artifacts.rmse,
        "sse": artifacts.sse,
        "ssr": artifacts.ssr,
        "sst": artifacts.sst,
        "estatistica_f": artifacts.f_statistic,
        "p_valor_f": artifacts.f_p_value,
        "sigma2": artifacts.sigma2,
        "coeficientes": artifacts.coefficients,
        "erros_padrao": artifacts.standard_errors,
        "estatisticas_t": artifacts.t_statistics,
        "p_valores": artifacts.p_values,
        "matriz_covariancia": artifacts.variance_covariance_matrix,
        "resumo_regressao": summary_df,
        "dataframe_regressao": regression_df,
        "artefatos": artifacts,
    }
