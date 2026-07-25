"""Serviços de diagnóstico estatístico do modelo.

Este módulo reúne testes e métricas de validação do modelo
regressivo com foco didático e aderência ao fluxo do treinamento.
As funções centrais avaliam poder explicativo, significância global,
significância individual, normalidade dos resíduos e independência
dos resíduos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from servicos.regressao import OLSRegressionArtifacts


try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover
    scipy_stats = None


@dataclass(frozen=True)
class NormalityTestResult:
    """Representa o resultado do teste de normalidade."""

    test_name: str
    statistic: float
    p_value: float
    status: str
    method_note: str


@dataclass(frozen=True)
class DurbinWatsonResult:
    """Representa o resultado do teste de independência serial."""

    statistic: float
    status: str
    lower_bound: float
    upper_bound: float


def compute_durbin_watson(residuals: pd.Series) -> float:
    """Calcula a estatística de Durbin-Watson."""
    residual_values = residuals.astype(float).to_numpy()

    if len(residual_values) < 2:
        return float("nan")

    diff = np.diff(residual_values)
    numerator = float(np.sum(diff**2))
    denominator = float(np.sum(residual_values**2))

    if denominator == 0:
        return float("nan")

    return numerator / denominator


def run_shapiro_wilk_test(
    residuals: pd.Series,
    *,
    alpha: float = 0.05,
) -> NormalityTestResult:
    """Executa Shapiro-Wilk, com fallback simples se SciPy não existir."""
    values = residuals.astype(float).to_numpy()

    if len(values) < 3:
        return NormalityTestResult(
            test_name="Shapiro-Wilk",
            statistic=float("nan"),
            p_value=float("nan"),
            status="NAO_AVALIADO",
            method_note="Amostra insuficiente para teste de normalidade.",
        )

    if scipy_stats is None:
        return NormalityTestResult(
            test_name="Shapiro-Wilk",
            statistic=float("nan"),
            p_value=float("nan"),
            status="NAO_AVALIADO",
            method_note=(
                "SciPy não disponível; o teste Shapiro-Wilk não foi executado."
            ),
        )

    statistic, p_value = scipy_stats.shapiro(values)
    status = "NORMAL" if p_value > alpha else "NAO_NORMAL"

    return NormalityTestResult(
        test_name="Shapiro-Wilk",
        statistic=float(statistic),
        p_value=float(p_value),
        status=status,
        method_note="Teste executado com scipy.stats.shapiro.",
    )


def evaluate_explanatory_power(
    artifacts: OLSRegressionArtifacts,
    *,
    minimum_adjusted_r_squared: float = 0.70,
) -> dict[str, Any]:
    """Avalia o poder explicativo via R² ajustado."""
    obtained_value = float(artifacts.adjusted_r_squared)
    approved = obtained_value >= minimum_adjusted_r_squared

    return {
        "pressuposto": "Poder Explicativo",
        "metrica_teste": "R2 Ajustado",
        "valor_obtido": obtained_value,
        "criterio_aceitacao": f"R2 Ajustado >= {minimum_adjusted_r_squared:.2f}",
        "status": "APROVADO" if approved else "REPROVADO",
    }


def evaluate_global_significance(
    artifacts: OLSRegressionArtifacts,
    *,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Avalia a significância global do modelo pelo teste F."""
    p_value = float(artifacts.f_p_value)
    approved = p_value < alpha

    return {
        "pressuposto": "Significancia Global",
        "metrica_teste": "Teste F p-valor",
        "valor_obtido": p_value,
        "criterio_aceitacao": f"p-valor F < {alpha:.2f}",
        "status": "APROVADO" if approved else "REPROVADO",
    }


def evaluate_individual_significance(
    artifacts: OLSRegressionArtifacts,
    *,
    alpha: float = 0.10,
) -> pd.DataFrame:
    """Avalia significância individual dos coeficientes pelo teste t."""
    table = pd.DataFrame(
        {
            "variavel_explicativa": artifacts.coefficients.index,
            "coeficiente": artifacts.coefficients.values,
            "erro_padrao": artifacts.standard_errors.values,
            "estatistica_t": artifacts.t_statistics.values,
            "p_valor_t": artifacts.p_values.values,
        }
    )

    table["significativo_10pct"] = np.where(table["p_valor_t"] <= alpha, "SIM", "NAO")
    return table


def evaluate_residual_normality(
    artifacts: OLSRegressionArtifacts,
    *,
    alpha: float = 0.05,
) -> tuple[dict[str, Any], NormalityTestResult]:
    """Avalia a normalidade dos resíduos."""
    result = run_shapiro_wilk_test(artifacts.residuals, alpha=alpha)

    if np.isnan(result.p_value):
        status = "NAO_AVALIADO"
        value = np.nan
        criterion = f"p-valor W > {alpha:.2f}"
    else:
        status = result.status
        value = float(result.p_value)
        criterion = f"p-valor W > {alpha:.2f}"

    row = {
        "pressuposto": "Normalidade dos Residuos",
        "metrica_teste": "Shapiro-Wilk p-valor",
        "valor_obtido": value,
        "criterio_aceitacao": criterion,
        "status": status,
    }
    return row, result


def evaluate_residual_independence(
    artifacts: OLSRegressionArtifacts,
    *,
    lower_bound: float = 1.5,
    upper_bound: float = 2.5,
) -> tuple[dict[str, Any], DurbinWatsonResult]:
    """Avalia a independência dos resíduos via Durbin-Watson."""
    statistic = compute_durbin_watson(artifacts.residuals)

    if np.isnan(statistic):
        status = "NAO_AVALIADO"
    elif lower_bound <= statistic <= upper_bound:
        status = "APROVADO"
    else:
        status = "ALERTA"

    result = DurbinWatsonResult(
        statistic=float(statistic),
        status=status,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )

    row = {
        "pressuposto": "Independencia de Residuos",
        "metrica_teste": "Durbin-Watson",
        "valor_obtido": float(statistic),
        "criterio_aceitacao": f"Entre {lower_bound:.1f} e {upper_bound:.1f}",
        "status": status,
    }
    return row, result


def build_nbr_diagnostics_report(
    artifacts: OLSRegressionArtifacts,
    *,
    minimum_adjusted_r_squared: float = 0.70,
    alpha_f: float = 0.05,
    alpha_t: float = 0.10,
    alpha_shapiro: float = 0.05,
    dw_lower_bound: float = 1.5,
    dw_upper_bound: float = 2.5,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Monta o relatório consolidado de diagnóstico do modelo."""
    explanatory_power_row = evaluate_explanatory_power(
        artifacts,
        minimum_adjusted_r_squared=minimum_adjusted_r_squared,
    )

    global_significance_row = evaluate_global_significance(
        artifacts,
        alpha=alpha_f,
    )

    individual_significance_table = evaluate_individual_significance(
        artifacts,
        alpha=alpha_t,
    )

    normality_row, normality_result = evaluate_residual_normality(
        artifacts,
        alpha=alpha_shapiro,
    )

    independence_row, dw_result = evaluate_residual_independence(
        artifacts,
        lower_bound=dw_lower_bound,
        upper_bound=dw_upper_bound,
    )

    diagnostics_report = pd.DataFrame(
        [
            explanatory_power_row,
            global_significance_row,
            normality_row,
            independence_row,
        ]
    )

    approved_count = int(
        diagnostics_report["status"].isin(["APROVADO", "NORMAL"]).sum()
    )
    total_count = int(len(diagnostics_report))
    overall_status = (
        "APROVADO"
        if diagnostics_report["status"].isin(["REPROVADO"]).sum() == 0
        else "REPROVADO"
    )

    summary = {
        "status_geral_modelo": overall_status,
        "itens_aprovados_ou_normais": approved_count,
        "itens_avaliados": total_count,
        "proporcao_aprovacao": approved_count / total_count if total_count else np.nan,
        "teste_normalidade_observacao": normality_result.method_note,
        "durbin_watson_valor": dw_result.statistic,
        "durbin_watson_faixa": (
            f"{dw_result.lower_bound:.1f} a {dw_result.upper_bound:.1f}"
        ),
        "coeficientes_significativos_10pct": int(
            (individual_significance_table["significativo_10pct"] == "SIM").sum()
        ),
        "coeficientes_totais": int(len(individual_significance_table)),
    }

    return diagnostics_report, individual_significance_table, summary


def format_diagnostics_for_display(report_df: pd.DataFrame) -> pd.DataFrame:
    """Formata o relatório para visualização em console ou notebook."""
    formatted = report_df.copy()
    formatted["valor_obtido"] = formatted["valor_obtido"].round(6)
    return formatted


def format_coefficients_for_display(coeff_df: pd.DataFrame) -> pd.DataFrame:
    """Formata a tabela de coeficientes para visualização."""
    formatted = coeff_df.copy()
    numeric_columns = ["coeficiente", "erro_padrao", "estatistica_t", "p_valor_t"]
    formatted[numeric_columns] = formatted[numeric_columns].round(6)
    return formatted
