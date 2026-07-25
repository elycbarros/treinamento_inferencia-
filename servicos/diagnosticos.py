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

    # O teste compara a variação entre resíduos consecutivos
    # com a energia total dos próprios resíduos.
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

    # O teste de normalidade requer tamanho mínimo de amostra.
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

    # O R² ajustado é usado aqui porque penaliza excesso de variáveis
    # e oferece leitura mais prudente do poder explicativo.
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

    # O teste F verifica se o conjunto das explicativas,
    # tomado em bloco, contribui para explicar a variável dependente.
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

    # Aqui avaliamos variável por variável.
    # Em contexto didático, isso ajuda a distinguir relevância global do modelo
    # e relevância individual de cada coeficiente.
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

    summary = {
        "pressuposto": "Normalidade dos Residuos",
        "metrica_teste": "Shapiro-Wilk p-valor",
        "valor_obtido": value,
        "criterio_aceitacao": criterion,
        "status": status,
    }

    return summary, result


def evaluate_residual_independence(
    artifacts: OLSRegressionArtifacts,
    *,
    lower_bound: float = 1.50,
    upper_bound: float = 2.50,
) -> tuple[dict[str, Any], DurbinWatsonResult]:
    """Avalia independência dos resíduos pela estatística de Durbin-Watson."""
    statistic = compute_durbin_watson(artifacts.residuals)

    if np.isnan(statistic):
        status = "NAO_AVALIADO"
    elif lower_bound <= statistic <= upper_bound:
        status = "APROVADO"
    else:
        status = "REPROVADO"

    # A leitura prática adotada nesta aula é por faixa:
    # valores próximos de 2 sugerem independência residual.
    result = DurbinWatsonResult(
        statistic=float(statistic),
        status=status,
        lower_bound=float(lower_bound),
        upper_bound=float(upper_bound),
    )

    summary = {
        "pressuposto": "Independencia dos Residuos",
        "metrica_teste": "Durbin-Watson",
        "valor_obtido": float(statistic),
        "criterio_aceitacao": f"{lower_bound:.2f} <= DW <= {upper_bound:.2f}",
        "status": status,
    }

    return summary, result


def build_nbr_diagnostics_report(
    artifacts: OLSRegressionArtifacts,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Consolida os diagnósticos centrais do modelo em formato tabular."""
    explanatory_power = evaluate_explanatory_power(artifacts)
    global_significance = evaluate_global_significance(artifacts)
    individual_significance = evaluate_individual_significance(artifacts)
    residual_normality, normality_result = evaluate_residual_normality(artifacts)
    residual_independence, dw_result = evaluate_residual_independence(artifacts)

    diagnostics_df = pd.DataFrame(
        [
            explanatory_power,
            global_significance,
            residual_normality,
            residual_independence,
        ]
    )

    approved_or_normal = diagnostics_df["status"].isin(["APROVADO", "NORMAL"]).sum()
    total_items = int(len(diagnostics_df))
    significant_10pct = int(
        (individual_significance["significativo_10pct"] == "SIM").sum()
    )
    total_coefficients = int(len(individual_significance))

    # O resumo executivo transforma vários testes em uma visão consolidada,
    # útil para aula, laudo e leitura rápida do status do modelo.
    summary = {
        "status_geral_modelo": (
            "APROVADO"
            if approved_or_normal == total_items
            else "APROVADO_COM_RESTRICOES"
            if approved_or_normal >= max(1, total_items - 1)
            else "REPROVADO"
        ),
        "itens_aprovados_ou_normais": int(approved_or_normal),
        "itens_avaliados": total_items,
        "proporcao_aprovacao": approved_or_normal / total_items if total_items else 0.0,
        "coeficientes_significativos_10pct": significant_10pct,
        "coeficientes_totais": total_coefficients,
        "teste_normalidade_observacao": normality_result.method_note,
        "durbin_watson_faixa": (
            f"{dw_result.lower_bound:.2f} a {dw_result.upper_bound:.2f}"
        ),
        "durbin_watson_valor": dw_result.statistic,
    }

    return diagnostics_df, individual_significance, summary


def format_diagnostics_for_display(report_df: pd.DataFrame) -> pd.DataFrame:
    """Formata o relatório de diagnóstico para exibição amigável."""
    display_df = report_df.copy()

    if "valor_obtido" in display_df.columns:
        display_df["valor_obtido"] = display_df["valor_obtido"].apply(
            lambda value: f"{value:.6f}" if pd.notna(value) else "NA"
        )

    return display_df


def format_coefficients_for_display(coeff_df: pd.DataFrame) -> pd.DataFrame:
    """Formata a tabela de coeficientes para exibição amigável."""
    display_df = coeff_df.copy()

    for column in ["coeficiente", "erro_padrao", "estatistica_t", "p_valor_t"]:
        if column in display_df.columns:
            display_df[column] = display_df[column].apply(
                lambda value: f"{value:.6f}" if pd.notna(value) else "NA"
            )

    return display_df
