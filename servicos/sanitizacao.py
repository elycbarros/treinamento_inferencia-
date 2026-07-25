"""Serviços de sanitização estatística da amostra.

Este módulo implementa o critério de Chauvenet de forma iterativa,
com rastreabilidade das exclusões realizadas em cada rodada.
O objetivo didático é deixar explícito por que uma amostra foi
mantida ou removida da série estatística.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from scipy import stats


DEFAULT_TARGET_COLUMN: Final[str] = "valor_unitario"
DEFAULT_ID_COLUMN: Final[str] = "id"


def validate_target_column(df: pd.DataFrame, target_col: str) -> None:
    """Valida a existência da coluna-alvo usada na sanitização."""
    if target_col not in df.columns:
        raise ValueError(f"A coluna '{target_col}' não existe no DataFrame informado.")


def calculate_chauvenet_threshold(series: pd.Series) -> dict[str, float]:
    """Calcula estatísticas e limite crítico do critério de Chauvenet."""
    cleaned_series = series.dropna().astype(float)
    n = len(cleaned_series)

    if n <= 1:
        raise ValueError("A série deve conter ao menos duas observações válidas.")

    mean_value = float(cleaned_series.mean())
    std_value = float(cleaned_series.std(ddof=1))
    probability_limit = 1 / (2 * n)
    z_critical = float(stats.norm.ppf(1 - (probability_limit / 2)))

    return {
        "n": float(n),
        "mean": mean_value,
        "std": std_value,
        "probability_limit": float(probability_limit),
        "z_critical": z_critical,
    }


def append_chauvenet_columns(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET_COLUMN,
) -> pd.DataFrame:
    """Retorna uma cópia do DataFrame com z-score e flag de outlier."""
    validate_target_column(df, target_col)
    result = df.copy()

    threshold = calculate_chauvenet_threshold(result[target_col])
    std_value = threshold["std"]

    if std_value == 0:
        result["z_score"] = 0.0
        result["chauvenet_outlier"] = False
        result["z_critical"] = threshold["z_critical"]
        result["probability_limit"] = threshold["probability_limit"]
        return result

    result["z_score"] = np.abs((result[target_col] - threshold["mean"]) / std_value)
    result["chauvenet_outlier"] = result["z_score"] > threshold["z_critical"]
    result["z_critical"] = threshold["z_critical"]
    result["probability_limit"] = threshold["probability_limit"]

    return result


def sanitize_sample_chauvenet(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET_COLUMN,
    id_col: str = DEFAULT_ID_COLUMN,
    max_iterations: int = 10,
    min_sample_size: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aplica o critério de Chauvenet de modo iterativo.

    Retorna:
    - DataFrame saneado;
    - DataFrame com amostras removidas;
    - histórico por iteração.
    """
    validate_target_column(df, target_col)

    if id_col not in df.columns:
        raise ValueError(f"A coluna identificadora '{id_col}' não existe no DataFrame.")

    clean_df = df.copy().reset_index(drop=True)
    removed_frames: list[pd.DataFrame] = []
    history_rows: list[dict[str, object]] = []

    for iteration in range(1, max_iterations + 1):
        n_before = len(clean_df)

        if n_before <= min_sample_size:
            history_rows.append(
                {
                    "iteracao": iteration,
                    "n_inicial": n_before,
                    "media": np.nan,
                    "desvio_padrao": np.nan,
                    "probabilidade_limite": np.nan,
                    "z_critico": np.nan,
                    "removidos": 0,
                    "status": "interrompido_amostra_minima",
                }
            )
            break

        threshold = calculate_chauvenet_threshold(clean_df[target_col])
        std_value = threshold["std"]

        if std_value == 0:
            history_rows.append(
                {
                    "iteracao": iteration,
                    "n_inicial": n_before,
                    "media": threshold["mean"],
                    "desvio_padrao": std_value,
                    "probabilidade_limite": threshold["probability_limit"],
                    "z_critico": threshold["z_critical"],
                    "removidos": 0,
                    "status": "encerrado_desvio_zero",
                }
            )
            break

        iteration_df = append_chauvenet_columns(clean_df, target_col=target_col)
        outliers_mask = iteration_df["chauvenet_outlier"]
        removed_count = int(outliers_mask.sum())

        history_rows.append(
            {
                "iteracao": iteration,
                "n_inicial": n_before,
                "media": threshold["mean"],
                "desvio_padrao": threshold["std"],
                "probabilidade_limite": threshold["probability_limit"],
                "z_critico": threshold["z_critical"],
                "removidos": removed_count,
                "status": "sem_outliers" if removed_count == 0 else "outliers_removidos",
            }
        )

        if removed_count == 0:
            break

        removed_in_iteration = iteration_df.loc[outliers_mask].copy()
        removed_in_iteration["iteracao_remocao"] = iteration
        removed_frames.append(removed_in_iteration)

        clean_df = iteration_df.loc[~outliers_mask].drop(
            columns=["z_score", "chauvenet_outlier", "z_critical", "probability_limit"]
        )
        clean_df = clean_df.reset_index(drop=True)

    removed_df = pd.concat(removed_frames, ignore_index=True) if removed_frames else pd.DataFrame()
    history_df = pd.DataFrame(history_rows)

    return clean_df, removed_df, history_df


def build_sanitization_report(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET_COLUMN,
    id_col: str = DEFAULT_ID_COLUMN,
) -> dict[str, object]:
    """Consolida os principais resultados da sanitização por Chauvenet."""
    clean_df, removed_df, history_df = sanitize_sample_chauvenet(
        df=df,
        target_col=target_col,
        id_col=id_col,
    )

    return {
        "amostra_inicial": int(len(df)),
        "amostra_saneada": int(len(clean_df)),
        "total_removido": int(len(removed_df)),
        "df_saneado": clean_df,
        "df_removidos": removed_df,
        "historico_iteracoes": history_df,
    }
