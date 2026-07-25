"""Fixtures compartilhadas para a suíte de testes do projeto."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def sample_market_df() -> pd.DataFrame:
    """Retorna uma amostra sintética coerente para regressão e diagnósticos."""
    rows = []
    for i, (area, vagas, dist) in enumerate(
        [
            (70, 1, 5.0),
            (72, 1, 4.8),
            (75, 1, 4.5),
            (78, 2, 4.2),
            (80, 2, 4.0),
            (85, 2, 3.8),
            (88, 2, 3.5),
            (92, 2, 3.2),
            (95, 3, 3.0),
            (98, 3, 2.8),
            (102, 3, 2.5),
            (105, 3, 2.2),
        ],
        start=1,
    ):
        preco = (
            120_000
            + 2_600 * area
            + 42_000 * vagas
            - 130 * dist
            + ((-1) ** i) * 2_000
        )
        rows.append(
            {
                "id": f"AP-{i:03d}",
                "preco": float(preco),
                "areaprivativa": float(area),
                "vagas": int(vagas),
                "distanciacentrokm": float(dist),
            }
        )
    return pd.DataFrame(rows)


@pytest.fixture
def sample_market_df_with_outlier(sample_market_df: pd.DataFrame) -> pd.DataFrame:
    """Retorna amostra com uma observação discrepante evidente."""
    outlier = pd.DataFrame(
        [
            {
                "id": "AP-999",
                "preco": 5_000_000.0,
                "areaprivativa": 70.0,
                "vagas": 1,
                "distanciacentrokm": 5.0,
            }
        ]
    )
    return pd.concat([sample_market_df, outlier], ignore_index=True)


@pytest.fixture
def market_csv_path(tmp_path: Path, sample_market_df: pd.DataFrame) -> Path:
    """Grava a amostra sintética em CSV temporário para testes de carga."""
    csv_path = tmp_path / "amostras_residencial35.csv"
    sample_market_df.to_csv(csv_path, index=False)
    return csv_path
