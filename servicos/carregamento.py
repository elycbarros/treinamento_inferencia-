"""Serviços de leitura e validação inicial da base de dados.

Este módulo concentra funções responsáveis por:
- localizar a base de dados;
- carregar o arquivo CSV;
- validar colunas obrigatórias;
- converter cada linha em contrato tipado.

A intenção didática é separar a etapa de entrada de dados da etapa
de análise estatística.
"""

from pathlib import Path
from typing import Final, Iterable

import pandas as pd

from dominio.models import RawRealEstateSample

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    "id",
    "preco",
    "areaprivativa",
    "vagas",
    "idadeaparente",
    "distanciacentrokm",
    "fontelink",
)


def resolve_project_root() -> Path:
    """Resolve a raiz do projeto de forma robusta.

    Se o código estiver sendo executado a partir de uma subpasta como
    `notebooks`, a função sobe um nível. Caso contrário, usa o diretório
    atual como raiz.
    """
    current_dir = Path.cwd()
    return current_dir.parent if current_dir.name == "notebooks" else current_dir


def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str] = REQUIRED_COLUMNS,
) -> None:
    """Valida a presença das colunas mínimas exigidas na base."""
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(
            "A base de dados não contém todas as colunas obrigatórias. "
            f"Colunas ausentes: {missing_columns}"
        )


def load_raw_dataset(csv_path: Path) -> pd.DataFrame:
    """Carrega o CSV bruto e valida sua estrutura mínima."""
    if not csv_path.exists():
        raise FileNotFoundError(
            f"O arquivo não foi encontrado em: {csv_path.resolve()}.\n"
            "Verifique se o nome do arquivo está correto dentro da pasta `data`."
        )

    df = pd.read_csv(csv_path)
    validate_required_columns(df)
    return df


def dataframe_to_samples(df: pd.DataFrame) -> list[RawRealEstateSample]:
    """Converte o DataFrame bruto em uma lista de amostras tipadas."""
    validate_required_columns(df)

    samples: list[RawRealEstateSample] = []

    for row in df.to_dict(orient="records"):
        sample = RawRealEstateSample(
            id=str(row["id"]),
            preco=float(row["preco"]),
            areaprivativa=float(row["areaprivativa"]),
            vagas=int(row["vagas"]),
            idadeaparente=float(row["idadeaparente"]),
            distanciacentrokm=float(row["distanciacentrokm"]),
            fontelink=str(row["fontelink"]),
            observacoes=str(row["observacoes"])
            if "observacoes" in row and pd.notna(row["observacoes"])
            else None,
        )
        samples.append(sample)

    return samples


def load_raw_samples_from_csv(csv_path: Path) -> list[RawRealEstateSample]:
    """Atalho didático para carregar e tipar a base em uma única chamada."""
    df = load_raw_dataset(csv_path)
    return dataframe_to_samples(df)
