"""Modelos centrais do domínio da avaliação imobiliária.

Este módulo define os contratos de dados utilizados no treinamento.
A proposta didática é transformar a coleta de campo em estruturas
claras, tipadas e validadas, para que o aluno enxergue com nitidez
o que é uma amostra bruta e o que é uma amostra já unitarizada.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RawRealEstateSample:
    """Representa uma coleta bruta de imóvel residencial.

    Cada instância corresponde a uma linha da base coletada em campo
    ou em portais imobiliários, antes de tratamentos estatísticos
    mais avançados.
    """

    id: str
    preco: float
    areaprivativa: float
    vagas: int
    idadeaparente: float
    distanciacentrokm: float
    fontelink: str
    observacoes: Optional[str] = None

    def __post_init__(self) -> None:
        """Aplica validações mínimas de consistência física e financeira."""
        if not self.id.strip():
            raise ValueError("O identificador da amostra não pode ser vazio.")

        if self.preco <= 0:
            raise ValueError(
                f"Preço inválido para a amostra {self.id}: R$ {self.preco}."
            )

        if self.areaprivativa <= 0:
            raise ValueError(
                f"Área privativa inválida para a amostra {self.id}: "
                f"{self.areaprivativa} m²."
            )

        if self.vagas < 0:
            raise ValueError(
                f"Número de vagas inválido para a amostra {self.id}: {self.vagas}."
            )

        if self.idadeaparente < 0:
            raise ValueError(
                f"Idade aparente inválida para a amostra {self.id}: "
                f"{self.idadeaparente} anos."
            )

        if self.distanciacentrokm < 0:
            raise ValueError(
                f"Distância inválida para a amostra {self.id}: "
                f"{self.distanciacentrokm} km."
            )


@dataclass(frozen=True)
class UnitizedRealEstateSample:
    """Representa uma amostra já convertida para valor unitário.

    Esta estrutura é útil para deixar explícita a passagem entre
    a coleta bruta e a série estatística homogênea em R$/m².
    """

    id: str
    preco: float
    areaprivativa: float
    valor_unitario: float
    vagas: int
    idadeaparente: float
    distanciacentrokm: float
    fontelink: str
    observacoes: Optional[str] = None

    def __post_init__(self) -> None:
        """Valida a coerência da amostra após a unitarização."""
        if self.valor_unitario <= 0:
            raise ValueError(
                f"Valor unitário inválido para a amostra {self.id}: "
                f"R$ {self.valor_unitario}/m²."
            )
