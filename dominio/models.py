"""
Modelos centrais do domínio da avaliação imobiliária.

Este módulo define os contratos de dados usados no treinamento.
A ideia é representar, de forma clara e validada, duas etapas da amostra:

1. coleta bruta do imóvel;
2. coleta já convertida para valor unitário (R$/m²).

Isso ajuda o aluno a enxergar a diferença entre dado coletado
e dado preparado para análise estatística.
"""

from dataclasses import dataclass
from typing import Optional


# ============================================================
# AMOSTRA BRUTA
# ============================================================
# Esta classe representa o imóvel como ele foi coletado em campo
# ou em portais imobiliários, antes de qualquer tratamento
# estatístico mais avançado.
#
# frozen=True torna a instância imutável após a criação.
# Isso é útil para preservar a integridade do dado.
@dataclass(frozen=True)
class RawRealEstateSample:
    """Representa uma coleta bruta de imóvel residencial."""

    # Identificador único da amostra.
    id: str

    # Preço total anunciado ou observado para o imóvel.
    preco: float

    # Área privativa do imóvel, em metros quadrados.
    areaprivativa: float

    # Número de vagas de garagem.
    vagas: int

    # Idade aparente do imóvel, em anos.
    idadeaparente: float

    # Distância aproximada até o centro urbano, em quilômetros.
    distanciacentrokm: float

    # Link da fonte da coleta, para rastreabilidade.
    fontelink: str

    # Observações livres do avaliador/coletor.
    observacoes: Optional[str] = None

    def __post_init__(self) -> None:
        """
        Executa validações logo após a criação do objeto.

        A ideia aqui é impedir que uma amostra fisicamente ou
        financeiramente incoerente entre no sistema.
        """

        # O identificador não pode ser vazio.
        if not self.id.strip():
            raise ValueError("O identificador da amostra não pode ser vazio.")

        # O preço deve ser positivo.
        if self.preco <= 0:
            raise ValueError(
                f"Preço inválido para a amostra {self.id}: R$ {self.preco}."
            )

        # A área privativa deve ser positiva.
        # Isso também evita problemas futuros ao calcular R$/m².
        if self.areaprivativa <= 0:
            raise ValueError(
                f"Área privativa inválida para a amostra {self.id}: "
                f"{self.areaprivativa} m²."
            )

        # O imóvel pode ter zero vagas, mas nunca vagas negativas.
        if self.vagas < 0:
            raise ValueError(
                f"Número de vagas inválido para a amostra {self.id}: {self.vagas}."
            )

        # A idade aparente não pode ser negativa.
        if self.idadeaparente < 0:
            raise ValueError(
                f"Idade aparente inválida para a amostra {self.id}: "
                f"{self.idadeaparente} anos."
            )

        # A distância até o centro também não pode ser negativa.
        if self.distanciacentrokm < 0:
            raise ValueError(
                f"Distância inválida para a amostra {self.id}: "
                f"{self.distanciacentrokm} km."
            )


# ============================================================
# AMOSTRA UNITARIZADA
# ============================================================
# Esta classe representa a mesma amostra, mas agora já convertida
# para valor unitário. Em avaliação imobiliária, isso normalmente
# significa trabalhar com R$/m² para tornar a série comparável.
@dataclass(frozen=True)
class UnitizedRealEstateSample:
    """Representa uma amostra já convertida para valor unitário."""

    # Identificador da amostra original.
    id: str

    # Preço total do imóvel.
    preco: float

    # Área privativa do imóvel.
    areaprivativa: float

    # Valor unitário calculado, normalmente em R$/m².
    valor_unitario: float

    # Variáveis explicativas mantidas para análise posterior.
    vagas: int
    idadeaparente: float
    distanciacentrokm: float

    # Fonte original da coleta.
    fontelink: str

    # Campo opcional para observações adicionais.
    observacoes: Optional[str] = None

    def __post_init__(self) -> None:
        """
        Valida a coerência mínima da amostra após a unitarização.
        """
        # O valor unitário precisa ser positivo.
        if self.valor_unitario <= 0:
            raise ValueError(
                f"Valor unitário inválido para a amostra {self.id}: "
                f"R$ {self.valor_unitario}/m²."
            )
