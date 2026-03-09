"""Domain configuration for different game datasets (geo, flat objects)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DomainConfig:
    """Configuracao do dominio do jogo.

    Atributos:
        leaf_type: Tipo de no folha ("city" | "object").
        node_id_prefix: Prefixo dos IDs dos nos folha ("city:" | "object:").
        target_noun: Termo para o alvo nos prompts ("city" | "object").
        domain_description: Descricao do dominio para prompts.
    """

    leaf_type: str
    node_id_prefix: str
    target_noun: str
    domain_description: str


# Configuracoes pre-definidas

GEO_DOMAIN = DomainConfig(
    leaf_type="city",
    node_id_prefix="city:",
    target_noun="city",
    domain_description="geographic (cities, countries, regions)",
)

OBJECTS_DOMAIN = DomainConfig(
    leaf_type="object",
    node_id_prefix="object:",
    target_noun="object",
    domain_description="concrete objects (sports, animals, fruits, vehicles, etc.)",
)

DISEASES_DOMAIN = DomainConfig(
    leaf_type="disease",
    node_id_prefix="disease:",
    target_noun="disease",
    domain_description="medical conditions (diseases with associated symptoms)",
)
