"""Loader for flat disease dataset from CSV.

Creates a KnowledgeGraph with only leaf nodes - no parent hierarchy.
Each node represents a disease with associated symptoms in attrs.

CSV format: disease,symptoms,aliases
- disease: disease name (e.g. panic disorder)
- symptoms: semicolon-separated list of symptoms
- aliases: optional semicolon-separated alternatives for Oracle matching
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import List, Set, Tuple

from ...graph import KnowledgeGraph, Node
from ..types import DomainConfig, DISEASES_DOMAIN


def _slug(text: str) -> str:
    """Create a URL-safe slug from text."""
    s = re.sub(r"[^\w\s-]", "", text.lower())
    return re.sub(r"[-\s]+", "_", s).strip("_")


def _parse_list(value: str) -> List[str]:
    """Parse semicolon-separated string into list."""
    if not value or not value.strip():
        return []
    return [p.strip() for p in value.split(";") if p.strip()]


def load_flat_disease_graph(
    csv_path: Path,
    domain_config: DomainConfig = DISEASES_DOMAIN,
) -> Tuple[KnowledgeGraph, DomainConfig]:
    """Load a flat knowledge graph from diseases CSV.

    CSV columns: disease, symptoms, aliases
    - disease: disease name
    - symptoms: semicolon-separated list of associated symptoms
    - aliases: optional semicolon-separated alternatives for Oracle matching

    Creates only leaf nodes (diseases). No hierarchy. Symptoms stored in attrs.

    Args:
        csv_path: Path to CSV file.
        domain_config: Domain configuration. Defaults to DISEASES_DOMAIN.

    Returns:
        Tuple of (KnowledgeGraph with disease nodes only, DomainConfig).
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Diseases CSV not found: {csv_path}")

    nodes: Set[Node] = set()
    prefix = domain_config.node_id_prefix.rstrip(":")
    idx = 0

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            disease = (row.get("disease") or "").strip()
            if not disease:
                continue
            symptoms = _parse_list(row.get("symptoms") or "")
            aliases = _parse_list(row.get("aliases") or "")

            disease_slug = _slug(disease)
            node_id = f"{prefix}:{disease_slug}:{idx}"
            idx += 1

            attrs: dict = {
                "type": domain_config.leaf_type,
                "category": "medical",
            }
            if symptoms:
                attrs["symptoms"] = symptoms
            if aliases:
                attrs["aliases"] = aliases

            nodes.add(
                Node(
                    id=node_id,
                    label=disease,
                    attrs=attrs,
                )
            )

    return KnowledgeGraph(nodes=nodes, edges=set()), domain_config
