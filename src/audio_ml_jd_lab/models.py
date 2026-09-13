"""Shared datatypes without heavy ML framework imports."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HFEmbeddingResult:
    model_id: str
    embedding_dim: int
    mean_activation: float
    std_activation: float
