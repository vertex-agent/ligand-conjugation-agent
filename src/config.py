from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    openai_api_key: str
    openai_model: str
    embedding_model: str
    embedding_enabled: bool
    hybrid_alpha: float  # weight for dense embeddings
    hybrid_beta: float  # weight for TF-IDF
    react_max_steps: int
    min_evidence_count: int
    min_confidence_score: float
    cache_dir: str

    @classmethod
    def from_env(cls) -> "AgentConfig":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        embedding_flag = os.getenv("VERTEX_EMBEDDINGS", "auto").strip().lower()
        if embedding_flag == "auto":
            embedding_enabled = bool(api_key)
        else:
            embedding_enabled = embedding_flag in {"1", "true", "yes", "on"}

        return cls(
            openai_api_key=api_key,
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o").strip() or "gpt-4o",
            embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"
            ).strip()
            or "text-embedding-3-small",
            embedding_enabled=embedding_enabled,
            hybrid_alpha=float(os.getenv("VERTEX_HYBRID_ALPHA", "0.65")),
            hybrid_beta=float(os.getenv("VERTEX_HYBRID_BETA", "0.35")),
            react_max_steps=int(os.getenv("VERTEX_REACT_MAX_STEPS", "5")),
            min_evidence_count=int(os.getenv("VERTEX_MIN_EVIDENCE", "2")),
            min_confidence_score=float(os.getenv("VERTEX_MIN_CONFIDENCE", "0.1")),
            cache_dir=os.getenv("VERTEX_CACHE_DIR", ".cache"),
        )


DEFAULT_CONFIG = AgentConfig.from_env()
