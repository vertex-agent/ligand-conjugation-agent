from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import DEFAULT_CONFIG, AgentConfig
from .models import EvidenceRecord

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None


@dataclass
class RetrievedEvidence:
    record: EvidenceRecord
    relevance: float


class EvidenceRetriever:
    def __init__(
        self,
        records: List[EvidenceRecord],
        config: Optional[AgentConfig] = None,
    ) -> None:
        self.records = records
        self.config = config or DEFAULT_CONFIG
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        texts = [r.retrieval_text for r in records]
        self.matrix = self.vectorizer.fit_transform(texts) if records else None
        self.dense_matrix: Optional[np.ndarray] = None
        if records and self.config.embedding_enabled:
            self.dense_matrix = self._load_or_compute_embeddings(texts)

    def _cache_path(self, texts: List[str]) -> Path:
        digest = hashlib.sha256("\n".join(texts).encode("utf-8")).hexdigest()[:16]
        cache_dir = Path(self.config.cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"embeddings_{digest}_{self.config.embedding_model.replace('/', '_')}.pkl"

    def _load_or_compute_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        cache_path = self._cache_path(texts)
        if cache_path.exists():
            try:
                with cache_path.open("rb") as handle:
                    cached = pickle.load(handle)
                if isinstance(cached, np.ndarray) and cached.shape[0] == len(texts):
                    return cached
            except Exception:
                pass

        embeddings = self._compute_embeddings(texts)
        if embeddings is not None:
            try:
                with cache_path.open("wb") as handle:
                    pickle.dump(embeddings, handle)
            except Exception:
                pass
        return embeddings

    def _compute_embeddings(self, texts: List[str]) -> Optional[np.ndarray]:
        if not self.config.openai_api_key or OpenAI is None:
            return None
        try:
            client = OpenAI(api_key=self.config.openai_api_key)
            batch_size = 64
            vectors: list[list[float]] = []
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                response = client.embeddings.create(
                    model=self.config.embedding_model,
                    input=batch,
                )
                vectors.extend(item.embedding for item in response.data)
            return np.array(vectors, dtype=np.float32)
        except Exception:
            return None

    def _tfidf_scores(self, query: str) -> np.ndarray:
        if not self.records or self.matrix is None:
            return np.array([])
        q_vec = self.vectorizer.transform([query])
        return cosine_similarity(q_vec, self.matrix).flatten()

    def _dense_scores(self, query: str) -> Optional[np.ndarray]:
        if self.dense_matrix is None or not self.config.openai_api_key or OpenAI is None:
            return None
        try:
            client = OpenAI(api_key=self.config.openai_api_key)
            response = client.embeddings.create(
                model=self.config.embedding_model,
                input=[query],
            )
            q_vec = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
            return cosine_similarity(q_vec, self.dense_matrix).flatten()
        except Exception:
            return None

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        ligand_filter: str = "",
        nanoparticle_filter: str = "",
        chemistry_filter: str = "",
    ) -> List[RetrievedEvidence]:
        if not self.records or self.matrix is None:
            return []

        tfidf = self._tfidf_scores(query)
        dense = self._dense_scores(query)

        if dense is not None and len(dense) == len(tfidf):
            alpha = self.config.hybrid_alpha
            beta = self.config.hybrid_beta
            sims = alpha * dense + beta * tfidf
        else:
            sims = tfidf

        ranked = sorted(
            enumerate(sims),
            key=lambda x: (x[1], self.records[x[0]].quality_score),
            reverse=True,
        )

        results: list[RetrievedEvidence] = []
        for idx, score in ranked:
            if score <= 0:
                continue
            record = self.records[idx]
            if ligand_filter and ligand_filter.lower() not in record.ligand_class.lower():
                continue
            if nanoparticle_filter and nanoparticle_filter.lower() not in record.nanoparticle_system.lower():
                continue
            if chemistry_filter and chemistry_filter.lower() not in record.conjugation_chemistry.lower():
                continue
            results.append(RetrievedEvidence(record=record, relevance=float(score)))
            if len(results) >= top_k:
                break
        return results

    def filter_by_compatibility(self, ligand: str, nanoparticle: str, top_k: int = 5) -> List[RetrievedEvidence]:
        query = f"{ligand} {nanoparticle} conjugation compatibility"
        return self.retrieve(
            query,
            top_k=top_k,
            ligand_filter=ligand,
            nanoparticle_filter=nanoparticle,
        )

    def get_protocol_parameters(self, chemistry: str, top_k: int = 5) -> List[RetrievedEvidence]:
        query = f"{chemistry} protocol conditions pH buffer incubation"
        return self.retrieve(query, top_k=top_k, chemistry_filter=chemistry)
