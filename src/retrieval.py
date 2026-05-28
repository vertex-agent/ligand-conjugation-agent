from __future__ import annotations

from dataclasses import dataclass
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .models import EvidenceRecord


@dataclass
class RetrievedEvidence:
    record: EvidenceRecord
    relevance: float


class EvidenceRetriever:
    def __init__(self, records: List[EvidenceRecord]) -> None:
        self.records = records
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.matrix = self.vectorizer.fit_transform([r.retrieval_text for r in records]) if records else None

    def retrieve(self, query: str, top_k: int = 8) -> List[RetrievedEvidence]:
        if not self.records or self.matrix is None:
            return []

        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.matrix).flatten()

        ranked = sorted(
            enumerate(sims),
            key=lambda x: (x[1], self.records[x[0]].quality_score),
            reverse=True,
        )[:top_k]

        return [
            RetrievedEvidence(record=self.records[idx], relevance=float(score))
            for idx, score in ranked
            if score > 0
        ]
