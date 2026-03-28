"""Approach 1: sentence embedding cosine similarity (e.g. Sentence-BERT)."""

from __future__ import annotations

from typing import Literal

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

InternalLabel = Literal["supported", "contradicted", "not_supported"]


class EmbeddingBaseline:
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,
    ):
        self.model = SentenceTransformer(model_name, device=device)

    def similarity(self, claim: str, evidence: str) -> float:
        e = self.model.encode(
            [claim, evidence],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return float(cosine_similarity([e[0]], [e[1]])[0, 0])

    def predict_three_way(
        self,
        claim: str,
        evidence: str,
        *,
        threshold: float = 0.35,
    ) -> InternalLabel:
        """
        Similarity-only heuristic: low cosine → not_supported; otherwise supported.
        Contradiction is not identifiable from cosine alone, so it is never returned.
        """
        s = self.similarity(claim, evidence)
        if s < threshold:
            return "not_supported"
        return "supported"

    def predict_binary_unsupported(
        self, claim: str, evidence: str, *, threshold: float = 0.35
    ) -> bool:
        """True if treated as unsupported / hallucinated (below threshold)."""
        return self.similarity(claim, evidence) < threshold
