#semantic_cache.py
"""
Semantic cache backed by Qdrant.

Instead of exact SHA-256 key matching, this cache:
  1. Embeds the incoming query using the same Cohere model.
  2. Performs a nearest-neighbour search in a dedicated Qdrant collection.
  3. If the top result has cosine similarity >= THRESHOLD, returns the cached answer.
  4. Otherwise proceeds to the LLM pipeline, then stores the new result.

Why this is better than Redis exact-match:
  - "What is the Q3 revenue?" and "Show me Q3 revenue figures" → same cached answer.
  - Scoped per document-workspace (active_file_hashes), so document A's cache
    never bleeds into document B's answers.
  - No TTL churn: old entries degrade gracefully as new content is indexed.
"""

import uuid
import logging
import time
from typing import Optional, List
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from app.services.embeddings import get_embedding_model
from app.core.config import get_settings

logger = logging.getLogger(__name__)

CACHE_COLLECTION = "semantic_cache"


def _scope_key(active_hashes: Optional[List[str]]) -> str:
    """Deterministic scope string for a set of active documents."""
    if not active_hashes:
        return "global"
    return "_".join(sorted(active_hashes))


class SemanticCache:
    """
    Vector-based semantic cache stored in a dedicated Qdrant collection.
    Thread-safe for concurrent async usage.
    """

    def __init__(self, qdrant_client: AsyncQdrantClient):
        self.client = qdrant_client
        self.embeddings = get_embedding_model()
        self.settings = get_settings()
        self._collection_ready = False

    async def _ensure_collection(self):
        if self._collection_ready:
            return
        exists = await self.client.collection_exists(CACHE_COLLECTION)
        if not exists:
            await self.client.create_collection(
                collection_name=CACHE_COLLECTION,
                vectors_config=models.VectorParams(
                    size=1024,                    # Cohere embed-english-v3.0 dimension
                    distance=models.Distance.COSINE,
                ),
            )
            logger.info(f"[SEMANTIC CACHE] Created Qdrant collection '{CACHE_COLLECTION}'")
        self._collection_ready = True

    async def get(
        self,
        query: str,
        active_file_hashes: Optional[List[str]] = None,
    ) -> Optional[dict]:
        """
        Look up a semantically similar cached answer.

        Returns a dict with 'answer', 'citations', and 'similarity_score'
        if a sufficiently similar cached query is found, else None.
        """
        await self._ensure_collection()
        threshold = self.settings.SEMANTIC_CACHE_THRESHOLD
        scope = _scope_key(active_file_hashes)

        try:
            query_vector = await self.embeddings.aembed_query(query)

            results = await self.client.search(
                collection_name=CACHE_COLLECTION,
                query_vector=query_vector,
                query_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="scope",
                            match=models.MatchValue(value=scope),
                        )
                    ]
                ),
                limit=1,
                score_threshold=threshold,
                with_payload=True,
            )

            if results:
                hit = results[0]
                logger.info(
                    f"[SEMANTIC CACHE] HIT — similarity={hit.score:.4f} "
                    f"(threshold={threshold}) | query='{query[:60]}...'"
                )
                return {
                    "answer": hit.payload["answer"],
                    "citations": hit.payload["citations"],
                    "similarity_score": hit.score,
                }

            logger.info(
                f"[SEMANTIC CACHE] MISS — no result above {threshold} | query='{query[:60]}...'"
            )
            return None

        except Exception as e:
            # Cache failure must never break the main retrieval pipeline.
            logger.warning(f"[SEMANTIC CACHE] get() error (graceful degradation): {e}")
            return None

    async def set(
        self,
        query: str,
        active_file_hashes: Optional[List[str]],
        answer: str,
        citations: list,
    ) -> None:
        """
        Store a query-answer pair in the semantic cache.
        Idempotent: duplicate near-identical queries will simply add another point
        that subsequent searches may hit first.
        """
        await self._ensure_collection()
        scope = _scope_key(active_file_hashes)

        try:
            query_vector = await self.embeddings.aembed_query(query)
            await self.client.upsert(
                collection_name=CACHE_COLLECTION,
                points=[
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=query_vector,
                        payload={
                            "query": query,
                            "scope": scope,
                            "answer": answer,
                            "citations": citations,
                            "created_at": int(time.time()),
                        },
                    )
                ],
            )
            logger.info(f"[SEMANTIC CACHE] Stored answer for query='{query[:60]}...'")
        except Exception as e:
            logger.warning(f"[SEMANTIC CACHE] set() error (graceful degradation): {e}")

    async def invalidate_scope(self, active_file_hashes: List[str]) -> None:
        """
        Remove all cached entries for a given document scope.
        Call this when a document is re-indexed so stale answers are evicted.
        """
        await self._ensure_collection()
        scope = _scope_key(active_file_hashes)
        try:
            await self.client.delete(
                collection_name=CACHE_COLLECTION,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key="scope",
                                match=models.MatchValue(value=scope),
                            )
                        ]
                    )
                ),
            )
            logger.info(f"[SEMANTIC CACHE] Invalidated scope='{scope}'")
        except Exception as e:
            logger.warning(f"[SEMANTIC CACHE] invalidate_scope() error: {e}")
