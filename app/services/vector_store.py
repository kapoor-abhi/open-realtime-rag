#vector_store.py
"""
Hybrid retrieval: Dense (Qdrant) + Sparse (BM25) + Cohere Reranker.

New in this version:
  - upsert_chunks_incremental(): embeds ONLY new/changed chunks, returns
    {chunk_hash → qdrant_point_id} for the chunk_index table.
  - delete_points(): removes specific Qdrant points by UUID list.
"""

import asyncio
import logging
import uuid as _uuid
from typing import List, Optional, Dict
import cohere
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models
from psycopg_pool import AsyncConnectionPool

from app.models.schemas import DocumentChunk
from app.services.embeddings import get_embedding_model
from app.services.bm25_store import BM25Store, reciprocal_rank_fusion
from app.services.chunk_index import compute_chunk_hash
from app.core.config import get_settings

logger = logging.getLogger(__name__)

DENSE_CANDIDATES = 20
SPARSE_CANDIDATES = 20
FINAL_TOP_N = 5


class QdrantService:
    def __init__(
        self,
        client: AsyncQdrantClient,
        db_pool: Optional[AsyncConnectionPool] = None,
        collection_name: str = "multirag_docs",
    ):
        self.client = client
        self.collection_name = collection_name
        self.embeddings = get_embedding_model()
        self.bm25 = BM25Store(db_pool) if db_pool else None
        self.settings = get_settings()

    async def init_collection(self):
        exists = await self.client.collection_exists(self.collection_name)
        if not exists:
            await self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
            )

    async def upsert_chunks(self, chunks: List[DocumentChunk]):
        """Full upsert — used on first-time indexing (no chunk_index exists yet)."""
        texts = [c.text for c in chunks]
        vectors = await self.embeddings.aembed_documents(texts)
        points = [
            models.PointStruct(
                id=str(_uuid.uuid4()),
                vector=v,
                payload={"text": c.text, **c.metadata.model_dump()},
            )
            for c, v in zip(chunks, vectors)
        ]
        await self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"[QDRANT] Full upsert: {len(points)} vectors")

    async def upsert_chunks_incremental(
        self, chunks: List[DocumentChunk]
    ) -> Dict[str, str]:
        """
        Embed and upsert ONLY the provided chunks (already diffed as new/changed).

        Returns {chunk_hash: qdrant_point_id} for recording in chunk_index.
        """
        if not chunks:
            return {}

        texts = [c.text for c in chunks]
        vectors = await self.embeddings.aembed_documents(texts)

        chunk_hash_to_point_id: Dict[str, str] = {}
        points = []
        for chunk, vector in zip(chunks, vectors):
            point_id = str(_uuid.uuid4())
            ch = compute_chunk_hash(chunk.text)
            chunk_hash_to_point_id[ch] = point_id
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={"text": chunk.text, **chunk.metadata.model_dump()},
                )
            )

        await self.client.upsert(collection_name=self.collection_name, points=points)
        logger.info(f"[QDRANT] Incremental upsert: {len(points)} chunk(s) embedded.")
        return chunk_hash_to_point_id

    async def delete_points(self, point_ids: List[str]):
        """Delete specific Qdrant vectors by their UUIDs."""
        if not point_ids:
            return
        await self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(points=point_ids),
        )
        logger.info(f"[QDRANT] Deleted {len(point_ids)} stale vector(s).")

    async def _dense_search(
        self,
        query: str,
        active_file_hashes: Optional[List[str]],
        page_number: Optional[int],
        limit: int,
    ) -> List[dict]:
        query_embedding = await self.embeddings.aembed_query(query)

        filter_conditions = []
        if active_file_hashes:
            filter_conditions.append(
                models.FieldCondition(
                    key="file_hash",
                    match=models.MatchAny(any=active_file_hashes),
                )
            )
        if page_number is not None:
            filter_conditions.append(
                models.FieldCondition(
                    key="page_number",
                    match=models.MatchValue(value=page_number),
                )
            )
            limit = max(limit, 20)

        query_filter = models.Filter(must=filter_conditions) if filter_conditions else None

        try:
            results = await self.client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                query_filter=query_filter,
                limit=limit,
            )
        except Exception as e:
            if "Not found" in str(e) or "doesn't exist" in str(e):
                return []
            raise

        return [
            {
                "text": h.payload["text"],
                "page_number": h.payload["page_number"],
                "source_file": h.payload["source_file"],
                "chunk_type": h.payload.get("chunk_type", "text"),
                "image_path": h.payload.get("image_path"),
                "dense_score": h.score,
            }
            for h in results
        ]

    async def _cohere_rerank(self, query: str, chunks: List[dict]) -> List[dict]:
        if not chunks:
            return []
        try:
            co = cohere.AsyncClient(api_key=self.settings.COHERE_API_KEY)
            response = await asyncio.wait_for(
                co.rerank(
                    model="rerank-english-v3.0",
                    query=query,
                    documents=[c["text"] for c in chunks],
                    top_n=FINAL_TOP_N,
                    return_documents=False,
                ),
                timeout=15.0,
            )
            reranked = []
            for r in response.results:
                chunk = chunks[r.index].copy()
                chunk["rerank_score"] = r.relevance_score
                reranked.append(chunk)
            logger.info(f"[COHERE RERANK] {len(chunks)} → {len(reranked)}")
            return reranked
        except Exception as e:
            logger.warning(f"[COHERE RERANK] Fallback ({e})")
            return chunks[:FINAL_TOP_N]

    async def search(
        self,
        query: str,
        limit: int = FINAL_TOP_N,
        page_number: Optional[int] = None,
        active_file_hashes: Optional[List[str]] = None,
    ) -> List[dict]:
        """Dense + BM25 → RRF → Cohere Rerank → top-N"""
        dense_task = self._dense_search(query, active_file_hashes, page_number, DENSE_CANDIDATES)

        if self.bm25:
            sparse_task = self.bm25.search(query, active_file_hashes, SPARSE_CANDIDATES)
        else:
            async def _empty(): return []
            sparse_task = _empty()

        dense_results, sparse_results = await asyncio.gather(
            dense_task, sparse_task, return_exceptions=True
        )
        if isinstance(dense_results, Exception):
            dense_results = []
        if isinstance(sparse_results, Exception):
            sparse_results = []

        fused = reciprocal_rank_fusion(dense_results, sparse_results)
        if not fused:
            return []

        return await self._cohere_rerank(query, fused)
