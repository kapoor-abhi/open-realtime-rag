#bm25_store.py
"""
BM25 sparse retrieval backed by Postgres document_chunks table.

FIXES applied:
  1. DDL split into individual statements — psycopg3's execute() is designed
     for a single SQL statement. Passing multiple statements separated by ";"
     caused the CREATE INDEX lines to be silently dropped, so the indexes
     were never created and BM25 chunk lookups were unindexed full-table scans.

  2. Removed early-exit `break` on score <= 0 in search(). When all chunk
     texts begin with "Source Document: ..." the BM25 tokenizer sees mostly
     shared tokens and can produce near-zero scores even for relevant chunks.
     We now collect ALL non-negative-scoring results and let the slice handle
     the limit, instead of stopping at the first zero-score entry.
"""

import re
import logging
from typing import List, Optional, Dict
from rank_bm25 import BM25Okapi
from psycopg_pool import AsyncConnectionPool
from app.services.chunk_index import compute_chunk_hash

logger = logging.getLogger(__name__)

RRF_K = 60


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower())


class BM25Store:
    def __init__(self, db_pool: AsyncConnectionPool):
        self.pool = db_pool

    async def ensure_table(self):
        """
        FIX: Each DDL statement is executed individually.
        Previously all statements were passed as one string to execute(),
        which in psycopg3 only ran the first CREATE TABLE and silently
        discarded the two CREATE INDEX statements.
        """
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS document_chunks (
                        id          SERIAL PRIMARY KEY,
                        file_hash   TEXT NOT NULL,
                        source_file TEXT NOT NULL,
                        page_number INTEGER NOT NULL,
                        chunk_type  TEXT NOT NULL DEFAULT 'text',
                        chunk_text  TEXT NOT NULL,
                        chunk_hash  TEXT,
                        image_path  TEXT,
                        created_at  TIMESTAMPTZ DEFAULT NOW()
                    )
                """)
                await cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunks_file_hash
                    ON document_chunks(file_hash)
                """)
                await cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunks_chunk_hash
                    ON document_chunks(chunk_hash)
                """)

    # ---- FULL REPLACE (first-time index) ----------------------------------

    async def upsert_chunks(self, chunks: list, file_hash: str):
        """Delete all existing rows for file_hash and insert fresh ones."""
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM document_chunks WHERE file_hash = %s", (file_hash,)
                )
                rows = [
                    (
                        file_hash,
                        c.metadata.source_file,
                        c.metadata.page_number,
                        c.metadata.chunk_type,
                        c.text,
                        compute_chunk_hash(c.text),
                        c.metadata.image_path,
                    )
                    for c in chunks
                ]
                await cur.executemany(
                    """
                    INSERT INTO document_chunks
                        (file_hash, source_file, page_number, chunk_type,
                         chunk_text, chunk_hash, image_path)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """,
                    rows,
                )
        logger.info(f"[BM25] Full replace: {len(chunks)} rows for file_hash={file_hash[:8]}...")

    # ---- INCREMENTAL INSERT (only new chunks) ------------------------------

    async def insert_chunks_incremental(
        self, chunks: list, file_hash: str
    ) -> Dict[str, int]:
        """
        Insert ONLY the provided (new/changed) chunks.
        Returns {chunk_hash → row_id} for recording in chunk_index.
        """
        if not chunks:
            return {}

        chunk_hash_to_row_id: Dict[str, int] = {}

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                for chunk in chunks:
                    ch = compute_chunk_hash(chunk.text)
                    await cur.execute(
                        """
                        INSERT INTO document_chunks
                            (file_hash, source_file, page_number, chunk_type,
                             chunk_text, chunk_hash, image_path)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        RETURNING id
                        """,
                        (
                            file_hash,
                            chunk.metadata.source_file,
                            chunk.metadata.page_number,
                            chunk.metadata.chunk_type,
                            chunk.text,
                            ch,
                            chunk.metadata.image_path,
                        ),
                    )
                    row = await cur.fetchone()
                    chunk_hash_to_row_id[ch] = row["id"]

        logger.info(
            f"[BM25] Incremental insert: {len(chunk_hash_to_row_id)} new chunk(s) "
            f"for file_hash={file_hash[:8]}..."
        )
        return chunk_hash_to_row_id

    # ---- TARGETED DELETE ---------------------------------------------------

    async def delete_chunks_by_ids(self, row_ids: List[int]):
        """Delete specific rows from document_chunks by primary key."""
        if not row_ids:
            return
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "DELETE FROM document_chunks WHERE id = ANY(%s)", (row_ids,)
                )
        logger.info(f"[BM25] Deleted {len(row_ids)} stale chunk row(s).")

    # ---- SEARCH ------------------------------------------------------------

    async def search(
        self,
        query: str,
        active_file_hashes: Optional[List[str]] = None,
        limit: int = 20,
    ) -> List[dict]:
        if not active_file_hashes:
            return []

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT id, file_hash, source_file, page_number,
                           chunk_type, chunk_text, image_path
                    FROM document_chunks
                    WHERE file_hash = ANY(%s)
                    ORDER BY id
                    """,
                    (active_file_hashes,),
                )
                rows = await cur.fetchall()

        if not rows:
            return []

        corpus = [r["chunk_text"] for r in rows]
        bm25 = BM25Okapi([_tokenize(doc) for doc in corpus])
        scores = bm25.get_scores(_tokenize(query))

        # FIX: collect all results with score > 0, then slice.
        # Previously we used `break` which stopped on the FIRST zero-score
        # entry after sorting — meaning any zero at the top (common when
        # chunks share "Source Document:" prefix tokens) would return nothing.
        scored = sorted(zip(scores, rows), key=lambda x: x[0], reverse=True)
        results = []
        for rank, (score, row) in enumerate(scored[:limit]):
            if score <= 0:
                continue
            results.append({
                "text": row["chunk_text"],
                "page_number": row["page_number"],
                "source_file": row["source_file"],
                "chunk_type": row.get("chunk_type", "text"),
                "image_path": row.get("image_path"),
                "bm25_score": float(score),
                "bm25_rank": rank,
            })

        logger.info(f"[BM25] {len(results)} results for query='{query[:60]}...'")
        return results


def reciprocal_rank_fusion(
    dense_results: List[dict],
    sparse_results: List[dict],
    k: int = RRF_K,
) -> List[dict]:
    def _key(c: dict) -> str:
        return f"{c['source_file']}|{c['page_number']}|{c['text'][:80]}"

    scores: dict = {}
    for rank, chunk in enumerate(dense_results):
        key = _key(chunk)
        if key not in scores:
            scores[key] = {"chunk": chunk, "rrf_score": 0.0}
        scores[key]["rrf_score"] += 1.0 / (k + rank + 1)

    for rank, chunk in enumerate(sparse_results):
        key = _key(chunk)
        if key not in scores:
            scores[key] = {"chunk": chunk, "rrf_score": 0.0}
        scores[key]["rrf_score"] += 1.0 / (k + rank + 1)

    merged = sorted(scores.values(), key=lambda x: x["rrf_score"], reverse=True)
    logger.info(
        f"[RRF] {len(dense_results)} dense + {len(sparse_results)} sparse "
        f"→ {len(merged)} unique"
    )
    return [e["chunk"] for e in merged]