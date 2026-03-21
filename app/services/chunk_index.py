#chunk_index.py
"""
Chunk-level incremental indexing service.

FIX: DDL split into individual statements — psycopg3's execute() handles
one statement at a time. The original combined DDL string caused the
CREATE INDEX statement to be silently dropped.
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import List, Optional, Dict, Set

from psycopg_pool import AsyncConnectionPool
from app.models.schemas import DocumentChunk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def compute_chunk_hash(text: str) -> str:
    """SHA-256 of the chunk text. Used as the identity key for a chunk."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass
class IndexedChunkRecord:
    chunk_hash: str
    qdrant_point_id: str
    bm25_row_id: Optional[int]


@dataclass
class DiffResult:
    """Result of diffing new chunks against the stored index."""
    chunks_to_add: List[DocumentChunk]
    chunk_hashes_to_add: List[str]
    qdrant_ids_to_delete: List[str]
    bm25_ids_to_delete: List[int]
    chunk_hashes_to_delete: List[str]
    unchanged_count: int


# ---------------------------------------------------------------------------
# SERVICE
# ---------------------------------------------------------------------------

class ChunkIndexService:

    def __init__(self, db_pool: AsyncConnectionPool):
        self.pool = db_pool

    async def ensure_table(self):
        """
        FIX: Split into two separate execute() calls.
        Previously both statements were sent in one string; psycopg3 only
        executed the first CREATE TABLE and silently discarded the CREATE INDEX.
        """
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("""
                    CREATE TABLE IF NOT EXISTS chunk_index (
                        id              SERIAL PRIMARY KEY,
                        file_hash       TEXT NOT NULL,
                        chunk_hash      TEXT NOT NULL,
                        qdrant_point_id TEXT NOT NULL,
                        bm25_row_id     INTEGER,
                        source_file     TEXT NOT NULL,
                        page_number     INTEGER NOT NULL,
                        chunk_type      TEXT NOT NULL DEFAULT 'text',
                        created_at      TIMESTAMPTZ DEFAULT NOW(),
                        CONSTRAINT uq_chunk_index_doc_chunk UNIQUE (file_hash, chunk_hash)
                    )
                """)
                await cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_chunk_index_file_hash
                    ON chunk_index (file_hash)
                """)
        logger.debug("[CHUNK INDEX] Table ensured.")

    # ---- READ ---------------------------------------------------------------

    async def load_existing(self, file_hash: str) -> Dict[str, IndexedChunkRecord]:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT chunk_hash, qdrant_point_id, bm25_row_id
                    FROM chunk_index
                    WHERE file_hash = %s
                    """,
                    (file_hash,),
                )
                rows = await cur.fetchall()

        result: Dict[str, IndexedChunkRecord] = {}
        for row in rows:
            result[row["chunk_hash"]] = IndexedChunkRecord(
                chunk_hash=row["chunk_hash"],
                qdrant_point_id=row["qdrant_point_id"],
                bm25_row_id=row.get("bm25_row_id"),
            )
        return result

    # ---- DIFF ---------------------------------------------------------------

    async def compute_diff(
        self,
        new_chunks: List[DocumentChunk],
        file_hash: str,
    ) -> DiffResult:
        existing: Dict[str, IndexedChunkRecord] = await self.load_existing(file_hash)

        new_hash_to_chunk: Dict[str, DocumentChunk] = {}
        for chunk in new_chunks:
            h = compute_chunk_hash(chunk.text)
            if h not in new_hash_to_chunk:
                new_hash_to_chunk[h] = chunk

        new_hashes: Set[str] = set(new_hash_to_chunk.keys())
        existing_hashes: Set[str] = set(existing.keys())

        to_add_hashes = new_hashes - existing_hashes
        to_delete_hashes = existing_hashes - new_hashes
        unchanged = new_hashes & existing_hashes

        chunks_to_add = [new_hash_to_chunk[h] for h in to_add_hashes]
        qdrant_ids_to_delete = [existing[h].qdrant_point_id for h in to_delete_hashes]
        bm25_ids_to_delete = [
            existing[h].bm25_row_id
            for h in to_delete_hashes
            if existing[h].bm25_row_id is not None
        ]

        logger.info(
            f"\n{'='*60}\n"
            f"[CHUNK INDEX DIFF] file_hash={file_hash[:8]}...\n"
            f"  Total new chunks  : {len(new_chunks)}\n"
            f"  Unique new hashes : {len(new_hashes)}\n"
            f"  Unchanged (skip)  : {len(unchanged)}\n"
            f"  To ADD            : {len(to_add_hashes)}\n"
            f"  To DELETE         : {len(to_delete_hashes)}\n"
            f"{'='*60}"
        )

        return DiffResult(
            chunks_to_add=chunks_to_add,
            chunk_hashes_to_add=list(to_add_hashes),
            qdrant_ids_to_delete=qdrant_ids_to_delete,
            bm25_ids_to_delete=bm25_ids_to_delete,
            chunk_hashes_to_delete=list(to_delete_hashes),
            unchanged_count=len(unchanged),
        )

    # ---- WRITE --------------------------------------------------------------

    async def record_added_chunks(
        self,
        file_hash: str,
        chunk_hash_to_qdrant_id: Dict[str, str],
        chunk_hash_to_bm25_id: Dict[str, int],
        new_chunk_map: Dict[str, DocumentChunk],
    ):
        rows = []
        for chunk_hash, qdrant_id in chunk_hash_to_qdrant_id.items():
            chunk = new_chunk_map.get(chunk_hash)
            if not chunk:
                continue
            rows.append((
                file_hash,
                chunk_hash,
                qdrant_id,
                chunk_hash_to_bm25_id.get(chunk_hash),
                chunk.metadata.source_file,
                chunk.metadata.page_number,
                chunk.metadata.chunk_type,
            ))

        if not rows:
            return

        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.executemany(
                    """
                    INSERT INTO chunk_index
                        (file_hash, chunk_hash, qdrant_point_id, bm25_row_id,
                         source_file, page_number, chunk_type)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (file_hash, chunk_hash) DO NOTHING
                    """,
                    rows,
                )
        logger.info(f"[CHUNK INDEX] Recorded {len(rows)} new chunk entries.")

    async def remove_deleted_chunks(
        self,
        file_hash: str,
        chunk_hashes: List[str],
    ):
        if not chunk_hashes:
            return
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    DELETE FROM chunk_index
                    WHERE file_hash = %s AND chunk_hash = ANY(%s)
                    """,
                    (file_hash, chunk_hashes),
                )
        logger.info(
            f"[CHUNK INDEX] Removed {len(chunk_hashes)} stale chunk records "
            f"for file_hash={file_hash[:8]}..."
        )

    async def get_summary(self, file_hash: str) -> dict:
        async with self.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    """
                    SELECT
                        chunk_type,
                        COUNT(*) AS count,
                        MIN(page_number) AS first_page,
                        MAX(page_number) AS last_page
                    FROM chunk_index
                    WHERE file_hash = %s
                    GROUP BY chunk_type
                    ORDER BY chunk_type
                    """,
                    (file_hash,),
                )
                rows = await cur.fetchall()
        return {
            row["chunk_type"]: {
                "count": row["count"],
                "page_range": f"{row['first_page']}–{row['last_page']}",
            }
            for row in rows
        }