#worker.py
"""
RQ background worker — with chunk-level incremental indexing.

FIX: langfuse.flush() is now called in the finally block of
async_process_document. The RQ worker process is short-lived — without
an explicit flush, the Langfuse SDK's background thread may not have time
to ship all telemetry before the process exits, causing silent trace loss.
"""

import asyncio
import json
import logging
from psycopg_pool import AsyncConnectionPool
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis as AsyncRedis
from psycopg.rows import dict_row

from app.core.config import get_settings
from app.services.parser import DocumentParser
from app.services.vector_store import QdrantService
from app.services.bm25_store import BM25Store
from app.services.chunk_index import ChunkIndexService, compute_chunk_hash

logger = logging.getLogger(__name__)


async def _publish(redis: AsyncRedis, file_hash: str, status: str):
    channel = f"doc_status:{file_hash}"
    await redis.publish(channel, json.dumps({"status": status, "file_hash": file_hash}))
    logger.info(f"[WORKER] Published status='{status}' on channel '{channel}'")


async def async_process_document(
    file_path: str, source_file_name: str, file_hash: str
):
    settings = get_settings()
    postgres_uri = (
        f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )

    db_pool = AsyncConnectionPool(
        postgres_uri, open=False,
        kwargs={"autocommit": True, "row_factory": dict_row},
    )
    await db_pool.open(wait=True)

    qdrant_client = AsyncQdrantClient(url=settings.QDRANT_URL)
    redis_client = AsyncRedis.from_url(settings.REDIS_BROKER_URL)

    try:
        # ── 1. Mark PROCESSING ──────────────────────────────────────────────
        async with db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE documents SET status = %s WHERE file_hash = %s",
                    ("PROCESSING", file_hash),
                )
        await _publish(redis_client, file_hash, "PROCESSING")

        # ── 2. Parse document (text + tables + images) ──────────────────────
        parser = DocumentParser()
        new_chunks = parser.parse_document(file_path, source_file_name, file_hash)

        logger.info(
            f"\n{'='*60}\n"
            f"[INDEXING] '{source_file_name}' — {len(new_chunks)} chunks parsed\n"
            f"{'='*60}"
        )
        for i, c in enumerate(new_chunks[:5]):
            logger.info(
                f"  Chunk {i+1} | Page {c.metadata.page_number} "
                f"| {c.metadata.chunk_type}: {c.text[:100]}..."
            )

        # ── 3. Ensure all tables exist ──────────────────────────────────────
        qdrant_service = QdrantService(client=qdrant_client, db_pool=db_pool)
        await qdrant_service.init_collection()

        bm25_store = BM25Store(db_pool)
        await bm25_store.ensure_table()

        chunk_index = ChunkIndexService(db_pool)
        await chunk_index.ensure_table()

        # ── 4. Diff: new vs stored chunks ───────────────────────────────────
        diff = await chunk_index.compute_diff(new_chunks, file_hash)

        logger.info(
            f"[DIFF] Unchanged={diff.unchanged_count} | "
            f"To add={len(diff.chunks_to_add)} | "
            f"To delete={len(diff.qdrant_ids_to_delete)}"
        )

        # ── 5. Delete stale chunks ──────────────────────────────────────────
        if diff.qdrant_ids_to_delete:
            await qdrant_service.delete_points(diff.qdrant_ids_to_delete)

        if diff.bm25_ids_to_delete:
            await bm25_store.delete_chunks_by_ids(diff.bm25_ids_to_delete)

        if diff.chunk_hashes_to_delete:
            await chunk_index.remove_deleted_chunks(file_hash, diff.chunk_hashes_to_delete)

        # ── 6. Embed and insert only new/changed chunks ─────────────────────
        chunk_hash_to_qdrant_id: dict = {}
        chunk_hash_to_bm25_id: dict = {}

        if diff.chunks_to_add:
            chunk_hash_to_qdrant_id = await qdrant_service.upsert_chunks_incremental(
                diff.chunks_to_add
            )
            chunk_hash_to_bm25_id = await bm25_store.insert_chunks_incremental(
                diff.chunks_to_add, file_hash
            )

        # ── 7. Record new chunk entries in chunk_index ──────────────────────
        if chunk_hash_to_qdrant_id:
            new_chunk_map = {
                compute_chunk_hash(c.text): c for c in diff.chunks_to_add
            }
            await chunk_index.record_added_chunks(
                file_hash=file_hash,
                chunk_hash_to_qdrant_id=chunk_hash_to_qdrant_id,
                chunk_hash_to_bm25_id=chunk_hash_to_bm25_id,
                new_chunk_map=new_chunk_map,
            )

        # ── 8. Log summary ──────────────────────────────────────────────────
        summary = await chunk_index.get_summary(file_hash)
        logger.info(
            f"\n[INDEX SUMMARY] '{source_file_name}'\n"
            + "\n".join(
                f"  {ctype}: {info['count']} chunks | pages {info['page_range']}"
                for ctype, info in summary.items()
            )
        )

        # ── 9. Mark COMPLETED ───────────────────────────────────────────────
        async with db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE documents SET status = %s WHERE file_hash = %s",
                    ("COMPLETED", file_hash),
                )
        await _publish(redis_client, file_hash, "COMPLETED")
        logger.info(f"[WORKER] Done: '{source_file_name}'")

    except Exception as e:
        logger.error(f"[WORKER] Failed: '{source_file_name}': {e}", exc_info=True)
        try:
            async with db_pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "UPDATE documents SET status = %s WHERE file_hash = %s",
                        ("FAILED", file_hash),
                    )
            await _publish(redis_client, file_hash, "FAILED")
        except Exception as inner:
            logger.error(f"[WORKER] Could not mark FAILED: {inner}")

    finally:
        await db_pool.close()
        await qdrant_client.close()
        await redis_client.aclose()

        # FIX: Flush Langfuse telemetry before the worker process exits.
        # Without this, the SDK's background thread may not finish shipping
        # all spans/generations before the short-lived RQ job process dies.
        try:
            from langfuse import get_client
            get_client().flush()
        except Exception as flush_err:
            logger.warning(f"[WORKER] Langfuse flush failed (non-fatal): {flush_err}")


def process_document(file_path: str, source_file_name: str, file_hash: str):
    """Synchronous RQ entry point. Runs async logic in an isolated event loop."""
    logger.info(f"[WORKER] Job started: '{source_file_name}' (hash={file_hash[:8]}...)")
    asyncio.run(async_process_document(file_path, source_file_name, file_hash))