#worker.py
import asyncio
import logging
from rq import get_current_job
from psycopg_pool import AsyncConnectionPool
from qdrant_client import AsyncQdrantClient
from psycopg.rows import dict_row
from app.core.config import get_settings
from app.services.parser import DocumentParser
from app.services.vector_store import QdrantService

logger = logging.getLogger(__name__)

async def async_process_document(file_path: str, source_file_name: str, file_hash: str):
    """The core asynchronous task that does the heavy lifting."""
    settings = get_settings()
    
    # IMPORTANT: Initialize fresh async clients INSIDE the worker process
    # IMPORTANT: Initialize fresh async clients INSIDE the worker process
    postgres_uri = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    
    # UPDATED: Set open=False and explicitly await pool.open()
    db_pool = AsyncConnectionPool(postgres_uri, open=False, kwargs={"autocommit": True, "row_factory": dict_row})

    await db_pool.open(wait=True)
    
    qdrant_client = AsyncQdrantClient(url=settings.QDRANT_URL)
    try:
        # 1. Update status to PROCESSING
        async with db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE documents SET status = %s WHERE file_hash = %s",
                    ("PROCESSING", file_hash)
                )
        
        # 2. Parse Document (Synchronous, but uses ThreadPool internally for speed)
        # 2. Parse Document
        parser = DocumentParser()
        chunks = parser.parse_document(file_path, source_file_name, file_hash)
        
        # --- NEW: CHUNK TRANSPARENCY LOGGING ---
        logger.info(f"\n{'='*60}\n[INDEXING] PREVIEWING CHUNKS FOR {source_file_name}\n{'='*60}")
        for i, chunk in enumerate(chunks[:5]): # Show the first 5 chunks to ensure clean extraction
            logger.info(f"Chunk {i+1} (Page {chunk.metadata.page_number} | {chunk.metadata.chunk_type}): {chunk.text[:120]}...")
        logger.info(f"... {len(chunks)} total chunks created.\n{'='*60}")
        # ---------------------------------------

        # 3. Upsert to Qdrant (Asynchronous)
        qdrant_service = QdrantService(qdrant_client)
        await qdrant_service.init_collection()
        await qdrant_service.upsert_chunks(chunks)
        
        # 4. Update status to COMPLETED
        async with db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE documents SET status = %s WHERE file_hash = %s",
                    ("COMPLETED", file_hash)
                )
        logger.info(f"Successfully processed and indexed document: {source_file_name}")
        
    except Exception as e:
        logger.error(f"Failed to process document {source_file_name}: {e}")
        # Mark as FAILED in database so the frontend doesn't poll forever
        async with db_pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE documents SET status = %s WHERE file_hash = %s",
                    ("FAILED", file_hash)
                )
    finally:
        # Always clean up connections to prevent memory leaks in the worker
        await db_pool.close()
        await qdrant_client.close()

def process_document(file_path: str, source_file_name: str, file_hash: str):
    """
    Synchronous wrapper for RQ. 
    RQ workers are strictly synchronous, so we use asyncio.run() to bootstrap 
    our async environment in an isolated event loop for this specific job.
    """
    logger.info(f"Worker picked up job for {source_file_name}...")
    asyncio.run(async_process_document(file_path, source_file_name, file_hash))