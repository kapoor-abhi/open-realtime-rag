import asyncio
import logging
from redis import Redis
from rq import Worker, Queue
from app.core.config import get_settings
from app.core.dependencies import db_manager, init_services, close_services
from app.services.parser import DocumentParser
from app.services.vector_store import QdrantService

# Transparent logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

def process_document(file_path: str, filename: str, file_hash: str):
    asyncio.run(_process_document_async(file_path, filename, file_hash))

async def _process_document_async(file_path: str, filename: str, file_hash: str):
    settings = get_settings()
    await init_services(settings)
    
    try:
        logger.info(f"Worker starting to process {filename}...")
        parser = DocumentParser()
        chunks = parser.parse_document(file_path, filename)
        
        # --- THE FIX: FILTER OUT EMPTY CHUNKS ---
        valid_chunks = []
        for c in chunks:
            # Only keep chunks that have actual string content
            if c.text and c.text.strip():
                valid_chunks.append(c)
        
        if not valid_chunks:
            logger.error(f"No valid text chunks could be extracted from {filename}!")
            raise ValueError("Empty document or extraction failed.")
            
        logger.info(f"Upserting {len(valid_chunks)} valid chunks to Qdrant (filtered out {len(chunks) - len(valid_chunks)} empty chunks)...")
        # ----------------------------------------
        
        qdrant_service = QdrantService(db_manager.qdrant)
        await qdrant_service.init_collection()
        await qdrant_service.upsert_chunks(valid_chunks)
        
        async with db_manager.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute(
                    "UPDATE documents SET status = 'COMPLETED' WHERE file_hash = %s",
                    (file_hash,)
                )
        logger.info(f"Worker successfully finished {filename}.")
        
    except Exception as e:
        logger.error(f"Worker failed processing {filename}: {str(e)}")
        # --- THE FAILSAFE: Tell the database we crashed ---
        try:
            async with db_manager.pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "UPDATE documents SET status = 'FAILED' WHERE file_hash = %s",
                        (file_hash,)
                    )
        except Exception as db_err:
            logger.error(f"Could not update database with FAILED status: {db_err}")
        # --------------------------------------------------
        raise e
        
    finally:
        await close_services()

if __name__ == '__main__':
    settings = get_settings()
    redis_conn = Redis.from_url(settings.REDIS_BROKER_URL)
    queue = Queue('default', connection=redis_conn)
    worker = Worker([queue], connection=redis_conn)
    worker.work()