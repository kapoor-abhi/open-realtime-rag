#dependencies.py
"""
FIX: init_services() now creates the documents table before the API
starts accepting requests. This replaces both the original approach
(CREATE TABLE inside every upload handler call) and the flawed
asyncio.Event approach (race condition between setting the flag and
the actual await completing).

Running DDL here is correct because:
  - init_services() is called from the FastAPI lifespan, which runs
    before any route handler can execute.
  - It runs exactly once per process lifetime.
  - No route handler needs to think about table existence at all.
"""

from typing import AsyncGenerator
from psycopg_pool import AsyncConnectionPool
from psycopg.rows import dict_row
from qdrant_client import AsyncQdrantClient
from redis.asyncio import Redis
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from app.core.config import Settings, get_settings


class DatabaseManager:
    def __init__(self):
        self.pool: AsyncConnectionPool | None = None
        self.qdrant: AsyncQdrantClient | None = None
        self.redis_broker: Redis | None = None
        self.redis_cache: Redis | None = None
        self.checkpointer: AsyncPostgresSaver | None = None


db_manager = DatabaseManager()


async def init_services(settings: Settings):
    postgres_uri = (
        f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}"
        f"@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    )

    db_manager.pool = AsyncConnectionPool(
        postgres_uri,
        open=False,
        kwargs={"autocommit": True, "row_factory": dict_row},
    )
    await db_manager.pool.open(wait=True)

    # FIX: create application tables once at startup, before any request arrives.
    # Previously the documents table was created inside the upload route handler,
    # meaning it ran on every upload and crashed on the very first request if
    # the CREATE TABLE call had been removed (as happened after the previous refactor).
    async with db_manager.pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    file_hash TEXT PRIMARY KEY,
                    filename  TEXT,
                    status    TEXT
                )
            """)

    # LangGraph checkpointer — also creates its own internal tables via .setup()
    db_manager.checkpointer = AsyncPostgresSaver(db_manager.pool)
    await db_manager.checkpointer.setup()

    db_manager.qdrant = AsyncQdrantClient(url=settings.QDRANT_URL)
    db_manager.redis_broker = Redis.from_url(
        settings.REDIS_BROKER_URL, decode_responses=True
    )
    db_manager.redis_cache = Redis.from_url(
        settings.REDIS_CACHE_URL, decode_responses=True
    )


async def close_services():
    if db_manager.pool:
        await db_manager.pool.close()
    if db_manager.qdrant:
        await db_manager.qdrant.close()
    if db_manager.redis_broker:
        await db_manager.redis_broker.aclose()
    if db_manager.redis_cache:
        await db_manager.redis_cache.aclose()


async def get_db_pool() -> AsyncGenerator[AsyncConnectionPool, None]:
    if not db_manager.pool:
        raise RuntimeError("Database pool not initialised")
    yield db_manager.pool


async def get_qdrant() -> AsyncGenerator[AsyncQdrantClient, None]:
    if not db_manager.qdrant:
        raise RuntimeError("Qdrant client not initialised")
    yield db_manager.qdrant


async def get_redis_cache() -> AsyncGenerator[Redis, None]:
    if not db_manager.redis_cache:
        raise RuntimeError("Redis cache not initialised")
    yield db_manager.redis_cache


async def get_redis_broker() -> AsyncGenerator[Redis, None]:
    if not db_manager.redis_broker:
        raise RuntimeError("Redis broker not initialised")
    yield db_manager.redis_broker