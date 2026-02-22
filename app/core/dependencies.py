from typing import AsyncGenerator
from fastapi import Depends
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
    postgres_uri = f"postgresql://{settings.POSTGRES_USER}:{settings.POSTGRES_PASSWORD}@{settings.POSTGRES_HOST}:{settings.POSTGRES_PORT}/{settings.POSTGRES_DB}"
    
    # UPDATED: Added dict_row to kwargs. LangGraph PostgresSaver requires this to access columns by name!
    db_manager.pool = AsyncConnectionPool(
        postgres_uri, 
        open=False, 
        kwargs={"autocommit": True, "row_factory": dict_row}
    )

    await db_manager.pool.open(wait=True)
    
    # SETUP CHECKPOINTER ONCE DURING LIFESPAN
    db_manager.checkpointer = AsyncPostgresSaver(db_manager.pool)
    await db_manager.checkpointer.setup()
    
    db_manager.qdrant = AsyncQdrantClient(url=settings.QDRANT_URL)
    db_manager.redis_broker = Redis.from_url(settings.REDIS_BROKER_URL, decode_responses=True)
    db_manager.redis_cache = Redis.from_url(settings.REDIS_CACHE_URL, decode_responses=True)

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
        raise RuntimeError()
    yield db_manager.pool

async def get_qdrant() -> AsyncGenerator[AsyncQdrantClient, None]:
    if not db_manager.qdrant:
        raise RuntimeError()
    yield db_manager.qdrant

async def get_redis_cache() -> AsyncGenerator[Redis, None]:
    if not db_manager.redis_cache:
        raise RuntimeError()
    yield db_manager.redis_cache

async def get_redis_broker() -> AsyncGenerator[Redis, None]:
    if not db_manager.redis_broker:
        raise RuntimeError()
    yield db_manager.redis_broker