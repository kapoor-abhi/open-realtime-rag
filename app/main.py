#main.py
import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

import logging
from app.core.config import get_settings
from app.core.dependencies import init_services, close_services
from app.api.routes import router

# ---------------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RATE LIMITER SETUP
# Keyed by client IP address. Limits are configurable via .env
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("uploads", exist_ok=True)
    settings = get_settings()
    await init_services(settings)
    yield
    await close_services()


app = FastAPI(
    title="OpenMultiRAG API",
    description="Multimodal RAG with hybrid BM25+dense retrieval, Cohere reranking, and semantic caching.",
    version="2.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# RATE LIMITING MIDDLEWARE
# Returns HTTP 429 with Retry-After header when limit is exceeded.
# ---------------------------------------------------------------------------
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# ---------------------------------------------------------------------------
# CORS
# In production: replace allow_origins=["*"] with your actual frontend domain.
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # TODO: tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


@app.get("/health")
async def health_check():
    """
    Lightweight health check for load balancers / Docker health checks.
    Returns 200 if the API process is alive.
    For deeper checks (Postgres, Redis, Qdrant), see /health/deep.
    """
    return {"status": "healthy", "version": "2.0.0"}


@app.get("/health/deep")
async def deep_health_check():
    """
    Deep health check: verifies connectivity to all downstream services.
    """
    from app.core.dependencies import db_manager
    results = {}

    # Postgres
    try:
        async with db_manager.pool.connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
        results["postgres"] = "ok"
    except Exception as e:
        results["postgres"] = f"error: {e}"

    # Redis
    try:
        await db_manager.redis_cache.ping()
        results["redis"] = "ok"
    except Exception as e:
        results["redis"] = f"error: {e}"

    # Qdrant
    try:
        info = await db_manager.qdrant.get_collections()
        results["qdrant"] = f"ok ({len(info.collections)} collections)"
    except Exception as e:
        results["qdrant"] = f"error: {e}"

    overall = "healthy" if all(v == "ok" or v.startswith("ok") for v in results.values()) else "degraded"
    return {"status": overall, "services": results}
