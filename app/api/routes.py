#routes.py
import asyncio
import hashlib
import json
import logging
import os
from typing import AsyncGenerator

from fastapi import APIRouter, Request, UploadFile, File, Depends, HTTPException
from fastapi.responses import StreamingResponse
from psycopg_pool import AsyncConnectionPool
from redis import Redis as SyncRedis
from rq import Queue
from langchain_core.messages import HumanMessage

from app.core.dependencies import get_db_pool, db_manager
from app.core.config import get_settings
from app.models.schemas import UploadResponse, ChatRequest, ChatResponse
from app.graph.workflow import build_graph
from app.services.storage import StorageService
from app.services.guardrails import validate_file, sanitize_query, detect_prompt_injection

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# UPLOAD
# ---------------------------------------------------------------------------

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    db_pool: AsyncConnectionPool = Depends(get_db_pool),
):
    settings = get_settings()
    content = await file.read()

    is_valid, error_msg = validate_file(
        content, file.filename or "", max_bytes=settings.MAX_UPLOAD_SIZE_BYTES
    )
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)

    # Always ensure the table exists before touching it.
    # CREATE TABLE IF NOT EXISTS is idempotent and takes microseconds —
    # running it on every upload is safe and guarantees no "relation does
    # not exist" crashes regardless of startup ordering.
    async with db_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    file_hash TEXT PRIMARY KEY,
                    filename  TEXT,
                    status    TEXT
                )
            """)

    file_hash = hashlib.sha256(content).hexdigest()

    async with db_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT status FROM documents WHERE file_hash = %s", (file_hash,)
            )
            result = await cur.fetchone()

            if result:
                existing_status = result["status"]
                # Allow FAILED documents to be re-queued.
                if existing_status in ("COMPLETED", "PROCESSING", "PENDING"):
                    return UploadResponse(
                        status="Document already indexed",
                        task_id="None",
                        file_hash=file_hash,
                    )
                # FAILED -> reset and re-queue
                await cur.execute(
                    "UPDATE documents SET status = %s WHERE file_hash = %s",
                    ("PENDING", file_hash),
                )
            else:
                await cur.execute(
                    "INSERT INTO documents (file_hash, filename, status) VALUES (%s, %s, %s)",
                    (file_hash, file.filename, "PENDING"),
                )

    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file_hash}_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(content)

    storage = StorageService()
    storage.upload_file(file_path, f"documents/{file_hash}_{file.filename}")

    redis_conn = SyncRedis.from_url(settings.REDIS_BROKER_URL)
    q = Queue(connection=redis_conn)
    job = q.enqueue(
        "worker.process_document",
        file_path, file.filename, file_hash,
        job_timeout=1200,
    )

    return UploadResponse(
        status="Processing started",
        task_id=job.id,
        file_hash=file_hash,
    )


# ---------------------------------------------------------------------------
# SSE STATUS STREAM
# ---------------------------------------------------------------------------

@router.get("/document/{file_hash}/stream")
async def stream_document_status(
    file_hash: str,
    db_pool: AsyncConnectionPool = Depends(get_db_pool),
):
    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            async with db_pool.connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(
                        "SELECT status FROM documents WHERE file_hash = %s", (file_hash,)
                    )
                    row = await cur.fetchone()
                    current_status = row["status"] if row else "NOT_FOUND"
        except Exception:
            current_status = "UNKNOWN"

        yield f"data: {json.dumps({'status': current_status, 'file_hash': file_hash})}\n\n"

        if current_status in ("COMPLETED", "FAILED", "NOT_FOUND"):
            return

        pubsub = db_manager.redis_broker.pubsub()
        channel = f"doc_status:{file_hash}"
        await pubsub.subscribe(channel)

        try:
            while True:
                msg = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=0.1
                )
                if msg and msg["type"] == "message":
                    data = json.loads(msg["data"])
                    yield f"data: {json.dumps(data)}\n\n"
                    if data.get("status") in ("COMPLETED", "FAILED"):
                        break
                else:
                    yield ": keepalive\n\n"
                    await asyncio.sleep(15)
        except asyncio.CancelledError:
            pass
        finally:
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/document/{file_hash}/status")
async def get_document_status(
    file_hash: str,
    db_pool: AsyncConnectionPool = Depends(get_db_pool),
):
    async with db_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT status FROM documents WHERE file_hash = %s", (file_hash,)
            )
            result = await cur.fetchone()
            if result:
                return {"file_hash": file_hash, "status": result["status"]}
            return {"file_hash": file_hash, "status": "NOT_FOUND"}


# ---------------------------------------------------------------------------
# CHAT
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    clean_query = sanitize_query(request.query)
    is_injection, pattern = detect_prompt_injection(clean_query)
    if is_injection:
        raise HTTPException(
            status_code=400,
            detail="Your query contains disallowed patterns. Please rephrase.",
        )

    graph = build_graph(db_manager.checkpointer)
    config = {"configurable": {"thread_id": request.thread_id}}

    active_documents = []
    active_file_hashes = []

    if request.active_documents:
        active_documents = [d.model_dump() for d in request.active_documents]
        active_file_hashes = [d["file_hash"] for d in active_documents]
    elif request.active_file_hashes:
        active_file_hashes = request.active_file_hashes
        active_documents = [{"file_hash": h, "filename": h[:8]} for h in active_file_hashes]
    elif request.file_hash:
        active_file_hashes = [request.file_hash]
        active_documents = [{"file_hash": request.file_hash, "filename": request.file_hash[:8]}]

    initial_state = {
        "messages": [HumanMessage(content=clean_query)],
        "query": clean_query,
        "active_file_hashes": active_file_hashes,
        "active_documents": active_documents,
        "query_type": "general",
        "resolved_targets": [],
    }

    result = await graph.ainvoke(initial_state, config)

    return ChatResponse(
        answer=result["final_answer"],
        citations=result.get("citations", []),
    )
