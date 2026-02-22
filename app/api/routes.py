import hashlib
import os
from fastapi import APIRouter, UploadFile, File, Depends
from psycopg_pool import AsyncConnectionPool
from redis import Redis as SyncRedis
from rq import Queue
from langchain_core.messages import HumanMessage

from app.core.dependencies import get_db_pool, db_manager
from app.core.config import get_settings
from app.models.schemas import UploadResponse, ChatRequest, ChatResponse
from app.graph.workflow import build_graph
from app.services.storage import StorageService

router = APIRouter()

@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    db_pool: AsyncConnectionPool = Depends(get_db_pool)
):
    content = await file.read()
    file_hash = hashlib.sha256(content).hexdigest()
    
    async with db_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "CREATE TABLE IF NOT EXISTS documents (file_hash TEXT PRIMARY KEY, filename TEXT, status TEXT)"
            )
            
            await cur.execute(
                "SELECT status FROM documents WHERE file_hash = %s", 
                (file_hash,)
            )
            result = await cur.fetchone()
            
            if result:
                return UploadResponse(
                    status="Document already indexed",
                    task_id="None",
                    file_hash=file_hash
                )
            
            await cur.execute(
                "INSERT INTO documents (file_hash, filename, status) VALUES (%s, %s, %s)",
                (file_hash, file.filename, "PENDING")
            )
    
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file_hash}_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(content)
        
    storage = StorageService()
    storage.upload_file(file_path, f"documents/{file_hash}_{file.filename}")
        
    settings = get_settings()
    # RQ requires a synchronous Redis connection
    redis_conn = SyncRedis.from_url(settings.REDIS_BROKER_URL)
    q = Queue(connection=redis_conn)
    
    # We pass file_hash as the third argument to match your updated parser.py
    job = q.enqueue("worker.process_document", file_path, file.filename, file_hash, job_timeout=1200)
    
    return UploadResponse(
        status="Processing started",
        task_id=job.get_id(),
        file_hash=file_hash
    )

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    # UPDATED: We use the globally initialized checkpointer, cutting out schema setup overhead
    graph = build_graph(db_manager.checkpointer)
    
    config = {"configurable": {"thread_id": request.thread_id}}
    
    # UPDATED: Mapping the file hash to source_file ensures context-aware routing
    initial_state = {
        "messages": [HumanMessage(content=request.query)],
        "query": request.query,
        "page_number": None,
        "file_hash": getattr(request, "file_hash", None)  # NEW: Map to file_hash, not source_file
    }
    
    result = await graph.ainvoke(initial_state, config)
    
    return ChatResponse(
        answer=result["final_answer"],
        citations=result.get("citations", [])
    )

@router.get("/document/{file_hash}/status")
async def get_document_status(
    file_hash: str, 
    db_pool: AsyncConnectionPool = Depends(get_db_pool)
):
    async with db_pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT status FROM documents WHERE file_hash = %s", 
                (file_hash,)
            )
            result = await cur.fetchone()
            if result:
                # UPDATED: result is now a dictionary, so we access it by key name
                return {"file_hash": file_hash, "status": result["status"]}
            
            return {"file_hash": file_hash, "status": "NOT_FOUND"}