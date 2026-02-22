from typing import List, Optional
from pydantic import BaseModel

class UploadResponse(BaseModel):
    status: str
    task_id: str
    file_hash: str

class ChatRequest(BaseModel):
    query: str
    thread_id: str
    # NEW: Add file_hash to support our context-aware retrieval and caching
    file_hash: Optional[str] = None

class SourceCitation(BaseModel):
    page_number: int
    source_file: str
    image_path: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    citations: List[SourceCitation]

class DocumentMetadata(BaseModel):
    source_file: str
    file_hash: str  # NEW: We will filter Qdrant using this!
    page_number: int
    chunk_type: str
    image_path: Optional[str] = None

class DocumentChunk(BaseModel):
    text: str
    metadata: DocumentMetadata