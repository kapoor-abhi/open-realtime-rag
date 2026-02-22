#schemas.py
from typing import List, Optional
from pydantic import BaseModel

class UploadResponse(BaseModel):
    status: str
    task_id: str
    file_hash: str

class ChatRequest(BaseModel):
    query: str
    thread_id: str

class SourceCitation(BaseModel):
    page_number: int
    source_file: str
    image_path: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    citations: List[SourceCitation]

class DocumentMetadata(BaseModel):
    source_file: str
    page_number: int
    chunk_type: str
    image_path: Optional[str] = None

class DocumentChunk(BaseModel):
    text: str
    metadata: DocumentMetadata