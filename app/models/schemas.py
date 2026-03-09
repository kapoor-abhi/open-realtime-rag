#schemas.py
from typing import List, Optional
from pydantic import BaseModel, Field

class UploadResponse(BaseModel):
    status: str
    task_id: str
    file_hash: str

class ChatRequest(BaseModel):
    query: str
    thread_id: str
    # Legacy support for a single document
    file_hash: Optional[str] = None
    # NEW: Multi-document workspace support. 
    # The frontend can pass an array of hashes for the documents active in the current chat.
    active_file_hashes: Optional[List[str]] = Field(default_factory=list)

class SourceCitation(BaseModel):
    page_number: int
    source_file: str
    image_path: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    citations: List[SourceCitation]

class DocumentMetadata(BaseModel):
    source_file: str
    file_hash: str  
    page_number: int
    chunk_type: str
    image_path: Optional[str] = None

class DocumentChunk(BaseModel):
    text: str
    metadata: DocumentMetadata