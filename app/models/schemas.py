#schemas.py
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# API REQUEST / RESPONSE
# ---------------------------------------------------------------------------

class UploadResponse(BaseModel):
    status: str
    task_id: str
    file_hash: str


class ActiveDocument(BaseModel):
    """
    One document open in the user's chat workspace.
    The frontend tracks these and sends them with every chat message.

    Plain English: when you upload finance.pdf and contract.pdf, the frontend
    holds [{file_hash: "abc123", filename: "finance.pdf"}, {file_hash: "def456",
    filename: "contract.pdf"}]. This list is 'active_documents'. It lets the
    backend resolve names like "the finance report" to the right file_hash.
    """
    file_hash: str
    filename: str


class ChatRequest(BaseModel):
    query: str
    thread_id: str
    # Legacy single-document support (kept for backward compat)
    file_hash: Optional[str] = None
    # Flat hash list (also kept for backward compat)
    active_file_hashes: Optional[List[str]] = Field(default_factory=list)
    # Preferred: rich list with filename + hash (enables document-name resolution)
    active_documents: Optional[List[ActiveDocument]] = Field(default_factory=list)


class SourceCitation(BaseModel):
    page_number: int
    source_file: str
    image_path: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    citations: List[SourceCitation]


# ---------------------------------------------------------------------------
# DOCUMENT TARGETING (used inside the graph, not in the API surface)
# ---------------------------------------------------------------------------

class DocumentTarget(BaseModel):
    """
    One (document, page) pair extracted from the user's natural language query.

    Examples:
      "compare page 5 of finance.pdf with page 9 of contract.pdf"
        → targets = [
            DocumentTarget(document_hint="finance.pdf",  page_number=5),
            DocumentTarget(document_hint="contract.pdf", page_number=9),
          ]

      "what is on page 3?"   (no specific doc named)
        → targets = [DocumentTarget(document_hint=None, page_number=3)]

      "summarise all documents"
        → targets = []  (general query, search everything)
    """
    document_hint: Optional[str] = Field(
        default=None,
        description="The document name/phrase the user mentioned, or None if not doc-specific.",
    )
    page_number: Optional[int] = Field(
        default=None,
        description="The page number the user mentioned for this document, or None.",
    )


class ResolvedTarget(BaseModel):
    """
    A DocumentTarget after fuzzy-matching the hint to an exact file_hash.
    This is what the retrieval node consumes.
    """
    file_hash: str
    filename: str
    page_number: Optional[int] = None


# ---------------------------------------------------------------------------
# DOCUMENT PARSING (internal)
# ---------------------------------------------------------------------------

class DocumentMetadata(BaseModel):
    source_file: str
    file_hash: str
    page_number: int
    chunk_type: str
    image_path: Optional[str] = None


class DocumentChunk(BaseModel):
    text: str
    metadata: DocumentMetadata
