#state.py
from typing import Annotated, TypedDict, List, Optional
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from app.models.schemas import SourceCitation, ActiveDocument, ResolvedTarget


class GraphState(TypedDict):
    # ── Core conversation ───────────────────────────────────────────────────
    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    rewritten_query: Optional[str]

    # ── Document workspace ──────────────────────────────────────────────────
    # All documents the user has open in this chat session.
    # Plain list of hashes (for Qdrant/BM25 filters when no specific doc targeted).
    active_file_hashes: List[str]
    # Rich version: hash + filename. Required for document-name resolution.
    active_documents: List[ActiveDocument]

    # ── Intent detection ────────────────────────────────────────────────────
    # What kind of query is this?
    #   "general"       — no specific doc/page, search everything
    #   "single_target" — one specific doc and/or page
    #   "comparison"    — multiple docs/pages to compare side-by-side
    query_type: str

    # Fully resolved targets after fuzzy-matching hints → file_hashes.
    # Empty list = general query (search all active docs).
    # One entry = single-target.
    # Two+ entries = comparison / multi-target.
    resolved_targets: List[ResolvedTarget]

    # ── Retrieval & generation ──────────────────────────────────────────────
    retrieved_chunks: List[dict]
    citations: List[SourceCitation]
    final_answer: str
    is_cached: bool
