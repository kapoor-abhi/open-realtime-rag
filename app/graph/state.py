#state.py

from typing import Annotated, TypedDict, List, Optional
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from app.models.schemas import SourceCitation

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    query: str
    page_number: Optional[int]
    source_file: Optional[str]
    retrieved_chunks: List[dict]
    citations: List[SourceCitation]
    final_answer: str
    is_cached: bool