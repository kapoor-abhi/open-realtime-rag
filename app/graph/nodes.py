#nodes.py
import json
import hashlib
from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableConfig

from langfuse import observe
from langfuse.langchain import CallbackHandler

from app.core.config import get_settings
from app.core.dependencies import db_manager
from app.services.vector_store import QdrantService
from app.graph.state import GraphState
from app.models.schemas import SourceCitation

class QueryIntent(BaseModel):
    is_page_specific: bool = Field(description="True if the user is asking about a specific page number.")
    page_number: Optional[int] = Field(default=None, description="The specific page number requested.")

def _get_cache_prefix(active_hashes: Optional[List[str]]) -> str:
    """Helper to generate a deterministic cache prefix for multiple documents."""
    if not active_hashes:
        return "global"
    return "_".join(sorted(active_hashes))

# --- NEW NODE: QUERY REWRITER (MEMORY FIX) ---
@observe(name="rewrite_query")
async def rewrite_query_node(state: GraphState, config: RunnableConfig) -> GraphState:
    """Rewrites the user's query using chat history to restore lost context."""
    if len(state["messages"]) <= 1:
        return {"rewritten_query": state["query"]}
        
    settings = get_settings()
    # Using an ultra-fast model for the rewriting task to minimize latency
    llm = ChatGroq(api_key=settings.GROQ_API_KEY, model_name="llama-3.1-8b-instant")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a query contextualization assistant. Look at the conversation history and rewrite the user's latest query into a standalone query that contains all necessary context (like document subjects or entities). Do NOT answer the query. Just return the rewritten text."),
        MessagesPlaceholder(variable_name="messages"),
    ])
    
    chain = prompt | llm
    response = await chain.ainvoke({"messages": state["messages"]})
    
    return {"rewritten_query": response.content.strip()}

@observe(name="check_semantic_cache")
async def check_cache_node(state: GraphState, config: RunnableConfig) -> GraphState:
    active_query = state.get("rewritten_query", state["query"]).strip().lower()
    
    # NEW: Multi-doc cache prefixing
    prefix = _get_cache_prefix(state.get("active_file_hashes"))
    cache_key = f"{prefix}:{active_query}"
    query_hash = hashlib.sha256(cache_key.encode()).hexdigest()
    
    cached_result = await db_manager.redis_cache.get(f"cache:{query_hash}")
    if cached_result:
        data = json.loads(cached_result)
        return {"is_cached": True, "final_answer": data["answer"], "citations": data["citations"]}
        
    return {"is_cached": False}

@observe(name="intent_routing")
async def intent_node(state: GraphState) -> GraphState:
    settings = get_settings()
    llm = ChatGroq(api_key=settings.GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    
    structured_llm = llm.with_structured_output(QueryIntent, method="json_mode")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a query analyzer. Respond ONLY with valid JSON. The key 'is_page_specific' MUST be a raw boolean. The key 'page_number' MUST be an integer or null."),
        ("user", "{query}")
    ])
    
    active_query = state.get("rewritten_query", state["query"])
    
    try:
        chain = prompt | structured_llm
        result = await chain.ainvoke({"query": active_query})
        page_num = result.page_number if result.is_page_specific else None
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"[INTENT] Groq schema hallucination caught. Defaulting to None. Error: {e}")
        page_num = None
        
    return {"page_number": page_num}

@observe(name="qdrant_retrieval")
async def retrieve_node(state: GraphState) -> GraphState:
    qdrant_service = QdrantService(db_manager.qdrant)
    
    active_query = state.get("rewritten_query", state["query"])
    
    # NEW: Pass the active_file_hashes list to Qdrant for strict Workspace filtering
    chunks = await qdrant_service.search(
        query=active_query,
        limit=5, 
        page_number=state.get("page_number"),
        active_file_hashes=state.get("active_file_hashes", []) 
    )
    return {"retrieved_chunks": chunks}

@observe(name="groq_generation")
async def generate_node(state: GraphState, config: RunnableConfig) -> GraphState:
    settings = get_settings()
    llm = ChatGroq(api_key=settings.GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    langfuse_handler = CallbackHandler()
    
    # Group chunks by source file to prevent context mixing
    from collections import defaultdict
    chunks_by_file = defaultdict(list)
    for i, chunk in enumerate(state.get("retrieved_chunks", [])):
        chunks_by_file[chunk['source_file']].append((i+1, chunk))
        
    context_blocks = []
    for source_file, chunks in chunks_by_file.items():
        doc_block = f"=== Document: {source_file} ===\n"
        for i, chunk in chunks:
            doc_block += f"--- Block {i} (Page {chunk['page_number']}) ---\n{chunk['text']}\n"
        context_blocks.append(doc_block)
        
    context_str = "\n\n".join(context_blocks)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert analyst. Answer using ONLY the provided context blocks.
Synthesize the information across the provided documents to form a cohesive, comprehensive answer. Do NOT mix facts from different documents.
Do NOT use inline citations (e.g., [Page X, Document Y]) for every single point.
Instead, provide a single synthesized answer, and at the absolute end of your response, include a "Sources:" section that lists the Document Name and Page Number for the sources you actually relied on.
If the information is not in the context, say you do not know."""),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "Here is the retrieved context to help you answer the query:\n\n{context}")
    ])
    
    chain = prompt | llm
    response = await chain.ainvoke(
        {"context": context_str, "messages": state["messages"]},
        config={"callbacks": [langfuse_handler]}
    )
    
    response_text = response.content
    unique_citations = {}
    for chunk in state.get("retrieved_chunks", []):
        file_name = chunk['source_file']
        page_num = str(chunk['page_number'])
        
        # Only include the citation if the LLM actively referenced it in its response
        if file_name in response_text and page_num in response_text:
            key = f"{file_name}_{page_num}"
            if key not in unique_citations:
                unique_citations[key] = SourceCitation(
                    page_number=chunk["page_number"], 
                    source_file=chunk["source_file"],
                    image_path=chunk.get("image_path")
                )
                
    citations = list(unique_citations.values())
    
    # Save to the new Multi-Doc context-aware cache
    active_query = state.get("rewritten_query", state["query"]).strip().lower()
    prefix = _get_cache_prefix(state.get("active_file_hashes"))
    cache_key = f"{prefix}:{active_query}"
    query_hash = hashlib.sha256(cache_key.encode()).hexdigest()
    
    cache_data = json.dumps({"answer": response.content, "citations": [c.model_dump() for c in citations]})
    await db_manager.redis_cache.setex(f"cache:{query_hash}", 300, cache_data)
    
    return {
        "messages": [response],
        "final_answer": response.content, 
        "citations": citations
    }