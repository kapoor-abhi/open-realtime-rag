#nodes.py
import json
import hashlib
from typing import Optional
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

@observe(name="check_semantic_cache")
async def check_cache_node(state: GraphState, config: RunnableConfig) -> GraphState:
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    query = state["query"]
    cache_key = f"{thread_id}:{query}"
    query_hash = hashlib.sha256(cache_key.encode()).hexdigest()
    
    cached_result = await db_manager.redis_cache.get(f"cache:{query_hash}")
    if cached_result:
        data = json.loads(cached_result)
        return {"is_cached": True, "final_answer": data["answer"], "citations": data["citations"]}
        
    return {"is_cached": False}

@observe(name="intent_routing")
async def intent_node(state: GraphState) -> GraphState:
    settings = get_settings()
    # UPDATED TO LLAMA-3.3-70B
    llm = ChatGroq(api_key=settings.GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    structured_llm = llm.with_structured_output(QueryIntent)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a query analyzer. Determine if the user is asking about a specific page number."),
        ("user", "{query}")
    ])
    
    chain = prompt | structured_llm
    result = await chain.ainvoke({"query": state["query"]})
    
    return {"page_number": result.page_number if result.is_page_specific else None}

@observe(name="qdrant_retrieval")
async def retrieve_node(state: GraphState) -> GraphState:
    qdrant_service = QdrantService(db_manager.qdrant)
    chunks = await qdrant_service.search(
        query=state["query"],
        limit=5,
        page_number=state.get("page_number"),
        source_file=state.get("source_file")
    )
    return {"retrieved_chunks": chunks}

@observe(name="groq_generation")
async def generate_node(state: GraphState, config: RunnableConfig) -> GraphState:
    settings = get_settings()
    # UPDATED TO LLAMA-3.3-70B
    llm = ChatGroq(api_key=settings.GROQ_API_KEY, model_name="llama-3.3-70b-versatile")
    
    langfuse_handler = CallbackHandler()
    
    context_str = "\n---\n".join(
        [f"Source: Page {chunk['page_number']}\nContent: {chunk['text']}" for chunk in state.get("retrieved_chunks", [])]
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert analyst. Answer using ONLY the provided context blocks. Every factual claim MUST be accompanied by an inline citation referencing the source page exactly like this: [Page X]."),
        MessagesPlaceholder(variable_name="messages"),
        ("user", "Here is the retrieved context to help you answer the query:\n\n{context}")
    ])
    
    chain = prompt | llm
    response = await chain.ainvoke(
        {"context": context_str, "messages": state["messages"]},
        config={"callbacks": [langfuse_handler]}
    )
    
    citations = [
        SourceCitation(
            page_number=chunk["page_number"], 
            source_file=chunk["source_file"],
            image_path=chunk.get("image_path")
        )
        for chunk in state.get("retrieved_chunks", [])
    ]
    
    thread_id = config.get("configurable", {}).get("thread_id", "default")
    cache_key = f"{thread_id}:{state['query']}"
    query_hash = hashlib.sha256(cache_key.encode()).hexdigest()
    
    cache_data = json.dumps({"answer": response.content, "citations": [c.model_dump() for c in citations]})
    await db_manager.redis_cache.setex(f"cache:{query_hash}", 300, cache_data)
    
    return {
        "messages": [response],
        "final_answer": response.content, 
        "citations": citations
    }