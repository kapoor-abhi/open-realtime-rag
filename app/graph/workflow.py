#workflow.py
"""
Graph topology (FIXED ORDER):

  rewrite_query
      │
  intent          ← extracts all targets + resolves file_hashes
      │               MUST run before cache so scope is correct
  check_cache ────→ END  (cache hit)
      │
  retrieve        ← parallel per-target Qdrant searches
      │
  generate
      │
     END

Why intent runs before check_cache:
  The semantic cache is scoped by resolved_targets (file_hash + page_number).
  If cache ran first, resolved_targets would be empty for every query and
  "page 5 of finance.pdf" would collide with "page 5 of contract.pdf".
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from app.graph.state import GraphState
from app.graph.nodes import (
    rewrite_query_node,
    intent_node,
    check_cache_node,
    retrieve_node,
    generate_node,
)


def _should_use_cache(state: GraphState) -> str:
    return "end" if state.get("is_cached") else "retrieve"


def build_graph(checkpointer: AsyncPostgresSaver):
    workflow = StateGraph(GraphState)

    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("intent", intent_node)
    workflow.add_node("check_cache", check_cache_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("rewrite_query")
    workflow.add_edge("rewrite_query", "intent")
    workflow.add_edge("intent", "check_cache")

    workflow.add_conditional_edges(
        "check_cache",
        _should_use_cache,
        {"retrieve": "retrieve", "end": END},
    )

    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile(checkpointer=checkpointer)
