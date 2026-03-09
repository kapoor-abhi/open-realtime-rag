#workflow.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from app.graph.state import GraphState
# NEW: Import the rewrite_query_node
from app.graph.nodes import rewrite_query_node, check_cache_node, intent_node, retrieve_node, generate_node

def should_route(state: GraphState) -> str:
    if state.get("is_cached"):
        return "end"
    return "intent"

def build_graph(checkpointer: AsyncPostgresSaver):
    workflow = StateGraph(GraphState)
    
    # Add all nodes to the graph
    workflow.add_node("rewrite_query", rewrite_query_node)  # NEW: Contextual memory node
    workflow.add_node("check_cache", check_cache_node)
    workflow.add_node("intent", intent_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("generate", generate_node)
    
    # NEW ENTRY POINT: Always contextualize the query first
    workflow.set_entry_point("rewrite_query")
    
    # NEW EDGE: Move from rewriting straight to cache checking
    workflow.add_edge("rewrite_query", "check_cache")
    
    workflow.add_conditional_edges(
        "check_cache",
        should_route,
        {
            "intent": "intent",
            "end": END
        }
    )
    workflow.add_edge("intent", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile(checkpointer=checkpointer)