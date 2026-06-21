"""
LangGraph workflow for multi-agent financial document analysis.

This module defines the agent graph structure and routing logic
for the swarm-based architecture where agents hand off to each other directly.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from models import GraphState

# Import agent functions
from agents.memory_agent import memory_agent
from agents.retriever_agent import retrieve_agent
from agents.validator_agent import validator_agent
from agents.websearch_agent import websearch_agent
from agents.summarizer_agent import summarizer_agent
from agents.table_agent import table_agent
from agents.math_agent import math_agent
from agents.aggregator_agent import aggregator_agent


def route_next_agent(state: GraphState):
    """
    Determine the next agent to execute based on current state.
    
    Args:
        state: Current graph state containing next_agent information
        
    Returns:
        str: Name of next agent or END to terminate execution
    """
    next_agent = state.get("next_agent", "Memory")
    print(f"Routing to: {next_agent}")
    return END if next_agent == "END" else next_agent


# FIX: Prevent runaway routing loops. All agent nodes are wrapped with this
# decorator before registration. If hop_count exceeds MAX_HOPS the wrapper
# forces a hard route to Aggregator instead of letting the graph spin forever.
MAX_HOPS = 12

def with_hop_limit(agent_name: str, agent_fn):
    """Wrap an agent function with a hop-count safety cutoff."""
    def wrapped(state: GraphState):
        hop_count = state.get("hop_count", 0) + 1
        if hop_count > MAX_HOPS:
            print(f"[HOP LIMIT] {agent_name} exceeded {MAX_HOPS} hops — forcing Aggregator")
            state["trace"].append({
                "agent": agent_name,
                "tool": "Hop limit",
                "output": f"Exceeded {MAX_HOPS} hops",
                "handoff-to": "Aggregator",
                "reasoning": "Safety cutoff to prevent runaway loops"
            })
            return {
                "hop_count": hop_count,
                "next_agent": "Aggregator",
                "context": {**state.get("context", {}), "hop_limit_exceeded": True}
            }
        result = agent_fn(state)
        result["hop_count"] = hop_count
        return result
    wrapped.__name__ = agent_fn.__name__
    return wrapped


def build_graph():
    """
    Build and compile the agent workflow graph.
    
    This creates a StateGraph where agents can route to any other agent
    based on their internal logic, enabling dynamic workflow execution.
    
    Returns:
        Compiled LangGraph workflow with memory checkpointing
    """
    workflow = StateGraph(GraphState)
    
    # Register all agent nodes (wrapped with hop-count safety)
    workflow.add_node("Memory",     with_hop_limit("Memory",     memory_agent))
    workflow.add_node("Retriever",  with_hop_limit("Retriever",  retrieve_agent))
    workflow.add_node("Validator",  with_hop_limit("Validator",  validator_agent))
    workflow.add_node("WebSearch",  with_hop_limit("WebSearch",  websearch_agent))
    workflow.add_node("Summarizer", with_hop_limit("Summarizer", summarizer_agent))
    workflow.add_node("Table",      with_hop_limit("Table",      table_agent))
    workflow.add_node("Math",       with_hop_limit("Math",       math_agent))
    workflow.add_node("Aggregator", with_hop_limit("Aggregator", aggregator_agent))
    
    # Set Memory as the entry point for all queries
    workflow.set_entry_point("Memory")
    
    # Define all possible agent transitions
    all_agents = [
        "Memory", 
        "Retriever",
        "Validator",
        "WebSearch", 
        "Summarizer", 
        "Table", 
        "Math", 
        "Aggregator"
    ]
    
    # Add conditional routing edges for each agent
    # Allows any agent to route to any other agent dynamically
    agent_routing = {
        "Memory": "Memory",
        "Retriever": "Retriever",
        "Validator": "Validator",
        "WebSearch": "WebSearch",
        "Summarizer": "Summarizer",
        "Table": "Table",
        "Math": "Math",
        "Aggregator": "Aggregator",
        END: END
    }
    
    for agent in all_agents:
        workflow.add_conditional_edges(
            agent,
            route_next_agent,
            agent_routing
        )
    
    # Compile workflow with memory checkpointing for conversation continuity
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)