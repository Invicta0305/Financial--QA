"""
State definitions and custom exceptions for the multi-agent system.

This module defines the graph state structure, custom exceptions,
and agent handoff rules for the swarm architecture.
"""

from typing import TypedDict, List, Dict, Any


class AgentError(Exception):
    """Base exception for agent errors."""
    pass


class RetrieverError(AgentError):
    """Error in retrieval agent."""
    pass


class LLMError(AgentError):
    """Error in LLM-based agents."""
    pass


class ToolError(AgentError):
    """Error in tool execution."""
    pass


class GraphState(TypedDict):
    """
    State that flows through the agent graph.
    
    Attributes:
        query: User's input query
        context: Additional context information
        trace: List of agent execution traces
        answer: Final generated answer
        next_agent: Name of the next agent to execute
        messages: Message history
        cache_hit: Whether answer was retrieved from cache
        cached_answer: Cached answer if available
        conversation_history: History of previous exchanges
        thread_id: Unique identifier for conversation thread
        bypass_cache: Flag to force fresh processing
    """
    query: str
    context: Dict[str, Any]
    trace: List[Dict]
    answer: str
    next_agent: str
    messages: List
    cache_hit: bool
    cached_answer: str
    conversation_history: List[str]
    thread_id: str
    bypass_cache: bool
    hop_count: int  # Safety counter — prevents runaway routing loops


# Agent handoff rules defining the swarm architecture
# Each agent can only hand off to specific downstream agents
AGENT_HANDOFFS = {
    "Memory": ["Retriever", "Aggregator"],
    "Retriever": ["Validator"],
    "Validator": ["WebSearch", "Table", "Summarizer", "Aggregator"],
    "WebSearch": ["Validator"],
    "Table": ["Math", "Aggregator"],
    "Math": ["Aggregator", "Summarizer"],   # FIX: Math can now hand off to Summarizer for compound queries
    "Summarizer": ["Aggregator"],
    "Aggregator": ["END"]
}


def validate_handoff(source_agent: str, target_agent: str) -> bool:
    """
    Validate if a handoff between two agents is allowed.
    
    Args:
        source_agent: Name of the agent initiating handoff
        target_agent: Name of the agent receiving handoff
        
    Returns:
        bool: True if handoff is allowed, False otherwise
    """
    allowed_targets = AGENT_HANDOFFS.get(source_agent, [])
    return target_agent in allowed_targets