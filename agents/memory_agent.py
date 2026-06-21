"""
Memory Agent — thread-aware query cache using shared ChromaDB client.
"""

from models import GraphState, AgentError, RetrieverError, AGENT_HANDOFFS
from decorators import agent_error_handler
from utils import validate_state
from chroma_client import get_vectorstore

ALLOWED_HANDOFFS = AGENT_HANDOFFS["Memory"]
CACHE_DIR = "query_cache_vectorstore"


@agent_error_handler
def memory_agent(state: GraphState):
    """Check for cached answers in the current conversation thread."""
    print("---NODE: Memory Agent---")

    if not validate_state(state):
        raise AgentError("Invalid state received")

    query = state["query"]
    thread_id = state.get("thread_id", "default")
    print(f"Thread ID: {thread_id}")

    if not query or len(query.strip()) == 0:
        raise ValueError("Empty query received")

    # Skip cache for time-sensitive queries since a cached answer may be stale
    TIME_SENSITIVE_KEYWORDS = [
        "current", "today", "now", "latest", "real-time", "real time",
        "live", "this week", "this month", "this year", "currently", "recent"
    ]
    if any(kw in query.lower() for kw in TIME_SENSITIVE_KEYWORDS):
        print("Time-sensitive query — bypassing cache")
        state["trace"].append({
            "agent": "Memory", "tool": "Freshness check",
            "output": "Time-sensitive query", "handoff-to": "Retriever",
            "reasoning": "Query asks for current info — cached answer may be stale"
        })
        return {"next_agent": "Retriever", "cache_hit": False}

    if state.get("bypass_cache", False):
        print("Cache bypass flag set")
        state["trace"].append({
            "agent": "Memory", "tool": "Cache bypass",
            "output": "Cache bypassed", "handoff-to": "Retriever",
            "reasoning": "Bypass requested"
        })
        return {"next_agent": "Retriever", "cache_hit": False}

    try:
        cache_store = get_vectorstore(CACHE_DIR, collection_name="query_cache")
        similar_queries = cache_store.similarity_search_with_score(
            query, k=5, filter={"thread_id": thread_id}
        )

        if not similar_queries:
            print("No cached queries found")
            state["trace"].append({
                "agent": "Memory", "tool": "Query Cache",
                "output": "No cache hit", "handoff-to": "Retriever",
                "reasoning": f"New query in thread {thread_id}"
            })
            return {"next_agent": "Retriever", "cache_hit": False}

        best_match, distance = similar_queries[0]

        if not best_match or not hasattr(best_match, 'page_content'):
            raise RetrieverError("Invalid cache data")

        similarity = 1.0 / (1.0 + distance)
        print(f"Best match similarity: {similarity:.3f}")

        if similarity >= 0.90:
            cached_answer = best_match.metadata.get("answer", "")
            if not cached_answer:
                return {"next_agent": "Retriever", "cache_hit": False}

            print(f"Cache hit in thread {thread_id}")
            state["trace"].append({
                "agent": "Memory", "tool": "Query Cache",
                "output": f"Cache hit ({similarity:.3f})", "handoff-to": "Aggregator",
                "reasoning": "Identical query found in this conversation"
            })
            return {
                "cache_hit": True,
                "cached_answer": cached_answer,
                "next_agent": "Aggregator"
            }
        else:
            print(f"Low similarity ({similarity:.3f}), treating as new query")
            state["trace"].append({
                "agent": "Memory", "tool": "Query Cache",
                "output": f"Low similarity ({similarity:.3f})", "handoff-to": "Retriever",
                "reasoning": "Different query"
            })
            return {"next_agent": "Retriever", "cache_hit": False}

    except Exception as e:
        print(f"Cache error: {e}")
        state["trace"].append({
            "agent": "Memory", "tool": "Query Cache",
            "error": str(e), "handoff-to": "Retriever",
            "reasoning": "Cache error, falling back to Retriever"
        })
        return {"next_agent": "Retriever", "cache_hit": False}