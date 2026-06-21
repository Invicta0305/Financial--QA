"""
Summarizer Agent for content summarization.

This agent generates concise summaries of retrieved documents or web search
results and forwards the summary to the Aggregator for final response generation.
"""

from langchain_groq import ChatGroq
from models import GraphState, AGENT_HANDOFFS
from decorators import agent_error_handler
from config import GROQ_API_KEY


llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=GROQ_API_KEY)

ALLOWED_HANDOFFS = AGENT_HANDOFFS["Summarizer"]


@agent_error_handler  # Catches unhandled exceptions and falls back to Aggregator instead of crashing the graph
def summarizer_agent(state: GraphState):
    """
    Generate a concise summary of available content.
    
    This agent summarizes documents or web search results using an LLM
    and forwards the summary to the Aggregator for final response generation.
    
    Args:
        state: Current graph state containing context with documents/web results
        
    Returns:
        dict: Updated state with summary and routing to Aggregator
    """
    print("---NODE: Summarizer---")
    print(f"Allowed handoffs: {ALLOWED_HANDOFFS}")
    
    context_texts = []
    context = state.get("context", {})
    
    if context.get("retrieved_docs"):
        context_texts.extend(context["retrieved_docs"])
    if context.get("web_results"):
        context_texts.extend(context["web_results"])
    
    # For compound queries, Math runs first and stores results in context["calculations"] —
    # include them so the summary covers the computed numbers, not just the raw documents.
    if context.get("calculations"):
        import json as _json
        context_texts.append("CALCULATION RESULTS:\n" + _json.dumps(context["calculations"], indent=2))
    
    if not context_texts:
        print("No content available for summarization, routing to Aggregator")
        
        state["trace"].append({
            "agent": "Summarizer",
            "tool": "N/A",
            "output": "No text",
            "handoff-to": "Aggregator",
            "reasoning": "No content to summarize"
        })
        
        return {"next_agent": "Aggregator"}
    
    context_str = "\n".join(context_texts)
    
    # Limit context length to avoid exceeding the prompt's token limit
    if len(context_str) > 30000:
        context_str = context_str[:30000] + "... [truncated]"
    
    prompt = f"""Summarize key points concisely:

Context:
{context_str}
"""
    
    try:
        response = llm.invoke(prompt)
        
        print("Summary generated successfully, routing to Aggregator")
        
        state["trace"].append({
            "agent": "Summarizer",
            "tool": "LLM",
            "output": "Summary generated",
            "handoff-to": "Aggregator",
            "reasoning": "Summarization complete"
        })
        
        return {
            "context": {**context, "summary": response.content},
            "next_agent": "Aggregator"
        }
    
    except Exception as e:
        print(f"Summarization failed: {e}")
        
        state["trace"].append({
            "agent": "Summarizer",
            "tool": "LLM",
            "error": str(e),
            "handoff-to": "Aggregator",
            "reasoning": "Summarization failed"
        })
        
        return {
            "context": {**context, "summarizer_failed": True},
            "next_agent": "Aggregator"
        }