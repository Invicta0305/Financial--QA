"""
WebSearch Agent for external data retrieval.

This agent queries external sources using the Tavily API when document
retrieval is insufficient, then forwards results to Validator for assessment.
"""

import os
from tavily import TavilyClient
from models import GraphState, ToolError, AGENT_HANDOFFS
from decorators import agent_error_handler, retry_with_exponential_backoff


ALLOWED_HANDOFFS = AGENT_HANDOFFS["WebSearch"]


@agent_error_handler
@retry_with_exponential_backoff(max_retries=2, initial_delay=2.0)
def websearch_agent(state: GraphState):
    """
    Retrieve external data from web sources using Tavily API.
    
    This agent performs web searches when document retrieval is insufficient
    and forwards results to the Validator for relevance assessment and routing.
    
    Args:
        state: Current graph state containing query and context
        
    Returns:
        dict: Updated state with web search results and next agent routing
        
    Raises:
        ValueError: If query is empty
        ToolError: If Tavily API is not configured
    """
    print("---NODE: WebSearch---")
    print("Retrieving external data from web sources")
    
    query = state["query"]
    context = state.get("context", {})
    
    if not query:
        raise ValueError("Empty query")
    
    try:
        # Initialize Tavily client
        if not os.environ.get("TAVILY_API_KEY"):
            raise ToolError("TAVILY_API_KEY not configured")
        
        client = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))
        
        print(f"Searching web for: '{query}'")
        results = client.search(query, max_results=5)
        
        # Extract text content from results
        web_texts = []
        if isinstance(results, dict) and "results" in results:
            for res in results["results"]:
                if "content" in res and res["content"]:
                    web_texts.append(res["content"])
        
        if not web_texts:
            print("Web search returned no results")
            
            # Check for document fallback
            if context.get("retrieved_docs"):
                print("Document fallback available, routing to Validator")
                
                state["trace"].append({
                    "agent": "WebSearch",
                    "tool": "Tavily",
                    "output": "No web results, document fallback available",
                    "handoff-to": "Validator",
                    "reasoning": "WebSearch empty, sending documents to Validator for processing decision"
                })
                
                return {
                    "context": {**context, "websearch_failed": True},
                    "next_agent": "Validator"
                }
            else:
                # No data sources available
                print("No data sources available, routing to Aggregator")
                
                state["trace"].append({
                    "agent": "WebSearch",
                    "tool": "Tavily",
                    "output": "No web results and no documents",
                    "handoff-to": "Aggregator",
                    "reasoning": "No data sources available"
                })
                
                return {
                    "context": {**context, "websearch_failed": True},
                    "next_agent": "Aggregator"
                }
        
        print(f"Retrieved {len(web_texts)} web results")
        print("Forwarding to Validator for assessment")
        
        state["trace"].append({
            "agent": "WebSearch",
            "tool": "Tavily",
            "output": f"Retrieved {len(web_texts)} web results",
            "handoff-to": "Validator",
            "reasoning": "Web data retrieved, sending to Validator for assessment"
        })
        
        return {
            "context": {
                **context,
                "web_results": web_texts,
                "data_source": "web"
            },
            "next_agent": "Validator"
        }
    
    except Exception as e:
        print(f"Web search failed: {e}")
        
        # Check for document fallback
        if context.get("retrieved_docs"):
            print("Web search failed, using document fallback")
            
            state["trace"].append({
                "agent": "WebSearch",
                "tool": "Tavily",
                "error": str(e),
                "handoff-to": "Validator",
                "reasoning": "WebSearch failed, validating document fallback"
            })
            
            return {
                "context": {**context, "websearch_failed": True},
                "next_agent": "Validator"
            }
        else:
            # No fallback available
            print("No alternative data sources available")
            
            state["trace"].append({
                "agent": "WebSearch",
                "tool": "Tavily",
                "error": str(e),
                "handoff-to": "Aggregator",
                "reasoning": "WebSearch failed with no fallback"
            })
            
            return {
                "context": {**context, "websearch_failed": True, "no_data": True},
                "next_agent": "Aggregator"
            }
