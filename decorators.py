"""
Reusable decorators for error handling and retry logic.

This module provides decorators for exponential backoff retry logic
and standardized error handling across agent functions.
"""

import time
from functools import wraps
from typing import Callable, Any, Dict
from models import GraphState


def retry_with_exponential_backoff(
    max_retries: int = 3,
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    errors_to_retry: tuple = (Exception,)
):
    """
    Retry a function with exponential backoff on failure.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before first retry
        exponential_base: Multiplier for delay after each retry
        errors_to_retry: Tuple of exception types to catch and retry
        
    Returns:
        Decorated function with retry logic
        
    Example:
        @retry_with_exponential_backoff(max_retries=3)
        def api_call():
            return make_request()
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            delay = initial_delay
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except errors_to_retry as e:
                    if attempt == max_retries - 1:
                        raise
                    
                    print(f"Attempt {attempt + 1} failed: {e}")
                    print(f"Retrying in {delay:.1f} seconds...")
                    time.sleep(delay)
                    delay *= exponential_base
            
            return func(*args, **kwargs)
        
        return wrapper
    return decorator


def agent_error_handler(func: Callable) -> Callable:
    """
    Handle errors in agent functions with standardized fallback behavior.
    
    This decorator catches exceptions during agent execution and routes
    to the Aggregator agent with error information added to the trace.
    
    Args:
        func: Agent function to wrap
        
    Returns:
        Wrapped function with error handling
        
    Example:
        @agent_error_handler
        def my_agent(state: GraphState) -> Dict:
            return process_state(state)
    """
    @wraps(func)
    def wrapper(state: GraphState) -> Dict:
        try:
            return func(state)
        
        except Exception as e:
            agent_name = func.__name__.replace("_agent", "").replace("_", " ").title()
            fallback_agent = "Aggregator"
            
            error_msg = f"{agent_name} Error: {type(e).__name__}: {str(e)}"
            print(error_msg)
            
            # Add error details to trace
            state["trace"].append({
                "agent": agent_name,
                "tool": "ERROR",
                "error": str(e),
                "error_type": type(e).__name__,
                "fallback": f"Routing to {fallback_agent}",
                "handoff-to": fallback_agent,
                "reasoning": f"Error occurred, falling back to {fallback_agent}"
            })
            
            # Return state with error context and fallback routing
            return {
                "context": {
                    **state.get("context", {}),
                    f"{agent_name.lower()}_failed": True,
                    f"{agent_name.lower()}_error": str(e)
                },
                "next_agent": fallback_agent
            }
    
    return wrapper
