"""
Utility functions for state validation and JSON parsing.

This module provides helper functions for validating graph state
and extracting JSON from LLM responses.
"""

import re
import json
from models import GraphState


def validate_state(state: GraphState) -> bool:
    """
    Validate that state contains all required fields.
    
    Args:
        state: GraphState object to validate
        
    Returns:
        bool: True if state is valid, False otherwise
    """
    required_fields = ["query", "trace", "next_agent"]
    
    for field in required_fields:
        if field not in state:
            print(f"Warning: Missing required field '{field}' in state")
            return False
    
    return True


def parse_json_from_llm(text: str):
    """
    Extract and parse JSON from LLM response text.
    
    This function handles various JSON formats that LLMs may return,
    including code blocks with backticks and embedded JSON within text.
    
    Args:
        text: Raw text from LLM that may contain JSON
        
    Returns:
        dict or list: Parsed JSON object, or None if parsing fails
        
    Example:
        >>> text = "``````"
        >>> parse_json_from_llm(text)
        {'key': 'value'}
    """
    if not text:
        return None
    
    text = text.strip()
    
    # Strip a trailing ``` code-block marker, if present
    text = re.sub(r'\s*```$', '', text)
    
    match = re.search(r'(\[.*\]|\{.*\})', text, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        json_str = text
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        print("Warning: Failed to parse JSON from LLM response")
        return None