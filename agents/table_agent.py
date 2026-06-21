"""
Table Agent for structured data extraction.

This agent extracts numeric and tabular data from documents or web results
and routes to either Math agent for calculations or Aggregator for final response.
"""

import json
from langchain_groq import ChatGroq
from models import GraphState, LLMError, AGENT_HANDOFFS
from decorators import agent_error_handler, retry_with_exponential_backoff
from utils import parse_json_from_llm
from config import GROQ_API_KEY


llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=GROQ_API_KEY)

ALLOWED_HANDOFFS = AGENT_HANDOFFS["Table"]


@agent_error_handler
@retry_with_exponential_backoff(max_retries=2, errors_to_retry=(LLMError,))
def table_agent(state: GraphState):
    """
    Extract structured numeric data from available content.
    
    This agent uses an LLM to extract tabular and numeric data in a standardized
    JSON format, then routes to Math agent if calculations are needed or directly
    to Aggregator if the extracted data is sufficient.
    
    Args:
        state: Current graph state containing query and context
        
    Returns:
        dict: Updated state with structured data and routing decision
        
    Raises:
        LLMError: If LLM extraction fails
    """
    print("---NODE: Table---")
    print(f"Allowed handoffs: {ALLOWED_HANDOFFS}")
    
    context = state.get("context", {})
    
    # Check for available data sources
    if not context.get("retrieved_docs") and not context.get("web_results"):
        print("No source data available, routing to Aggregator")
        
        state["trace"].append({
            "agent": "Table",
            "tool": "N/A",
            "output": "No data",
            "handoff-to": "Aggregator",
            "reasoning": "No documents to extract from"
        })
        
        return {"next_agent": "Aggregator"}
    
    # Prepare context string with length limit
    context_str = json.dumps(context, default=str)
    
    if len(context_str) > 50000:
        context_str = context_str[:50000] + "... [truncated]"
    
    # Data extraction prompt
    prompt = f"""You are a specialized Table Extraction agent for financial document analysis.

TASK: Extract structured numeric data from the provided context in a standardized JSON format.

===== USER'S QUERY =====
{state['query']}

===== SOURCE CONTEXT =====
{context_str}

===== EXTRACTION INSTRUCTIONS =====

1. **Identify all numeric data** mentioned in the context (revenue, expenses, ratios, percentages, growth rates, etc.)

2. **Extract each data point** in this standardized format:
   - category: What type of data is this? (e.g., "Revenue", "Operating Expense", "Profit Margin", "Stock Price")
   - period: Time period (e.g., "Q1 2024", "FY 2023", "Jan 2024", "2024")
   - value: The numeric value (keep units: "$1.2M", "15%", "1,234,567")
   - unit: Unit of measurement ("USD", "percent", "millions", "thousands")
   - entity: Company/entity name if mentioned

3. **Focus on data relevant to the query** but extract all available numeric information

4. **Preserve context**: If a table shows quarterly comparisons, extract all quarters

===== OUTPUT FORMAT =====
Return a JSON array of data points:

[
  {{
    "category": "Revenue",
    "period": "Q1 2024",
    "value": "$12,836,000",
    "unit": "USD",
    "entity": "Company ABC"
  }},
  {{
    "category": "Operating Expense",
    "period": "Q1 2024",
    "value": "$8,245,000",
    "unit": "USD",
    "entity": "Company ABC"
  }}
]

If no numeric data found, return: {{"error": "No numeric data found in context"}}

===== YOUR EXTRACTION =====
"""
    
    try:
        response = llm.invoke(prompt)
        
        if not response or not response.content:
            raise LLMError("LLM returned empty response")
        
        # Parse JSON from LLM response
        parsed = parse_json_from_llm(response.content)
        
        if parsed is None:
            parsed = {"raw_text": response.content}
        
        # Determine next agent based on query requirements
        query_lower = state["query"].lower()
        calculation_keywords = ["compare", "calculate", "ratio", "growth", "change"]
        needs_calculation = any(kw in query_lower for kw in calculation_keywords)
        
        if needs_calculation and parsed:
            next_agent = "Math"
            reasoning = "Data extracted, performing calculations"
        else:
            next_agent = "Aggregator"
            reasoning = "Data extracted, sufficient for answer"
        
        print(f"Data extraction complete, routing to {next_agent}")
        
        state["trace"].append({
            "agent": "Table",
            "tool": "LLM",
            "output": f"Extracted data - routing to {next_agent}",
            "handoff-to": next_agent,
            "reasoning": reasoning
        })
        
        return {
            "context": {**context, "structured_data": parsed},
            "next_agent": next_agent
        }
    
    except Exception as e:
        print(f"Table extraction failed: {e}")
        
        state["trace"].append({
            "agent": "Table",
            "tool": "LLM",
            "error": str(e),
            "handoff-to": "Aggregator",
            "reasoning": "Extraction failed"
        })
        
        return {
            "context": {**context, "table_failed": True},
            "next_agent": "Aggregator"
        }