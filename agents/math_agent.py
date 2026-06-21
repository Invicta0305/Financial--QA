"""
Math Agent for financial calculations and analysis.

This agent performs calculations on structured data extracted by the Table agent,
including comparisons, growth rates, ratios, and other financial metrics.
"""

import json
from langchain_groq import ChatGroq
from models import GraphState, LLMError, AGENT_HANDOFFS
from decorators import agent_error_handler, retry_with_exponential_backoff
from utils import parse_json_from_llm
from config import GROQ_API_KEY


llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=GROQ_API_KEY)

ALLOWED_HANDOFFS = AGENT_HANDOFFS["Math"]


@agent_error_handler
@retry_with_exponential_backoff(max_retries=2)
def math_agent(state: GraphState):
    """
    Perform financial calculations on structured data.
    
    This agent executes calculations such as comparisons, growth rates, ratios,
    and aggregations based on the user's query and structured data from Table agent.
    
    Args:
        state: Current graph state containing query and structured data context
        
    Returns:
        dict: Updated state with calculation results and routing to Aggregator
        
    Raises:
        LLMError: If LLM calculation fails
    """
    print("---NODE: Math---")
    print(f"Allowed handoffs: {ALLOWED_HANDOFFS}")
    
    context = state.get("context", {})
    
    if not context.get("structured_data"):
        print("No structured data available, routing to Aggregator")
        
        state["trace"].append({
            "agent": "Math",
            "tool": "N/A",
            "output": "No data",
            "handoff-to": "Aggregator",
            "reasoning": "No structured data for calculations"
        })
        
        return {"next_agent": "Aggregator"}
    
    context_str = json.dumps(context, default=str)
    
    prompt = f"""You are a Math agent specialized in financial calculations and analysis.

TASK: Perform calculations based on the user's query and the extracted structured data.

===== USER'S QUERY =====
{state['query']}

===== EXTRACTED STRUCTURED DATA =====
{context_str}

===== CALCULATION INSTRUCTIONS =====

1. **Understand what calculation is needed** from the query:
   - Comparison (Q1 vs Q2, Year-over-year)
   - Growth rate / Percentage change
   - Ratios (profit margin, debt-to-equity, etc.)
   - Totals, averages, or aggregations
   - Differences (increase/decrease)

2. **Perform calculations step-by-step**:
   - Show your work
   - Use the actual values from the structured data
   - Perform arithmetic correctly
   - Calculate percentages accurately

3. **Provide context** for the results:
   - What the numbers mean
   - Whether increase/decrease is significant
   - Relevant comparisons

===== OUTPUT FORMAT =====
Return JSON with calculated results:

{{
  "calculation_type": "comparison" | "growth_rate" | "ratio" | "total" | "difference",
  "results": [
    {{
      "metric": "Q1 Expenses",
      "value": "$12,836,000",
      "period": "Q1 2024"
    }},
    {{
      "metric": "Q4 Expenses",
      "value": "$14,500,000",
      "period": "Q4 2023"
    }},
    {{
      "metric": "Difference",
      "value": "$1,664,000 decrease",
      "percentage_change": "-11.5%"
    }}
  ],
  "interpretation": "Q1 2024 expenses decreased by 11.5% compared to Q4 2023, indicating cost reduction efforts."
}}

===== YOUR CALCULATIONS =====
"""
    
    try:
        response = llm.invoke(prompt)
        
        if not response or not response.content:
            raise LLMError("LLM returned empty response")
        
        parsed = parse_json_from_llm(response.content)
        
        if parsed is None:
            parsed = {"raw_calculation": response.content}
        
        print("Calculations complete")
        
        # Route to Summarizer instead of Aggregator if Validator flagged this as a compound query (calc + summary)
        needs_summary = context.get("needs_summary", False)
        next_agent = "Summarizer" if needs_summary else "Aggregator"

        state["trace"].append({
            "agent": "Math",
            "tool": "LLM",
            "output": "Calculations done",
            "handoff-to": next_agent,
            "reasoning": "Computed results" + ("; handing to Summarizer for compound query" if needs_summary else "")
        })
        
        return {
            "context": {**context, "calculations": parsed},
            "next_agent": next_agent
        }
    
    except Exception as e:
        print(f"Calculations failed: {e}")
        
        state["trace"].append({
            "agent": "Math",
            "tool": "LLM",
            "error": str(e),
            "handoff-to": "Aggregator",
            "reasoning": "Calculation failed"
        })
        
        return {
            "context": {**context, "math_failed": True},
            "next_agent": "Aggregator"
        }