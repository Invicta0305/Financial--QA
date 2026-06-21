"""
Validator Agent for intent classification and query routing.

This agent uses LLM-based classification to determine query intent
and routes to appropriate downstream agents based on query type and data availability.
"""

from langchain_groq import ChatGroq
from models import GraphState, AGENT_HANDOFFS
from utils import parse_json_from_llm
from decorators import agent_error_handler 
from config import GROQ_API_KEY


llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0, api_key=GROQ_API_KEY)

ALLOWED_HANDOFFS = AGENT_HANDOFFS["Validator"]


@agent_error_handler
def validator_agent(state: GraphState):
    """
    Classify query intent and route to appropriate agent.
    
    This agent performs three-tier classification:
    1. Intent classification (casual, conversation history, or informational)
    2. Query type detection (calculation, summary, etc.)
    3. Data relevance validation
    
    Args:
        state: Current graph state containing query and context
        
    Returns:
        dict: Updated state with routing decision to next agent
    """
    print("---NODE: Validator---")
    print("Classifying query intent and determining routing")
    
    query = state["query"]
    context = state.get("context", {})
    
    # Intent classification prompt
    intent_prompt = f"""You are an intent classifier for a financial document analysis system.

Classify the user's query into ONE of these categories:

**A) CASUAL** - Greetings, thanks, compliments, non-informational statements
Examples: "hello", "thank you", "you're great", "good job", "hi there"

**B) CONVERSATION_HISTORY** - Questions about previous queries/conversation
Examples: "what did I ask before", "previous queries", "our conversation"

**C) INFORMATIONAL** - Questions needing document retrieval, calculation, or web search
Examples: "what is the revenue", "calculate margin", "who is the CEO", "summarize risks"

===== USER QUERY =====
{query}

===== CLASSIFICATION RULES =====
- If query is a greeting, thanks, or compliment → A (CASUAL)
- If query asks about conversation history → B (CONVERSATION_HISTORY)
- If query asks for information, facts, or analysis → C (INFORMATIONAL)
- If query is a statement without a question → likely A (CASUAL)
- If unsure, default to C (INFORMATIONAL)

Return JSON:
{{
  "intent": "A" | "B" | "C",
  "confidence": "high" | "medium" | "low",
  "reasoning": "brief explanation"
}}

Your classification:
"""
    
    try:
        print("Classifying query intent with LLM...")
        intent_response = llm.invoke(intent_prompt)
        intent_result = parse_json_from_llm(intent_response.content)
        
        if not intent_result:
            print("Intent classification failed, treating as INFORMATIONAL")
            intent_type = "C"
        else:
            intent_type = intent_result.get("intent", "C")
            confidence = intent_result.get("confidence", "medium")
            reasoning = intent_result.get("reasoning", "Intent classified")
            
            print(f"Intent: {intent_type} (confidence: {confidence})")
            print(f"Reasoning: {reasoning}")
        
        # Route based on intent classification
        
        # Casual queries (greetings, thanks)
        if intent_type == "A":
            print("Casual query detected, routing to Aggregator")
            
            state["trace"].append({
                "agent": "Validator",
                "tool": "LLM intent classification",
                "output": "Casual query",
                "handoff-to": "Aggregator",
                "reasoning": f"Classified as casual: {reasoning}"
            })
            
            return {
                "context": {**context, "casual_query": True},
                "next_agent": "Aggregator"
            }
        
        # Conversation history queries
        elif intent_type == "B":
            print("Conversation history query, routing to Aggregator")
            
            state["trace"].append({
                "agent": "Validator",
                "tool": "LLM intent classification",
                "output": "Conversation history query",
                "handoff-to": "Aggregator",
                "reasoning": f"Asking about conversation: {reasoning}"
            })
            
            return {
                "context": {**context, "conversation_query": True},
                "next_agent": "Aggregator"
            }
        
        # Informational queries
        else:
            print("Informational query, proceeding with data validation")
        
    except Exception as e:
        print(f"Intent classification failed: {e}")
        print("Defaulting to INFORMATIONAL (safe fallback)")
    
    # Check data availability for informational queries
    has_docs = bool(context.get("retrieved_docs"))
    has_web = bool(context.get("web_results"))
    
    print(f"Data availability - Documents: {has_docs}, Web: {has_web}")
    
    # No data available - route to WebSearch
    if not has_docs and not has_web:
        print("No data available, routing to WebSearch")
        
        state["trace"].append({
            "agent": "Validator",
            "tool": "Data check",
            "output": "No data",
            "handoff-to": "WebSearch",
            "reasoning": "No data sources available"
        })
        return {"next_agent": "WebSearch"}
    
    # Query type detection for data processing
    query_lower = query.lower()
    
    # FIX: Detect BOTH flags before routing — previously an early-return on
    # needs_calculation meant needs_summary was never evaluated, so compound
    # queries like "calculate revenue growth and summarize findings" always
    # skipped the Summarizer entirely.
    calculation_keywords = ["calculate", "compute", "compare", "difference",
                            "ratio", "margin", "growth", "percentage"]
    summary_keywords = ["summarize", "summary", "overview", "explain", "describe"]

    needs_calculation = any(kw in query_lower for kw in calculation_keywords)
    needs_summary = any(kw in query_lower for kw in summary_keywords)

    if needs_calculation and (has_docs or has_web):
        print("Calculation query detected, routing to Table agent")
        # Carry needs_summary flag so Math agent can decide whether to
        # continue to Summarizer or go straight to Aggregator.
        state["trace"].append({
            "agent": "Validator",
            "tool": "Query-type detection",
            "output": "Calculation query" + (" + summary" if needs_summary else ""),
            "handoff-to": "Table",
            "reasoning": "Query needs numeric extraction and calculation"
        })
        return {
            "context": {**context, "needs_summary": needs_summary},
            "next_agent": "Table"
        }

    if needs_summary and (has_docs or has_web):
        print("Summary query detected, routing to Summarizer")
        state["trace"].append({
            "agent": "Validator",
            "tool": "Query-type detection",
            "output": "Summary query",
            "handoff-to": "Summarizer",
            "reasoning": "Query needs summarization"
        })
        return {"context": context, "next_agent": "Summarizer"}
    
    # Data relevance validation
    
    # FIX: If WebSearch already ran (websearch_failed or web_results present),
    # never send back to WebSearch — that's the main recursion trigger.
    # Just use whatever data we have and proceed to Aggregator.
    websearch_already_ran = (
        context.get("websearch_failed") or
        context.get("web_results") or
        context.get("retriever_failed")
    )

    # Prepare data sample for validation
    if has_web:
        data_sample = "\n\n".join(context["web_results"][:3])[:2500]
        data_type = "web search results"
    else:
        data_sample = "\n\n".join(context["retrieved_docs"][:3])[:2500]
        data_type = "retrieved documents"
    
    validation_prompt = f"""Check if the data can answer the query.

QUERY: {query}

DATA ({data_type.upper()}):
{data_sample}

Is data relevant? Answer YES or NO with brief reason.

Return JSON:
{{
  "relevant": true | false,
  "reasoning": "why"
}}
"""
    
    try:
        response = llm.invoke(validation_prompt)
        result = parse_json_from_llm(response.content)
        
        if not result:
            next_agent = "Aggregator"
            reasoning = "Parsing failed"
        else:
            is_relevant = result.get("relevant", False)
            reasoning = result.get("reasoning", "Validation complete")
            
            # Determine next agent based on relevance
            if not is_relevant and not has_web and not websearch_already_ran:
                # Try web search only if we haven't already tried it
                next_agent = "WebSearch"
            else:
                # Data is relevant, or we've already tried all sources — go to Aggregator
                if not is_relevant:
                    context["no_usable_data"] = True
                next_agent = "Aggregator"
            
            print(f"Relevance check: {is_relevant} - Routing to {next_agent}")
        
        # Validate handoff is allowed
        if next_agent not in ALLOWED_HANDOFFS:
            next_agent = "Aggregator"
        
        state["trace"].append({
            "agent": "Validator",
            "tool": "LLM validation",
            "output": f"Validated - routing to {next_agent}",
            "handoff-to": next_agent,
            "reasoning": reasoning
        })
        
        return {"context": context, "next_agent": next_agent}
    
    except Exception as e:
        print(f"Validation failed: {e}")
        
        # FIX: Never route back to WebSearch on exception if it already ran
        if not has_web and not websearch_already_ran:
            return {"next_agent": "WebSearch"}
        else:
            return {
                "context": {**context, "no_usable_data": True},
                "next_agent": "Aggregator"
            }