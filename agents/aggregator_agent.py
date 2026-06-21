"""
Aggregator Agent — generates final answer and caches results.
"""

import json
from datetime import datetime
from langchain_groq import ChatGroq
from models import GraphState, AGENT_HANDOFFS
from decorators import agent_error_handler
from config import GROQ_API_KEY
from chroma_client import get_vectorstore

llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, api_key=GROQ_API_KEY)
ALLOWED_HANDOFFS = AGENT_HANDOFFS["Aggregator"]
CACHE_DIR = "query_cache_vectorstore"


@agent_error_handler
def aggregator_agent(state: GraphState):
    """Generate final answer from all available information."""
    print("---NODE: Aggregator---")

    context = state.get("context", {})
    user_query = state['query']
    query_lower = user_query.lower()
    thread_id = state.get("thread_id", "default")
    print(f"Processing in thread: {thread_id}")

    # Return cached answer if available
    if state.get("cache_hit") and state.get("cached_answer"):
        return {"answer": state['cached_answer'], "next_agent": "END"}

    # Handle casual queries
    if context.get("casual_query"):
        if "thank" in query_lower:
            answer = "You're welcome! Let me know if you have any questions about financial data."
        elif any(w in query_lower for w in ["hello", "hi", "hey"]):
            answer = "Hello! I'm here to help with financial document analysis. What would you like to know?"
        else:
            answer = "I'm here to help with financial document analysis. What would you like to know?"
        state["trace"].append({
            "agent": "Aggregator", "tool": "Casual response",
            "output": "Polite response", "handoff-to": "END",
            "reasoning": "Casual query handled"
        })
        return {"answer": answer, "next_agent": "END"}

    # Handle no usable data
    if context.get("no_usable_data"):
        checked_docs = "Yes" if context.get("retrieved_docs") else "No"
        checked_web = "Yes" if context.get("web_results") else "No"
        error_answer = (
            f"I couldn't find relevant information to answer: \"{user_query}\"\n\n"
            f"**What I tried:**\n- Searched local documents: {checked_docs}\n"
            f"- Searched the web: {checked_web}\n\n"
            "Please try rephrasing your query or uploading relevant documents."
        )
        state["trace"].append({
            "agent": "Aggregator", "tool": "Error Handler",
            "output": "No usable data", "handoff-to": "END",
            "reasoning": "All data sources insufficient"
        })
        return {"answer": error_answer, "next_agent": "END"}

    # Handle conversation history queries
    conversation_keywords = [
        "previous query", "what did i ask", "earlier", "before", "last question",
        "all queries", "conversation", "queries i asked", "brief of", "history"
    ]
    if any(phrase in query_lower for phrase in conversation_keywords):
        conversation_history = state.get("conversation_history", [])
        if not conversation_history:
            answer = "This is the first query in this conversation."
        else:
            history_text = "\n".join(conversation_history[-10:])
            try:
                response = llm.invoke(f"Summarize this conversation history:\n{history_text}\n\nUser asked: {user_query}")
                answer = response.content
            except Exception:
                answer = f"Conversation history:\n\n{history_text}"
        return {"answer": answer, "next_agent": "END"}

    # Check for available data
    has_data = (
        context.get("retrieved_docs") or context.get("web_results") or
        context.get("structured_data") or context.get("calculations") or
        context.get("summary")
    )
    if not has_data:
        return {
            "answer": f"Unable to retrieve information to answer: \"{user_query}\"\n\nPlease ensure documents are uploaded.",
            "next_agent": "END"
        }

    # Build context
    context_parts = []
    if context.get("retrieved_docs"):
        context_parts.append("**Documents:**\n" + "\n".join(context["retrieved_docs"][:3])[:3000])
    if context.get("web_results"):
        context_parts.append("**Web Results:**\n" + "\n".join(context["web_results"][:2])[:2000])
    if context.get("structured_data"):
        context_parts.append("**Data:**\n" + json.dumps(context["structured_data"], indent=2)[:2000])
    if context.get("calculations"):
        context_parts.append("**Calculations:**\n" + json.dumps(context["calculations"], indent=2)[:1000])
    if context.get("summary"):
        context_parts.append(f"**Summary:**\n{context['summary']}")

    context_text = "\n\n".join(context_parts) if context_parts else "(No context)"

    prompt = f"""You are an expert financial analyst. Answer clearly and accurately.

QUESTION: {user_query}

INFORMATION:
{context_text}

INSTRUCTIONS:
- Start with a direct answer (1-2 sentences)
- Use specific numbers and facts from the context
- Use **bold** for key numbers
- Do NOT say "based on the context"

ANSWER:
"""

    try:
        response = llm.invoke(prompt)
        answer = response.content

        # Quality check before caching
        LOW_QUALITY = [
            "i couldn't find", "unable to retrieve", "error generating",
            "i don't know", "no information", "cannot answer"
        ]
        if len(answer.strip()) >= 80 and not any(p in answer.lower() for p in LOW_QUALITY):
            try:
                cache_store = get_vectorstore(CACHE_DIR, collection_name="query_cache")
                cache_store.add_texts(
                    texts=[state["query"]],
                    metadatas=[{
                        "answer": answer,
                        "timestamp": datetime.now().isoformat(),
                        "thread_id": thread_id
                    }]
                )
                print(f"Answer cached in thread: {thread_id}")
            except Exception as cache_error:
                print(f"Cache save failed: {cache_error}")
        else:
            print("Answer quality check failed — skipping cache write")

        state["trace"].append({
            "agent": "Aggregator", "tool": "LLM",
            "output": "Answer generated", "handoff-to": "END",
            "reasoning": "Answer synthesized"
        })
        return {"answer": answer, "next_agent": "END"}

    except Exception as e:
        print(f"Answer generation failed: {e}")
        return {"answer": f"Error generating response for: {user_query}", "next_agent": "END"}