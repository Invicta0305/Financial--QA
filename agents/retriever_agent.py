"""
Retriever Agent — fetches relevant docs from the main vectorstore.
"""

from models import GraphState, RetrieverError, AGENT_HANDOFFS
from decorators import agent_error_handler, retry_with_exponential_backoff
from config import DEFAULT_DB_PATH
from chroma_client import get_vectorstore

ALLOWED_HANDOFFS = AGENT_HANDOFFS["Retriever"]


@agent_error_handler
@retry_with_exponential_backoff(max_retries=3, initial_delay=1.0)
def retrieve_agent(state: GraphState):
    """Retrieve relevant documents from vector store."""
    print("---NODE: Retriever---")
    print("Retrieving documents from vector store")

    query = state["query"]
    if not query:
        raise ValueError("Empty query for retrieval")

    try:
        # Uses shared PersistentClient — no tenant conflict
        vectorstore = get_vectorstore(DEFAULT_DB_PATH, collection_name="langchain")
        retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
        docs = retriever.invoke(query)

        if not docs:
            raise RetrieverError("Vector store returned no documents")

        valid_context = [
            doc.page_content for doc in docs
            if doc.page_content and len(doc.page_content.strip()) > 10
        ]

        if not valid_context:
            raise RetrieverError("Retrieved documents have no useful content")

        print(f"Retrieved {len(valid_context)} valid documents")
        print("Forwarding to Validator for assessment")

        state["trace"].append({
            "agent": "Retriever",
            "tool": "vector_store",
            "input": query,
            "output": f"Retrieved {len(valid_context)} documents",
            "handoff-to": "Validator",
            "reasoning": "Documents retrieved successfully"
        })

        return {
            "context": {
                **state.get("context", {}),
                "retrieved_docs": valid_context,
                "data_source": "documents"
            },
            "next_agent": "Validator"
        }

    except Exception as e:
        print(f"Retrieval failed: {e}")
        print("Routing to WebSearch for external data")

        state["trace"].append({
            "agent": "Retriever",
            "tool": "vector_store",
            "error": str(e),
            "output": "Document retrieval failed",
            "handoff-to": "WebSearch",
            "reasoning": "Vector store retrieval failed, trying WebSearch"
        })

        return {
            "context": {
                **state.get("context", {}),
                "retriever_failed": True
            },
            "next_agent": "WebSearch"
        }