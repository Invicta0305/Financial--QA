"""
Streamlit Chat Interface for Financial Document Analysis.

This application provides a multi-agent chatbot interface for querying
financial documents with conversation management, caching, and PDF upload capabilities.

Run with: streamlit run app_streamlit.py
"""

import streamlit as st
import json
import os
from datetime import datetime
from pathlib import Path
import shutil
import uuid

from graph import build_graph
from ingest import create_vector_store
from config import DEFAULT_DB_PATH


# Configuration
CONVERSATIONS_DIR = Path("conversations")
CONVERSATIONS_DIR.mkdir(exist_ok=True)


def save_conversation(thread_id, messages, conversation_history):
    """
    Save conversation to disk for persistence.
    
    Args:
        thread_id: Unique identifier for the conversation thread
        messages: List of message dictionaries with role and content
        conversation_history: List of conversation context strings
    """
    conv_file = CONVERSATIONS_DIR / f"{thread_id}.json"
    data = {
        "thread_id": thread_id,
        "created_at": datetime.now().isoformat(),
        "messages": messages,
        "conversation_history": conversation_history,
        "title": messages[0]["content"][:50] if messages else "New Conversation"
    }
    with open(conv_file, 'w') as f:
        json.dump(data, f, indent=2)


def load_conversation(thread_id):
    """
    Load a previously saved conversation from disk.
    
    Args:
        thread_id: Unique identifier for the conversation to load
        
    Returns:
        dict: Conversation data or None if not found
    """
    conv_file = CONVERSATIONS_DIR / f"{thread_id}.json"
    if conv_file.exists():
        with open(conv_file, 'r') as f:
            return json.load(f)
    return None


def list_conversations():
    """
    List all saved conversations sorted by creation time.
    
    Returns:
        list: List of conversation metadata dictionaries
    """
    conversations = []
    for conv_file in CONVERSATIONS_DIR.glob("*.json"):
        try:
            with open(conv_file, 'r') as f:
                data = json.load(f)
                conversations.append({
                    "thread_id": data["thread_id"],
                    "title": data.get("title", "Untitled"),
                    "created_at": data.get("created_at", "")
                })
        except:
            pass
    
    conversations.sort(key=lambda x: x["created_at"], reverse=True)
    return conversations


def clear_query_cache():
    """
    Clear the query cache vector store.
    
    Returns:
        bool: True if successful, False otherwise
    """
    cache_path = "query_cache_vectorstore"
    if os.path.exists(cache_path):
        try:
            shutil.rmtree(cache_path)
            return True
        except Exception as e:
            print(f"Cache clear error: {e}")
            return False
    return True


# Page configuration
st.set_page_config(
    page_title="Multi-Agent Q&A for Financial Documents",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .cache-badge {
        background-color: #90EE90;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        font-weight: bold;
        color: #006400;
    }
    .agent-flow {
        background-color: #e0f7fa;
        color: black;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        font-family: monospace;
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        margin-top: 1rem;
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border: 2px dashed #1f77b4;
    }
    .doc-item {
        background-color: #e8f4f8;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.3rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize Streamlit session state variables with default values."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "processed_documents" not in st.session_state:
        st.session_state.processed_documents = []
    
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.bypass_cache_once = True
    
    if "vector_store_ready" not in st.session_state:
        st.session_state.vector_store_ready = os.path.exists(DEFAULT_DB_PATH)
    
    if "conversation_count" not in st.session_state:
        st.session_state.conversation_count = 0
    
    if "processing_queue" not in st.session_state:
        st.session_state.processing_queue = []
    
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []


@st.cache_resource
def get_graph():
    """
    Get or create the agent graph instance with caching.
    
    Returns:
        Compiled LangGraph workflow or None if initialization fails
    """
    if not os.environ.get("GROQ_API_KEY"):
        st.error("GROQ_API_KEY not configured. Please check your .env file.")
        return None
    
    try:
        return build_graph()
    except Exception as e:
        st.error(f"Failed to initialize graph: {e}")
        return None


def process_pdf_file(uploaded_file, temp_dir="temp_uploads"):
    """
    Process uploaded PDF file and create vector store.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        temp_dir: Temporary directory for file storage
        
    Returns:
        tuple: (success: bool, message: str)
    """
    os.makedirs(temp_dir, exist_ok=True)
    
    pdf_path = os.path.join(temp_dir, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        progress_placeholder = st.empty()
        
        with progress_placeholder.container():
            st.info("Processing PDF document...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Step 1/3: Extracting content from PDF...")
            progress_bar.progress(33)
            
            # FIX: ChromaDB error code 14 = "unable to open database file"
            # happens when the vectorstore directory doesn't exist yet.
            # Create it explicitly before calling create_vector_store.
            os.makedirs(DEFAULT_DB_PATH, exist_ok=True)
            
            create_vector_store(pdf_path, db_path=DEFAULT_DB_PATH)
            
            status_text.text("Step 2/3: Creating embeddings...")
            progress_bar.progress(66)
            
            status_text.text("Step 3/3: Building vector store...")
            progress_bar.progress(100)
        
        progress_placeholder.empty()
        
        doc_info = {
            "name": uploaded_file.name,
            "size": uploaded_file.size,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if doc_info not in st.session_state.processed_documents:
            st.session_state.processed_documents.append(doc_info)
        
        st.session_state.vector_store_ready = True
        
        return True, "Document processed successfully!"
    
    except Exception as e:
        err_str = str(e)
        # FIX: WinError 32 = file locked by another process (leftover from ingest.py).
        # Give a clear actionable message instead of a cryptic OS error.
        if "WinError 32" in err_str or "being used by another process" in err_str:
            import glob, os as _os
            # Clean up SQLite WAL/SHM leftover lock files
            for lock_file in glob.glob(f"{DEFAULT_DB_PATH}/chroma.sqlite3-*"):
                try:
                    _os.remove(lock_file)
                except Exception:
                    pass
            return False, (
                "The vector store file is locked (WinError 32).\n\n"
                "Fix: Close any terminal that ran ingest.py, wait 5 seconds, then click 'Process' again. "
                "If it persists, restart VS Code completely and try again."
            )
        return False, f"Error processing document: {err_str}"
    
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


def display_agent_trace(trace_data):
    """
    Display agent execution trace in a formatted UI.
    
    Args:
        trace_data: List of trace step dictionaries
    """
    if not trace_data:
        st.info("No trace data available")
        return
    
    agents_used = [t["agent"] for t in trace_data]
    
    st.markdown("**Agent Flow:**")
    flow_text = " → ".join(agents_used)
    st.markdown(f'<div class="agent-flow">{flow_text}</div>', unsafe_allow_html=True)
    
    st.markdown("**Detailed Steps:**")
    for i, step in enumerate(trace_data, 1):
        with st.container():
            col1, col2 = st.columns([1, 4])
            with col1:
                st.markdown(f"**Step {i}**")
            with col2:
                st.markdown(f"**Agent:** `{step.get('agent', 'N/A')}`")
                st.markdown(f"**Tool:** {step.get('tool', 'N/A')}")
                st.markdown(f"**Handoff to:** {step.get('handoff-to', 'N/A')}")
                
                if 'reasoning' in step:
                    st.markdown(f"*Reasoning:* {step['reasoning']}")
                
                if 'error' in step:
                    st.error(f"Error: {step['error']}")
            
            if i < len(trace_data):
                st.markdown("---")


def process_query(query, graph_app):
    """
    Process user query through the agent graph.
    
    Args:
        query: User's input query string
        graph_app: Compiled LangGraph workflow
        
    Returns:
        dict: Final state from graph execution
    """
    bypass_cache = st.session_state.get('force_bypass_cache', False) or \
                   st.session_state.get('bypass_cache_once', False)
    
    inputs = {
        "query": query,
        "context": {},
        "trace": [],
        "answer": "",
        "next_agent": "Memory",
        "messages": [],
        "cache_hit": False,
        "cached_answer": "",
        "conversation_history": st.session_state.conversation_history,
        "thread_id": st.session_state.thread_id,
        "bypass_cache": bypass_cache,
        "hop_count": 0  # FIX: initialize hop counter for our hop-limit safety wrapper
    }
    
    # FIX: Raise recursion limit above default 25 — our hop_limit wrapper stops
    # at 12 hops, but LangGraph counts every node visit. 50 gives plenty of headroom.
    config = {
        "configurable": {"thread_id": st.session_state.thread_id},
        "recursion_limit": 50
    }
    
    with st.spinner("Thinking..."):
        final_state = graph_app.invoke(inputs, config=config)
    
    # Reset bypass flags after use
    if st.session_state.get('bypass_cache_once', False):
        st.session_state.bypass_cache_once = False
    if st.session_state.get('force_bypass_cache', False):
        st.session_state.force_bypass_cache = False
    
    return final_state


def render_sidebar():
    """Render sidebar with conversation management, document upload, and settings."""
    with st.sidebar:
        st.header("💬 Conversations")
        
        # New conversation button
        if st.button("➕ New Conversation", use_container_width=True):
            if st.session_state.messages:
                save_conversation(
                    st.session_state.thread_id,
                    st.session_state.messages,
                    st.session_state.conversation_history
                )
            
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.conversation_history = []
            st.session_state.conversation_count = 0
            st.session_state.bypass_cache_once = True
            
            st.success("New conversation started!")
            st.rerun()
        
        st.divider()
        
        # Past conversations list
        st.subheader("📚 Past Conversations")
        
        past_convs = list_conversations()
        
        if past_convs:
            for conv in past_convs[:10]:
                is_current = conv["thread_id"] == st.session_state.thread_id
                button_label = f"{'🟢 ' if is_current else ''}{conv['title'][:35]}..."
                
                if st.button(button_label, key=conv["thread_id"], use_container_width=True):
                    if not is_current:
                        if st.session_state.messages:
                            save_conversation(
                                st.session_state.thread_id,
                                st.session_state.messages,
                                st.session_state.conversation_history
                            )
                        
                        loaded = load_conversation(conv["thread_id"])
                        if loaded:
                            st.session_state.thread_id = loaded["thread_id"]
                            st.session_state.messages = loaded["messages"]
                            st.session_state.conversation_history = loaded.get("conversation_history", [])
                            st.session_state.conversation_count = len([m for m in loaded["messages"] if m["role"] == "user"])
                            st.rerun()
        else:
            st.info("No past conversations yet")
        
        st.divider()
        
        # Document management
        st.header("📁 Document Management")
        
        with st.expander("📤 Upload PDF Document", expanded=not st.session_state.vector_store_ready):
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Choose a PDF file",
                type=['pdf'],
                help="Upload a financial document (PDF format)"
            )
            
            if uploaded_file is not None:
                st.write(f"📄 **File:** {uploaded_file.name}")
                st.write(f"📊 **Size:** {uploaded_file.size / 1024:.2f} KB")
                
                if st.button("🚀 Process Document", use_container_width=True):
                    success, message = process_pdf_file(uploaded_file)
                    
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Processed documents list
        if st.session_state.processed_documents:
            st.markdown("### 📚 Processed Documents")
            for doc in st.session_state.processed_documents:
                st.markdown(f"""
                <div class="doc-item">
                    📄 <b>{doc['name']}</b><br>
                    <small>Size: {doc['size']/1024:.1f} KB | {doc['timestamp']}</small>
                </div>
                """, unsafe_allow_html=True)
        
        st.divider()
        
        # Cache control settings
        with st.expander("⚙️ Cache Settings", expanded=False):
            bypass_next = st.checkbox(
                "🔄 Bypass cache for next query", 
                value=st.session_state.get('force_bypass_cache', False),
                help="Skip cache lookup for your next question"
            )
            
            if bypass_next:
                st.session_state.force_bypass_cache = True
                st.info("Next query will bypass cache")
            else:
                st.session_state.force_bypass_cache = False
            
            st.divider()
            
            if st.button("🗑️ Clear All Cache", use_container_width=True):
                if clear_query_cache():
                    st.success("Cache cleared! System reloading...")
                    import time
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error("Failed to clear cache")
        
        st.divider()
        
        # Action buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.conversation_history = []
                st.session_state.conversation_count = 0
                st.rerun()
        
        with col2:
            if st.button("🔄 Reset Vector", use_container_width=True):
                from chroma_client import reset_collection
                reset_collection(DEFAULT_DB_PATH, "langchain")
                st.session_state.processed_documents = []
                st.session_state.vector_store_ready = False
                st.success("Vector store reset!")
                st.rerun()
        
        st.divider()
        
        # Session statistics
        st.markdown("### 📊 Session Stats")
        st.metric("Documents Processed", len(st.session_state.processed_documents))
        st.metric("Queries in Session", st.session_state.conversation_count)
        st.metric("Thread ID", st.session_state.thread_id[:8] + "...")
        
        cache_status = "🔴 Bypassing" if st.session_state.get('force_bypass_cache', False) else "🟢 Active"
        st.metric("Cache Status", cache_status)
        
        st.divider()
        
        # Information and help
        st.markdown("### ℹ️ About")
        st.info("""
        This AI assistant uses a **multi-agent swarm system** powered by:
        - 🧠 LangGraph orchestration
        - 📚 Vector database retrieval
        - 🌐 Web search (Tavily)
        - 🔢 Specialized calculation agents
        - 📄 Thread-aware caching
        - 💬 Conversation memory
        """)
        
        # Example queries
        if st.session_state.vector_store_ready:
            st.markdown("### 💡 Example Queries")
            example_queries = [
                "What was the revenue in 2023?",
                "Calculate the profit margin",
                "Summarize the financial risks",
                "Compare Q1 and Q2 expenses"
            ]
            
            for example in example_queries:
                if st.button(f"📝 {example}", key=f"example_{example}", use_container_width=True):
                    st.session_state.next_query = example


def main():
    """Main application entry point with chat interface and agent orchestration."""
    initialize_session_state()
    
    # Header
    st.markdown('<div class="main-header">💼 Multi-Agent Q&A for Financial Documents</div>', unsafe_allow_html=True)
    st.markdown("Upload financial documents and ask questions about them!")
    
    # Render sidebar
    render_sidebar()
    
    # Check if vector store is ready
    if not st.session_state.vector_store_ready:
        st.info("👈 Please upload a PDF document to get started!")
        
        st.markdown("""
        ### 🚀 Getting Started
        
        1. **Upload Document**: Click on "Upload PDF Document" in the sidebar
        2. **Process**: Click "Process Document" to analyze your PDF
        3. **Ask Questions**: Once processed, start asking questions in the chat
        
        ### 📝 What You Can Ask
        
        - Financial metrics and ratios
        - Revenue and expense information
        - Summaries of key points
        - Comparisons across time periods
        - Calculations based on document data
        - Previous conversation history
        """)
        
        return
    
    # Initialize graph
    graph_app = get_graph()
    
    if graph_app is None:
        st.error("Failed to initialize the system. Please check your configuration.")
        return
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "metadata" in message:
                metadata = message["metadata"]
                
                if metadata.get("cache_hit"):
                    st.markdown(
                        '<span class="cache-badge">✅ Cached Response</span>', 
                        unsafe_allow_html=True
                    )
                
                with st.expander("🔍 View Agent Execution Trace", expanded=False):
                    display_agent_trace(metadata.get("trace", []))
                
                with st.expander("📄 View Full JSON Log", expanded=False):
                    st.json(metadata.get("full_log", {}))
    
    # Handle user input
    if hasattr(st.session_state, 'next_query'):
        query = st.session_state.next_query
        del st.session_state.next_query
    else:
        query = st.chat_input("💬 Ask a financial question...")
    
    # Process new query
    if query:
        st.session_state.conversation_count += 1
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })
        
        # Process through agent graph
        final_state = process_query(query, graph_app)
        
        answer = final_state.get("answer", "No answer generated")
        trace = final_state.get("trace", [])
        cache_hit = final_state.get("cache_hit", False)
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(answer)
            
            if cache_hit:
                st.markdown(
                    '<span class="cache-badge">✅ Cached Response</span>', 
                    unsafe_allow_html=True
                )
            elif st.session_state.get('force_bypass_cache_used', False):
                st.markdown(
                    '<span class="cache-badge" style="background-color: #FFB347;">🔄 Cache Bypassed</span>', 
                    unsafe_allow_html=True
                )
            
            with st.expander("🔍 View Agent Execution Trace", expanded=False):
                display_agent_trace(trace)
            
            with st.expander("📄 View Full JSON Log", expanded=False):
                full_log = {
                    "query": query,
                    "trace": trace,
                    "final_answer": answer,
                    "cache_hit": cache_hit,
                    "thread_id": st.session_state.thread_id,
                    "timestamp": datetime.now().isoformat()
                }
                st.json(full_log)
        
        # Add assistant message to history
        assistant_message = {
            "role": "assistant",
            "content": answer,
            "metadata": {
                "trace": trace,
                "cache_hit": cache_hit,
                "full_log": {
                    "query": query,
                    "trace": trace,
                    "final_answer": answer,
                    "cache_hit": cache_hit,
                    "thread_id": st.session_state.thread_id,
                    "timestamp": datetime.now().isoformat()
                }
            }
        }
        
        st.session_state.messages.append(assistant_message)
        
        # Update conversation history
        st.session_state.conversation_history.append(f"User: {query}")
        st.session_state.conversation_history.append(f"Assistant: {answer[:500]}")
        
        # Auto-save conversation
        save_conversation(
            st.session_state.thread_id,
            st.session_state.messages,
            st.session_state.conversation_history
        )
        
        st.rerun()


if __name__ == "__main__":
    main()