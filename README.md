# 🤖 Multi Agent QA for Financial Documents

<div align="center">

**An intelligent LangGraph-based multi-agent system for financial document analysis, powered by LLM orchestration**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-green.svg)](https://github.com/langchain-ai/langgraph)
[![Streamlit](https://img.shields.io/badge/Streamlit-UI-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 🚀 Quick Start

1. **Get API Keys** (all required):
   - GROQ: https://console.groq.com
   - Tavily: https://app.tavily.com
   - Unstructured: https://unstructured.io

2. **Clone & Install**:
   ```
   git clone https://github.com/Aditya-ad48/INTERIIT-NLP-Prepathon.git
   cd INTERIIT-NLP-Prepathon
   python -m venv finance_env
   source finance_env/bin/activate  # On Windows: finance_env\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   ```
   cp .env.example .env
   # Edit .env and add your three API keys
   ```

4. **Ingest Documents**:
   ```
   mkdir -p data
   # Add PDFs to data/ folder
   python ingest.py
   ```

5. **Run Application**:
   ```
   streamlit run app_streamlit.py
   ```

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Agent Descriptions](#-agent-descriptions)
- [PDF Processing Pipeline](#-pdf-processing-pipeline)
- [Chart & Image Analysis with Groq Vision](#-chart--image-analysis-with-groq-vision)
- [Project Structure](#-project-structure)
- [Configuration](#-configuration)
- [Technologies Used](#-technologies-used)
- [Contributing](#-contributing)

---

## 🎯 Overview

This project implements a **swarm-based multi-agent system** for financial document analysis using **LangGraph** orchestration. Unlike hierarchical architectures with central supervisors, this system enables **autonomous agent coordination** where each specialized agent independently decides the next handoff based on query requirements and data availability.

- **🧠 Intelligent Intent Classification**: Distinguishes between casual queries, conversation history requests, and informational questions
- **🔄 Dynamic Agent Orchestration**: Agents autonomously decide next steps based on query requirements
- **📊 Multi-Modal Processing**: Handles document retrieval, web search, table extraction, and mathematical calculations
- **💾 Thread-Aware Caching**: Maintains conversation context across sessions
- **⚡ Fault-Tolerant Design**: Graceful error handling with automatic fallback mechanisms

---

## ✨ Key Features

### 🎯 Core Capabilities

| Feature | Description |
|---------|-------------|
| **Multi-Agent System** | 8 specialized agents working in coordinated swarm architecture |
| **RAG Pipeline** | Retrieval-augmented generation with vector store and web search |
| **Intent Classification** | LLM-based query classification (casual/informational/history) |
| **Smart Routing** | Dynamic agent handoffs based on query type and data availability |
| **Financial Analysis** | Specialized in extracting numbers, calculating ratios, and comparisons |
| **Conversation Memory** | Thread-aware caching and conversation history tracking |
| **PDF Processing** | Advanced document parsing and chunking |
| **Web Search Integration** | Real-time external data retrieval via Tavily API |
| **Vision Analysis** | Groq Vision API extracts data from charts/graphs/images |

### 🛠️ Technical Features

- **Stateful Graph Execution**: LangGraph StateGraph with persistent memory
- **Retry Logic**: Exponential backoff for transient failures
- **Error Handling**: Comprehensive exception handling with fallback agents
- **Type Safety**: Pydantic models for state validation
- **Modular Design**: Clean separation of concerns across agents
- **Streamlit Interface**: User-friendly web UI with file upload

---

## 🏗️ System Architecture

### Agent Flow Diagram

```
                    User Query
                        ↓
                ┌───────────────┐
                │    Memory     │ ← Checks cache (90% similarity)
                └───────┬───────┘
                        ↓
                ┌───────┴────────┐
                │  Cache Hit?    │
                └───────┬────────┘
                        │
                ┌───────┴────────┐
                ↓                ↓
          YES: Skip         NO: Continue
          to Aggregator     Pipeline
                │                │
                │                ↓
                │        ┌───────────────┐
                │        │   Retriever   │ ← Vector store search (k=10)
                │        └───────┬───────┘
                │                ↓
                │        ┌───────────────┐
                │        │   Validator   │ ← Intent + data validation
                │        └───────┬───────┘
                │                │
                │        ┌───────┴────────────────────────────┐
                │        │  Validator Decision Logic          │
                │        │  1. Casual? → Aggregator           │
                │        │  2. Insufficient data? → WebSearch │
                │        │  3. Needs calculation? → Table     │
                │        │  4. Needs summary? → Summarizer    │
                │        └───────┬────────────────────────────┘
                │                │
                │    ┌───────────┼───────────┬────────────┐
                │    ↓           ↓           ↓            ↓
                │ ┌─────────┐ ┌────────┐ ┌──────────┐ ┌───────────┐
                │ │WebSearch│ │ Table  │ │Summarizer│ │ Direct to │
                │ └────┬────┘ └───┬────┘ └────┬─────┘ │ Aggregator│
                │      │          │           │       └─────┬─────┘
                │      │          │           │             │
                │      ↓          │           │             │
                │ ┌─────────┐     │           │             │
                │ │Validator│     │           │             │
                │ │(Re-eval)│     │           │             │
                │ └────┬────┘     │           │             │
                │      │          │           │             │
                │  ┌───┴──────────┼───────────┤             │
                │  ↓              ↓           ↓             │
                │ WebSearch → Validator decides:            │
                │  results      Table/Summarizer/Aggregator │
                │                │           │              │
                │                ↓           ↓              │
                │            ┌────────┐  (goes to           │
                │            │  Math  │  Aggregator)        │
                │            └───┬────┘                     │
                │                │                          │
                └────────────────┼──────────────────────────┘
                                 ↓
                        ┌────────────────┐
                        │  Aggregator    │ ← Synthesizes final answer
                        │  (Caches)      │
                        └────────┬───────┘
                                 ↓
                          Final Answer

```

### Agent Responsibilities

```
┌──────────────────────────────────────────────────────────────┐
│ Memory Agent                                                 │
│ -  Query cache lookup using vector similarity                │
│ -  Thread-aware conversation history                         │
├──────────────────────────────────────────────────────────────┤
│ Retriever Agent                                              │
│ -  Document retrieval from Chroma vector store               │
│ -  Semantic search with embeddings                           │
├──────────────────────────────────────────────────────────────┤
│ Validator Agent                                              │
│ -  Intent classification (casual/informational/history)      │
│ -  Query type detection (calculation/summary/general)        │
│ -  Data relevance validation                                 │
├──────────────────────────────────────────────────────────────┤
│ WebSearch Agent                                              │
│ -  External data retrieval via Tavily API                    │
│ -  Fallback for insufficient document data                   │
├──────────────────────────────────────────────────────────────┤
│ Summarizer Agent                                             │
│ -  Concise content summarization                             │
│ -  Handles large document contexts                           │
├──────────────────────────────────────────────────────────────┤
│ Table Agent                                                  │
│ -  Structured numeric data extraction                        │
│ -  JSON formatting of financial metrics                      │
├──────────────────────────────────────────────────────────────┤
│ Math Agent                                                   │
│ -  Financial calculations (ratios, growth rates, etc.)       │
│ -  Comparative analysis                                      │
├──────────────────────────────────────────────────────────────┤
│ Aggregator Agent                                             │
│ -  Final answer synthesis                                    │
│ -  Error handling and user-friendly responses                │
│ -  Query caching for future lookups                          │
└──────────────────────────────────────────────────────────────┘
```

---

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager
- API keys (GROQ, Tavily, Unstructured)

### Step 1: Clone the Repository

```
git clone https://github.com/Aditya-ad48/INTERIIT-NLP-Prepathon.git
cd INTERIIT-NLP-Prepathon
```

### Step 2: Create Virtual Environment

```
# Create virtual environment
python -m venv finance_env

# Activate (Mac/Linux)
source finance_env/bin/activate

# Activate (Windows)
finance_env\Scripts\activate
```

### Step 3: Install Dependencies

```
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the project root:

```
cp .env.example .env
```

Edit `.env` and add your API keys:

```
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
UNSTRUCTURED_API_KEY=your_unstructured_api_key_here
```

**Get API Keys:**
- **GROQ**: [console.groq.com](https://console.groq.com) - For LLM inference
- **Tavily**: [app.tavily.com](https://app.tavily.com) - For web search
- **Unstructured**: [unstructured.io](https://unstructured.io) - For advanced PDF parsing

### Step 5: Prepare Document Store

```
# Create directory for PDFs
mkdir -p data

# Add your financial documents (PDFs) to the data/ folder
# Example: data/financial_report_q1.pdf

# Ingest documents into vector store
python ingest.py
```

---

## 💻 Usage

### Running the Application

```
streamlit run app_streamlit.py
```

The application will open in your browser at `http://localhost:8501`

### Using the Interface

1. **Upload Documents** (optional)
   - Click "Browse files" to upload PDF documents
   - Documents are processed and added to the vector store

2. **Ask Questions**
   - Type your query in the chat input
   - Examples:
     - "What was the Q1 revenue?"
     - "Calculate the profit margin for 2024"
     - "Compare expenses between Q1 and Q2"
     - "Summarize the risk factors"

3. **View Results**
   - See the agent execution trace
   - Get comprehensive answers with sources
   - Cached responses load instantly for repeated queries

### Example Queries

**Financial Analysis:**
```
"What is the total revenue for Q1 2024?"
"Calculate the year-over-year growth rate"
"Compare operating expenses between quarters"
```

**Document Summarization:**
```
"Summarize the key risks mentioned in the report"
"Give me an overview of the financial performance"
```

**Casual Interaction:**
```
"Hello"
"Thank you"
"What did I ask earlier?"
```

---

## 🤖 Agent Descriptions

### 1. Memory Agent
**Purpose**: Query cache and conversation history management

- Searches cache using vector similarity
- Returns cached answers for duplicate queries
- Maintains thread-aware conversation history
- Reduces latency and API costs

**Handoffs**: → Retriever (on cache miss)

---

### 2. Retriever Agent
**Purpose**: Document retrieval from vector store

- Semantic search using HuggingFace embeddings
- Retrieves top-k relevant document chunks
- Validates document quality
- Fallback to WebSearch on failure

**Handoffs**: → Validator (always)

---

### 3. Validator Agent
**Purpose**: Intent classification and routing orchestration

**Three-tier classification:**
1. **Intent Classification**: Casual vs Informational vs History
2. **Query Type Detection**: Calculation vs Summary vs General
3. **Data Relevance Validation**: Ensures data can answer query

**Handoffs**: → WebSearch | Summarizer | Table | Aggregator

---

### 4. WebSearch Agent
**Purpose**: External data retrieval

- Tavily API integration for web search
- Fallback when documents are insufficient
- Returns relevant web content
- Handles API errors gracefully

**Handoffs**: → Validator (for assessment)

---

### 5. Summarizer Agent
**Purpose**: Content summarization

- Generates concise summaries
- Handles large contexts (30K+ chars)
- Extracts key points
- Uses LLM for natural language generation

**Handoffs**: → Aggregator (always)

---

### 6. Table Agent
**Purpose**: Structured data extraction

- Extracts numeric data (revenue, expenses, ratios)
- Formats as JSON with metadata
- Preserves temporal context (periods)
- Identifies calculation requirements

**Handoffs**: → Math | Aggregator

---

### 7. Math Agent
**Purpose**: Financial calculations

**Capabilities:**
- Comparisons (Q1 vs Q2)
- Growth rates & percentage changes
- Financial ratios (profit margin, etc.)
- Aggregations (totals, averages)
- Step-by-step calculation breakdown

**Handoffs**: → Aggregator (always)

---

### 8. Aggregator Agent
**Purpose**: Final answer synthesis

- Combines information from all sources
- Generates natural language responses
- Handles error cases gracefully
- Caches answers for future queries
- Responds to casual queries directly

**Handoffs**: → END (terminal node)

---

## 📄 PDF Processing Pipeline

### Unstructured API Integration

This system uses **Unstructured.io** for advanced PDF document processing, enabling robust extraction of text, tables, and structured data from financial documents.

### Processing Flow

```
PDF Upload
    ↓
┌───────────────────────────┐
│  Unstructured API         │
│  -  Text extraction       │
│  -  Table detection       │
│  -  Layout analysis       │
│  -  Element classification│
└──────────┬────────────────┘
           ↓
┌───────────────────────────┐
│  Document Chunking        │
│  -  Semantic segmentation │
│  -  Overlap handling      │
│  -  Metadata preservation │
└──────────┬────────────────┘
           ↓
┌───────────────────────────┐
│  Vector Store (Chroma)    │
│  -  Embedding generation  │
│  -  Index creation        │
│  -  Semantic search ready │
└───────────────────────────┘
```

### Why Unstructured API?

| Feature | Benefit |
|---------|---------|
| **Multi-format Support** | Handles PDFs, images, tables seamlessly |
| **Layout Preservation** | Maintains document structure and hierarchy |
| **Table Extraction** | Accurately extracts financial tables and data |
| **Cloud-based** | Scalable processing without local dependencies |
| **OCR Support** | Handles scanned documents |

### Usage in Code

```
# pdf_utils.py
from unstructured_client import UnstructuredClient
from config import UNSTRUCTURED_API_KEY

client = UnstructuredClient(api_key_auth=UNSTRUCTURED_API_KEY)

# Process PDF with advanced parsing
elements = partition_pdf_flexible(
    filename=pdf_path,
    strategy="hi_res",  # High-resolution mode
    extract_images=False
)

# Elements include text, tables, titles, etc.
documents = process_elements(elements)
```

### Configuration

Edit `config.py` to customize PDF processing:

```
# PDF Processing Settings
UNSTRUCTURED_STRATEGY = "hi_res"  # or "fast" for speed
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
```

---

## 📊 Chart & Image Analysis with Groq Vision

### Multimodal Document Understanding

This system uses **Groq's Vision API** (`meta-llama/llama-4-scout-17b-16e-instruct`) to extract structured data from charts, graphs, and images embedded in financial PDFs.

### How It Works

```
PDF Document
    ↓
┌─────────────────────────────┐
│ Unstructured API            │
│ - Detects images/charts     │
│ - Extracts as base64        │
│ - Identifies element types  │
└──────────┬──────────────────┘
           ↓
┌─────────────────────────────┐
│ Groq Vision API             │
│ Model: llama-4-scout-17b    │
│ - Analyzes image content    │
│ - Extracts numeric data     │
│ - Identifies trends         │
└──────────┬──────────────────┘
           ↓
┌─────────────────────────────┐
│ Structured Text Output      │
│ - Chart titles              │
│ - Data points with labels   │
│ - Axis information          │
│ - Annotations               │
└─────────────────────────────┘
```

### Vision Extraction Capabilities

The `analyze_chart_with_groq()` function extracts:

| Element Type       | Extracted Information                   |
|--------------------|----------------------------------------|
| **Bar Charts**     | Values, labels, comparisons            |
| **Line Graphs**    | Trends, data points, time series       |
| **Pie Charts**     | Percentages, category breakdowns       |
| **Financial Tables** | Row/column headers, numeric values   |
| **Diagrams**       | Annotations, relationships             |
| **Mixed Visuals**  | Text overlays, legends                 |

Vision analysis is **automatically triggered** during PDF ingestion when the Unstructured API detects visual elements:

### Vision API Configuration

**Model Details**:
- **Model**: `meta-llama/llama-4-scout-17b-16e-instruct`
- **Context Window**: 16K tokens
- **Max Output**: 2048 tokens
- **Temperature**: 0 (deterministic)
- **Image Size Limit**: 5MB

**Usage Limits**:
- GROQ free tier: Shared with text inference quota
- Image processing: ~2-4 seconds per chart
- Automatic fallback: If vision fails, uses text extraction

### Advantages Over OCR

| Traditional OCR           | Groq Vision (LLM-based)         |
|--------------------------|---------------------------------|
| Extracts raw text only   | Understands context & structure |
| Misses chart relationships| Identifies trends & patterns   |
| No numeric interpretation | Extracts labeled data points   |
| Fails on complex layouts | Handles multi-element visuals   |
---

## 📁 Project Structure

```
INTERIIT-NLP-Prepathon/
│
├── agents/                      # Agent implementations
│   ├── __init__.py
│   ├── memory_agent.py         # Cache & conversation history
│   ├── retriever_agent.py      # Document retrieval
│   ├── validator_agent.py      # Intent classification & routing
│   ├── websearch_agent.py      # External data retrieval
│   ├── summarizer_agent.py     # Content summarization
│   ├── table_agent.py          # Data extraction
│   ├── math_agent.py           # Financial calculations
│   └── aggregator_agent.py     # Answer synthesis
│
├── config.py                    # Configuration & environment variables
├── models.py                    # Pydantic models & type definitions
├── decorators.py                # Error handling & retry logic
├── utils.py                     # Utility functions
├── pdf_utils.py                 # PDF processing utilities
│
├── graph.py                     # LangGraph orchestration
├── ingest.py                    # Document ingestion pipeline
├── app_streamlit.py             # Streamlit web interface
│
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

---

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GROQ_API_KEY` | GROQ LLM API key for inference | ✅ Yes |
| `TAVILY_API_KEY` | Tavily search API key for web search | ✅ Yes |
| `UNSTRUCTURED_API_KEY` | Unstructured API key for PDF processing | ✅ Yes |

### Configurable Parameters

Edit `config.py` to customize:

```
# LLM Configuration
LLM_MODEL = "llama-3.3-70b-versatile"
and "openai/gpt-oss-120b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Vector Store
DEFAULT_DB_PATH = "vectorstore_final"

# Retrieval Settings
RETRIEVAL_K = 10  # Number of documents to retrieve

# Timeouts
AGENT_TIMEOUT = 30  # seconds

# PDF Processing Settings
UNSTRUCTURED_STRATEGY = "hi_res"  # or "fast" for speed
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Vision API Settings
VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
MAX_IMAGE_SIZE = 5000000  # 5MB
VISION_MAX_WORKERS = 4  # Parallel image processing
VISION_TEMPERATURE = 0
VISION_MAX_TOKENS = 2048

# Unstructured API configuration
EXTRACT_IMAGE_TYPES = ["Image", "Table"]  # Element types to extract
PARALLEL_WORKERS = 4  # Concurrent element processing
```

---

## 🛠️ Technologies Used

### Core Frameworks
- **LangGraph**: Multi-agent orchestration and state management
- **LangChain**: LLM abstraction and RAG pipeline
- **Streamlit**: Web interface

### AI & ML
- **GROQ**: High-performance LLM inference
  - Text: `llama-3.3-70b-versatile` & `openai/gpt-oss-120b`
  - Vision: `meta-llama/llama-4-scout-17b-16e-instruct`
- **Chroma**: Vector database for semantic search
- **HuggingFace**: Embedding models (`all-MiniLM-L6-v2`)

### APIs
- **Tavily**: Web search API for real-time external data
- **Unstructured**: Advanced PDF parsing with image/chart detection
- **GROQ Vision**: Multimodal chart and image analysis

### Data Processing
- **Pandas**: Data manipulation and table parsing
- **PyPDF**: PDF document handling
- **Base64**: Image encoding for vision API

---

## 🎓 Learning Resources

This project demonstrates:

- **LangGraph State Management**: Persistent conversation state
- **Multi-Agent Coordination**: Autonomous agent handoffs
- **RAG Architecture**: Retrieval-augmented generation
- **Intent Classification**: LLM-based query understanding
- **Error Handling**: Production-grade fault tolerance
- **Caching Strategies**: Vector similarity-based cache

---

## 📊 Performance

- **Average Response Time**: 2-5 seconds (first query)
- **Cache Hit Response**: < 0.5 seconds
- **Document Retrieval**: ~1 second for 10 documents
- **Web Search Fallback**: 2-3 seconds

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📝 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Aditya Ahirwar**
- GitHub: [@Aditya-ad48](https://github.com/Aditya-ad48)
- Email: adityaraj13.ahirwar@gmail.com

---

## 🙏 Acknowledgments

- **LangChain Team** for LangGraph framework
- **GROQ** for high-performance LLM inference
- **Tavily** for web search capabilities
- **Unstructured.io** for advanced PDF processing

---


## 📚 Research Papers & Foundations

This project implements concepts from the following research papers:

### Core Architecture

1. **Retrieval-Augmented Generation (RAG)**
   - Lewis et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
   - Meta AI Research
   - [arXiv:2005.11401](https://arxiv.org/abs/2005.11401)
   - **Implementation**: Document retrieval with Chroma vector store + LLM generation

2. **LangGraph Multi-Agent Orchestration**
   - Duan & Wang (2024). "Exploration of LLM Multi-Agent Application Implementation Based on LangGraph+CrewAI"
   - [arXiv:2411.18241](https://arxiv.org/html/2411.18241v1)
   - **Implementation**: StateGraph with 8 specialized agents and conditional routing

### Supporting Concepts

3. **Swarm Intelligence for Coordination**
   - Gnanamani & Kumaravel (2025). "Coordination and Collaboration in Multi-Agent Autonomous Systems: A Swarm Intelligence Approach"
   - **Implementation**: Decentralized agent decision-making and autonomous handoffs

4. **Advanced RAG Techniques**
   - Gao et al. (2023). "Retrieval-Augmented Generation for Large Language Models: A Survey"
   - [arXiv:2312.10997](https://arxiv.org/abs/2312.10997)
   - **Implementation**: Thread-aware caching, semantic similarity search

- **State Persistence**: LangGraph checkpointing with MemorySaver

---
#   F i n a n c i a l - - Q A  
 