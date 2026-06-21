"""
Document ingestion pipeline for creating vector stores.

This module handles PDF processing, element extraction, and vector store
creation with batch processing for large documents.
"""

import os
import shutil
import time
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from config import GROQ_API_KEY, LLM_MODEL, EMBEDDING_MODEL, DEFAULT_DB_PATH
from pdf_utils import process_elements, partition_pdf_flexible


def create_vector_store(pdf_path, db_path=DEFAULT_DB_PATH):
    """
    Create a vector store from a PDF document.
    
    This function partitions the PDF, processes all elements (text, tables, charts),
    and creates a searchable vector store using embeddings.
    
    Args:
        pdf_path: Path to the PDF file to process
        db_path: Directory path for vector store (defaults to config value)
        
    Returns:
        None
        
    Raises:
        ValueError: If GROQ_API_KEY is not configured
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not configured in environment")

    start_time = time.time()

    # Step 1: Partition PDF into elements
    print("Partitioning PDF...")
    partition_start = time.time()
    elements = partition_pdf_flexible(pdf_path)
    partition_time = time.time() - partition_start
    print(f"Extracted {len(elements)} elements in {partition_time:.1f}s")

    # Step 2: Process elements into documents
    print("\nProcessing elements...")
    process_start = time.time()
    llm = ChatGroq(model=LLM_MODEL, temperature=0, api_key=GROQ_API_KEY)
    docs = process_elements(elements, llm, pdf_path)
    process_time = time.time() - process_start
    print(f"Created {len(docs)} documents in {process_time:.1f}s")

    # Step 3: Create vector store with embeddings
    print("\nCreating vector store...")
    store_start = time.time()
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Remove existing vector store if present
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    
    # Batch processing for large document sets
    BATCH_SIZE = 100

    if len(docs) > BATCH_SIZE:
        print(f"Using batch processing ({BATCH_SIZE} documents per batch)...")
        
        # Create initial store with first batch
        vector_store = Chroma.from_documents(
            documents=docs[:BATCH_SIZE],
            embedding=embedding_model,
            persist_directory=db_path
        )
        
        # Add remaining documents in batches
        total_batches = (len(docs) - 1) // BATCH_SIZE + 1
        for i in range(BATCH_SIZE, len(docs), BATCH_SIZE):
            batch = docs[i:i + BATCH_SIZE]
            vector_store.add_documents(batch)
            current_batch = i // BATCH_SIZE + 1
            print(f"Added batch {current_batch}/{total_batches}")
    else:
        # Process all documents at once for small sets
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embedding_model,
            persist_directory=db_path
        )

    # FIX: Explicitly release the ChromaDB connection so Windows unlocks the
    # SQLite file. Without this, the lock persists until Python GC runs —
    # which may be AFTER Streamlit tries to open the same file, causing WinError 32.
    try:
        vector_store._client._system.stop()
    except Exception:
        pass
    del vector_store
    import gc
    gc.collect()
    time.sleep(1)  # Give Windows a moment to fully release the file handle

    store_time = time.time() - store_start
    print(f"Vector store created in {store_time:.1f}s")
    
    # Print summary statistics
    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total elements extracted: {len(elements)}")
    print(f"Total documents created: {len(docs)}")
    print(f"Total processing time: {total_time:.1f}s")
    print(f"Average time per element: {total_time / len(elements):.2f}s")
    print("=" * 70)