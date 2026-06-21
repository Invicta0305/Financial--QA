"""
Document ingestion pipeline for creating vector stores.
"""

import os
import shutil
import time
import gc
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from config import GROQ_API_KEY, LLM_MODEL, EMBEDDING_MODEL, DEFAULT_DB_PATH
from pdf_utils import process_elements, partition_pdf_flexible


def create_vector_store(pdf_path, db_path=DEFAULT_DB_PATH):
    """
    Create a vector store from a PDF document.
    """
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not configured in environment")

    start_time = time.time()

    # Step 1: Partition PDF
    print("Partitioning PDF...")
    partition_start = time.time()
    elements = partition_pdf_flexible(pdf_path)
    partition_time = time.time() - partition_start
    print(f"Extracted {len(elements)} elements in {partition_time:.1f}s")

    # Step 2: Process elements
    print("\nProcessing elements...")
    process_start = time.time()
    llm = ChatGroq(model=LLM_MODEL, temperature=0, api_key=GROQ_API_KEY)
    docs = process_elements(elements, llm, pdf_path)
    process_time = time.time() - process_start
    print(f"Created {len(docs)} documents in {process_time:.1f}s")

    # Step 3: Create vector store
    print("\nCreating vector store...")
    store_start = time.time()

    # FIX: Before deleting the old vectorstore, tell chroma_client.py to
    # release its cached PersistentClient for this directory. If we delete
    # the directory while chroma_client still holds an open client, Windows
    # locks the sqlite file (WinError 32).
    try:
        from chroma_client import invalidate_client
        invalidate_client(db_path)
        time.sleep(0.5)  # Let Windows release the file handle
    except Exception as e:
        print(f"[ingest] Warning: could not invalidate client cache: {e}")

    # Now safe to delete the old vectorstore
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    os.makedirs(db_path, exist_ok=True)

    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # FIX: Use chromadb.PersistentClient directly — same as chroma_client.py uses.
    # This avoids the conflict between langchain_chroma's internal client and
    # our shared client that was causing "Could not connect to tenant default_tenant".
    persistent_client = chromadb.PersistentClient(path=db_path)

    BATCH_SIZE = 100

    if len(docs) > BATCH_SIZE:
        print(f"Using batch processing ({BATCH_SIZE} documents per batch)...")
        vector_store = Chroma(
            client=persistent_client,
            collection_name="langchain",
            embedding_function=embedding_model
        )
        total_batches = (len(docs) + BATCH_SIZE - 1) // BATCH_SIZE
        for i in range(0, len(docs), BATCH_SIZE):
            batch = docs[i:i + BATCH_SIZE]
            vector_store.add_documents(batch)
            current_batch = i // BATCH_SIZE + 1
            print(f"Added batch {current_batch}/{total_batches}")
    else:
        vector_store = Chroma(
            client=persistent_client,
            collection_name="langchain",
            embedding_function=embedding_model
        )
        vector_store.add_documents(docs)

    # Release the client so Windows unlocks the sqlite file
    try:
        persistent_client._system.stop()
    except Exception:
        pass
    del vector_store
    del persistent_client
    gc.collect()
    time.sleep(1)

    # Now register a fresh client in chroma_client's cache so the
    # next query immediately gets a valid client without needing a reload.
    try:
        from chroma_client import _get_client
        _get_client(db_path)
        print("[ingest] Fresh client registered in chroma_client cache.")
    except Exception as e:
        print(f"[ingest] Note: could not pre-register client: {e}")

    store_time = time.time() - store_start
    print(f"Vector store created in {store_time:.1f}s")

    total_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("PROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total elements extracted: {len(elements)}")
    print(f"Total documents created: {len(docs)}")
    print(f"Total processing time: {total_time:.1f}s")
    print(f"Average time per element: {total_time / len(elements):.2f}s")
    print("=" * 70)