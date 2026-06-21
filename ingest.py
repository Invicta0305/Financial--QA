"""
Document ingestion pipeline for creating vector stores.
"""

import time

from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from config import GROQ_API_KEY, LLM_MODEL, DEFAULT_DB_PATH
from pdf_utils import process_elements, partition_pdf_flexible
from chroma_client import reset_collection, get_embedding_model


def create_vector_store(pdf_path, db_path=DEFAULT_DB_PATH):
    """
    Create a vector store from a PDF document.

    Re-ingesting a new document no longer deletes the on-disk directory or
    creates a brand new chromadb client -- it reuses the single shared
    PersistentClient (from chroma_client.py) and just clears the collection.
    This is what avoids the "Could not connect to tenant default_tenant" error.
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

    # Clear out any previous document's data by resetting the COLLECTION,
    # not the directory/client. Uses the one shared PersistentClient for the
    # whole process (see chroma_client.py).
    client = reset_collection(db_path, "langchain")
    embedding_model = get_embedding_model()

    vector_store = Chroma(
        client=client,
        collection_name="langchain",
        embedding_function=embedding_model
    )

    BATCH_SIZE = 100
    if len(docs) > BATCH_SIZE:
        print(f"Using batch processing ({BATCH_SIZE} documents per batch)...")
        total_batches = (len(docs) + BATCH_SIZE - 1) // BATCH_SIZE
        for i in range(0, len(docs), BATCH_SIZE):
            batch = docs[i:i + BATCH_SIZE]
            vector_store.add_documents(batch)
            current_batch = i // BATCH_SIZE + 1
            print(f"Added batch {current_batch}/{total_batches}")
    else:
        vector_store.add_documents(docs)

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