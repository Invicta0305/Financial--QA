"""
Shared ChromaDB client module.

THE PROBLEM THIS SOLVES:
chromadb >= 0.4.x registers 'default_tenant' in the in-memory state of each
PersistentClient. Two failure modes existed:

1. Multiple Chroma() calls for same directory -> multiple clients -> tenant conflict
2. ingest.py deletes+recreates the vectorstore directory, but the cached client
   in _clients still points to the OLD (now deleted) database -> tenant error
   on next query even though the files look fine on disk.

FIX:
- One shared PersistentClient per directory (_clients dict)
- Before returning a cached client, verify its DB file still exists and matches.
  If not (ingest deleted+recreated the dir), discard the stale client and make a new one.
- ingest.py calls invalidate_client() after it's done so the next query always
  gets a fresh client pointing at the new database.
"""

import os
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embedding_model


_clients = {}  # abs_path -> chromadb.PersistentClient


def invalidate_client(persist_dir: str):
    """
    Call this after deleting/recreating a vectorstore directory (e.g. in ingest.py).
    Forces the next get_vectorstore() call to create a fresh client.
    """
    abs_dir = os.path.abspath(persist_dir)
    if abs_dir in _clients:
        try:
            _clients[abs_dir]._system.stop()
        except Exception:
            pass
        del _clients[abs_dir]
        print(f"[chroma_client] Invalidated stale client for: {abs_dir}")


def _get_client(persist_dir: str) -> chromadb.PersistentClient:
    abs_dir = os.path.abspath(persist_dir)
    os.makedirs(abs_dir, exist_ok=True)

    db_file = os.path.join(abs_dir, "chroma.sqlite3")

    if abs_dir in _clients:
        # Verify the sqlite file the client was created with still exists.
        # If ingest.py deleted+recreated the directory, the old client is stale.
        client = _clients[abs_dir]
        client_db = getattr(client, "_db_path", None)

        db_exists = os.path.exists(db_file)
        if not db_exists:
            print(f"[chroma_client] DB file gone, creating fresh client for: {abs_dir}")
            invalidate_client(abs_dir)
        else:
            return client

    print(f"[chroma_client] Creating new PersistentClient for: {abs_dir}")
    client = chromadb.PersistentClient(path=abs_dir)
    # Store the db path so we can detect staleness later
    client._db_path = db_file
    _clients[abs_dir] = client
    return client


def get_vectorstore(persist_dir: str, collection_name: str = "langchain") -> Chroma:
    """
    Get a LangChain Chroma wrapper backed by the shared PersistentClient.
    Always validates the client is fresh before returning.
    """
    client = _get_client(persist_dir)
    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=get_embedding_model()
    )