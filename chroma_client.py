"""
Shared ChromaDB client module.

THE PROBLEM THIS SOLVED (root cause):
chromadb keeps its own internal, library-level cache of "System" objects,
keyed by the persist directory path. This cache lives INSIDE chromadb itself
(chromadb.api.client.SharedSystemClient) -- it is separate from any cache an
application builds on top of it.

The old version of this file deleted the on-disk vectorstore directory and
created a brand new chromadb.PersistentClient on every re-ingest. Even after
clearing this module's own cache, chromadb's internal cache could still be
holding state tied to the now-deleted directory, which produced:

    ValueError: Could not connect to tenant default_tenant. Are you sure it exists?

It also meant a half-finished ingest could leave the OLD vectorstore data on
disk, so a fresh `streamlit run` would load the previous document instead of
the new one.

FIX:
- Exactly ONE PersistentClient is ever created per directory, for the life of
  the process (_clients dict acts as a true singleton cache).
- To replace a document's data, we delete + recreate the COLLECTION through
  that same client (reset_collection), instead of deleting the directory and
  creating a new client. This means chromadb's internal System cache is never
  disturbed mid-process, which is what was causing the tenant error.
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


_clients = {}  # abs_path -> chromadb.PersistentClient (one per process, ever)


def _get_client(persist_dir: str) -> chromadb.PersistentClient:
    """
    Return the single shared PersistentClient for this directory, creating it
    on first use. It is never deleted or recreated for the life of the process.
    """
    abs_dir = os.path.abspath(persist_dir)
    os.makedirs(abs_dir, exist_ok=True)

    if abs_dir not in _clients:
        print(f"[chroma_client] Creating PersistentClient for: {abs_dir}")
        _clients[abs_dir] = chromadb.PersistentClient(path=abs_dir)

    return _clients[abs_dir]


def reset_collection(persist_dir: str, collection_name: str = "langchain"):
    """
    Wipe a collection's contents without touching the directory or the client.
    Call this instead of shutil.rmtree() whenever you need to re-ingest fresh
    data into an existing vectorstore directory.
    """
    client = _get_client(persist_dir)
    try:
        client.delete_collection(collection_name)
        print(f"[chroma_client] Cleared collection '{collection_name}'")
    except Exception as e:
        # Collection may not exist yet on first run -- that's fine.
        print(f"[chroma_client] No existing collection to clear ({e})")
    return client


def invalidate_client(persist_dir: str):
    """
    Kept only for backward compatibility with any old call sites.
    No longer needed/used now that the client is a true process-wide
    singleton and we never delete the underlying directory.
    """
    print("[chroma_client] invalidate_client() is a no-op now (deprecated).")


def get_vectorstore(persist_dir: str, collection_name: str = "langchain") -> Chroma:
    """
    Get a LangChain Chroma wrapper backed by the shared PersistentClient.
    """
    client = _get_client(persist_dir)
    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=get_embedding_model()
    )