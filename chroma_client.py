"""
Shared ChromaDB client module.

chromadb >= 0.4.x registers 'default_tenant' in the in-process state of each
PersistentClient. Creating multiple Chroma() objects for the same directory
within one Python process causes them to conflict — one client can't find the
tenant registered by another, giving:
  "Could not connect to tenant default_tenant"

Fix: one PersistentClient per directory, shared across the entire process.
All agents import from here instead of creating their own Chroma() objects.
"""

import os
import chromadb
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Shared embedding model — expensive to load, only load once
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embedding_model


# Shared PersistentClients — one per directory, created once
_clients = {}

def _get_client(persist_dir: str) -> chromadb.PersistentClient:
    """Return the shared PersistentClient for a given directory."""
    persist_dir = os.path.abspath(persist_dir)
    if persist_dir not in _clients:
        os.makedirs(persist_dir, exist_ok=True)
        _clients[persist_dir] = chromadb.PersistentClient(path=persist_dir)
    return _clients[persist_dir]


def get_vectorstore(persist_dir: str, collection_name: str = "langchain") -> Chroma:
    """
    Get a LangChain Chroma wrapper backed by the shared PersistentClient.
    Safe to call multiple times — always returns a wrapper around the same client.
    """
    client = _get_client(persist_dir)
    return Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=get_embedding_model()
    )