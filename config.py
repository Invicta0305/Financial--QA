"""
Configuration module for Financial Document Analysis System.

This module loads environment variables and sets up project paths
and model configurations used throughout the application.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
UNSTRUCTURED_API_KEY = os.getenv("UNSTRUCTURED_API_KEY")

# Model Configuration
LLM_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Project Paths
PROJECT_ROOT = Path(__file__).parent.absolute()
DEFAULT_DB_PATH = str(PROJECT_ROOT / "vectorstore_final")

# FIX: Create the vectorstore directory immediately at config load time.
# ChromaDB throws "error code 14: unable to open database file" if the
# directory doesn't exist — this ensures it always does, regardless of
# whether ingest.py or app_streamlit.py is the first script to run.
os.makedirs(DEFAULT_DB_PATH, exist_ok=True)