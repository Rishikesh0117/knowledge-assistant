import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = ROOT / "data" / "papers"
DOCS_DIR = ROOT / "data" / "langgraph_docs"
VECTORSTORE_DIR = ROOT / "vectorstore"

NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY")
if not NEBIUS_API_KEY:
    try:
        import streamlit as st
        NEBIUS_API_KEY = st.secrets["NEBIUS_API_KEY"]
    except Exception:
        pass
NEBIUS_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"

# Copy the exact model names from the Token Factory Model catalog
FAST_MODEL = "nvidia/Nemotron-3_5-Lightning"
REASONING_MODEL = "nvidia/Nemotron-3-Ultra-550b-a55b"
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-8B"

CHUNK_MAX_TOKENS = 700
TOP_K = 6