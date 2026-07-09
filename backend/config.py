import os
from dotenv import load_dotenv

load_dotenv()

# ── LLM — Groq (active) ───────────────────────────────────────────────────────
GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL: str   = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# ── LLM — OpenAI (alternative) ───────────────────────────────────────────────
# OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
# OPENAI_MODEL: str   = os.getenv("OPENAI_MODEL", "gpt-4o")

MAX_TOKENS: int      = int(os.getenv("MAX_TOKENS", "1024"))
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K: int = int(os.getenv("TOP_K", "5"))

# ── Chunking ──────────────────────────────────────────────────────────────────
# CHUNKING_METHOD  — which index the SERVER loads (one value)
# CHUNKING_METHODS — which indexes INGEST builds (comma-separated list)
CHUNKING_METHOD: str       = os.getenv("CHUNKING_METHOD", "fixed")
CHUNKING_METHODS: list[str] = [
    m.strip()
    for m in os.getenv("CHUNKING_METHODS", "fixed").split(",")
    # if m.strip() in ("fixed", "paragraph", "semantic")
    if m.strip() # no need to check methods. whatever .env gives will path through 
]

CHUNK_SIZE: int           = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int        = int(os.getenv("CHUNK_OVERLAP", "64"))
SEMANTIC_THRESHOLD: float = float(os.getenv("SEMANTIC_THRESHOLD", "0.5"))

# ── Knowledge source ──────────────────────────────────────────────────────────
SOURCE_TYPE: str    = os.getenv("SOURCE_TYPE", "documents")
DOCUMENTS_DIR: str  = os.getenv("DOCUMENTS_DIR", "data/documents")
DB_URL: str         = os.getenv("DB_URL", "")
DB_QUERY: str       = os.getenv("DB_QUERY", "")
WEB_URLS: list[str] = [u.strip() for u in os.getenv("WEB_URLS", "").split(",") if u.strip()]

# ── Storage ───────────────────────────────────────────────────────────────────
VECTOR_STORE_BASE: str = os.getenv("VECTOR_STORE_PATH", "vector_store")

def get_vector_store_path(method: str = None) -> str:
    """
    Returns the vector store subfolder for a given chunking method.
    If method is not provided, uses CHUNKING_METHOD (the active server method).

    Examples:
        get_vector_store_path()           → vector_store/fixed
        get_vector_store_path("semantic") → vector_store/semantic
    """
    return os.path.join(VECTOR_STORE_BASE, method or CHUNKING_METHOD)

# Convenience — active path the server loads from
VECTOR_STORE_PATH: str = get_vector_store_path()

LOG_PATH: str = os.getenv("LOG_PATH", "logs/requests.jsonl")

# ── Server ────────────────────────────────────────────────────────────────────
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))